use pest_derive::Parser;
use pest_consume::{Error, Parser as PestConsumer};
use std::result;

#[derive(Parser)]
#[grammar = "rpython.pest"]
pub struct rpythonParser;

type Result<T> = result::Result<T, Error<Rule>>;
type Node<'i> = pest_consume::Node<'i, Rule, ()>;

#[pest_consume::parser]
impl rpythonParser {
    fn EOI(_input: Node) -> Result<()> { Ok(()) }

    fn identifier(node: Node) -> Result<String> { Ok(node.as_str().to_string()) }

    fn number(node: Node) -> Result<i64> {
        node.as_str().parse::<i64>().map_err(|e| node.error(e))
    }

    fn float(node: Node) -> Result<f64> {
        node.as_str().parse::<f64>().map_err(|e| node.error(e))
    }

    fn string(node: Node) -> Result<String> {
        let s = node.as_str();
        Ok(s[1..s.len()-1].to_string())
    }

    fn r#type(node: Node) -> Result<String> { Ok(node.as_str().to_string()) }

    fn comp(node: Node) -> Result<String> { Ok(node.as_str().to_string()) }
    fn add(node: Node) -> Result<String> { Ok(node.as_str().to_string()) }
    fn mul(node: Node) -> Result<String> { Ok(node.as_str().to_string()) }

    fn param(node: Node) -> Result<(String, String)> {
        let mut children = node.into_children();
        let name_node = children.next().unwrap();
        let type_node = children.next().unwrap();
        let name = Self::identifier(name_node)?;
        let ty = Self::r#type(type_node)?;
        Ok((name, ty))
    }

    fn var_ref(node: Node) -> Result<Expr> { Ok(Expr::Var(node.as_str().to_string())) }

    fn call(node: Node) -> Result<Expr> {
        let mut children = node.into_children();
        let name_node = children.next().unwrap();
        let name = Self::identifier(name_node)?;
        let mut args = Vec::new();
        for child in children {
            if child.as_rule() == Rule::expr {
                args.push(Self::expr(child)?);
            }
        }
        Ok(Expr::Call { func: name, args })
    }

    fn primary(node: Node) -> Result<Expr> {
        let child = node.into_children().next().unwrap();
        match child.as_rule() {
            Rule::call => Self::call(child),
            Rule::var_ref => Self::var_ref(child),
            Rule::number => Ok(Expr::Number(Self::number(child)?)),
            Rule::float => Ok(Expr::Float(Self::float(child)?)),
            Rule::string => Ok(Expr::String(Self::string(child)?)),
            Rule::expr => Self::expr(child),
            _ => unreachable!(),
        }
    }

    fn dotcall(node: Node) -> Result<Expr> {
        let mut all: Vec<Node> = node.into_children().collect();
        let mut idx = 0;

        // first child is always primary
        let mut res = Self::primary(all.remove(0))?;

        // process remaining children as dot-suffix tokens
        while !all.is_empty() {
            // ext must be an identifier (the attr / method name)
            let name_node = all.remove(0);
            let name = Self::identifier(name_node)?;

            // Collect consecutive expr children as method arguments
            let mut args: Vec<Expr> = Vec::new();
            while !all.is_empty() && all[0].as_rule() == Rule::expr {
                let expr_node = all.remove(0);
                args.push(Self::expr(expr_node)?);
            }

            res = if args.is_empty() {
                Expr::GetAttr(Box::new(res), name)
            } else {
                Expr::MethodCall(Box::new(res), name, args)
            };
        }
        Ok(res)
    }

    fn mul_expr(node: Node) -> Result<Expr> {
        let mut children = node.into_children();
        let mut res = Self::dotcall(children.next().unwrap())?;
        while let Some(op_node) = children.next() {
            let op = match op_node.as_rule() {
                Rule::mul => match op_node.as_str() {
                    "*" => Op::Mul,
                    "/" => Op::Div,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            let right = Self::dotcall(children.next().unwrap())?;
            res = Expr::BinOp {
                left: Box::new(res),
                op,
                right: Box::new(right),
            };
        }
        Ok(res)
    }

    fn add_expr(node: Node) -> Result<Expr> {
        let mut children = node.into_children();
        let mut res = Self::mul_expr(children.next().unwrap())?;
        while let Some(op_node) = children.next() {
            let op = match op_node.as_rule() {
                Rule::add => match op_node.as_str() {
                    "+" => Op::Add,
                    "-" => Op::Sub,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            let right = Self::mul_expr(children.next().unwrap())?;
            res = Expr::BinOp {
                left: Box::new(res),
                op,
                right: Box::new(right),
            };
        }
        Ok(res)
    }

    fn comp_expr(node: Node) -> Result<Expr> {
        let mut children = node.into_children();
        let mut res = Self::add_expr(children.next().unwrap())?;
        while let Some(op_node) = children.next() {
            let op = match op_node.as_rule() {
                Rule::comp => match op_node.as_str() {
                    "<" => Op::Lt,
                    "<=" => Op::Le,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            let right = Self::add_expr(children.next().unwrap())?;
            res = Expr::BinOp {
                left: Box::new(res),
                op,
                right: Box::new(right),
            };
        }
        Ok(res)
    }

    fn expr(node: Node) -> Result<Expr> {
        let child = node.into_children().next().unwrap();
        Self::comp_expr(child)
    }

    fn block(node: Node) -> Result<Vec<Statement>> {
        let mut stmts = Vec::new();
        for child in node.into_children() {
            if child.as_rule() == Rule::statement {
                stmts.push(Self::statement(child)?);
            }
        }
        Ok(stmts)
    }

    fn while_statement(node: Node) -> Result<Statement> {
        let mut children = node.into_children();
        let cond_node = children.next().unwrap();
        let cond = Self::expr(cond_node)?;
        let block_node = children.next().unwrap();
        let body = Self::block(block_node)?;
        Ok(Statement::While(WhileStmt { condition: cond, body }))
    }

    fn if_statement(node: Node) -> Result<Statement> {
        let mut children = node.into_children();
        let cond_node = children.next().unwrap();
        let cond = Self::expr(cond_node)?;
        let then_block_node = children.next().unwrap();
        let then_block = Self::block(then_block_node)?;
        let else_block = if let Some(else_node) = children.next() {
            match else_node.as_rule() {
                Rule::block => Some(Self::block(else_node)?),
                Rule::if_statement => Some(vec![Self::if_statement(else_node)?]),
                _ => unreachable!(),
            }
        } else {
            None
        };
        Ok(Statement::If(IfStmt { condition: cond, then_block, else_block }))
    }

    fn assign_statement(node: Node) -> Result<Statement> {
        let mut children = node.into_children();
        let name_node = children.next().unwrap();
        let val_node = children.next().unwrap();
        let name = Self::identifier(name_node)?;
        let val = Self::expr(val_node)?;
        Ok(Statement::Assign(name, val))
    }

    fn var_decl(node: Node) -> Result<Statement> {
        let mut children = node.into_children();
        let name_node = children.next().unwrap();
        let type_node = children.next().unwrap();
        let val_node = children.next().unwrap();
        let name = Self::identifier(name_node)?;
        let ty = Self::r#type(type_node)?;
        let val = Self::expr(val_node)?;
        Ok(Statement::VarDecl(VarDecl { name, type_def: ty, value: val }))
    }

    fn return_statement(node: Node) -> Result<Statement> {
        let mut children = node.into_children();
        let expr_node = children.next().unwrap();
        let expr = Self::expr(expr_node)?;
        Ok(Statement::Return(expr))
    }

    fn function_def(node: Node) -> Result<Statement> {
        let mut children = node.into_children();
        let name_node = children.next().unwrap();
        let name = Self::identifier(name_node)?;
        let mut args = Vec::new();
        let mut return_type = None;
        let mut block_node = None;
        for child in children {
            match child.as_rule() {
                Rule::param => args.push(Self::param(child)?),
                Rule::r#type => return_type = Some(Self::r#type(child)?),
                Rule::block => block_node = Some(child),
                _ => {}
            }
        }
        let return_type = return_type.expect("Missing return type");
        let block_node = block_node.expect("Missing function body");
        let body = Self::block(block_node)?;
        Ok(Statement::FunctionDef(FunctionDef { name, args, return_type, body }))
    }

    fn import_statement(node: Node) -> Result<Statement> {
        let mut children = node.into_children();
        let string_node = children.next().unwrap();
        let filename = Self::string(string_node)?;
        Ok(Statement::Import(filename))
    }

    fn py_import_statement(node: Node) -> Result<Statement> {
        let mut children = node.into_children();
        let module_node = children.next().unwrap();
        let module = Self::identifier(module_node)?;
        let alias = if let Some(alias_node) = children.next() {
            Self::identifier(alias_node)?
        } else {
            module.clone()
        };
        // Encode as PyImport variant
        Ok(Statement::PyImport { alias, module })
    }

    fn statement(node: Node) -> Result<Statement> {
        let child = node.into_children().next().unwrap();
        match child.as_rule() {
            Rule::import_statement => Self::import_statement(child),
            Rule::py_import_statement => Self::py_import_statement(child),
            Rule::var_decl => Self::var_decl(child),
            Rule::function_def => Self::function_def(child),
            Rule::return_statement => Self::return_statement(child),
            Rule::if_statement => Self::if_statement(child),
            Rule::while_statement => Self::while_statement(child),
            Rule::assign_statement => Self::assign_statement(child),
            Rule::expr => Ok(Statement::Expr(Self::expr(child)?)),
            _ => unreachable!(),
        }
    }

    fn program(node: Node) -> Result<Program> {
        let mut body = Vec::new();
        for child in node.into_children() {
            match child.as_rule() {
                Rule::statement => body.push(Self::statement(child)?),
                Rule::EOI => break,
                _ => {}
            }
        }
        Ok(Program { body })
    }
}

#[derive(Debug, Clone)]
pub enum Op {
    Add, Sub, Mul, Div, Lt, Le,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Number(i64),
    Float(f64),
    String(String),
    Var(String),
    BinOp { left: Box<Expr>, op: Op, right: Box<Expr> },
    Call { func: String, args: Vec<Expr> },
    GetAttr(Box<Expr>, String),
    MethodCall(Box<Expr>, String, Vec<Expr>),
}

#[derive(Debug, Clone)]
pub struct VarDecl {
    pub name: String,
    pub type_def: String,
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub struct WhileStmt {
    pub condition: Expr,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub struct IfStmt {
    pub condition: Expr,
    pub then_block: Vec<Statement>,
    pub else_block: Option<Vec<Statement>>,
}

#[derive(Debug, Clone)]
pub struct FunctionDef {
    pub name: String,
    pub args: Vec<(String, String)>,
    pub return_type: String,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub enum Statement {
    VarDecl(VarDecl),
    FunctionDef(FunctionDef),
    Return(Expr),
    While(WhileStmt),
    If(IfStmt),
    Assign(String, Expr),
    Expr(Expr),
    Import(String),         // file-import: `import "path/to/file.ry"`
    PyImport {              // python-import: `import math` or `import numpy as np`
        alias: String,
        module: String,
    },
}

#[derive(Debug, Clone)]
pub struct Program {
    pub body: Vec<Statement>,
}

pub fn parse_rpython_code(code: &str) -> Result<Program> {
    let nodes = rpythonParser::parse(Rule::program, code)?;
    let program_node = nodes.single()?;
    rpythonParser::program(program_node)
}