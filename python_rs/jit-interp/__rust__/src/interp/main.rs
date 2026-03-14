// main.rs — python_rs standalone runner.
// usage:  python_rs file.ry
//         cat file.ry | python_rs
//
// accepts 100% python syntax (via rustpython‑parser)

mod value;
mod bytecode;
mod vm;
mod arena;
mod ir;
mod lower;
mod frontend;
mod jit;
mod parser;

use frontend::CodeGen;
use lower::LoweringContext;
use vm::VM;

use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    // read source
    let (source, filename) = if args.len() > 1 {
        let path = &args[1];
        (fs::read_to_string(path)?, path.clone())
    } else {
        print!("enter python code (end with ctrl+d / ctrl+z):\n");
        io::stdout().flush()?;
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        (buf, "<stdin>".to_string())
    };

    // parse
    let stmts = match parser::parse_python(&source) {
        Ok(s)  => s,
        Err(e) => { eprintln!("parse error: {e}"); return Ok(()); }
    };

    // lower to frontend ir
    let dir = Path::new(&filename).parent().unwrap_or(Path::new("."));
    let mut declared     = HashSet::new();
    let mut extra_funcs  = Vec::new();
    let mut already      = HashSet::new();

    // re‑use the same split_stmts logic from lib.rs.
    // since main.rs doesn't link lib.rs, inline a minimal version here.
    let main_stmts = convert_stmts(stmts, &mut declared, dir, &mut already, &mut extra_funcs);

    // compile
    let mut codegen   = CodeGen::new();
    let ir_program    = codegen.generate(extra_funcs, main_stmts);
    let mut lower_ctx = LoweringContext::new();
    let (functions, string_pool, float_pool) = lower_ctx.lower_program(&ir_program);

    let mut vm = VM::new(functions, string_pool, float_pool, 1024 * 1024, 1000, 1024);
    match vm.run() {
        Ok(val)  => { if !val.is_nil() { println!("{:?}", val); } }
        Err(e)   => eprintln!("runtime error: {e}"),
    }

    Ok(())
}

// minimal inline conversion (mirrors lib.rs)

use rustpython_ast::{Stmt as PyStmt, Expr as PyExpr, Operator, Constant, BoolOp, CmpOp, UnaryOp};
use frontend::{
    Stmt as FStmt, Expr as FExpr, Func as FFunc,
    BinOp, CmpOp as FCmpOp, BoolOpKind, UnaryOpKind,
};

fn convert_expr(expr: &PyExpr) -> FExpr {
    match expr {
        PyExpr::Constant(c) => match &c.value {
            Constant::Int(n)   => FExpr::Number(n.try_into().unwrap_or(0)),
            Constant::Float(f) => FExpr::Float(*f),
            Constant::Bool(b)  => FExpr::Bool(*b),
            Constant::Str(s)   => FExpr::String(s.clone()),
            Constant::None     => FExpr::None,
            _                  => FExpr::None,
        },
        PyExpr::Name(n) => match n.id.as_str() {
            "true"  => FExpr::Bool(true),
            "false" => FExpr::Bool(false),
            "none"  => FExpr::None,
            other   => FExpr::Variable(other.to_string()),
        },
        PyExpr::BinOp(b) => {
            let op = match b.op {
                Operator::Add    => BinOp::Add,
                Operator::Sub    => BinOp::Sub,
                Operator::Mult   => BinOp::Mul,
                Operator::Div    => BinOp::Div,
                Operator::FloorDiv => BinOp::FloorDiv,
                Operator::Mod    => BinOp::Mod,
                Operator::Pow    => BinOp::Pow,
                _                => BinOp::Add,
            };
            FExpr::Binary(op, Box::new(convert_expr(&b.left)), Box::new(convert_expr(&b.right)))
        }
        PyExpr::Compare(c) => {
            let mut exprs = Vec::new();
            let mut left = convert_expr(&c.left);
            for (op, right_py) in c.ops.iter().zip(c.comparators.iter()) {
                let right = convert_expr(right_py);
                let cmp = match op {
                    CmpOp::Eq    => FCmpOp::Eq,
                    CmpOp::NotEq => FCmpOp::NotEq,
                    CmpOp::Lt    => FCmpOp::Lt,
                    CmpOp::LtE  => FCmpOp::Le,
                    CmpOp::Gt   => FCmpOp::Gt,
                    CmpOp::GtE  => FCmpOp::Ge,
                    _            => FCmpOp::Eq,
                };
                exprs.push(FExpr::Compare(cmp, Box::new(left.clone()), Box::new(right.clone())));
                left = right;
            }
            if exprs.len() == 1 { exprs.remove(0) }
            else { FExpr::BoolOp(BoolOpKind::And, exprs) }
        }
        PyExpr::BoolOp(b) => {
            let kind = match b.op { BoolOp::And => BoolOpKind::And, _ => BoolOpKind::Or };
            FExpr::BoolOp(kind, b.values.iter().map(convert_expr).collect())
        }
        PyExpr::UnaryOp(u) => match u.op {
            UnaryOp::Not  => FExpr::Not(Box::new(convert_expr(&u.operand))),
            UnaryOp::USub => FExpr::Unary(UnaryOpKind::Neg, Box::new(convert_expr(&u.operand))),
            _             => convert_expr(&u.operand),
        },
        PyExpr::Call(call) => {
            let args: Vec<_> = call.args.iter().map(convert_expr).collect();
            match call.func.as_ref() {
                PyExpr::Name(n)      => FExpr::Call(n.id.as_str().to_string(), args),
                PyExpr::Attribute(a) => FExpr::MethodCall {
                    obj:    Box::new(convert_expr(&a.value)),
                    method: a.attr.as_str().to_string(),
                    args,
                },
                _                    => FExpr::PyCallExpr {
                    callable: Box::new(convert_expr(&call.func)), args,
                },
            }
        }
        PyExpr::Attribute(a) => FExpr::GetAttr {
            obj:  Box::new(convert_expr(&a.value)),
            attr: a.attr.as_str().to_string(),
        },
        PyExpr::Subscript(s) => FExpr::Subscript {
            obj: Box::new(convert_expr(&s.value)),
            idx: Box::new(convert_expr(&s.slice)),
        },
        PyExpr::List(l) => FExpr::List(l.elts.iter().map(convert_expr).collect()),
        _ => FExpr::None,
    }
}

fn convert_stmts(
    stmts: Vec<PyStmt>,
    declared: &mut HashSet<String>,
    source_dir: &Path,
    already_loaded: &mut HashSet<String>,
    extra_funcs: &mut Vec<FFunc>,
) -> Vec<FStmt> {
    let mut out = Vec::new();
    for stmt in stmts {
        convert_one_stmt(stmt, declared, source_dir, already_loaded, extra_funcs, &mut out);
    }
    out
}

fn convert_one_stmt(
    stmt: PyStmt,
    declared: &mut HashSet<String>,
    source_dir: &Path,
    already_loaded: &mut HashSet<String>,
    extra_funcs: &mut Vec<FFunc>,
    out: &mut Vec<FStmt>,
) {
    match stmt {
        PyStmt::FunctionDef(f) => {
            let name   = f.name.as_str().to_string();
            let params = f.args.args.iter().map(|a| a.arg.as_str().to_string()).collect();
            let mut fn_dec = HashSet::new();
            for a in &f.args.args { fn_dec.insert(a.arg.as_str().to_string()); }
            let body = convert_stmts(f.body, &mut fn_dec, source_dir, already_loaded, extra_funcs);
            extra_funcs.push(FFunc { name, params, body });
        }
        PyStmt::Import(i) => {
            for alias in i.names {
                let module = alias.name.as_str().to_string();
                let alias_str = alias.asname
                    .map(|a| a.as_str().to_string())
                    .unwrap_or_else(|| module.clone());
                out.push(FStmt::PythonImport { module, alias: alias_str });
            }
        }
        PyStmt::ImportFrom(i) => {
            let base = i.module.as_ref().map(|m| m.as_str().to_string()).unwrap_or_default();
            for alias in i.names {
                let name     = alias.name.as_str().to_string();
                let full     = if base.is_empty() { name.clone() } else { format!("{base}.{name}") };
                let alias_str = alias.asname.map(|a| a.as_str().to_string()).unwrap_or(name);
                out.push(FStmt::PythonImport { module: full, alias: alias_str });
            }
        }
        PyStmt::Assign(a) => {
            let rhs = convert_expr(&a.value);
            for target in &a.targets {
                if let PyExpr::Name(n) = target {
                    let name = n.id.as_str().to_string();
                    if declared.insert(name.clone()) {
                        out.push(FStmt::Let(name, rhs.clone()));
                    } else {
                        out.push(FStmt::Assign(name, rhs.clone()));
                    }
                }
            }
        }
        PyStmt::AnnAssign(a) => {
            if let Some(val) = a.value {
                let rhs = convert_expr(&val);
                if let PyExpr::Name(n) = a.target.as_ref() {
                    let name = n.id.as_str().to_string();
                    if declared.insert(name.clone()) { out.push(FStmt::Let(name, rhs)); }
                    else { out.push(FStmt::Assign(name, rhs)); }
                }
            }
        }
        PyStmt::AugAssign(a) => {
            if let PyExpr::Name(n) = a.target.as_ref() {
                let name = n.id.as_str().to_string();
                let op   = match a.op {
                    Operator::Add      => BinOp::Add,
                    Operator::Sub      => BinOp::Sub,
                    Operator::Mult     => BinOp::Mul,
                    Operator::Div      => BinOp::Div,
                    Operator::FloorDiv => BinOp::FloorDiv,
                    _                  => BinOp::Add,
                };
                out.push(FStmt::AugAssign(name, op, convert_expr(&a.value)));
            }
        }
        PyStmt::Return(r) => {
            let e = r.value.map(|e| convert_expr(&e)).unwrap_or(FExpr::None);
            out.push(FStmt::Return(e));
        }
        PyStmt::If(i) => {
            let cond    = convert_expr(&i.test);
            let then_bl = convert_stmts(i.body, declared, source_dir, already_loaded, extra_funcs);
            let else_bl = if !i.orelse.is_empty() {
                Some(Box::new(FStmt::Block(
                    convert_stmts(i.orelse, declared, source_dir, already_loaded, extra_funcs)
                )))
            } else { None };
            out.push(FStmt::If(cond, Box::new(FStmt::Block(then_bl)), else_bl));
        }
        PyStmt::While(w) => {
            let cond = convert_expr(&w.test);
            let body = convert_stmts(w.body, declared, source_dir, already_loaded, extra_funcs);
            out.push(FStmt::While(cond, Box::new(FStmt::Block(body))));
        }
        PyStmt::For(f) => {
            let iter   = convert_expr(&f.iter);
            let target = match f.target.as_ref() {
                PyExpr::Name(n) => n.id.as_str().to_string(),
                _               => "__iter_target__".into(),
            };
            declared.insert(target.clone());
            let body = convert_stmts(f.body, declared, source_dir, already_loaded, extra_funcs);
            out.push(FStmt::For { target, iter, body: Box::new(FStmt::Block(body)) });
        }
        PyStmt::Expr(e) => {
            match e.value.as_ref() {
                PyExpr::Call(call) => {
                    if let PyExpr::Name(n) = call.func.as_ref() {
                        if n.id.as_str() == "print" {
                            for arg in &call.args {
                                out.push(FStmt::Print(convert_expr(arg)));
                            }
                            if call.args.is_empty() {
                                out.push(FStmt::Print(FExpr::String("".into())));
                            }
                            return;
                        }
                    }
                    out.push(FStmt::Expr(convert_expr(&e.value)));
                }
                _ => { out.push(FStmt::Expr(convert_expr(&e.value))); }
            }
        }
        PyStmt::Pass(_) | PyStmt::Break(_) | PyStmt::Continue(_) => {}
        _ => {}
    }
}