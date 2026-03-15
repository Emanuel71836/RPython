use crate::ir::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Expr {
    Number(i64),
    Float(f64),
    Bool(bool),
    None,
    String(String),
    Variable(String),
    Assign(String, Box<Expr>),
    AugAssign(String, BinOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Compare(CmpOp, Box<Expr>, Box<Expr>),
    BoolOp(BoolOpKind, Vec<Expr>),
    Not(Box<Expr>),
    Unary(UnaryOpKind, Box<Expr>),
    Call(String, Vec<Expr>),
    GetAttr { obj: Box<Expr>, attr: String },
    MethodCall { obj: Box<Expr>, method: String, args: Vec<Expr> },
    Subscript { obj: Box<Expr>, idx: Box<Expr> },
    PyCallExpr { callable: Box<Expr>, args: Vec<Expr> },
    List(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),  // new variant for dict literals
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CmpOp {
    Eq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    Is,
    IsNot,
    In,
    NotIn,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoolOpKind {
    And,
    Or,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOpKind {
    Neg,
    Pos,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let(String, Expr),
    Assign(String, Expr),
    AugAssign(String, BinOp, Expr),
    Return(Expr),
    Print(Expr),
    While(Expr, Box<Stmt>),
    For {
        target: String,
        iter: Expr,
        body: Box<Stmt>,
    },
    Expr(Expr),
    Block(Vec<Stmt>),
    If(Expr, Box<Stmt>, Option<Box<Stmt>>),
    PythonImport { module: String, alias: String },
}

#[derive(Debug, Clone)]
pub struct Func {
    pub name: String,
    pub params: Vec<String>,
    pub body: Vec<Stmt>,
}

pub struct CodeGen {
    func_index: HashMap<String, FunctionId>,
    next_func_id: FunctionId,
    current_func: FunctionId,
    symbols: HashMap<String, ValueId>,
    declared: std::collections::HashSet<String>,
    next_val: ValueId,
    ir: IrProgram,
    current_block: BasicBlockId,
    last_expr_value: Option<ValueId>,
    builtins_value: Option<ValueId>,
}

impl CodeGen {
    pub fn new() -> Self {
        CodeGen {
            func_index: HashMap::new(),
            next_func_id: 0,
            current_func: 0,
            symbols: HashMap::new(),
            declared: std::collections::HashSet::new(),
            next_val: 1,
            ir: IrProgram::new(),
            current_block: 0,
            last_expr_value: None,
            builtins_value: None,
        }
    }

    fn new_value(&mut self) -> ValueId {
        let v = self.next_val;
        self.next_val += 1;
        v
    }

    fn new_block(&mut self) -> BasicBlockId {
        let func = &mut self.ir.functions[self.current_func];
        let id = func.blocks.len() as BasicBlockId;
        func.blocks.push(BasicBlock {
            id,
            instructions: Vec::new(),
        });
        id
    }

    fn add_insn(&mut self, node: IrNode) {
        let cb = self.current_block as usize;
        self.ir.functions[self.current_func].blocks[cb]
            .instructions
            .push(node);
    }

    fn ends_with_return(&self, block: BasicBlockId) -> bool {
        let func = &self.ir.functions[self.current_func];
        matches!(
            func.blocks[block as usize].instructions.last(),
            Some(IrNode::Return(_))
        )
    }

    fn get_or_declare(&mut self, name: &str) -> ValueId {
        if let Some(&v) = self.symbols.get(name) {
            return v;
        }
        let v = self.new_value();
        self.symbols.insert(name.to_string(), v);
        v
    }

    pub fn generate(&mut self, funcs: Vec<Func>, main_stmts: Vec<Stmt>) -> IrProgram {
        // synthetic main
        self.ir.functions.push(Function {
            name: "__main__".into(),
            params: Vec::new(),
            blocks: vec![BasicBlock {
                id: 0,
                instructions: Vec::new(),
            }],
            entry: 0,
        });
        self.func_index.insert("__main__".into(), 0);
        self.next_func_id = 1;

        // register user functions
        for f in &funcs {
            let id = self.next_func_id;
            self.ir.functions.push(Function {
                name: f.name.clone(),
                params: f.params.clone(),
                blocks: vec![BasicBlock {
                    id: 0,
                    instructions: Vec::new(),
                }],
                entry: 0,
            });
            self.func_index.insert(f.name.clone(), id);
            self.next_func_id += 1;
        }

        // generate main body
        self.current_func = 0;
        self.current_block = 0;
        self.symbols.clear();
        self.declared.clear();
        self.last_expr_value = None;
        self.builtins_value = None;
        for stmt in &main_stmts {
            self.gen_stmt(stmt);
        }
        self.implicit_return();

        // generate user functions
        for f in &funcs {
            let id = *self.func_index.get(&f.name).unwrap();
            self.current_func = id;
            self.current_block = 0;
            self.symbols.clear();
            self.declared.clear();
            self.last_expr_value = None;
            self.builtins_value = None;
            for (idx, param) in f.params.iter().enumerate() {
                let v = self.new_value();
                self.symbols.insert(param.clone(), v);
                self.declared.insert(param.clone());
                self.add_insn(IrNode::Param(v, idx));
            }
            for stmt in &f.body {
                self.gen_stmt(stmt);
            }
            self.implicit_return();
        }

        self.ir.main = 0;
        self.ir.clone()
    }

    fn implicit_return(&mut self) {
        if self.ends_with_return(self.current_block) {
            return;
        }
        if let Some(val) = self.last_expr_value {
            self.add_insn(IrNode::Return(val));
        } else {
            let v = self.new_value();
            self.add_insn(IrNode::Const(v, Constant::Nil));
            self.add_insn(IrNode::Return(v));
        }
    }

    fn gen_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(name, expr) | Stmt::Assign(name, expr) => {
                let val = self.gen_expr(expr);
                if let Some(&existing) = self.symbols.get(name) {
                    self.add_insn(IrNode::Move(existing, val));
                    self.last_expr_value = Some(existing);
                } else {
                    self.symbols.insert(name.clone(), val);
                    self.declared.insert(name.clone());
                    self.last_expr_value = Some(val);
                }
            }
            Stmt::AugAssign(name, op, rhs) => {
                let var_reg = self.get_or_declare(name);
                let rv = self.gen_expr(rhs);
                let node = match op {
                    BinOp::Add => IrNode::Add(var_reg, var_reg, rv),
                    BinOp::Sub => IrNode::Sub(var_reg, var_reg, rv),
                    BinOp::Mul => IrNode::Mul(var_reg, var_reg, rv),
                    BinOp::Div => IrNode::Div(var_reg, var_reg, rv),
                    BinOp::FloorDiv => IrNode::Div(var_reg, var_reg, rv),
                    BinOp::Mod => IrNode::Sub(var_reg, var_reg, rv),
                    BinOp::Pow => IrNode::Mul(var_reg, var_reg, rv),
                };
                self.add_insn(node);
                self.last_expr_value = Some(var_reg);
            }
            Stmt::Return(expr) => {
                let val = self.gen_expr(expr);
                self.add_insn(IrNode::Return(val));
                self.last_expr_value = None;
            }
            Stmt::Print(expr) => {
                let val = self.gen_expr(expr);
                self.add_insn(IrNode::Print(val));
                self.last_expr_value = None;
            }
            Stmt::While(cond, body) => {
                let cond_block = self.new_block();
                let body_block = self.new_block();
                let after_block = self.new_block();
                self.add_insn(IrNode::Jump(cond_block));
                self.current_block = cond_block;
                let cv = self.gen_expr(cond);
                self.add_insn(IrNode::Branch(cv, body_block, after_block));
                self.current_block = body_block;
                self.gen_stmt(body);
                if !self.ends_with_return(self.current_block) {
                    self.add_insn(IrNode::Jump(cond_block));
                }
                self.current_block = after_block;
            }
            Stmt::For { target, iter, body } => match iter {
                Expr::Call(f, args) if f == "range" && (args.len() == 1 || args.len() == 2) => {
                    let (start_expr, stop_expr) = if args.len() == 1 {
                        (Expr::Number(0), args[0].clone())
                    } else {
                        (args[0].clone(), args[1].clone())
                    };
                    let start_val = self.gen_expr(&start_expr);
                    let target_reg = self.get_or_declare(target);
                    self.add_insn(IrNode::Move(target_reg, start_val));
                    let stop_val = self.gen_expr(&stop_expr);

                    let cond_block = self.new_block();
                    let body_block = self.new_block();
                    let after_block = self.new_block();
                    self.add_insn(IrNode::Jump(cond_block));

                    self.current_block = cond_block;
                    let cond_dst = self.new_value();
                    self.add_insn(IrNode::Lt(cond_dst, target_reg, stop_val));
                    self.add_insn(IrNode::Branch(cond_dst, body_block, after_block));

                    self.current_block = body_block;
                    self.gen_stmt(body);
                    let one = self.new_value();
                    self.add_insn(IrNode::Const(one, Constant::Int(1)));
                    self.add_insn(IrNode::Add(target_reg, target_reg, one));
                    if !self.ends_with_return(self.current_block) {
                        self.add_insn(IrNode::Jump(cond_block));
                    }
                    self.current_block = after_block;
                }
                _ => {
                    let iter_val = self.gen_expr(iter);
                    let iter_obj = self.new_value();
                    self.add_insn(IrNode::CallMethod(
                        iter_obj,
                        iter_val,
                        "__iter__".into(),
                        vec![],
                    ));

                    let cond_block = self.new_block();
                    let body_block = self.new_block();
                    let after_block = self.new_block();
                    self.add_insn(IrNode::Jump(cond_block));

                    self.current_block = cond_block;
                    let item = self.new_value();
                    self.add_insn(IrNode::CallMethod(
                        item,
                        iter_obj,
                        "__next__".into(),
                        vec![],
                    ));
                    let is_nil = self.new_value();
                    self.add_insn(IrNode::Const(is_nil, Constant::Nil));
                    self.add_insn(IrNode::Branch(item, body_block, after_block));

                    self.current_block = body_block;
                    let target_reg = self.get_or_declare(target);
                    self.add_insn(IrNode::Move(target_reg, item));
                    self.gen_stmt(body);
                    if !self.ends_with_return(self.current_block) {
                        self.add_insn(IrNode::Jump(cond_block));
                    }
                    self.current_block = after_block;
                }
            },
            Stmt::Expr(expr) => {
                let val = self.gen_expr(expr);
                self.last_expr_value = Some(val);
            }
            Stmt::Block(stmts) => {
                for s in stmts {
                    self.gen_stmt(s);
                }
            }
            Stmt::If(cond, then_stmt, else_stmt) => {
                let cv = self.gen_expr(cond);
                let then_block = self.new_block();
                let else_block = self.new_block();
                let merge_block = self.new_block();
                self.add_insn(IrNode::Branch(cv, then_block, else_block));

                self.current_block = then_block;
                self.gen_stmt(then_stmt);
                if !self.ends_with_return(self.current_block) {
                    self.add_insn(IrNode::Jump(merge_block));
                }

                self.current_block = else_block;
                if let Some(es) = else_stmt {
                    self.gen_stmt(es);
                }
                if !self.ends_with_return(self.current_block) {
                    self.add_insn(IrNode::Jump(merge_block));
                }
                self.current_block = merge_block;
                self.last_expr_value = None;
            }
            Stmt::PythonImport { module, alias } => {
                let dst = self.new_value();
                self.add_insn(IrNode::ImportPython(dst, module.clone()));
                self.symbols.insert(alias.clone(), dst);
                self.declared.insert(alias.clone());
                self.last_expr_value = Some(dst);
            }
        }
    }

    fn gen_expr(&mut self, expr: &Expr) -> ValueId {
        match expr {
            Expr::Number(n) => {
                let v = self.new_value();
                self.add_insn(IrNode::Const(v, Constant::Int(*n)));
                v
            }
            Expr::Float(f) => {
                let v = self.new_value();
                self.add_insn(IrNode::Const(v, Constant::Float(*f)));
                v
            }
            Expr::Bool(b) => {
                let v = self.new_value();
                self.add_insn(IrNode::Const(v, Constant::Bool(*b)));
                v
            }
            Expr::None => {
                let v = self.new_value();
                self.add_insn(IrNode::Const(v, Constant::Nil));
                v
            }
            Expr::String(s) => {
                let v = self.new_value();
                self.add_insn(IrNode::Const(v, Constant::String(s.clone())));
                v
            }
            Expr::Variable(name) => self
                .symbols
                .get(name)
                .copied()
                .unwrap_or_else(|| panic!("undefined variable '{}'", name)),
            Expr::Assign(name, rhs) => {
                let rv = self.gen_expr(rhs);
                if let Some(&existing) = self.symbols.get(name) {
                    self.add_insn(IrNode::Move(existing, rv));
                    existing
                } else {
                    self.symbols.insert(name.clone(), rv);
                    self.declared.insert(name.clone());
                    rv
                }
            }
            Expr::AugAssign(name, op, rhs) => {
                let var_reg = self.get_or_declare(name);
                let rv = self.gen_expr(rhs);
                let node = match op {
                    BinOp::Add => IrNode::Add(var_reg, var_reg, rv),
                    BinOp::Sub => IrNode::Sub(var_reg, var_reg, rv),
                    BinOp::Mul => IrNode::Mul(var_reg, var_reg, rv),
                    BinOp::Div | BinOp::FloorDiv => IrNode::Div(var_reg, var_reg, rv),
                    BinOp::Mod | BinOp::Pow => IrNode::Add(var_reg, var_reg, rv),
                };
                self.add_insn(node);
                var_reg
            }
            Expr::Binary(op, left, right) => {
                let l = self.gen_expr(left);
                let r = self.gen_expr(right);
                let dst = self.new_value();
                self.add_insn(match op {
                    BinOp::Add => IrNode::Add(dst, l, r),
                    BinOp::Sub => IrNode::Sub(dst, l, r),
                    BinOp::Mul => IrNode::Mul(dst, l, r),
                    BinOp::Div | BinOp::FloorDiv => IrNode::Div(dst, l, r),
                    BinOp::Mod => IrNode::Sub(dst, l, r),
                    BinOp::Pow => IrNode::Mul(dst, l, r),
                });
                dst
            }
            Expr::Compare(op, left, right) => {
                let l = self.gen_expr(left);
                let r = self.gen_expr(right);
                let dst = self.new_value();
                self.add_insn(match op {
                    CmpOp::Lt | CmpOp::NotIn => IrNode::Lt(dst, l, r),
                    CmpOp::Le => IrNode::Le(dst, l, r),
                    CmpOp::Gt => IrNode::Lt(dst, r, l),
                    CmpOp::Ge => IrNode::Le(dst, r, l),
                    _ => IrNode::Le(dst, l, r),
                });
                dst
            }
            Expr::BoolOp(kind, exprs) => {
                let mut acc = self.gen_expr(&exprs[0]);
                for e in &exprs[1..] {
                    let next = self.gen_expr(e);
                    let dst = self.new_value();
                    match kind {
                        BoolOpKind::And => self.add_insn(IrNode::Mul(dst, acc, next)),
                        BoolOpKind::Or => self.add_insn(IrNode::Add(dst, acc, next)),
                    }
                    acc = dst;
                }
                acc
            }
            Expr::Not(inner) => {
                let v = self.gen_expr(inner);
                let one = self.new_value();
                let dst = self.new_value();
                self.add_insn(IrNode::Const(one, Constant::Int(1)));
                self.add_insn(IrNode::Sub(dst, one, v));
                dst
            }
            Expr::Unary(op, inner) => {
                let v = self.gen_expr(inner);
                let dst = self.new_value();
                match op {
                    UnaryOpKind::Neg => {
                        let zero = self.new_value();
                        self.add_insn(IrNode::Const(zero, Constant::Int(0)));
                        self.add_insn(IrNode::Sub(dst, zero, v));
                    }
                    UnaryOpKind::Pos => {
                        self.add_insn(IrNode::Move(dst, v));
                    }
                }
                dst
            }
            Expr::Call(name, args) => {
                if let Some(&func_id) = self.func_index.get(name) {
                    // user-defined function
                    let arg_vals: Vec<_> = args.iter().map(|a| self.gen_expr(a)).collect();
                    let dst = self.new_value();
                    self.add_insn(IrNode::Call(dst, func_id, arg_vals));
                    dst
                } else {
                    // built-in function: get it from the builtins module
                    let builtins = if let Some(b) = self.builtins_value {
                        b
                    } else {
                        let b = self.new_value();
                        self.add_insn(IrNode::ImportPython(b, "builtins".into()));
                        self.builtins_value = Some(b);
                        b
                    };
                    let func = self.new_value();
                    self.add_insn(IrNode::GetAttr(func, builtins, name.clone()));
                    let arg_vals: Vec<_> = args.iter().map(|a| self.gen_expr(a)).collect();
                    let dst = self.new_value();
                    self.add_insn(IrNode::PyCall(dst, func, arg_vals));
                    dst
                }
            }
            Expr::List(items) => {
                let builtins = self.new_value();
                self.add_insn(IrNode::ImportPython(builtins, "builtins".into()));
                let list_fn = self.new_value();
                self.add_insn(IrNode::GetAttr(list_fn, builtins, "list".into()));

                let empty_list = self.new_value();
                self.add_insn(IrNode::PyCall(empty_list, list_fn, vec![]));

                let append_method = self.new_value();
                self.add_insn(IrNode::GetAttr(append_method, empty_list, "append".into()));

                for item in items {
                    let item_val = self.gen_expr(item);
                    let temp = self.new_value();
                    self.add_insn(IrNode::PyCall(temp, append_method, vec![item_val]));
                }

                empty_list
            }
                        Expr::Dict(items) => {
                // create an empty dict via builtins.dict()
                let builtins = if let Some(b) = self.builtins_value {
                    b
                } else {
                    let b = self.new_value();
                    self.add_insn(IrNode::ImportPython(b, "builtins".into()));
                    self.builtins_value = Some(b);
                    b
                };
                let dict_fn = self.new_value();
                self.add_insn(IrNode::GetAttr(dict_fn, builtins, "dict".into()));
                let empty_dict = self.new_value();
                self.add_insn(IrNode::PyCall(empty_dict, dict_fn, vec![]));

                // get the __setitem__ method
                let setitem = self.new_value();
                self.add_insn(IrNode::GetAttr(setitem, empty_dict, "__setitem__".into()));

                for (key_expr, value_expr) in items {
                    let key_val = self.gen_expr(key_expr);
                    let value_val = self.gen_expr(value_expr);
                    let temp = self.new_value();
                    self.add_insn(IrNode::PyCall(temp, setitem, vec![key_val, value_val]));
                }

                empty_dict
            }
            Expr::GetAttr { obj, attr } => {
                let obj_val = self.gen_expr(obj);
                let dst = self.new_value();
                self.add_insn(IrNode::GetAttr(dst, obj_val, attr.clone()));
                dst
            }
            Expr::MethodCall { obj, method, args } => {
                let obj_val = self.gen_expr(obj);
                let arg_vals: Vec<_> = args.iter().map(|a| self.gen_expr(a)).collect();
                let dst = self.new_value();
                self.add_insn(IrNode::CallMethod(dst, obj_val, method.clone(), arg_vals));
                dst
            }
            Expr::Subscript { obj, idx } => {
                let obj_val = self.gen_expr(obj);
                let idx_val = self.gen_expr(idx);
                let dst = self.new_value();
                self.add_insn(IrNode::CallMethod(dst, obj_val, "__getitem__".into(), vec![idx_val]));
                dst
            }
            Expr::PyCallExpr { callable, args } => {
                let c = self.gen_expr(callable);
                let arg_vals: Vec<_> = args.iter().map(|a| self.gen_expr(a)).collect();
                let dst = self.new_value();
                self.add_insn(IrNode::PyCall(dst, c, arg_vals));
                dst
            }
        }
    }
}