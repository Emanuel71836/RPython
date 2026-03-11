use crate::ir::*;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Number(i64),
    String(String),
    Ident(String),
    Fn,
    Let,
    If,
    Else,
    While,
    Return,
    Print,
    Import,   // NEW: `import` keyword
    Plus,
    Minus,
    Star,
    Slash,
    Less,
    Assign,
    Semicolon,
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Dot,      // NEW: `.` for attribute access  (obj.method)
    EOF,
}

pub struct Lexer<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Lexer { chars: input.chars().peekable() }
    }

    pub fn next_token(&mut self) -> Token {
        while let Some(&c) = self.chars.peek() {
            if c.is_whitespace() {
                self.chars.next();
                continue;
            }
            match c {
                '.' => { self.chars.next(); return Token::Dot; }
                '+' => { self.chars.next(); return Token::Plus; }
                '-' => { self.chars.next(); return Token::Minus; }
                '*' => { self.chars.next(); return Token::Star; }
                '/' => { self.chars.next(); return Token::Slash; }
                '<' => { self.chars.next(); return Token::Less; }
                '=' => { self.chars.next(); return Token::Assign; }
                ';' => { self.chars.next(); return Token::Semicolon; }
                '(' => { self.chars.next(); return Token::LParen; }
                ')' => { self.chars.next(); return Token::RParen; }
                '{' => { self.chars.next(); return Token::LBrace; }
                '}' => { self.chars.next(); return Token::RBrace; }
                ',' => { self.chars.next(); return Token::Comma; }
                '"' => {
                    self.chars.next();
                    let mut s = String::new();
                    while let Some(&ch) = self.chars.peek() {
                        if ch == '"' { self.chars.next(); break; }
                        s.push(ch);
                        self.chars.next();
                    }
                    return Token::String(s);
                }
                '0'..='9' => {
                    let mut num = 0;
                    while let Some(&d) = self.chars.peek() {
                        if d.is_ascii_digit() {
                            num = num * 10 + (d as i64 - '0' as i64);
                            self.chars.next();
                        } else { break; }
                    }
                    return Token::Number(num);
                }
                'a'..='z' | 'A'..='Z' | '_' => {
                    let mut ident = String::new();
                    while let Some(&c) = self.chars.peek() {
                        if c.is_alphanumeric() || c == '_' {
                            ident.push(c);
                            self.chars.next();
                        } else { break; }
                    }
                    match ident.as_str() {
                        "fn" => return Token::Fn,
                        "let" => return Token::Let,
                        "if" => return Token::If,
                        "else" => return Token::Else,
                        "while" => return Token::While,
                        "return" => return Token::Return,
                        "print" => return Token::Print,
                        "import" => return Token::Import,
                        _ => return Token::Ident(ident),
                    }
                }
                _ => panic!("Unexpected character: {}", c),
            }
        }
        Token::EOF
    }
}

#[derive(Debug)]
pub enum Expr {
    Number(i64),
    String(String),
    Variable(String),
    Assign(String, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
    /// `obj.attr`  — attribute get on a Python object
    GetAttr(Box<Expr>, String),
    /// `obj.method(args…)` — method call on a Python object
    MethodCall(Box<Expr>, String, Vec<Expr>),
}

#[derive(Debug)]
pub enum BinOp {
    Add, Sub, Mul, Div, Lt, Le,
}

#[derive(Debug)]
pub enum Stmt {
    Let(String, Expr),
    Return(Expr),
    Print(Expr),
    While(Expr, Box<Stmt>),
    Expr(Expr),
    Block(Vec<Stmt>),
    If(Expr, Box<Stmt>, Option<Box<Stmt>>),
    /// `import <name>` — imports a Python module, binds to name
    PyImport(String, String),  // (variable_name, module_name)
}

#[derive(Debug)]
pub struct Func {
    pub name: String,
    pub params: Vec<String>,
    pub body: Vec<Stmt>,
}

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
}

impl<'a> Parser<'a> {
    pub fn new(mut lexer: Lexer<'a>) -> Self {
        let current = lexer.next_token();
        Parser { lexer, current }
    }

    fn advance(&mut self) { self.current = self.lexer.next_token(); }

    fn expect(&mut self, expected: Token) {
        if self.current == expected {
            self.advance();
        } else {
            panic!("Expected {:?}, got {:?}", expected, self.current);
        }
    }

    pub fn parse_program(&mut self) -> (Vec<Func>, Vec<Stmt>) {
        let mut funcs = Vec::new();
        let mut stmts = Vec::new();
        loop {
            match &self.current {
                Token::Fn => { funcs.push(self.parse_func()); }
                Token::EOF => break,
                _ => { stmts.push(self.parse_stmt()); }
            }
        }
        (funcs, stmts)
    }

    fn parse_func(&mut self) -> Func {
        self.expect(Token::Fn);
        let name = if let Token::Ident(name) = self.current.clone() {
            self.advance();
            name
        } else { panic!("Expected function name"); };
        self.expect(Token::LParen);
        let mut params = Vec::new();
        if let Token::Ident(_) = self.current {
            if let Token::Ident(p) = self.current.clone() {
                params.push(p);
                self.advance();
                while let Token::Comma = self.current {
                    self.advance();
                    if let Token::Ident(p) = self.current.clone() {
                        params.push(p);
                        self.advance();
                    } else { panic!("Expected parameter name after comma"); }
                }
            }
        }
        self.expect(Token::RParen);
        self.expect(Token::LBrace);
        let mut body = Vec::new();
        while self.current != Token::RBrace && self.current != Token::EOF {
            body.push(self.parse_stmt());
        }
        self.expect(Token::RBrace);
        Func { name, params, body }
    }

    fn parse_stmt(&mut self) -> Stmt {
        match self.current {
            // `import math` or `import numpy as np`
            Token::Import => {
                self.advance();
                let module = if let Token::Ident(name) = self.current.clone() {
                    self.advance();
                    name
                } else {
                    panic!("Expected module name after 'import'");
                };
                // optional `as alias`
                let alias = if let Token::Ident(ref kw) = self.current.clone() {
                    if kw == "as" {
                        self.advance();
                        if let Token::Ident(alias) = self.current.clone() {
                            self.advance();
                            alias
                        } else {
                            panic!("Expected alias after 'as'");
                        }
                    } else {
                        module.clone()
                    }
                } else {
                    module.clone()
                };
                // consume optional semicolon
                if self.current == Token::Semicolon { self.advance(); }
                Stmt::PyImport(alias, module)
            }
            Token::Let => {
                self.advance();
                if let Token::Ident(name) = self.current.clone() {
                    self.advance();
                    self.expect(Token::Assign);
                    let expr = self.parse_expr();
                    self.expect(Token::Semicolon);
                    Stmt::Let(name, expr)
                } else { panic!("Expected identifier after let"); }
            }
            Token::Return => {
                self.advance();
                let expr = self.parse_expr();
                self.expect(Token::Semicolon);
                Stmt::Return(expr)
            }
            Token::Print => {
                self.advance();
                self.expect(Token::LParen);
                let expr = self.parse_expr();
                self.expect(Token::RParen);
                self.expect(Token::Semicolon);
                Stmt::Print(expr)
            }
            Token::While => {
                self.advance();
                self.expect(Token::LParen);
                let cond = self.parse_expr();
                self.expect(Token::RParen);
                let body = self.parse_stmt();
                Stmt::While(cond, Box::new(body))
            }
            Token::LBrace => {
                self.advance();
                let mut stmts = Vec::new();
                while self.current != Token::RBrace && self.current != Token::EOF {
                    stmts.push(self.parse_stmt());
                }
                self.expect(Token::RBrace);
                Stmt::Block(stmts)
            }
            Token::If => {
                self.advance();
                self.expect(Token::LParen);
                let cond = self.parse_expr();
                self.expect(Token::RParen);
                let then_stmt = Box::new(self.parse_stmt());
                let else_stmt = if self.current == Token::Else {
                    self.advance();
                    Some(Box::new(self.parse_stmt()))
                } else { None };
                Stmt::If(cond, then_stmt, else_stmt)
            }
            _ => {
                let expr = self.parse_expr();
                self.expect(Token::Semicolon);
                Stmt::Expr(expr)
            }
        }
    }

    fn parse_expr(&mut self) -> Expr { self.parse_assign() }

    fn parse_assign(&mut self) -> Expr {
        let expr = self.parse_lt();
        if self.current == Token::Assign {
            match expr {
                Expr::Variable(name) => {
                    self.advance();
                    let right = self.parse_assign();
                    Expr::Assign(name, Box::new(right))
                }
                _ => panic!("Invalid left-hand side in assignment"),
            }
        } else { expr }
    }

    fn parse_lt(&mut self) -> Expr {
        let mut expr = self.parse_add_sub();
        while self.current == Token::Less {
            self.advance();
            let right = self.parse_add_sub();
            expr = Expr::Binary(BinOp::Lt, Box::new(expr), Box::new(right));
        }
        expr
    }

    fn parse_add_sub(&mut self) -> Expr {
        let mut expr = self.parse_mul_div();
        loop {
            match self.current {
                Token::Plus => {
                    self.advance();
                    let right = self.parse_mul_div();
                    expr = Expr::Binary(BinOp::Add, Box::new(expr), Box::new(right));
                }
                Token::Minus => {
                    self.advance();
                    let right = self.parse_mul_div();
                    expr = Expr::Binary(BinOp::Sub, Box::new(expr), Box::new(right));
                }
                _ => break,
            }
        }
        expr
    }

    fn parse_mul_div(&mut self) -> Expr {
        let mut expr = self.parse_primary();
        loop {
            match self.current {
                Token::Star => {
                    self.advance();
                    let right = self.parse_primary();
                    expr = Expr::Binary(BinOp::Mul, Box::new(expr), Box::new(right));
                }
                Token::Slash => {
                    self.advance();
                    let right = self.parse_primary();
                    expr = Expr::Binary(BinOp::Div, Box::new(expr), Box::new(right));
                }
                _ => break,
            }
        }
        expr
    }

    fn parse_primary(&mut self) -> Expr {
        let mut expr = match &self.current {
            Token::Number(n) => { let n = *n; self.advance(); Expr::Number(n) }
            Token::String(s) => { let s = s.clone(); self.advance(); Expr::String(s) }
            Token::Ident(name) => {
                let name = name.clone();
                self.advance();
                if self.current == Token::LParen {
                    self.advance();
                    let mut args = Vec::new();
                    if self.current != Token::RParen {
                        args.push(self.parse_expr());
                        while self.current == Token::Comma {
                            self.advance();
                            args.push(self.parse_expr());
                        }
                    }
                    self.expect(Token::RParen);
                    Expr::Call(name, args)
                } else {
                    Expr::Variable(name)
                }
            }
            Token::LParen => {
                self.advance();
                let expr = self.parse_expr();
                self.expect(Token::RParen);
                expr
            }
            _ => panic!("Unexpected token in primary: {:?}", self.current),
        };

        // Postfix: dot-access chains — `obj.attr` and `obj.method(args…)`
        while self.current == Token::Dot {
            self.advance(); // consume '.'
            let attr = if let Token::Ident(a) = self.current.clone() {
                self.advance();
                a
            } else {
                panic!("Expected attribute name after '.'");
            };
            // Check if it's a method call
            if self.current == Token::LParen {
                self.advance();
                let mut args = Vec::new();
                if self.current != Token::RParen {
                    args.push(self.parse_expr());
                    while self.current == Token::Comma {
                        self.advance();
                        args.push(self.parse_expr());
                    }
                }
                self.expect(Token::RParen);
                expr = Expr::MethodCall(Box::new(expr), attr, args);
            } else {
                expr = Expr::GetAttr(Box::new(expr), attr);
            }
        }

        expr
    }
}

pub struct CodeGen {
    func_index: HashMap<String, FunctionId>,
    next_func_id: FunctionId,
    current_func: FunctionId,
    symbols: HashMap<String, ValueId>,
    next_val: ValueId,
    ir: IrProgram,
    current_block: BasicBlockId,
    last_expr_value: Option<ValueId>,
}

impl CodeGen {
    pub fn new() -> Self {
        CodeGen {
            func_index: HashMap::new(),
            next_func_id: 0,
            current_func: 0,
            symbols: HashMap::new(),
            next_val: 0,
            ir: IrProgram::new(),
            current_block: 0,
            last_expr_value: None,
        }
    }

    fn new_value(&mut self) -> ValueId { let v = self.next_val; self.next_val += 1; v }

    fn new_block(&mut self) -> BasicBlockId {
        let func = &mut self.ir.functions[self.current_func];
        let id = func.blocks.len() as BasicBlockId;
        func.blocks.push(BasicBlock { id, instructions: Vec::new() });
        id
    }

    fn add_insn(&mut self, node: IrNode) {
        let func = &mut self.ir.functions[self.current_func];
        func.blocks[self.current_block as usize].instructions.push(node);
    }

    fn ends_with_return(&self, block: BasicBlockId) -> bool {
        let func = &self.ir.functions[self.current_func];
        if let Some(last) = func.blocks[block as usize].instructions.last() {
            matches!(last, IrNode::Return(_))
        } else { false }
    }

    pub fn generate(&mut self, funcs: Vec<Func>, main_stmts: Vec<Stmt>) -> IrProgram {
        // Create synthetic main function
        let main_func = Function {
            name: "main".to_string(),
            params: Vec::new(),
            blocks: vec![BasicBlock { id: 0, instructions: Vec::new() }],
            entry: 0,
        };
        self.ir.functions.push(main_func);
        self.func_index.insert("main".to_string(), 0);
        self.next_func_id = 1;

        // Add user functions
        for f in &funcs {
            let id = self.next_func_id;
            self.ir.functions.push(Function {
                name: f.name.clone(),
                params: f.params.clone(),
                blocks: vec![BasicBlock { id: 0, instructions: Vec::new() }],
                entry: 0,
            });
            self.func_index.insert(f.name.clone(), id);
            self.next_func_id += 1;
        }

        // Generate synthetic main body
        self.current_func = 0;
        self.current_block = 0;
        self.symbols.clear();
        self.last_expr_value = None;
        
        for stmt in &main_stmts {
            self.gen_stmt(stmt);
        }

        if !self.ends_with_return(self.current_block) {
            if let Some(val) = self.last_expr_value {
                self.add_insn(IrNode::Return(val));
            } else {
                let nil_val = self.new_value();
                self.add_insn(IrNode::Const(nil_val, Constant::Nil));
                self.add_insn(IrNode::Return(nil_val));
            }
        }

        // Generate user functions
        for f in &funcs {
            let id = *self.func_index.get(&f.name).unwrap();
            self.current_func = id;
            self.current_block = 0;
            self.symbols.clear();
            self.last_expr_value = None;

            // Parameters
            for (idx, param) in f.params.iter().enumerate() {
                let v = self.new_value();
                self.symbols.insert(param.clone(), v);
                self.add_insn(IrNode::Param(v, idx));
            }

            // Function body
            for stmt in &f.body {
                self.gen_stmt(stmt);
            }

            // Implicit return if needed
            if !self.ends_with_return(self.current_block) {
                if let Some(val) = self.last_expr_value {
                    self.add_insn(IrNode::Return(val));
                } else {
                    let nil_val = self.new_value();
                    self.add_insn(IrNode::Const(nil_val, Constant::Nil));
                    self.add_insn(IrNode::Return(nil_val));
                }
            }
        }

        self.ir.main = 0;
        self.ir.clone()
    }

    fn gen_func_body(&mut self, stmts: &[Stmt]) {
        for stmt in stmts {
            self.gen_stmt(stmt);
        }
    }

    fn gen_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::PyImport(var_name, module_name) => {
                // Emit ImportPython IR node; bind result to var_name in symbol table
                let dst = self.new_value();
                self.add_insn(IrNode::ImportPython(dst, module_name.clone()));
                self.symbols.insert(var_name.clone(), dst);
                self.last_expr_value = Some(dst);
            }
            Stmt::Let(name, expr) => {
                let val = self.gen_expr(expr);
                self.symbols.insert(name.clone(), val);
                self.last_expr_value = Some(val);
            }
            Stmt::Return(expr) => {
                let val = self.gen_expr(expr);
                self.add_insn(IrNode::Return(val));
                self.last_expr_value = None;
            }
            Stmt::Print(expr) => {
                let val = self.gen_expr(expr);
                self.add_insn(IrNode::Print(val));
                self.last_expr_value = Some(val);
            }
            Stmt::While(cond, body) => {
    let cond_block = self.new_block();
    let after_block = self.new_block();
    let body_block = self.new_block();

    self.add_insn(IrNode::Jump(cond_block));

    self.current_block = cond_block;
    let cond_val = self.gen_expr(cond);
    self.add_insn(IrNode::Branch(cond_val, body_block, after_block));

    self.current_block = body_block;
    self.gen_stmt(body);
    self.add_insn(IrNode::Jump(cond_block));

    self.current_block = after_block;
}
            Stmt::Expr(expr) => {
                let val = self.gen_expr(expr);
                self.last_expr_value = Some(val);
            }
            Stmt::Block(block) => {
                for s in block {
                    self.gen_stmt(s);
                }
            }
            Stmt::If(cond, then_stmt, else_stmt) => {
                let cond_val = self.gen_expr(cond);
                let then_block = self.new_block();
                let else_block = self.new_block();
                let merge_block = self.new_block();

                self.add_insn(IrNode::Branch(cond_val, then_block, else_block));

                self.current_block = then_block;
                self.gen_stmt(then_stmt);
                self.add_insn(IrNode::Jump(merge_block));

                self.current_block = else_block;
                if let Some(else_stmt) = else_stmt {
                    self.gen_stmt(else_stmt);
                }
                self.add_insn(IrNode::Jump(merge_block));

                self.current_block = merge_block;
                self.last_expr_value = None;
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
            Expr::String(s) => {
                let v = self.new_value();
                self.add_insn(IrNode::Const(v, Constant::String(s.clone())));
                v
            }
            Expr::Variable(name) => *self.symbols.get(name).unwrap_or_else(|| panic!("Undefined variable {}", name)),
            Expr::Assign(name, expr) => {
                let var_reg = *self.symbols.get(name).expect("Variable not declared");
                // Special case: x = x + y  ->  directly use var_reg as destination
                match &**expr {
                    Expr::Binary(op, left, right) if matches!(**left, Expr::Variable(ref n) if n == name) => {
                        let right_val = self.gen_expr(right);
                        let node = match op {
                            BinOp::Add => IrNode::Add(var_reg, var_reg, right_val),
                            BinOp::Sub => IrNode::Sub(var_reg, var_reg, right_val),
                            BinOp::Mul => IrNode::Mul(var_reg, var_reg, right_val),
                            BinOp::Div => IrNode::Div(var_reg, var_reg, right_val),
                            _ => panic!("Unsupported binary op in assignment"),
                        };
                        self.add_insn(node);
                        var_reg
                    }
                    _ => {
                        let rhs_val = self.gen_expr(expr);
                        self.add_insn(IrNode::Move(var_reg, rhs_val));
                        var_reg
                    }
                }
            }
            Expr::GetAttr(obj_expr, attr_name) => {
                let obj_val = self.gen_expr(obj_expr);
                let dst = self.new_value();
                self.add_insn(IrNode::GetAttr(dst, obj_val, attr_name.clone()));
                dst
            }
            Expr::MethodCall(obj_expr, method_name, args) => {
                // obj.method(args) ≡ tmp = GetAttr(obj, method); CallPython(tmp, args)
                let obj_val = self.gen_expr(obj_expr);
                let callable = self.new_value();
                self.add_insn(IrNode::GetAttr(callable, obj_val, method_name.clone()));
                let mut arg_vals = Vec::new();
                for arg in args {
                    arg_vals.push(self.gen_expr(arg));
                }
                let dst = self.new_value();
                self.add_insn(IrNode::CallPython(dst, callable, arg_vals));
                dst
            }
            Expr::Call(name, args) => {
                if let Some(&func_id) = self.func_index.get(name) {
                    // Known user-defined function — emit a native Call
                    let mut arg_vals = Vec::new();
                    for arg in args {
                        arg_vals.push(self.gen_expr(arg));
                    }
                    let dst = self.new_value();
                    self.add_insn(IrNode::Call(dst, func_id, arg_vals));
                    dst
                } else if let Some(&var_id) = self.symbols.get(name) {
                    // Variable in scope — treat as Python callable
                    let mut arg_vals = Vec::new();
                    for arg in args {
                        arg_vals.push(self.gen_expr(arg));
                    }
                    let dst = self.new_value();
                    self.add_insn(IrNode::CallPython(dst, var_id, arg_vals));
                    dst
                } else {
                    panic!("Unknown function or variable '{}'", name)
                }
            }
            Expr::Binary(op, left, right) => {
                let l = self.gen_expr(left);
                let r = self.gen_expr(right);
                let dst = self.new_value();
                let node = match op {
                    BinOp::Add => IrNode::Add(dst, l, r),
                    BinOp::Sub => IrNode::Sub(dst, l, r),
                    BinOp::Mul => IrNode::Mul(dst, l, r),
                    BinOp::Div => IrNode::Div(dst, l, r),
                    BinOp::Lt => IrNode::Lt(dst, l, r),
                    BinOp::Le => IrNode::Le(dst, l, r),
                };
                self.add_insn(node);
                dst
            }
        }
    }
}