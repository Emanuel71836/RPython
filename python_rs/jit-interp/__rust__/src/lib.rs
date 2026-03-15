// lib.rs — python_rs runtime library.
// parses 100% python syntax via rustpython‑parser, lowers to our ir, and
// executes with the cranelift jit + bytecode interpreter vm.
//
// cli usage (from python):  python_rs.python_rs("path/to/file.ry")
// pyo3 api:                 python_rs.run_file("path/to/file.ry")
//                           python_rs.run_code("x = 1 + 2\nprint(x)")

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use once_cell::sync::Lazy;
use std::sync::Mutex;
use rayon::prelude::*;
use std::path::Path;
use std::fs;

// sub‑modules
#[path = "interp/arena.rs"]    mod arena;
#[path = "interp/bytecode.rs"] mod bytecode;
#[path = "interp/frontend.rs"] mod frontend;
#[path = "interp/ir.rs"]       mod ir;
#[path = "interp/lower.rs"]    mod lower;
#[path = "interp/value.rs"]    mod value;
#[path = "interp/vm.rs"]       mod vm;
#[path = "interp/jit.rs"]      mod jit;

mod parser;

use frontend::{
    CodeGen,
    Func          as FrontendFunc,
    Stmt          as FrontendStmt,
    Expr          as FrontendExpr,
    BinOp         as FrontendBinOp,
    CmpOp         as FrontendCmpOp,
    BoolOpKind,
    UnaryOpKind,
};
use lower::LoweringContext;
use vm::VM;
use value::Value;

// rustpython ast types
use parser::{PyStmt, PyExpr, Operator, UnaryOp, BoolOp, CmpOp, Constant};

static mut JIT_THRESHOLD: usize = 10;

static COMPILATION_CACHE: Lazy<Mutex<HashMap<String,
    (Vec<(Vec<bytecode::Instruction>, usize, usize)>, Vec<String>)>>>
    = Lazy::new(|| Mutex::new(HashMap::new()));

// python ast → frontend expr

fn convert_expr(expr: &PyExpr) -> FrontendExpr {
    match expr {
        // constants
        PyExpr::Constant(c) => {
            match &c.value {
                Constant::Int(n)   => FrontendExpr::Number(n.try_into().unwrap_or(0)),
                Constant::Float(f) => FrontendExpr::Float(*f),
                Constant::Bool(b)  => FrontendExpr::Bool(*b),
                Constant::Str(s)   => FrontendExpr::String(s.clone()),
                Constant::None     => FrontendExpr::None,
                _                  => FrontendExpr::None,
            }
        }

        // name / variable
        PyExpr::Name(n) => {
            match n.id.as_str() {
                "true"  => FrontendExpr::Bool(true),
                "false" => FrontendExpr::Bool(false),
                "none"  => FrontendExpr::None,
                other   => FrontendExpr::Variable(other.to_string()),
            }
        }

        // binary operations
        PyExpr::BinOp(b) => {
            let op = match b.op {
                Operator::Add      => FrontendBinOp::Add,
                Operator::Sub      => FrontendBinOp::Sub,
                Operator::Mult     => FrontendBinOp::Mul,
                Operator::Div      => FrontendBinOp::Div,
                Operator::FloorDiv => FrontendBinOp::FloorDiv,
                Operator::Mod      => FrontendBinOp::Mod,
                Operator::Pow      => FrontendBinOp::Pow,
                _                  => FrontendBinOp::Add,
            };
            FrontendExpr::Binary(
                op,
                Box::new(convert_expr(&b.left)),
                Box::new(convert_expr(&b.right)),
            )
        }

        // comparisons
        PyExpr::Compare(c) => {
            let mut exprs: Vec<FrontendExpr> = Vec::new();
            let mut left = convert_expr(&c.left);
            for (op, right_py) in c.ops.iter().zip(c.comparators.iter()) {
                let right  = convert_expr(right_py);
                let cmp_op = match op {
                    CmpOp::Eq    => FrontendCmpOp::Eq,
                    CmpOp::NotEq => FrontendCmpOp::NotEq,
                    CmpOp::Lt    => FrontendCmpOp::Lt,
                    CmpOp::LtE   => FrontendCmpOp::Le,
                    CmpOp::Gt    => FrontendCmpOp::Gt,
                    CmpOp::GtE   => FrontendCmpOp::Ge,
                    CmpOp::Is    => FrontendCmpOp::Is,
                    CmpOp::IsNot => FrontendCmpOp::IsNot,
                    CmpOp::In    => FrontendCmpOp::In,
                    CmpOp::NotIn => FrontendCmpOp::NotIn,
                };
                exprs.push(FrontendExpr::Compare(
                    cmp_op,
                    Box::new(left.clone()),
                    Box::new(right.clone()),
                ));
                left = right;
            }
            if exprs.len() == 1 { exprs.remove(0) }
            else { FrontendExpr::BoolOp(BoolOpKind::And, exprs) }
        }

        // boolean operations
        PyExpr::BoolOp(b) => {
            let kind = match b.op {
                BoolOp::And => BoolOpKind::And,
                BoolOp::Or  => BoolOpKind::Or,
            };
            FrontendExpr::BoolOp(kind, b.values.iter().map(convert_expr).collect())
        }

        // unary operations
        PyExpr::UnaryOp(u) => {
            match u.op {
                UnaryOp::Not  => FrontendExpr::Not(Box::new(convert_expr(&u.operand))),
                UnaryOp::USub => FrontendExpr::Unary(UnaryOpKind::Neg, Box::new(convert_expr(&u.operand))),
                UnaryOp::UAdd => FrontendExpr::Unary(UnaryOpKind::Pos, Box::new(convert_expr(&u.operand))),
                _             => convert_expr(&u.operand),
            }
        }

        // function call
        PyExpr::Call(call) => {
            let pos_args: Vec<FrontendExpr> = call.args.iter().map(convert_expr).collect();

            // collect keyword arguments.
            let kw_pairs: Vec<(String, FrontendExpr)> = call.keywords.iter()
                .filter_map(|kw| {
                    // kw.arg is none for **kwargs splat — skip those for now.
                    let key = kw.arg.as_ref()?.as_str().to_string();
                    let val = convert_expr(&kw.value);
                    Some((key, val))
                })
                .collect();

            if kw_pairs.is_empty() {
                // fast path — no kwargs.
                match call.func.as_ref() {
                    PyExpr::Name(n) => FrontendExpr::Call(n.id.as_str().to_string(), pos_args),
                    PyExpr::Attribute(a) => FrontendExpr::MethodCall {
                        obj:    Box::new(convert_expr(&a.value)),
                        method: a.attr.as_str().to_string(),
                        args:   pos_args,
                    },
                    other => FrontendExpr::PyCallExpr {
                        callable: Box::new(convert_expr(other)),
                        args:     pos_args,
                    },
                }
            } else {
                // kwargs path — encode as special sentinel strings.
                let callable_expr = match call.func.as_ref() {
                    PyExpr::Attribute(a) => {
                        FrontendExpr::GetAttr {
                            obj:  Box::new(convert_expr(&a.value)),
                            attr: a.attr.as_str().to_string(),
                        }
                    }
                    other => convert_expr(other),
                };

                let mut all_args = pos_args;
                for (key, val) in kw_pairs {
                    all_args.push(FrontendExpr::String(format!("__kw__:{}", key)));
                    all_args.push(val);
                }

                FrontendExpr::PyCallExpr {
                    callable: Box::new(callable_expr),
                    args:     all_args,
                }
            }
        }

        // attribute access
        PyExpr::Attribute(a) => FrontendExpr::GetAttr {
            obj:  Box::new(convert_expr(&a.value)),
            attr: a.attr.as_str().to_string(),
        },

        // subscript obj[idx]
        PyExpr::Subscript(s) => FrontendExpr::Subscript {
            obj: Box::new(convert_expr(&s.value)),
            idx: Box::new(convert_expr(&s.slice)),
        },

        // list literal
        PyExpr::List(l) => FrontendExpr::List(l.elts.iter().map(convert_expr).collect()),

                PyExpr::Dict(d) => {
            let mut items = Vec::new();
            // zip keys and values; keys are Option<Expr> (None for ** unpacking)
            for (k_opt, v) in d.keys.iter().zip(d.values.iter()) {
                if let Some(k) = k_opt {
                    items.push((convert_expr(k), convert_expr(v)));
                } else {
                    // skip ** unpacking for now
                    continue;
                }
            }
            FrontendExpr::Dict(items)
        }

        // tuple (treat as list)
        PyExpr::Tuple(t) => FrontendExpr::List(t.elts.iter().map(convert_expr).collect()),

        // ternary (a if cond else b)
        PyExpr::IfExp(i) => convert_expr(&i.body), // todo: full ternary support

        // walrus operator (x := expr)
        PyExpr::NamedExpr(n) => {
            let name = match n.target.as_ref() {
                PyExpr::Name(nm) => nm.id.as_str().to_string(),
                _                => "__walrus__".into(),
            };
            FrontendExpr::Assign(name, Box::new(convert_expr(&n.value)))
        }

        // f‑string
        PyExpr::JoinedStr(parts) => {
            let mut string_parts = Vec::new();
            for part in &parts.values {
                let part_expr = match part {
                    PyExpr::Constant(c) => {
                        if let Constant::Str(s) = &c.value {
                            FrontendExpr::String(s.clone())
                        } else {
                            FrontendExpr::String(String::new())
                        }
                    }
                    PyExpr::FormattedValue(fv) => {
                        let value_expr = convert_expr(&fv.value);
                        FrontendExpr::MethodCall {
                            obj: Box::new(value_expr),
                            method: "__format__".to_string(),
                            args: vec![FrontendExpr::String("".to_string())],
                        }
                    }
                    _ => FrontendExpr::String(String::new()),
                };
                string_parts.push(part_expr);
            }
            // build list of parts and join with empty string
            let list_expr = FrontendExpr::List(string_parts);
            let empty_str = FrontendExpr::String("".to_string());
            FrontendExpr::MethodCall {
                obj: Box::new(empty_str),
                method: "join".to_string(),
                args: vec![list_expr],
            }
        }

        // formatted value (standalone, e.g. {x})
        PyExpr::FormattedValue(fv) => {
            let value_expr = convert_expr(&fv.value);
            FrontendExpr::MethodCall {
                obj: Box::new(value_expr),
                method: "__format__".to_string(),
                args: vec![FrontendExpr::String("".to_string())],
            }
        }

        _ => FrontendExpr::None,
    }
}

// python ast → frontend stmt

fn split_stmts(
    stmts: Vec<PyStmt>,
    declared: &mut HashSet<String>,
    source_dir: &Path,
    already_loaded: &mut HashSet<String>,
    extra_funcs: &mut Vec<FrontendFunc>,
) -> Vec<FrontendStmt> {
    let mut out = Vec::new();
    for stmt in stmts {
        convert_stmt(stmt, declared, source_dir, already_loaded, extra_funcs, &mut out);
    }
    out
}

fn convert_stmt(
    stmt: PyStmt,
    declared: &mut HashSet<String>,
    source_dir: &Path,
    already_loaded: &mut HashSet<String>,
    extra_funcs: &mut Vec<FrontendFunc>,
    out: &mut Vec<FrontendStmt>,
) {
    match stmt {
        // function definition
        PyStmt::FunctionDef(f) => {
            let name   = f.name.as_str().to_string();
            let params: Vec<String> = f.args.args.iter()
                .map(|a| a.def.arg.as_str().to_string())
                .collect();
            let mut fn_declared: HashSet<String> = params.iter().cloned().collect();
            let mut nested = Vec::new();
            let body = split_stmts(f.body, &mut fn_declared, source_dir, already_loaded, &mut nested);
            extra_funcs.extend(nested);
            extra_funcs.push(FrontendFunc { name, params, body });
        }

        // import
        PyStmt::Import(i) => {
            for alias in i.names {
                let module    = alias.name.as_str().to_string();
                let alias_str = alias.asname
                    .as_ref()
                    .map(|a| a.as_str().to_string())
                    .unwrap_or_else(|| module.clone());
                let ry_path = source_dir.join(format!("{}.ry", module.replace('.', "/")));
                if ry_path.exists() {
                    if let Err(e) = load_ry_file(ry_path.to_str().unwrap(), already_loaded, extra_funcs) {
                        eprintln!("warning: {e}");
                    }
                } else {
                    out.push(FrontendStmt::PythonImport { module, alias: alias_str });
                }
            }
        }

        PyStmt::ImportFrom(i) => {
            let base = i.module.as_ref()
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            for alias in i.names {
                let name      = alias.name.as_str().to_string();
                let full      = if base.is_empty() { name.clone() } else { format!("{base}.{name}") };
                let alias_str = alias.asname
                    .as_ref()
                    .map(|a| a.as_str().to_string())
                    .unwrap_or_else(|| name.clone());
                out.push(FrontendStmt::PythonImport { module: full, alias: alias_str });
            }
        }

        // assignments
        PyStmt::Assign(a) => {
            let rhs = convert_expr(&a.value);
            for target in &a.targets {
                match target {
                    PyExpr::Name(n) => {
                        let name = n.id.as_str().to_string();
                        if declared.insert(name.clone()) {
                            out.push(FrontendStmt::Let(name, rhs.clone()));
                        } else {
                            out.push(FrontendStmt::Assign(name, rhs.clone()));
                        }
                    }
                    _ => { out.push(FrontendStmt::Expr(rhs.clone())); }
                }
            }
        }

        PyStmt::AnnAssign(a) => {
            if let Some(val) = a.value {
                let rhs = convert_expr(&val);
                if let PyExpr::Name(n) = a.target.as_ref() {
                    let name = n.id.as_str().to_string();
                    if declared.insert(name.clone()) {
                        out.push(FrontendStmt::Let(name, rhs));
                    } else {
                        out.push(FrontendStmt::Assign(name, rhs));
                    }
                }
            }
        }

        PyStmt::AugAssign(a) => {
            if let PyExpr::Name(n) = a.target.as_ref() {
                let name = n.id.as_str().to_string();
                let op   = match a.op {
                    Operator::Add      => FrontendBinOp::Add,
                    Operator::Sub      => FrontendBinOp::Sub,
                    Operator::Mult     => FrontendBinOp::Mul,
                    Operator::Div      => FrontendBinOp::Div,
                    Operator::FloorDiv => FrontendBinOp::FloorDiv,
                    Operator::Mod      => FrontendBinOp::Mod,
                    Operator::Pow      => FrontendBinOp::Pow,
                    _                  => FrontendBinOp::Add,
                };
                out.push(FrontendStmt::AugAssign(name, op, convert_expr(&a.value)));
            }
        }

        // return
        PyStmt::Return(r) => {
            let expr = r.value
                .as_ref()
                .map(|e| convert_expr(e))
                .unwrap_or(FrontendExpr::None);
            out.push(FrontendStmt::Return(expr));
        }

        // if / elif / else
        PyStmt::If(i) => {
            let cond = convert_expr(&i.test);
            let then_body = split_stmts(i.body, declared, source_dir, already_loaded, extra_funcs);
            let else_body = if !i.orelse.is_empty() {
                let stmts = split_stmts(i.orelse, declared, source_dir, already_loaded, extra_funcs);
                Some(Box::new(FrontendStmt::Block(stmts)))
            } else {
                None
            };
            out.push(FrontendStmt::If(
                cond,
                Box::new(FrontendStmt::Block(then_body)),
                else_body,
            ));
        }

        // while
        PyStmt::While(w) => {
            let cond = convert_expr(&w.test);
            let body = split_stmts(w.body, declared, source_dir, already_loaded, extra_funcs);
            out.push(FrontendStmt::While(cond, Box::new(FrontendStmt::Block(body))));
        }

        // for
        PyStmt::For(f) => {
            let iter   = convert_expr(&f.iter);
            let target = match f.target.as_ref() {
                PyExpr::Name(n) => n.id.as_str().to_string(),
                _               => "__iter_target__".into(),
            };
            declared.insert(target.clone());
            let body = split_stmts(f.body, declared, source_dir, already_loaded, extra_funcs);
            out.push(FrontendStmt::For {
                target,
                iter,
                body: Box::new(FrontendStmt::Block(body)),
            });
        }

        // expr statement
        PyStmt::Expr(e) => {
            if let PyExpr::Call(call) = e.value.as_ref() {
                if let PyExpr::Name(n) = call.func.as_ref() {
                    if n.id.as_str() == "print" {
                        for arg in &call.args {
                            out.push(FrontendStmt::Print(convert_expr(arg)));
                        }
                        if call.args.is_empty() {
                            out.push(FrontendStmt::Print(FrontendExpr::String(String::new())));
                        }
                        return;
                    }
                }
            }
            out.push(FrontendStmt::Expr(convert_expr(&e.value)));
        }

        // pass / break / continue
        PyStmt::Pass(_) | PyStmt::Break(_) | PyStmt::Continue(_) => {}

        // class
        PyStmt::ClassDef(c) => {
            let body = split_stmts(c.body, declared, source_dir, already_loaded, extra_funcs);
            out.extend(body);
        }

        _ => {}
    }
}

// .ry file loader

fn load_ry_file(
    filename: &str,
    already_loaded: &mut HashSet<String>,
    extra_funcs: &mut Vec<FrontendFunc>,
) -> Result<(), String> {
    let canonical = Path::new(filename)
        .canonicalize()
        .map_err(|e| format!("cannot canonicalize {filename}: {e}"))?;
    let key = canonical.to_string_lossy().to_string();
    if !already_loaded.insert(key) { return Ok(()); }

    let code = fs::read_to_string(filename)
        .map_err(|e| format!("cannot read {filename}: {e}"))?;
    let stmts = parser::parse_python(&code)
        .map_err(|e| format!("parse error in {filename}: {e}"))?;

    let dir = Path::new(filename).parent().unwrap_or(Path::new(".")).to_path_buf();
    let _declared: HashSet<String> = HashSet::new();
    let mut nested = Vec::new();

    for stmt in stmts {
        if let PyStmt::FunctionDef(f) = stmt {
            let name   = f.name.as_str().to_string();
            let params: Vec<String> = f.args.args.iter()
                .map(|a| a.def.arg.as_str().to_string())
                .collect();
            let mut fn_declared: HashSet<String> = params.iter().cloned().collect();
            let body = split_stmts(f.body, &mut fn_declared, &dir, already_loaded, &mut nested);
            extra_funcs.push(FrontendFunc { name, params, body });
        }
    }
    extra_funcs.extend(nested);
    Ok(())
}

// pipeline that returns a Value

fn run_pipeline_value(funcs: Vec<FrontendFunc>, main_stmts: Vec<FrontendStmt>) -> Result<Value, String> {
    let mut codegen   = CodeGen::new();
    let ir_program    = codegen.generate(funcs, main_stmts);
    let mut lower_ctx = LoweringContext::new();
    let (functions, string_pool, float_pool) = lower_ctx.lower_program(&ir_program);
    let mut vm = VM::new(
        functions,
        string_pool,
        float_pool,
        1024 * 1024,
        unsafe { JIT_THRESHOLD },
        1024,
    );
    vm.run()
}

fn run_pipeline_string(funcs: Vec<FrontendFunc>, main_stmts: Vec<FrontendStmt>) -> Result<String, String> {
    run_pipeline_value(funcs, main_stmts).map(|val| format!("{:?}", val))
}

fn compile_source_value(code: &str) -> Result<Value, String> {
    let stmts = parser::parse_python(code)?;
    let dir   = Path::new(".");
    let mut declared     = HashSet::new();
    let mut extra_funcs  = Vec::new();
    let mut already      = HashSet::new();
    let main_stmts = split_stmts(stmts, &mut declared, dir, &mut already, &mut extra_funcs);
    run_pipeline_value(extra_funcs, main_stmts)
}

fn compile_source(code: &str) -> Result<String, String> {
    compile_source_value(code).map(|val| format!("{:?}", val))
}

pub fn compile_file_value(filename: &str) -> Result<Value, String> {
    let code  = fs::read_to_string(filename)
        .map_err(|e| format!("cannot read {filename}: {e}"))?;
    let stmts = parser::parse_python(&code)
        .map_err(|e| format!("parse error in {filename}: {e}"))?;

    let dir = Path::new(filename).parent().unwrap_or(Path::new("."));
    let mut declared    = HashSet::new();
    let mut extra_funcs = Vec::new();
    let mut already     = HashSet::new();
    let main_stmts = split_stmts(stmts, &mut declared, dir, &mut already, &mut extra_funcs);
    run_pipeline_value(extra_funcs, main_stmts)
}

pub fn compile_file(filename: &str) -> Result<String, String> {
    compile_file_value(filename).map(|val| format!("{:?}", val))
}

// pyo3 entry‑points

/// `python_rs.python_rs("file.ry")` — primary entry point.
#[pyfunction]
#[pyo3(name = "python_rs")]
fn python_rs_run(py: Python, filename: &str) -> PyResult<String> {
    py.allow_threads(|| {
        compile_file(filename)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    })
}

/// `python_rs.run_code("x = 1 + 2\nprint(x)")`
#[pyfunction]
fn run_code(py: Python, code: &str) -> PyResult<String> {
    py.allow_threads(|| {
        compile_source(code)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    })
}

/// `python_rs.run_file("file.ry")` — alias for python_rs_run.
#[pyfunction]
fn run_file(py: Python, filename: &str) -> PyResult<String> {
    py.allow_threads(|| {
        compile_file(filename)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    })
}

/// compile many snippets in parallel.
#[pyfunction]
fn compile_many(py: Python, codes: Vec<String>) -> PyResult<Vec<String>> {
    let results: Vec<Result<String, String>> = py.allow_threads(|| {
        codes.into_par_iter()
             .map(|code| compile_source(&code))
             .collect()
    });
    results.into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

#[pyfunction]
fn set_jit_threshold(threshold: usize) { unsafe { JIT_THRESHOLD = threshold; } }

#[pyfunction]
fn set_thread_count(n: usize) {
    rayon::ThreadPoolBuilder::new().num_threads(n).build_global().unwrap();
}

// cli entry point
use pyo3::types::PyModule;
use std::process;

#[pyfunction]
fn run_cli(py: Python) -> PyResult<()> {
    // get sys.argv from python (works even when embedded)
    let sys = py.import("sys")?;
    let argv: Vec<String> = sys.getattr("argv")?.extract()?;

    if argv.len() != 2 {
        eprintln!("usage: python.rs <file.ry>");
        process::exit(1);
    }

    let filename = &argv[1];
    match py.allow_threads(|| compile_file_value(filename)) {
        Ok(val) => {
            if !val.is_nil() {
                println!("{:?}", val);
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

#[pymodule]
fn python_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python_rs_run,       m)?)?;
    m.add_function(wrap_pyfunction!(run_code,          m)?)?;
    m.add_function(wrap_pyfunction!(run_file,          m)?)?;
    m.add_function(wrap_pyfunction!(compile_many,      m)?)?;
    m.add_function(wrap_pyfunction!(set_jit_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(set_thread_count,  m)?)?;
    m.add_function(wrap_pyfunction!(run_cli,            m)?)?;
    Ok(())
}