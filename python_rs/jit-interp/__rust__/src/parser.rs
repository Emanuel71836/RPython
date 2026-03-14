// parser.rs — wraps rustpython-parser to give us a full Python AST.
// The downstream code (lib.rs) converts this AST to the frontend IR types.

pub use rustpython_ast::{
    Stmt     as PyStmt,
    Expr     as PyExpr,
    Operator, UnaryOp, BoolOp, CmpOp,
    Constant,
};

pub use rustpython_parser::{parse, Mode};

/// Parse a Python source string and return the top-level statement list.
pub fn parse_python(source: &str) -> Result<Vec<PyStmt>, String> {
    let ast = parse(source, Mode::Module, "<python_rs>")
        .map_err(|e| format!("Parse error: {e}"))?;
    match ast {
        rustpython_ast::Mod::Module(m) => Ok(m.body),
        _ => Err("Expected a module".into()),
    }
}