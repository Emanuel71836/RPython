pub type ValueId     = u32;
pub type BasicBlockId = u32;
pub type FunctionId  = usize;

#[derive(Debug, Clone)]
pub enum Constant {
    Int(i64),
    Bool(bool),
    Nil,
    String(String),
    Float(f64),  // new
}

#[derive(Debug, Clone)]
pub enum IrNode {
    // arithmetic comparisons
    Add(ValueId, ValueId, ValueId),
    Sub(ValueId, ValueId, ValueId),
    Mul(ValueId, ValueId, ValueId),
    Div(ValueId, ValueId, ValueId),
    Lt (ValueId, ValueId, ValueId),
    Le (ValueId, ValueId, ValueId),
    // constants loads
    Const(ValueId, Constant),
    Param(ValueId, usize),
    Move (ValueId, ValueId),
    // control flow
    Jump  (BasicBlockId),
    Branch(ValueId, BasicBlockId, BasicBlockId),
    Return(ValueId),
    // builtins
    Print(ValueId),
    // ry to ry call
    Call(ValueId, FunctionId, Vec<ValueId>),

    // python interop
    ImportPython(ValueId, String),
    GetAttr(ValueId, ValueId, String),
    PyCall(ValueId, ValueId, Vec<ValueId>),
    CallMethod(ValueId, ValueId, String, Vec<ValueId>),
    ConvertFromPy(ValueId, ValueId),
}

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id:           BasicBlockId,
    pub instructions: Vec<IrNode>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name:   String,
    pub params: Vec<String>,
    pub blocks: Vec<BasicBlock>,
    pub entry:  BasicBlockId,
}

#[derive(Debug, Clone)]
pub struct IrProgram {
    pub functions: Vec<Function>,
    pub main:      FunctionId,
}

impl IrProgram {
    pub fn new() -> Self { IrProgram { functions: Vec::new(), main: 0 } }
}