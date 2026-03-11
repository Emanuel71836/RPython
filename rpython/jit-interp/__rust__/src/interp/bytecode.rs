#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCode {
    Add = 0,
    Sub,
    Mul,
    Div,
    Lt,
    Le,
    LoadConst,
    LoadBool,
    LoadNil,
    LoadString,
    Jump,
    Branch,
    Return,
    Print,
    Move,
    Call,
    Inc,
    // Python interop opcodes ---------------------------------------------------
    /// ImportPython dst, string_pool_idx
    /// Load a Python module by name; store as PyObject in dst.
    ImportPython,
    /// GetAttr dst, obj_reg, string_pool_idx
    /// Retrieve attribute `string_pool[imm]` from the PyObject in src1.
    GetAttr,
    /// CallPython dst, callable_reg, nargs
    /// Call the PyObject in src1 with `src2` arguments placed in the
    /// following consecutive registers (reg_base+dst+1 … +nargs).
    /// Result (converted back to Value) is written to dst.
    CallPython,
    /// ConvertToPy dst, src
    /// Convert a native Value (int/bool/string) in src to a Python object in dst.
    ConvertToPy,
    /// ConvertFromPy dst, src
    /// Convert a PyObject in src back to the closest native Value.
    ConvertFromPy,
}

#[derive(Clone, Copy)]
pub struct Instruction(pub u32);

impl Instruction {
    #[inline(always)]
    pub fn encode_rrr(op: OpCode, dst: u8, src1: u8, src2: u8) -> Self {
        let word = ((op as u32) << 24) | ((dst as u32) << 16) | ((src1 as u32) << 8) | (src2 as u32);
        Instruction(word)
    }

    #[inline(always)]
    pub fn encode_rr(op: OpCode, dst: u8, src: u8) -> Self {
        Instruction::encode_rrr(op, dst, src, 0)
    }

    #[inline(always)]
    pub fn encode_imm(op: OpCode, dst: u8, imm: u16) -> Self {
        let word = ((op as u32) << 24) | ((dst as u32) << 16) | (imm as u32);
        Instruction(word)
    }

    #[inline(always)]
    pub fn opcode(&self) -> OpCode {
        unsafe { std::mem::transmute(((self.0 >> 24) & 0xff) as u8) }
    }

    #[inline(always)]
    pub fn dst(&self) -> u8 {
        ((self.0 >> 16) & 0xff) as u8
    }

    #[inline(always)]
    pub fn src1(&self) -> u8 {
        ((self.0 >> 8) & 0xff) as u8
    }

    #[inline(always)]
    pub fn src2(&self) -> u8 {
        (self.0 & 0xff) as u8
    }

    #[inline(always)]
    pub fn imm(&self) -> u16 {
        (self.0 & 0xffff) as u16
    }
}