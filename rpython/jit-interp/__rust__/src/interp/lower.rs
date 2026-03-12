use crate::ir::*;
use crate::bytecode::{Instruction, OpCode};
use std::collections::HashMap;
use std::rc::Rc;

pub struct LoweringContext {
    reg_map: Vec<u8>,
    next_reg: u8,
    bytecode: Vec<Instruction>,
    block_starts: HashMap<BasicBlockId, usize>,
    pending_jumps: Vec<(usize, BasicBlockId, bool)>,
    string_pool: Vec<String>,
    string_map: HashMap<String, u16>,
}

impl LoweringContext {
    pub fn new() -> Self {
        LoweringContext {
            reg_map: Vec::new(),
            next_reg: 0,
            bytecode: Vec::new(),
            block_starts: HashMap::new(),
            pending_jumps: Vec::new(),
            string_pool: Vec::new(),
            string_map: HashMap::new(),
        }
    }

    /// Intern a string into the pool and return its u16 index.
    fn intern_string(&mut self, s: &str) -> u16 {
        if let Some(&i) = self.string_map.get(s) {
            i
        } else {
            let i = self.string_pool.len() as u16;
            self.string_pool.push(s.to_string());
            self.string_map.insert(s.to_string(), i);
            i
        }
    }

    fn ensure_reg(&mut self, v: ValueId) -> u8 {
        while self.reg_map.len() <= v as usize {
            self.reg_map.push(self.next_reg);
            self.next_reg += 1;
        }
        self.reg_map[v as usize]
    }

    fn lower_function(&mut self, func: &Function) -> (Vec<Instruction>, usize) {
        self.reg_map.clear();
        self.next_reg = 0;
        self.bytecode.clear();
        self.block_starts.clear();
        self.pending_jumps.clear();

        for (idx, _) in func.params.iter().enumerate() {
            self.ensure_reg(idx as ValueId);
        }

        for block in &func.blocks {
            self.block_starts.insert(block.id, self.bytecode.len());
            for node in &block.instructions {
                match node {
                    IrNode::Add(dst, a, b) => {
                        let rd = self.ensure_reg(*dst);
                        let rs1 = self.ensure_reg(*a);
                        let rs2 = self.ensure_reg(*b);
                        self.bytecode.push(Instruction::encode_rrr(OpCode::Add, rd, rs1, rs2));
                    }
                    IrNode::Sub(dst, a, b) => {
                        let rd = self.ensure_reg(*dst);
                        let rs1 = self.ensure_reg(*a);
                        let rs2 = self.ensure_reg(*b);
                        self.bytecode.push(Instruction::encode_rrr(OpCode::Sub, rd, rs1, rs2));
                    }
                    IrNode::Mul(dst, a, b) => {
                        let rd = self.ensure_reg(*dst);
                        let rs1 = self.ensure_reg(*a);
                        let rs2 = self.ensure_reg(*b);
                        self.bytecode.push(Instruction::encode_rrr(OpCode::Mul, rd, rs1, rs2));
                    }
                    IrNode::Div(dst, a, b) => {
                        let rd = self.ensure_reg(*dst);
                        let rs1 = self.ensure_reg(*a);
                        let rs2 = self.ensure_reg(*b);
                        self.bytecode.push(Instruction::encode_rrr(OpCode::Div, rd, rs1, rs2));
                    }
                    IrNode::Lt(dst, a, b) => {
                        let rd = self.ensure_reg(*dst);
                        let rs1 = self.ensure_reg(*a);
                        let rs2 = self.ensure_reg(*b);
                        self.bytecode.push(Instruction::encode_rrr(OpCode::Lt, rd, rs1, rs2));
                    }
                    IrNode::Le(dst, a, b) => {
                        let rd = self.ensure_reg(*dst);
                        let rs1 = self.ensure_reg(*a);
                        let rs2 = self.ensure_reg(*b);
                        self.bytecode.push(Instruction::encode_rrr(OpCode::Le, rd, rs1, rs2));
                    }
                    IrNode::Const(dst, Constant::Int(i)) => {
                        let rd = self.ensure_reg(*dst);
                        self.bytecode.push(Instruction::encode_imm(OpCode::LoadConst, rd, *i as u16));
                    }
                    IrNode::Const(dst, Constant::Bool(b)) => {
                        let rd = self.ensure_reg(*dst);
                        let imm = if *b { 1 } else { 0 };
                        self.bytecode.push(Instruction::encode_imm(OpCode::LoadBool, rd, imm));
                    }
                    IrNode::Const(dst, Constant::Nil) => {
                        let rd = self.ensure_reg(*dst);
                        self.bytecode.push(Instruction::encode_imm(OpCode::LoadNil, rd, 0));
                    }
                    IrNode::Const(dst, Constant::String(s)) => {
                        let rd = self.ensure_reg(*dst);
                        let idx = if let Some(&i) = self.string_map.get(s) {
                            i
                        } else {
                            let i = self.string_pool.len() as u16;
                            self.string_pool.push(s.clone());
                            self.string_map.insert(s.clone(), i);
                            i
                        };
                        self.bytecode.push(Instruction::encode_imm(OpCode::LoadString, rd, idx));
                    }
                    IrNode::Param(dst, idx) => {
                        let rd = self.ensure_reg(*dst);
                        self.bytecode.push(Instruction::encode_rr(OpCode::Move, rd, *idx as u8));
                    }
                    IrNode::Jump(target) => {
                        let pos = self.bytecode.len();
                        self.bytecode.push(Instruction::encode_imm(OpCode::Jump, 0, 0));
                        self.pending_jumps.push((pos, *target, false));
                    }
                    IrNode::Branch(cond, _t, f) => {
                        // branch jumps to f (else block) when condition is FALSE
                        // falls through to the next block (then block) when TRUE
                        let cond_reg = self.ensure_reg(*cond);
                        let pos = self.bytecode.len();
                        self.bytecode.push(Instruction::encode_imm(OpCode::Branch, cond_reg, 0));
                        self.pending_jumps.push((pos, *f, true));
                    }
                    IrNode::Return(val) => {
                        let rval = self.ensure_reg(*val);
                        self.bytecode.push(Instruction::encode_imm(OpCode::Return, rval, 0));
                    }
                    IrNode::Print(val) => {
                        let rval = self.ensure_reg(*val);
                        self.bytecode.push(Instruction::encode_imm(OpCode::Print, rval, 0));
                    }
                    IrNode::Call(dst, func_idx, args) => {
                        for (i, arg_reg) in args.iter().enumerate() {
                            let src = self.ensure_reg(*arg_reg);
                            self.bytecode.push(Instruction::encode_rr(OpCode::Move, i as u8, src));
                        }
                        self.bytecode.push(Instruction::encode_imm(OpCode::Call, 0, *func_idx as u16));
                        let rd = self.ensure_reg(*dst);
                        if rd != 0 {
                            self.bytecode.push(Instruction::encode_rr(OpCode::Move, rd, 0));
                        }
                    }
                    IrNode::Move(dst, src) => {
                        let rd = self.ensure_reg(*dst);
                        let rs = self.ensure_reg(*src);
                        self.bytecode.push(Instruction::encode_rr(OpCode::Move, rd, rs));
                    }
                    // ----- Python interop IR nodes -----
                    //
                    // Encoding strategy: the string name/attr is stored in the
                    // string pool and referenced via its u16 index in imm.
                    // For GetAttr / CallPython the object/callable register is
                    // stored in src1 (via encode_rrr with src2=nargs or 0).

                    IrNode::ImportPython(dst, module_name) => {
                        let rd = self.ensure_reg(*dst);
                        let idx = self.intern_string(module_name);
                        // encode_imm: op=ImportPython, dst=rd, imm=string_idx
                        self.bytecode.push(Instruction::encode_imm(OpCode::ImportPython, rd, idx));
                    }

                    IrNode::GetAttr(dst, obj, attr_name) => {
                        let rd  = self.ensure_reg(*dst);
                        let ro  = self.ensure_reg(*obj);
                        let idx = self.intern_string(attr_name);
                        // encode_rrr: op=GetAttr, dst=rd, src1=ro, src2=0; imm lives in lower 16 bits
                        // We reuse encode_imm but pack the object register into the dst field and
                        // store the string index in imm, then recover obj reg from src1 in the VM.
                        // Use a full rrr form: dst=rd, src1=ro, then patch imm via the word.
                        // Simplest: emit encode_rrr so src1 carries ro, and imm in lower 16.
                        // Encoding: op[31:24] | dst[23:16] | obj_reg[15:8] | attr_idx[7:0]
                        // attr_idx is asserted < 256 below.
                        let raw = ((OpCode::GetAttr as u32) << 24)
                            | ((rd as u32) << 16)
                            | ((ro as u32) << 8)
                            | (idx & 0xff) as u32;
                        // idx may be > 255; store high byte separately using src2 as high byte.
                        // For now assert idx < 256 (practically safe for typical programs).
                        assert!(idx <= 0xff, "Too many strings in pool for GetAttr imm8 (>255)");
                        self.bytecode.push(Instruction(raw));
                    }

                    IrNode::CallPython(dst, callable, args) => {
                        let rd   = self.ensure_reg(*dst);
                        let rc   = self.ensure_reg(*callable);
                        let nargs = args.len() as u8;
                        // Arguments are placed into registers rd+1, rd+2, …
                        // The VM reads nargs Values starting at reg_base + rd + 1.
                        for (i, arg_id) in args.iter().enumerate() {
                            let rsrc = self.ensure_reg(*arg_id);
                            // Move arg into slot rd+1+i relative to frame
                            let slot = rd + 1 + i as u8;
                            self.bytecode.push(Instruction::encode_rr(OpCode::Move, slot, rsrc));
                        }
                        // encode: dst=rd, src1=rc, src2=nargs
                        self.bytecode.push(Instruction::encode_rrr(OpCode::CallPython, rd, rc, nargs));
                    }

                    IrNode::ConvertToPy(dst, src) => {
                        let rd = self.ensure_reg(*dst);
                        let rs = self.ensure_reg(*src);
                        self.bytecode.push(Instruction::encode_rr(OpCode::ConvertToPy, rd, rs));
                    }

                    IrNode::ConvertFromPy(dst, src) => {
                        let rd = self.ensure_reg(*dst);
                        let rs = self.ensure_reg(*src);
                        self.bytecode.push(Instruction::encode_rr(OpCode::ConvertFromPy, rd, rs));
                    }
                } // end match node
            } // end for node in &block.instructions
        } // end for block in &func.blocks

        for (pos, target, is_branch) in &self.pending_jumps {
            let target_pc = *self.block_starts.get(target).unwrap() as u16;
            let insn = self.bytecode[*pos];
            let op = insn.opcode();
            let dst = insn.dst();
            let new_insn = if *is_branch {
                Instruction::encode_imm(op, dst, target_pc)
            } else {
                Instruction::encode_imm(op, 0, target_pc)
            };
            self.bytecode[*pos] = new_insn;
        }

        let optimized = self.bytecode.clone();
        #[cfg(debug_assertions)]
        if func.name == "main" {
            println!("Bytecode for {}:", func.name);
            for (i, insn) in optimized.iter().enumerate() {
                println!("  {}: {:?} dst={} src1={} src2={} imm={}",
                    i, insn.opcode(), insn.dst(), insn.src1(), insn.src2(), insn.imm());
            }
        }
        (optimized, self.next_reg as usize)
    }

    pub fn lower_program(&mut self, program: &IrProgram) -> (Vec<(Rc<Vec<Instruction>>, usize, usize)>, Vec<String>) {
    let mut functions = Vec::new();
    for (_idx, func) in program.functions.iter().enumerate() {
        let (bytecode, max_reg) = self.lower_function(func);
        let param_count = func.params.len();
        functions.push((Rc::new(bytecode), param_count, max_reg));
    }
    (functions, self.string_pool.clone())
    }
}