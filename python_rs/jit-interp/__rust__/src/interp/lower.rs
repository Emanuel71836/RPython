use crate::ir::*;
use crate::bytecode::{Instruction, OpCode};
use std::collections::HashMap;
use std::rc::Rc;

pub struct LoweringContext {
    reg_map:       Vec<u8>,
    next_reg:      u8,
    bytecode:      Vec<Instruction>,
    block_starts:  HashMap<BasicBlockId, usize>,
    pending_jumps: Vec<(usize, BasicBlockId, bool)>,
    string_pool:   Vec<String>,
    string_map:    HashMap<String, u16>,
    float_pool:    Vec<f64>,          // new
    float_map:     HashMap<u64, u16>, // new: map f64 bits to index
}

impl LoweringContext {
    pub fn new() -> Self {
        LoweringContext {
            reg_map: Vec::new(), next_reg: 0,
            bytecode: Vec::new(),
            block_starts: HashMap::new(), pending_jumps: Vec::new(),
            string_pool: Vec::new(), string_map: HashMap::new(),
            float_pool: Vec::new(), float_map: HashMap::new(),
        }
    }

    fn ensure_reg(&mut self, v: ValueId) -> u8 {
        while self.reg_map.len() <= v as usize {
            self.reg_map.push(self.next_reg);
            self.next_reg += 1;
        }
        self.reg_map[v as usize]
    }

    fn intern(&mut self, s: &str) -> u16 {
        if let Some(&i) = self.string_map.get(s) { return i; }
        let i = self.string_pool.len() as u16;
        self.string_pool.push(s.to_string());
        self.string_map.insert(s.to_string(), i);
        i
    }

    fn intern_float(&mut self, f: f64) -> u16 {
        let bits = f.to_bits();
        if let Some(&i) = self.float_map.get(&bits) { return i; }
        let i = self.float_pool.len() as u16;
        self.float_pool.push(f);
        self.float_map.insert(bits, i);
        i
    }

    fn lower_function(&mut self, func: &Function) -> (Vec<Instruction>, usize) {
        self.reg_map.clear();   self.next_reg = 0;
        self.bytecode.clear();
        self.block_starts.clear(); self.pending_jumps.clear();

        for (idx, _) in func.params.iter().enumerate() { self.ensure_reg(idx as ValueId); }

        for block in &func.blocks {
            self.block_starts.insert(block.id, self.bytecode.len());
            for node in &block.instructions {
                match node {
                    IrNode::Add(d, a, b) => { let (rd, r1, r2) = self.rrr(*d, *a, *b); self.bytecode.push(Instruction::encode_rrr(OpCode::Add, rd, r1, r2)); }
                    IrNode::Sub(d, a, b) => { let (rd, r1, r2) = self.rrr(*d, *a, *b); self.bytecode.push(Instruction::encode_rrr(OpCode::Sub, rd, r1, r2)); }
                    IrNode::Mul(d, a, b) => { let (rd, r1, r2) = self.rrr(*d, *a, *b); self.bytecode.push(Instruction::encode_rrr(OpCode::Mul, rd, r1, r2)); }
                    IrNode::Div(d, a, b) => { let (rd, r1, r2) = self.rrr(*d, *a, *b); self.bytecode.push(Instruction::encode_rrr(OpCode::Div, rd, r1, r2)); }
                    IrNode::Lt (d, a, b) => { let (rd, r1, r2) = self.rrr(*d, *a, *b); self.bytecode.push(Instruction::encode_rrr(OpCode::Lt,  rd, r1, r2)); }
                    IrNode::Le (d, a, b) => { let (rd, r1, r2) = self.rrr(*d, *a, *b); self.bytecode.push(Instruction::encode_rrr(OpCode::Le,  rd, r1, r2)); }

                    IrNode::Const(dst, Constant::Int(i)) => {
                        let rd = self.ensure_reg(*dst);
                        self.bytecode.push(Instruction::encode_imm(OpCode::LoadConst, rd, *i as u16));
                    }
                    IrNode::Const(dst, Constant::Float(f)) => {
                        let rd = self.ensure_reg(*dst);
                        let idx = self.intern_float(*f);
                        self.bytecode.push(Instruction::encode_imm(OpCode::LoadFloat, rd, idx));
                    }
                    IrNode::Const(dst, Constant::Bool(b)) => {
                        let rd = self.ensure_reg(*dst);
                        self.bytecode.push(Instruction::encode_imm(OpCode::LoadBool, rd, if *b { 1 } else { 0 }));
                    }
                    IrNode::Const(dst, Constant::Nil) => {
                        let rd = self.ensure_reg(*dst);
                        self.bytecode.push(Instruction::encode_imm(OpCode::LoadNil, rd, 0));
                    }
                    IrNode::Const(dst, Constant::String(s)) => {
                        let rd  = self.ensure_reg(*dst);
                        let idx = self.intern(s);
                        self.bytecode.push(Instruction::encode_imm(OpCode::LoadString, rd, idx));
                    }

                    IrNode::Param(dst, idx) => {
                        let rd = self.ensure_reg(*dst);
                        self.bytecode.push(Instruction::encode_rr(OpCode::Move, rd, *idx as u8));
                    }
                    IrNode::Move(dst, src) => {
                        let rd = self.ensure_reg(*dst);
                        let rs = self.ensure_reg(*src);
                        self.bytecode.push(Instruction::encode_rr(OpCode::Move, rd, rs));
                    }

                    IrNode::Jump(target) => {
                        let pos = self.bytecode.len();
                        self.bytecode.push(Instruction::encode_imm(OpCode::Jump, 0, 0));
                        self.pending_jumps.push((pos, *target, false));
                    }
                    IrNode::Branch(cond, _t, f) => {
                        let cr  = self.ensure_reg(*cond);
                        let pos = self.bytecode.len();
                        self.bytecode.push(Instruction::encode_imm(OpCode::Branch, cr, 0));
                        self.pending_jumps.push((pos, *f, true));
                    }
                    IrNode::Return(val) => {
                        let rv = self.ensure_reg(*val);
                        self.bytecode.push(Instruction::encode_imm(OpCode::Return, rv, 0));
                    }
                    IrNode::Print(val) => {
                        let rv = self.ensure_reg(*val);
                        self.bytecode.push(Instruction::encode_imm(OpCode::Print, rv, 0));
                    }
                    IrNode::Call(dst, func_idx, args) => {
                        for (i, arg) in args.iter().enumerate() {
                            let src = self.ensure_reg(*arg);
                            self.bytecode.push(Instruction::encode_rr(OpCode::Move, i as u8, src));
                        }
                        self.bytecode.push(Instruction::encode_imm(OpCode::Call, 0, *func_idx as u16));
                        let rd = self.ensure_reg(*dst);
                        if rd != 0 { self.bytecode.push(Instruction::encode_rr(OpCode::Move, rd, 0)); }
                    }

                    IrNode::ImportPython(dst, module) => {
                        let rd  = self.ensure_reg(*dst);
                        let idx = self.intern(module);
                        self.bytecode.push(Instruction::encode_imm(OpCode::ImportPython, rd, idx));
                    }

                    IrNode::GetAttr(dst, obj, attr) => {
                        let rd       = self.ensure_reg(*dst);
                        let ro       = self.ensure_reg(*obj);
                        let attr_idx = self.intern(attr);
                        assert!(attr_idx < 256, "attribute pool overflow");
                        self.bytecode.push(Instruction::encode_rrr(OpCode::GetAttr, rd, ro, attr_idx as u8));
                    }

                    IrNode::PyCall(dst, callable, args) => {
                        let rd = self.ensure_reg(*dst);
                        let rc = self.ensure_reg(*callable);
                        let arg_regs: Vec<u8> = args.iter()
                            .map(|a| self.ensure_reg(*a))
                            .collect();
                        let arg_base = self.next_reg;
                        for (i, &src) in arg_regs.iter().enumerate() {
                            let slot = arg_base + i as u8;
                            self.bytecode.push(Instruction::encode_rr(OpCode::Move, slot, src));
                        }
                        self.bytecode.push(Instruction::encode_rrr(OpCode::PyCall, rd, rc, args.len() as u8));
                        self.bytecode.push(Instruction::encode_imm(OpCode::LoadNil, arg_base, 0));
                    }

                    IrNode::ConvertFromPy(dst, src) => {
                        let rd = self.ensure_reg(*dst);
                        let rs = self.ensure_reg(*src);
                        self.bytecode.push(Instruction::encode_rr(OpCode::ConvertFromPy, rd, rs));
                    }

                    IrNode::CallMethod(dst, obj, method, args) => {
                        let rd       = self.ensure_reg(*dst);
                        let ro       = self.ensure_reg(*obj);
                        let attr_idx = self.intern(method);

                        let arg_regs: Vec<u8> = args.iter()
                            .map(|a| self.ensure_reg(*a))
                            .collect();

                        let arg_base = self.next_reg;
                        self.next_reg = self.next_reg.saturating_add(args.len() as u8);

                        for (i, &src) in arg_regs.iter().enumerate() {
                            let slot = arg_base + i as u8;
                            self.bytecode.push(Instruction::encode_rr(OpCode::Move, slot, src));
                        }

                        self.bytecode.push(Instruction::encode_rrr(
                            OpCode::CallMethod, rd, ro, args.len() as u8,
                        ));
                        self.bytecode.push(Instruction::encode_imm(
                            OpCode::LoadNil, arg_base, attr_idx,
                        ));
                    }
                }
            }
        }

        for (pos, target, is_branch) in &self.pending_jumps {
            let target_pc = *self.block_starts.get(target).unwrap() as u16;
            let insn = self.bytecode[*pos];
            self.bytecode[*pos] = if *is_branch {
                Instruction::encode_imm(insn.opcode(), insn.dst(), target_pc)
            } else {
                Instruction::encode_imm(insn.opcode(), 0, target_pc)
            };
        }

        (self.bytecode.clone(), self.next_reg as usize)
    }

    fn rrr(&mut self, d: ValueId, a: ValueId, b: ValueId) -> (u8, u8, u8) {
        let rd = self.ensure_reg(d);
        let r1 = self.ensure_reg(a);
        let r2 = self.ensure_reg(b);
        (rd, r1, r2)
    }

    pub fn lower_program(&mut self, program: &IrProgram)
        -> (Vec<(Rc<Vec<Instruction>>, usize, usize)>, Vec<String>, Vec<f64>) // also return float_pool
    {
        let mut functions = Vec::new();
        for func in &program.functions {
            let (bytecode, max_reg) = self.lower_function(func);
            let param_count = func.params.len();
            functions.push((Rc::new(bytecode), param_count, max_reg));
        }
        (functions, self.string_pool.clone(), self.float_pool.clone())
    }
}