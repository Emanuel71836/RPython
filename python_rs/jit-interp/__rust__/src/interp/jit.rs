use crate::bytecode::{Instruction, OpCode};
use crate::vm::VM;
use cranelift::prelude::*;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_module::{Linkage, Module};
use cranelift_jit::{JITBuilder, JITModule};
use std::collections::HashMap;
use std::mem;

type NativeFunc = unsafe extern "C" fn(*mut VM, *mut u64, *mut u64) -> u64;

extern "C" {
    fn ry_jit_call(vm: *mut VM, regs: *mut u64, func_idx: usize, nargs: usize) -> u64;
    fn ry_jit_import(vm: *mut VM, name_idx: u64) -> u64;
    fn ry_jit_getattr(vm: *mut VM, obj: u64, attr_idx: u64) -> u64;
    fn ry_jit_callpython(vm: *mut VM, callable: u64, args_ptr: *mut u64, nargs: u64) -> u64;
    fn ry_jit_convert(vm: *mut VM, src: u64) -> u64;
    fn ry_load_float(vm: *mut VM, idx: u64) -> u64;
    fn ry_load_int(vm: *mut VM, idx: u64) -> u64;   // new
    fn ry_print_value(vm: *mut VM, val: u64) -> u64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Interpreted = 0,
    Baseline = 1,
    Optimized = 2,
}

pub struct FuncProfile {
    pub call_count: u64,
    pub tier: Tier,
}

impl FuncProfile {
    fn new() -> Self {
        FuncProfile {
            call_count: 0,
            tier: Tier::Interpreted,
        }
    }
}

// nan‑boxing constants (must match value.rs)
const INT_TAG: i64 = 0x7ffa_0000_0000_0000_u64 as i64;
const BOOL_TAG: i64 = 0x7ff9_0000_0000_0000_u64 as i64;
const PAYLOAD_MASK: i64 = 0x0000_ffff_ffff_ffff_u64 as i64;

#[inline]
fn reg_addr(builder: &mut FunctionBuilder, regs_ptr: Value, r: usize) -> Value {
    let off = builder.ins().iconst(types::I64, (r as i64) * 8);
    builder.ins().iadd(regs_ptr, off)
}

#[inline]
fn load_reg(builder: &mut FunctionBuilder, regs_ptr: Value, r: usize) -> Value {
    let addr = reg_addr(builder, regs_ptr, r);
    builder.ins().load(types::I64, MemFlags::new(), addr, 0)
}

#[inline]
fn store_reg(builder: &mut FunctionBuilder, regs_ptr: Value, r: usize, v: Value) {
    let addr = reg_addr(builder, regs_ptr, r);
    builder.ins().store(MemFlags::new(), v, addr, 0);
}

#[inline]
fn unbox_int(builder: &mut FunctionBuilder, v: Value) -> Value {
    let masked = builder.ins().band_imm(v, PAYLOAD_MASK);
    let shifted = builder.ins().ishl_imm(masked, 16);
    builder.ins().sshr_imm(shifted, 16)
}

#[inline]
fn box_int(builder: &mut FunctionBuilder, v: Value) -> Value {
    let masked = builder.ins().band_imm(v, PAYLOAD_MASK);
    let tag = builder.ins().iconst(types::I64, INT_TAG);
    builder.ins().bor(tag, masked)
}

#[inline]
fn box_bool(builder: &mut FunctionBuilder, bit: Value) -> Value {
    let tag = builder.ins().iconst(types::I64, BOOL_TAG);
    builder.ins().bor(tag, bit)
}

pub struct JitCompiler {
    baseline_module: JITModule,
    optimized_module: JITModule,
    pub profiles: Vec<FuncProfile>,
    pub baseline_threshold: u64,
    pub optimized_threshold: u64,
}

impl JitCompiler {
    pub fn new(
        num_functions: usize,
        baseline_threshold: u64,
        optimized_threshold: u64,
    ) -> Option<Self> {
        let mut t1_flags = settings::builder();
        t1_flags.set("opt_level", "none").unwrap();
        t1_flags.set("enable_verifier", "false").unwrap();
        let t1_flags = settings::Flags::new(t1_flags);

        let host_arch = std::env::consts::ARCH;
        let t1_isa = isa::lookup_by_name(host_arch)
            .map_err(|e| eprintln!("[jit] isa lookup failed for {}: {}", host_arch, e))
            .ok()?
            .finish(t1_flags)
            .map_err(|e| eprintln!("[jit] isa finish failed: {}", e))
            .ok()?;

        let mut t1_jb = JITBuilder::with_isa(t1_isa, cranelift_module::default_libcall_names());
        t1_jb.symbol("ry_jit_call", ry_jit_call as *const u8);
        t1_jb.symbol("ry_jit_import", ry_jit_import as *const u8);
        t1_jb.symbol("ry_jit_getattr", ry_jit_getattr as *const u8);
        t1_jb.symbol("ry_jit_callpython", ry_jit_callpython as *const u8);
        t1_jb.symbol("ry_jit_convert", ry_jit_convert as *const u8);
        t1_jb.symbol("ry_load_float", ry_load_float as *const u8);
        t1_jb.symbol("ry_load_int", ry_load_int as *const u8);
        t1_jb.symbol("ry_print_value", ry_print_value as *const u8);
        let baseline_module = JITModule::new(t1_jb);

        let mut t2_flags = settings::builder();
        t2_flags.set("opt_level", "speed_and_size").unwrap();
        t2_flags.set("enable_verifier", "false").unwrap();
        t2_flags.set("enable_alias_analysis", "true").unwrap();
        let t2_flags = settings::Flags::new(t2_flags);
        let t2_isa = isa::lookup_by_name(host_arch)
            .map_err(|e| eprintln!("[jit] isa lookup failed for {}: {}", host_arch, e))
            .ok()?
            .finish(t2_flags)
            .map_err(|e| eprintln!("[jit] isa finish failed: {}", e))
            .ok()?;

        let mut t2_jb = JITBuilder::with_isa(t2_isa, cranelift_module::default_libcall_names());
        t2_jb.symbol("ry_jit_call", ry_jit_call as *const u8);
        t2_jb.symbol("ry_jit_import", ry_jit_import as *const u8);
        t2_jb.symbol("ry_jit_getattr", ry_jit_getattr as *const u8);
        t2_jb.symbol("ry_jit_callpython", ry_jit_callpython as *const u8);
        t2_jb.symbol("ry_jit_convert", ry_jit_convert as *const u8);
        t2_jb.symbol("ry_load_float", ry_load_float as *const u8);
        t2_jb.symbol("ry_load_int", ry_load_int as *const u8);
        t2_jb.symbol("ry_print_value", ry_print_value as *const u8);
        let optimized_module = JITModule::new(t2_jb);

        let profiles = (0..num_functions).map(|_| FuncProfile::new()).collect();

        if cfg!(debug_assertions) {
            println!(
                "[jit] tiered compiler ready  baseline≥{}  optimized≥{}  (python interop on)",
                baseline_threshold, optimized_threshold
            );
        }

        Some(JitCompiler {
            baseline_module,
            optimized_module,
            profiles,
            baseline_threshold,
            optimized_threshold,
        })
    }

    pub fn record_and_maybe_compile(
        &mut self,
        func_idx: usize,
        bytecode: &[Instruction],
        num_params: usize,
        num_regs: usize,
    ) -> Option<(Tier, NativeFunc)> {
        if func_idx >= self.profiles.len() {
            return None;
        }

        self.profiles[func_idx].call_count += 1;
        let count = self.profiles[func_idx].call_count;
        let current_tier = self.profiles[func_idx].tier;

        let target_tier = if count >= self.optimized_threshold && current_tier < Tier::Optimized {
            Tier::Optimized
        } else if count >= self.baseline_threshold && current_tier < Tier::Baseline {
            Tier::Baseline
        } else {
            return None;
        };

        if cfg!(debug_assertions) {
            println!(
                "[jit] func_{}: {} calls  {:?} → {:?}",
                func_idx, count, current_tier, target_tier
            );
        }

        if Self::has_unsupported_opcodes(bytecode) || !Self::validate_jump_targets(bytecode) {
            return None;
        }

        let fn_ptr = self.compile_inner(func_idx, bytecode, num_params, num_regs, target_tier)?;
        self.profiles[func_idx].tier = target_tier;
        Some((target_tier, fn_ptr))
    }

    pub fn compile_immediately(
        &mut self,
        func_idx: usize,
        bytecode: &[Instruction],
        num_params: usize,
        num_regs: usize,
    ) -> Option<NativeFunc> {
        if func_idx >= self.profiles.len() {
            return None;
        }
        if self.profiles[func_idx].tier >= Tier::Baseline {
            return None;
        }

        if Self::has_unsupported_opcodes(bytecode) || !Self::validate_jump_targets(bytecode) {
            return None;
        }

        if cfg!(debug_assertions) {
            println!("[jit] func_{}: compile_immediately → baseline", func_idx);
        }

        let fn_ptr = self.compile_inner(func_idx, bytecode, num_params, num_regs, Tier::Baseline)?;
        self.profiles[func_idx].tier = Tier::Baseline;
        Some(fn_ptr)
    }

    // pre‑flight checks: print is now supported (via helper)
    fn has_unsupported_opcodes(bytecode: &[Instruction]) -> bool {
        bytecode.iter().any(|i| {
            matches!(
                i.opcode(),
                OpCode::LoadString
                    | OpCode::LoadInt          // new
                    | OpCode::ImportPython
                    | OpCode::GetAttr
                    | OpCode::PyCall
                    | OpCode::ConvertFromPy
                    | OpCode::CallMethod
            )
        })
    }

    fn validate_jump_targets(bytecode: &[Instruction]) -> bool {
        let len = bytecode.len();
        bytecode.iter().all(|i| match i.opcode() {
            OpCode::Jump | OpCode::Branch => (i.imm() as usize) < len,
            _ => true,
        })
    }

    fn compile_inner(
        &mut self,
        func_idx: usize,
        bytecode: &[Instruction],
        _num_params: usize,
        num_regs: usize,
        tier: Tier,
    ) -> Option<NativeFunc> {
        let tier_tag = match tier {
            Tier::Baseline => "t1",
            Tier::Optimized => "t2",
            _ => return None,
        };
        if cfg!(debug_assertions) {
            println!(
                "[jit] func_{} compiling ({} insns, {} regs):",
                func_idx,
                bytecode.len(),
                num_regs
            );
            for (i, insn) in bytecode.iter().enumerate() {
                println!(
                    "  {}: {:?} dst={} src1={} src2={} imm={}",
                    i,
                    insn.opcode(),
                    insn.dst(),
                    insn.src1(),
                    insn.src2(),
                    insn.imm()
                );
            }
        }

        let sym = format!("func_{}_{}_{}", func_idx, tier_tag, self.profiles[func_idx].call_count);

        let module: &mut JITModule = match tier {
            Tier::Baseline => &mut self.baseline_module,
            Tier::Optimized => &mut self.optimized_module,
            _ => return None,
        };

        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));

        let func_id = module.declare_function(&sym, Linkage::Export, &sig).ok()?;

        let mut call_sig = module.make_signature();
        for _ in 0..4 {
            call_sig.params.push(AbiParam::new(types::I64));
        }
        call_sig.returns.push(AbiParam::new(types::I64));

        let mut import_sig = module.make_signature();
        import_sig.params.push(AbiParam::new(types::I64));
        import_sig.params.push(AbiParam::new(types::I64));
        import_sig.returns.push(AbiParam::new(types::I64));

        let mut getattr_sig = module.make_signature();
        getattr_sig.params.push(AbiParam::new(types::I64));
        getattr_sig.params.push(AbiParam::new(types::I64));
        getattr_sig.params.push(AbiParam::new(types::I64));
        getattr_sig.returns.push(AbiParam::new(types::I64));

        let mut callpy_sig = module.make_signature();
        callpy_sig.params.push(AbiParam::new(types::I64));
        callpy_sig.params.push(AbiParam::new(types::I64));
        callpy_sig.params.push(AbiParam::new(types::I64));
        callpy_sig.params.push(AbiParam::new(types::I64));
        callpy_sig.returns.push(AbiParam::new(types::I64));

        let mut convert_sig = module.make_signature();
        convert_sig.params.push(AbiParam::new(types::I64));
        convert_sig.params.push(AbiParam::new(types::I64));
        convert_sig.returns.push(AbiParam::new(types::I64));

        let mut load_float_sig = module.make_signature();
        load_float_sig.params.push(AbiParam::new(types::I64));
        load_float_sig.params.push(AbiParam::new(types::I64));
        load_float_sig.returns.push(AbiParam::new(types::I64));

        let mut load_int_sig = module.make_signature();
        load_int_sig.params.push(AbiParam::new(types::I64));
        load_int_sig.params.push(AbiParam::new(types::I64));
        load_int_sig.returns.push(AbiParam::new(types::I64));

        let mut print_sig = module.make_signature();
        print_sig.params.push(AbiParam::new(types::I64));
        print_sig.params.push(AbiParam::new(types::I64));
        print_sig.returns.push(AbiParam::new(types::I64));

        let mut ctx = codegen::Context::new();
        ctx.func.signature = sig;

        let helper_call_id = module.declare_function("ry_jit_call", Linkage::Import, &call_sig).ok()?;
        let helper_import_id = module.declare_function("ry_jit_import", Linkage::Import, &import_sig).ok()?;
        let helper_getattr_id = module.declare_function("ry_jit_getattr", Linkage::Import, &getattr_sig).ok()?;
        let helper_callpy_id = module.declare_function("ry_jit_callpython", Linkage::Import, &callpy_sig).ok()?;
        let helper_convert_id = module.declare_function("ry_jit_convert", Linkage::Import, &convert_sig).ok()?;
        let helper_load_float_id = module.declare_function("ry_load_float", Linkage::Import, &load_float_sig).ok()?;
        let helper_load_int_id = module.declare_function("ry_load_int", Linkage::Import, &load_int_sig).ok()?;
        let helper_print_id = module.declare_function("ry_print_value", Linkage::Import, &print_sig).ok()?;

        let helper_call = module.declare_func_in_func(helper_call_id, &mut ctx.func);
        let helper_import = module.declare_func_in_func(helper_import_id, &mut ctx.func);
        let helper_getattr = module.declare_func_in_func(helper_getattr_id, &mut ctx.func);
        let _helper_callpy = module.declare_func_in_func(helper_callpy_id, &mut ctx.func);
        let helper_convert = module.declare_func_in_func(helper_convert_id, &mut ctx.func);
        let helper_load_float = module.declare_func_in_func(helper_load_float_id, &mut ctx.func);
        let helper_load_int = module.declare_func_in_func(helper_load_int_id, &mut ctx.func);
        let helper_print = module.declare_func_in_func(helper_print_id, &mut ctx.func);

        {
            let mut bctx = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut bctx);

            // liveness analysis
            let nr = num_regs.max(
                bytecode
                    .iter()
                    .map(|insn| {
                        [insn.dst(), insn.src1(), insn.src2()]
                            .iter()
                            .map(|&x| x as usize + 1)
                            .max()
                            .unwrap_or(0)
                    })
                    .max()
                    .unwrap_or(0),
            );

            let mut is_target = vec![false; bytecode.len()];
            is_target[0] = true;
            for (idx, insn) in bytecode.iter().enumerate() {
                match insn.opcode() {
                    OpCode::Jump => {
                        let t = insn.imm() as usize;
                        if t < bytecode.len() {
                            is_target[t] = true;
                        }
                    }
                    OpCode::Branch => {
                        let t = insn.imm() as usize;
                        if t < bytecode.len() {
                            is_target[t] = true;
                        }
                        if idx + 1 < bytecode.len() {
                            is_target[idx + 1] = true;
                        }
                    }
                    _ => {}
                }
            }

            let block_starts: Vec<usize> = (0..bytecode.len()).filter(|&k| is_target[k]).collect();
            let nb = block_starts.len();
            let bs_index: HashMap<usize, usize> = block_starts
                .iter()
                .enumerate()
                .map(|(bi, &pc)| (pc, bi))
                .collect();

            let mut block_use: Vec<Vec<bool>> = vec![vec![false; nr]; nb];
            let mut block_def: Vec<Vec<bool>> = vec![vec![false; nr]; nb];
            let mut block_succ: Vec<Vec<usize>> = vec![vec![]; nb];

            for (bi, &start) in block_starts.iter().enumerate() {
                let end = block_starts.get(bi + 1).copied().unwrap_or(bytecode.len());
                let mut written = vec![false; nr];

                for (ii, insn) in bytecode[start..end].iter().enumerate() {
                    let op = insn.opcode();
                    let dst = insn.dst() as usize;
                    let s1 = insn.src1() as usize;
                    let s2 = insn.src2() as usize;
                    let tgt = insn.imm() as usize;

                    let reads: &[usize] = match op {
                        OpCode::Add | OpCode::Sub | OpCode::Mul | OpCode::Div | OpCode::Lt | OpCode::Le => {
                            &[s1, s2]
                        }
                        OpCode::Branch | OpCode::Return => &[dst],
                        OpCode::Move | OpCode::Inc | OpCode::Print | OpCode::ConvertFromPy => &[s1],
                        OpCode::GetAttr => &[s1],
                        OpCode::PyCall => &[s1],
                        _ => &[],
                    };
                    for &r in reads {
                        if r < nr && !written[r] {
                            block_use[bi][r] = true;
                        }
                    }

                    match op {
                        OpCode::Add
                        | OpCode::Sub
                        | OpCode::Mul
                        | OpCode::Div
                        | OpCode::Lt
                        | OpCode::Le
                        | OpCode::LoadConst
                        | OpCode::LoadInt
                        | OpCode::LoadBool
                        | OpCode::LoadNil
                        | OpCode::Move
                        | OpCode::Inc
                        | OpCode::Call
                        | OpCode::ImportPython
                        | OpCode::GetAttr
                        | OpCode::PyCall
                        | OpCode::CallMethod
                        | OpCode::ConvertFromPy
                        | OpCode::LoadFloat => {
                            if dst < nr {
                                written[dst] = true;
                                block_def[bi][dst] = true;
                            }
                        }
                        _ => {}
                    }

                    let is_last = ii == end - start - 1;
                    if is_last {
                        match op {
                            OpCode::Jump => {
                                if tgt < bytecode.len() {
                                    block_succ[bi].push(bs_index[&tgt]);
                                }
                            }
                            OpCode::Branch => {
                                if tgt < bytecode.len() {
                                    block_succ[bi].push(bs_index[&tgt]);
                                }
                                let fall = block_starts.get(bi + 1).copied().unwrap_or(bytecode.len());
                                if fall < bytecode.len() && bs_index.contains_key(&fall) {
                                    block_succ[bi].push(bs_index[&fall]);
                                }
                            }
                            OpCode::Return => {}
                            _ => {
                                if bi + 1 < nb {
                                    block_succ[bi].push(bi + 1);
                                }
                            }
                        }
                    }
                }
                block_def[bi] = written;
            }

            let mut live_in_bits: Vec<Vec<bool>> = vec![vec![false; nr]; nb];
            let mut live_out_bits: Vec<Vec<bool>> = vec![vec![false; nr]; nb];
            loop {
                let mut changed = false;
                for bi in (0..nb).rev() {
                    let mut new_out = vec![false; nr];
                    for &s in &block_succ[bi] {
                        for r in 0..nr {
                            if live_in_bits[s][r] {
                                new_out[r] = true;
                            }
                        }
                    }
                    let mut new_in = block_use[bi].clone();
                    for r in 0..nr {
                        if new_out[r] && !block_def[bi][r] {
                            new_in[r] = true;
                        }
                    }
                    if new_in != live_in_bits[bi] || new_out != live_out_bits[bi] {
                        live_in_bits[bi] = new_in;
                        live_out_bits[bi] = new_out;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }

            let live_in: Vec<Vec<usize>> = live_in_bits
                .iter()
                .map(|bits| (0..nr).filter(|&r| bits[r]).collect())
                .collect();

            let blocks: Vec<Block> = block_starts.iter().map(|_| builder.create_block()).collect();
            builder.append_block_params_for_function_params(blocks[0]);
            for bi in 1..blocks.len() {
                for _ in &live_in[bi] {
                    builder.append_block_param(blocks[bi], types::I64);
                }
            }

            builder.switch_to_block(blocks[0]);
            let entry_params = builder.block_params(blocks[0]).to_vec();
            if entry_params.len() < 3 {
                return None;
            }
            let vm_ptr = entry_params[0];
            let regs_ptr = entry_params[1];

            // simple cache for non‑python values only
            let mut cache: Vec<Option<Value>> = vec![None; nr];
            let mut is_raw: Vec<bool> = vec![false; nr];
            let mut dirty: Vec<bool> = vec![false; nr];
            let mut is_pyobj: Vec<bool> = vec![false; nr];

            macro_rules! get_boxed {
                ($r:expr) => {{
                    let r: usize = $r;
                    if is_pyobj[r] {
                        load_reg(&mut builder, regs_ptr, r)
                    } else {
                        match cache[r] {
                            Some(v) if is_raw[r] => box_int(&mut builder, v),
                            Some(v) => v,
                            None => load_reg(&mut builder, regs_ptr, r),
                        }
                    }
                }};
            }
            macro_rules! get_raw {
                ($r:expr) => {{
                    let r: usize = $r;
                    if is_pyobj[r] {
                        let boxed = load_reg(&mut builder, regs_ptr, r);
                        unbox_int(&mut builder, boxed)
                    } else {
                        match cache[r] {
                            Some(v) if is_raw[r] => v,
                            Some(v) => unbox_int(&mut builder, v),
                            None => {
                                let boxed = load_reg(&mut builder, regs_ptr, r);
                                unbox_int(&mut builder, boxed)
                            }
                        }
                    }
                }};
            }
            macro_rules! flush_dirty {
                () => {
                    for r in 0..nr {
                        if dirty[r] && !is_pyobj[r] {
                            if let Some(v) = cache[r] {
                                let boxed = if is_raw[r] { box_int(&mut builder, v) } else { v };
                                store_reg(&mut builder, regs_ptr, r, boxed);
                            }
                            dirty[r] = false;
                        }
                    }
                };
            }
            macro_rules! invalidate_cache {
                () => {
                    cache.iter_mut().for_each(|c| *c = None);
                    is_raw.iter_mut().for_each(|r| *r = false);
                    dirty.iter_mut().for_each(|d| *d = false);
                };
            }
            macro_rules! write_reg {
                ($r:expr, $v:expr, $py:expr) => {{
                    let r: usize = $r;
                    if $py {
                        store_reg(&mut builder, regs_ptr, r, $v);
                        is_pyobj[r] = true;
                        cache[r] = None;
                        is_raw[r] = false;
                        dirty[r] = false;
                    } else {
                        cache[r] = Some($v);
                        is_raw[r] = false;
                        dirty[r] = true;
                        is_pyobj[r] = false;
                    }
                }};
            }
            macro_rules! write_raw {
                ($r:expr, $v:expr, $py:expr) => {{
                    let r: usize = $r;
                    if $py {
                        let boxed = box_int(&mut builder, $v);
                        store_reg(&mut builder, regs_ptr, r, boxed);
                        is_pyobj[r] = true;
                        cache[r] = None;
                        is_raw[r] = false;
                        dirty[r] = false;
                    } else {
                        cache[r] = Some($v);
                        is_raw[r] = true;
                        dirty[r] = true;
                        is_pyobj[r] = false;
                    }
                }};
            }
            macro_rules! emit_jump_to {
                ($tgt_pc:expr) => {{
                    flush_dirty!();
                    let tgt_bi = bs_index[&$tgt_pc];
                    let args: Vec<Value> = live_in[tgt_bi].iter().map(|&r| get_raw!(r)).collect();
                    builder.ins().jump(blocks[tgt_bi], &args);
                }};
            }
            macro_rules! emit_brif_to {
                ($flag:expr, $tgt_pc:expr, $fall_pc:expr) => {{
                    flush_dirty!();
                    let tgt_bi = bs_index[&$tgt_pc];
                    let fall_bi = bs_index[&$fall_pc];
                    let tgt_args: Vec<Value> = live_in[tgt_bi].iter().map(|&r| get_raw!(r)).collect();
                    let fall_args: Vec<Value> = live_in[fall_bi].iter().map(|&r| get_raw!(r)).collect();
                    builder.ins().brif($flag, blocks[tgt_bi], &tgt_args, blocks[fall_bi], &fall_args);
                }};
            }

            let mut _cur_bi = 0usize;
            let mut block_terminated = false;

            let mut i = 0usize;
            while i < bytecode.len() {
                let insn = bytecode[i];
                let op = insn.opcode();

                if is_target[i] && i > 0 {
                    if !block_terminated {
                        emit_jump_to!(i);
                    }
                    _cur_bi = bs_index[&i];
                    builder.switch_to_block(blocks[_cur_bi]);
                    block_terminated = false;

                    invalidate_cache!();
                    let bp = builder.block_params(blocks[_cur_bi]).to_vec();
                    for (param_idx, &r) in live_in[_cur_bi].iter().enumerate() {
                        cache[r] = Some(bp[param_idx]);
                        is_raw[r] = true;
                        dirty[r] = true;
                        is_pyobj[r] = false;
                    }
                }

                if block_terminated {
                    i += 1;
                    continue;
                }

                match op {
                    OpCode::Add | OpCode::Sub | OpCode::Mul | OpCode::Div => {
                        let (d, s1, s2) = (insn.dst() as usize, insn.src1() as usize, insn.src2() as usize);
                        let i1 = get_raw!(s1);
                        let i2 = get_raw!(s2);
                        let raw = match op {
                            OpCode::Add => builder.ins().iadd(i1, i2),
                            OpCode::Sub => builder.ins().isub(i1, i2),
                            OpCode::Mul => builder.ins().imul(i1, i2),
                            OpCode::Div => builder.ins().sdiv(i1, i2),
                            _ => unreachable!(),
                        };
                        write_raw!(d, raw, false);
                    }
                    OpCode::Lt | OpCode::Le => {
                        let (d, s1, s2) = (insn.dst() as usize, insn.src1() as usize, insn.src2() as usize);
                        let i1 = get_raw!(s1);
                        let i2 = get_raw!(s2);
                        let cc = if op == OpCode::Lt {
                            IntCC::SignedLessThan
                        } else {
                            IntCC::SignedLessThanOrEqual
                        };
                        let cmp = builder.ins().icmp(cc, i1, i2);
                        let bit = builder.ins().uextend(types::I64, cmp);
                        let boxed = box_bool(&mut builder, bit);
                        write_reg!(d, boxed, false);
                    }
                    OpCode::LoadConst => {
                        let val = builder.ins().iconst(types::I64, insn.imm() as i64);
                        write_raw!(insn.dst() as usize, val, false);
                    }
                    OpCode::LoadBool => {
                        let bit = builder.ins().iconst(types::I64, if insn.imm() != 0 { 1 } else { 0 });
                        let boxed = box_bool(&mut builder, bit);
                        write_reg!(insn.dst() as usize, boxed, false);
                    }
                    OpCode::LoadNil => {
                        let nil = builder.ins().iconst(types::I64, 0i64);
                        write_reg!(insn.dst() as usize, nil, false);
                    }
                    OpCode::LoadString => unreachable!("filtered by pre‑flight check"),
                    OpCode::LoadFloat => {
                        let dst = insn.dst() as usize;
                        let idx = insn.imm() as i64;
                        let idx_val = builder.ins().iconst(types::I64, idx);
                        let call = builder.ins().call(helper_load_float, &[vm_ptr, idx_val]);
                        let res = builder.inst_results(call)[0];
                        write_reg!(dst, res, false);
                    }
                    OpCode::LoadInt => unreachable!("loadint should have been filtered by pre‑flight check"),
                    OpCode::Jump => {
                        let tgt = insn.imm() as usize;
                        emit_jump_to!(tgt);
                        block_terminated = true;
                    }
                    OpCode::Branch => {
                        let cond = get_boxed!(insn.dst() as usize);
                        let bit = builder.ins().band_imm(cond, 1i64);
                        let zero = builder.ins().iconst(types::I64, 0i64);
                        let flag = builder.ins().icmp(IntCC::Equal, bit, zero);
                        let tgt = insn.imm() as usize;
                        let fall = i + 1;
                        emit_brif_to!(flag, tgt, fall);
                        block_terminated = true;
                    }
                    OpCode::Return => {
                        flush_dirty!();
                        let v = get_boxed!(insn.dst() as usize);
                        builder.ins().return_(&[v]);
                        block_terminated = true;
                    }
                    OpCode::Print => {
                        flush_dirty!();
                        let v = get_boxed!(insn.dst() as usize);
                        builder.ins().call(helper_print, &[vm_ptr, v]);
                    }
                    OpCode::Move => {
                        let s = insn.src1() as usize;
                        let d = insn.dst() as usize;
                        if is_pyobj[s] {
                            let v = load_reg(&mut builder, regs_ptr, s);
                            store_reg(&mut builder, regs_ptr, d, v);
                            is_pyobj[d] = true;
                            cache[d] = None;
                            dirty[d] = false;
                        } else {
                            if let Some(v) = cache[s] {
                                cache[d] = Some(v);
                                is_raw[d] = is_raw[s];
                                dirty[d] = true;
                                is_pyobj[d] = false;
                            } else {
                                let v = load_reg(&mut builder, regs_ptr, s);
                                write_reg!(d, v, false);
                            }
                        }
                    }
                    OpCode::Inc => {
                        let d = insn.dst() as usize;
                        let raw = get_raw!(d);
                        let one = builder.ins().iconst(types::I64, 1i64);
                        let sum = builder.ins().iadd(raw, one);
                        write_raw!(d, sum, false);
                    }
                    OpCode::Call => {
                        flush_dirty!();
                        invalidate_cache!();
                        let callee = builder.ins().iconst(types::I64, insn.imm() as i64);
                        let nargs = builder.ins().iconst(types::I64, 0i64);
                        let call = builder.ins().call(helper_call, &[vm_ptr, regs_ptr, callee, nargs]);
                        let res = builder.inst_results(call)[0];
                        write_reg!(insn.dst() as usize, res, false);
                    }

                    // python interop opcodes
                    OpCode::ImportPython => {
                        let dst = insn.dst() as usize;
                        let name_idx = insn.imm() as i64;
                        flush_dirty!();
                        let name_imm = builder.ins().iconst(types::I64, name_idx);
                        let call = builder.ins().call(helper_import, &[vm_ptr, name_imm]);
                        let res = builder.inst_results(call)[0];
                        invalidate_cache!();
                        write_reg!(dst, res, true);
                    }
                    OpCode::GetAttr => {
                        let dst = insn.dst() as usize;
                        let obj_reg = insn.src1() as usize;
                        let attr_idx = insn.src2() as i64;
                        flush_dirty!();
                        let obj_val = get_boxed!(obj_reg);
                        let attr_imm = builder.ins().iconst(types::I64, attr_idx);
                        let call = builder.ins().call(helper_getattr, &[vm_ptr, obj_val, attr_imm]);
                        let res = builder.inst_results(call)[0];
                        write_reg!(dst, res, true);
                    }
                    OpCode::PyCall => {
                        unreachable!("pycall reached jit compiler — should have been filtered");
                    }
                    OpCode::ConvertFromPy => {
                        let dst = insn.dst() as usize;
                        let src = insn.src1() as usize;
                        flush_dirty!();
                        let src_val = get_boxed!(src);
                        let call = builder.ins().call(helper_convert, &[vm_ptr, src_val]);
                        let res = builder.inst_results(call)[0];
                        write_reg!(dst, res, false);
                    }
                    OpCode::CallMethod => {
                        unreachable!("callmethod reached jit compiler — should have been filtered");
                    }
                }

                if !block_terminated && i + 1 >= bytecode.len() {
                    flush_dirty!();
                    let z = builder.ins().iconst(types::I64, 0i64);
                    builder.ins().return_(&[z]);
                    block_terminated = true;
                }

                i += 1;
            }

            builder.seal_all_blocks();
            builder.finalize();
        }

        let module: &mut JITModule = match tier {
            Tier::Baseline => &mut self.baseline_module,
            Tier::Optimized => &mut self.optimized_module,
            _ => return None,
        };

        if let Err(e) = module.define_function(func_id, &mut ctx) {
            if cfg!(debug_assertions) {
                eprintln!(
                    "[jit] define_function FAILED ({:?}) func_{}: {:?}",
                    tier, func_idx, e
                );
            }
            return None;
        }

        module.finalize_definitions().ok()?;
        let code_ptr = module.get_finalized_function(func_id);

        if cfg!(debug_assertions) {
            println!("[jit] func_{} compiled {:?} @ {:p}", func_idx, tier, code_ptr);
        }

        Some(unsafe { mem::transmute(code_ptr) })
    }
}