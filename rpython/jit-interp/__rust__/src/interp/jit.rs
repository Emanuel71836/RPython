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
}

// tier definitions

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Interpreted = 0,
// baseline JIT: fast to compile, no optimisations
    Baseline    = 1,
// optimised JIT: full Cranelift opt passes, best throughput
    Optimized   = 2,
}

pub struct FuncProfile {
    pub call_count: u64,
    pub tier: Tier,
}
impl FuncProfile {
    fn new() -> Self { FuncProfile { call_count: 0, tier: Tier::Interpreted } }
}

// NaN-boxing helpers (free functions, no borrow conflicts)

// value layout (matches value.rs):
// int(i)  = 0x7ffa_0000_0000_0000 | (i & 0x0000_ffff_ffff_ffff)
// bool(b) = 0x7ff9_0000_0000_0000 | (b & 1)
// nil     = 0x0000_0000_0000_0000

const INT_TAG:      i64 = 0x7ffa_0000_0000_0000_u64 as i64;
const BOOL_TAG:     i64 = 0x7ff9_0000_0000_0000_u64 as i64;
const PAYLOAD_MASK: i64 = 0x0000_ffff_ffff_ffff_u64 as i64;

// emit: address of register slot `r` (= regs_ptr + r*8)
#[inline]
fn reg_addr(builder: &mut FunctionBuilder, regs_ptr: Value, r: usize) -> Value {
    let off  = builder.ins().iconst(types::I64, (r as i64) * 8);
    builder.ins().iadd(regs_ptr, off)
}

// emit: load raw Value bits for register `r`
#[inline]
fn load_reg(builder: &mut FunctionBuilder, regs_ptr: Value, r: usize) -> Value {
    let addr = reg_addr(builder, regs_ptr, r);
    builder.ins().load(types::I64, MemFlags::new(), addr, 0)
}

// emit: store `v` into register slot `r`
#[inline]
fn store_reg(builder: &mut FunctionBuilder, regs_ptr: Value, r: usize, v: Value) {
    let addr = reg_addr(builder, regs_ptr, r);
    builder.ins().store(MemFlags::new(), v, addr, 0);
}

// emit: unbox Value(int) → sign-extended i64 payload
#[inline]
fn unbox_int(builder: &mut FunctionBuilder, v: Value) -> Value {
    let masked  = builder.ins().band_imm(v, PAYLOAD_MASK);
    let shifted = builder.ins().ishl_imm(masked, 16);
    builder.ins().sshr_imm(shifted, 16)
}

// emit: box raw i64 → Value(int)
#[inline]
fn box_int(builder: &mut FunctionBuilder, v: Value) -> Value {
    let masked = builder.ins().band_imm(v, PAYLOAD_MASK);
    let tag    = builder.ins().iconst(types::I64, INT_TAG);
    builder.ins().bor(tag, masked)
}

// emit: box 1-bit i64 (0/1) → Value(bool)
#[inline]
fn box_bool(builder: &mut FunctionBuilder, bit: Value) -> Value {
    let tag = builder.ins().iconst(types::I64, BOOL_TAG);
    builder.ins().bor(tag, bit)
}

// JitCompiler

pub struct JitCompiler {
    baseline_module:  JITModule,
    optimized_module: JITModule,
    pub profiles:              Vec<FuncProfile>,
    pub baseline_threshold:    u64,
    pub optimized_threshold:   u64,
}

impl JitCompiler {
    pub fn new(
        num_functions:      usize,
        baseline_threshold: u64,
        optimized_threshold: u64,
    ) -> Option<Self> {
// tier-1: opt_level = none
        let mut t1_flags = settings::builder();
        t1_flags.set("opt_level",        "none").unwrap();
        t1_flags.set("enable_verifier",  "false").unwrap();
        let t1_flags = settings::Flags::new(t1_flags);

// detect the host architecture name at compile time so we can look up
// the right ISA backend.  cranelift_codegen::isa::lookup_by_name accepts
// the same strings as `target_triple` (e.g. "x86_64", "aarch64")
        let host_arch = std::env::consts::ARCH;  // "x86_64", "aarch64", …
        let t1_isa = isa::lookup_by_name(host_arch)
            .map_err(|e| eprintln!("[JIT] ISA lookup failed for {}: {}", host_arch, e)).ok()?
            .finish(t1_flags)
            .map_err(|e| eprintln!("[JIT] ISA finish failed: {}", e)).ok()?;
        let mut t1_jb = JITBuilder::with_isa(t1_isa, cranelift_module::default_libcall_names());
        t1_jb.symbol("ry_jit_call", ry_jit_call as *const u8);
        let baseline_module = JITModule::new(t1_jb);

// tier-2: opt_level = speed_and_size
        let mut t2_flags = settings::builder();
        t2_flags.set("opt_level",             "speed_and_size").unwrap();
        t2_flags.set("enable_verifier",       "false").unwrap();
        t2_flags.set("enable_alias_analysis", "true").unwrap();
        let t2_flags = settings::Flags::new(t2_flags);
        let t2_isa = isa::lookup_by_name(host_arch)
            .map_err(|e| eprintln!("[JIT] ISA lookup failed for {}: {}", host_arch, e)).ok()?
            .finish(t2_flags)
            .map_err(|e| eprintln!("[JIT] ISA finish failed: {}", e)).ok()?;
        let mut t2_jb = JITBuilder::with_isa(t2_isa, cranelift_module::default_libcall_names());
        t2_jb.symbol("ry_jit_call", ry_jit_call as *const u8);
        let optimized_module = JITModule::new(t2_jb);

        let profiles = (0..num_functions).map(|_| FuncProfile::new()).collect();

        if cfg!(debug_assertions) {
            println!(
                "[JIT] Tiered compiler ready  baseline≥{}  optimized≥{}",
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

// called by the VM on every function invocation

    pub fn record_and_maybe_compile(
        &mut self,
        func_idx:   usize,
        bytecode:   &[Instruction],
        num_params: usize,
        num_regs:   usize,
    ) -> Option<(Tier, NativeFunc)> {
        if func_idx >= self.profiles.len() { return None; }

        self.profiles[func_idx].call_count += 1;
        let count        = self.profiles[func_idx].call_count;
        let current_tier = self.profiles[func_idx].tier;

        let target_tier =
            if count >= self.optimized_threshold && current_tier < Tier::Optimized {
                Tier::Optimized
            } else if count >= self.baseline_threshold && current_tier < Tier::Baseline {
                Tier::Baseline
            } else {
                return None;
            };

        if cfg!(debug_assertions) {
            if cfg!(debug_assertions) { println!("[JIT] func_{}: {} calls  {:?} → {:?}", func_idx, count, current_tier, target_tier); }
        }

        if Self::has_unsupported_opcodes(bytecode) || !Self::validate_jump_targets(bytecode) {
            return None;
        }

        let fn_ptr = self.compile_inner(func_idx, bytecode, num_params, num_regs, target_tier)?;
        self.profiles[func_idx].tier = target_tier;
        Some((target_tier, fn_ptr))
    }

// compile a function immediately at Baseline tier, bypassing the call-count
// threshold.  Used for on-stack replacement at function entry so that
// functions called only once (e.g. the outer loop) still get JIT-compiled
    pub fn compile_immediately(
        &mut self,
        func_idx:   usize,
        bytecode:   &[Instruction],
        num_params: usize,
        num_regs:   usize,
    ) -> Option<NativeFunc> {
        if func_idx >= self.profiles.len() { return None; }
        if self.profiles[func_idx].tier >= Tier::Baseline { return None; }  // already compiled

        if Self::has_unsupported_opcodes(bytecode) || !Self::validate_jump_targets(bytecode) {
            return None;
        }

        if cfg!(debug_assertions) {
            if cfg!(debug_assertions) { println!("[JIT] func_{}: compile_immediately → Baseline", func_idx); }
        }

        let fn_ptr = self.compile_inner(func_idx, bytecode, num_params, num_regs, Tier::Baseline)?;
        self.profiles[func_idx].tier = Tier::Baseline;
        Some(fn_ptr)
    }

// pre-flight checks

    fn has_unsupported_opcodes(bytecode: &[Instruction]) -> bool {
        bytecode.iter().any(|i| matches!(i.opcode(),
            OpCode::LoadString
            | OpCode::ImportPython
            | OpCode::GetAttr
            | OpCode::CallPython
            | OpCode::ConvertToPy
            | OpCode::ConvertFromPy
        ))
    }

    fn validate_jump_targets(bytecode: &[Instruction]) -> bool {
        let len = bytecode.len();
        bytecode.iter().all(|i| match i.opcode() {
            OpCode::Jump | OpCode::Branch => (i.imm() as usize) < len,
            _ => true,
        })
    }

// core compiler

// two-pass SSA compiler with block parameters (phi nodes):

// pass 1 – liveness analysis:
// for each block-entry point, compute the set of bytecode registers that
// are live (read before being written in at least one predecessor path)
// these become Cranelift block parameters so values flow through SSA edges
// rather than memory

// pass 2 – code emission:
// maintain a `cache: Vec<Option<CrValue>>` that maps bytecode regs to the
// current SSA value.  At jump/branch edges, pass the cached values as
// block arguments instead of storing them to memory.  Memory is only
// touched for Call (ry_jit_call reads args from regs_ptr) and Return

// this eliminates ALL load/store overhead in hot loops

    fn compile_inner(
        &mut self,
        func_idx:    usize,
        bytecode:    &[Instruction],
        _num_params: usize,
        num_regs:    usize,
        tier:        Tier,
    ) -> Option<NativeFunc> {
        let tier_tag = match tier { Tier::Baseline => "t1", Tier::Optimized => "t2", _ => return None };
        if cfg!(debug_assertions) {
            if cfg!(debug_assertions) { println!("[JIT] func_{} bytecode ({} insns, {} regs):", func_idx, bytecode.len(), num_regs); }
            for (i, insn) in bytecode.iter().enumerate() {
                println!("  {}: {:?} dst={} src1={} src2={} imm={}",
                    i, insn.opcode(), insn.dst(), insn.src1(), insn.src2(), insn.imm());
            }
        }
        let sym = format!("func_{}_{}_{}", func_idx, tier_tag, self.profiles[func_idx].call_count);

        let module: &mut JITModule = match tier {
            Tier::Baseline  => &mut self.baseline_module,
            Tier::Optimized => &mut self.optimized_module,
            _ => return None,
        };

// signature
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));  // *mut VM
        sig.params.push(AbiParam::new(types::I64));  // *mut u64 register file
        sig.params.push(AbiParam::new(types::I64));  // *mut u64 dispatch table
        sig.returns.push(AbiParam::new(types::I64));

        let func_id = module.declare_function(&sym, Linkage::Export, &sig).ok()?;

        let mut helper_sig = module.make_signature();
        for _ in 0..4 { helper_sig.params.push(AbiParam::new(types::I64)); }
        helper_sig.returns.push(AbiParam::new(types::I64));

        let mut ctx = codegen::Context::new();
        ctx.func.signature = sig;

        let helper_id  = module.declare_function("ry_jit_call", Linkage::Import, &helper_sig).unwrap();
        let helper_ref = module.declare_func_in_func(helper_id, &mut ctx.func);

// iR emission
        {
            let mut bctx    = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut bctx);

// pass 1: identify block boundaries and live-in sets
            let nr = num_regs.max(
                bytecode.iter().map(|insn| {
                    [insn.dst(), insn.src1(), insn.src2()]
                        .iter().map(|&x| x as usize + 1).max().unwrap_or(0)
                }).max().unwrap_or(0)
            );

// is_target[i] = true means instruction i starts a new block
            let mut is_target = vec![false; bytecode.len()];
            is_target[0] = true;
            for (idx, insn) in bytecode.iter().enumerate() {
                match insn.opcode() {
                    OpCode::Jump => {
                        let t = insn.imm() as usize;
                        if t < bytecode.len() { is_target[t] = true; }
                    }
                    OpCode::Branch => {
                        let t = insn.imm() as usize;
                        if t < bytecode.len() { is_target[t] = true; }
                        if idx + 1 < bytecode.len() { is_target[idx + 1] = true; }
                    }
                    _ => {}
                }
            }

// proper backward dataflow liveness analysis
// live_in[bi]  = registers live at entry of block bi
// live_out[bi] = registers live at exit of block bi

// equations (standard backward dataflow):
// use[bi]     = regs read before written in bi
// def[bi]     = regs written in bi
// live_in[bi] = use[bi] ∪ (live_out[bi] − def[bi])
// live_out[bi]= ∪ live_in[succ] for each successor succ of bi

// iterate to fixpoint (converges in ≤ #blocks passes for acyclic;
// one extra pass for back-edges)

            let block_starts: Vec<usize> = (0..bytecode.len()).filter(|&k| is_target[k]).collect();
            let nb = block_starts.len();
            let bs_index: HashMap<usize, usize> = block_starts.iter().enumerate()
                .map(|(bi, &pc)| (pc, bi)).collect();

// compute use[], def[], and successor list for each block
            let mut block_use:  Vec<Vec<bool>> = vec![vec![false; nr]; nb];
            let mut block_def:  Vec<Vec<bool>> = vec![vec![false; nr]; nb];
            let mut block_succ: Vec<Vec<usize>> = vec![vec![]; nb];

            for (bi, &start) in block_starts.iter().enumerate() {
                let end = block_starts.get(bi + 1).copied().unwrap_or(bytecode.len());
                let mut written = vec![false; nr];

                for (ii, insn) in bytecode[start..end].iter().enumerate() {
                    let op  = insn.opcode();
                    let dst = insn.dst() as usize;
                    let s1  = insn.src1() as usize;
                    let s2  = insn.src2() as usize;
                    let tgt = insn.imm() as usize;

// record reads (use)
                    let reads: &[usize] = match op {
                        OpCode::Add | OpCode::Sub | OpCode::Mul | OpCode::Div
                        | OpCode::Lt | OpCode::Le => &[s1, s2],
                        OpCode::Branch | OpCode::Return => &[dst],
                        OpCode::Move | OpCode::Inc     => &[s1],
                        _ => &[],
                    };
                    for &r in reads {
                        if r < nr && !written[r] { block_use[bi][r] = true; }
                    }

// record writes (def)
                    match op {
                        OpCode::Add | OpCode::Sub | OpCode::Mul | OpCode::Div
                        | OpCode::Lt | OpCode::Le | OpCode::LoadConst | OpCode::LoadBool
                        | OpCode::LoadNil | OpCode::Move | OpCode::Inc | OpCode::Call => {
                            if dst < nr { written[dst] = true; block_def[bi][dst] = true; }
                        }
                        _ => {}
                    }

// record successors (from the last instruction of this block)
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
                            OpCode::Return => {}  // no successors
                            _ => {
// fall-through to next block
                                if bi + 1 < nb {
                                    block_succ[bi].push(bi + 1);
                                }
                            }
                        }
                    }
                }
                block_def[bi] = written;  // def = everything written
            }

// iterative backward pass until fixpoint
            let mut live_in_bits:  Vec<Vec<bool>> = vec![vec![false; nr]; nb];
            let mut live_out_bits: Vec<Vec<bool>> = vec![vec![false; nr]; nb];
            loop {
                let mut changed = false;
// process blocks in reverse order (helps convergence)
                for bi in (0..nb).rev() {
// live_out[bi] = union of live_in[succ]
                    let mut new_out = vec![false; nr];
                    for &s in &block_succ[bi] {
                        for r in 0..nr {
                            if live_in_bits[s][r] { new_out[r] = true; }
                        }
                    }
// live_in[bi] = use[bi] ∪ (live_out[bi] − def[bi])
                    let mut new_in = block_use[bi].clone();
                    for r in 0..nr {
                        if new_out[r] && !block_def[bi][r] { new_in[r] = true; }
                    }
                    if new_in != live_in_bits[bi] || new_out != live_out_bits[bi] {
                        live_in_bits[bi]  = new_in;
                        live_out_bits[bi] = new_out;
                        changed = true;
                    }
                }
                if !changed { break; }
            }

// convert bit-vectors to sorted register lists
            let live_in: Vec<Vec<usize>> = live_in_bits.iter()
                .map(|bits| (0..nr).filter(|&r| bits[r]).collect())
                .collect();

// create one Cranelift block per target, with block params for live-in regs
            let blocks: Vec<Block> = block_starts.iter().map(|_| builder.create_block()).collect();

// entry block: function params (vm, regs, dispatch)
            builder.append_block_params_for_function_params(blocks[0]);
// non-entry blocks: one I64 param per live-in register
            for bi in 1..blocks.len() {
                for _ in &live_in[bi] {
                    builder.append_block_param(blocks[bi], types::I64);
                }
            }

            builder.switch_to_block(blocks[0]);
            let entry_params = builder.block_params(blocks[0]).to_vec();
            if entry_params.len() < 3 { return None; }
            let vm_ptr   = entry_params[0];
            let regs_ptr = entry_params[1];

// sSA value cache
// cache[r]   = current SSA Value for bytecode register r
// is_raw[r]  = true  → cache[r] is a raw i64 (unboxed integer)
// false → cache[r] is a boxed Value (NaN-tagged)
// dirty[r]   = needs to be written (boxed) to regs_ptr before Call/jump

// storing raw i64 eliminates the box→unbox round-trip that would
// otherwise occur on every arithmetic op in a loop
            let mut cache:  Vec<Option<Value>> = vec![None; nr];
            let mut is_raw: Vec<bool>          = vec![false; nr];
            let mut dirty:  Vec<bool>          = vec![false; nr];

// get the cached value as a boxed Value (for stores, returns, jumps)
            macro_rules! get_boxed {
                ($r:expr) => {{
                    let r: usize = $r;
                    match cache[r] {
                        Some(v) if is_raw[r] => box_int(&mut builder, v),
                        Some(v)              => v,
                        None                 => load_reg(&mut builder, regs_ptr, r),
                    }
                }}
            }

// get the cached value as a raw i64 (for arithmetic)
            macro_rules! get_raw {
                ($r:expr) => {{
                    let r: usize = $r;
                    match cache[r] {
                        Some(v) if is_raw[r] => v,
                        Some(v)              => unbox_int(&mut builder, v),
                        None => {
                            let boxed = load_reg(&mut builder, regs_ptr, r);
                            unbox_int(&mut builder, boxed)
                        }
                    }
                }}
            }

// macros
            macro_rules! flush_for_call {
                () => {
                    for r in 0..nr {
                        if dirty[r] {
                            if let Some(v) = cache[r] {
                                let boxed = if is_raw[r] { box_int(&mut builder, v) } else { v };
                                store_reg(&mut builder, regs_ptr, r, boxed);
                            }
                            dirty[r] = false;
                        }
                    }
                }
            }

            macro_rules! read_reg {
                ($r:expr) => {{ get_boxed!($r) }}
            }

            macro_rules! write_reg {
                ($r:expr, $v:expr) => {{
                    let r: usize = $r;
                    cache[r]  = Some($v);
                    is_raw[r] = false;
                    dirty[r]  = true;
                }}
            }

// write a raw i64 (unboxed) to the cache — avoids box/unbox round-trip
            macro_rules! write_raw {
                ($r:expr, $v:expr) => {{
                    let r: usize = $r;
                    cache[r]  = Some($v);
                    is_raw[r] = true;
                    dirty[r]  = true;
                }}
            }

// emit a jump to target block, passing current cache values as block args
// raw-value jump: pass values as raw i64 block params
// no memory flush needed — values flow via SSA edges
// box only at Call/Return boundaries
            macro_rules! emit_jump_to {
                ($tgt_pc:expr) => {{
                    let tgt_bi = bs_index[&$tgt_pc];
                    let args: Vec<Value> = live_in[tgt_bi].iter().map(|&r| get_raw!(r)).collect();
                    builder.ins().jump(blocks[tgt_bi], &args);
                }}
            }

            macro_rules! emit_brif_to {
                ($flag:expr, $tgt_pc:expr, $fall_pc:expr) => {{
                    let tgt_bi  = bs_index[&$tgt_pc];
                    let fall_bi = bs_index[&$fall_pc];
                    let tgt_args: Vec<Value>  = live_in[tgt_bi].iter().map(|&r| get_raw!(r)).collect();
                    let fall_args: Vec<Value> = live_in[fall_bi].iter().map(|&r| get_raw!(r)).collect();
                    builder.ins().brif($flag, blocks[tgt_bi], &tgt_args,
                                              blocks[fall_bi], &fall_args);
                }}
            }

// pass 2: emit code
            let mut cur_bi = 0usize;  // current block index into `blocks`
            let mut block_terminated = false;

            let mut i = 0usize;
            while i < bytecode.len() {
                let insn = bytecode[i];
                let op   = insn.opcode();

// block boundary: switch to the new block, populate cache from params
                if is_target[i] && i > 0 {
                    if !block_terminated {
                        emit_jump_to!(i);
                    }
                    cur_bi = bs_index[&i];
                    builder.switch_to_block(blocks[cur_bi]);
                    block_terminated = false;

// populate cache from block params (raw i64 SSA values)
// mark dirty so flush_for_call boxes and stores them to
// regs_ptr before any Call that reads args from memory
                    cache.iter_mut().for_each(|c| *c = None);
                    is_raw.iter_mut().for_each(|r| *r = false);
                    dirty.iter_mut().for_each(|d| *d = false);
                    let bp = builder.block_params(blocks[cur_bi]).to_vec();
                    for (param_idx, &r) in live_in[cur_bi].iter().enumerate() {
                        cache[r]  = Some(bp[param_idx]);
                        is_raw[r] = true;  // block params are raw i64
                        dirty[r]  = true;  // must box+flush before Call
                    }
                }

                if block_terminated { i += 1; continue; }

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
                        write_raw!(d, raw);  // store as raw i64 — no boxing until needed
                    }
                    OpCode::Lt | OpCode::Le => {
                        let (d, s1, s2) = (insn.dst() as usize, insn.src1() as usize, insn.src2() as usize);
                        let i1  = get_raw!(s1);
                        let i2  = get_raw!(s2);
                        let cc  = if op == OpCode::Lt { IntCC::SignedLessThan } else { IntCC::SignedLessThanOrEqual };
                        let cmp = builder.ins().icmp(cc, i1, i2);
                        let bit = builder.ins().uextend(types::I64, cmp);
                        let bv  = box_bool(&mut builder, bit);
                        write_reg!(d, bv);  // bool stays boxed (Branch reads bit 0)
                    }
                    OpCode::LoadConst => {
                        let raw = builder.ins().iconst(types::I64, insn.imm() as i64);
                        write_raw!(insn.dst() as usize, raw);  // raw i64 constant
                    }
                    OpCode::LoadBool => {
                        let bit = builder.ins().iconst(types::I64, if insn.imm() != 0 { 1 } else { 0 });
                        write_reg!(insn.dst() as usize, box_bool(&mut builder, bit));
                    }
                    OpCode::LoadNil => {
                        write_reg!(insn.dst() as usize, builder.ins().iconst(types::I64, 0i64));
                    }
                    OpCode::LoadString => unreachable!("filtered by pre-flight check"),
                    OpCode::Jump => {
                        let tgt = insn.imm() as usize;
                        emit_jump_to!(tgt);
                        block_terminated = true;
                    }
                    OpCode::Branch => {
                        // branch jumps to tgt when cond is FALSE (else block)
                        // falls through to fall (then block) when TRUE
                        let cond = read_reg!(insn.dst() as usize);
                        let bit  = builder.ins().band_imm(cond, 1i64);
                        let zero = builder.ins().iconst(types::I64, 0i64);
                        let flag = builder.ins().icmp(IntCC::Equal, bit, zero);  // TRUE when cond=false
                        let tgt  = insn.imm() as usize;
                        let fall = i + 1;
                        emit_brif_to!(flag, tgt, fall);
                        block_terminated = true;
                    }
                    OpCode::Return => {
// caller expects a boxed Value
                        let v = get_boxed!(insn.dst() as usize);
                        builder.ins().return_(&[v]);
                        block_terminated = true;
                    }
                    OpCode::Print => {}
                    OpCode::Move => {
                        let s = insn.src1() as usize;
                        let d = insn.dst() as usize;
// propagate raw flag so we don't box/unbox unnecessarily
                        if let Some(v) = cache[s] {
                            cache[d]  = Some(v);
                            is_raw[d] = is_raw[s];
                            dirty[d]  = true;
                        } else {
// not in cache: load boxed from memory
                            let v = load_reg(&mut builder, regs_ptr, s);
                            write_reg!(d, v);
                        }
                    }
                    OpCode::Call => {
// flush live values so ry_jit_call can read args from regs_ptr
                        flush_for_call!();
// also write any values that ry_jit_call's callee needs to see
// (args are placed at regs[0..num_params] by preceding Move insns)
// after the call, invalidate cache (callee may have side-effects
// on regs_ptr... it doesn't, but we reload the result anyway)
                        cache.iter_mut().for_each(|c| *c = None);
                        dirty.iter_mut().for_each(|d| *d = false);
                        let callee_imm = builder.ins().iconst(types::I64, insn.imm() as i64);
                        let nargs      = builder.ins().iconst(types::I64, 0i64);
                        let call       = builder.ins().call(helper_ref, &[vm_ptr, regs_ptr, callee_imm, nargs]);
                        let res        = builder.inst_results(call)[0];
                        write_reg!(insn.dst() as usize, res);
                    }
                    OpCode::Inc => {
                        let d   = insn.dst() as usize;
                        let raw = get_raw!(d);
                        let one = builder.ins().iconst(types::I64, 1i64);
                        let inc = builder.ins().iadd(raw, one);
                        write_raw!(d, inc);
                    }
                    // Python interop opcodes are excluded by has_unsupported_opcodes()
                    // so these arms should never be reached in practice.
                    OpCode::ImportPython
                    | OpCode::GetAttr
                    | OpCode::CallPython
                    | OpCode::ConvertToPy
                    | OpCode::ConvertFromPy => {
                        unreachable!("Python interop opcode {:?} should have been caught by pre-flight check", op);
                    }
                }

// implicit return at end of bytecode
                if !block_terminated && i + 1 >= bytecode.len() {
                    let z = builder.ins().iconst(types::I64, 0i64);
                    builder.ins().return_(&[z]);
                    block_terminated = true;
                }

                i += 1;
            }

            builder.seal_all_blocks();
            builder.finalize();
        }

// define + finalise
        let module: &mut JITModule = match tier {
            Tier::Baseline  => &mut self.baseline_module,
            Tier::Optimized => &mut self.optimized_module,
            _ => return None,
        };

        if let Err(e) = module.define_function(func_id, &mut ctx) {
            if cfg!(debug_assertions) {
                eprintln!("[JIT] define_function FAILED ({:?}) func_{}: {:?}", tier, func_idx, e);
            }
            return None;
        }

        module.finalize_definitions().ok()?;
        let code_ptr = module.get_finalized_function(func_id);

        if cfg!(debug_assertions) {
            if cfg!(debug_assertions) { println!("[JIT] func_{} compiled {:?} @ {:p}", func_idx, tier, code_ptr); }
        }

        Some(unsafe { mem::transmute(code_ptr) })
    }
}