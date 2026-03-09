use crate::value::Value;
use crate::bytecode::{Instruction, OpCode};
use crate::arena::Arena;
use crate::jit::JitCompiler;
use std::rc::Rc;
use std::collections::HashMap;

struct Function {
    bytecode: Rc<Vec<Instruction>>,
    num_params: usize,
    num_regs: usize,
}

struct Frame {
    reg_base: usize,
    func_idx: usize,
    pc: usize,
    return_reg: usize,
}

pub struct Profiler {
    call_counts: Vec<usize>,
    hot_threshold: usize,
}

impl Profiler {
    pub fn new(num_functions: usize, hot_threshold: usize) -> Self {
        Profiler {
            call_counts: vec![0; num_functions],
            hot_threshold,
        }
    }

    #[inline(always)]
    pub fn record_call(&mut self, func_idx: usize) {
        if func_idx < self.call_counts.len() {
            self.call_counts[func_idx] += 1;
        }
    }
}

pub struct VM {
    register_pool: Vec<Value>,
    frames: Vec<Frame>,
    pub functions: Vec<Function>,
    pub string_pool: Vec<String>,
    arena: Arena,
    pub profiler: Profiler,
    max_call_depth: usize,
    pub jit_compiler: Option<JitCompiler>,
    pub compiled_functions: HashMap<usize, unsafe extern "C" fn(*mut VM, *mut u64, *mut u64) -> u64>,
    pub dispatch_table: Vec<u64>,
}

impl VM {
    pub fn new(
        functions: Vec<(Rc<Vec<Instruction>>, usize, usize)>,
        string_pool: Vec<String>,
        arena_capacity: usize,
        hot_threshold: usize,
        max_call_depth: usize,
    ) -> Self {
        let num_functions = functions.len();
        let funcs: Vec<Function> = functions
            .into_iter()
            .map(|(bytecode, num_params, num_regs)| Function {
                bytecode,
                num_params,
                num_regs,
            })
            .collect();

        let max_regs = funcs.iter().map(|f| f.num_regs).max().unwrap_or(0);
        let pool_size = max_regs * max_call_depth;
        let register_pool = vec![Value::nil(); pool_size];
        let frames = Vec::with_capacity(max_call_depth);

        let baseline_t  = hot_threshold.max(1) as u64;
        let optimized_t = (baseline_t * 3).max(baseline_t + 1);
        let jit_compiler = JitCompiler::new(num_functions, baseline_t, optimized_t);
        let compiled_functions = HashMap::new();
        let dispatch_table = vec![0u64; num_functions];

        let mut vm = VM {
            register_pool,
            frames,
            functions: funcs,
            string_pool,
            arena: Arena::new(arena_capacity, std::ptr::null_mut()),
            profiler: Profiler::new(num_functions, hot_threshold),
            max_call_depth,
            jit_compiler,
            compiled_functions,
            dispatch_table,
        };
        vm.arena.vm_ptr = &mut vm as *mut VM;

        let main_frame = Frame {
            reg_base: 0,
            func_idx: 0,
            pc: 0,
            return_reg: 0,
        };
        vm.frames.push(main_frame);
        vm
    }

    pub fn run(&mut self) -> Result<Value, String> {
        let mut instruction_count = 0;
        const MAX_INSTRUCTIONS: usize = 10_000_000_000;

        while let Some(frame) = self.frames.last_mut() {
            instruction_count += 1;
            if instruction_count > MAX_INSTRUCTIONS {
                return Err(format!("Infinite loop detected after {} instructions", instruction_count));
            }

            if frame.pc == 0 {
                let func_idx  = frame.func_idx;
                let reg_base  = frame.reg_base;
                // skip if already compiled or not compilable
                if self.dispatch_table.get(func_idx).copied().unwrap_or(0) == 0 {
                    let promoted = if let Some(jit) = &mut self.jit_compiler {
                        let (bc, np, nr) = {
                            let f = &self.functions[func_idx];
                            (f.bytecode.clone(), f.num_params, f.num_regs)
                        };
                        jit.compile_immediately(func_idx, &bc, np, nr)
                    } else { None };

                    if let Some(native_fn) = promoted {
                        self.compiled_functions.insert(func_idx, native_fn);
                        if func_idx < self.dispatch_table.len() {
                            self.dispatch_table[func_idx] = native_fn as u64;
                        }
                        // pop current frame and run native code
                        self.frames.pop();
                        let regs_ptr     = self.register_pool.as_mut_ptr().wrapping_add(reg_base) as *mut u64;
                        let dispatch_ptr = self.dispatch_table.as_mut_ptr();
                        let ret_val = unsafe { native_fn(self as *mut VM, regs_ptr, dispatch_ptr) };
                        // store result in caller's return slot (or return from run())
                        if let Some(caller) = self.frames.last_mut() {
                            self.register_pool[caller.return_reg] = Value(ret_val);
                        } else {
                            return Ok(Value(ret_val));
                        }
                        continue;
                    }
                }
            }

            let func = &self.functions[frame.func_idx];
            if frame.pc >= func.bytecode.len() {
                self.frames.pop();
                continue;
            }

            let insn = func.bytecode[frame.pc];
            frame.pc += 1;

            match insn.opcode() {
                OpCode::Add => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    let src1 = frame.reg_base + insn.src1() as usize;
                    let src2 = frame.reg_base + insn.src2() as usize;
                    self.register_pool[dst] = self.register_pool[src1] + self.register_pool[src2];
                }
                OpCode::Sub => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    let src1 = frame.reg_base + insn.src1() as usize;
                    let src2 = frame.reg_base + insn.src2() as usize;
                    self.register_pool[dst] = self.register_pool[src1] - self.register_pool[src2];
                }
                OpCode::Mul => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    let src1 = frame.reg_base + insn.src1() as usize;
                    let src2 = frame.reg_base + insn.src2() as usize;
                    self.register_pool[dst] = self.register_pool[src1] * self.register_pool[src2];
                }
                OpCode::Div => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    let src1 = frame.reg_base + insn.src1() as usize;
                    let src2 = frame.reg_base + insn.src2() as usize;
                    self.register_pool[dst] = self.register_pool[src1] / self.register_pool[src2];
                }
                OpCode::Lt => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    let src1 = frame.reg_base + insn.src1() as usize;
                    let src2 = frame.reg_base + insn.src2() as usize;
                    let a = self.register_pool[src1];
                    let b = self.register_pool[src2];
                    self.register_pool[dst] = if let (Some(ai), Some(bi)) = (a.to_int(), b.to_int()) {
                        Value::from_bool(ai < bi)
                    } else {
                        Value::from_bool(false)
                    };
                }
                OpCode::Le => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    let src1 = frame.reg_base + insn.src1() as usize;
                    let src2 = frame.reg_base + insn.src2() as usize;
                    let a = self.register_pool[src1];
                    let b = self.register_pool[src2];
                    self.register_pool[dst] = if let (Some(ai), Some(bi)) = (a.to_int(), b.to_int()) {
                        Value::from_bool(ai <= bi)
                    } else {
                        Value::from_bool(false)
                    };
                }
                OpCode::LoadConst => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    let imm = insn.imm() as i64;
                    if cfg!(debug_assertions) && frame.func_idx == 0 {
                        println!("main LoadConst: imm={}, dst_reg={}", imm, insn.dst());
                    }
                    self.register_pool[dst] = Value::from_int(imm);
                }
                OpCode::LoadBool => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    self.register_pool[dst] = Value::from_bool(insn.imm() != 0);
                }
                OpCode::LoadNil => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    self.register_pool[dst] = Value::nil();
                }
                OpCode::LoadString => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    let idx = insn.imm() as usize;
                    if idx >= self.string_pool.len() {
                        panic!("LoadString: index {} out of bounds", idx);
                    }
                    let s = &self.string_pool[idx];
                    let val = Value::from_string_in_arena(s, &mut self.arena);
                    self.register_pool[dst] = val;
                }
                OpCode::Jump => {
                    frame.pc = insn.imm() as usize;
                }
                OpCode::Branch => {
                    let cond_reg = frame.reg_base + insn.dst() as usize;
                    let cond = self.register_pool[cond_reg];
                    let take = if let Some(b) = cond.to_bool() {
                        b
                    } else if let Some(i) = cond.to_int() {
                        i != 0
                    } else if let Some(f) = cond.to_f64() {
                        f != 0.0
                    } else {
                        false
                    };
                    if take {
                        frame.pc = insn.imm() as usize;
                    }
                }
                OpCode::Return => {
                    let val_reg = frame.reg_base + insn.dst() as usize;
                    let ret_val = self.register_pool[val_reg];
                    if cfg!(debug_assertions) && frame.func_idx == 1 {
                        println!("heavy returning: {:?}", ret_val);
                    }
                    self.frames.pop(); // remove current frame
                    if let Some(caller) = self.frames.last_mut() {
                        self.register_pool[caller.return_reg] = ret_val;
                    } else {
                        return Ok(ret_val);
                    }
                }
                OpCode::Print => {
                    let reg = frame.reg_base + insn.dst() as usize;
                    println!("{:?}", self.register_pool[reg]);
                }
                OpCode::Move => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    let src = frame.reg_base + insn.src1() as usize;
                    self.register_pool[dst] = self.register_pool[src];
                }
                OpCode::Call => {
                    let func_idx = insn.imm() as usize;
                    if func_idx >= self.functions.len() {
                        panic!("Call to undefined function index {}", func_idx);
                    }

                    // record call for profiling
                    self.profiler.record_call(func_idx);
                    if cfg!(debug_assertions) {
                        println!("[VM] Call to function {} (count now: {:?})", func_idx, self.profiler.call_counts.get(func_idx));
                    }

                    // extract necessary data before potential mutable borrow of self for JIT compilation
                    let current_frame_func_idx = frame.func_idx;
                    let current_frame_num_regs = self.functions[current_frame_func_idx].num_regs;
                    let dst_reg = insn.dst() as usize;
                    let return_reg = frame.reg_base + dst_reg;
                    let num_params = self.functions[func_idx].num_params;
                    let new_base = frame.reg_base + current_frame_num_regs;

                    // check stack overflow
                    if new_base + self.functions[func_idx].num_regs > self.register_pool.len() {
                        panic!("Call stack overflow");
                    }

                    // copy arguments to new register frame
                    for i in 0..num_params {
                        self.register_pool[new_base + i] = self.register_pool[frame.reg_base + i];
                    }

                    // fast path: dispatch table lookup (O(1), no HashMap)
                    if func_idx < self.dispatch_table.len() && self.dispatch_table[func_idx] != 0 {
                        let fn_bits = self.dispatch_table[func_idx];
                        let native_fn: unsafe extern "C" fn(*mut VM, *mut u64, *mut u64) -> u64 =
                            unsafe { std::mem::transmute(fn_bits) };
                        let regs_ptr     = self.register_pool.as_mut_ptr().wrapping_add(new_base) as *mut u64;
                        let dispatch_ptr = self.dispatch_table.as_mut_ptr();
                        let ret_val = unsafe { native_fn(self as *mut VM, regs_ptr, dispatch_ptr) };
                        self.register_pool[return_reg] = Value(ret_val);
                        continue;
                    }

                    // Not compiled yet – maybe compile it if hot
                    if let Some(jit) = &mut self.jit_compiler {
                        self.profiler.record_call(func_idx);
                        let (bytecode, np, nr) = {
                            let f = &self.functions[func_idx];
                            (f.bytecode.clone(), f.num_params, f.num_regs)
                        };
                        if let Some((_tier, native_fn)) = jit.record_and_maybe_compile(func_idx, &bytecode, np, nr) {
                            self.compiled_functions.insert(func_idx, native_fn);
                            if func_idx < self.dispatch_table.len() {
                                self.dispatch_table[func_idx] = native_fn as u64;
                            }
                            let regs_ptr     = self.register_pool.as_mut_ptr().wrapping_add(new_base) as *mut u64;
                            let dispatch_ptr = self.dispatch_table.as_mut_ptr();
                            let ret_val = unsafe { native_fn(self as *mut VM, regs_ptr, dispatch_ptr) };
                            self.register_pool[return_reg] = Value(ret_val);
                            continue;
                        }
                    }

                    // Interpreted call (including the first time after compilation attempt)
                    let new_frame = Frame {
                        reg_base: new_base,
                        func_idx,
                        pc: 0,
                        return_reg,
                    };
                    self.frames.push(new_frame);
                }
                OpCode::Inc => {
                    let reg = frame.reg_base + insn.dst() as usize;
                    self.register_pool[reg] = self.register_pool[reg] + Value::from_int(1);
                }
            }
        }
        Ok(Value::nil())
    }

    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }
}

const MAX_REGS_PER_FRAME: usize = 256;

#[no_mangle]
pub unsafe extern "C" fn ry_jit_call(
    vm: *mut VM,
    regs_ptr: *mut u64,
    func_idx: usize,
    _nargs: usize,
) -> u64 {
    let vm = &mut *vm;
    let num_params = vm.functions[func_idx].num_params;
    let num_regs   = vm.functions[func_idx].num_regs;

    let mut frame_buf = [0u64; MAX_REGS_PER_FRAME];
    let usable = num_regs.min(MAX_REGS_PER_FRAME);
    for i in 0..num_params.min(usable) {
        frame_buf[i] = *regs_ptr.add(i);
    }
    let callee_ptr   = frame_buf.as_mut_ptr();
    let dispatch_ptr = vm.dispatch_table.as_mut_ptr();

    if func_idx < vm.dispatch_table.len() && vm.dispatch_table[func_idx] != 0 {
        let native_fn: unsafe extern "C" fn(*mut VM, *mut u64, *mut u64) -> u64 =
            std::mem::transmute(vm.dispatch_table[func_idx]);
        return native_fn(vm as *mut VM, callee_ptr, dispatch_ptr);
    }

    if let Some(jit) = &mut vm.jit_compiler {
        let (bc, np, nr) = {
            let f = &vm.functions[func_idx];
            (f.bytecode.clone(), f.num_params, f.num_regs)
        };
        let result = jit.compile_immediately(func_idx, &bc, np, nr)
            .map(|f| f)
            .or_else(|| jit.record_and_maybe_compile(func_idx, &bc, np, nr).map(|(_, f)| f));
        if let Some(native_fn) = result {
            vm.compiled_functions.insert(func_idx, native_fn);
            if func_idx < vm.dispatch_table.len() {
                vm.dispatch_table[func_idx] = native_fn as u64;
            }
            return native_fn(vm as *mut VM, callee_ptr, dispatch_ptr);
        }
    }

    let bytecode = vm.functions[func_idx].bytecode.clone();
    let registers = std::slice::from_raw_parts_mut(callee_ptr as *mut Value, usable);

    let mut pc = 0usize;
    loop {
        if pc >= bytecode.len() { break; }
        let insn = bytecode[pc];
        pc += 1;
        match insn.opcode() {
            OpCode::Return => return registers[insn.dst() as usize].0,
            OpCode::Add => {
                let (d,s1,s2) = (insn.dst() as usize, insn.src1() as usize, insn.src2() as usize);
                registers[d] = registers[s1] + registers[s2];
            }
            OpCode::Sub => {
                let (d,s1,s2) = (insn.dst() as usize, insn.src1() as usize, insn.src2() as usize);
                registers[d] = registers[s1] - registers[s2];
            }
            OpCode::Mul => {
                let (d,s1,s2) = (insn.dst() as usize, insn.src1() as usize, insn.src2() as usize);
                registers[d] = registers[s1] * registers[s2];
            }
            OpCode::Div => {
                let (d,s1,s2) = (insn.dst() as usize, insn.src1() as usize, insn.src2() as usize);
                registers[d] = registers[s1] / registers[s2];
            }
            OpCode::Lt => {
                let (d,s1,s2) = (insn.dst() as usize, insn.src1() as usize, insn.src2() as usize);
                registers[d] = if let (Some(a),Some(b)) = (registers[s1].to_int(), registers[s2].to_int()) {
                    Value::from_bool(a < b) } else { Value::from_bool(false) };
            }
            OpCode::Le => {
                let (d,s1,s2) = (insn.dst() as usize, insn.src1() as usize, insn.src2() as usize);
                registers[d] = if let (Some(a),Some(b)) = (registers[s1].to_int(), registers[s2].to_int()) {
                    Value::from_bool(a <= b) } else { Value::from_bool(false) };
            }
            OpCode::LoadConst  => { registers[insn.dst() as usize] = Value::from_int(insn.imm() as i64); }
            OpCode::LoadBool   => { registers[insn.dst() as usize] = Value::from_bool(insn.imm() != 0); }
            OpCode::LoadNil    => { registers[insn.dst() as usize] = Value::nil(); }
            OpCode::LoadString => {
                let s = vm.string_pool[insn.imm() as usize].clone();
                registers[insn.dst() as usize] = Value::from_string_in_arena(&s, &mut vm.arena);
            }
            OpCode::Jump   => { pc = insn.imm() as usize; }
            OpCode::Branch => {
                let cond = registers[insn.dst() as usize];
                let take = cond.to_bool().unwrap_or_else(|| cond.to_int().map(|i| i != 0).unwrap_or(false));
                if take { pc = insn.imm() as usize; }
            }
            OpCode::Print => { println!("{:?}", registers[insn.dst() as usize]); }
            OpCode::Move  => {
                let (d,s) = (insn.dst() as usize, insn.src1() as usize);
                registers[d] = registers[s];
            }
            OpCode::Call  => {
                // nested call: recurse through ry_jit_call with our frame as arg source
                let ret = ry_jit_call(vm as *mut VM, callee_ptr, insn.imm() as usize, 0);
                registers[insn.dst() as usize] = Value(ret);
            }
            OpCode::Inc => {
                let r = insn.dst() as usize;
                registers[r] = registers[r] + Value::from_int(1);
            }
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn ry_load_string(arena_ptr: *mut Arena, idx: usize) -> Value {
    let arena = &mut *arena_ptr;
    let vm = &mut *(arena.vm_ptr as *mut VM);
    if idx >= vm.string_pool.len() {
        panic!("ry_load_string: index {} out of bounds", idx);
    }
    let s = &vm.string_pool[idx];
    Value::from_string_in_arena(s, arena)
}

#[no_mangle]
pub unsafe extern "C" fn ry_print(val: Value) {
    println!("{:?}", val);
}

#[no_mangle]
pub unsafe extern "C" fn ry_call_function(
    arena_ptr: *mut Arena,
    args_ptr: *const Value,
    nargs: usize,
    func_idx: usize,
) -> Value {
    let arena = &mut *arena_ptr;
    let vm = &mut *(arena.vm_ptr as *mut VM);

    let (bytecode, num_params, num_regs) = (
        vm.functions[func_idx].bytecode.clone(),
        vm.functions[func_idx].num_params,
        vm.functions[func_idx].num_regs,
    );
    let mut registers = vec![Value::nil(); num_regs];
    for i in 0..num_params.min(nargs) {
        registers[i] = *args_ptr.add(i);
    }
    let mut pc = 0;
    while pc < bytecode.len() {
        let insn = bytecode[pc];
        pc += 1;
        match insn.opcode() {
            OpCode::Return => {
                let reg = insn.dst() as usize;
                return registers[reg];
            }
            _ => {}
        }
    }
    Value::nil()
}   