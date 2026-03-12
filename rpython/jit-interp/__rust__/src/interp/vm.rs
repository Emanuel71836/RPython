use crate::value::Value;
use crate::bytecode::{Instruction, OpCode};
use crate::arena::Arena;
use crate::jit::JitCompiler;
use std::rc::Rc;
use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::BoundObject;
use pyo3::types::{PyFloat, PyInt, PyBool, PyString};

/// Convert a native [`Value`] to an owned Python object.
/// The returned [`PyObject`] has a fresh reference (owned).
fn value_to_pyobject(py: Python<'_>, v: Value) -> PyResult<PyObject> {
    if let Some(b) = v.to_bool() {
        // Check bool before int — bool is a subtype of int in Python.
        // into_pyobject for bool returns Borrowed<PyBool>; call into_bound() to
        // get an owned Bound, then convert to PyObject.
        return Ok(b.into_pyobject(py)?.into_bound().into_any().unbind());
    }
    if let Some(i) = v.to_int() {
        return Ok(i.into_pyobject(py)?.into_any().unbind());
    }
    if let Some(f) = v.to_f64() {
        return Ok(f.into_pyobject(py)?.into_any().unbind());
    }
    if v.is_pyobject() {
        // Already a Python object — return a new borrowed (bumped) reference.
        let raw = v.to_pyobject().unwrap();
        return Ok(unsafe { PyObject::from_borrowed_ptr(py, raw) });
    }
    if let Some(s) = v.to_string_from_arena() {
        return Ok(s.into_pyobject(py)?.into_any().unbind());
    }
    // Nil → Python None
    Ok(py.None())
}

/// Convert a Python object to the closest native [`Value`].
fn pyobject_to_value(py: Python<'_>, obj: PyObject) -> PyResult<Value> {
    let bound = obj.bind(py);
    // Try bool first (bool is a subtype of int in Python, must check first)
    if bound.is_instance_of::<PyBool>() {
        let b: bool = bound.extract()?;
        return Ok(Value::from_bool(b));
    }
    if bound.is_instance_of::<PyInt>() {
        let i: i64 = bound.extract()?;
        return Ok(Value::from_int(i));
    }
    if bound.is_instance_of::<PyFloat>() {
        let f: f64 = bound.extract()?;
        return Ok(Value::from_f64(f));
    }
    if bound.is_instance_of::<PyString>() {
        // Keep strings as opaque PyObject values; caller can use ConvertFromPy
        let raw = obj.into_ptr();
        return Ok(Value::from_pyobject(raw));
    }
    // Everything else: wrap as opaque PyObject
    Ok(Value::from_pyobject(obj.into_ptr()))
}


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
// dispatch_table[func_idx] = fn_ptr as u64, or 0 = not compiled
// flat array: one load + branch = hot-path call dispatch
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
// promote to Tier-2 after 3× baseline calls (not 10×) — hot loops hit
// this quickly and benefit the most from speed_and_size optimisations
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

// on-stack replacement (OSR) at function entry
// when the interpreter is about to execute the first instruction of
// any function, try to JIT-compile it immediately.  This ensures
// functions that are only called once (e.g. the outer loop) also get
// compiled, not just frequently-called callees
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
                    // branch jumps to imm when condition is FALSE (else block)
                    // falls through when TRUE (then block immediately follows)
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
                    if !take {
                        frame.pc = insn.imm() as usize;
                    }
                }
                OpCode::Return => {
                    let val_reg = frame.reg_base + insn.dst() as usize;
                    let ret_val = self.register_pool[val_reg];
                    if cfg!(debug_assertions) && frame.func_idx == 1 {
                        println!("heavy returning: {:?}", ret_val);
                    }
                    self.frames.pop();  // remove current frame
                    if let Some(caller) = self.frames.last_mut() {
                        self.register_pool[caller.return_reg] = ret_val;
                    } else {
                        return Ok(ret_val);
                    }
                }
                OpCode::Print => {
                    let reg = frame.reg_base + insn.dst() as usize;
                    let val = self.register_pool[reg];
                    if val.is_pyobject() {
                        // Ask Python to stringify — bind the raw ptr, call __str__
                        pyo3::Python::with_gil(|py| {
                            if let Some(raw) = val.to_pyobject() {
                                let obj = unsafe { pyo3::PyObject::from_borrowed_ptr(py, raw) };
                                let bound = obj.bind(py);
                                let s = bound.str()
                                    .map(|ps| ps.to_string())
                                    .unwrap_or_else(|_| "<PyObject>".to_string());
                                println!("{}", s);
                            }
                        });
                    } else {
                        println!("{:?}", val);
                    }
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

// not compiled yet – maybe compile it if hot
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

// interpreted call (including the first time after compilation attempt)
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

                // ── Python interop opcodes ────────────────────────────────────

                OpCode::ImportPython => {
                    // encoding: dst=rd, imm=string_pool_index
                    let dst = frame.reg_base + insn.dst() as usize;
                    let idx = insn.imm() as usize;
                    let module_name = self.string_pool[idx].clone();
                    let val = pyo3::Python::with_gil(|py| -> pyo3::PyResult<Value> {
                        let module = pyo3::types::PyModule::import(py, module_name.as_str())?;
                        // `into_ptr` consumes the reference and returns an OWNED (ref-count +1) pointer
                        let raw = pyo3::PyObject::from(module).into_ptr();
                        Ok(Value::from_pyobject(raw))
                    }).unwrap_or_else(|e| {
                        pyo3::Python::with_gil(|py| e.print(py));
                        panic!("ImportPython: failed to import module '{}'", module_name);
                    });
                    self.register_pool[dst] = val;
                }

                OpCode::GetAttr => {
                    // encoding: raw word — bits[23:16]=dst, bits[15:8]=obj_reg, bits[7:0]=attr_idx
                    let rd      = insn.dst() as usize;
                    let ro      = insn.src1() as usize;
                    let attr_idx = insn.src2() as usize; // attr string index stored in src2/lower byte
                    let dst     = frame.reg_base + rd;
                    let obj_reg = frame.reg_base + ro;
                    let attr    = self.string_pool[attr_idx].clone();
                    let obj_val = self.register_pool[obj_reg];
                    let raw_obj = obj_val.to_pyobject()
                        .unwrap_or_else(|| panic!("GetAttr: register is not a PyObject"));
                    let val = pyo3::Python::with_gil(|py| -> pyo3::PyResult<Value> {
                        let obj = unsafe { pyo3::PyObject::from_borrowed_ptr(py, raw_obj) };
                        let attr_obj = obj.getattr(py, attr.as_str())?;
                        let raw = attr_obj.into_ptr();
                        Ok(Value::from_pyobject(raw))
                    }).unwrap_or_else(|e| {
                        pyo3::Python::with_gil(|py| e.print(py));
                        panic!("GetAttr: failed to get attribute '{}'", attr);
                    });
                    self.register_pool[dst] = val;
                }

                OpCode::CallPython => {
                    // encoding: dst=rd, src1=callable_reg, src2=nargs
                    // Args live at frame.reg_base + rd + 1 .. + nargs  (placed by lowerer)
                    let rd   = insn.dst() as usize;
                    let rc   = insn.src1() as usize;
                    let nargs = insn.src2() as usize;
                    let dst          = frame.reg_base + rd;
                    let callable_reg = frame.reg_base + rc;
                    let callable_val = self.register_pool[callable_reg];
                    let raw_callable = callable_val.to_pyobject()
                        .unwrap_or_else(|| panic!("CallPython: register is not a callable PyObject"));

                    // Collect argument Values from the register window (rd+1, rd+2, …)
                    let mut arg_vals: Vec<Value> = Vec::with_capacity(nargs);
                    for i in 0..nargs {
                        arg_vals.push(self.register_pool[frame.reg_base + rd + 1 + i]);
                    }

                    let val = pyo3::Python::with_gil(|py| -> pyo3::PyResult<Value> {
                        let callable = unsafe { pyo3::PyObject::from_borrowed_ptr(py, raw_callable) };
                        // Convert each arg to a Python object
                        let py_args: Vec<pyo3::PyObject> = arg_vals.iter().map(|v| {
                            value_to_pyobject(py, *v)
                        }).collect::<pyo3::PyResult<Vec<_>>>()?;
                        // PyTuple::new returns PyResult<Bound<PyTuple>>; Bound<PyTuple>
                        // implements IntoPyObject<Target=PyTuple> so pass it directly to call1.
                        let tuple = pyo3::types::PyTuple::new(py, py_args)?;
                        let result = callable.call1(py, tuple)?;
                        // Convert result back to Value
                        pyobject_to_value(py, result)
                    }).unwrap_or_else(|e| {
                        pyo3::Python::with_gil(|py| e.print(py));
                        panic!("CallPython: call failed");
                    });
                    self.register_pool[dst] = val;
                }

                OpCode::ConvertToPy => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    let src = frame.reg_base + insn.src1() as usize;
                    let native = self.register_pool[src];
                    let py_val = pyo3::Python::with_gil(|py| -> pyo3::PyResult<Value> {
                        let obj = value_to_pyobject(py, native)?;
                        Ok(Value::from_pyobject(obj.into_ptr()))
                    }).unwrap_or_else(|e| {
                        pyo3::Python::with_gil(|py| e.print(py));
                        panic!("ConvertToPy: conversion failed");
                    });
                    self.register_pool[dst] = py_val;
                }

                OpCode::ConvertFromPy => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    let src = frame.reg_base + insn.src1() as usize;
                    let py_val = self.register_pool[src];
                    let raw = py_val.to_pyobject()
                        .unwrap_or_else(|| panic!("ConvertFromPy: src is not a PyObject"));
                    let native = pyo3::Python::with_gil(|py| -> pyo3::PyResult<Value> {
                        let obj = unsafe { pyo3::PyObject::from_borrowed_ptr(py, raw) };
                        pyobject_to_value(py, obj)
                    }).unwrap_or_else(|e| {
                        pyo3::Python::with_gil(|py| e.print(py));
                        panic!("ConvertFromPy: conversion failed");
                    });
                    self.register_pool[dst] = native;
                }
            }
        }
        Ok(Value::nil())
    }

    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }
}

// max registers per function frame — stack-allocated, no heap
const MAX_REGS_PER_FRAME: usize = 256;

// runtime helper called by JIT-compiled code to invoke another function

// hot path  : dispatch_table[func_idx] != 0  → one array load + indirect call
// cold path : record call, maybe compile, then call or interpret
#[no_mangle]
pub unsafe extern "C" fn ry_jit_call(
    vm: *mut VM,
    regs_ptr: *mut u64,  // caller frame; args at [0..num_params]
    func_idx: usize,
    _nargs: usize,
) -> u64 {
    let vm = &mut *vm;
    let num_params = vm.functions[func_idx].num_params;
    let num_regs   = vm.functions[func_idx].num_regs;

// stack-allocate a fresh callee frame — zero cost, no malloc
    let mut frame_buf = [0u64; MAX_REGS_PER_FRAME];
    let usable = num_regs.min(MAX_REGS_PER_FRAME);
    for i in 0..num_params.min(usable) {
        frame_buf[i] = *regs_ptr.add(i);
    }
    let callee_ptr   = frame_buf.as_mut_ptr();
    let dispatch_ptr = vm.dispatch_table.as_mut_ptr();

// hot path: dispatch table (single array index)
    if func_idx < vm.dispatch_table.len() && vm.dispatch_table[func_idx] != 0 {
        let native_fn: unsafe extern "C" fn(*mut VM, *mut u64, *mut u64) -> u64 =
            std::mem::transmute(vm.dispatch_table[func_idx]);
        return native_fn(vm as *mut VM, callee_ptr, dispatch_ptr);
    }

// cold path: maybe compile
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

// slow path: interpret inline
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
                // jump to imm when FALSE
                let cond = registers[insn.dst() as usize];
                let take = cond.to_bool().unwrap_or_else(|| cond.to_int().map(|i| i != 0).unwrap_or(false));
                if !take { pc = insn.imm() as usize; }
            }
            OpCode::Print => { println!("{:?}", registers[insn.dst() as usize]); }
            OpCode::Move  => {
                let (d,s) = (insn.dst() as usize, insn.src1() as usize);
                registers[d] = registers[s];
            }
            OpCode::Call  => {
// copy args from current frame into callee frame before recursing
                let nested_func = insn.imm() as usize;
                let nested_np = if nested_func < vm.functions.len() {
                    vm.functions[nested_func].num_params } else { 0 };
                let mut nested_buf = [0u64; MAX_REGS_PER_FRAME];
                for i in 0..nested_np.min(MAX_REGS_PER_FRAME) {
                    nested_buf[i] = *callee_ptr.add(i);
                }
                let ret = ry_jit_call(vm as *mut VM, nested_buf.as_mut_ptr(), nested_func, 0);
                registers[insn.dst() as usize] = Value(ret);
            }
            OpCode::Inc => {
                let r = insn.dst() as usize;
                registers[r] = registers[r] + Value::from_int(1);
            }
            // Python interop opcodes are not reachable via the JIT slow path
            // (the JIT defers to the main VM loop for these). If somehow hit,
            // panic with a clear message rather than silently miscompiling.
            OpCode::ImportPython | OpCode::GetAttr | OpCode::CallPython
            | OpCode::ConvertToPy | OpCode::ConvertFromPy => {
                panic!("Python interop opcode {:?} encountered in JIT slow-path interpreter; \
                        this should be executed by the main VM loop, not ry_jit_call", insn.opcode());
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