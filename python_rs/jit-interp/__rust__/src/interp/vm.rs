use crate::value::Value;
use crate::bytecode::{Instruction, OpCode};
use crate::arena::Arena;
use crate::jit::JitCompiler;
use std::rc::Rc;
use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::BoundObject;
use pyo3::types::{PyFloat, PyInt, PyBool, PyString, PyTuple};

/// convert a native [`Value`] to an owned python object.
fn value_to_pyobject(py: Python<'_>, v: Value) -> PyResult<PyObject> {
    if let Some(b) = v.to_bool() {
        return Ok(b.into_pyobject(py)?.into_bound().into_any().unbind());
    }
    if let Some(i) = v.to_int() {
        return Ok(i.into_pyobject(py)?.into_any().unbind());
    }
    if let Some(f) = v.to_f64() {
        return Ok(f.into_pyobject(py)?.into_any().unbind());
    }
    if v.is_pyobject() {
        let raw = v.to_pyobject().unwrap();
        return Ok(unsafe { PyObject::from_borrowed_ptr(py, raw) });
    }
    if let Some(s) = v.to_string_from_arena() {
        return Ok(s.into_pyobject(py)?.into_any().unbind());
    }
    Ok(py.None())
}

/// convert a python object to the closest native [`Value`].
fn pyobject_to_value(py: Python<'_>, obj: PyObject) -> PyResult<Value> {
    let bound = obj.bind(py);
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
        let raw = obj.into_ptr();
        return Ok(Value::from_pyobject(raw));
    }
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
    pub float_pool: Vec<f64>,
    /// interned python string objects for each string_pool entry.
    /// built lazily on first python interop use; avoids rust→pyunicode on every attr access.
    py_string_cache: Vec<Option<*mut pyo3::ffi::PyObject>>,

    /// inline method cache: maps (obj_ptr, attr_pool_idx) → cached bound method pyobject*.
    /// keyed by object identity so each instance gets its own bound method cached correctly.
    method_cache: std::collections::HashMap<(usize, usize), *mut pyo3::ffi::PyObject>,
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
        float_pool: Vec<f64>,
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

        let baseline_t = hot_threshold.max(1) as u64;
        let optimized_t = (baseline_t * 3).max(baseline_t + 1);
        let jit_compiler = JitCompiler::new(num_functions, baseline_t, optimized_t);
        let compiled_functions = HashMap::new();
        let dispatch_table = vec![0u64; num_functions];

        let pool_len = string_pool.len();
        let mut vm = VM {
            register_pool,
            frames,
            functions: funcs,
            string_pool,
            float_pool,
            py_string_cache: vec![None; pool_len],

            method_cache: std::collections::HashMap::new(),
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

    // helper: returns true if the bytecode contains any python‑interop opcode.
    fn has_python_interop(bytecode: &[Instruction]) -> bool {
    bytecode.iter().any(|insn| matches!(insn.opcode(),
        OpCode::ImportPython | OpCode::GetAttr | OpCode::PyCall | 
        OpCode::ConvertFromPy | OpCode::CallMethod | OpCode::Print))
}

    pub fn run(&mut self) -> Result<Value, String> {
        let needs_gil = self.functions.iter()
            .any(|f| VM::has_python_interop(&f.bytecode));
        if needs_gil {
            return pyo3::Python::with_gil(|_py| {
                self.run_inner()
            });
        }
        self.run_inner()
    }

    fn run_inner(&mut self) -> Result<Value, String> {
        let mut instruction_count = 0;
        const MAX_INSTRUCTIONS: usize = 10_000_000_000;

        while let Some(frame) = self.frames.last_mut() {
            instruction_count += 1;
            if instruction_count > MAX_INSTRUCTIONS {
                return Err(format!("infinite loop detected after {} instructions", instruction_count));
            }

            // osr at function entry
            if frame.pc == 0 {
                let func_idx = frame.func_idx;
                let reg_base = frame.reg_base;
                if self.dispatch_table.get(func_idx).copied().unwrap_or(0) == 0 {
                    let f = &self.functions[func_idx];
                    // do not compile if python interop is present
                    if !VM::has_python_interop(&f.bytecode) {
                        if let Some(jit) = &mut self.jit_compiler {
                            let (bc, np, nr) = (f.bytecode.clone(), f.num_params, f.num_regs);
                            if let Some(native_fn) = jit.compile_immediately(func_idx, &bc, np, nr) {
                                self.compiled_functions.insert(func_idx, native_fn);
                                if func_idx < self.dispatch_table.len() {
                                    self.dispatch_table[func_idx] = native_fn as u64;
                                }
                                self.frames.pop();
                                let regs_ptr = self.register_pool.as_mut_ptr().wrapping_add(reg_base) as *mut u64;
                                let dispatch_ptr = self.dispatch_table.as_mut_ptr();
                                let ret_val = unsafe { native_fn(self as *mut VM, regs_ptr, dispatch_ptr) };
                                if let Some(caller) = self.frames.last_mut() {
                                    self.register_pool[caller.return_reg] = Value(ret_val);
                                } else {
                                    return Ok(Value(ret_val));
                                }
                                continue;
                            }
                        }
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
                        panic!("loadstring: index {} out of bounds", idx);
                    }
                    let s = &self.string_pool[idx];
                    let py = unsafe { pyo3::Python::assume_gil_acquired() };
                    let py_str = pyo3::types::PyString::new(py, s);
                    let val = Value::from_pyobject(py_str.into_ptr());
                    if cfg!(debug_assertions) {
                        eprintln!("loadstring: idx={}, s='{}', value={:?}", idx, s, val);
                    }
                    self.register_pool[dst] = val;
                }
                OpCode::LoadFloat => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    let idx = insn.imm() as usize;
                    if idx >= self.float_pool.len() {
                        panic!("loadfloat: index {} out of bounds", idx);
                    }
                    let f = self.float_pool[idx];
                    self.register_pool[dst] = Value::from_f64(f);
                    if cfg!(debug_assertions) {
                        eprintln!("loadfloat: idx={}, f={}", idx, f);
                    }
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
                    if !take {
                        frame.pc = insn.imm() as usize;
                    }
                }
                OpCode::Return => {
                    let val_reg = frame.reg_base + insn.dst() as usize;
                    let ret_val = self.register_pool[val_reg];
                    self.frames.pop();
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
                        let py = unsafe { pyo3::Python::assume_gil_acquired() };
                        if let Some(raw) = val.to_pyobject() {
                            let obj = unsafe { pyo3::PyObject::from_borrowed_ptr(py, raw) };
                            let s = obj.bind(py).str()
                                .map(|ps| ps.to_string())
                                .unwrap_or_else(|_| "<pyobject>".to_string());
                            println!("{}", s);
                        }
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
                        panic!("call to undefined function index {}", func_idx);
                    }

                    let current_frame_func_idx = frame.func_idx;
                    let current_frame_num_regs = self.functions[current_frame_func_idx].num_regs;
                    let dst_reg = insn.dst() as usize;
                    let return_reg = frame.reg_base + dst_reg;
                    let num_params = self.functions[func_idx].num_params;
                    let new_base = frame.reg_base + current_frame_num_regs;

                    if new_base + self.functions[func_idx].num_regs > self.register_pool.len() {
                        panic!("call stack overflow");
                    }

                    for i in 0..num_params {
                        self.register_pool[new_base + i] = self.register_pool[frame.reg_base + i];
                    }

                    if func_idx < self.dispatch_table.len() && self.dispatch_table[func_idx] != 0 {
                        let fn_bits = self.dispatch_table[func_idx];
                        let native_fn: unsafe extern "C" fn(*mut VM, *mut u64, *mut u64) -> u64 =
                            unsafe { std::mem::transmute(fn_bits) };
                        let regs_ptr = self.register_pool.as_mut_ptr().wrapping_add(new_base) as *mut u64;
                        let dispatch_ptr = self.dispatch_table.as_mut_ptr();
                        let ret_val = unsafe { native_fn(self as *mut VM, regs_ptr, dispatch_ptr) };
                        self.register_pool[return_reg] = Value(ret_val);
                        continue;
                    }

                    if let Some(jit) = &mut self.jit_compiler {
                        self.profiler.record_call(func_idx);
                        let f = &self.functions[func_idx];
                        if !VM::has_python_interop(&f.bytecode) {
                            let (bytecode, np, nr) = (f.bytecode.clone(), f.num_params, f.num_regs);
                            if let Some((_tier, native_fn)) = jit.record_and_maybe_compile(func_idx, &bytecode, np, nr) {
                                self.compiled_functions.insert(func_idx, native_fn);
                                if func_idx < self.dispatch_table.len() {
                                    self.dispatch_table[func_idx] = native_fn as u64;
                                }
                                let regs_ptr = self.register_pool.as_mut_ptr().wrapping_add(new_base) as *mut u64;
                                let dispatch_ptr = self.dispatch_table.as_mut_ptr();
                                let ret_val = unsafe { native_fn(self as *mut VM, regs_ptr, dispatch_ptr) };
                                self.register_pool[return_reg] = Value(ret_val);
                                continue;
                            }
                        }
                    }

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

                // python interop opcodes (interpreted only)

                OpCode::ImportPython => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    let idx = insn.imm() as usize;
                    let module_name = self.string_pool[idx].clone();
                    let py = unsafe { pyo3::Python::assume_gil_acquired() };
                    match pyo3::types::PyModule::import(py, module_name.as_str()) {
                        Ok(module) => {
                            self.register_pool[dst] = Value::from_pyobject(PyObject::from(module).into_ptr());
                        }
                        Err(e) => {
                            let err_str = e.to_string();
                            return Err(format!("importpython: failed to import '{}': {}", module_name, err_str));
                        }
                    }
                }

                OpCode::GetAttr => {
    let rd = insn.dst() as usize;
    let ro = insn.src1() as usize;
    let attr_idx = insn.src2() as usize;
    let dst = frame.reg_base + rd;
    let obj_reg = frame.reg_base + ro;
    let mut obj_val = self.register_pool[obj_reg];
    if !obj_val.is_pyobject() {
        let py = unsafe { pyo3::Python::assume_gil_acquired() };
        let py_obj = value_to_pyobject(py, obj_val).unwrap_or_else(|_| py.None());
        obj_val = Value::from_pyobject(py_obj.into_ptr());
        self.register_pool[obj_reg] = obj_val;
    }
    let obj_ptr = obj_val.to_pyobject().unwrap();
    // key by object identity, not type
    let cache_key = (obj_ptr as usize, attr_idx);
    let cached_ptr: Option<*mut pyo3::ffi::PyObject> =
        self.method_cache.get(&cache_key).copied();
    let result = if let Some(cached) = cached_ptr {
        unsafe { pyo3::ffi::Py_INCREF(cached); }
        cached
    } else {
        let attr_str = self.get_py_str(attr_idx);
        let r = unsafe { pyo3::ffi::PyObject_GetAttr(obj_ptr, attr_str) };
        if r.is_null() {
            let py = unsafe { pyo3::Python::assume_gil_acquired() };
            let attr_name = self.string_pool[attr_idx].clone();
            let err = PyErr::fetch(py);
            let err_str = err.to_string();
            return Err(format!("getattr: failed to get attribute '{}': {}", attr_name, err_str));
        }
        self.method_cache.insert(cache_key, r);
        unsafe { pyo3::ffi::Py_INCREF(r); }
        r
    };
    self.register_pool[dst] = Value::from_pyobject(result);
}

                OpCode::PyCall => {
                    let rd = insn.dst() as usize;
                    let rc = insn.src1() as usize;
                    let nargs = insn.src2() as usize;
                    let dst = frame.reg_base + rd;
                    let callable_reg = frame.reg_base + rc;

                    // consume payload before acquiring py — frame is &mut from last_mut().
                    let arg_base = {
                        let func_idx = frame.func_idx;
                        let p = self.functions[func_idx].bytecode[frame.pc];
                        frame.pc += 1;
                        p.dst() as usize
                    };

                    // now safe to get py — frame borrow from last_mut() was used only above.
                    let py = unsafe { pyo3::Python::assume_gil_acquired() };
                    let mut callable_val = self.register_pool[callable_reg];
                    if !callable_val.is_pyobject() {
                        let py_obj = value_to_pyobject(py, callable_val).unwrap_or_else(|_| py.None());
                        callable_val = Value::from_pyobject(py_obj.into_ptr());
                        self.register_pool[callable_reg] = callable_val;
                    }
                    let callable = unsafe { PyObject::from_borrowed_ptr(py, callable_val.to_pyobject().unwrap()) };

                    let arg_vals: Vec<Value> = (0..nargs)
                        .map(|i| self.register_pool[frame.reg_base + arg_base + i])
                        .collect();

                    // helper: try to extract a rust string from a value (arena string or pystr).
                    let val_to_str = |v: Value| -> Option<String> {
                        if let Some(s) = v.to_string_from_arena() { return Some(s); }
                        if v.is_pyobject() {
                            if let Some(ptr) = v.to_pyobject() {
                                let obj = unsafe { PyObject::from_borrowed_ptr(py, ptr) };
                                return obj.bind(py).extract::<String>().ok();
                            }
                        }
                        None
                    };

                    // detect __kw__:key sentinel pattern injected by lib.rs for calls
                    // with keyword arguments, e.g. torch.tensor(3.0, requires_grad=true).
                    // arg list layout: [...pos_args, string("__kw__:key0"), val0, …]
                    let has_kwargs = arg_vals.iter().any(|v| {
                        val_to_str(*v).map_or(false, |s| s.starts_with("__kw__:"))
                    });

                    let result_raw = if has_kwargs {
                        let mut pos_py: Vec<PyObject> = Vec::new();
                        let kwargs = pyo3::types::PyDict::new(py);
                        let mut i = 0;
                        while i < arg_vals.len() {
                            let v = arg_vals[i];
                            if let Some(s) = val_to_str(v) {
                                if s.starts_with("__kw__:") {
                                    let key = &s["__kw__:".len()..];
                                    i += 1;
                                    if i < arg_vals.len() {
                                        let kw_val = value_to_pyobject(py, arg_vals[i])
                                            .unwrap_or_else(|_| py.None());
                                        kwargs.set_item(key, kw_val).unwrap();
                                        i += 1;
                                    }
                                    continue;
                                }
                            }
                            pos_py.push(value_to_pyobject(py, v).unwrap_or_else(|_| py.None()));
                            i += 1;
                        }
                        let args_tuple = PyTuple::new(py, pos_py).unwrap();
                        let r = unsafe {
                            pyo3::ffi::PyObject_Call(
                                callable.as_ptr(),
                                args_tuple.as_ptr(),
                                kwargs.as_ptr(),
                            )
                        };
                        if r.is_null() {
                            let err = PyErr::fetch(py);
                            let err_str = err.to_string();
                            return Err(format!("pycall (kwargs): call failed: {}", err_str));
                        }
                        r
                    } else {
                        let build_py_args = |coerce_to_bool: bool| -> Vec<PyObject> {
                            arg_vals.iter().map(|v| {
                                if coerce_to_bool {
                                    if let Some(i) = v.to_int() {
                                        if i == 0 || i == 1 {
                                            let obj = (i != 0).into_pyobject(py).unwrap();
                                            return obj.into_bound().into_any().unbind();
                                        }
                                    }
                                }
                                value_to_pyobject(py, *v).unwrap_or_else(|_| py.None())
                            }).collect()
                        };
                        let fast_call = |args: Vec<PyObject>| -> *mut pyo3::ffi::PyObject {
                            unsafe {
                                match args.as_slice() {
                                    [] => pyo3::ffi::PyObject_CallNoArgs(callable.as_ptr()),
                                    [a0] => pyo3::ffi::PyObject_CallOneArg(callable.as_ptr(), a0.as_ptr()),
                                    _ => {
                                        let t = PyTuple::new(py, args).unwrap();
                                        pyo3::ffi::PyObject_Call(callable.as_ptr(), t.as_ptr(), std::ptr::null_mut())
                                    }
                                }
                            }
                        };
                        let result = fast_call(build_py_args(false));
                        if result.is_null() {
                            let err = PyErr::fetch(py);
                            if err.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                                let result2 = fast_call(build_py_args(true));
                                if result2.is_null() {
                                    let err2 = PyErr::fetch(py);
                                    let err_str = err2.to_string();
                                    return Err(format!("pycall: call failed (after bool‑coerce retry): {}", err_str));
                                }
                                result2
                            } else {
                                let err_str = err.to_string();
                                return Err(format!("pycall: call failed: {}", err_str));
                            }
                        } else {
                            result
                        }
                    };
                    self.register_pool[dst] = Value::from_pyobject(result_raw);
                }

                OpCode::ConvertFromPy => {
                    let dst = frame.reg_base + insn.dst() as usize;
                    let src = frame.reg_base + insn.src1() as usize;
                    let src_val = self.register_pool[src];
                    if !src_val.is_pyobject() {
                        self.register_pool[dst] = src_val;
                        continue;
                    }
                    let py = unsafe { pyo3::Python::assume_gil_acquired() };
                    let obj = unsafe { PyObject::from_borrowed_ptr(py, src_val.to_pyobject().unwrap()) };
                    let bound = obj.bind(py);
                    self.register_pool[dst] = if let Ok(b) = bound.extract::<bool>() {
                        Value::from_bool(b)
                    } else if let Ok(i) = bound.extract::<i64>() {
                        Value::from_int(i)
                    } else if let Ok(f) = bound.extract::<f64>() {
                        Value::from_f64(f)
                    } else {
                        src_val
                    };
                }

                // fused method call: getattr + pycall in one gil acquisition
                OpCode::CallMethod => {
                    // instruction layout:
                    //   insn:    encode_rrr(callmethod, dst, obj_reg, nargs)
                    //   insn+1:  encode_imm(loadnil, 0, attr_pool_idx)  ← payload
                    let rd = insn.dst() as usize;
                    let ro = insn.src1() as usize;
                    let nargs = insn.src2() as usize;
                    let dst = frame.reg_base + rd;
                    let obj_reg = frame.reg_base + ro;

                    // consume the payload instruction: dst=arg_base, imm=attr_pool_idx
                    let (attr_idx, arg_base) = {
                        let func_idx = frame.func_idx;
                        let p = self.functions[func_idx].bytecode[frame.pc];
                        frame.pc += 1;
                        (p.imm() as usize, p.dst() as usize)
                    };

                    let reg_base = frame.reg_base;
                    let mut obj_val = self.register_pool[obj_reg];
                    let attr_name = self.string_pool[attr_idx].clone();

                    if cfg!(debug_assertions) {
                        eprintln!("callmethod: obj={:?} (py={}), method='{}', nargs={}",
                                  obj_val, obj_val.is_pyobject(), attr_name, nargs);
                    }

                    // collect args from scratch slots above all live vars.
                    let arg_vals: Vec<Value> = (0..nargs)
                        .map(|i| self.register_pool[reg_base + arg_base + i])
                        .collect();

                    // ensure obj is a pyobject — coerce before cache lookup.
                    if !obj_val.is_pyobject() {
                        let py = unsafe { pyo3::Python::assume_gil_acquired() };
                        let py_obj = value_to_pyobject(py, obj_val).unwrap_or_else(|_| py.None());
                        obj_val = Value::from_pyobject(py_obj.into_ptr());
                        self.register_pool[obj_reg] = obj_val;
                        if cfg!(debug_assertions) {
                            eprintln!("converted obj_val to pyobject: {:?}", obj_val);
                        }
                    }
                    let obj_raw_ptr = obj_val.to_pyobject().unwrap();
                    // key by object identity (obj_ptr), not type_ptr.
                    // type_ptr was wrong: all tensors share the same type_ptr, so
                    // arange(4).float() would cache the bound method, and then
                    // arange(6).float() would reuse the same bound method — which
                    // is still bound to the 4‑element tensor — producing size=4 errors.
                    let cache_key = (obj_raw_ptr as usize, attr_idx);

                    // immutable cache lookup ends before any mut borrow.
                    let cached_ptr: Option<*mut pyo3::ffi::PyObject> =
                        self.method_cache.get(&cache_key).copied();

                    // get_py_str is &mut self — call before acquiring py.
                    let attr_str_for_miss = if cached_ptr.is_none() {
                        Some(self.get_py_str(attr_idx))
                    } else { None };

                    let py = unsafe { pyo3::Python::assume_gil_acquired() };
                    let result_raw: *mut pyo3::ffi::PyObject = {
                        let method_ptr = if let Some(cached) = cached_ptr {
                            unsafe { pyo3::ffi::Py_INCREF(cached); }
                            cached
                        } else {
                            let r = unsafe { pyo3::ffi::PyObject_GetAttr(obj_raw_ptr, attr_str_for_miss.unwrap()) };
                            if r.is_null() {
                                let err = PyErr::fetch(py);
                                let err_str = err.to_string();
                                return Err(format!("callmethod: getattr '{}' failed: {}", attr_name, err_str));
                            }
                            self.method_cache.insert(cache_key, r);
                            unsafe { pyo3::ffi::Py_INCREF(r); }
                            r
                        };
                        let method = unsafe { PyObject::from_owned_ptr(py, method_ptr) };

                        let build_args = |coerce_to_bool: bool| -> Vec<PyObject> {
                            arg_vals.iter().map(|v| {
                                if coerce_to_bool {
                                    if let Some(i) = v.to_int() {
                                        if i == 0 || i == 1 {
                                            let b: bool = i != 0;
                                            let obj = b.into_pyobject(py).unwrap();
                                            return obj.into_bound().into_any().unbind();
                                        }
                                    }
                                }
                                value_to_pyobject(py, *v).unwrap_or_else(|_| py.None())
                            }).collect()
                        };

                        let call_with = |args: Vec<PyObject>| -> *mut pyo3::ffi::PyObject {
                            // for 0‑3 args use pyobject_callnoargs / _callonearg / pytuple_pack
                            // to avoid the vec→pytuple round‑trip on the hot path.
                            unsafe {
                                match args.as_slice() {
                                    [] => pyo3::ffi::PyObject_CallNoArgs(method.as_ptr()),
                                    [a0] => pyo3::ffi::PyObject_CallOneArg(method.as_ptr(), a0.as_ptr()),
                                    _ => {
                                        let tuple = PyTuple::new(py, args).unwrap();
                                        pyo3::ffi::PyObject_Call(method.as_ptr(), tuple.as_ptr(), std::ptr::null_mut())
                                    }
                                }
                            }
                        };

                        let result = call_with(build_args(false));
                        if result.is_null() {
                            // check if it is a typeerror — if so, retry with bool coercion.
                            let err = PyErr::fetch(py);
                            if err.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                                let result2 = call_with(build_args(true));
                                if result2.is_null() {
                                    let err2 = PyErr::fetch(py);
                                    let err_str = err2.to_string();
                                    return Err(format!("callmethod '{}': call failed (after bool‑coerce retry): {}", attr_name, err_str));
                                }
                                result2
                            } else {
                                let err_str = err.to_string();
                                return Err(format!("callmethod '{}': call failed: {}", attr_name, err_str));
                            }
                        } else {
                            result
                        }
                    };
                    self.register_pool[dst] = Value::from_pyobject(result_raw);
                }
            }
        }
        Ok(Value::nil())
    }

    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }

    /// return a borrowed pyobject* for string_pool[idx], creating and interning it on first use.
    /// must be called while the gil is held.
    #[inline]
    fn get_py_str(&mut self, idx: usize) -> *mut pyo3::ffi::PyObject {
        if let Some(ptr) = self.py_string_cache[idx] {
            return ptr;
        }
        let s = &self.string_pool[idx];
        // pyunicode_interninplace modifies the pointer in‑place to point to the
        // canonical interned object. must pass a proper *mut *mut and read back
        // the (possibly changed) pointer after the call.
        let ptr = unsafe {
            let mut interned = pyo3::ffi::PyUnicode_FromStringAndSize(
                s.as_ptr() as *const std::os::raw::c_char,
                s.len() as isize,
            );
            pyo3::ffi::PyUnicode_InternInPlace(&mut interned);
            interned  // now points to the canonical interned string
        };
        self.py_string_cache[idx] = Some(ptr);
        ptr
    }
}

impl Drop for VM {
    fn drop(&mut self) {
        // decref all cached method objects.
        if !self.method_cache.is_empty() {
            if let Ok(_guard) = pyo3::Python::with_gil(|_py| -> Result<(), ()> {
                for &ptr in self.method_cache.values() {
                    unsafe { pyo3::ffi::Py_DECREF(ptr); }
                }
                Ok(())
            }) {}
        }
    }
}

// jit runtime helpers

#[no_mangle]
pub unsafe extern "C" fn ry_jit_import(vm: *mut VM, name_idx: u64) -> u64 {
    let vm = &mut *vm;
    let name = &vm.string_pool[name_idx as usize];
    pyo3::Python::with_gil(|py| {
        let module = pyo3::types::PyModule::import(py, name)
            .unwrap_or_else(|e| {
                e.print(py);
                panic!("ry_jit_import: failed to import module '{}'", name);
            });
        let raw = PyObject::from(module).into_ptr();
        Value::from_pyobject(raw).0
    })
}

#[no_mangle]
pub unsafe extern "C" fn ry_jit_getattr(vm: *mut VM, obj: u64, attr_idx: u64) -> u64 {
    let vm = &mut *vm;
    let obj_val = Value(obj);
    let attr_name = &vm.string_pool[attr_idx as usize];
    if !obj_val.is_pyobject() {
        panic!("ry_jit_getattr: object is not a pyobject");
    }
    pyo3::Python::with_gil(|py| {
        let py_obj = unsafe { PyObject::from_borrowed_ptr(py, obj_val.to_pyobject().unwrap()) };
        let attr = py_obj.getattr(py, attr_name).unwrap_or_else(|e| {
            e.print(py);
            panic!("ry_jit_getattr: failed to get attribute '{}'", attr_name);
        });
        let raw = attr.into_ptr();
        Value::from_pyobject(raw).0
    })
}

#[no_mangle]
pub unsafe extern "C" fn ry_jit_callpython(
    vm: *mut VM,
    callable: u64,
    args_ptr: *mut u64,
    nargs: u64,
) -> u64 {
    let _vm = &mut *vm;
    let callable_val = Value(callable);
    if !callable_val.is_pyobject() {
        panic!("ry_jit_callpython: callable is not a pyobject");
    }
    let args_slice = std::slice::from_raw_parts(args_ptr as *const Value, nargs as usize);
    pyo3::Python::with_gil(|py| {
        let py_callable = unsafe { PyObject::from_borrowed_ptr(py, callable_val.to_pyobject().unwrap()) };

        let build_py_args = |coerce_to_bool: bool| -> Vec<PyObject> {
            args_slice.iter().map(|&arg| {
                if coerce_to_bool {
                    if let Some(i) = arg.to_int() {
                        if i == 0 || i == 1 {
                            let b: bool = i != 0;
                            let obj = b.into_pyobject(py).unwrap();
                            return obj.into_bound().into_any().unbind();
                        }
                    }
                }
                value_to_pyobject(py, arg).unwrap_or_else(|e| {
                    panic!("ry_jit_callpython: argument conversion failed: {}", e);
                })
            }).collect()
        };

        let tuple = PyTuple::new(py, build_py_args(false)).unwrap();
        match py_callable.call1(py, tuple) {
            Ok(result) => Value::from_pyobject(result.into_ptr()).0,
            Err(e) => {
                if e.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                    let tuple2 = PyTuple::new(py, build_py_args(true)).unwrap();
                    match py_callable.call1(py, tuple2) {
                        Ok(result) => Value::from_pyobject(result.into_ptr()).0,
                        Err(e2) => { e2.print(py); panic!("ry_jit_callpython: call failed (after bool‑coerce retry)"); }
                    }
                } else {
                    e.print(py);
                    panic!("ry_jit_callpython: call failed");
                }
            }
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn ry_jit_convert(_vm: *mut VM, src: u64) -> u64 {
    let src_val = Value(src);
    if !src_val.is_pyobject() {
        return src;
    }
    let maybe_native = pyo3::Python::with_gil(|py| {
        let obj = unsafe { PyObject::from_borrowed_ptr(py, src_val.to_pyobject().unwrap()) };
        let bound = obj.bind(py);
        if let Ok(i) = bound.extract::<i64>() {
            return Some(Value::from_int(i).0);
        }
        if let Ok(b) = bound.extract::<bool>() {
            return Some(Value::from_bool(b).0);
        }
        if let Ok(f) = bound.extract::<f64>() {
            return Some(Value::from_f64(f).0);
        }
        None
    });
    maybe_native.unwrap_or(src)
}

/// helper for jit: load a float constant from the float pool.
#[no_mangle]
pub unsafe extern "C" fn ry_load_float(vm: *mut VM, idx: u64) -> u64 {
    let vm = &mut *vm;
    let idx = idx as usize;
    if idx >= vm.float_pool.len() {
        panic!("ry_load_float: index {} out of bounds", idx);
    }
    let f = vm.float_pool[idx];
    Value::from_f64(f).0
}

/// helper for jit: print a value.
#[no_mangle]
pub unsafe extern "C" fn ry_print_value(vm: *mut VM, val: u64) -> u64 {
    let val = Value(val);
    if val.is_pyobject() {
        pyo3::Python::with_gil(|py| {
            if let Some(raw) = val.to_pyobject() {
                let obj = unsafe { pyo3::PyObject::from_borrowed_ptr(py, raw) };
                let s = obj.bind(py).str()
                    .map(|ps| ps.to_string())
                    .unwrap_or_else(|_| "<pyobject>".to_string());
                println!("{}", s);
            }
        });
    } else {
        println!("{:?}", val);
    }
    0 // return value unused
}

// slow‑path interpreter with python opcode support

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

    const MAX_REGS_PER_FRAME: usize = 256;
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

    // slow path: interpret inline, with gil if needed
    let bytecode = vm.functions[func_idx].bytecode.clone();
    let registers = std::slice::from_raw_parts_mut(callee_ptr as *mut Value, usable);

    // determine if any python opcode is present in this function
    let has_python = bytecode.iter().any(|insn| matches!(insn.opcode(),
        OpCode::ImportPython | OpCode::GetAttr | OpCode::PyCall | OpCode::ConvertFromPy | OpCode::CallMethod));

    if has_python {
        // run with gil
        return pyo3::Python::with_gil(|py| {
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
                    OpCode::LoadFloat => {
                        let f = vm.float_pool[insn.imm() as usize];
                        registers[insn.dst() as usize] = Value::from_f64(f);
                    }
                    OpCode::Jump   => { pc = insn.imm() as usize; }
                    OpCode::Branch => {
                        let cond = registers[insn.dst() as usize];
                        let take = cond.to_bool().unwrap_or_else(|| cond.to_int().map(|i| i != 0).unwrap_or(false));
                        if !take { pc = insn.imm() as usize; }
                    }
                    OpCode::Print => {
                        let v = registers[insn.dst() as usize].0;
                        ry_print_value(vm as *mut VM, v);
                    }
                    OpCode::Move  => {
                        let (d,s) = (insn.dst() as usize, insn.src1() as usize);
                        registers[d] = registers[s];
                    }
                    OpCode::Call  => {
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

                    // python interop opcodes
                    OpCode::ImportPython => {
                        let dst = insn.dst() as usize;
                        let idx = insn.imm() as usize;
                        let module_name = vm.string_pool[idx].clone();
                        match pyo3::types::PyModule::import(py, module_name.as_str()) {
                            Ok(module) => {
                                registers[dst] = Value::from_pyobject(PyObject::from(module).into_ptr());
                            }
                            Err(e) => {
                                panic!("importpython failed: {}", e);
                            }
                        }
                    }
                    OpCode::GetAttr => {
                        let dst = insn.dst() as usize;
                        let obj_reg = insn.src1() as usize;
                        let attr_idx = insn.src2() as usize;
                        let mut obj_val = registers[obj_reg];
                        if !obj_val.is_pyobject() {
                            let py_obj = value_to_pyobject(py, obj_val).unwrap_or_else(|_| py.None());
                            obj_val = Value::from_pyobject(py_obj.into_ptr());
                            registers[obj_reg] = obj_val;
                        }
                        let obj_ptr = obj_val.to_pyobject().unwrap();
                        let attr_name = vm.string_pool[attr_idx].as_str();
                        let attr = unsafe { PyObject::from_borrowed_ptr(py, pyo3::ffi::PyObject_GetAttrString(obj_ptr, attr_name.as_ptr() as *const i8)) };
                        if attr.is_none(py) {
                            panic!("getattr failed for '{}'", attr_name);
                        }
                        registers[dst] = Value::from_pyobject(attr.into_ptr());
                    }
                    OpCode::PyCall => {
                        let dst = insn.dst() as usize;
                        let callable_reg = insn.src1() as usize;
                        let nargs = insn.src2() as usize;
                        let arg_base = {
                            let p = bytecode[pc];
                            pc += 1;
                            p.dst() as usize
                        };
                        let mut callable_val = registers[callable_reg];
                        if !callable_val.is_pyobject() {
                            let py_obj = value_to_pyobject(py, callable_val).unwrap_or_else(|_| py.None());
                            callable_val = Value::from_pyobject(py_obj.into_ptr());
                            registers[callable_reg] = callable_val;
                        }
                        let callable = unsafe { PyObject::from_borrowed_ptr(py, callable_val.to_pyobject().unwrap()) };
                        let args: Vec<PyObject> = (0..nargs)
                            .map(|i| {
                                let v = registers[arg_base + i];
                                value_to_pyobject(py, v).unwrap_or_else(|_| py.None())
                            })
                            .collect();
                        let tuple = PyTuple::new(py, args).unwrap();
                        match callable.call1(py, tuple) {
                            Ok(result) => registers[dst] = Value::from_pyobject(result.into_ptr()),
                            Err(e) => panic!("pycall failed: {}", e),
                        }
                    }
                    OpCode::ConvertFromPy => {
                        let dst = insn.dst() as usize;
                        let src = insn.src1() as usize;
                        let src_val = registers[src];
                        if !src_val.is_pyobject() {
                            registers[dst] = src_val;
                        } else {
                            let obj = unsafe { PyObject::from_borrowed_ptr(py, src_val.to_pyobject().unwrap()) };
                            let bound = obj.bind(py);
                            registers[dst] = if let Ok(i) = bound.extract::<i64>() {
                                Value::from_int(i)
                            } else if let Ok(b) = bound.extract::<bool>() {
                                Value::from_bool(b)
                            } else if let Ok(f) = bound.extract::<f64>() {
                                Value::from_f64(f)
                            } else {
                                src_val
                            };
                        }
                    }
                    OpCode::CallMethod => {
                        let dst = insn.dst() as usize;
                        let obj_reg = insn.src1() as usize;
                        let nargs = insn.src2() as usize;
                        let (attr_idx, arg_base) = {
                            let p = bytecode[pc];
                            pc += 1;
                            (p.imm() as usize, p.dst() as usize)
                        };
                        let mut obj_val = registers[obj_reg];
                        if !obj_val.is_pyobject() {
                            let py_obj = value_to_pyobject(py, obj_val).unwrap_or_else(|_| py.None());
                            obj_val = Value::from_pyobject(py_obj.into_ptr());
                            registers[obj_reg] = obj_val;
                        }
                        let obj_ptr = obj_val.to_pyobject().unwrap();
                        let method_name = vm.string_pool[attr_idx].as_str();
                        let method = unsafe { PyObject::from_borrowed_ptr(py, pyo3::ffi::PyObject_GetAttrString(obj_ptr, method_name.as_ptr() as *const i8)) };
                        if method.is_none(py) {
                            panic!("callmethod: method '{}' not found", method_name);
                        }
                        let args: Vec<PyObject> = (0..nargs)
                            .map(|i| {
                                let v = registers[arg_base + i];
                                value_to_pyobject(py, v).unwrap_or_else(|_| py.None())
                            })
                            .collect();
                        let tuple = PyTuple::new(py, args).unwrap();
                        match method.call1(py, tuple) {
                            Ok(result) => registers[dst] = Value::from_pyobject(result.into_ptr()),
                            Err(e) => panic!("callmethod '{}' failed: {}", method_name, e),
                        }
                    }
                }
            }
            0
        });
    } else {
        // no python opcodes, run without gil
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
                OpCode::LoadFloat => {
                    let f = vm.float_pool[insn.imm() as usize];
                    registers[insn.dst() as usize] = Value::from_f64(f);
                }
                OpCode::Jump   => { pc = insn.imm() as usize; }
                OpCode::Branch => {
                    let cond = registers[insn.dst() as usize];
                    let take = cond.to_bool().unwrap_or_else(|| cond.to_int().map(|i| i != 0).unwrap_or(false));
                    if !take { pc = insn.imm() as usize; }
                }
                OpCode::Print => {
                    let v = registers[insn.dst() as usize].0;
                    ry_print_value(vm as *mut VM, v);
                }
                OpCode::Move  => {
                    let (d,s) = (insn.dst() as usize, insn.src1() as usize);
                    registers[d] = registers[s];
                }
                OpCode::Call  => {
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
                // if a python opcode appears here, the `has_python` check was wrong.
                op @ (OpCode::ImportPython | OpCode::GetAttr | OpCode::PyCall | OpCode::ConvertFromPy | OpCode::CallMethod) => {
                    panic!("python interop opcode {:?} encountered in jit slow‑path interpreter without gil (has_python={})", op, has_python);
                }
            }
        }
        0
    }
}

// other runtime helpers

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