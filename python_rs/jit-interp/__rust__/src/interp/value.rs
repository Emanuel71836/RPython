use std::fmt;
use std::ops::{Add, Sub, Mul, Div};
use crate::arena::Arena;
use pyo3::prelude::*;
use pyo3::ffi;

const QNAN: u64 = 0x7ff8_0000_0000_0000;
const TAG_MASK: u64 = 0xffff_0000_0000_0000;
const PAYLOAD_MASK: u64 = 0x0000_ffff_ffff_ffff;
const TAG_BOOL: u64 = 0x0001_0000_0000_0000;
const TAG_INT: u64 = 0x0002_0000_0000_0000;
const TAG_OBJ: u64 = 0x0003_0000_0000_0000;
const TAG_PYOBJECT: u64 = 0x0004_0000_0000_0000;

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Value(pub u64);

impl Value {
    pub fn from_bool(b: bool) -> Self {
        Value(QNAN | TAG_BOOL | (b as u64))
    }
    pub fn to_bool(self) -> Option<bool> {
        if (self.0 & QNAN) == QNAN && (self.0 & TAG_MASK) == (QNAN | TAG_BOOL) {
            Some((self.0 & 1) != 0)
        } else {
            None
        }
    }

    pub fn from_int(i: i64) -> Self {
        Value(QNAN | TAG_INT | ((i as u64) & PAYLOAD_MASK))
    }
    pub fn to_int(self) -> Option<i64> {
        if (self.0 & QNAN) == QNAN && (self.0 & TAG_MASK) == (QNAN | TAG_INT) {
            let i = (self.0 & PAYLOAD_MASK) as i64;
            Some((i << 16) >> 16)
        } else {
            None
        }
    }

    pub fn from_f64(f: f64) -> Self {
        Value(f.to_bits())
    }
    pub fn to_f64(self) -> Option<f64> {
        if (self.0 & QNAN) != QNAN {
            Some(f64::from_bits(self.0))
        } else {
            None
        }
    }

    pub fn from_ptr<T>(ptr: *mut T) -> Self {
        let ptr_bits = ptr as u64;
        assert!(ptr_bits & !PAYLOAD_MASK == 0);
        Value(QNAN | TAG_OBJ | ptr_bits)
    }
    pub fn to_ptr<T>(self) -> Option<*mut T> {
        if (self.0 & QNAN) == QNAN && (self.0 & TAG_MASK) == (QNAN | TAG_OBJ) {
            Some((self.0 & PAYLOAD_MASK) as *mut T)
        } else {
            None
        }
    }

    pub fn from_string_in_arena(s: &str, arena: &mut Arena) -> Self {
        let len = s.len();
        let total_size = std::mem::size_of::<usize>() + len;
        let align = std::mem::align_of::<usize>();
        let ptr = arena.alloc_aligned(total_size, align);
        unsafe {
            *(ptr as *mut usize) = len;
            if len > 0 {
                let data_ptr = ptr.add(std::mem::size_of::<usize>());
                std::ptr::copy_nonoverlapping(s.as_ptr(), data_ptr, len);
            }
        }
        Self::from_ptr(ptr as *mut ())
    }
    pub fn to_string_from_arena(self) -> Option<String> {
        if let Some(ptr) = self.to_ptr::<()>() {
            unsafe {
                let len_ptr = ptr as *const usize;
                let len = *len_ptr;
                let data_ptr = (ptr as *const u8).add(std::mem::size_of::<usize>());
                let slice = std::slice::from_raw_parts(data_ptr, len);
                Some(String::from_utf8_lossy(slice).to_string())
            }
        } else {
            None
        }
    }

    pub fn nil() -> Self {
        Value(0)
    }
    pub fn is_nil(self) -> bool {
        self.0 == 0
    }

    // python object support
    pub fn from_pyobject(obj: *mut pyo3::ffi::PyObject) -> Self {
        let ptr = obj as u64;
        assert!(ptr & !PAYLOAD_MASK == 0, "pyobject pointer too large");
        Value(QNAN | TAG_PYOBJECT | ptr)
    }
    pub fn to_pyobject(self) -> Option<*mut pyo3::ffi::PyObject> {
        if (self.0 & QNAN) == QNAN && (self.0 & TAG_MASK) == (QNAN | TAG_PYOBJECT) {
            Some((self.0 & PAYLOAD_MASK) as *mut pyo3::ffi::PyObject)
        } else {
            None
        }
    }
    pub fn is_pyobject(self) -> bool {
        (self.0 & QNAN) == QNAN && (self.0 & TAG_MASK) == (QNAN | TAG_PYOBJECT)
    }
    pub unsafe fn pyobject_decref(self) {
        if let Some(ptr) = self.to_pyobject() {
            pyo3::ffi::Py_DECREF(ptr);
        }
    }

    // convert to owned python object pointer (caller must decref)
    fn to_pyobject_owned(&self, _py: Python) -> *mut ffi::PyObject {
        if let Some(ptr) = self.to_pyobject() {
            unsafe { ffi::Py_INCREF(ptr); }
            return ptr;
        }
        if let Some(i) = self.to_int() {
            return unsafe { ffi::PyLong_FromLong(i as std::os::raw::c_long) };
        }
        if let Some(b) = self.to_bool() {
            return unsafe { ffi::PyBool_FromLong(b as std::os::raw::c_long) };
        }
        if let Some(f) = self.to_f64() {
            return unsafe { ffi::PyFloat_FromDouble(f) };
        }
        if let Some(s) = self.to_string_from_arena() {
            return unsafe { ffi::PyUnicode_FromStringAndSize(s.as_ptr() as *const _, s.len() as isize) };
        }
        unsafe {
            ffi::Py_INCREF(ffi::Py_None());
            ffi::Py_None()
        }
    }

    // python arithmetic helpers
    fn py_add(&self, rhs: &Self) -> Self {
        let py = unsafe { Python::assume_gil_acquired() };
        let a = self.to_pyobject_owned(py);
        let b = rhs.to_pyobject_owned(py);
        if cfg!(debug_assertions) {
            eprintln!("py_add: a={:?}, b={:?}", self, rhs);
        }
        let result = unsafe { ffi::PyNumber_Add(a, b) };
        unsafe {
            ffi::Py_DECREF(a);
            ffi::Py_DECREF(b);
        }
        if result.is_null() {
            let err = pyo3::PyErr::fetch(py);
            if cfg!(debug_assertions) {
                eprintln!("python addition failed: {}", err);
            }
            panic!("python addition failed: {}", err);
        }
        Value::from_pyobject(result)
    }
    fn py_sub(&self, rhs: &Self) -> Self {
        let py = unsafe { Python::assume_gil_acquired() };
        let a = self.to_pyobject_owned(py);
        let b = rhs.to_pyobject_owned(py);
        if cfg!(debug_assertions) {
            eprintln!("py_sub: a={:?}, b={:?}", self, rhs);
        }
        let result = unsafe { ffi::PyNumber_Subtract(a, b) };
        unsafe {
            ffi::Py_DECREF(a);
            ffi::Py_DECREF(b);
        }
        if result.is_null() {
            let err = pyo3::PyErr::fetch(py);
            if cfg!(debug_assertions) {
                eprintln!("python subtraction failed: {}", err);
            }
            panic!("python subtraction failed: {}", err);
        }
        Value::from_pyobject(result)
    }
    fn py_mul(&self, rhs: &Self) -> Self {
        let py = unsafe { Python::assume_gil_acquired() };
        let a = self.to_pyobject_owned(py);
        let b = rhs.to_pyobject_owned(py);
        if cfg!(debug_assertions) {
            eprintln!("py_mul: a={:?}, b={:?}", self, rhs);
        }
        let result = unsafe { ffi::PyNumber_Multiply(a, b) };
        unsafe {
            ffi::Py_DECREF(a);
            ffi::Py_DECREF(b);
        }
        if result.is_null() {
            let err = pyo3::PyErr::fetch(py);
            if cfg!(debug_assertions) {
                eprintln!("python multiplication failed: {}", err);
            }
            panic!("python multiplication failed: {}", err);
        }
        Value::from_pyobject(result)
    }
    fn py_div(&self, rhs: &Self) -> Self {
        let py = unsafe { Python::assume_gil_acquired() };
        let a = self.to_pyobject_owned(py);
        let b = rhs.to_pyobject_owned(py);
        if cfg!(debug_assertions) {
            eprintln!("py_div: a={:?}, b={:?}", self, rhs);
        }
        let result = unsafe { ffi::PyNumber_TrueDivide(a, b) };
        unsafe {
            ffi::Py_DECREF(a);
            ffi::Py_DECREF(b);
        }
        if result.is_null() {
            let err = pyo3::PyErr::fetch(py);
            if cfg!(debug_assertions) {
                eprintln!("python true division failed: {}", err);
            }
            panic!("python true division failed: {}", err);
        }
        Value::from_pyobject(result)
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        if self.is_pyobject() || rhs.is_pyobject() {
            return self.py_add(&rhs);
        }
        if let (Some(a), Some(b)) = (self.to_int(), rhs.to_int()) {
            return Value::from_int(a + b);
        }
        if let (Some(a), Some(b)) = (self.to_f64(), rhs.to_f64()) {
            return Value::from_f64(a + b);
        }
        if let (Some(i), Some(f)) = (self.to_int(), rhs.to_f64()) {
            return Value::from_f64(i as f64 + f);
        }
        if let (Some(f), Some(i)) = (self.to_f64(), rhs.to_int()) {
            return Value::from_f64(f + i as f64);
        }
        panic!("type error in +");
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Self::Output {
        if self.is_pyobject() || rhs.is_pyobject() {
            return self.py_sub(&rhs);
        }
        if let (Some(a), Some(b)) = (self.to_int(), rhs.to_int()) {
            return Value::from_int(a - b);
        }
        if let (Some(a), Some(b)) = (self.to_f64(), rhs.to_f64()) {
            return Value::from_f64(a - b);
        }
        if let (Some(i), Some(f)) = (self.to_int(), rhs.to_f64()) {
            return Value::from_f64(i as f64 - f);
        }
        if let (Some(f), Some(i)) = (self.to_f64(), rhs.to_int()) {
            return Value::from_f64(f - i as f64);
        }
        panic!("type error in -");
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_pyobject() || rhs.is_pyobject() {
            return self.py_mul(&rhs);
        }
        if let (Some(a), Some(b)) = (self.to_int(), rhs.to_int()) {
            return Value::from_int(a * b);
        }
        if let (Some(a), Some(b)) = (self.to_f64(), rhs.to_f64()) {
            return Value::from_f64(a * b);
        }
        if let (Some(i), Some(f)) = (self.to_int(), rhs.to_f64()) {
            return Value::from_f64(i as f64 * f);
        }
        if let (Some(f), Some(i)) = (self.to_f64(), rhs.to_int()) {
            return Value::from_f64(f * i as f64);
        }
        panic!("type error in *");
    }
}

impl Div for Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Self::Output {
        if self.is_pyobject() || rhs.is_pyobject() {
            return self.py_div(&rhs);
        }
        if let (Some(a), Some(b)) = (self.to_int(), rhs.to_int()) {
            if b == 0 {
                panic!("division by zero");
            }
            return Value::from_int(a / b); // integer division
        }
        let a_float = self.to_f64().or_else(|| self.to_int().map(|i| i as f64));
        let b_float = rhs.to_f64().or_else(|| rhs.to_int().map(|i| i as f64));
        match (a_float, b_float) {
            (Some(a), Some(b)) => {
                if b == 0.0 {
                    panic!("division by zero");
                }
                Value::from_f64(a / b)
            }
            _ => panic!("type error in /"),
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_nil() {
            write!(f, "nil")
        } else if let Some(b) = self.to_bool() {
            write!(f, "{}", b)
        } else if let Some(i) = self.to_int() {
            write!(f, "{}", i)
        } else if let Some(x) = self.to_f64() {
            write!(f, "{}", x)
        } else if let Some(s) = self.to_string_from_arena() {
            write!(f, "{}", s)
        } else if self.is_pyobject() {
            write!(f, "<pyobject@{:#x}>", self.0 & PAYLOAD_MASK)
        } else {
            write!(f, "value({:#x})", self.0)
        }
    }
}