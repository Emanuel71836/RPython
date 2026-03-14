use pyo3::prelude::*;

/// used to define functions - currently empty
#[pyfunction]
#[pyo3(name = "function_define")]
pub fn function_define(func_name: String) -> PyResult<String> {
    if cfg!(debug_assertions) {
    println!("Received code: {}", func_name);
    Ok(func_name)
}