use pyo3::prelude::*;

mod tensor;
mod dlpack;

// Force jemalloc on Unix environments to replace Python's allocator for Rust memory
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Raw PyO3 extension module hook.
#[pymodule]
fn _mohu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tensor::PyTensor>()?;
    Ok(())
}
