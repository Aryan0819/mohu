use pyo3::prelude::*;
use pyo3::ffi::{Py_buffer, PyBUF_WRITABLE};
use pyo3::exceptions::{PyValueError, PyTypeError};
use mohu_core::{Tensor, DType};
use std::os::raw::{c_void, c_int};

#[pyclass(name = "Tensor", subclass)]
pub struct PyTensor {
    pub inner: Tensor,
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(shape: Vec<usize>, dtype_str: &str) -> PyResult<Self> {
        let dtype = match dtype_str {
            "float32" => DType::F32,
            "float64" => DType::F64,
            "int32" => DType::I32,
            _ => return Err(PyValueError::new_err(format!("Unsupported dtype: {}", dtype_str))),
        };
        
        let inner = Tensor::empty(shape, dtype);
        Ok(PyTensor { inner })
    }

    @getter
    fn dtype_str(&self) -> &'static str {
        match self.inner.dtype() {
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::I32 => "int32",
            _ => "unknown",
        }
    }

    @getter
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    // ── Python Buffer Protocol (PEP 3118) Implementation ──
    unsafe fn __getbuffer__(
        &self,
        view: *mut Py_buffer,
        _flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyValueError::new_err("Buffer view structure pointer is null"));
        }

        let item_size = self.inner.dtype().size_in_bytes() as isize;
        let total_bytes = (self.inner.len() * self.inner.dtype().size_in_bytes()) as isize;

        (*view).buf = self.inner.as_ptr() as *mut c_void;
        (*view).obj = std::ptr::null_mut(); 
        (*view).len = total_bytes;
        (*view).itemsize = item_size;
        (*view).readonly = 0; 
        
        (*view).format = match self.inner.dtype() {
            DType::F32 => b"f\0".as_ptr() as *mut i8,
            DType::F64 => b"d\0".as_ptr() as *mut i8,
            DType::I32 => b"i\0".as_ptr() as *mut i8,
            _ => b"B\0".as_ptr() as *mut i8,
        };

        (*view).ndim = self.inner.ndim() as i32;
        (*view).shape = self.inner.shape().as_ptr() as *mut isize;
        (*view).strides = self.inner.strides().as_ptr() as *mut isize;
        (*view).suboffsets = std::ptr::null_mut();
        (*view).internal = std::ptr::null_mut();

        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut Py_buffer) {
        // Shared reference cleanup is handled implicitly through Rust dropped object cycles
    }
}
