use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi::{PyCapsule_New, PyCapsule_GetPointer};
use crate::tensor::PyTensor;
use std::os::raw::c_void;

const DL_CPU_DEVICE_TYPE: i32 = 1;

#[pymethods]
impl PyTensor {
    pub fn __dlpack__<'py>(&self, py: Python<'py>, _stream: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        // Request the raw DLManagedTensor representation from mohu-buffer / mohu-core primitives
        let raw_dl_struct = Box::into_raw(Box::new(self.inner.to_dlpack_struct()));
        
        unsafe {
            let name = b"dltensor\0".as_ptr() as *const i8;
            let capsule = PyCapsule_New(raw_dl_struct as *mut c_void, name, Some(dlpack_capsule_deleter));
            if capsule.is_null() {
                return Err(PyRuntimeError::new_err("Failed to construct DLPack capsule context"));
            }
            Ok(Bound::from_owned_ptr(py, capsule))
        }
    }

    pub fn __dlpack_device__(&self) -> (i32, i32) {
        // Return protocol device mapping tuple: (device_type, device_id)
        (DL_CPU_DEVICE_TYPE, 0)
    }
}

unsafe extern "C" fn dlpack_capsule_deleter(capsule: *mut pyo3::ffi::PyObject) {
    let name = b"dltensor\0".as_ptr() as *const i8;
    let raw_ptr = PyCapsule_GetPointer(capsule, name);
    if !raw_ptr.is_null() {
        // Safe context recovery of structural resource allocation memory context blocks
        let _ = Box::from_raw(raw_ptr as *mut mohu_core::DLManagedTensor);
    }
}
