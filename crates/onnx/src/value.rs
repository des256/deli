use {crate::*, std::sync::Arc};

/// Sealed trait for types that can be used as tensor elements
mod sealed {
    pub trait Sealed {}
}

/// Types that can be used as tensor elements
pub trait TensorElement: sealed::Sealed + Sized + Copy {
    fn element_type() -> ffi::ONNXTensorElementDataType;
}

impl sealed::Sealed for f32 {}
impl TensorElement for f32 {
    fn element_type() -> ffi::ONNXTensorElementDataType {
        ffi::ONNXTensorElementDataType::Float
    }
}

impl sealed::Sealed for f64 {}
impl TensorElement for f64 {
    fn element_type() -> ffi::ONNXTensorElementDataType {
        ffi::ONNXTensorElementDataType::Double
    }
}

impl sealed::Sealed for i64 {}
impl TensorElement for i64 {
    fn element_type() -> ffi::ONNXTensorElementDataType {
        ffi::ONNXTensorElementDataType::Int64
    }
}

impl sealed::Sealed for i32 {}
impl TensorElement for i32 {
    fn element_type() -> ffi::ONNXTensorElementDataType {
        ffi::ONNXTensorElementDataType::Int32
    }
}

impl sealed::Sealed for bool {}
impl TensorElement for bool {
    fn element_type() -> ffi::ONNXTensorElementDataType {
        ffi::ONNXTensorElementDataType::Bool
    }
}

/// ONNX Runtime value (tensor)
pub struct Value {
    onnx: Arc<Onnx>,
    value: *mut ffi::OrtValue,
    // Own the data buffer to ensure it lives as long as the Value
    _data: Box<[u8]>,
}

unsafe impl Send for Value {}

impl Value {
    pub fn from_slice<T: TensorElement>(
        onnx: &Arc<Onnx>,
        shape: &[usize],
        data: &[T],
    ) -> Result<Self, OnnxError> {
        // Verify data length matches shape
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(OnnxError::runtime_error(&format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_len
            )));
        }

        // Copy data into owned buffer
        let byte_len = data.len() * std::mem::size_of::<T>();
        let mut buffer = vec![0u8; byte_len].into_boxed_slice();
        let src_ptr = data.as_ptr() as *const u8;
        unsafe { std::ptr::copy_nonoverlapping(src_ptr, buffer.as_mut_ptr(), byte_len) };

        // Create memory info for CPU
        let mut memory_info: *mut ffi::OrtMemoryInfo = std::ptr::null_mut();
        let status = unsafe {
            (onnx.create_memory_info)(
                ffi::OrtAllocatorType::Device,
                ffi::OrtMemType::CpuOutput,
                &mut memory_info as *mut _,
            )
        };
        if !status.is_null() {
            return Err(OnnxError::from_status(onnx.api, status));
        }

        // Convert shape to i64
        let shape_i64: Vec<i64> = shape.iter().map(|&s| s as i64).collect();

        // Create tensor
        let mut value: *mut ffi::OrtValue = std::ptr::null_mut();
        let status = unsafe {
            (onnx.create_tensor)(
                memory_info,
                buffer.as_mut_ptr() as *mut std::ffi::c_void,
                byte_len,
                shape_i64.as_ptr(),
                shape_i64.len(),
                T::element_type(),
                &mut value as *mut _,
            )
        };
        if !status.is_null() {
            unsafe { (onnx.release_memory_info)(memory_info) };
            return Err(OnnxError::from_status(onnx.api, status));
        }

        // Release memory info
        unsafe { (onnx.release_memory_info)(memory_info) };

        Ok(Value {
            onnx: Arc::clone(&onnx),
            value,
            _data: buffer,
        })
    }

    pub fn empty_typed(
        onnx: &Arc<Onnx>,
        shape: &[usize],
        element_type: ffi::ONNXTensorElementDataType,
    ) -> Result<Self, OnnxError> {
        let buffer = vec![0u8; 0].into_boxed_slice();

        let mut memory_info: *mut ffi::OrtMemoryInfo = std::ptr::null_mut();
        let status = unsafe {
            (onnx.create_memory_info)(
                ffi::OrtAllocatorType::Device,
                ffi::OrtMemType::CpuOutput,
                &mut memory_info as *mut _,
            )
        };
        if !status.is_null() {
            return Err(OnnxError::from_status(onnx.api, status));
        }

        let shape_i64: Vec<i64> = shape.iter().map(|&s| s as i64).collect();

        let mut value: *mut ffi::OrtValue = std::ptr::null_mut();
        let status = unsafe {
            (onnx.create_tensor)(
                memory_info,
                buffer.as_ptr() as *mut std::ffi::c_void,
                0,
                shape_i64.as_ptr(),
                shape_i64.len(),
                element_type,
                &mut value as *mut _,
            )
        };
        if !status.is_null() {
            unsafe { (onnx.release_memory_info)(memory_info) };
            return Err(OnnxError::from_status(onnx.api, status));
        }

        unsafe { (onnx.release_memory_info)(memory_info) };

        Ok(Value {
            onnx: Arc::clone(&onnx),
            value,
            _data: buffer,
        })
    }

    pub fn zeros<T: TensorElement + Default>(
        onnx: &Arc<Onnx>,
        shape: &[i64],
    ) -> Result<Self, OnnxError> {
        let resolved: Vec<usize> = shape
            .iter()
            .map(|&d| if d < 0 { 1 } else { d as usize })
            .collect();
        let total: usize = resolved.iter().product();
        let data = vec![T::default(); total];
        Self::from_slice(onnx, &resolved, &data)
    }

    pub fn extract_tensor<T: TensorElement>(&self) -> Result<&[T], OnnxError> {
        // Get tensor type and shape info
        let mut type_info: *mut ffi::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status =
            unsafe { (self.onnx.get_tensor_type_and_shape)(self.value, &mut type_info as *mut _) };
        if !status.is_null() {
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        // Verify element type matches
        let mut element_type = ffi::ONNXTensorElementDataType::Undefined;
        let status =
            unsafe { (self.onnx.get_tensor_element_type)(type_info, &mut element_type as *mut _) };
        if !status.is_null() {
            unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        // Get total element count from shape info
        let mut elem_count: usize = 0;
        let status = unsafe {
            (self.onnx.get_tensor_shape_element_count)(type_info, &mut elem_count as *mut _)
        };
        if !status.is_null() {
            unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        // Release type info
        unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };

        if element_type != T::element_type() {
            return Err(OnnxError::runtime_error(&format!(
                "Element type mismatch: expected {:?}, got {:?}",
                T::element_type(),
                element_type
            )));
        }

        // Handle empty tensors (elem_count = 0) - avoid null pointer in from_raw_parts
        if elem_count == 0 {
            return Ok(&[]);
        }

        // Get mutable data pointer
        let mut data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let status =
            unsafe { (self.onnx.get_tensor_mutable_data)(self.value, &mut data_ptr as *mut _) };
        if !status.is_null() {
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        Ok(unsafe { std::slice::from_raw_parts(data_ptr as *const T, elem_count) })
    }

    pub fn extract_as_f32(&self) -> Result<Vec<f32>, OnnxError> {
        let elem_type = self.tensor_element_type()?;
        match elem_type {
            ffi::ONNXTensorElementDataType::Float => {
                let data = self.extract_tensor::<f32>()?;
                Ok(data.to_vec())
            }
            ffi::ONNXTensorElementDataType::Float16 => {
                let mut type_info: *mut ffi::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
                let status = unsafe {
                    (self.onnx.get_tensor_type_and_shape)(self.value, &mut type_info as *mut _)
                };
                if !status.is_null() {
                    return Err(OnnxError::from_status(self.onnx.api, status));
                }

                let mut elem_count: usize = 0;
                let status = unsafe {
                    (self.onnx.get_tensor_shape_element_count)(type_info, &mut elem_count as *mut _)
                };
                if !status.is_null() {
                    unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
                    return Err(OnnxError::from_status(self.onnx.api, status));
                }

                unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };

                if elem_count == 0 {
                    return Ok(Vec::new());
                }

                let mut data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
                let status = unsafe {
                    (self.onnx.get_tensor_mutable_data)(self.value, &mut data_ptr as *mut _)
                };
                if !status.is_null() {
                    return Err(OnnxError::from_status(self.onnx.api, status));
                }

                let f16_data =
                    unsafe { std::slice::from_raw_parts(data_ptr as *const u16, elem_count) };
                Ok(f16_data.iter().map(|&h| f16::f16_to_f32(h)).collect())
            }
            other => Err(OnnxError::runtime_error(&format!(
                "extract_as_f32: unsupported element type {:?}",
                other
            ))),
        }
    }

    pub fn tensor_shape(&self) -> Result<Vec<i64>, OnnxError> {
        let mut type_info: *mut ffi::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status =
            unsafe { (self.onnx.get_tensor_type_and_shape)(self.value, &mut type_info as *mut _) };
        if !status.is_null() {
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        let mut dim_count: usize = 0;
        let status =
            unsafe { (self.onnx.get_dimensions_count)(type_info, &mut dim_count as *mut _) };
        if !status.is_null() {
            unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        let mut dims = vec![0i64; dim_count];
        let status = unsafe { (self.onnx.get_dimensions)(type_info, dims.as_mut_ptr(), dim_count) };
        if !status.is_null() {
            unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };

        Ok(dims)
    }

    pub fn tensor_element_type(&self) -> Result<ffi::ONNXTensorElementDataType, OnnxError> {
        let mut type_info: *mut ffi::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status =
            unsafe { (self.onnx.get_tensor_type_and_shape)(self.value, &mut type_info as *mut _) };
        if !status.is_null() {
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        let mut element_type = ffi::ONNXTensorElementDataType::Undefined;
        let status =
            unsafe { (self.onnx.get_tensor_element_type)(type_info, &mut element_type as *mut _) };
        if !status.is_null() {
            unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        unsafe { (self.onnx.release_tensor_type_and_shape_info)(type_info) };

        Ok(element_type)
    }

    pub(crate) fn as_ptr(&self) -> *const ffi::OrtValue {
        self.value
    }

    pub(crate) unsafe fn from_raw(onnx: &Arc<Onnx>, value: *mut ffi::OrtValue) -> Self {
        Value {
            onnx: Arc::clone(&onnx),
            value,
            _data: Box::new([]),
        }
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        if !self.value.is_null() {
            unsafe { (self.onnx.release_value)(self.value) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Value>();
    }

    #[test]
    fn test_tensor_element_types() {
        assert_eq!(f32::element_type(), ffi::ONNXTensorElementDataType::Float);
        assert_eq!(f64::element_type(), ffi::ONNXTensorElementDataType::Double);
        assert_eq!(i64::element_type(), ffi::ONNXTensorElementDataType::Int64);
        assert_eq!(i32::element_type(), ffi::ONNXTensorElementDataType::Int32);
        assert_eq!(bool::element_type(), ffi::ONNXTensorElementDataType::Bool);
    }

    #[test]
    fn test_bool_from_slice() {
        let onnx = Onnx::new(17).unwrap();
        let val = Value::from_slice::<bool>(&onnx, &[1], &[true]).unwrap();
        let extracted = val.extract_tensor::<bool>().unwrap();
        assert_eq!(extracted, &[true]);
    }

    #[test]
    fn test_bool_zeros() {
        let onnx = Onnx::new(17).unwrap();
        let val = Value::zeros::<bool>(&onnx, &[1]).unwrap();
        let extracted = val.extract_tensor::<bool>().unwrap();
        assert_eq!(extracted, &[false]);
    }

    #[test]
    fn test_bool_zeros_multi_dim() {
        let onnx = Onnx::new(17).unwrap();
        let val = Value::zeros::<bool>(&onnx, &[1, 10]).unwrap();
        let extracted = val.extract_tensor::<bool>().unwrap();
        assert_eq!(extracted.len(), 10);
        assert!(extracted.iter().all(|&b| b == false));
    }

    #[test]
    fn test_empty_tensor() {
        let onnx = Onnx::new(17).unwrap();
        // Test empty tensor with shape [0] - needed for flow_lm current_end states
        let val = Value::from_slice::<f32>(&onnx, &[0], &[]).unwrap();
        let shape = val.tensor_shape().unwrap();
        assert_eq!(shape, vec![0]);
        let extracted = val.extract_tensor::<f32>().unwrap();
        assert_eq!(extracted.len(), 0);
    }
}
