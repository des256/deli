use crate::{error::{self, Result}, ffi, get_api};

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

/// ONNX Runtime value (tensor)
pub struct Value {
    value: *mut ffi::OrtValue,
    api: *const ffi::OrtApi,
    // Own the data buffer to ensure it lives as long as the Value
    _data: Box<[u8]>,
}

unsafe impl Send for Value {}

impl Value {
    /// Create a tensor from a slice of data
    ///
    /// # Errors
    /// Returns an error if tensor creation fails
    pub fn from_slice<T: TensorElement>(shape: &[usize], data: &[T]) -> Result<Self> {
        unsafe {
            let api = get_api();

            // Verify data length matches shape
            let expected_len: usize = shape.iter().product();
            if data.len() != expected_len {
                return Err(error::OnnxError::runtime_error(
                    &format!("Data length {} doesn't match shape {:?} (expected {})",
                             data.len(), shape, expected_len)
                ));
            }

            // Copy data into owned buffer
            let byte_len = data.len() * std::mem::size_of::<T>();
            let mut buffer = vec![0u8; byte_len].into_boxed_slice();
            let src_ptr = data.as_ptr() as *const u8;
            std::ptr::copy_nonoverlapping(src_ptr, buffer.as_mut_ptr(), byte_len);

            // Create memory info for CPU
            let mut memory_info: *mut ffi::OrtMemoryInfo = std::ptr::null_mut();
            let create_memory_info: ffi::CreateCpuMemoryInfoFn =
                (*api).get_fn(ffi::IDX_CREATE_CPU_MEMORY_INFO);
            let status = create_memory_info(
                ffi::OrtAllocatorType::Device,
                ffi::OrtMemType::CpuOutput,
                &mut memory_info as *mut _,
            );
            error::check_status(api, status, ())?;

            // Convert shape to i64
            let shape_i64: Vec<i64> = shape.iter().map(|&s| s as i64).collect();

            // Create tensor
            let mut value: *mut ffi::OrtValue = std::ptr::null_mut();
            let create_tensor: ffi::CreateTensorWithDataAsOrtValueFn =
                (*api).get_fn(ffi::IDX_CREATE_TENSOR_WITH_DATA_AS_ORT_VALUE);
            let status = create_tensor(
                memory_info,
                buffer.as_mut_ptr() as *mut std::ffi::c_void,
                byte_len,
                shape_i64.as_ptr(),
                shape_i64.len(),
                T::element_type(),
                &mut value as *mut _,
            );

            // Release memory info
            let release_memory_info: ffi::ReleaseMemoryInfoFn =
                (*api).get_fn(ffi::IDX_RELEASE_MEMORY_INFO);
            release_memory_info(memory_info);

            error::check_status(api, status, ())?;

            Ok(Value {
                value,
                api,
                _data: buffer,
            })
        }
    }

    /// Extract tensor data as a slice
    ///
    /// # Errors
    /// Returns an error if the element type doesn't match or data extraction fails
    pub fn extract_tensor<T: TensorElement>(&self) -> Result<&[T]> {
        unsafe {
            // Get tensor type and shape info
            let mut type_info: *mut ffi::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
            let get_type_shape: ffi::GetTensorTypeAndShapeFn =
                (*self.api).get_fn(ffi::IDX_GET_TENSOR_TYPE_AND_SHAPE);
            let status = get_type_shape(self.value, &mut type_info as *mut _);
            error::check_status(self.api, status, ())?;

            // Verify element type matches
            let mut element_type = ffi::ONNXTensorElementDataType::Undefined;
            let get_element_type: ffi::GetTensorElementTypeFn =
                (*self.api).get_fn(ffi::IDX_GET_TENSOR_ELEMENT_TYPE);
            let status = get_element_type(type_info, &mut element_type as *mut _);
            error::check_status(self.api, status, ())?;

            // Get total element count from shape info
            let mut elem_count: usize = 0;
            let get_count: ffi::GetTensorShapeElementCountFn =
                (*self.api).get_fn(ffi::IDX_GET_TENSOR_SHAPE_ELEMENT_COUNT);
            let status = get_count(type_info, &mut elem_count as *mut _);

            // Release type info
            let release_type_info: ffi::ReleaseTensorTypeAndShapeInfoFn =
                (*self.api).get_fn(ffi::IDX_RELEASE_TENSOR_TYPE_AND_SHAPE_INFO);
            release_type_info(type_info);

            error::check_status(self.api, status, ())?;

            if element_type != T::element_type() {
                return Err(error::OnnxError::runtime_error(
                    &format!("Element type mismatch: expected {:?}, got {:?}",
                             T::element_type(), element_type)
                ));
            }

            // Get mutable data pointer
            let mut data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let get_data: ffi::GetTensorMutableDataFn =
                (*self.api).get_fn(ffi::IDX_GET_TENSOR_MUTABLE_DATA);
            let status = get_data(self.value, &mut data_ptr as *mut _);
            error::check_status(self.api, status, ())?;

            Ok(std::slice::from_raw_parts(data_ptr as *const T, elem_count))
        }
    }

    /// Get the raw OrtValue pointer (for internal use)
    pub(crate) fn as_ptr(&self) -> *const ffi::OrtValue {
        self.value
    }

    /// Create a Value from a raw OrtValue pointer (for outputs)
    ///
    /// # Safety
    /// - value must be a valid OrtValue pointer
    /// - api must be a valid OrtApi pointer
    pub(crate) unsafe fn from_raw(value: *mut ffi::OrtValue, api: *const ffi::OrtApi) -> Self {
        // For output values, we don't own the data buffer - it's managed by OrtValue
        // Use an empty buffer as a placeholder
        Value {
            value,
            api,
            _data: Box::new([]),
        }
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        unsafe {
            if !self.value.is_null() {
                let release_value: ffi::ReleaseValueFn =
                    (*self.api).get_fn(ffi::IDX_RELEASE_VALUE);
                release_value(self.value);
            }
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
    }
}
