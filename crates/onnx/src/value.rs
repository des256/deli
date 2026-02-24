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

impl sealed::Sealed for bool {}
impl TensorElement for bool {
    fn element_type() -> ffi::ONNXTensorElementDataType {
        ffi::ONNXTensorElementDataType::Bool
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

    /// Create an empty tensor with a specified element type.
    ///
    /// Useful for creating KV cache tensors where a dimension is 0 (no cached
    /// keys/values yet) and the element type may not have a Rust equivalent
    /// (e.g., float16).
    ///
    /// # Errors
    /// Returns an error if tensor creation fails
    pub fn empty_typed(shape: &[usize], element_type: ffi::ONNXTensorElementDataType) -> Result<Self> {
        unsafe {
            let api = get_api();

            let buffer = vec![0u8; 0].into_boxed_slice();

            let mut memory_info: *mut ffi::OrtMemoryInfo = std::ptr::null_mut();
            let create_memory_info: ffi::CreateCpuMemoryInfoFn =
                (*api).get_fn(ffi::IDX_CREATE_CPU_MEMORY_INFO);
            let status = create_memory_info(
                ffi::OrtAllocatorType::Device,
                ffi::OrtMemType::CpuOutput,
                &mut memory_info as *mut _,
            );
            error::check_status(api, status, ())?;

            let shape_i64: Vec<i64> = shape.iter().map(|&s| s as i64).collect();

            let mut value: *mut ffi::OrtValue = std::ptr::null_mut();
            let create_tensor: ffi::CreateTensorWithDataAsOrtValueFn =
                (*api).get_fn(ffi::IDX_CREATE_TENSOR_WITH_DATA_AS_ORT_VALUE);
            let status = create_tensor(
                memory_info,
                buffer.as_ptr() as *mut std::ffi::c_void,
                0,
                shape_i64.as_ptr(),
                shape_i64.len(),
                element_type,
                &mut value as *mut _,
            );

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

    /// Create a zero-filled tensor with the given shape
    ///
    /// Dimensions of -1 (dynamic) are replaced with 1.
    ///
    /// # Errors
    /// Returns an error if tensor creation fails
    pub fn zeros<T: TensorElement + Default>(shape: &[i64]) -> Result<Self> {
        let resolved: Vec<usize> = shape.iter().map(|&d| if d < 0 { 1 } else { d as usize }).collect();
        let total: usize = resolved.iter().product();
        let data = vec![T::default(); total];
        Self::from_slice(&resolved, &data)
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

            // Handle empty tensors (elem_count = 0) - avoid null pointer in from_raw_parts
            if elem_count == 0 {
                return Ok(&[]);
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

    /// Extract tensor data as f32, converting from f16 if needed.
    ///
    /// Handles both f32 (zero-copy) and f16 (converted to f32) tensors.
    /// Returns owned Vec for f16 data and a wrapper for f32 data.
    ///
    /// # Errors
    /// Returns an error if the tensor is not f32 or f16, or data extraction fails
    pub fn extract_as_f32(&self) -> Result<Vec<f32>> {
        let elem_type = self.tensor_element_type()?;
        match elem_type {
            ffi::ONNXTensorElementDataType::Float => {
                let data = self.extract_tensor::<f32>()?;
                Ok(data.to_vec())
            }
            ffi::ONNXTensorElementDataType::Float16 => {
                unsafe {
                    let mut type_info: *mut ffi::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
                    let get_type_shape: ffi::GetTensorTypeAndShapeFn =
                        (*self.api).get_fn(ffi::IDX_GET_TENSOR_TYPE_AND_SHAPE);
                    let status = get_type_shape(self.value, &mut type_info as *mut _);
                    error::check_status(self.api, status, ())?;

                    let mut elem_count: usize = 0;
                    let get_count: ffi::GetTensorShapeElementCountFn =
                        (*self.api).get_fn(ffi::IDX_GET_TENSOR_SHAPE_ELEMENT_COUNT);
                    let status = get_count(type_info, &mut elem_count as *mut _);

                    let release_type_info: ffi::ReleaseTensorTypeAndShapeInfoFn =
                        (*self.api).get_fn(ffi::IDX_RELEASE_TENSOR_TYPE_AND_SHAPE_INFO);
                    release_type_info(type_info);
                    error::check_status(self.api, status, ())?;

                    if elem_count == 0 {
                        return Ok(Vec::new());
                    }

                    let mut data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
                    let get_data: ffi::GetTensorMutableDataFn =
                        (*self.api).get_fn(ffi::IDX_GET_TENSOR_MUTABLE_DATA);
                    let status = get_data(self.value, &mut data_ptr as *mut _);
                    error::check_status(self.api, status, ())?;

                    let f16_data = std::slice::from_raw_parts(data_ptr as *const u16, elem_count);
                    Ok(f16_data.iter().map(|&h| f16_to_f32(h)).collect())
                }
            }
            other => Err(error::OnnxError::runtime_error(
                &format!("extract_as_f32: unsupported element type {:?}", other),
            )),
        }
    }

    /// Get the tensor shape as a Vec of dimensions.
    ///
    /// # Errors
    /// Returns an error if the value is not a tensor or shape extraction fails
    pub fn tensor_shape(&self) -> Result<Vec<i64>> {
        unsafe {
            let mut type_info: *mut ffi::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
            let get_type_shape: ffi::GetTensorTypeAndShapeFn =
                (*self.api).get_fn(ffi::IDX_GET_TENSOR_TYPE_AND_SHAPE);
            let status = get_type_shape(self.value, &mut type_info as *mut _);
            error::check_status(self.api, status, ())?;

            let mut dim_count: usize = 0;
            let get_dims_count: ffi::GetDimensionsCountFn =
                (*self.api).get_fn(ffi::IDX_GET_DIMENSIONS_COUNT);
            let status = get_dims_count(type_info, &mut dim_count as *mut _);
            if !status.is_null() {
                let release: ffi::ReleaseTensorTypeAndShapeInfoFn =
                    (*self.api).get_fn(ffi::IDX_RELEASE_TENSOR_TYPE_AND_SHAPE_INFO);
                release(type_info);
                return Err(error::OnnxError::from_status(self.api, status));
            }

            let mut dims = vec![0i64; dim_count];
            let get_dims: ffi::GetDimensionsFn =
                (*self.api).get_fn(ffi::IDX_GET_DIMENSIONS);
            let status = get_dims(type_info, dims.as_mut_ptr(), dim_count);

            let release: ffi::ReleaseTensorTypeAndShapeInfoFn =
                (*self.api).get_fn(ffi::IDX_RELEASE_TENSOR_TYPE_AND_SHAPE_INFO);
            release(type_info);

            error::check_status(self.api, status, ())?;
            Ok(dims)
        }
    }

    /// Get the tensor element type
    ///
    /// # Errors
    /// Returns an error if the value is not a tensor or type extraction fails
    pub fn tensor_element_type(&self) -> Result<ffi::ONNXTensorElementDataType> {
        unsafe {
            let mut type_info: *mut ffi::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
            let get_type_shape: ffi::GetTensorTypeAndShapeFn =
                (*self.api).get_fn(ffi::IDX_GET_TENSOR_TYPE_AND_SHAPE);
            let status = get_type_shape(self.value, &mut type_info as *mut _);
            error::check_status(self.api, status, ())?;

            let mut element_type = ffi::ONNXTensorElementDataType::Undefined;
            let get_element_type: ffi::GetTensorElementTypeFn =
                (*self.api).get_fn(ffi::IDX_GET_TENSOR_ELEMENT_TYPE);
            let status = get_element_type(type_info, &mut element_type as *mut _);

            let release: ffi::ReleaseTensorTypeAndShapeInfoFn =
                (*self.api).get_fn(ffi::IDX_RELEASE_TENSOR_TYPE_AND_SHAPE_INFO);
            release(type_info);

            error::check_status(self.api, status, ())?;
            Ok(element_type)
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

/// Convert IEEE 754 half-precision (f16) to single-precision (f32).
fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exponent = ((half >> 10) & 0x1f) as u32;
    let mantissa = (half & 0x3ff) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // ±0
            f32::from_bits(sign << 31)
        } else {
            // Denormalized: shift mantissa until hidden bit appears
            let mut e = 0i32;
            let mut m = mantissa;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let f32_exp = (127 - 15 + 1 + e) as u32;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exponent == 31 {
        // Infinity or NaN
        f32::from_bits((sign << 31) | (0xff << 23) | (mantissa << 13))
    } else {
        // Normalized
        let f32_exp = (exponent as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
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
        let _ = crate::init();
        let val = Value::from_slice::<bool>(&[1], &[true]).unwrap();
        let extracted = val.extract_tensor::<bool>().unwrap();
        assert_eq!(extracted, &[true]);
    }

    #[test]
    fn test_bool_zeros() {
        let _ = crate::init();
        let val = Value::zeros::<bool>(&[1]).unwrap();
        let extracted = val.extract_tensor::<bool>().unwrap();
        assert_eq!(extracted, &[false]);
    }

    #[test]
    fn test_bool_zeros_multi_dim() {
        let _ = crate::init();
        let val = Value::zeros::<bool>(&[1, 10]).unwrap();
        let extracted = val.extract_tensor::<bool>().unwrap();
        assert_eq!(extracted.len(), 10);
        assert!(extracted.iter().all(|&b| b == false));
    }

    #[test]
    fn test_empty_tensor() {
        let _ = crate::init();
        // Test empty tensor with shape [0] - needed for flow_lm current_end states
        let val = Value::from_slice::<f32>(&[0], &[]).unwrap();
        let shape = val.tensor_shape().unwrap();
        assert_eq!(shape, vec![0]);
        let extracted = val.extract_tensor::<f32>().unwrap();
        assert_eq!(extracted.len(), 0);
    }
}
