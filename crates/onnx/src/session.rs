use crate::{error::{self, Result}, ffi, get_api, get_env};
use std::{ffi::CString, path::Path};

/// Session builder for configuring ONNX sessions
pub struct SessionBuilder {
    options: *mut ffi::OrtSessionOptions,
    api: *const ffi::OrtApi,
}

impl SessionBuilder {
    /// Create a new session builder
    ///
    /// # Safety
    /// Must call init() first
    pub(crate) unsafe fn new() -> Result<Self> {
        unsafe {
            let api = get_api();
            let mut options: *mut ffi::OrtSessionOptions = std::ptr::null_mut();

            let create_options: ffi::CreateSessionOptionsFn = (*api).get_fn(ffi::IDX_CREATE_SESSION_OPTIONS);
            let status = create_options(&mut options as *mut _);

            error::check_status(api, status, ())?;

            Ok(SessionBuilder { options, api })
        }
    }

    /// Configure to use CPU execution provider
    pub fn with_cpu(self) -> Self {
        // CPU is the default, no configuration needed
        self
    }

    /// Configure to use CUDA execution provider
    #[cfg(feature = "cuda")]
    pub fn with_cuda(self, device_id: i32) -> Result<Self> {
        unsafe {
            let status = ffi::OrtSessionOptionsAppendExecutionProvider_CUDA(self.options, device_id);
            error::check_status(self.api, status, ())?;
        }
        Ok(self)
    }

    /// Set graph optimization level
    pub fn with_optimization_level(self, level: ffi::GraphOptimizationLevel) -> Result<Self> {
        unsafe {
            let set_opt_level: ffi::SetSessionGraphOptimizationLevelFn =
                (*self.api).get_fn(ffi::IDX_SET_SESSION_GRAPH_OPTIMIZATION_LEVEL);
            let status = set_opt_level(self.options, level);
            error::check_status(self.api, status, ())?;
        }
        Ok(self)
    }

    /// Set number of intra-op threads
    pub fn with_intra_threads(self, num_threads: i32) -> Result<Self> {
        unsafe {
            let set_threads: ffi::SetIntraOpNumThreadsFn =
                (*self.api).get_fn(ffi::IDX_SET_INTRA_OP_NUM_THREADS);
            let status = set_threads(self.options, num_threads);
            error::check_status(self.api, status, ())?;
        }
        Ok(self)
    }

    /// Create a session from an ONNX model file
    pub fn commit_from_file<P>(self, model_path: P) -> Result<Session>
    where
        P: AsRef<Path>,
    {
        unsafe {
            let path = model_path.as_ref();
            let path_str = path.to_str()
                .ok_or_else(|| error::OnnxError::runtime_error("Invalid UTF-8 in model path"))?;
            let c_path = CString::new(path_str)
                .map_err(|_| error::OnnxError::runtime_error("Null byte in model path"))?;

            let env = get_env();
            let mut session: *mut ffi::OrtSession = std::ptr::null_mut();
            let api = self.api;
            let options = self.options;

            let create_session: ffi::CreateSessionFn = (*api).get_fn(ffi::IDX_CREATE_SESSION);
            let status = create_session(
                env,
                c_path.as_ptr(),
                options,
                &mut session as *mut _,
            );

            // Release options after session creation (whether success or failure)
            let release_options: ffi::ReleaseSessionOptionsFn =
                (*api).get_fn(ffi::IDX_RELEASE_SESSION_OPTIONS);
            release_options(options);
            std::mem::forget(self); // Prevent double-free in Drop

            error::check_status(api, status, ())?;

            Ok(Session { session, api })
        }
    }
}

impl Drop for SessionBuilder {
    fn drop(&mut self) {
        unsafe {
            if !self.options.is_null() {
                let release_options: ffi::ReleaseSessionOptionsFn =
                    (*self.api).get_fn(ffi::IDX_RELEASE_SESSION_OPTIONS);
                release_options(self.options);
            }
        }
    }
}

/// ONNX Runtime session
pub struct Session {
    session: *mut ffi::OrtSession,
    api: *const ffi::OrtApi,
}

// Session is Send because OrtSession is thread-safe with external synchronization
// (which Arc<Mutex<Session>> provides)
unsafe impl Send for Session {}

impl Session {
    /// Get the number of model inputs
    ///
    /// # Errors
    /// Returns an error if the operation fails
    pub fn input_count(&self) -> Result<usize> {
        unsafe {
            let mut count: usize = 0;
            let get_input_count: ffi::SessionGetInputCountFn =
                (*self.api).get_fn(ffi::IDX_SESSION_GET_INPUT_COUNT);
            let status = get_input_count(self.session, &mut count as *mut _);
            error::check_status(self.api, status, count)
        }
    }

    /// Get the number of model outputs
    ///
    /// # Errors
    /// Returns an error if the operation fails
    pub fn output_count(&self) -> Result<usize> {
        unsafe {
            let mut count: usize = 0;
            let get_output_count: ffi::SessionGetOutputCountFn =
                (*self.api).get_fn(ffi::IDX_SESSION_GET_OUTPUT_COUNT);
            let status = get_output_count(self.session, &mut count as *mut _);
            error::check_status(self.api, status, count)
        }
    }

    /// Get the name of an input by index
    ///
    /// # Errors
    /// Returns an error if the index is invalid or the operation fails
    pub fn input_name(&self, index: usize) -> Result<String> {
        unsafe {
            // Get allocator
            let mut allocator: *mut ffi::OrtAllocator = std::ptr::null_mut();
            let get_allocator: ffi::GetAllocatorWithDefaultOptionsFn =
                (*self.api).get_fn(ffi::IDX_GET_ALLOCATOR_WITH_DEFAULT_OPTIONS);
            let status = get_allocator(&mut allocator as *mut _);
            error::check_status(self.api, status, ())?;

            // Get input name
            let mut name_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
            let get_input_name: ffi::SessionGetInputNameFn =
                (*self.api).get_fn(ffi::IDX_SESSION_GET_INPUT_NAME);
            let status = get_input_name(self.session, index, allocator, &mut name_ptr as *mut _);

            if !status.is_null() {
                return Err(error::OnnxError::from_status(self.api, status));
            }

            // Convert C string to Rust String
            let name = std::ffi::CStr::from_ptr(name_ptr)
                .to_str()
                .map_err(|_| error::OnnxError::runtime_error("Invalid UTF-8 in input name"))?
                .to_string();

            // Free the allocated string
            let allocator_free: ffi::AllocatorFreeFn =
                (*self.api).get_fn(ffi::IDX_ALLOCATOR_FREE);
            allocator_free(allocator, name_ptr as *mut std::ffi::c_void);

            Ok(name)
        }
    }

    /// Get the name of an output by index
    ///
    /// # Errors
    /// Returns an error if the index is invalid or the operation fails
    pub fn output_name(&self, index: usize) -> Result<String> {
        unsafe {
            // Get allocator
            let mut allocator: *mut ffi::OrtAllocator = std::ptr::null_mut();
            let get_allocator: ffi::GetAllocatorWithDefaultOptionsFn =
                (*self.api).get_fn(ffi::IDX_GET_ALLOCATOR_WITH_DEFAULT_OPTIONS);
            let status = get_allocator(&mut allocator as *mut _);
            error::check_status(self.api, status, ())?;

            // Get output name
            let mut name_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
            let get_output_name: ffi::SessionGetOutputNameFn =
                (*self.api).get_fn(ffi::IDX_SESSION_GET_OUTPUT_NAME);
            let status = get_output_name(self.session, index, allocator, &mut name_ptr as *mut _);

            if !status.is_null() {
                return Err(error::OnnxError::from_status(self.api, status));
            }

            // Convert C string to Rust String
            let name = std::ffi::CStr::from_ptr(name_ptr)
                .to_str()
                .map_err(|_| error::OnnxError::runtime_error("Invalid UTF-8 in output name"))?
                .to_string();

            // Free the allocated string
            let allocator_free: ffi::AllocatorFreeFn =
                (*self.api).get_fn(ffi::IDX_ALLOCATOR_FREE);
            allocator_free(allocator, name_ptr as *mut std::ffi::c_void);

            Ok(name)
        }
    }

    /// Get the shape of an input tensor by index
    ///
    /// Returns the dimensions as `Vec<i64>`. Dynamic dimensions are represented as `-1`.
    ///
    /// # Errors
    /// Returns an error if the index is invalid or the operation fails
    pub fn input_shape(&self, index: usize) -> Result<Vec<i64>> {
        unsafe {
            // Get input type info
            let mut type_info: *mut ffi::OrtTypeInfo = std::ptr::null_mut();
            let get_input_type_info: ffi::SessionGetInputTypeInfoFn =
                (*self.api).get_fn(ffi::IDX_SESSION_GET_INPUT_TYPE_INFO);
            let status = get_input_type_info(self.session, index, &mut type_info as *mut _);
            if !status.is_null() {
                return Err(error::OnnxError::from_status(self.api, status));
            }

            // Cast OrtTypeInfo to OrtTensorTypeAndShapeInfo
            let mut tensor_info: *const ffi::OrtTensorTypeAndShapeInfo = std::ptr::null();
            let cast_fn: ffi::CastTypeInfoToTensorInfoFn =
                (*self.api).get_fn(ffi::IDX_CAST_TYPE_INFO_TO_TENSOR_INFO);
            let status = cast_fn(type_info, &mut tensor_info as *mut _);
            if !status.is_null() {
                let release_type_info: ffi::ReleaseTypeInfoFn =
                    (*self.api).get_fn(ffi::IDX_RELEASE_TYPE_INFO);
                release_type_info(type_info);
                return Err(error::OnnxError::from_status(self.api, status));
            }

            if tensor_info.is_null() {
                let release_type_info: ffi::ReleaseTypeInfoFn =
                    (*self.api).get_fn(ffi::IDX_RELEASE_TYPE_INFO);
                release_type_info(type_info);
                return Err(error::OnnxError::runtime_error("Input is not a tensor type"));
            }

            // Get dimensions count
            let mut dim_count: usize = 0;
            let get_dims_count: ffi::GetDimensionsCountFn =
                (*self.api).get_fn(ffi::IDX_GET_DIMENSIONS_COUNT);
            let status = get_dims_count(tensor_info, &mut dim_count as *mut _);
            if !status.is_null() {
                let release_type_info: ffi::ReleaseTypeInfoFn =
                    (*self.api).get_fn(ffi::IDX_RELEASE_TYPE_INFO);
                release_type_info(type_info);
                return Err(error::OnnxError::from_status(self.api, status));
            }

            // Get dimensions
            let mut dims = vec![0i64; dim_count];
            let get_dims: ffi::GetDimensionsFn =
                (*self.api).get_fn(ffi::IDX_GET_DIMENSIONS);
            let status = get_dims(tensor_info, dims.as_mut_ptr(), dim_count);

            // Release type info (tensor_info is a borrowed pointer, not owned)
            let release_type_info: ffi::ReleaseTypeInfoFn =
                (*self.api).get_fn(ffi::IDX_RELEASE_TYPE_INFO);
            release_type_info(type_info);

            if !status.is_null() {
                return Err(error::OnnxError::from_status(self.api, status));
            }

            Ok(dims)
        }
    }

    /// Get the element type of an input tensor by index
    ///
    /// # Errors
    /// Returns an error if the index is invalid or the input is not a tensor
    pub fn input_element_type(&self, index: usize) -> Result<ffi::ONNXTensorElementDataType> {
        unsafe {
            let mut type_info: *mut ffi::OrtTypeInfo = std::ptr::null_mut();
            let get_input_type_info: ffi::SessionGetInputTypeInfoFn =
                (*self.api).get_fn(ffi::IDX_SESSION_GET_INPUT_TYPE_INFO);
            let status = get_input_type_info(self.session, index, &mut type_info as *mut _);
            if !status.is_null() {
                return Err(error::OnnxError::from_status(self.api, status));
            }

            let mut tensor_info: *const ffi::OrtTensorTypeAndShapeInfo = std::ptr::null();
            let cast_fn: ffi::CastTypeInfoToTensorInfoFn =
                (*self.api).get_fn(ffi::IDX_CAST_TYPE_INFO_TO_TENSOR_INFO);
            let status = cast_fn(type_info, &mut tensor_info as *mut _);
            if !status.is_null() {
                let release_type_info: ffi::ReleaseTypeInfoFn =
                    (*self.api).get_fn(ffi::IDX_RELEASE_TYPE_INFO);
                release_type_info(type_info);
                return Err(error::OnnxError::from_status(self.api, status));
            }

            let mut elem_type = ffi::ONNXTensorElementDataType::Undefined;
            let get_elem_type: ffi::GetTensorElementTypeFn =
                (*self.api).get_fn(ffi::IDX_GET_TENSOR_ELEMENT_TYPE);
            let status = get_elem_type(tensor_info, &mut elem_type as *mut _);

            let release_type_info: ffi::ReleaseTypeInfoFn =
                (*self.api).get_fn(ffi::IDX_RELEASE_TYPE_INFO);
            release_type_info(type_info);

            if !status.is_null() {
                return Err(error::OnnxError::from_status(self.api, status));
            }

            Ok(elem_type)
        }
    }

    /// Run the model with named inputs and outputs
    ///
    /// # Errors
    /// Returns an error if inference fails
    pub fn run(&mut self, inputs: &[(&str, &crate::value::Value)], output_names: &[&str]) -> Result<Vec<crate::value::Value>> {
        unsafe {
            // Convert input names to CStrings
            let input_name_cstrings: Result<Vec<_>> = inputs
                .iter()
                .map(|(name, _)| {
                    CString::new(*name)
                        .map_err(|_| error::OnnxError::runtime_error("Null byte in input name"))
                })
                .collect();
            let input_name_cstrings = input_name_cstrings?;

            let input_name_ptrs: Vec<_> = input_name_cstrings
                .iter()
                .map(|s| s.as_ptr())
                .collect();

            // Get input value pointers
            let input_value_ptrs: Vec<_> = inputs
                .iter()
                .map(|(_, value)| value.as_ptr())
                .collect();

            // Convert output names to CStrings
            let output_name_cstrings: Result<Vec<_>> = output_names
                .iter()
                .map(|name| {
                    CString::new(*name)
                        .map_err(|_| error::OnnxError::runtime_error("Null byte in output name"))
                })
                .collect();
            let output_name_cstrings = output_name_cstrings?;

            let output_name_ptrs: Vec<_> = output_name_cstrings
                .iter()
                .map(|s| s.as_ptr())
                .collect();

            // Allocate output value slots (initialized to null)
            let mut output_value_ptrs: Vec<*mut ffi::OrtValue> =
                vec![std::ptr::null_mut(); output_names.len()];

            // Run inference
            let run: ffi::RunFn = (*self.api).get_fn(ffi::IDX_RUN);
            let status = run(
                self.session,
                std::ptr::null(), // run_options (null = default)
                input_name_ptrs.as_ptr(),
                input_value_ptrs.as_ptr(),
                inputs.len(),
                output_name_ptrs.as_ptr(),
                output_names.len(),
                output_value_ptrs.as_mut_ptr(),
            );

            if !status.is_null() {
                // On error, release any non-null output values
                for &output_ptr in &output_value_ptrs {
                    if !output_ptr.is_null() {
                        let release_value: ffi::ReleaseValueFn =
                            (*self.api).get_fn(ffi::IDX_RELEASE_VALUE);
                        release_value(output_ptr);
                    }
                }
                return Err(error::OnnxError::from_status(self.api, status));
            }

            // Wrap output values
            let outputs: Vec<_> = output_value_ptrs
                .into_iter()
                .map(|value_ptr| {
                    // Create an empty data buffer - output data is owned by OrtValue
                    crate::value::Value::from_raw(value_ptr, self.api)
                })
                .collect();

            Ok(outputs)
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            if !self.session.is_null() {
                let release_session: ffi::ReleaseSessionFn =
                    (*self.api).get_fn(ffi::IDX_RELEASE_SESSION);
                release_session(self.session);
            }
        }
    }
}

/// Create a new session builder
///
/// # Errors
/// Returns an error if session options cannot be created
///
/// # Panics
/// Panics if init() has not been called
pub fn session_builder() -> Result<SessionBuilder> {
    unsafe { SessionBuilder::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Session>();
    }

    #[test]
    fn test_session_builder_nonexistent_file() {
        // This test requires onnxruntime to be installed
        // It will fail at link time without it, but the logic is correct
        let _ = crate::init();

        if let Ok(builder) = session_builder() {
            let result = builder.commit_from_file("/nonexistent/model.onnx");
            assert!(result.is_err(), "Should error for non-existent file");
        }
    }
}
