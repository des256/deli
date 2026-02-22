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
