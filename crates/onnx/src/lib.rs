pub mod error;
pub mod ffi;
pub mod session;
pub mod value;

pub use error::{OnnxError, Result};
pub use session::{session_builder, Session, SessionBuilder};
pub use value::{TensorElement, Value};
use ffi::{OrtApi, OrtEnv, OrtGetApiBase, OrtLoggingLevel, ORT_API_VERSION};
use std::ffi::CString;
use std::sync::{Mutex, OnceLock};

/// Global API and environment state
struct OnnxRuntime {
    api: *const OrtApi,
    env: *mut OrtEnv,
}

unsafe impl Send for OnnxRuntime {}
unsafe impl Sync for OnnxRuntime {}

static RUNTIME: OnceLock<Mutex<std::result::Result<OnnxRuntime, String>>> = OnceLock::new();

/// Initialize the ONNX Runtime
///
/// This must be called before using any other functionality.
/// Calling this multiple times is safe - it will return the same result.
///
/// # Errors
/// Returns an error if:
/// - The installed onnxruntime version doesn't support API version 17
/// - Environment creation fails
pub fn init() -> Result<()> {
    let mutex = RUNTIME.get_or_init(|| {
        let result = unsafe {
            // Get the API base
            let api_base = OrtGetApiBase();
            if api_base.is_null() {
                return Mutex::new(Err("Failed to get ONNX Runtime API base".to_string()));
            }

            // Get the specific API version
            let get_api = (*api_base).GetApi;
            let api = get_api(ORT_API_VERSION);
            if api.is_null() {
                return Mutex::new(Err(
                    "ONNX Runtime doesn't support API version 17. Your runtime may be too old."
                        .to_string(),
                ));
            }

            // Create the environment
            let log_id = CString::new("onnx").unwrap();
            let mut env: *mut OrtEnv = std::ptr::null_mut();

            let create_env: ffi::CreateEnvFn = (*api).get_fn(ffi::IDX_CREATE_ENV);
            let status = create_env(
                OrtLoggingLevel::Warning,
                log_id.as_ptr(),
                &mut env as *mut _,
            );

            if status.is_null() {
                Mutex::new(Ok(OnnxRuntime { api, env }))
            } else {
                let err = error::OnnxError::from_status(api, status);
                Mutex::new(Err(err.message().to_string()))
            }
        };

        result
    });

    match mutex.lock().unwrap().as_ref() {
        Ok(_) => Ok(()),
        Err(msg) => Err(OnnxError::runtime_error(msg)),
    }
}

/// Get the API pointer (for internal use)
///
/// # Safety
/// - Must call init() first
pub(crate) unsafe fn get_api() -> *const OrtApi {
    let mutex = RUNTIME
        .get()
        .expect("ONNX Runtime not initialized - call init() first");

    match mutex.lock().unwrap().as_ref() {
        Ok(runtime) => runtime.api,
        Err(_) => panic!("ONNX Runtime initialization failed"),
    }
}

/// Get the environment pointer (for internal use)
///
/// # Safety
/// - Must call init() first
pub(crate) unsafe fn get_env() -> *const OrtEnv {
    let mutex = RUNTIME
        .get()
        .expect("ONNX Runtime not initialized - call init() first");

    match mutex.lock().unwrap().as_ref() {
        Ok(runtime) => runtime.env,
        Err(_) => panic!("ONNX Runtime initialization failed"),
    }
}

// Ensure runtime is dropped properly (though it lives for 'static in practice)
impl Drop for OnnxRuntime {
    fn drop(&mut self) {
        unsafe {
            if !self.env.is_null() {
                let release_env: ffi::ReleaseEnvFn = (*self.api).get_fn(ffi::IDX_RELEASE_ENV);
                release_env(self.env);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_init_is_safe() {
        // First init - may fail if onnxruntime not installed, that's ok
        let _ = init();
        // Second init should not panic
        let _ = init();
    }
}
