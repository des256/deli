use crate::ffi::{OrtApi, OrtErrorCode, OrtStatus};
use std::ffi::CStr;
use std::fmt;

/// Error type for ONNX operations
#[derive(Debug, Clone)]
pub struct OnnxError {
    code: OrtErrorCode,
    message: String,
}

impl OnnxError {
    /// Create a runtime error with a message
    pub(crate) fn runtime_error(msg: &str) -> Self {
        OnnxError {
            code: OrtErrorCode::RuntimeException,
            message: msg.to_string(),
        }
    }

    /// Create an error from an OrtStatus pointer
    ///
    /// # Safety
    /// - status must be a valid non-null OrtStatus pointer
    /// - api must be a valid OrtApi pointer
    pub(crate) unsafe fn from_status(api: *const OrtApi, status: *mut OrtStatus) -> Self {
        unsafe {
            let api_ref = &*api;

            // Get error code
            let get_error_code: crate::ffi::GetErrorCodeFn = api_ref.get_fn(crate::ffi::IDX_GET_ERROR_CODE);
            let code = get_error_code(status);

            // Get error message
            let get_error_message: crate::ffi::GetErrorMessageFn = api_ref.get_fn(crate::ffi::IDX_GET_ERROR_MESSAGE);
            let msg_ptr = get_error_message(status);
            let message = if msg_ptr.is_null() {
                String::from("Unknown error")
            } else {
                CStr::from_ptr(msg_ptr)
                    .to_string_lossy()
                    .into_owned()
            };

            // Release the status
            let release_status: crate::ffi::ReleaseStatusFn = api_ref.get_fn(crate::ffi::IDX_RELEASE_STATUS);
            release_status(status);

            OnnxError { code, message }
        }
    }

    pub fn code(&self) -> OrtErrorCode {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ONNX error ({:?}): {}", self.code, self.message)
    }
}

impl std::error::Error for OnnxError {}

/// Result type for ONNX operations
pub type Result<T> = std::result::Result<T, OnnxError>;

/// Helper function to check OrtStatus and convert to Result
///
/// # Safety
/// - status can be null (indicates success)
/// - if non-null, status must be a valid OrtStatus pointer
/// - api must be a valid OrtApi pointer
pub(crate) unsafe fn check_status<T>(
    api: *const OrtApi,
    status: *mut OrtStatus,
    value: T,
) -> Result<T> {
    if status.is_null() {
        Ok(value)
    } else {
        unsafe { Err(OnnxError::from_status(api, status)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_implements_error_trait() {
        let err = OnnxError {
            code: OrtErrorCode::Fail,
            message: "test error".to_string(),
        };

        // Should implement Error trait
        fn assert_error<T: std::error::Error>(_: &T) {}
        assert_error(&err);
    }

    #[test]
    fn test_error_display() {
        let err = OnnxError {
            code: OrtErrorCode::InvalidArgument,
            message: "invalid input".to_string(),
        };

        let display = format!("{}", err);
        assert!(display.contains("invalid input"));
        assert!(display.contains("InvalidArgument"));
    }
}
