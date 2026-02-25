use std::{ffi::CStr, fmt};

mod onnx;
pub use onnx::*;

mod session;
pub use session::*;

mod ffi;

mod value;
pub use value::*;

pub(crate) mod f16;

#[derive(Debug, Clone)]
pub struct OnnxError {
    pub code: ffi::OrtErrorCode,
    pub message: String,
}

impl OnnxError {
    pub(crate) fn runtime_error(message: &str) -> Self {
        OnnxError {
            code: ffi::OrtErrorCode::RuntimeException,
            message: message.to_string(),
        }
    }

    pub(crate) fn from_status(api: *const ffi::OrtApi, status: *mut ffi::OrtStatus) -> Self {
        let get_error_code: crate::ffi::GetErrorCodeFn =
            unsafe { (*api).get_fn(crate::ffi::IDX_GET_ERROR_CODE) };
        let code = unsafe { get_error_code(status) };

        let get_error_message: crate::ffi::GetErrorMessageFn =
            unsafe { (*api).get_fn(crate::ffi::IDX_GET_ERROR_MESSAGE) };
        let msg_ptr = unsafe { get_error_message(status) };
        let message = if msg_ptr.is_null() {
            String::from("Unknown error")
        } else {
            unsafe { CStr::from_ptr(msg_ptr).to_string_lossy().into_owned() }
        };

        let release_status: crate::ffi::ReleaseStatusFn =
            unsafe { (*api).get_fn(crate::ffi::IDX_RELEASE_STATUS) };
        unsafe { release_status(status) };

        OnnxError { code, message }
    }
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ONNX error ({:?}): {}", self.code, self.message)
    }
}

pub(crate) fn check_status<T>(
    api: *const ffi::OrtApi,
    status: *mut ffi::OrtStatus,
    value: T,
) -> Result<T, OnnxError> {
    if status.is_null() {
        Ok(value)
    } else {
        Err(OnnxError::from_status(api, status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = OnnxError {
            code: ffi::OrtErrorCode::InvalidArgument,
            message: "invalid input".to_string(),
        };

        let display = format!("{}", err);
        assert!(display.contains("invalid input"));
        assert!(display.contains("InvalidArgument"));
    }
}
