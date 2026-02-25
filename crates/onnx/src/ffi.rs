use std::ffi::{c_char, c_void};

#[repr(C)]
pub struct OrtEnv {
    _private: [u8; 0],
}

unsafe impl Send for OrtEnv {}
unsafe impl Sync for OrtEnv {}

#[repr(C)]
pub struct OrtSession {
    _private: [u8; 0],
}

unsafe impl Send for OrtSession {}
unsafe impl Sync for OrtSession {}

#[repr(C)]
pub struct OrtSessionOptions {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtValue {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtStatus {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtMemoryInfo {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtAllocator {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtRunOptions {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtTensorTypeAndShapeInfo {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtTypeInfo {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OrtModelMetadata {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtLoggingLevel {
    Verbose = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
    Fatal = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ONNXTensorElementDataType {
    Undefined = 0,
    Float = 1,  // f32
    Uint8 = 2,  // u8
    Int8 = 3,   // i8
    Uint16 = 4, // u16
    Int16 = 5,  // i16
    Int32 = 6,  // i32
    Int64 = 7,  // i64
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11, // f64
    Uint32 = 12, // u32
    Uint64 = 13, // u64
    Complex64 = 14,
    Complex128 = 15,
    BFloat16 = 16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtErrorCode {
    Ok = 0,
    Fail = 1,
    InvalidArgument = 2,
    NoSuchFile = 3,
    NoModel = 4,
    EngineError = 5,
    RuntimeException = 6,
    InvalidProtobuf = 7,
    ModelLoaded = 8,
    NotImplemented = 9,
    InvalidGraph = 10,
    EpFail = 11,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphOptimizationLevel {
    DisableAll = 0,
    EnableBasic = 1,
    EnableExtended = 2,
    EnableAll = 99,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtMemType {
    CpuInput = -2,
    CpuOutput = -1,
    Default = 0,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtAllocatorType {
    Invalid = -1,
    Device = 0,
    Arena = 1,
}

#[repr(C)]
#[allow(non_snake_case)]
pub struct OrtApiBase {
    pub GetApi: unsafe extern "C" fn(version: u32) -> *const OrtApi,
    pub GetVersionString: unsafe extern "C" fn() -> *const c_char,
}

unsafe impl Send for OrtApiBase {}

#[repr(C)]
pub struct OrtApi {
    _private: [u8; 0],
}

unsafe impl Send for OrtApi {}
unsafe impl Sync for OrtApi {}

unsafe extern "C" {
    pub fn OrtGetApiBase() -> *const OrtApiBase;
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    pub fn OrtSessionOptionsAppendExecutionProvider_CUDA(
        options: *mut OrtSessionOptions,
        device_id: i32,
    ) -> *mut OrtStatus;
}

pub type CreateStatusFn =
    unsafe extern "C" fn(code: OrtErrorCode, msg: *const c_char) -> *mut OrtStatus;
pub type GetErrorCodeFn = unsafe extern "C" fn(status: *const OrtStatus) -> OrtErrorCode;
pub type GetErrorMessageFn = unsafe extern "C" fn(status: *const OrtStatus) -> *const c_char;
pub type CreateEnvFn = unsafe extern "C" fn(
    log_level: OrtLoggingLevel,
    log_id: *const c_char,
    out: *mut *mut OrtEnv,
) -> *mut OrtStatus;
pub type ReleaseEnvFn = unsafe extern "C" fn(env: *mut OrtEnv);
pub type GetAllocatorWithDefaultOptionsFn =
    unsafe extern "C" fn(out: *mut *mut OrtAllocator) -> *mut OrtStatus;
pub type AllocatorFreeFn = unsafe extern "C" fn(allocator: *mut OrtAllocator, ptr: *mut c_void);
pub type ReleaseStatusFn = unsafe extern "C" fn(status: *mut OrtStatus);
pub type CreateSessionFn = unsafe extern "C" fn(
    env: *const OrtEnv,
    model_path: *const c_char,
    options: *const OrtSessionOptions,
    out: *mut *mut OrtSession,
) -> *mut OrtStatus;
pub type ReleaseSessionFn = unsafe extern "C" fn(session: *mut OrtSession);
pub type RunFn = unsafe extern "C" fn(
    session: *mut OrtSession,
    run_options: *const OrtRunOptions,
    input_names: *const *const c_char,
    inputs: *const *const OrtValue,
    input_len: usize,
    output_names: *const *const c_char,
    output_names_len: usize,
    outputs: *mut *mut OrtValue,
) -> *mut OrtStatus;

pub type SessionGetInputCountFn =
    unsafe extern "C" fn(session: *const OrtSession, out: *mut usize) -> *mut OrtStatus;

pub type SessionGetOutputCountFn =
    unsafe extern "C" fn(session: *const OrtSession, out: *mut usize) -> *mut OrtStatus;

pub type SessionGetInputNameFn = unsafe extern "C" fn(
    session: *const OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
    value: *mut *mut c_char,
) -> *mut OrtStatus;

pub type SessionGetOutputNameFn = unsafe extern "C" fn(
    session: *const OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
    value: *mut *mut c_char,
) -> *mut OrtStatus;

pub type SessionGetInputTypeInfoFn = unsafe extern "C" fn(
    session: *const OrtSession,
    index: usize,
    type_info: *mut *mut OrtTypeInfo,
) -> *mut OrtStatus;

pub type CreateSessionOptionsFn =
    unsafe extern "C" fn(out: *mut *mut OrtSessionOptions) -> *mut OrtStatus;
pub type ReleaseSessionOptionsFn = unsafe extern "C" fn(options: *mut OrtSessionOptions);
pub type SetSessionGraphOptimizationLevelFn = unsafe extern "C" fn(
    options: *mut OrtSessionOptions,
    level: GraphOptimizationLevel,
) -> *mut OrtStatus;
pub type SetIntraOpNumThreadsFn =
    unsafe extern "C" fn(options: *mut OrtSessionOptions, num_threads: i32) -> *mut OrtStatus;
pub type CreateCpuMemoryInfoFn = unsafe extern "C" fn(
    allocator_type: OrtAllocatorType,
    mem_type: OrtMemType,
    out: *mut *mut OrtMemoryInfo,
) -> *mut OrtStatus;
pub type ReleaseMemoryInfoFn = unsafe extern "C" fn(info: *mut OrtMemoryInfo);
pub type CreateTensorWithDataAsOrtValueFn = unsafe extern "C" fn(
    memory_info: *const OrtMemoryInfo,
    data: *mut std::ffi::c_void,
    data_len: usize,
    shape: *const i64,
    shape_len: usize,
    element_type: ONNXTensorElementDataType,
    out: *mut *mut OrtValue,
) -> *mut OrtStatus;
pub type ReleaseValueFn = unsafe extern "C" fn(value: *mut OrtValue);
pub type GetTensorMutableDataFn =
    unsafe extern "C" fn(value: *mut OrtValue, out: *mut *mut std::ffi::c_void) -> *mut OrtStatus;
pub type GetTensorTypeAndShapeFn = unsafe extern "C" fn(
    value: *const OrtValue,
    out: *mut *mut OrtTensorTypeAndShapeInfo,
) -> *mut OrtStatus;
pub type ReleaseTensorTypeAndShapeInfoFn =
    unsafe extern "C" fn(info: *mut OrtTensorTypeAndShapeInfo);
pub type ReleaseTypeInfoFn = unsafe extern "C" fn(info: *mut OrtTypeInfo);
pub type GetTensorElementTypeFn = unsafe extern "C" fn(
    info: *const OrtTensorTypeAndShapeInfo,
    out: *mut ONNXTensorElementDataType,
) -> *mut OrtStatus;
pub type CastTypeInfoToTensorInfoFn = unsafe extern "C" fn(
    type_info: *const OrtTypeInfo,
    out: *mut *const OrtTensorTypeAndShapeInfo,
) -> *mut OrtStatus;
pub type GetDimensionsCountFn =
    unsafe extern "C" fn(info: *const OrtTensorTypeAndShapeInfo, out: *mut usize) -> *mut OrtStatus;
pub type GetDimensionsFn = unsafe extern "C" fn(
    info: *const OrtTensorTypeAndShapeInfo,
    dim_values: *mut i64,
    dim_count: usize,
) -> *mut OrtStatus;
pub type GetTensorShapeElementCountFn =
    unsafe extern "C" fn(info: *const OrtTensorTypeAndShapeInfo, out: *mut usize) -> *mut OrtStatus;

// Model metadata
pub type SessionGetModelMetadataFn = unsafe extern "C" fn(
    session: *const OrtSession,
    out: *mut *mut OrtModelMetadata,
) -> *mut OrtStatus;
pub type ModelMetadataLookupCustomMetadataMapFn = unsafe extern "C" fn(
    model_metadata: *const OrtModelMetadata,
    allocator: *mut OrtAllocator,
    key: *const c_char,
    value: *mut *mut c_char,
) -> *mut OrtStatus;
pub type ModelMetadataGetCustomMetadataMapKeysFn = unsafe extern "C" fn(
    model_metadata: *const OrtModelMetadata,
    allocator: *mut OrtAllocator,
    keys: *mut *mut *mut c_char,
    num_keys: *mut i64,
) -> *mut OrtStatus;
pub type ReleaseModelMetadataFn = unsafe extern "C" fn(metadata: *mut OrtModelMetadata);

// OrtApi vtable indices — verified against onnxruntime_c_api.h v1.24.2
// on Jetson (aarch64). These indices are stable across versions since
// the vtable only grows (new entries appended, existing entries never move).
//
// Indices 0-2: raw function pointers (CreateStatus, GetErrorCode, GetErrorMessage)
// Indices 3+: ORT_API2_STATUS and ORT_CLASS_RELEASE entries
pub const IDX_CREATE_STATUS: usize = 0;
pub const IDX_GET_ERROR_CODE: usize = 1;
pub const IDX_GET_ERROR_MESSAGE: usize = 2;
pub const IDX_CREATE_ENV: usize = 3;
pub const IDX_CREATE_SESSION: usize = 7;
pub const IDX_RUN: usize = 9;
pub const IDX_CREATE_SESSION_OPTIONS: usize = 10;
pub const IDX_SESSION_GET_INPUT_COUNT: usize = 30;
pub const IDX_SESSION_GET_OUTPUT_COUNT: usize = 31;
pub const IDX_SESSION_GET_INPUT_TYPE_INFO: usize = 33;
pub const IDX_SESSION_GET_INPUT_NAME: usize = 36;
pub const IDX_SESSION_GET_OUTPUT_NAME: usize = 37;
pub const IDX_ALLOCATOR_FREE: usize = 76;
pub const IDX_GET_ALLOCATOR_WITH_DEFAULT_OPTIONS: usize = 78;
pub const IDX_SET_SESSION_GRAPH_OPTIMIZATION_LEVEL: usize = 23;
pub const IDX_SET_INTRA_OP_NUM_THREADS: usize = 24;
// Tensor creation and data access
pub const IDX_CREATE_TENSOR_WITH_DATA_AS_ORT_VALUE: usize = 49;
pub const IDX_GET_TENSOR_MUTABLE_DATA: usize = 51;
// Type info casting
pub const IDX_CAST_TYPE_INFO_TO_TENSOR_INFO: usize = 55;
// Tensor type/shape info
pub const IDX_GET_DIMENSIONS_COUNT: usize = 61;
pub const IDX_GET_DIMENSIONS: usize = 62;
pub const IDX_GET_TENSOR_ELEMENT_TYPE: usize = 60;
pub const IDX_GET_TENSOR_SHAPE_ELEMENT_COUNT: usize = 64;
pub const IDX_GET_TENSOR_TYPE_AND_SHAPE: usize = 65;
// Memory info
pub const IDX_CREATE_CPU_MEMORY_INFO: usize = 69;
// Release functions
pub const IDX_RELEASE_ENV: usize = 92;
pub const IDX_RELEASE_STATUS: usize = 93;
pub const IDX_RELEASE_MEMORY_INFO: usize = 94;
pub const IDX_RELEASE_SESSION: usize = 95;
pub const IDX_RELEASE_VALUE: usize = 96;
pub const IDX_RELEASE_RUN_OPTIONS: usize = 97;
pub const IDX_RELEASE_TYPE_INFO: usize = 98;
pub const IDX_RELEASE_TENSOR_TYPE_AND_SHAPE_INFO: usize = 99;
pub const IDX_RELEASE_SESSION_OPTIONS: usize = 100;
// Model metadata
pub const IDX_SESSION_GET_MODEL_METADATA: usize = 111;
pub const IDX_MODEL_METADATA_LOOKUP_CUSTOM_METADATA_MAP: usize = 116;
pub const IDX_RELEASE_MODEL_METADATA: usize = 118;
pub const IDX_MODEL_METADATA_GET_CUSTOM_METADATA_MAP_KEYS: usize = 123;

impl OrtApi {
    pub unsafe fn get_fn<F>(&self, index: usize) -> F {
        unsafe {
            let vtable = self as *const _ as *const *const ();
            let fn_ptr = *vtable.add(index);
            std::mem::transmute_copy(&fn_ptr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opaque_types_are_zero_sized() {
        // Opaque types should be zero-sized for correct FFI
        assert_eq!(std::mem::size_of::<OrtEnv>(), 0);
        assert_eq!(std::mem::size_of::<OrtSession>(), 0);
        assert_eq!(std::mem::size_of::<OrtSessionOptions>(), 0);
        assert_eq!(std::mem::size_of::<OrtValue>(), 0);
        assert_eq!(std::mem::size_of::<OrtStatus>(), 0);
    }

    #[test]
    fn test_enum_discriminants() {
        // Verify enum values match the C API
        assert_eq!(OrtLoggingLevel::Warning as i32, 2);
        assert_eq!(OrtErrorCode::Ok as i32, 0);
        assert_eq!(OrtErrorCode::Fail as i32, 1);
        assert_eq!(ONNXTensorElementDataType::Float as i32, 1);
        assert_eq!(ONNXTensorElementDataType::Int64 as i32, 7);
        assert_eq!(ONNXTensorElementDataType::Double as i32, 11);
        assert_eq!(GraphOptimizationLevel::EnableAll as i32, 99);
    }
}
