use {
    crate::*,
    std::{ffi::CString, path::Path, ptr::null_mut, sync::Arc},
};

#[derive(Debug, Clone)]
pub enum Executor {
    Cpu,
    Cuda(usize),
}

#[derive(Debug, Clone)]
pub enum OptimizationLevel {
    Disabled,
    EnableBasic,
    EnableExtended,
    EnableAll,
}

#[derive(Debug, Clone)]
pub struct Onnx {
    pub(crate) api: *const ffi::OrtApi,
    pub(crate) environment: *mut ffi::OrtEnv,
    pub(crate) allocator: *mut ffi::OrtAllocator,
    pub(crate) allocator_free: ffi::AllocatorFreeFn,
    pub(crate) create_session: ffi::CreateSessionFn,
    pub(crate) create_session_options: ffi::CreateSessionOptionsFn,
    pub(crate) set_session_graph_optimization_level: ffi::SetSessionGraphOptimizationLevelFn,
    pub(crate) set_intra_op_num_threads: ffi::SetIntraOpNumThreadsFn,
    pub(crate) release_session_options: ffi::ReleaseSessionOptionsFn,
    pub(crate) session_get_input_count: ffi::SessionGetInputCountFn,
    pub(crate) session_get_output_count: ffi::SessionGetOutputCountFn,
    pub(crate) session_get_input_name: ffi::SessionGetInputNameFn,
    pub(crate) session_get_output_name: ffi::SessionGetOutputNameFn,
    pub(crate) session_get_input_type_info: ffi::SessionGetInputTypeInfoFn,
    pub(crate) release_type_info: ffi::ReleaseTypeInfoFn,
    pub(crate) cast_type_info_to_tensor_info: ffi::CastTypeInfoToTensorInfoFn,
    pub(crate) get_dimensions_count: ffi::GetDimensionsCountFn,
    pub(crate) get_dimensions: ffi::GetDimensionsFn,
    pub(crate) get_tensor_element_type: ffi::GetTensorElementTypeFn,
    pub(crate) run: ffi::RunFn,
    pub(crate) release_value: ffi::ReleaseValueFn,
    pub(crate) release_session: ffi::ReleaseSessionFn,
    pub(crate) create_memory_info: ffi::CreateCpuMemoryInfoFn,
    pub(crate) create_tensor: ffi::CreateTensorWithDataAsOrtValueFn,
    pub(crate) release_memory_info: ffi::ReleaseMemoryInfoFn,
    pub(crate) get_tensor_type_and_shape: ffi::GetTensorTypeAndShapeFn,
    pub(crate) get_tensor_shape_element_count: ffi::GetTensorShapeElementCountFn,
    pub(crate) release_tensor_type_and_shape_info: ffi::ReleaseTensorTypeAndShapeInfoFn,
    pub(crate) get_tensor_mutable_data: ffi::GetTensorMutableDataFn,
    // Model metadata
    pub(crate) session_get_model_metadata: ffi::SessionGetModelMetadataFn,
    pub(crate) model_metadata_lookup_custom_metadata_map:
        ffi::ModelMetadataLookupCustomMetadataMapFn,
    pub(crate) model_metadata_get_custom_metadata_map_keys:
        ffi::ModelMetadataGetCustomMetadataMapKeysFn,
    pub(crate) release_model_metadata: ffi::ReleaseModelMetadataFn,
}

unsafe impl Send for Onnx {}
unsafe impl Sync for Onnx {}

impl Onnx {
    pub fn new(version: usize) -> Result<Arc<Self>, OnnxError> {
        // get API base
        let api_base = unsafe { ffi::OrtGetApiBase() };
        if api_base.is_null() {
            panic!("Failed to get ONNX runtime API base");
        }

        // get versioned API
        let get_api = unsafe { (*api_base).GetApi };
        let api = unsafe { get_api(version as u32) };
        if api.is_null() {
            panic!("ONNX runtime doesn't support API version {}", version);
        }

        // get function pointers
        let create_env: ffi::CreateEnvFn = unsafe { (*api).get_fn(ffi::IDX_CREATE_ENV) };
        let allocator_free: ffi::AllocatorFreeFn =
            unsafe { (*api).get_fn(ffi::IDX_ALLOCATOR_FREE) };
        let create_session: ffi::CreateSessionFn =
            unsafe { (*api).get_fn(ffi::IDX_CREATE_SESSION) };
        let create_session_options: ffi::CreateSessionOptionsFn =
            unsafe { (*api).get_fn(ffi::IDX_CREATE_SESSION_OPTIONS) };
        let set_session_graph_optimization_level: ffi::SetSessionGraphOptimizationLevelFn =
            unsafe { (*api).get_fn(ffi::IDX_SET_SESSION_GRAPH_OPTIMIZATION_LEVEL) };
        let set_intra_op_num_threads: ffi::SetIntraOpNumThreadsFn =
            unsafe { (*api).get_fn(ffi::IDX_SET_INTRA_OP_NUM_THREADS) };
        let release_session_options: ffi::ReleaseSessionOptionsFn =
            unsafe { (*api).get_fn(ffi::IDX_RELEASE_SESSION_OPTIONS) };
        let session_get_input_count: ffi::SessionGetInputCountFn =
            unsafe { (*api).get_fn(ffi::IDX_SESSION_GET_INPUT_COUNT) };
        let session_get_output_count: ffi::SessionGetOutputCountFn =
            unsafe { (*api).get_fn(ffi::IDX_SESSION_GET_OUTPUT_COUNT) };
        let session_get_input_name: ffi::SessionGetInputNameFn =
            unsafe { (*api).get_fn(ffi::IDX_SESSION_GET_INPUT_NAME) };
        let session_get_output_name: ffi::SessionGetOutputNameFn =
            unsafe { (*api).get_fn(ffi::IDX_SESSION_GET_OUTPUT_NAME) };
        let session_get_input_type_info: ffi::SessionGetInputTypeInfoFn =
            unsafe { (*api).get_fn(ffi::IDX_SESSION_GET_INPUT_TYPE_INFO) };
        let release_type_info: ffi::ReleaseTypeInfoFn =
            unsafe { (*api).get_fn(ffi::IDX_RELEASE_TYPE_INFO) };
        let cast_type_info_to_tensor_info: ffi::CastTypeInfoToTensorInfoFn =
            unsafe { (*api).get_fn(ffi::IDX_CAST_TYPE_INFO_TO_TENSOR_INFO) };
        let get_dimensions_count: ffi::GetDimensionsCountFn =
            unsafe { (*api).get_fn(ffi::IDX_GET_DIMENSIONS_COUNT) };
        let get_dimensions: ffi::GetDimensionsFn =
            unsafe { (*api).get_fn(ffi::IDX_GET_DIMENSIONS) };
        let get_tensor_element_type: ffi::GetTensorElementTypeFn =
            unsafe { (*api).get_fn(ffi::IDX_GET_TENSOR_ELEMENT_TYPE) };
        let run: ffi::RunFn = unsafe { (*api).get_fn(ffi::IDX_RUN) };
        let release_value: ffi::ReleaseValueFn = unsafe { (*api).get_fn(ffi::IDX_RELEASE_VALUE) };
        let release_session: ffi::ReleaseSessionFn =
            unsafe { (*api).get_fn(ffi::IDX_RELEASE_SESSION) };
        let create_memory_info: ffi::CreateCpuMemoryInfoFn =
            unsafe { (*api).get_fn(ffi::IDX_CREATE_CPU_MEMORY_INFO) };
        let create_tensor: ffi::CreateTensorWithDataAsOrtValueFn =
            unsafe { (*api).get_fn(ffi::IDX_CREATE_TENSOR_WITH_DATA_AS_ORT_VALUE) };
        let release_memory_info: ffi::ReleaseMemoryInfoFn =
            unsafe { (*api).get_fn(ffi::IDX_RELEASE_MEMORY_INFO) };
        let get_tensor_type_and_shape: ffi::GetTensorTypeAndShapeFn =
            unsafe { (*api).get_fn(ffi::IDX_GET_TENSOR_TYPE_AND_SHAPE) };
        let get_tensor_shape_element_count: ffi::GetTensorShapeElementCountFn =
            unsafe { (*api).get_fn(ffi::IDX_GET_TENSOR_SHAPE_ELEMENT_COUNT) };
        let release_tensor_type_and_shape_info: ffi::ReleaseTensorTypeAndShapeInfoFn =
            unsafe { (*api).get_fn(ffi::IDX_RELEASE_TENSOR_TYPE_AND_SHAPE_INFO) };
        let get_tensor_mutable_data: ffi::GetTensorMutableDataFn =
            unsafe { (*api).get_fn(ffi::IDX_GET_TENSOR_MUTABLE_DATA) };
        let session_get_model_metadata: ffi::SessionGetModelMetadataFn =
            unsafe { (*api).get_fn(ffi::IDX_SESSION_GET_MODEL_METADATA) };
        let model_metadata_lookup_custom_metadata_map: ffi::ModelMetadataLookupCustomMetadataMapFn =
            unsafe { (*api).get_fn(ffi::IDX_MODEL_METADATA_LOOKUP_CUSTOM_METADATA_MAP) };
        let model_metadata_get_custom_metadata_map_keys: ffi::ModelMetadataGetCustomMetadataMapKeysFn =
            unsafe { (*api).get_fn(ffi::IDX_MODEL_METADATA_GET_CUSTOM_METADATA_MAP_KEYS) };
        let release_model_metadata: ffi::ReleaseModelMetadataFn =
            unsafe { (*api).get_fn(ffi::IDX_RELEASE_MODEL_METADATA) };

        // create environment
        let log_id = CString::new("onnx").unwrap();
        let mut environment: *mut ffi::OrtEnv = null_mut();
        let status = unsafe {
            create_env(
                ffi::OrtLoggingLevel::Warning,
                log_id.as_ptr(),
                &mut environment as *mut _,
            )
        };
        if !status.is_null() {
            let err = OnnxError::from_status(api, status);
            panic!("Failed to create ONNX runtime environment: {}", err.message);
        }

        // get allocator
        let mut allocator: *mut ffi::OrtAllocator = std::ptr::null_mut();
        let get_allocator_with_default_options: ffi::GetAllocatorWithDefaultOptionsFn =
            unsafe { (*api).get_fn(ffi::IDX_GET_ALLOCATOR_WITH_DEFAULT_OPTIONS) };
        let status = unsafe { get_allocator_with_default_options(&mut allocator as *mut _) };
        if !status.is_null() {
            return Err(OnnxError::from_status(api, status));
        }

        Ok(Arc::new(Self {
            api,
            environment,
            allocator,
            allocator_free,
            create_session,
            create_session_options,
            set_session_graph_optimization_level,
            set_intra_op_num_threads,
            release_session_options,
            session_get_input_count,
            session_get_output_count,
            session_get_input_name,
            session_get_output_name,
            session_get_input_type_info,
            release_type_info,
            cast_type_info_to_tensor_info,
            get_dimensions_count,
            get_dimensions,
            get_tensor_element_type,
            run,
            release_value,
            release_session,
            create_memory_info,
            create_tensor,
            release_memory_info,
            get_tensor_type_and_shape,
            get_tensor_shape_element_count,
            release_tensor_type_and_shape_info,
            get_tensor_mutable_data,
            session_get_model_metadata,
            model_metadata_lookup_custom_metadata_map,
            model_metadata_get_custom_metadata_map_keys,
            release_model_metadata,
        }))
    }

    pub fn create_session(
        self: &Arc<Self>,
        executor: &Executor,
        optimization_level: &OptimizationLevel,
        threads: usize,
        model_path: impl AsRef<Path>,
    ) -> Result<Session, OnnxError> {
        let mut options: *mut ffi::OrtSessionOptions = null_mut();
        let status = unsafe { (self.create_session_options)(&mut options as *mut _) };
        if !status.is_null() {
            return Err(OnnxError::from_status(self.api, status));
        }

        #[cfg(feature = "cuda")]
        if let Executor::Cuda(id) = executor {
            let status =
                unsafe { ffi::OrtSessionOptionsAppendExecutionProvider_CUDA(options, *id as i32) };
            if !status.is_null() {
                unsafe { (self.release_session_options)(options) };
                return Err(OnnxError::from_status(self.api, status));
            }
        }

        let status = unsafe {
            (self.set_session_graph_optimization_level)(
                options,
                match optimization_level {
                    OptimizationLevel::Disabled => ffi::GraphOptimizationLevel::DisableAll,
                    OptimizationLevel::EnableBasic => ffi::GraphOptimizationLevel::EnableBasic,
                    OptimizationLevel::EnableExtended => {
                        ffi::GraphOptimizationLevel::EnableExtended
                    }
                    OptimizationLevel::EnableAll => ffi::GraphOptimizationLevel::EnableAll,
                },
            )
        };
        if !status.is_null() {
            unsafe { (self.release_session_options)(options) };
            return Err(OnnxError::from_status(self.api, status));
        }

        let status = unsafe { (self.set_intra_op_num_threads)(options, threads as i32) };
        if !status.is_null() {
            unsafe { (self.release_session_options)(options) };
            return Err(OnnxError::from_status(self.api, status));
        }

        let path_str = model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| OnnxError::runtime_error("Invalid UTF-8 in model path"))?;
        let c_path = CString::new(path_str)
            .map_err(|_| OnnxError::runtime_error("Null byte in model path"))?;
        let mut session: *mut ffi::OrtSession = null_mut();
        let status = unsafe {
            (self.create_session)(
                self.environment,
                c_path.as_ptr(),
                options,
                &mut session as *mut _,
            )
        };
        if !status.is_null() {
            unsafe { (self.release_session_options)(options) };
            return Err(OnnxError::from_status(self.api, status));
        }

        unsafe { (self.release_session_options)(options) };

        Ok(Session {
            onnx: Arc::clone(&self),
            session,
        })
    }
}

impl Drop for Onnx {
    fn drop(&mut self) {
        let release_env: ffi::ReleaseEnvFn = unsafe { (*self.api).get_fn(ffi::IDX_RELEASE_ENV) };
        unsafe { release_env(self.environment) };
    }
}
