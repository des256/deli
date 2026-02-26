use {
    crate::*,
    std::{collections::HashMap, ffi::CString, sync::Arc},
};

pub struct Session {
    pub onnx: Arc<Onnx>,
    pub session: *mut ffi::OrtSession,
}

unsafe impl Send for Session {}

impl Session {
    pub fn input_count(&self) -> Result<usize, OnnxError> {
        let mut count: usize = 0;

        let status =
            unsafe { (self.onnx.session_get_input_count)(self.session, &mut count as *mut _) };
        if !status.is_null() {
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        Ok(count)
    }

    pub fn output_count(&self) -> Result<usize, OnnxError> {
        let mut count: usize = 0;

        let status =
            unsafe { (self.onnx.session_get_output_count)(self.session, &mut count as *mut _) };
        if !status.is_null() {
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        Ok(count)
    }

    pub fn input_name(&self, index: usize) -> Result<String, OnnxError> {
        let mut name_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
        let status = unsafe {
            (self.onnx.session_get_input_name)(
                self.session,
                index,
                self.onnx.allocator,
                &mut name_ptr as *mut _,
            )
        };
        if !status.is_null() {
            unsafe {
                (self.onnx.allocator_free)(self.onnx.allocator, name_ptr as *mut std::ffi::c_void)
            };
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        let name = unsafe {
            std::ffi::CStr::from_ptr(name_ptr)
                .to_str()
                .map_err(|_| OnnxError::runtime_error("Invalid UTF-8 in input name"))?
                .to_string()
        };

        unsafe {
            (self.onnx.allocator_free)(self.onnx.allocator, name_ptr as *mut std::ffi::c_void)
        };

        Ok(name)
    }

    pub fn output_name(&self, index: usize) -> Result<String, OnnxError> {
        let mut name_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
        let status = unsafe {
            (self.onnx.session_get_output_name)(
                self.session,
                index,
                self.onnx.allocator,
                &mut name_ptr as *mut _,
            )
        };
        if !status.is_null() {
            unsafe {
                (self.onnx.allocator_free)(self.onnx.allocator, name_ptr as *mut std::ffi::c_void)
            };
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        let name = unsafe {
            std::ffi::CStr::from_ptr(name_ptr)
                .to_str()
                .map_err(|_| OnnxError::runtime_error("Invalid UTF-8 in output name"))?
                .to_string()
        };

        unsafe {
            (self.onnx.allocator_free)(self.onnx.allocator, name_ptr as *mut std::ffi::c_void)
        };

        Ok(name)
    }

    fn get_type_info(&self, index: usize) -> Result<*mut ffi::OrtTypeInfo, OnnxError> {
        let mut type_info: *mut ffi::OrtTypeInfo = std::ptr::null_mut();
        let status = unsafe {
            (self.onnx.session_get_input_type_info)(self.session, index, &mut type_info as *mut _)
        };
        if !status.is_null() {
            return Err(OnnxError::from_status(self.onnx.api, status));
        }
        Ok(type_info)
    }

    fn release_type_info(&self, type_info: *mut ffi::OrtTypeInfo) {
        unsafe { (self.onnx.release_type_info)(type_info) };
    }

    pub fn input_shape(&self, index: usize) -> Result<Vec<i64>, OnnxError> {
        let type_info = self.get_type_info(index)?;

        let mut tensor_info: *const ffi::OrtTensorTypeAndShapeInfo = std::ptr::null();
        let status = unsafe {
            (self.onnx.cast_type_info_to_tensor_info)(type_info, &mut tensor_info as *mut _)
        };
        if !status.is_null() {
            self.release_type_info(type_info);
            return Err(OnnxError::from_status(self.onnx.api, status));
        }
        if tensor_info.is_null() {
            self.release_type_info(type_info);
            return Err(OnnxError::runtime_error("Input is not a tensor type"));
        }
        let mut dim_count: usize = 0;
        let status =
            unsafe { (self.onnx.get_dimensions_count)(tensor_info, &mut dim_count as *mut _) };
        if !status.is_null() {
            self.release_type_info(type_info);
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        let mut dims = vec![0i64; dim_count];
        let status =
            unsafe { (self.onnx.get_dimensions)(tensor_info, dims.as_mut_ptr(), dim_count) };
        if !status.is_null() {
            self.release_type_info(type_info);
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        self.release_type_info(type_info);

        Ok(dims)
    }

    pub fn input_element_type(
        &self,
        index: usize,
    ) -> Result<ffi::ONNXTensorElementDataType, OnnxError> {
        let mut type_info: *mut ffi::OrtTypeInfo = std::ptr::null_mut();
        let status = unsafe {
            (self.onnx.session_get_input_type_info)(self.session, index, &mut type_info as *mut _)
        };
        if !status.is_null() {
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        let mut tensor_info: *const ffi::OrtTensorTypeAndShapeInfo = std::ptr::null();
        let status = unsafe {
            (self.onnx.cast_type_info_to_tensor_info)(type_info, &mut tensor_info as *mut _)
        };
        if !status.is_null() {
            self.release_type_info(type_info);
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        let mut elem_type = ffi::ONNXTensorElementDataType::Undefined;
        let status =
            unsafe { (self.onnx.get_tensor_element_type)(tensor_info, &mut elem_type as *mut _) };

        self.release_type_info(type_info);

        if !status.is_null() {
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        Ok(elem_type)
    }

    /// Get custom metadata from the model as a key-value map.
    pub fn metadata(&self) -> Result<HashMap<String, String>, OnnxError> {
        // Get model metadata handle
        let mut metadata: *mut ffi::OrtModelMetadata = std::ptr::null_mut();
        let status = unsafe {
            (self.onnx.session_get_model_metadata)(self.session, &mut metadata as *mut _)
        };
        if !status.is_null() {
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        // Get all custom metadata keys
        let mut keys_ptr: *mut *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut num_keys: i64 = 0;
        let status = unsafe {
            (self.onnx.model_metadata_get_custom_metadata_map_keys)(
                metadata,
                self.onnx.allocator,
                &mut keys_ptr as *mut _,
                &mut num_keys as *mut _,
            )
        };
        if !status.is_null() {
            unsafe { (self.onnx.release_model_metadata)(metadata) };
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        let mut map = HashMap::new();

        for i in 0..num_keys as usize {
            let key_ptr = unsafe { *keys_ptr.add(i) };
            let key = unsafe {
                std::ffi::CStr::from_ptr(key_ptr)
                    .to_string_lossy()
                    .into_owned()
            };

            // Look up the value for this key
            let key_cstr = CString::new(key.as_str())
                .map_err(|_| OnnxError::runtime_error("Null byte in metadata key"))?;
            let mut value_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
            let status = unsafe {
                (self.onnx.model_metadata_lookup_custom_metadata_map)(
                    metadata,
                    self.onnx.allocator,
                    key_cstr.as_ptr(),
                    &mut value_ptr as *mut _,
                )
            };
            if !status.is_null() {
                // Free the key, remaining keys, and metadata before returning
                unsafe { (self.onnx.allocator_free)(self.onnx.allocator, key_ptr as *mut _) };
                for j in (i + 1)..num_keys as usize {
                    unsafe {
                        (self.onnx.allocator_free)(self.onnx.allocator, *keys_ptr.add(j) as *mut _)
                    };
                }
                unsafe { (self.onnx.allocator_free)(self.onnx.allocator, keys_ptr as *mut _) };
                unsafe { (self.onnx.release_model_metadata)(metadata) };
                return Err(OnnxError::from_status(self.onnx.api, status));
            }

            if !value_ptr.is_null() {
                let value = unsafe {
                    std::ffi::CStr::from_ptr(value_ptr)
                        .to_string_lossy()
                        .into_owned()
                };
                map.insert(key, value);
                unsafe { (self.onnx.allocator_free)(self.onnx.allocator, value_ptr as *mut _) };
            }

            unsafe { (self.onnx.allocator_free)(self.onnx.allocator, key_ptr as *mut _) };
        }

        // Free the keys array and metadata handle
        if !keys_ptr.is_null() {
            unsafe { (self.onnx.allocator_free)(self.onnx.allocator, keys_ptr as *mut _) };
        }
        unsafe { (self.onnx.release_model_metadata)(metadata) };

        Ok(map)
    }

    pub fn run(
        &mut self,
        inputs: &[(&str, &Value)],
        output_names: &[&str],
    ) -> Result<Vec<Value>, OnnxError> {
        // Convert input names to CStrings
        let input_name_cstrings: Result<Vec<_>, OnnxError> = inputs
            .iter()
            .map(|(name, _)| {
                CString::new(*name).map_err(|_| OnnxError::runtime_error("Null byte in input name"))
            })
            .collect();
        let input_name_cstrings = input_name_cstrings?;

        let input_name_ptrs: Vec<_> = input_name_cstrings.iter().map(|s| s.as_ptr()).collect();

        // Get input value pointers
        let input_value_ptrs: Vec<_> = inputs.iter().map(|(_, value)| value.as_ptr()).collect();

        // Convert output names to CStrings
        let output_name_cstrings: Result<Vec<_>, OnnxError> = output_names
            .iter()
            .map(|name| {
                CString::new(*name)
                    .map_err(|_| OnnxError::runtime_error("Null byte in output name"))
            })
            .collect();
        let output_name_cstrings = output_name_cstrings?;

        let output_name_ptrs: Vec<_> = output_name_cstrings.iter().map(|s| s.as_ptr()).collect();

        // Allocate output value slots (initialized to null)
        let mut output_value_ptrs: Vec<*mut ffi::OrtValue> =
            vec![std::ptr::null_mut(); output_names.len()];

        // Run inference
        let status = unsafe {
            (self.onnx.run)(
                self.session,
                std::ptr::null(), // run_options (null = default)
                input_name_ptrs.as_ptr(),
                input_value_ptrs.as_ptr(),
                inputs.len(),
                output_name_ptrs.as_ptr(),
                output_names.len(),
                output_value_ptrs.as_mut_ptr(),
            )
        };

        if !status.is_null() {
            // On error, release any non-null output values
            for &output_ptr in &output_value_ptrs {
                if !output_ptr.is_null() {
                    unsafe { (self.onnx.release_value)(output_ptr) };
                }
            }
            return Err(OnnxError::from_status(self.onnx.api, status));
        }

        // Wrap output values
        let outputs: Vec<_> = output_value_ptrs
            .into_iter()
            .map(|value_ptr| {
                // Create an empty data buffer - output data is owned by OrtValue
                unsafe { Value::from_raw(&self.onnx, value_ptr) }
            })
            .collect();

        Ok(outputs)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.session.is_null() {
            unsafe { (self.onnx.release_session)(self.session) };
        }
    }
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
        let onnx = Onnx::new(17).unwrap();
        let result = onnx.create_session(
            &Executor::Cpu,
            &OptimizationLevel::Disabled,
            1,
            "/nonexistent/model.onnx",
        );
        assert!(result.is_err(), "Should error for non-existent file");
    }
}
