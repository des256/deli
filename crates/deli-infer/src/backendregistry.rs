use crate::Backend;
use std::collections::HashMap;

pub struct BackendRegistry {
    backends: HashMap<String, Box<dyn Backend>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
        }
    }

    pub fn register(&mut self, backend: Box<dyn Backend>) {
        let name = backend.name().to_string();
        self.backends.insert(name, backend);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Backend> {
        self.backends.get(name).map(|b| &**b as &dyn Backend)
    }

    pub fn list(&self) -> Vec<&str> {
        self.backends.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn create_registry() -> BackendRegistry {
    #[cfg(feature = "onnx")]
    let mut registry = BackendRegistry::new();
    #[cfg(not(feature = "onnx"))]
    let registry = BackendRegistry::new();

    #[cfg(feature = "onnx")]
    {
        use crate::backends::OnnxBackend;
        registry.register(Box::new(OnnxBackend));
    }

    registry
}
