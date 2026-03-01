use {
    crate::*,
    base::*,
    std::{fmt::Write, path::Path, sync::Arc},
};

const ONNX_VERSION: usize = 24;

#[derive(Debug)]
pub struct Inference {
    onnx: Arc<onnx::Onnx>,
}

impl Inference {
    pub fn new() -> Result<Self, InferError> {
        let onnx = match onnx::Onnx::new(ONNX_VERSION) {
            Ok(onnx) => onnx,
            Err(error) => return Err(InferError::Onnx(error.to_string())),
        };
        Ok(Self { onnx })
    }

    pub fn use_parakeet<T: Clone + Send + 'static>(
        &self,
        executor: &onnx::Executor,
    ) -> Result<
        (
            crate::asr::parakeet::ParakeetHandle<T>,
            crate::asr::parakeet::ParakeetListener<T>,
        ),
        InferError,
    > {
        crate::asr::parakeet::create(&self.onnx, &executor)
    }

    pub fn use_pocket<T: Clone + Send + 'static>(
        &self,
        executor: &onnx::Executor,
        voice_path: impl AsRef<Path>,
        epoch: Epoch,
    ) -> Result<
        (
            crate::tts::pocket::PocketHandle<T>,
            crate::tts::pocket::PocketListener<T>,
        ),
        InferError,
    > {
        crate::tts::pocket::create(&self.onnx, &executor, voice_path.as_ref(), epoch)
    }

    pub fn use_phi3<T: Clone + Send + 'static>(
        &self,
        executor: &onnx::Executor,
        epoch: Epoch,
    ) -> Result<
        (
            crate::llm::phi3::Phi3Handle<T>,
            crate::llm::phi3::Phi3Listener<T>,
        ),
        InferError,
    > {
        crate::llm::phi3::create(&self.onnx, executor, epoch)
    }

    pub fn use_llama3_3b<T: Clone + Send + 'static>(
        &self,
        executor: &onnx::Executor,
        epoch: Epoch,
    ) -> Result<
        (
            crate::llm::llama3::Llama3Handle<T>,
            crate::llm::llama3::Llama3Listener<T>,
        ),
        InferError,
    > {
        crate::llm::llama3::create(
            &self.onnx,
            executor,
            epoch,
            crate::llm::llama3::Llama3Flavor::ThreeB,
        )
    }

    pub fn use_llama3_8b<T: Clone + Send + 'static>(
        &self,
        executor: &onnx::Executor,
        epoch: Epoch,
    ) -> Result<
        (
            crate::llm::llama3::Llama3Handle<T>,
            crate::llm::llama3::Llama3Listener<T>,
        ),
        InferError,
    > {
        crate::llm::llama3::create(
            &self.onnx,
            executor,
            epoch,
            crate::llm::llama3::Llama3Flavor::EightB,
        )
    }

    pub fn use_gemma3<T: Clone + Send + 'static>(
        &self,
        executor: &onnx::Executor,
        epoch: Epoch,
    ) -> Result<
        (
            crate::llm::gemma3::Gemma3Handle<T>,
            crate::llm::gemma3::Gemma3Listener<T>,
        ),
        InferError,
    > {
        crate::llm::gemma3::create(&self.onnx, executor, epoch)
    }

    pub fn use_silero(
        &self,
        executor: &onnx::Executor,
        sample_rate: usize,
    ) -> Result<crate::vad::Silero, InferError> {
        crate::vad::Silero::new(&self.onnx, &executor, sample_rate)
    }

    pub fn use_parakeet_diar(
        &self,
        executor: &onnx::Executor,
    ) -> Result<crate::diar::parakeet::Sortformer, InferError> {
        crate::diar::parakeet::Sortformer::new(&self.onnx, executor)
    }

    pub fn onnx_session(&self, model_path: impl AsRef<Path>) -> Result<onnx::Session, InferError> {
        self.onnx
            .create_session(
                &onnx::Executor::Cpu,
                &onnx::OptimizationLevel::EnableAll,
                4,
                model_path.as_ref(),
            )
            .map_err(|e| InferError::Onnx(e.to_string()))
    }

    pub fn device(&self) -> &onnx::Executor {
        &onnx::Executor::Cpu
    }

    /// Returns a human-readable summary of memory usage.
    /// Parses /proc/meminfo for system RAM and NvMapMemUsed (Jetson GPU-mapped memory).
    /// When compiled with the `cuda` feature, also reports CUDA free/total memory.
    pub fn mem_info(&self) -> String {
        let mut out = String::new();

        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            let mut mem_total_kb = 0u64;
            let mut mem_available_kb = 0u64;
            let mut nvmap_kb = 0u64;

            for line in contents.lines() {
                if let Some(val) = parse_meminfo_kb(line, "MemTotal:") {
                    mem_total_kb = val;
                } else if let Some(val) = parse_meminfo_kb(line, "MemAvailable:") {
                    mem_available_kb = val;
                } else if let Some(val) = parse_meminfo_kb(line, "NvMapMemUsed:") {
                    nvmap_kb = val;
                }
            }

            let used_mb = (mem_total_kb.saturating_sub(mem_available_kb)) / 1024;
            let total_mb = mem_total_kb / 1024;
            let _ = write!(out, "RAM: {}/{} MB used", used_mb, total_mb);

            if nvmap_kb > 0 {
                let _ = write!(out, ", NvMap: {} MB", nvmap_kb / 1024);
            }
        }

        #[cfg(feature = "cuda")]
        {
            if let Some((free, total)) = cuda_mem_get_info() {
                let used_mb = total.saturating_sub(free) / (1024 * 1024);
                let total_mb = total / (1024 * 1024);
                if !out.is_empty() {
                    out.push_str(" | ");
                }
                let _ = write!(out, "CUDA: {}/{} MB used", used_mb, total_mb);
            }
        }

        out
    }
}

/// Parse a line like "MemTotal:       16384000 kB" into the numeric value in kB.
fn parse_meminfo_kb(line: &str, prefix: &str) -> Option<u64> {
    let rest = line.strip_prefix(prefix)?;
    rest.trim().strip_suffix("kB")?.trim().parse().ok()
}

#[cfg(feature = "cuda")]
fn cuda_mem_get_info() -> Option<(usize, usize)> {
    #[link(name = "cuda")]
    unsafe extern "C" {
        fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> i32;
    }

    let mut free = 0usize;
    let mut total = 0usize;
    let result = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
    if result == 0 {
        Some((free, total))
    } else {
        None
    }
}
