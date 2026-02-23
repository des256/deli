use crate::diar::parakeet::compression;
use crate::diar::parakeet::features::compute_mel_features;
use crate::diar::parakeet::postprocess;
use crate::diar::parakeet::{DiarizationConfig, SpeakerSegment};
use crate::error::{InferError, Result};
use onnx::Session;

// Constants from reference implementation
const CHUNK_LEN: usize = 124;
const FIFO_LEN: usize = 124;
const SPKCACHE_LEN: usize = 188;
const SUBSAMPLING: usize = 8;
const EMB_DIM: usize = 512;
const NUM_SPEAKERS: usize = 4;
const SIL_THRESHOLD: f32 = 0.2;
const SPKCACHE_SIL_FRAMES_PER_SPK: usize = 3;

/// Sortformer streaming speaker diarization.
pub struct Sortformer {
    session: Session,
    config: DiarizationConfig,
    // Streaming constants
    chunk_len: usize,
    fifo_len: usize,
    spkcache_len: usize,
    // Streaming state
    spkcache: Vec<f32>,               // [T_cache, EMB_DIM] flattened
    fifo: Vec<f32>,                   // [T_fifo, EMB_DIM] flattened
    fifo_preds: Vec<f32>,             // [T_fifo, NUM_SPEAKERS] flattened
    spkcache_preds: Option<Vec<f32>>, // [T_cache, NUM_SPEAKERS] flattened
    mean_sil_emb: Vec<f32>,           // [EMB_DIM]
    n_sil_frames: usize,
}

impl Sortformer {
    /// Create a new Sortformer with zero-initialized streaming state.
    pub fn new(session: Session, config: DiarizationConfig) -> Result<Self> {
        Ok(Self {
            session,
            config,
            chunk_len: CHUNK_LEN,
            fifo_len: FIFO_LEN,
            spkcache_len: SPKCACHE_LEN,
            spkcache: Vec::new(),
            fifo: Vec::new(),
            fifo_preds: Vec::new(),
            spkcache_preds: None,
            mean_sil_emb: vec![0.0; EMB_DIM],
            n_sil_frames: 0,
        })
    }

    /// Reset all streaming state to zero.
    pub fn reset(&mut self) -> Result<()> {
        self.spkcache.clear();
        self.fifo.clear();
        self.fifo_preds.clear();
        self.spkcache_preds = None;
        self.mean_sil_emb.fill(0.0);
        self.n_sil_frames = 0;
        Ok(())
    }

    /// Process a chunk of mel features through the streaming Sortformer model.
    ///
    /// # Arguments
    /// - `chunk_feat`: Mel features [chunk_feat_frames, 128] flattened
    /// - `chunk_feat_frames`: Number of mel feature frames (may include padding)
    /// - `actual_len`: Actual mel feature frames before padding
    ///
    /// # Returns
    /// Predictions for valid chunk frames [valid_frames, NUM_SPEAKERS] flattened
    pub fn streaming_update(
        &mut self,
        chunk_feat: &[f32],
        chunk_feat_frames: usize,
        actual_len: usize,
    ) -> Result<Vec<f32>> {
        if chunk_feat.len() != chunk_feat_frames * 128 {
            return Err(InferError::Runtime(format!(
                "Chunk features length {} does not match frames {} * 128",
                chunk_feat.len(),
                chunk_feat_frames
            )));
        }

        // Valid output frames after model subsampling
        let valid_chunk_frames = actual_len.div_ceil(SUBSAMPLING);

        // Prepare ONNX inputs
        let chunk_lengths = vec![actual_len as i64];

        let chunk = onnx::Value::from_slice(&[1, chunk_feat_frames, 128], chunk_feat)
            .map_err(|e| InferError::Runtime(format!("Failed to create chunk tensor: {}", e)))?;

        let chunk_lengths_tensor = onnx::Value::from_slice(&[1], &chunk_lengths)
            .map_err(|e| InferError::Runtime(format!("Failed to create chunk_lengths: {}", e)))?;

        let spkcache_frames = self.spkcache.len() / EMB_DIM;
        let spkcache = if spkcache_frames > 0 {
            onnx::Value::from_slice(&[1, spkcache_frames, EMB_DIM], &self.spkcache)
                .map_err(|e| InferError::Runtime(format!("Failed to create spkcache: {}", e)))?
        } else {
            onnx::Value::zeros::<f32>(&[1, 0, EMB_DIM as i64])
                .map_err(|e| InferError::Runtime(format!("Failed to create empty spkcache: {}", e)))?
        };

        let spkcache_lengths = vec![spkcache_frames as i64];
        let spkcache_lengths_tensor = onnx::Value::from_slice(&[1], &spkcache_lengths)
            .map_err(|e| InferError::Runtime(format!("Failed to create spkcache_lengths: {}", e)))?;

        let fifo_frames = self.fifo.len() / EMB_DIM;
        let fifo = if fifo_frames > 0 {
            onnx::Value::from_slice(&[1, fifo_frames, EMB_DIM], &self.fifo)
                .map_err(|e| InferError::Runtime(format!("Failed to create fifo: {}", e)))?
        } else {
            onnx::Value::zeros::<f32>(&[1, 0, EMB_DIM as i64])
                .map_err(|e| InferError::Runtime(format!("Failed to create empty fifo: {}", e)))?
        };

        let fifo_lengths = vec![fifo_frames as i64];
        let fifo_lengths_tensor = onnx::Value::from_slice(&[1], &fifo_lengths)
            .map_err(|e| InferError::Runtime(format!("Failed to create fifo_lengths: {}", e)))?;

        // Run ONNX model
        let outputs = self
            .session
            .run(
                &[
                    ("chunk", &chunk),
                    ("chunk_lengths", &chunk_lengths_tensor),
                    ("spkcache", &spkcache),
                    ("spkcache_lengths", &spkcache_lengths_tensor),
                    ("fifo", &fifo),
                    ("fifo_lengths", &fifo_lengths_tensor),
                ],
                &["spkcache_fifo_chunk_preds", "chunk_pre_encode_embs", "chunk_pre_encode_lengths"],
            )
            .map_err(|e| InferError::Runtime(format!("Sortformer inference failed: {}", e)))?;

        // Extract outputs
        let all_preds = outputs[0]
            .extract_tensor::<f32>()
            .map_err(|e| InferError::Runtime(format!("Failed to extract predictions: {}", e)))?
            .to_vec();

        let all_chunk_embs = outputs[1]
            .extract_tensor::<f32>()
            .map_err(|e| InferError::Runtime(format!("Failed to extract embeddings: {}", e)))?
            .to_vec();

        // Slice predictions: [spkcache_preds | fifo_preds | chunk_preds]
        // Only take valid_chunk_frames (exclude padding frames)
        let chunk_preds_start = (spkcache_frames + fifo_frames) * NUM_SPEAKERS;
        let chunk_preds_end = chunk_preds_start + valid_chunk_frames * NUM_SPEAKERS;
        let chunk_preds = all_preds[chunk_preds_start..chunk_preds_end].to_vec();

        // Slice chunk embeddings to valid frames only (exclude padding)
        let chunk_embs = &all_chunk_embs[..valid_chunk_frames * EMB_DIM];

        // Update fifo predictions with model's re-evaluated fifo preds + new chunk preds
        let fifo_preds_start = spkcache_frames * NUM_SPEAKERS;
        let fifo_preds_end = fifo_preds_start + fifo_frames * NUM_SPEAKERS;
        if fifo_frames > 0 {
            self.fifo_preds = all_preds[fifo_preds_start..fifo_preds_end].to_vec();
            self.fifo_preds.extend_from_slice(&chunk_preds);
        } else {
            self.fifo_preds = chunk_preds.clone();
        }

        // NOTE: Do NOT replace spkcache_preds from model output here.
        // spkcache_preds is only built up from overflow predictions (matching reference).

        // Append valid chunk embeddings to fifo
        self.fifo.extend_from_slice(chunk_embs);

        // Pop from fifo to spkcache when fifo exceeds limit
        let fifo_frames_after = self.fifo.len() / EMB_DIM;
        if fifo_frames_after > self.fifo_len {
            // Reference pops at least chunk_len frames (aggressive flush)
            let mut pop_out_len = self.chunk_len;
            pop_out_len = pop_out_len.max(
                valid_chunk_frames.saturating_sub(self.fifo_len) + fifo_frames,
            );
            pop_out_len = pop_out_len.min(fifo_frames_after);

            let pop_out_embs: Vec<f32> = self.fifo.drain(0..pop_out_len * EMB_DIM).collect();
            let pop_out_preds: Vec<f32> =
                self.fifo_preds.drain(0..pop_out_len * NUM_SPEAKERS).collect();

            // Update silence profile from popped frames
            for i in 0..pop_out_len {
                let pred_start = i * NUM_SPEAKERS;
                let pred_end = pred_start + NUM_SPEAKERS;
                let speaker_sum: f32 = pop_out_preds[pred_start..pred_end].iter().sum();

                if speaker_sum < SIL_THRESHOLD {
                    let emb_start = i * EMB_DIM;
                    let emb_end = emb_start + EMB_DIM;
                    for (j, &emb_val) in pop_out_embs[emb_start..emb_end].iter().enumerate() {
                        let old_mean = self.mean_sil_emb[j];
                        let n = self.n_sil_frames as f32;
                        self.mean_sil_emb[j] = (old_mean * n + emb_val) / (n + 1.0);
                    }
                    self.n_sil_frames += 1;
                }
            }

            // Append popped embeddings to spkcache
            self.spkcache.extend_from_slice(&pop_out_embs);

            // Extend spkcache_preds if it exists (otherwise stays None until needed)
            if let Some(ref mut cache_preds) = self.spkcache_preds {
                cache_preds.extend_from_slice(&pop_out_preds);
            }

            // Smart compression when cache exceeds limit
            if self.spkcache.len() / EMB_DIM > self.spkcache_len {
                if self.spkcache_preds.is_none() {
                    // First-time initialization: model's spkcache predictions + overflow preds
                    let mut initial_preds =
                        all_preds[0..spkcache_frames * NUM_SPEAKERS].to_vec();
                    initial_preds.extend_from_slice(&pop_out_preds);
                    self.spkcache_preds = Some(initial_preds);
                }
                self.compress_spkcache();
            }
        }

        Ok(chunk_preds)
    }

    /// Compress the speaker cache using NeMo-style smart compression.
    fn compress_spkcache(&mut self) {
        let cache_frames = self.spkcache.len() / EMB_DIM;
        if cache_frames == 0 {
            return;
        }

        let cache_preds = match &self.spkcache_preds {
            Some(preds) => preds.clone(),
            None => return,
        };

        let pred_frames = cache_preds.len() / NUM_SPEAKERS;
        let effective_frames = cache_frames.min(pred_frames);

        if effective_frames == 0 {
            return;
        }

        let mut scores = compression::get_log_pred_scores(&cache_preds, effective_frames);
        compression::disable_low_scores(&mut scores, effective_frames);
        compression::boost_topk_scores(&mut scores, effective_frames);

        let target_frames_per_speaker = self.spkcache_len / NUM_SPEAKERS;
        let selected_indices =
            compression::get_topk_indices(&scores, effective_frames, target_frames_per_speaker);

        self.gather_spkcache(&selected_indices);
    }

    /// Rebuild spkcache from selected indices plus silence frames.
    fn gather_spkcache(&mut self, selected_indices: &[usize]) {
        let mut new_cache = Vec::new();
        let mut new_cache_preds = Vec::new();

        let mut sorted_indices = selected_indices.to_vec();
        sorted_indices.sort_unstable();

        for &idx in &sorted_indices {
            let emb_start = idx * EMB_DIM;
            let emb_end = emb_start + EMB_DIM;

            if emb_end <= self.spkcache.len() {
                new_cache.extend_from_slice(&self.spkcache[emb_start..emb_end]);
            } else {
                continue;
            }

            if let Some(ref preds) = self.spkcache_preds {
                let pred_start = idx * NUM_SPEAKERS;
                let pred_end = pred_start + NUM_SPEAKERS;

                if pred_end <= preds.len() {
                    new_cache_preds.extend_from_slice(&preds[pred_start..pred_end]);
                } else {
                    new_cache_preds.extend_from_slice(&[0.0; NUM_SPEAKERS]);
                }
            }
        }

        // Add silence frames
        let num_sil_frames = SPKCACHE_SIL_FRAMES_PER_SPK * NUM_SPEAKERS;
        for _ in 0..num_sil_frames {
            new_cache.extend_from_slice(&self.mean_sil_emb);
            new_cache_preds.extend_from_slice(&[0.0; NUM_SPEAKERS]);
        }

        self.spkcache = new_cache;
        self.spkcache_preds = Some(new_cache_preds);
    }

    /// Diarize an audio chunk and return speaker segments.
    ///
    /// # Arguments
    /// - `audio_16k_mono`: Audio samples in f32 format, normalized to [-1, 1] range, at 16kHz
    ///
    /// # Returns
    /// Vec of SpeakerSegments with start/end times in seconds and speaker IDs
    pub fn diarize_chunk(&mut self, audio_16k_mono: &[f32]) -> Result<Vec<SpeakerSegment>> {
        if audio_16k_mono.is_empty() {
            return Ok(vec![]);
        }

        let (features, num_frames) = compute_mel_features(audio_16k_mono, 16000)?;

        // Each model chunk processes chunk_len * SUBSAMPLING mel frames
        let chunk_stride = self.chunk_len * SUBSAMPLING;
        let num_chunks = num_frames.div_ceil(chunk_stride);

        let mut all_preds = Vec::new();

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_stride;
            let end = (start + chunk_stride).min(num_frames);
            let current_len = end - start;

            let chunk_start = start * 128;
            let chunk_end = end * 128;
            let chunk_feat = &features[chunk_start..chunk_end];

            // Zero-pad if needed
            let (padded_feat, actual_len) = if current_len < chunk_stride {
                let mut padded = chunk_feat.to_vec();
                padded.resize(chunk_stride * 128, 0.0);
                (padded, current_len)
            } else {
                (chunk_feat.to_vec(), current_len)
            };

            let chunk_preds = self.streaming_update(&padded_feat, chunk_stride, actual_len)?;
            all_preds.extend_from_slice(&chunk_preds);
        }

        let total_out_frames = all_preds.len() / NUM_SPEAKERS;
        let smoothed = postprocess::median_filter(&all_preds, total_out_frames, 11);
        let segments = postprocess::binarize(&smoothed, total_out_frames, &self.config);

        Ok(segments)
    }
}
