// Whisper model architecture for speech recognition.
//
// Ported from candle-transformers (Apache-2.0/MIT)
// https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/whisper/model.rs

use {
    super::{attention::ResidualAttentionBlock, config::Config},
    candle_core::{Module, Result, Tensor},
    candle_nn::{layer_norm, Conv1d, Conv1dConfig, LayerNorm, VarBuilder},
};

// Helper: 1D convolution with configurable stride
fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let config = Conv1dConfig {
        padding: (kernel_size - 1) / 2,
        stride,
        dilation: 1,
        groups: 1,
        cudnn_fwd_algo: None,
    };
    candle_nn::conv1d(in_channels, out_channels, kernel_size, config, vb)
}

#[derive(Debug, Clone)]
pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
}

impl AudioEncoder {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let n_state = cfg.d_model;
        let n_head = cfg.encoder_attention_heads;
        let n_ctx = cfg.max_source_positions;

        let conv1 = conv1d(cfg.num_mel_bins, n_state, 3, 1, vb.pp("conv1"))?;
        let conv2 = conv1d(n_state, n_state, 3, 2, vb.pp("conv2"))?; // stride=2 for downsampling

        let positional_embedding = vb.get((n_ctx, n_state), "embed_positions.weight")?;

        let mut blocks = Vec::with_capacity(cfg.encoder_layers);
        let vb_b = vb.pp("layers");
        for i in 0..cfg.encoder_layers {
            let block =
                ResidualAttentionBlock::load(vb_b.pp(&i.to_string()), n_state, n_head, false)?;
            blocks.push(block);
        }

        let ln_post = layer_norm(n_state, 1e-5, vb.pp("layer_norm"))?;

        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
        })
    }

    pub fn forward(&self, x: &Tensor, flush: bool) -> Result<Tensor> {
        let x = self.conv1.forward(x)?;
        let x = x.gelu()?;
        let x = self.conv2.forward(&x)?;
        let x = x.gelu()?;

        let x = x.transpose(1, 2)?;
        let (_b_sz, seq_len, _hidden) = x.dims3()?;
        let positional_embedding = if flush {
            self.positional_embedding.narrow(0, 0, seq_len)?
        } else {
            self.positional_embedding.clone()
        };
        let x = x.broadcast_add(&positional_embedding)?;

        let mut x = x;
        for block in &self.blocks {
            x = block.forward_encoder(&x)?;
        }

        self.ln_post.forward(&x)
    }
}

#[derive(Debug, Clone)]
pub struct TextDecoder {
    token_embedding: candle_nn::Embedding,
    positional_embedding: Tensor,
    pub(crate) blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    mask: Tensor,
}

impl TextDecoder {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let n_state = cfg.d_model;
        let n_head = cfg.decoder_attention_heads;
        let n_ctx = cfg.max_target_positions;

        let token_embedding = candle_nn::embedding(cfg.vocab_size, n_state, vb.pp("embed_tokens"))?;
        let positional_embedding = vb.get((n_ctx, n_state), "embed_positions.weight")?;

        let mut blocks = Vec::with_capacity(cfg.decoder_layers);
        let vb_b = vb.pp("layers");
        for i in 0..cfg.decoder_layers {
            let block =
                ResidualAttentionBlock::load(vb_b.pp(&i.to_string()), n_state, n_head, true)?;
            blocks.push(block);
        }

        let ln = layer_norm(n_state, 1e-5, vb.pp("layer_norm"))?;

        let mask: Vec<_> = (0..n_ctx)
            .flat_map(|i| (0..n_ctx).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_vec(mask, (n_ctx, n_ctx), vb.device())?;

        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
        })
    }

    pub fn forward(&mut self, tokens: &Tensor, xa: &Tensor) -> Result<Tensor> {
        let (_b_sz, seq_len) = tokens.dims2()?;
        let token_embedding = self.token_embedding.forward(tokens)?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, seq_len)?;
        let x = token_embedding.broadcast_add(&positional_embedding)?;

        let mask = self.mask.narrow(0, 0, seq_len)?.narrow(1, 0, seq_len)?;

        let mut x = x;
        for block in &mut self.blocks {
            x = block.forward(&x, Some(xa), Some(&mask))?;
        }

        let x = self.ln.forward(&x)?;
        // Weight tying: reuse token_embedding weights for final projection
        // Reshape from [batch, seq, hidden] to [batch*seq, hidden] for matmul
        let (b_sz, seq_len, hidden) = x.dims3()?;
        let x_flat = x.reshape((b_sz * seq_len, hidden))?;
        let logits = x_flat.matmul(&self.token_embedding.embeddings().t()?)?;
        logits.reshape((b_sz, seq_len, ()))
    }
}

#[derive(Debug, Clone)]
pub struct Whisper {
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
    pub config: Config,
}

impl Whisper {
    pub fn load(vb: VarBuilder, config: Config) -> Result<Self> {
        let encoder = AudioEncoder::load(vb.pp("model.encoder"), &config)?;
        let decoder = TextDecoder::load(vb.pp("model.decoder"), &config)?;
        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    pub fn forward(&mut self, mel: &Tensor, tokens: &Tensor) -> Result<Tensor> {
        let encoder_output = self.encoder.forward(mel, true)?;
        self.decoder.forward(tokens, &encoder_output)
    }

    pub(crate) fn encoder_forward(&self, mel: &Tensor) -> Result<Tensor> {
        self.encoder.forward(mel, true)
    }

    pub(crate) fn decoder_forward(&mut self, tokens: &Tensor, xa: &Tensor) -> Result<Tensor> {
        self.decoder.forward(tokens, xa)
    }

    pub(crate) fn reset_kv_cache(&mut self) {
        for block in &mut self.decoder.blocks {
            block.attn.reset_kv_cache();
            if let Some((cross_attn, _)) = &mut block.cross_attn {
                cross_attn.reset_kv_cache();
            }
        }
    }
}
