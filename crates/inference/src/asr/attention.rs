// Multi-head attention and residual attention blocks for Whisper.
//
// Ported from candle-transformers (Apache-2.0/MIT)
// https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/whisper/model.rs

use {candle_core::{Module, Result, Tensor}, candle_nn::{layer_norm, linear, linear_no_bias, LayerNorm, Linear, VarBuilder}};

#[derive(Debug, Clone)]
pub(crate) struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl MultiHeadAttention {
    pub(crate) fn load(vb: VarBuilder, n_state: usize, n_head: usize) -> Result<Self> {
        let query = linear(n_state, n_state, vb.pp("q_proj"))?;
        let value = linear(n_state, n_state, vb.pp("v_proj"))?;
        let key = linear_no_bias(n_state, n_state, vb.pp("k_proj"))?;
        let out = linear(n_state, n_state, vb.pp("out_proj"))?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
            kv_cache: None,
        })
    }

    pub(crate) fn forward(
        &mut self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.query.forward(x)?;
        let (k, v) = match xa {
            None => {
                let k = self.key.forward(x)?;
                let v = self.value.forward(x)?;
                (k, v)
            }
            Some(xa) => {
                if let Some((k, v)) = &self.kv_cache {
                    (k.clone(), v.clone())
                } else {
                    let k = self.key.forward(xa)?;
                    let v = self.value.forward(xa)?;
                    self.kv_cache = Some((k.clone(), v.clone()));
                    (k, v)
                }
            }
        };

        self.qkv_attention(&q, &k, &v, b_sz, seq_len, mask)
    }

    /// Forward without KV cache mutation — for encoder self-attention
    pub(crate) fn forward_no_cache(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;
        self.qkv_attention(&q, &k, &v, b_sz, seq_len, mask)
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        b_sz: usize,
        seq_len: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let q = q
            .reshape((b_sz, seq_len, self.n_head, ()))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, (), self.n_head, q.dim(3)?))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, (), self.n_head, q.dim(3)?))?
            .transpose(1, 2)?
            .contiguous()?;

        let n_state = q.dim(3)?;
        let scale = (n_state as f64).powf(-0.25);
        let q = (q * scale)?;
        let k = (k * scale)?;

        let att = q.matmul(&k.t()?)?;
        let att = match mask {
            None => att,
            Some(mask) => att.broadcast_add(mask)?,
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?.transpose(1, 2)?.flatten_from(2)?;
        self.out.forward(&out)
    }

    pub(crate) fn reset_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ResidualAttentionBlock {
    pub(crate) attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    pub(crate) cross_attn: Option<(MultiHeadAttention, LayerNorm)>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
}

impl ResidualAttentionBlock {
    pub(crate) fn load(
        vb: VarBuilder,
        n_state: usize,
        n_head: usize,
        ca: bool,
    ) -> Result<Self> {
        let attn = MultiHeadAttention::load(vb.pp("self_attn"), n_state, n_head)?;
        let attn_ln = layer_norm(n_state, 1e-5, vb.pp("self_attn_layer_norm"))?;

        let cross_attn = if ca {
            let cross_attn = MultiHeadAttention::load(vb.pp("encoder_attn"), n_state, n_head)?;
            let cross_attn_ln = layer_norm(n_state, 1e-5, vb.pp("encoder_attn_layer_norm"))?;
            Some((cross_attn, cross_attn_ln))
        } else {
            None
        };

        let n_mlp = n_state * 4;
        let mlp_linear1 = linear(n_state, n_mlp, vb.pp("fc1"))?;
        let mlp_linear2 = linear(n_mlp, n_state, vb.pp("fc2"))?;
        let mlp_ln = layer_norm(n_state, 1e-5, vb.pp("final_layer_norm"))?;

        Ok(Self {
            attn,
            attn_ln,
            cross_attn,
            mlp_linear1,
            mlp_linear2,
            mlp_ln,
        })
    }

    pub(crate) fn forward(
        &mut self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let attn_out = self.attn.forward(&self.attn_ln.forward(x)?, None, mask)?;
        let mut x = (x + attn_out)?;

        if let Some((cross_attn, cross_attn_ln)) = &mut self.cross_attn {
            let cross_out = cross_attn.forward(&cross_attn_ln.forward(&x)?, xa, None)?;
            x = (x + cross_out)?;
        }

        self.mlp_forward(&x)
    }

    /// Forward for encoder blocks — no KV cache mutation, no cross-attention
    pub(crate) fn forward_encoder(&self, x: &Tensor) -> Result<Tensor> {
        let attn_out = self.attn.forward_no_cache(&self.attn_ln.forward(x)?, None)?;
        let x = (x + attn_out)?;
        self.mlp_forward(&x)
    }

    fn mlp_forward(&self, x: &Tensor) -> Result<Tensor> {
        let mlp_out = self.mlp_linear1.forward(&self.mlp_ln.forward(x)?)?;
        let mlp_out = mlp_out.gelu()?;
        let mlp_out = self.mlp_linear2.forward(&mlp_out)?;
        x + mlp_out
    }
}
