// YOLOv8 DarkNet backbone
// Ported from HuggingFace candle-wasm-examples/yolo
// Source: https://github.com/huggingface/candle/tree/main/candle-wasm-examples/yolo
// License: Apache-2.0 OR MIT

use super::{C2f, ConvBlock, Sppf};
use crate::model::Multiples;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// DarkNet backbone - extracts features at 3 scales
#[derive(Debug)]
pub struct DarkNet {
    b1_0: ConvBlock,
    b1_1: ConvBlock,
    b2_0: C2f,
    b2_1: ConvBlock,
    b2_2: C2f,
    b3_0: ConvBlock,
    b3_1: C2f,
    b4_0: ConvBlock,
    b4_1: C2f,
    b5_0: Sppf,
}

impl DarkNet {
    pub fn load(vb: VarBuilder, m: &Multiples) -> Result<Self> {
        let w = m.width;
        let c1 = (64.0 * w) as usize;
        let c2 = (128.0 * w) as usize;
        let [c3, c4, c5] = m.filters();

        // Stage 1: Two initial convs (/2 then /4)
        let b1_0 = ConvBlock::load(vb.pp("b1.0"), 3, c1, 3, 2, 1)?;
        let b1_1 = ConvBlock::load(vb.pp("b1.1"), c1, c2, 3, 2, 1)?;

        // Stage 2: C2f -> downsample -> C2f
        let b2_0 = C2f::load(vb.pp("b2.0"), c2, c2, m.n_short(), true, 0.5)?;
        let b2_1 = ConvBlock::load(vb.pp("b2.1"), c2, c3, 3, 2, 1)?;
        let b2_2 = C2f::load(vb.pp("b2.2"), c3, c3, m.n_long(), true, 0.5)?;

        // Stage 3: downsample -> C2f (P4)
        let b3_0 = ConvBlock::load(vb.pp("b3.0"), c3, c4, 3, 2, 1)?;
        let b3_1 = C2f::load(vb.pp("b3.1"), c4, c4, m.n_long(), true, 0.5)?;

        // Stage 4: downsample -> C2f (P5 before SPPF)
        let b4_0 = ConvBlock::load(vb.pp("b4.0"), c4, c5, 3, 2, 1)?;
        let b4_1 = C2f::load(vb.pp("b4.1"), c5, c5, m.n_short(), true, 0.5)?;

        // Stage 5: SPPF
        let b5_0 = Sppf::load(vb.pp("b5.0"), c5, c5, 5)?;

        Ok(Self {
            b1_0, b1_1,
            b2_0, b2_1, b2_2,
            b3_0, b3_1,
            b4_0, b4_1,
            b5_0,
        })
    }
}

impl DarkNet {
    pub fn forward_features(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        use candle_nn::Module;

        let xs = xs.apply(&self.b1_0)?.apply(&self.b1_1)?;

        // P3: after b2.2
        let xs = xs.apply(&self.b2_0)?;
        let p3 = self.b2_1.forward(&xs)?.apply(&self.b2_2)?;

        // P4: after b3.1
        let p4 = self.b3_0.forward(&p3)?.apply(&self.b3_1)?;

        // P5: after b5.0 (SPPF)
        let p5 = self.b4_0.forward(&p4)?.apply(&self.b4_1)?;
        let p5 = self.b5_0.forward(&p5)?;

        Ok((p3, p4, p5))
    }
}
