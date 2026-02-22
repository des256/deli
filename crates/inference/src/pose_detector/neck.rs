// YOLOv8 Feature Pyramid Network (FPN) neck
// Ported from HuggingFace candle-wasm-examples/yolo
// Source: https://github.com/huggingface/candle/tree/main/candle-wasm-examples/yolo
// License: Apache-2.0 OR MIT

use {super::{C2f, ConvBlock, Upsample}, crate::pose_detector::Multiples, candle_core::{Result, Tensor}, candle_nn::{Module, VarBuilder}};

/// YOLOv8 neck with Feature Pyramid Network
#[derive(Debug)]
pub struct YoloV8Neck {
    // Top-down path
    up1: Upsample,
    n1: C2f,      // fpn.n1: P5+P4 fusion
    up2: Upsample,
    n2: C2f,      // fpn.n2: P4+P3 fusion

    // Bottom-up path
    n3: ConvBlock, // fpn.n3: P3 downsample
    n4: C2f,       // fpn.n4: P3+P4 fusion
    n5: ConvBlock, // fpn.n5: P4 downsample
    n6: C2f,       // fpn.n6: P4+P5 fusion
}

impl YoloV8Neck {
    pub fn load(vb: VarBuilder, m: &Multiples) -> Result<Self> {
        let [c3, c4, c5] = m.filters();

        // Top-down path: P5 -> P4 -> P3
        let up1 = Upsample::load(vb.pp("up1"), 2)?;
        let n1 = C2f::load(vb.pp("n1"), c5 + c4, c4, m.n_short(), false, 0.5)?;

        let up2 = Upsample::load(vb.pp("up2"), 2)?;
        let n2 = C2f::load(vb.pp("n2"), c4 + c3, c3, m.n_short(), false, 0.5)?;

        // Bottom-up path: P3 -> P4 -> P5
        let n3 = ConvBlock::load(vb.pp("n3"), c3, c3, 3, 2, 1)?;
        let n4 = C2f::load(vb.pp("n4"), c3 + c4, c4, m.n_short(), false, 0.5)?;

        let n5 = ConvBlock::load(vb.pp("n5"), c4, c4, 3, 2, 1)?;
        let n6 = C2f::load(vb.pp("n6"), c4 + c5, c5, m.n_short(), false, 0.5)?;

        Ok(Self { up1, n1, up2, n2, n3, n4, n5, n6 })
    }

    /// Forward pass through FPN, returns features at 3 scales
    pub fn forward(&self, p3: &Tensor, p4: &Tensor, p5: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // Top-down: P5 -> P4
        let p5_up = self.up1.forward(p5)?;
        let p4_fused = Tensor::cat(&[&p5_up, p4], 1)?;
        let p4_out = self.n1.forward(&p4_fused)?;

        // Top-down: P4 -> P3
        let p4_up = self.up2.forward(&p4_out)?;
        let p3_fused = Tensor::cat(&[&p4_up, p3], 1)?;
        let p3_out = self.n2.forward(&p3_fused)?;

        // Bottom-up: P3 -> P4
        let p3_down = self.n3.forward(&p3_out)?;
        let p4_fused = Tensor::cat(&[&p3_down, &p4_out], 1)?;
        let p4_final = self.n4.forward(&p4_fused)?;

        // Bottom-up: P4 -> P5
        let p4_down = self.n5.forward(&p4_final)?;
        let p5_fused = Tensor::cat(&[&p4_down, p5], 1)?;
        let p5_final = self.n6.forward(&p5_fused)?;

        Ok((p3_out, p4_final, p5_final))
    }
}
