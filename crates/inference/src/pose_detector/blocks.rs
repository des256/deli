// YOLOv8 neural network building blocks
// Ported from HuggingFace candle-wasm-examples/yolo
// Source: https://github.com/huggingface/candle/tree/main/candle-wasm-examples/yolo
// License: Apache-2.0 OR MIT

use candle_core::{Result, Tensor};
use candle_nn::{batch_norm, conv2d_no_bias, Conv2dConfig, Module, VarBuilder};

/// Convolution block: Conv2d + BatchNorm + SiLU activation
#[derive(Debug)]
pub struct ConvBlock {
    conv: candle_nn::Conv2d,
    bn: candle_nn::BatchNorm,
}

impl ConvBlock {
    pub fn load(
        vb: VarBuilder,
        c1: usize,
        c2: usize,
        k: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let conv = conv2d_no_bias(
            c1,
            c2,
            k,
            Conv2dConfig {
                stride,
                padding,
                groups: 1,
                dilation: 1,
                ..Default::default()
            },
            vb.pp("conv"),
        )?;
        let bn = batch_norm(c2, 1e-3, vb.pp("bn"))?;
        Ok(Self { conv, bn })
    }
}

impl Module for ConvBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.conv)?.apply_t(&self.bn, false)?.silu()
    }
}

/// Bottleneck block with optional residual connection
#[derive(Debug)]
pub struct Bottleneck {
    cv1: ConvBlock,
    cv2: ConvBlock,
    add: bool,
}

impl Bottleneck {
    pub fn load(
        vb: VarBuilder,
        c1: usize,
        c2: usize,
        shortcut: bool,
        expansion: f64,
    ) -> Result<Self> {
        let c_ = (c2 as f64 * expansion) as usize;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c1, c_, 3, 1, 1)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), c_, c2, 3, 1, 1)?;
        let add = shortcut && c1 == c2;
        Ok(Self { cv1, cv2, add })
    }
}

impl Module for Bottleneck {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = xs.apply(&self.cv1)?.apply(&self.cv2)?;
        if self.add {
            out + xs
        } else {
            Ok(out)
        }
    }
}

/// Cross Stage Partial bottleneck with 2 convolutions (C2f)
#[derive(Debug)]
pub struct C2f {
    cv1: ConvBlock,
    cv2: ConvBlock,
    bottlenecks: Vec<Bottleneck>,
}

impl C2f {
    pub fn load(
        vb: VarBuilder,
        c1: usize,
        c2: usize,
        n: usize,
        shortcut: bool,
        expansion: f64,
    ) -> Result<Self> {
        let c_ = (c2 as f64 * expansion) as usize;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c1, 2 * c_, 1, 1, 0)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), (2 + n) * c_, c2, 1, 1, 0)?;
        let mut bottlenecks = Vec::with_capacity(n);
        for i in 0..n {
            bottlenecks.push(Bottleneck::load(
                vb.pp(format!("bottleneck.{i}")),
                c_,
                c_,
                shortcut,
                1.0,
            )?);
        }
        Ok(Self {
            cv1,
            cv2,
            bottlenecks,
        })
    }
}

impl Module for C2f {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.cv1)?;
        let xs = xs.chunk(2, 1)?;
        let mut ys = vec![xs[0].clone(), xs[1].clone()];
        for bottleneck in &self.bottlenecks {
            ys.push(ys.last().unwrap().apply(bottleneck)?);
        }
        Tensor::cat(&ys, 1)?.apply(&self.cv2)
    }
}

/// Spatial Pyramid Pooling - Fast (SPPF)
#[derive(Debug)]
pub struct Sppf {
    cv1: ConvBlock,
    cv2: ConvBlock,
    k: usize,
}

impl Sppf {
    pub fn load(vb: VarBuilder, c1: usize, c2: usize, k: usize) -> Result<Self> {
        let c_ = c1 / 2;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c1, c_, 1, 1, 0)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), c_ * 4, c2, 1, 1, 0)?;
        Ok(Self { cv1, cv2, k })
    }
}

impl Module for Sppf {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.cv1)?;
        // Pad manually to maintain spatial dimensions with stride=1
        let pad = self.k / 2;
        let y1 = xs.pad_with_zeros(2, pad, pad)?.pad_with_zeros(3, pad, pad)?;
        let y1 = y1.max_pool2d_with_stride(self.k, 1)?;
        let y2 = y1.pad_with_zeros(2, pad, pad)?.pad_with_zeros(3, pad, pad)?;
        let y2 = y2.max_pool2d_with_stride(self.k, 1)?;
        let y3 = y2.pad_with_zeros(2, pad, pad)?.pad_with_zeros(3, pad, pad)?;
        let y3 = y3.max_pool2d_with_stride(self.k, 1)?;
        Tensor::cat(&[&xs, &y1, &y2, &y3], 1)?.apply(&self.cv2)
    }
}

/// Upsample layer using nearest neighbor interpolation
#[derive(Debug)]
pub struct Upsample {
    scale_factor: usize,
}

impl Upsample {
    pub fn load(_vb: VarBuilder, scale_factor: usize) -> Result<Self> {
        Ok(Self { scale_factor })
    }
}

impl Module for Upsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = xs.dims4()?;
        xs.upsample_nearest2d(h * self.scale_factor, w * self.scale_factor)
    }
}

/// Distribution Focal Loss (DFL) for bounding box regression
#[derive(Debug)]
pub struct Dfl {
    conv: candle_nn::Conv2d,
    c1: usize,
}

impl Dfl {
    pub fn load(vb: VarBuilder, c1: usize) -> Result<Self> {
        let conv = conv2d_no_bias(
            c1,
            1,
            1,
            Conv2dConfig {
                stride: 1,
                padding: 0,
                groups: 1,
                dilation: 1,
                ..Default::default()
            },
            vb.pp("conv"),
        )?;
        Ok(Self { conv, c1 })
    }
}

impl Module for Dfl {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, _c, a) = xs.dims3()?;
        // Reshape: [b, 4*c1, a] -> [b*4, c1, a]
        let xs = xs.reshape((b * 4, self.c1, a))?;
        let xs = candle_nn::ops::softmax(&xs, 1)?;
        // Add spatial dimensions for conv2d: [b*4, c1, a] -> [b*4, c1, a, 1]
        let xs = xs.unsqueeze(3)?;
        // Apply conv: [b*4, c1, a, 1] -> [b*4, 1, a, 1]
        let xs = xs.apply(&self.conv)?;
        // Remove extra dimensions: [b*4, 1, a, 1] -> [b*4, a]
        let xs = xs.squeeze(1)?.squeeze(2)?;
        // Reshape back: [b*4, a] -> [b, 4, a]
        xs.reshape((b, 4, a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn test_conv_block_output_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let block = ConvBlock::load(vb.pp("conv"), 3, 32, 3, 1, 1).unwrap();
        let input = Tensor::zeros(&[1, 3, 64, 64], DType::F32, &device).unwrap();
        let output = block.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 32, 64, 64]);
    }

    #[test]
    fn test_conv_block_stride_reduces_spatial() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let block = ConvBlock::load(vb.pp("conv"), 3, 32, 3, 2, 1).unwrap();
        let input = Tensor::zeros(&[1, 3, 64, 64], DType::F32, &device).unwrap();
        let output = block.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 32, 32, 32]);
    }

    #[test]
    fn test_bottleneck_output_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let block = Bottleneck::load(vb.pp("bottleneck"), 64, 64, true, 0.5).unwrap();
        let input = Tensor::zeros(&[1, 64, 32, 32], DType::F32, &device).unwrap();
        let output = block.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 64, 32, 32]);
    }

    #[test]
    fn test_c2f_output_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let block = C2f::load(vb.pp("c2f"), 64, 128, 2, false, 0.5).unwrap();
        let input = Tensor::zeros(&[1, 64, 32, 32], DType::F32, &device).unwrap();
        let output = block.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 128, 32, 32]);
    }

    #[test]
    fn test_sppf_output_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let block = Sppf::load(vb.pp("sppf"), 256, 256, 5).unwrap();
        let input = Tensor::zeros(&[1, 256, 32, 32], DType::F32, &device).unwrap();
        let output = block.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 256, 32, 32]);
    }

    #[test]
    fn test_upsample_doubles_spatial() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let block = Upsample::load(vb.pp("upsample"), 2).unwrap();
        let input = Tensor::zeros(&[1, 64, 16, 16], DType::F32, &device).unwrap();
        let output = block.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 64, 32, 32]);
    }

    #[test]
    fn test_dfl_output_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let block = Dfl::load(vb.pp("dfl"), 16).unwrap();
        let input = Tensor::zeros(&[1, 64, 8400], DType::F32, &device).unwrap();
        let output = block.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 4, 8400]);
    }
}
