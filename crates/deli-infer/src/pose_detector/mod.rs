// YOLOv8-Pose model architecture
// Ported from HuggingFace candle-wasm-examples/yolo
// Source: https://github.com/huggingface/candle/tree/main/candle-wasm-examples/yolo
// License: Apache-2.0 OR MIT

mod backbone;
mod blocks;
mod detector;
mod head;
mod neck;
mod postprocess;
mod types;

pub use blocks::{C2f, ConvBlock, Dfl, Sppf, Upsample};
pub use detector::PoseDetector;
pub use types::{CocoKeypoint, Keypoint, PoseDetection, PoseDetections};

use backbone::DarkNet;
use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};
use head::PoseHead;
use neck::YoloV8Neck;

/// Model size configuration (width, depth, and ratio multipliers)
#[derive(Debug, Clone, Copy)]
pub struct Multiples {
    pub depth: f64,
    pub width: f64,
    pub ratio: f64,
}

impl Multiples {
    pub fn n() -> Self { Self { depth: 0.33, width: 0.25, ratio: 2.0 } }
    pub fn s() -> Self { Self { depth: 0.33, width: 0.50, ratio: 2.0 } }
    pub fn m() -> Self { Self { depth: 0.67, width: 0.75, ratio: 1.5 } }
    pub fn l() -> Self { Self { depth: 1.00, width: 1.00, ratio: 1.0 } }
    pub fn x() -> Self { Self { depth: 1.00, width: 1.25, ratio: 1.0 } }

    /// P3, P4, P5 filter sizes for FPN neck and head
    pub fn filters(&self) -> [usize; 3] {
        [
            (256.0 * self.width) as usize,
            (512.0 * self.width) as usize,
            (512.0 * self.width * self.ratio) as usize,
        ]
    }

    /// Short repeat count (stages 2, 5, and FPN)
    pub fn n_short(&self) -> usize { (3.0 * self.depth).round() as usize }

    /// Long repeat count (stages 3, 4)
    pub fn n_long(&self) -> usize { (6.0 * self.depth).round() as usize }
}

/// YOLOv8-Pose model (backbone + neck + pose head)
#[derive(Debug)]
pub struct YoloV8Pose {
    backbone: DarkNet,
    neck: YoloV8Neck,
    head: PoseHead,
}

impl YoloV8Pose {
    pub fn load(vb: VarBuilder, m: Multiples) -> Result<Self> {
        let backbone = DarkNet::load(vb.pp("net"), &m)?;
        let neck = YoloV8Neck::load(vb.pp("fpn"), &m)?;

        let filters = m.filters();
        // nc=1 for pose (person class), 17 COCO keypoints
        let head = PoseHead::load(vb.pp("head"), 1, &filters, 17)?;

        Ok(Self {
            backbone,
            neck,
            head,
        })
    }
}

impl Module for YoloV8Pose {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Extract multi-scale features
        let (p3, p4, p5) = self.backbone.forward_features(xs)?;

        // Fuse features with FPN
        let (p3_out, p4_out, p5_out) = self.neck.forward(&p3, &p4, &p5)?;

        // Detect poses at all scales
        self.head.forward(&[p3_out, p4_out, p5_out])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn test_multiples_n() {
        let m = Multiples::n();
        assert_eq!(m.filters(), [64, 128, 256]);
    }

    #[test]
    fn test_multiples_s() {
        let m = Multiples::s();
        // c3=256*0.5=128, c4=512*0.5=256, c5=512*0.5*2.0=512
        assert_eq!(m.filters(), [128, 256, 512]);
    }

    #[test]
    fn test_yolov8_pose_forward_nano() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = YoloV8Pose::load(vb, Multiples::n()).unwrap();

        let input = Tensor::zeros(&[1, 3, 640, 640], DType::F32, &device).unwrap();
        let output = model.forward(&input).unwrap();

        // [1, 56, 8400] where 8400 = 80*80 + 40*40 + 20*20
        assert_eq!(output.dims(), &[1, 56, 8400]);
    }

    #[test]
    fn test_yolov8_pose_forward_small() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = YoloV8Pose::load(vb, Multiples::s()).unwrap();

        let input = Tensor::zeros(&[1, 3, 640, 640], DType::F32, &device).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 56, 8400]);
    }

    #[test]
    fn test_yolov8_pose_all_sizes() {
        let device = Device::Cpu;

        for (name, multiples) in [
            ("n", Multiples::n()),
            ("s", Multiples::s()),
            ("m", Multiples::m()),
            ("l", Multiples::l()),
            ("x", Multiples::x()),
        ] {
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let model = YoloV8Pose::load(vb, multiples);
            assert!(model.is_ok(), "Failed to load YoloV8Pose variant {}", name);
        }
    }
}
