// YOLOv8-Pose detection head
// Ported from HuggingFace candle-wasm-examples/yolo
// Source: https://github.com/huggingface/candle/tree/main/candle-wasm-examples/yolo
// License: Apache-2.0 OR MIT

use {super::{ConvBlock, Dfl}, candle_core::{Device, Result, Tensor}, candle_nn::{conv2d, Conv2dConfig, Module, VarBuilder}};

/// Pose detection head for YOLOv8-Pose
#[derive(Debug)]
pub struct PoseHead {
    // Bbox regression branch: 2 ConvBlocks + 1 Conv2d per scale
    cv2: [([ConvBlock; 2], candle_nn::Conv2d); 3],
    // Class prediction branch: 2 ConvBlocks + 1 Conv2d per scale
    cv3: [([ConvBlock; 2], candle_nn::Conv2d); 3],
    // Keypoint branch: 2 ConvBlocks + 1 Conv2d per scale
    cv4: [([ConvBlock; 2], candle_nn::Conv2d); 3],
    dfl: Dfl,
    reg_max: usize,
    num_keypoints: usize,
    strides: Vec<usize>,
}

impl PoseHead {
    pub fn load(vb: VarBuilder, nc: usize, filters: &[usize; 3], num_keypoints: usize) -> Result<Self> {
        let reg_max = 16;
        let c2 = (filters[0] / 4).max(16).max(reg_max * 4);
        let c3 = filters[0].max(nc.min(100));
        let c4 = (filters[0] / 4).max(num_keypoints * 3);
        let conv1x1 = Conv2dConfig { ..Default::default() };

        let mut cv2_vec = Vec::with_capacity(3);
        let mut cv3_vec = Vec::with_capacity(3);
        let mut cv4_vec = Vec::with_capacity(3);

        for (i, &f) in filters.iter().enumerate() {
            // Bbox regression: 2 ConvBlocks + Conv2d(→4*reg_max)
            let cb0 = ConvBlock::load(vb.pp(format!("cv2.{i}.0")), f, c2, 3, 1, 1)?;
            let cb1 = ConvBlock::load(vb.pp(format!("cv2.{i}.1")), c2, c2, 3, 1, 1)?;
            let final_cv = conv2d(c2, 4 * reg_max, 1, conv1x1, vb.pp(format!("cv2.{i}.2")))?;
            cv2_vec.push(([cb0, cb1], final_cv));

            // Class prediction: 2 ConvBlocks + Conv2d(→nc)
            let cb0 = ConvBlock::load(vb.pp(format!("cv3.{i}.0")), f, c3, 3, 1, 1)?;
            let cb1 = ConvBlock::load(vb.pp(format!("cv3.{i}.1")), c3, c3, 3, 1, 1)?;
            let final_cv = conv2d(c3, nc, 1, conv1x1, vb.pp(format!("cv3.{i}.2")))?;
            cv3_vec.push(([cb0, cb1], final_cv));

            // Keypoint: 2 ConvBlocks + Conv2d(→nkpt*3)
            let cb0 = ConvBlock::load(vb.pp(format!("cv4.{i}.0")), f, c4, 3, 1, 1)?;
            let cb1 = ConvBlock::load(vb.pp(format!("cv4.{i}.1")), c4, c4, 3, 1, 1)?;
            let final_cv = conv2d(c4, num_keypoints * 3, 1, conv1x1, vb.pp(format!("cv4.{i}.2")))?;
            cv4_vec.push(([cb0, cb1], final_cv));
        }

        let dfl = Dfl::load(vb.pp("dfl"), reg_max)?;

        Ok(Self {
            cv2: cv2_vec.try_into().map_err(|_| candle_core::Error::Msg("cv2 size".into()))?,
            cv3: cv3_vec.try_into().map_err(|_| candle_core::Error::Msg("cv3 size".into()))?,
            cv4: cv4_vec.try_into().map_err(|_| candle_core::Error::Msg("cv4 size".into()))?,
            dfl,
            reg_max,
            num_keypoints,
            strides: vec![8, 16, 32],
        })
    }

    pub fn forward(&self, xs: &[Tensor]) -> Result<Tensor> {
        let device = xs[0].device();
        let mut outputs = Vec::new();

        for (i, x) in xs.iter().enumerate() {
            // Bbox regression branch
            let bbox = x.apply(&self.cv2[i].0[0])?.apply(&self.cv2[i].0[1])?.apply(&self.cv2[i].1)?;
            let (b, _, h, w) = bbox.dims4()?;

            // Class prediction branch
            let cls = x.apply(&self.cv3[i].0[0])?.apply(&self.cv3[i].0[1])?.apply(&self.cv3[i].1)?;

            // Keypoint branch
            let kpt = x.apply(&self.cv4[i].0[0])?.apply(&self.cv4[i].0[1])?.apply(&self.cv4[i].1)?;

            // Reshape: [b, C, h, w] -> [b, C, h*w]
            let bbox = bbox.reshape((b, 4 * self.reg_max, h * w))?;
            let cls = cls.reshape((b, cls.dim(1)?, h * w))?;
            let kpt = kpt.reshape((b, self.num_keypoints * 3, h * w))?;

            // Apply DFL to bbox predictions
            let bbox = self.dfl.forward(&bbox)?;

            // Generate anchors and convert bbox from ltrb to xywh
            let anchors = make_anchors(h, w, self.strides[i], device)?;
            let bbox = dist2bbox(&bbox, &anchors, true)?;

            // Concatenate: bbox (4) + cls + kpt (nkpt*3) -> [b, 4 + nc + nkpt*3, h*w]
            let output = Tensor::cat(&[&bbox, &cls, &kpt], 1)?;
            outputs.push(output);
        }

        // Concatenate all scales: [b, 56, N] where N = sum of all h*w
        Tensor::cat(&outputs, 2)
    }
}

/// Generate anchor grid for a feature map
pub fn make_anchors(h: usize, w: usize, stride: usize, device: &Device) -> Result<Tensor> {
    let mut anchors = Vec::with_capacity(h * w * 2);

    for y in 0..h {
        for x in 0..w {
            anchors.push((x as f32 + 0.5) * stride as f32);
            anchors.push((y as f32 + 0.5) * stride as f32);
        }
    }

    Tensor::from_vec(anchors, (h * w, 2), device)
}

/// Convert distance predictions to bounding boxes
pub fn dist2bbox(distance: &Tensor, anchor_points: &Tensor, xywh: bool) -> Result<Tensor> {
    let lt = distance.narrow(1, 0, 2)?;
    let rb = distance.narrow(1, 2, 2)?;
    let lt = lt.neg()?;

    // Broadcast anchor_points to [1, 2, N]
    let anchor_points = anchor_points.transpose(0, 1)?.unsqueeze(0)?;

    let x1y1 = anchor_points.broadcast_add(&lt)?;
    let x2y2 = anchor_points.broadcast_add(&rb)?;

    if xywh {
        let c_xy = ((x1y1.clone() + x2y2.clone())? * 0.5)?;
        let wh = (x2y2 - x1y1)?;
        Tensor::cat(&[&c_xy, &wh], 1)
    } else {
        Tensor::cat(&[&x1y1, &x2y2], 1)
    }
}
