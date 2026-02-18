use deli_base::{Rect, Vec2};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    pub position: Vec2<f32>,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PoseDetection {
    pub bbox: Rect<f32>,
    pub confidence: f32,
    pub keypoints: [Keypoint; 17],
}

pub type PoseDetections = Vec<PoseDetection>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum CocoKeypoint {
    Nose = 0,
    LeftEye = 1,
    RightEye = 2,
    LeftEar = 3,
    RightEar = 4,
    LeftShoulder = 5,
    RightShoulder = 6,
    LeftElbow = 7,
    RightElbow = 8,
    LeftWrist = 9,
    RightWrist = 10,
    LeftHip = 11,
    RightHip = 12,
    LeftKnee = 13,
    RightKnee = 14,
    LeftAnkle = 15,
    RightAnkle = 16,
}

impl From<CocoKeypoint> for usize {
    fn from(kp: CocoKeypoint) -> usize {
        kp as usize
    }
}
