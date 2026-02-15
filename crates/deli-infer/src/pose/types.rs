use deli_math::{Rect, Vec2};

/// Number of keypoints in COCO pose format
pub const COCO_KEYPOINT_COUNT: usize = 17;

/// A single keypoint with 2D position and confidence score
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    pub position: Vec2<f32>,
    /// Confidence/visibility score in [0.0, 1.0] range.
    /// YOLO pose models output continuous confidence values, not COCO categorical (0/1/2).
    pub confidence: f32,
}

/// COCO keypoint indices for human pose
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeypointIndex {
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

impl From<KeypointIndex> for usize {
    fn from(index: KeypointIndex) -> usize {
        index as usize
    }
}

impl TryFrom<usize> for KeypointIndex {
    type Error = String;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(KeypointIndex::Nose),
            1 => Ok(KeypointIndex::LeftEye),
            2 => Ok(KeypointIndex::RightEye),
            3 => Ok(KeypointIndex::LeftEar),
            4 => Ok(KeypointIndex::RightEar),
            5 => Ok(KeypointIndex::LeftShoulder),
            6 => Ok(KeypointIndex::RightShoulder),
            7 => Ok(KeypointIndex::LeftElbow),
            8 => Ok(KeypointIndex::RightElbow),
            9 => Ok(KeypointIndex::LeftWrist),
            10 => Ok(KeypointIndex::RightWrist),
            11 => Ok(KeypointIndex::LeftHip),
            12 => Ok(KeypointIndex::RightHip),
            13 => Ok(KeypointIndex::LeftKnee),
            14 => Ok(KeypointIndex::RightKnee),
            15 => Ok(KeypointIndex::LeftAnkle),
            16 => Ok(KeypointIndex::RightAnkle),
            _ => Err(format!(
                "Invalid keypoint index: {}. Must be in range 0-16.",
                value
            )),
        }
    }
}

/// A detected person with bounding box and keypoints
#[derive(Debug, Clone, PartialEq)]
pub struct PoseDetection {
    /// Bounding box of the detected person
    pub bbox: Rect<f32>,
    /// Confidence score for the person detection
    pub confidence: f32,
    /// Array of 17 COCO keypoints
    pub keypoints: [Keypoint; COCO_KEYPOINT_COUNT],
}

impl PoseDetection {
    /// Get a keypoint by its semantic index
    pub fn keypoint(&self, index: KeypointIndex) -> &Keypoint {
        &self.keypoints[usize::from(index)]
    }
}

/// Letterbox transformation parameters for coordinate rescaling
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterboxInfo {
    /// Scale factor applied to the image (min(640/H, 640/W))
    pub scale: f32,
    /// Horizontal padding added (in pixels)
    pub pad_x: f32,
    /// Vertical padding added (in pixels)
    pub pad_y: f32,
}
