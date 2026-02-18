use std::ops::Mul;

use crate::{Mat4, Quat, Vec3, Vec4};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose<T> {
    pub offset: Vec3<T>,
    pub rotation: Quat<T>,
}

impl<T> Pose<T> {
    pub fn new(offset: Vec3<T>, rotation: Quat<T>) -> Self {
        Self { offset, rotation }
    }
}

// --- f64 operations ---

impl Pose<f64> {
    pub fn identity() -> Self {
        Self {
            offset: Vec3::zero(),
            rotation: Quat::<f64>::identity(),
        }
    }

    /// Transform a point from child space to parent space (rotate then translate).
    pub fn transform_point(self, point: Vec3<f64>) -> Vec3<f64> {
        self.rotation.rotate(point) + self.offset
    }

    /// Transform a direction vector (rotation only, no translation).
    pub fn transform_vector(self, dir: Vec3<f64>) -> Vec3<f64> {
        self.rotation.rotate(dir)
    }

    /// Inverse pose: maps parent space back to child space.
    pub fn inverse(self) -> Self {
        let inv_rot = self.rotation.conjugate();
        let inv_offset = inv_rot.rotate(Vec3::zero() - self.offset);
        Self {
            offset: inv_offset,
            rotation: inv_rot,
        }
    }

    /// Convert to a 4x4 homogeneous transformation matrix.
    pub fn to_mat4(self) -> Mat4<f64> {
        let m3 = self.rotation.to_mat3();
        let t = self.offset;
        Mat4::from_cols(
            Vec4::new(m3.cols[0].x, m3.cols[0].y, m3.cols[0].z, 0.0),
            Vec4::new(m3.cols[1].x, m3.cols[1].y, m3.cols[1].z, 0.0),
            Vec4::new(m3.cols[2].x, m3.cols[2].y, m3.cols[2].z, 0.0),
            Vec4::new(t.x, t.y, t.z, 1.0),
        )
    }
}

/// Compose two poses: (parent * child) such that the result transforms
/// from child space through parent space.
impl Mul for Pose<f64> {
    type Output = Self;
    fn mul(self, child: Self) -> Self {
        Self {
            rotation: self.rotation * child.rotation,
            offset: self.rotation.rotate(child.offset) + self.offset,
        }
    }
}

// --- f32 operations ---

impl Pose<f32> {
    pub fn identity() -> Self {
        Self {
            offset: Vec3::zero(),
            rotation: Quat::<f32>::identity(),
        }
    }

    pub fn transform_point(self, point: Vec3<f32>) -> Vec3<f32> {
        self.rotation.rotate(point) + self.offset
    }

    pub fn transform_vector(self, dir: Vec3<f32>) -> Vec3<f32> {
        self.rotation.rotate(dir)
    }

    pub fn inverse(self) -> Self {
        let inv_rot = self.rotation.conjugate();
        let inv_offset = inv_rot.rotate(Vec3::zero() - self.offset);
        Self {
            offset: inv_offset,
            rotation: inv_rot,
        }
    }

    pub fn to_mat4(self) -> Mat4<f32> {
        let m3 = self.rotation.to_mat3();
        let t = self.offset;
        Mat4::from_cols(
            Vec4::new(m3.cols[0].x, m3.cols[0].y, m3.cols[0].z, 0.0),
            Vec4::new(m3.cols[1].x, m3.cols[1].y, m3.cols[1].z, 0.0),
            Vec4::new(m3.cols[2].x, m3.cols[2].y, m3.cols[2].z, 0.0),
            Vec4::new(t.x, t.y, t.z, 1.0),
        )
    }
}

impl Mul for Pose<f32> {
    type Output = Self;
    fn mul(self, child: Self) -> Self {
        Self {
            rotation: self.rotation * child.rotation,
            offset: self.rotation.rotate(child.offset) + self.offset,
        }
    }
}
