use std::ops::{Add, Mul, Neg, Sub};

use crate::Vec4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4<T> {
    pub cols: [Vec4<T>; 4],
}

impl<T> Mat4<T> {
    pub fn from_cols(c0: Vec4<T>, c1: Vec4<T>, c2: Vec4<T>, c3: Vec4<T>) -> Self {
        Self {
            cols: [c0, c1, c2, c3],
        }
    }
}

impl<T: Default> Mat4<T> {
    pub fn zero() -> Self {
        Self {
            cols: [Vec4::zero(), Vec4::zero(), Vec4::zero(), Vec4::zero()],
        }
    }
}

impl Mat4<f64> {
    pub fn identity() -> Self {
        Self::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }

    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-14 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(self.adjugate() * inv_det)
    }
}

impl Mat4<f32> {
    pub fn identity() -> Self {
        Self::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }

    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-7 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(self.adjugate() * inv_det)
    }
}

impl<T: Copy> Mat4<T> {
    pub fn transpose(self) -> Self {
        let [c0, c1, c2, c3] = self.cols;
        Self::from_cols(
            Vec4::new(c0.x, c1.x, c2.x, c3.x),
            Vec4::new(c0.y, c1.y, c2.y, c3.y),
            Vec4::new(c0.z, c1.z, c2.z, c3.z),
            Vec4::new(c0.w, c1.w, c2.w, c3.w),
        )
    }
}

impl<T: Mul<Output = T> + Sub<Output = T> + Add<Output = T> + Copy> Mat4<T> {
    pub fn determinant(self) -> T {
        let [c0, c1, c2, c3] = self.cols;

        let s0 = c0.x * c1.y - c1.x * c0.y;
        let s1 = c0.x * c1.z - c1.x * c0.z;
        let s2 = c0.x * c1.w - c1.x * c0.w;
        let s3 = c0.y * c1.z - c1.y * c0.z;
        let s4 = c0.y * c1.w - c1.y * c0.w;
        let s5 = c0.z * c1.w - c1.z * c0.w;

        let a0 = c2.z * c3.w - c3.z * c2.w;
        let a1 = c2.y * c3.w - c3.y * c2.w;
        let a2 = c2.y * c3.z - c3.y * c2.z;
        let a3 = c2.x * c3.w - c3.x * c2.w;
        let a4 = c2.x * c3.z - c3.x * c2.z;
        let a5 = c2.x * c3.y - c3.x * c2.y;

        s0 * a0 - s1 * a1 + s2 * a2 + s3 * a3 - s4 * a4 + s5 * a5
    }
}

impl<T: Mul<Output = T> + Sub<Output = T> + Add<Output = T> + Neg<Output = T> + Copy> Mat4<T> {
    fn adjugate(self) -> Self {
        let [c0, c1, c2, c3] = self.cols;

        // 2x2 sub-determinants of columns 0,1
        let s0 = c0.x * c1.y - c1.x * c0.y;
        let s1 = c0.x * c1.z - c1.x * c0.z;
        let s2 = c0.x * c1.w - c1.x * c0.w;
        let s3 = c0.y * c1.z - c1.y * c0.z;
        let s4 = c0.y * c1.w - c1.y * c0.w;
        let s5 = c0.z * c1.w - c1.z * c0.w;

        // 2x2 sub-determinants of columns 2,3
        let a0 = c2.x * c3.y - c3.x * c2.y;
        let a1 = c2.x * c3.z - c3.x * c2.z;
        let a2 = c2.x * c3.w - c3.x * c2.w;
        let a3 = c2.y * c3.z - c3.y * c2.z;
        let a4 = c2.y * c3.w - c3.y * c2.w;
        let a5 = c2.z * c3.w - c3.z * c2.w;

        // Adjugate = transpose of cofactor matrix
        Self::from_cols(
            Vec4::new(
                c1.y * a5 - c1.z * a4 + c1.w * a3,
                -(c0.y * a5 - c0.z * a4 + c0.w * a3),
                c3.y * s5 - c3.z * s4 + c3.w * s3,
                -(c2.y * s5 - c2.z * s4 + c2.w * s3),
            ),
            Vec4::new(
                -(c1.x * a5 - c1.z * a2 + c1.w * a1),
                c0.x * a5 - c0.z * a2 + c0.w * a1,
                -(c3.x * s5 - c3.z * s2 + c3.w * s1),
                c2.x * s5 - c2.z * s2 + c2.w * s1,
            ),
            Vec4::new(
                c1.x * a4 - c1.y * a2 + c1.w * a0,
                -(c0.x * a4 - c0.y * a2 + c0.w * a0),
                c3.x * s4 - c3.y * s2 + c3.w * s0,
                -(c2.x * s4 - c2.y * s2 + c2.w * s0),
            ),
            Vec4::new(
                -(c1.x * a3 - c1.y * a1 + c1.z * a0),
                c0.x * a3 - c0.y * a1 + c0.z * a0,
                -(c3.x * s3 - c3.y * s1 + c3.z * s0),
                c2.x * s3 - c2.y * s1 + c2.z * s0,
            ),
        )
    }
}

// --- Arithmetic ---

impl<T: Add<Output = T> + Copy> Add for Mat4<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            cols: [
                self.cols[0] + rhs.cols[0],
                self.cols[1] + rhs.cols[1],
                self.cols[2] + rhs.cols[2],
                self.cols[3] + rhs.cols[3],
            ],
        }
    }
}

impl<T: Sub<Output = T> + Copy> Sub for Mat4<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            cols: [
                self.cols[0] - rhs.cols[0],
                self.cols[1] - rhs.cols[1],
                self.cols[2] - rhs.cols[2],
                self.cols[3] - rhs.cols[3],
            ],
        }
    }
}

impl<T: Neg<Output = T> + Copy> Neg for Mat4<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            cols: [-self.cols[0], -self.cols[1], -self.cols[2], -self.cols[3]],
        }
    }
}

impl<T: Mul<Output = T> + Copy> Mul<T> for Mat4<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self {
            cols: [
                self.cols[0] * rhs,
                self.cols[1] * rhs,
                self.cols[2] * rhs,
                self.cols[3] * rhs,
            ],
        }
    }
}

// Mat4 * Vec4
impl<T: Mul<Output = T> + Add<Output = T> + Copy> Mul<Vec4<T>> for Mat4<T> {
    type Output = Vec4<T>;
    fn mul(self, v: Vec4<T>) -> Vec4<T> {
        self.cols[0] * v.x + self.cols[1] * v.y + self.cols[2] * v.z + self.cols[3] * v.w
    }
}

// Mat4 * Mat4
impl<T: Mul<Output = T> + Add<Output = T> + Copy> Mul<Mat4<T>> for Mat4<T> {
    type Output = Self;
    fn mul(self, rhs: Mat4<T>) -> Self {
        Self::from_cols(
            self * rhs.cols[0],
            self * rhs.cols[1],
            self * rhs.cols[2],
            self * rhs.cols[3],
        )
    }
}
