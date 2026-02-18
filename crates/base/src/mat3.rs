use std::ops::{Add, Mul, Neg, Sub};

use crate::Vec3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3<T> {
    pub cols: [Vec3<T>; 3],
}

impl<T> Mat3<T> {
    pub fn from_cols(c0: Vec3<T>, c1: Vec3<T>, c2: Vec3<T>) -> Self {
        Self { cols: [c0, c1, c2] }
    }
}

impl<T: Default> Mat3<T> {
    pub fn zero() -> Self {
        Self {
            cols: [Vec3::zero(), Vec3::zero(), Vec3::zero()],
        }
    }
}

impl Mat3<f64> {
    pub fn identity() -> Self {
        Self::from_cols(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
    }

    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-14 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(self.cofactor_matrix().transpose() * inv_det)
    }
}

impl Mat3<f32> {
    pub fn identity() -> Self {
        Self::from_cols(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
    }

    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-7 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(self.cofactor_matrix().transpose() * inv_det)
    }
}

impl<T: Copy> Mat3<T> {
    pub fn transpose(self) -> Self {
        let [c0, c1, c2] = self.cols;
        Self::from_cols(
            Vec3::new(c0.x, c1.x, c2.x),
            Vec3::new(c0.y, c1.y, c2.y),
            Vec3::new(c0.z, c1.z, c2.z),
        )
    }
}

impl<T: Mul<Output = T> + Sub<Output = T> + Add<Output = T> + Copy> Mat3<T> {
    pub fn determinant(self) -> T {
        let [c0, c1, c2] = self.cols;
        c0.x * (c1.y * c2.z - c1.z * c2.y)
            - c1.x * (c0.y * c2.z - c0.z * c2.y)
            + c2.x * (c0.y * c1.z - c0.z * c1.y)
    }
}

impl<T: Mul<Output = T> + Sub<Output = T> + Neg<Output = T> + Copy> Mat3<T> {
    fn cofactor_matrix(self) -> Self {
        let [c0, c1, c2] = self.cols;
        // Cofactor (i,j) = (-1)^(i+j) * minor(i,j)
        Self::from_cols(
            Vec3::new(
                c1.y * c2.z - c1.z * c2.y,
                -(c1.x * c2.z - c1.z * c2.x),
                c1.x * c2.y - c1.y * c2.x,
            ),
            Vec3::new(
                -(c0.y * c2.z - c0.z * c2.y),
                c0.x * c2.z - c0.z * c2.x,
                -(c0.x * c2.y - c0.y * c2.x),
            ),
            Vec3::new(
                c0.y * c1.z - c0.z * c1.y,
                -(c0.x * c1.z - c0.z * c1.x),
                c0.x * c1.y - c0.y * c1.x,
            ),
        )
    }
}

// --- Arithmetic ---

impl<T: Add<Output = T> + Copy> Add for Mat3<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            cols: [
                self.cols[0] + rhs.cols[0],
                self.cols[1] + rhs.cols[1],
                self.cols[2] + rhs.cols[2],
            ],
        }
    }
}

impl<T: Sub<Output = T> + Copy> Sub for Mat3<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            cols: [
                self.cols[0] - rhs.cols[0],
                self.cols[1] - rhs.cols[1],
                self.cols[2] - rhs.cols[2],
            ],
        }
    }
}

impl<T: Neg<Output = T> + Copy> Neg for Mat3<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            cols: [-self.cols[0], -self.cols[1], -self.cols[2]],
        }
    }
}

impl<T: Mul<Output = T> + Copy> Mul<T> for Mat3<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self {
            cols: [self.cols[0] * rhs, self.cols[1] * rhs, self.cols[2] * rhs],
        }
    }
}

// Mat3 * Vec3
impl<T: Mul<Output = T> + Add<Output = T> + Copy> Mul<Vec3<T>> for Mat3<T> {
    type Output = Vec3<T>;
    fn mul(self, v: Vec3<T>) -> Vec3<T> {
        self.cols[0] * v.x + self.cols[1] * v.y + self.cols[2] * v.z
    }
}

// Mat3 * Mat3
impl<T: Mul<Output = T> + Add<Output = T> + Copy> Mul<Mat3<T>> for Mat3<T> {
    type Output = Self;
    fn mul(self, rhs: Mat3<T>) -> Self {
        Self::from_cols(self * rhs.cols[0], self * rhs.cols[1], self * rhs.cols[2])
    }
}
