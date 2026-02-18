use std::ops::{Add, Mul, Neg, Sub};

use crate::Vec2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat2<T> {
    pub cols: [Vec2<T>; 2],
}

impl<T> Mat2<T> {
    pub fn from_cols(c0: Vec2<T>, c1: Vec2<T>) -> Self {
        Self { cols: [c0, c1] }
    }
}

impl<T: Default> Mat2<T> {
    pub fn zero() -> Self {
        Self {
            cols: [Vec2::zero(), Vec2::zero()],
        }
    }
}

impl Mat2<f64> {
    pub fn identity() -> Self {
        Self::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0))
    }

    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-14 {
            return None;
        }
        let inv_det = 1.0 / det;
        let a = self.cols[0].x;
        let c = self.cols[0].y;
        let b = self.cols[1].x;
        let d = self.cols[1].y;
        Some(Self::from_cols(
            Vec2::new(d * inv_det, -c * inv_det),
            Vec2::new(-b * inv_det, a * inv_det),
        ))
    }
}

impl Mat2<f32> {
    pub fn identity() -> Self {
        Self::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0))
    }

    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-7 {
            return None;
        }
        let inv_det = 1.0 / det;
        let a = self.cols[0].x;
        let c = self.cols[0].y;
        let b = self.cols[1].x;
        let d = self.cols[1].y;
        Some(Self::from_cols(
            Vec2::new(d * inv_det, -c * inv_det),
            Vec2::new(-b * inv_det, a * inv_det),
        ))
    }
}

impl<T: Mul<Output = T> + Sub<Output = T> + Copy> Mat2<T> {
    pub fn determinant(self) -> T {
        self.cols[0].x * self.cols[1].y - self.cols[1].x * self.cols[0].y
    }
}

impl<T: Copy> Mat2<T> {
    pub fn transpose(self) -> Self {
        Self::from_cols(
            Vec2::new(self.cols[0].x, self.cols[1].x),
            Vec2::new(self.cols[0].y, self.cols[1].y),
        )
    }
}

// --- Arithmetic ---

impl<T: Add<Output = T> + Copy> Add for Mat2<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            cols: [self.cols[0] + rhs.cols[0], self.cols[1] + rhs.cols[1]],
        }
    }
}

impl<T: Sub<Output = T> + Copy> Sub for Mat2<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            cols: [self.cols[0] - rhs.cols[0], self.cols[1] - rhs.cols[1]],
        }
    }
}

impl<T: Neg<Output = T> + Copy> Neg for Mat2<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            cols: [-self.cols[0], -self.cols[1]],
        }
    }
}

impl<T: Mul<Output = T> + Copy> Mul<T> for Mat2<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self {
            cols: [self.cols[0] * rhs, self.cols[1] * rhs],
        }
    }
}

// Mat2 * Vec2
impl<T: Mul<Output = T> + Add<Output = T> + Copy> Mul<Vec2<T>> for Mat2<T> {
    type Output = Vec2<T>;
    fn mul(self, v: Vec2<T>) -> Vec2<T> {
        self.cols[0] * v.x + self.cols[1] * v.y
    }
}

// Mat2 * Mat2
impl<T: Mul<Output = T> + Add<Output = T> + Copy> Mul<Mat2<T>> for Mat2<T> {
    type Output = Self;
    fn mul(self, rhs: Mat2<T>) -> Self {
        Self::from_cols(self * rhs.cols[0], self * rhs.cols[1])
    }
}
