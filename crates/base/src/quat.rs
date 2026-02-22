use {crate::{Mat3, Vec3}, std::{fmt, ops::{Add, Div, Mul, Neg, Sub}}};

#[derive(Clone, Copy, PartialEq)]
pub struct Quat<T> {
    pub w: T,
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: fmt::Debug> fmt::Debug for Quat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Quat")
            .field("w", &self.w)
            .field("x", &self.x)
            .field("y", &self.y)
            .field("z", &self.z)
            .finish()
    }
}

impl<T: Default> Default for Quat<T> {
    fn default() -> Self {
        Self {
            w: T::default(),
            x: T::default(),
            y: T::default(),
            z: T::default(),
        }
    }
}

impl<T> Quat<T> {
    pub fn new(w: T, x: T, y: T, z: T) -> Self {
        Self { w, x, y, z }
    }
}

// --- Arithmetic operators ---

impl<T: Add<Output = T>> Add for Quat<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            w: self.w + rhs.w,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: Sub<Output = T>> Sub for Quat<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            w: self.w - rhs.w,
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T: Neg<Output = T>> Neg for Quat<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            w: -self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl<T: Mul<Output = T> + Copy> Mul<T> for Quat<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self {
            w: self.w * rhs,
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T: Div<Output = T> + Copy> Div<T> for Quat<T> {
    type Output = Self;
    fn div(self, rhs: T) -> Self {
        Self {
            w: self.w / rhs,
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

// --- Generic linear algebra ---

impl<T: Mul<Output = T> + Add<Output = T> + Copy> Quat<T> {
    pub fn dot(self, rhs: Self) -> T {
        self.w * rhs.w + self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn length_squared(self) -> T {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }
}

impl<T: Neg<Output = T> + Copy> Quat<T> {
    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

// --- Hamilton product (generic) ---

impl<T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy> Mul<Quat<T>> for Quat<T> {
    type Output = Self;
    fn mul(self, rhs: Quat<T>) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

// --- Float operations (f64) ---

impl Quat<f64> {
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub fn from_axis_angle(axis: Vec3<f64>, angle: f64) -> Self {
        let half = angle / 2.0;
        let s = half.sin();
        let c = half.cos();
        Self {
            w: c,
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
        }
    }

    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        self / len
    }

    pub fn inverse(self) -> Self {
        let len_sq = self.length_squared();
        self.conjugate() / len_sq
    }

    pub fn rotate(self, v: Vec3<f64>) -> Vec3<f64> {
        let qv = Quat::new(0.0, v.x, v.y, v.z);
        let rotated = self * qv * self.conjugate();
        Vec3::new(rotated.x, rotated.y, rotated.z)
    }

    pub fn to_axis_angle(self) -> (Vec3<f64>, f64) {
        let n = self.normalized();
        let angle = 2.0 * n.w.acos();
        let s = (1.0 - n.w * n.w).sqrt();
        if s < 1e-10 {
            // Angle is ~0, axis is arbitrary
            (Vec3::new(1.0, 0.0, 0.0), 0.0)
        } else {
            (Vec3::new(n.x / s, n.y / s, n.z / s), angle)
        }
    }

    pub fn to_mat3(self) -> Mat3<f64> {
        let q = self.normalized();
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);

        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;

        Mat3::from_cols(
            Vec3::new(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)),
            Vec3::new(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)),
            Vec3::new(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)),
        )
    }

    pub fn slerp(self, other: Self, t: f64) -> Self {
        let mut dot = self.dot(other);

        // If dot is negative, negate one to take the short path
        let other = if dot < 0.0 {
            dot = -dot;
            -other
        } else {
            other
        };

        // If quaternions are very close, use linear interpolation
        if dot > 0.9995 {
            return (self * (1.0 - t) + other * t).normalized();
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;

        self * a + other * b
    }
}

// --- Float operations (f32) ---

impl Quat<f32> {
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub fn from_axis_angle(axis: Vec3<f32>, angle: f32) -> Self {
        let half = angle / 2.0;
        let s = half.sin();
        let c = half.cos();
        Self {
            w: c,
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
        }
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        self / len
    }

    pub fn inverse(self) -> Self {
        let len_sq = self.length_squared();
        self.conjugate() / len_sq
    }

    pub fn rotate(self, v: Vec3<f32>) -> Vec3<f32> {
        let qv = Quat::new(0.0, v.x, v.y, v.z);
        let rotated = self * qv * self.conjugate();
        Vec3::new(rotated.x, rotated.y, rotated.z)
    }

    pub fn to_axis_angle(self) -> (Vec3<f32>, f32) {
        let n = self.normalized();
        let angle = 2.0 * n.w.acos();
        let s = (1.0 - n.w * n.w).sqrt();
        if s < 1e-6 {
            (Vec3::new(1.0, 0.0, 0.0), 0.0)
        } else {
            (Vec3::new(n.x / s, n.y / s, n.z / s), angle)
        }
    }

    pub fn to_mat3(self) -> Mat3<f32> {
        let q = self.normalized();
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);

        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;

        Mat3::from_cols(
            Vec3::new(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)),
            Vec3::new(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)),
            Vec3::new(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)),
        )
    }

    pub fn slerp(self, other: Self, t: f32) -> Self {
        let mut dot = self.dot(other);

        let other = if dot < 0.0 {
            dot = -dot;
            -other
        } else {
            other
        };

        if dot > 0.9995 {
            return (self * (1.0 - t) + other * t).normalized();
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;

        self * a + other * b
    }
}
