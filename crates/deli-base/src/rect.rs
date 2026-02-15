use crate::Vec2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect<T> {
    pub origin: Vec2<T>,
    pub size: Vec2<T>,
}

impl<T: Default> Default for Rect<T> {
    fn default() -> Self {
        Self {
            origin: Vec2::zero(),
            size: Vec2::zero(),
        }
    }
}

impl<T> Rect<T> {
    pub fn new(origin: Vec2<T>, size: Vec2<T>) -> Self {
        Self { origin, size }
    }
}

impl<T: Default> Rect<T> {
    pub fn zero() -> Self {
        Self::default()
    }
}

impl<T: std::ops::Add<Output = T> + Copy> Rect<T> {
    pub fn min(&self) -> Vec2<T> {
        self.origin
    }

    pub fn max(&self) -> Vec2<T> {
        self.origin + self.size
    }
}

impl<T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy> Rect<T> {
    pub fn area(&self) -> T {
        self.size.x * self.size.y
    }
}

impl<T: std::ops::Add<Output = T> + std::ops::Div<Output = T> + Copy + From<u8>> Rect<T> {
    pub fn center(&self) -> Vec2<T> {
        let two: T = T::from(2u8);
        Vec2::new(
            self.origin.x + self.size.x / two,
            self.origin.y + self.size.y / two,
        )
    }
}

impl<T: std::ops::Sub<Output = T> + Copy> Rect<T> {
    pub fn from_min_max(min: Vec2<T>, max: Vec2<T>) -> Self {
        Self {
            origin: min,
            size: max - min,
        }
    }
}

impl<T: std::ops::Add<Output = T> + PartialOrd + Copy> Rect<T> {
    pub fn contains_point(&self, point: Vec2<T>) -> bool {
        let max = self.max();
        point.x >= self.origin.x
            && point.y >= self.origin.y
            && point.x < max.x
            && point.y < max.y
    }

    pub fn contains_rect(&self, other: Rect<T>) -> bool {
        let self_max = self.max();
        let other_max = other.max();
        other.origin.x >= self.origin.x
            && other.origin.y >= self.origin.y
            && other_max.x <= self_max.x
            && other_max.y <= self_max.y
    }

    pub fn intersects(&self, other: Rect<T>) -> bool {
        let self_max = self.max();
        let other_max = other.max();
        self.origin.x < other_max.x
            && other.origin.x < self_max.x
            && self.origin.y < other_max.y
            && other.origin.y < self_max.y
    }
}

impl<T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + PartialOrd + Copy> Rect<T> {
    pub fn intersection(&self, other: Rect<T>) -> Option<Rect<T>> {
        if !self.intersects(other) {
            return None;
        }
        let self_max = self.max();
        let other_max = other.max();

        let min_x = if self.origin.x > other.origin.x {
            self.origin.x
        } else {
            other.origin.x
        };
        let min_y = if self.origin.y > other.origin.y {
            self.origin.y
        } else {
            other.origin.y
        };
        let max_x = if self_max.x < other_max.x {
            self_max.x
        } else {
            other_max.x
        };
        let max_y = if self_max.y < other_max.y {
            self_max.y
        } else {
            other_max.y
        };

        Some(Rect {
            origin: Vec2::new(min_x, min_y),
            size: Vec2::new(max_x - min_x, max_y - min_y),
        })
    }

    pub fn union(&self, other: Rect<T>) -> Rect<T> {
        let self_max = self.max();
        let other_max = other.max();

        let min_x = if self.origin.x < other.origin.x {
            self.origin.x
        } else {
            other.origin.x
        };
        let min_y = if self.origin.y < other.origin.y {
            self.origin.y
        } else {
            other.origin.y
        };
        let max_x = if self_max.x > other_max.x {
            self_max.x
        } else {
            other_max.x
        };
        let max_y = if self_max.y > other_max.y {
            self_max.y
        } else {
            other_max.y
        };

        Rect {
            origin: Vec2::new(min_x, min_y),
            size: Vec2::new(max_x - min_x, max_y - min_y),
        }
    }
}
