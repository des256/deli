use deli_math::Vec3;

#[test]
fn test_new_and_fields() {
    let v = Vec3::new(1.0_f64, 2.0, 3.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);
    assert_eq!(v.z, 3.0);
}

#[test]
fn test_zero() {
    let v = Vec3::<f64>::zero();
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 0.0);
    assert_eq!(v.z, 0.0);
}

// --- Arithmetic operators ---

#[test]
fn test_add() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);
    let c = a + b;
    assert_eq!(c.x, 5.0);
    assert_eq!(c.y, 7.0);
    assert_eq!(c.z, 9.0);
}

#[test]
fn test_sub() {
    let a = Vec3::new(5.0, 7.0, 9.0);
    let b = Vec3::new(1.0, 2.0, 3.0);
    let c = a - b;
    assert_eq!(c.x, 4.0);
    assert_eq!(c.y, 5.0);
    assert_eq!(c.z, 6.0);
}

#[test]
fn test_mul_scalar() {
    let v = Vec3::new(2.0, 3.0, 4.0);
    let r = v * 3.0;
    assert_eq!(r.x, 6.0);
    assert_eq!(r.y, 9.0);
    assert_eq!(r.z, 12.0);
}

#[test]
fn test_div_scalar() {
    let v = Vec3::new(8.0, 6.0, 4.0);
    let r = v / 2.0;
    assert_eq!(r.x, 4.0);
    assert_eq!(r.y, 3.0);
    assert_eq!(r.z, 2.0);
}

#[test]
fn test_neg() {
    let v = Vec3::new(1.0, -2.0, 3.0);
    let r = -v;
    assert_eq!(r.x, -1.0);
    assert_eq!(r.y, 2.0);
    assert_eq!(r.z, -3.0);
}

#[test]
fn test_mul_componentwise() {
    let a = Vec3::new(2.0, 3.0, 4.0);
    let b = Vec3::new(5.0, 6.0, 7.0);
    let c = a * b;
    assert_eq!(c.x, 10.0);
    assert_eq!(c.y, 18.0);
    assert_eq!(c.z, 28.0);
}

#[test]
fn test_div_componentwise() {
    let a = Vec3::new(10.0, 18.0, 28.0);
    let b = Vec3::new(5.0, 6.0, 7.0);
    let c = a / b;
    assert_eq!(c.x, 2.0);
    assert_eq!(c.y, 3.0);
    assert_eq!(c.z, 4.0);
}

// --- Assign operators ---

#[test]
fn test_add_assign() {
    let mut v = Vec3::new(1.0, 2.0, 3.0);
    v += Vec3::new(4.0, 5.0, 6.0);
    assert_eq!(v.x, 5.0);
    assert_eq!(v.y, 7.0);
    assert_eq!(v.z, 9.0);
}

#[test]
fn test_sub_assign() {
    let mut v = Vec3::new(5.0, 7.0, 9.0);
    v -= Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(v.x, 4.0);
    assert_eq!(v.y, 5.0);
    assert_eq!(v.z, 6.0);
}

#[test]
fn test_mul_assign_scalar() {
    let mut v = Vec3::new(2.0, 3.0, 4.0);
    v *= 3.0;
    assert_eq!(v.x, 6.0);
    assert_eq!(v.y, 9.0);
    assert_eq!(v.z, 12.0);
}

#[test]
fn test_div_assign_scalar() {
    let mut v = Vec3::new(8.0, 6.0, 4.0);
    v /= 2.0;
    assert_eq!(v.x, 4.0);
    assert_eq!(v.y, 3.0);
    assert_eq!(v.z, 2.0);
}

// --- Linear algebra ---

#[test]
fn test_dot() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_eq!(a.dot(b), 32.0);
}

#[test]
fn test_cross() {
    let a = Vec3::new(1.0_f64, 0.0, 0.0);
    let b = Vec3::new(0.0, 1.0, 0.0);
    let c = a.cross(b);
    assert_eq!(c.x, 0.0);
    assert_eq!(c.y, 0.0);
    assert_eq!(c.z, 1.0);
}

#[test]
fn test_cross_anticommutative() {
    let a = Vec3::new(2.0_f64, 3.0, 4.0);
    let b = Vec3::new(5.0, 6.0, 7.0);
    let ab = a.cross(b);
    let ba = b.cross(a);
    assert!((ab.x + ba.x).abs() < 1e-10);
    assert!((ab.y + ba.y).abs() < 1e-10);
    assert!((ab.z + ba.z).abs() < 1e-10);
}

#[test]
fn test_cross_perpendicular() {
    let a = Vec3::new(2.0_f64, 3.0, 4.0);
    let b = Vec3::new(5.0, 6.0, 7.0);
    let c = a.cross(b);
    assert!(a.dot(c).abs() < 1e-10);
    assert!(b.dot(c).abs() < 1e-10);
}

#[test]
fn test_length_squared() {
    let v = Vec3::new(1.0_f64, 2.0, 3.0);
    // 1 + 4 + 9 = 14
    assert_eq!(v.length_squared(), 14.0);
}

#[test]
fn test_length() {
    let v = Vec3::new(2.0_f64, 3.0, 6.0);
    // sqrt(4+9+36) = sqrt(49) = 7
    assert!((v.length() - 7.0).abs() < 1e-10);
}

#[test]
fn test_normalized() {
    let v = Vec3::new(0.0_f64, 3.0, 4.0);
    let n = v.normalized();
    assert!((n.length() - 1.0).abs() < 1e-10);
    assert!((n.x - 0.0).abs() < 1e-10);
    assert!((n.y - 0.6).abs() < 1e-10);
    assert!((n.z - 0.8).abs() < 1e-10);
}

#[test]
fn test_distance_to() {
    let a = Vec3::new(1.0_f64, 2.0, 3.0);
    let b = Vec3::new(4.0, 6.0, 3.0);
    // sqrt(9+16+0) = 5
    assert!((a.distance_to(b) - 5.0).abs() < 1e-10);
}

#[test]
fn test_lerp() {
    let a = Vec3::new(0.0_f64, 0.0, 0.0);
    let b = Vec3::new(10.0, 20.0, 30.0);
    let c = a.lerp(b, 0.25);
    assert!((c.x - 2.5).abs() < 1e-10);
    assert!((c.y - 5.0).abs() < 1e-10);
    assert!((c.z - 7.5).abs() < 1e-10);
}

#[test]
fn test_reflect() {
    // Reflect (1, -1, 0) over normal (0, 1, 0)
    let v = Vec3::new(1.0_f64, -1.0, 0.0);
    let n = Vec3::new(0.0, 1.0, 0.0);
    let r = v.reflect(n);
    assert!((r.x - 1.0).abs() < 1e-10);
    assert!((r.y - 1.0).abs() < 1e-10);
    assert!((r.z - 0.0).abs() < 1e-10);
}

// --- Traits ---

#[test]
fn test_debug_display() {
    let v = Vec3::new(1.5, 2.5, 3.5);
    let s = format!("{v:?}");
    assert!(s.contains("1.5"));
    assert!(s.contains("2.5"));
    assert!(s.contains("3.5"));
}

#[test]
fn test_clone_copy() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = a;
    let c = a.clone();
    assert_eq!(a.x, b.x);
    assert_eq!(a.x, c.x);
}

#[test]
fn test_partial_eq() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(1.0, 2.0, 3.0);
    let c = Vec3::new(1.0, 2.0, 4.0);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_default_is_zero() {
    let v = Vec3::<f64>::default();
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 0.0);
    assert_eq!(v.z, 0.0);
}

// --- Integer support ---

#[test]
fn test_integer_vec3() {
    let a = Vec3::new(1_i32, 2, 3);
    let b = Vec3::new(4, 5, 6);
    let c = a + b;
    assert_eq!(c.x, 5);
    assert_eq!(c.y, 7);
    assert_eq!(c.z, 9);
    assert_eq!(a.dot(b), 32);
}
