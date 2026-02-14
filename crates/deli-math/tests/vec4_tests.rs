use deli_math::Vec4;

#[test]
fn test_new_and_fields() {
    let v = Vec4::new(1.0_f64, 2.0, 3.0, 4.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);
    assert_eq!(v.z, 3.0);
    assert_eq!(v.w, 4.0);
}

#[test]
fn test_zero() {
    let v = Vec4::<f64>::zero();
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 0.0);
    assert_eq!(v.z, 0.0);
    assert_eq!(v.w, 0.0);
}

// --- Arithmetic operators ---

#[test]
fn test_add() {
    let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
    let c = a + b;
    assert_eq!(c.x, 6.0);
    assert_eq!(c.y, 8.0);
    assert_eq!(c.z, 10.0);
    assert_eq!(c.w, 12.0);
}

#[test]
fn test_sub() {
    let a = Vec4::new(5.0, 7.0, 9.0, 11.0);
    let b = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let c = a - b;
    assert_eq!(c.x, 4.0);
    assert_eq!(c.y, 5.0);
    assert_eq!(c.z, 6.0);
    assert_eq!(c.w, 7.0);
}

#[test]
fn test_mul_scalar() {
    let v = Vec4::new(2.0, 3.0, 4.0, 5.0);
    let r = v * 3.0;
    assert_eq!(r.x, 6.0);
    assert_eq!(r.y, 9.0);
    assert_eq!(r.z, 12.0);
    assert_eq!(r.w, 15.0);
}

#[test]
fn test_div_scalar() {
    let v = Vec4::new(8.0, 6.0, 4.0, 2.0);
    let r = v / 2.0;
    assert_eq!(r.x, 4.0);
    assert_eq!(r.y, 3.0);
    assert_eq!(r.z, 2.0);
    assert_eq!(r.w, 1.0);
}

#[test]
fn test_neg() {
    let v = Vec4::new(1.0, -2.0, 3.0, -4.0);
    let r = -v;
    assert_eq!(r.x, -1.0);
    assert_eq!(r.y, 2.0);
    assert_eq!(r.z, -3.0);
    assert_eq!(r.w, 4.0);
}

#[test]
fn test_mul_componentwise() {
    let a = Vec4::new(2.0, 3.0, 4.0, 5.0);
    let b = Vec4::new(6.0, 7.0, 8.0, 9.0);
    let c = a * b;
    assert_eq!(c.x, 12.0);
    assert_eq!(c.y, 21.0);
    assert_eq!(c.z, 32.0);
    assert_eq!(c.w, 45.0);
}

#[test]
fn test_div_componentwise() {
    let a = Vec4::new(12.0, 21.0, 32.0, 45.0);
    let b = Vec4::new(6.0, 7.0, 8.0, 9.0);
    let c = a / b;
    assert_eq!(c.x, 2.0);
    assert_eq!(c.y, 3.0);
    assert_eq!(c.z, 4.0);
    assert_eq!(c.w, 5.0);
}

// --- Assign operators ---

#[test]
fn test_add_assign() {
    let mut v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    v += Vec4::new(5.0, 6.0, 7.0, 8.0);
    assert_eq!(v.x, 6.0);
    assert_eq!(v.y, 8.0);
    assert_eq!(v.z, 10.0);
    assert_eq!(v.w, 12.0);
}

#[test]
fn test_sub_assign() {
    let mut v = Vec4::new(5.0, 7.0, 9.0, 11.0);
    v -= Vec4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(v.x, 4.0);
    assert_eq!(v.y, 5.0);
    assert_eq!(v.z, 6.0);
    assert_eq!(v.w, 7.0);
}

#[test]
fn test_mul_assign_scalar() {
    let mut v = Vec4::new(2.0, 3.0, 4.0, 5.0);
    v *= 3.0;
    assert_eq!(v.x, 6.0);
    assert_eq!(v.y, 9.0);
    assert_eq!(v.z, 12.0);
    assert_eq!(v.w, 15.0);
}

#[test]
fn test_div_assign_scalar() {
    let mut v = Vec4::new(8.0, 6.0, 4.0, 2.0);
    v /= 2.0;
    assert_eq!(v.x, 4.0);
    assert_eq!(v.y, 3.0);
    assert_eq!(v.z, 2.0);
    assert_eq!(v.w, 1.0);
}

// --- Linear algebra ---

#[test]
fn test_dot() {
    let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    assert_eq!(a.dot(b), 70.0);
}

#[test]
fn test_length_squared() {
    let v = Vec4::new(1.0_f64, 2.0, 3.0, 4.0);
    // 1 + 4 + 9 + 16 = 30
    assert_eq!(v.length_squared(), 30.0);
}

#[test]
fn test_length() {
    let v = Vec4::new(2.0_f64, 0.0, 0.0, 0.0);
    assert!((v.length() - 2.0).abs() < 1e-10);
}

#[test]
fn test_normalized() {
    let v = Vec4::new(0.0_f64, 0.0, 3.0, 4.0);
    let n = v.normalized();
    assert!((n.length() - 1.0).abs() < 1e-10);
    assert!((n.x - 0.0).abs() < 1e-10);
    assert!((n.y - 0.0).abs() < 1e-10);
    assert!((n.z - 0.6).abs() < 1e-10);
    assert!((n.w - 0.8).abs() < 1e-10);
}

#[test]
fn test_distance_to() {
    let a = Vec4::new(1.0_f64, 0.0, 0.0, 0.0);
    let b = Vec4::new(4.0, 4.0, 0.0, 0.0);
    // sqrt(9+16) = 5
    assert!((a.distance_to(b) - 5.0).abs() < 1e-10);
}

#[test]
fn test_lerp() {
    let a = Vec4::new(0.0_f64, 0.0, 0.0, 0.0);
    let b = Vec4::new(10.0, 20.0, 30.0, 40.0);
    let c = a.lerp(b, 0.5);
    assert!((c.x - 5.0).abs() < 1e-10);
    assert!((c.y - 10.0).abs() < 1e-10);
    assert!((c.z - 15.0).abs() < 1e-10);
    assert!((c.w - 20.0).abs() < 1e-10);
}

// --- Traits ---

#[test]
fn test_debug_display() {
    let v = Vec4::new(1.5, 2.5, 3.5, 4.5);
    let s = format!("{v:?}");
    assert!(s.contains("1.5"));
    assert!(s.contains("2.5"));
    assert!(s.contains("3.5"));
    assert!(s.contains("4.5"));
}

#[test]
fn test_clone_copy() {
    let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let b = a;
    let c = a.clone();
    assert_eq!(a.x, b.x);
    assert_eq!(a.x, c.x);
}

#[test]
fn test_partial_eq() {
    let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let b = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let c = Vec4::new(1.0, 2.0, 3.0, 5.0);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_default_is_zero() {
    let v = Vec4::<f64>::default();
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 0.0);
    assert_eq!(v.z, 0.0);
    assert_eq!(v.w, 0.0);
}

// --- Integer support ---

#[test]
fn test_integer_vec4() {
    let a = Vec4::new(1_i32, 2, 3, 4);
    let b = Vec4::new(5, 6, 7, 8);
    let c = a + b;
    assert_eq!(c.x, 6);
    assert_eq!(c.y, 8);
    assert_eq!(c.z, 10);
    assert_eq!(c.w, 12);
    assert_eq!(a.dot(b), 70);
}
