use deli_base::Vec2;

#[test]
fn test_new_and_fields() {
    let v = Vec2::new(3.0_f64, 4.0);
    assert_eq!(v.x, 3.0);
    assert_eq!(v.y, 4.0);
}

#[test]
fn test_zero() {
    let v = Vec2::<f64>::zero();
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 0.0);
}

// --- Arithmetic operators ---

#[test]
fn test_add() {
    let a = Vec2::new(1.0, 2.0);
    let b = Vec2::new(3.0, 4.0);
    let c = a + b;
    assert_eq!(c.x, 4.0);
    assert_eq!(c.y, 6.0);
}

#[test]
fn test_sub() {
    let a = Vec2::new(5.0, 7.0);
    let b = Vec2::new(2.0, 3.0);
    let c = a - b;
    assert_eq!(c.x, 3.0);
    assert_eq!(c.y, 4.0);
}

#[test]
fn test_mul_scalar() {
    let v = Vec2::new(2.0, 3.0);
    let r = v * 4.0;
    assert_eq!(r.x, 8.0);
    assert_eq!(r.y, 12.0);
}

#[test]
fn test_div_scalar() {
    let v = Vec2::new(8.0, 6.0);
    let r = v / 2.0;
    assert_eq!(r.x, 4.0);
    assert_eq!(r.y, 3.0);
}

#[test]
fn test_neg() {
    let v = Vec2::new(3.0, -4.0);
    let r = -v;
    assert_eq!(r.x, -3.0);
    assert_eq!(r.y, 4.0);
}

#[test]
fn test_mul_componentwise() {
    let a = Vec2::new(2.0, 3.0);
    let b = Vec2::new(4.0, 5.0);
    let c = a * b;
    assert_eq!(c.x, 8.0);
    assert_eq!(c.y, 15.0);
}

#[test]
fn test_div_componentwise() {
    let a = Vec2::new(8.0, 15.0);
    let b = Vec2::new(4.0, 5.0);
    let c = a / b;
    assert_eq!(c.x, 2.0);
    assert_eq!(c.y, 3.0);
}

// --- Assign operators ---

#[test]
fn test_add_assign() {
    let mut v = Vec2::new(1.0, 2.0);
    v += Vec2::new(3.0, 4.0);
    assert_eq!(v.x, 4.0);
    assert_eq!(v.y, 6.0);
}

#[test]
fn test_sub_assign() {
    let mut v = Vec2::new(5.0, 7.0);
    v -= Vec2::new(2.0, 3.0);
    assert_eq!(v.x, 3.0);
    assert_eq!(v.y, 4.0);
}

#[test]
fn test_mul_assign_scalar() {
    let mut v = Vec2::new(2.0, 3.0);
    v *= 4.0;
    assert_eq!(v.x, 8.0);
    assert_eq!(v.y, 12.0);
}

#[test]
fn test_div_assign_scalar() {
    let mut v = Vec2::new(8.0, 6.0);
    v /= 2.0;
    assert_eq!(v.x, 4.0);
    assert_eq!(v.y, 3.0);
}

// --- Linear algebra ---

#[test]
fn test_dot() {
    let a = Vec2::new(1.0, 2.0);
    let b = Vec2::new(3.0, 4.0);
    assert_eq!(a.dot(b), 11.0);
}

#[test]
fn test_cross() {
    let a = Vec2::new(1.0, 2.0);
    let b = Vec2::new(3.0, 4.0);
    // cross = a.x*b.y - a.y*b.x = 4 - 6 = -2
    assert_eq!(a.cross(b), -2.0);
}

#[test]
fn test_length_squared() {
    let v = Vec2::new(3.0_f64, 4.0);
    assert_eq!(v.length_squared(), 25.0);
}

#[test]
fn test_length() {
    let v = Vec2::new(3.0_f64, 4.0);
    assert!((v.length() - 5.0).abs() < 1e-10);
}

#[test]
fn test_normalized() {
    let v = Vec2::new(3.0_f64, 4.0);
    let n = v.normalized();
    assert!((n.length() - 1.0).abs() < 1e-10);
    assert!((n.x - 0.6).abs() < 1e-10);
    assert!((n.y - 0.8).abs() < 1e-10);
}

#[test]
fn test_distance_to() {
    let a = Vec2::new(1.0_f64, 2.0);
    let b = Vec2::new(4.0, 6.0);
    assert!((a.distance_to(b) - 5.0).abs() < 1e-10);
}

#[test]
fn test_lerp() {
    let a = Vec2::new(0.0_f64, 0.0);
    let b = Vec2::new(10.0, 20.0);
    let c = a.lerp(b, 0.25);
    assert!((c.x - 2.5).abs() < 1e-10);
    assert!((c.y - 5.0).abs() < 1e-10);
}

#[test]
fn test_perp() {
    let v = Vec2::new(3.0, 4.0);
    let p = v.perp();
    assert_eq!(p.x, -4.0);
    assert_eq!(p.y, 3.0);
    // perpendicular => dot == 0
    assert_eq!(v.dot(p), 0.0);
}

#[test]
fn test_reflect() {
    // Reflect (1, -1) over horizontal normal (0, 1)
    let v = Vec2::new(1.0_f64, -1.0);
    let n = Vec2::new(0.0, 1.0);
    let r = v.reflect(n);
    assert!((r.x - 1.0).abs() < 1e-10);
    assert!((r.y - 1.0).abs() < 1e-10);
}

// --- Traits ---

#[test]
fn test_debug_display() {
    let v = Vec2::new(1.5, 2.5);
    let s = format!("{v:?}");
    assert!(s.contains("1.5"));
    assert!(s.contains("2.5"));
}

#[test]
fn test_clone_copy() {
    let a = Vec2::new(1.0, 2.0);
    let b = a;
    let c = a.clone();
    assert_eq!(a.x, b.x);
    assert_eq!(a.x, c.x);
}

#[test]
fn test_partial_eq() {
    let a = Vec2::new(1.0, 2.0);
    let b = Vec2::new(1.0, 2.0);
    let c = Vec2::new(1.0, 3.0);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_default_is_zero() {
    let v = Vec2::<f64>::default();
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 0.0);
}

// --- Integer support ---

#[test]
fn test_integer_vec2() {
    let a = Vec2::new(1_i32, 2);
    let b = Vec2::new(3, 4);
    let c = a + b;
    assert_eq!(c.x, 4);
    assert_eq!(c.y, 6);
    assert_eq!(a.dot(b), 11);
    assert_eq!(a.cross(b), -2);
}
