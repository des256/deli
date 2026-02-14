use deli_math::{Quat, Vec3};
use std::f64::consts::{FRAC_PI_2, PI};

const EPS: f64 = 1e-10;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
}

fn quat_approx_eq(a: Quat<f64>, b: Quat<f64>) -> bool {
    approx_eq(a.w, b.w) && approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
}

fn vec3_approx_eq(a: Vec3<f64>, b: Vec3<f64>) -> bool {
    approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
}

// --- Construction ---

#[test]
fn test_new() {
    let q = Quat::new(1.0_f64, 2.0, 3.0, 4.0);
    assert_eq!(q.w, 1.0);
    assert_eq!(q.x, 2.0);
    assert_eq!(q.y, 3.0);
    assert_eq!(q.z, 4.0);
}

#[test]
fn test_identity() {
    let q = Quat::<f64>::identity();
    assert_eq!(q.w, 1.0);
    assert_eq!(q.x, 0.0);
    assert_eq!(q.y, 0.0);
    assert_eq!(q.z, 0.0);
}

#[test]
fn test_from_axis_angle() {
    // 90 degrees around Z axis
    let q = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
    let half = FRAC_PI_2 / 2.0;
    assert!(approx_eq(q.w, half.cos()));
    assert!(approx_eq(q.x, 0.0));
    assert!(approx_eq(q.y, 0.0));
    assert!(approx_eq(q.z, half.sin()));
}

#[test]
fn test_from_axis_angle_identity() {
    // 0 degrees around any axis = identity
    let q = Quat::<f64>::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.0);
    assert!(quat_approx_eq(q, Quat::<f64>::identity()));
}

// --- Arithmetic ---

#[test]
fn test_add() {
    let a = Quat::new(1.0, 2.0, 3.0, 4.0);
    let b = Quat::new(5.0, 6.0, 7.0, 8.0);
    let c = a + b;
    assert_eq!(c.w, 6.0);
    assert_eq!(c.x, 8.0);
    assert_eq!(c.y, 10.0);
    assert_eq!(c.z, 12.0);
}

#[test]
fn test_sub() {
    let a = Quat::new(5.0, 6.0, 7.0, 8.0);
    let b = Quat::new(1.0, 2.0, 3.0, 4.0);
    let c = a - b;
    assert_eq!(c.w, 4.0);
    assert_eq!(c.x, 4.0);
    assert_eq!(c.y, 4.0);
    assert_eq!(c.z, 4.0);
}

#[test]
fn test_neg() {
    let q = Quat::new(1.0, -2.0, 3.0, -4.0);
    let r = -q;
    assert_eq!(r.w, -1.0);
    assert_eq!(r.x, 2.0);
    assert_eq!(r.y, -3.0);
    assert_eq!(r.z, 4.0);
}

#[test]
fn test_mul_scalar() {
    let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    let r = q * 2.0;
    assert_eq!(r.w, 2.0);
    assert_eq!(r.x, 4.0);
    assert_eq!(r.y, 6.0);
    assert_eq!(r.z, 8.0);
}

#[test]
fn test_div_scalar() {
    let q = Quat::new(2.0, 4.0, 6.0, 8.0);
    let r = q / 2.0;
    assert_eq!(r.w, 1.0);
    assert_eq!(r.x, 2.0);
    assert_eq!(r.y, 3.0);
    assert_eq!(r.z, 4.0);
}

// --- Quaternion multiplication (Hamilton product) ---

#[test]
fn test_mul_identity() {
    let q = Quat::new(1.0_f64, 2.0, 3.0, 4.0);
    let i = Quat::<f64>::identity();
    assert!(quat_approx_eq(q * i, q));
    assert!(quat_approx_eq(i * q, q));
}

#[test]
fn test_mul_quat() {
    // Two 90-degree rotations around Z = 180-degree rotation around Z
    let q = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
    let qq = q * q;
    let expected = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), PI);
    assert!(quat_approx_eq(qq, expected));
}

#[test]
fn test_mul_quat_noncommutative() {
    let a = Quat::<f64>::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), FRAC_PI_2);
    let b = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), FRAC_PI_2);
    // Quaternion multiplication is NOT commutative in general
    assert!(!quat_approx_eq(a * b, b * a));
}

// --- Linear algebra ---

#[test]
fn test_dot() {
    let a = Quat::new(1.0, 2.0, 3.0, 4.0);
    let b = Quat::new(5.0, 6.0, 7.0, 8.0);
    // 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
    assert_eq!(a.dot(b), 70.0);
}

#[test]
fn test_length_squared() {
    let q = Quat::new(1.0_f64, 2.0, 3.0, 4.0);
    // 1+4+9+16 = 30
    assert_eq!(q.length_squared(), 30.0);
}

#[test]
fn test_length() {
    let q = Quat::new(1.0_f64, 0.0, 0.0, 0.0);
    assert!(approx_eq(q.length(), 1.0));
}

#[test]
fn test_normalized() {
    let q = Quat::new(1.0_f64, 2.0, 3.0, 4.0);
    let n = q.normalized();
    assert!(approx_eq(n.length(), 1.0));
}

#[test]
fn test_conjugate() {
    let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    let c = q.conjugate();
    assert_eq!(c.w, 1.0);
    assert_eq!(c.x, -2.0);
    assert_eq!(c.y, -3.0);
    assert_eq!(c.z, -4.0);
}

#[test]
fn test_inverse() {
    let q = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
    let inv = q.inverse();
    let product = q * inv;
    assert!(quat_approx_eq(product, Quat::<f64>::identity()));
}

#[test]
fn test_inverse_is_conjugate_for_unit() {
    // For unit quaternions, inverse == conjugate
    let q = Quat::<f64>::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 1.0).normalized();
    let inv = q.inverse();
    let conj = q.conjugate();
    assert!(quat_approx_eq(inv, conj));
}

// --- Rotation ---

#[test]
fn test_rotate_vec3_identity() {
    let q = Quat::<f64>::identity();
    let v = Vec3::new(1.0, 2.0, 3.0);
    assert!(vec3_approx_eq(q.rotate(v), v));
}

#[test]
fn test_rotate_vec3_90_around_z() {
    // Rotating (1,0,0) by 90 degrees around Z should give (0,1,0)
    let q = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
    let v = Vec3::new(1.0, 0.0, 0.0);
    let r = q.rotate(v);
    assert!(vec3_approx_eq(r, Vec3::new(0.0, 1.0, 0.0)));
}

#[test]
fn test_rotate_vec3_180_around_z() {
    // Rotating (1,0,0) by 180 degrees around Z should give (-1,0,0)
    let q = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), PI);
    let v = Vec3::new(1.0, 0.0, 0.0);
    let r = q.rotate(v);
    assert!(vec3_approx_eq(r, Vec3::new(-1.0, 0.0, 0.0)));
}

#[test]
fn test_rotate_vec3_90_around_x() {
    // Rotating (0,1,0) by 90 degrees around X should give (0,0,1)
    let q = Quat::<f64>::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), FRAC_PI_2);
    let v = Vec3::new(0.0, 1.0, 0.0);
    let r = q.rotate(v);
    assert!(vec3_approx_eq(r, Vec3::new(0.0, 0.0, 1.0)));
}

// --- Conversion ---

#[test]
fn test_to_mat3_identity() {
    let q = Quat::<f64>::identity();
    let m = q.to_mat3();
    let i = deli_math::Mat3::<f64>::identity();
    for col in 0..3 {
        assert!(approx_eq(m.cols[col].x, i.cols[col].x));
        assert!(approx_eq(m.cols[col].y, i.cols[col].y));
        assert!(approx_eq(m.cols[col].z, i.cols[col].z));
    }
}

#[test]
fn test_to_mat3_rotation() {
    // Rotation by quat and by mat3 should produce same result
    let q = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
    let m = q.to_mat3();
    let v = Vec3::new(1.0, 0.0, 0.0);
    let rotated_q = q.rotate(v);
    let rotated_m = m * v;
    assert!(vec3_approx_eq(rotated_q, rotated_m));
}

#[test]
fn test_to_axis_angle() {
    let axis = Vec3::new(0.0_f64, 0.0, 1.0);
    let angle = FRAC_PI_2;
    let q = Quat::<f64>::from_axis_angle(axis, angle);
    let (out_axis, out_angle) = q.to_axis_angle();
    assert!(approx_eq(out_angle, angle));
    assert!(vec3_approx_eq(out_axis, axis));
}

#[test]
fn test_to_axis_angle_identity() {
    let q = Quat::<f64>::identity();
    let (_, angle) = q.to_axis_angle();
    assert!(approx_eq(angle, 0.0));
}

// --- Slerp ---

#[test]
fn test_slerp_endpoints() {
    let a = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.0);
    let b = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
    assert!(quat_approx_eq(a.slerp(b, 0.0), a));
    assert!(quat_approx_eq(a.slerp(b, 1.0), b));
}

#[test]
fn test_slerp_midpoint() {
    let a = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.0);
    let b = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
    let mid = a.slerp(b, 0.5);
    // Midpoint should be 45-degree rotation around Z
    let expected = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2 / 2.0);
    assert!(quat_approx_eq(mid, expected));
}

#[test]
fn test_slerp_same() {
    let q = Quat::<f64>::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 1.0);
    assert!(quat_approx_eq(q.slerp(q, 0.5), q));
}

// --- Traits ---

#[test]
fn test_debug() {
    let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    let s = format!("{q:?}");
    assert!(s.contains("1.0"));
    assert!(s.contains("4.0"));
}

#[test]
fn test_clone_copy() {
    let a = Quat::new(1.0, 2.0, 3.0, 4.0);
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_partial_eq() {
    let a = Quat::new(1.0, 2.0, 3.0, 4.0);
    let b = Quat::new(1.0, 2.0, 3.0, 4.0);
    let c = Quat::new(1.0, 2.0, 3.0, 5.0);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_default_is_zero() {
    let q = Quat::<f64>::default();
    assert_eq!(q.w, 0.0);
    assert_eq!(q.x, 0.0);
    assert_eq!(q.y, 0.0);
    assert_eq!(q.z, 0.0);
}
