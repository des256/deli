use deli_math::{Mat4, Pose, Quat, Vec3, Vec4};
use std::f64::consts::FRAC_PI_2;

const EPS: f64 = 1e-10;

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
}

fn v3_approx(a: Vec3<f64>, b: Vec3<f64>) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

// --- Construction ---

#[test]
fn test_new() {
    let offset = Vec3::new(1.0_f64, 2.0, 3.0);
    let rotation = Quat::<f64>::identity();
    let p = Pose::new(offset, rotation);
    assert_eq!(p.offset, offset);
    assert_eq!(p.rotation, rotation);
}

#[test]
fn test_identity() {
    let p = Pose::<f64>::identity();
    assert_eq!(p.offset, Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(p.rotation, Quat::<f64>::identity());
}

// --- transform_point: rotate then translate (child→parent) ---

#[test]
fn test_transform_point_translation_only() {
    let p = Pose::new(Vec3::new(10.0_f64, 0.0, 0.0), Quat::<f64>::identity());
    let v = Vec3::new(1.0, 2.0, 3.0);
    let result = p.transform_point(v);
    assert!(v3_approx(result, Vec3::new(11.0, 2.0, 3.0)));
}

#[test]
fn test_transform_point_rotation_only() {
    // 90 degrees around Z: (1,0,0) → (0,1,0)
    let rot = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
    let p = Pose::new(Vec3::new(0.0_f64, 0.0, 0.0), rot);
    let result = p.transform_point(Vec3::new(1.0, 0.0, 0.0));
    assert!(v3_approx(result, Vec3::new(0.0, 1.0, 0.0)));
}

#[test]
fn test_transform_point_rotation_then_translation() {
    // Rotate (1,0,0) by 90° around Z → (0,1,0), then translate by (5,0,0) → (5,1,0)
    let rot = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
    let p = Pose::new(Vec3::new(5.0_f64, 0.0, 0.0), rot);
    let result = p.transform_point(Vec3::new(1.0, 0.0, 0.0));
    assert!(v3_approx(result, Vec3::new(5.0, 1.0, 0.0)));
}

// --- transform_vector: rotate only, no translation (for directions) ---

#[test]
fn test_transform_vector_ignores_translation() {
    let p = Pose::new(Vec3::new(100.0_f64, 200.0, 300.0), Quat::<f64>::identity());
    let v = Vec3::new(1.0, 0.0, 0.0);
    let result = p.transform_vector(v);
    assert!(v3_approx(result, Vec3::new(1.0, 0.0, 0.0)));
}

#[test]
fn test_transform_vector_applies_rotation() {
    let rot = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
    let p = Pose::new(Vec3::new(100.0_f64, 200.0, 300.0), rot);
    let result = p.transform_vector(Vec3::new(1.0, 0.0, 0.0));
    assert!(v3_approx(result, Vec3::new(0.0, 1.0, 0.0)));
}

// --- inverse ---

#[test]
fn test_inverse_identity() {
    let p = Pose::<f64>::identity();
    let inv = p.inverse();
    assert!(v3_approx(inv.offset, Vec3::new(0.0, 0.0, 0.0)));
}

#[test]
fn test_inverse_translation_only() {
    let p = Pose::new(Vec3::new(5.0_f64, 3.0, 1.0), Quat::<f64>::identity());
    let inv = p.inverse();
    assert!(v3_approx(inv.offset, Vec3::new(-5.0, -3.0, -1.0)));
}

#[test]
fn test_inverse_round_trip() {
    // p.inverse().transform_point(p.transform_point(v)) == v
    let rot = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 1.2);
    let p = Pose::new(Vec3::new(3.0_f64, -4.0, 7.0), rot);
    let v = Vec3::new(1.0, 2.0, 3.0);
    let transformed = p.transform_point(v);
    let back = p.inverse().transform_point(transformed);
    assert!(v3_approx(back, v));
}

#[test]
fn test_inverse_rotation_only() {
    let rot = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
    let p = Pose::new(Vec3::new(0.0_f64, 0.0, 0.0), rot);
    let inv = p.inverse();
    // Rotating (0,1,0) by inverse (-90° around Z) should give (1,0,0)
    let result = inv.transform_point(Vec3::new(0.0, 1.0, 0.0));
    assert!(v3_approx(result, Vec3::new(1.0, 0.0, 0.0)));
}

// --- compose (Mul): chaining parent * child ---

#[test]
fn test_compose_identity() {
    let p = Pose::new(
        Vec3::new(1.0_f64, 2.0, 3.0),
        Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.5),
    );
    let i = Pose::<f64>::identity();
    let a = p * i;
    let b = i * p;
    let v = Vec3::new(7.0, -3.0, 2.0);
    assert!(v3_approx(a.transform_point(v), p.transform_point(v)));
    assert!(v3_approx(b.transform_point(v), p.transform_point(v)));
}

#[test]
fn test_compose_matches_sequential_transform() {
    // (parent * child).transform_point(v) == parent.transform_point(child.transform_point(v))
    let child = Pose::new(
        Vec3::new(1.0_f64, 0.0, 0.0),
        Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2),
    );
    let parent = Pose::new(
        Vec3::new(0.0_f64, 5.0, 0.0),
        Quat::<f64>::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), FRAC_PI_2),
    );
    let composed = parent * child;
    let v = Vec3::new(2.0, 3.0, 4.0);
    let sequential = parent.transform_point(child.transform_point(v));
    let direct = composed.transform_point(v);
    assert!(v3_approx(direct, sequential));
}

#[test]
fn test_compose_inverse_is_identity() {
    let p = Pose::new(
        Vec3::new(3.0_f64, -1.0, 5.0),
        Quat::<f64>::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.8),
    );
    let composed = p * p.inverse();
    let v = Vec3::new(7.0, -2.0, 4.0);
    assert!(v3_approx(composed.transform_point(v), v));
}

// --- to_mat4 ---

#[test]
fn test_to_mat4_identity() {
    let p = Pose::<f64>::identity();
    let m = p.to_mat4();
    assert_eq!(m, Mat4::<f64>::identity());
}

#[test]
fn test_to_mat4_matches_transform() {
    let rot = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
    let p = Pose::new(Vec3::new(5.0_f64, 3.0, 1.0), rot);
    let v = Vec3::new(1.0, 2.0, 3.0);

    let from_pose = p.transform_point(v);
    let m = p.to_mat4();
    let v4 = m * Vec4::new(v.x, v.y, v.z, 1.0);
    assert!(approx(v4.x, from_pose.x));
    assert!(approx(v4.y, from_pose.y));
    assert!(approx(v4.z, from_pose.z));
    assert!(approx(v4.w, 1.0));
}

#[test]
fn test_to_mat4_direction_vector() {
    // w=0 for direction vectors: translation should not apply
    let rot = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
    let p = Pose::new(Vec3::new(100.0_f64, 200.0, 300.0), rot);
    let m = p.to_mat4();
    let dir = m * Vec4::new(1.0, 0.0, 0.0, 0.0);
    assert!(approx(dir.x, 0.0));
    assert!(approx(dir.y, 1.0));
    assert!(approx(dir.z, 0.0));
    assert!(approx(dir.w, 0.0));
}

// --- Traits ---

#[test]
fn test_debug() {
    let p = Pose::new(Vec3::new(1.0, 2.0, 3.0), Quat::new(1.0, 0.0, 0.0, 0.0));
    let s = format!("{p:?}");
    assert!(s.contains("offset"));
    assert!(s.contains("rotation"));
}

#[test]
fn test_clone_copy() {
    let p = Pose::<f64>::identity();
    let a = p;
    let b = p.clone();
    assert_eq!(a, b);
}

#[test]
fn test_partial_eq() {
    let a = Pose::<f64>::identity();
    let b = Pose::<f64>::identity();
    let c = Pose::new(Vec3::new(1.0_f64, 0.0, 0.0), Quat::<f64>::identity());
    assert_eq!(a, b);
    assert_ne!(a, c);
}
