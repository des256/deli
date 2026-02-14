use deli_math::{Mat3, Vec3};

#[test]
fn test_from_cols() {
    let c0 = Vec3::new(1.0, 2.0, 3.0);
    let c1 = Vec3::new(4.0, 5.0, 6.0);
    let c2 = Vec3::new(7.0, 8.0, 9.0);
    let m = Mat3::from_cols(c0, c1, c2);
    assert_eq!(m.cols[0], c0);
    assert_eq!(m.cols[1], c1);
    assert_eq!(m.cols[2], c2);
}

#[test]
fn test_identity() {
    let m = Mat3::<f64>::identity();
    assert_eq!(m.cols[0], Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(m.cols[1], Vec3::new(0.0, 1.0, 0.0));
    assert_eq!(m.cols[2], Vec3::new(0.0, 0.0, 1.0));
}

#[test]
fn test_zero() {
    let m = Mat3::<f64>::zero();
    for col in 0..3 {
        assert_eq!(m.cols[col], Vec3::new(0.0, 0.0, 0.0));
    }
}

// --- Arithmetic ---

#[test]
fn test_add() {
    let a = Mat3::from_cols(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
    );
    let b = Mat3::from_cols(
        Vec3::new(9.0, 8.0, 7.0),
        Vec3::new(6.0, 5.0, 4.0),
        Vec3::new(3.0, 2.0, 1.0),
    );
    let c = a + b;
    assert_eq!(c.cols[0], Vec3::new(10.0, 10.0, 10.0));
    assert_eq!(c.cols[1], Vec3::new(10.0, 10.0, 10.0));
    assert_eq!(c.cols[2], Vec3::new(10.0, 10.0, 10.0));
}

#[test]
fn test_sub() {
    let a = Mat3::from_cols(
        Vec3::new(9.0, 8.0, 7.0),
        Vec3::new(6.0, 5.0, 4.0),
        Vec3::new(3.0, 2.0, 1.0),
    );
    let b = Mat3::from_cols(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
    );
    let c = a - b;
    assert_eq!(c.cols[0], Vec3::new(8.0, 6.0, 4.0));
    assert_eq!(c.cols[1], Vec3::new(2.0, 0.0, -2.0));
    assert_eq!(c.cols[2], Vec3::new(-4.0, -6.0, -8.0));
}

#[test]
fn test_mul_scalar() {
    let m = Mat3::from_cols(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
    );
    let r = m * 2.0;
    assert_eq!(r.cols[0], Vec3::new(2.0, 4.0, 6.0));
    assert_eq!(r.cols[1], Vec3::new(8.0, 10.0, 12.0));
    assert_eq!(r.cols[2], Vec3::new(14.0, 16.0, 18.0));
}

#[test]
fn test_neg() {
    let m = Mat3::from_cols(
        Vec3::new(1.0, -2.0, 3.0),
        Vec3::new(-4.0, 5.0, -6.0),
        Vec3::new(7.0, -8.0, 9.0),
    );
    let r = -m;
    assert_eq!(r.cols[0], Vec3::new(-1.0, 2.0, -3.0));
    assert_eq!(r.cols[1], Vec3::new(4.0, -5.0, 6.0));
    assert_eq!(r.cols[2], Vec3::new(-7.0, 8.0, -9.0));
}

#[test]
fn test_mul_vec3() {
    // Identity * v = v
    let m = Mat3::<f64>::identity();
    let v = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(m * v, v);
}

#[test]
fn test_mul_vec3_nonidentity() {
    // | 2 0 0 |   | 1 |   | 2 |
    // | 0 3 0 | * | 2 | = | 6 |
    // | 0 0 4 |   | 3 |   | 12|
    let m = Mat3::from_cols(
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 3.0, 0.0),
        Vec3::new(0.0, 0.0, 4.0),
    );
    let v = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(m * v, Vec3::new(2.0, 6.0, 12.0));
}

#[test]
fn test_mul_mat3() {
    let a = Mat3::<f64>::identity();
    let b = Mat3::from_cols(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
    );
    assert_eq!(a * b, b);
    assert_eq!(b * a, b);
}

#[test]
fn test_mul_mat3_nonidentity() {
    // A = |1 4 7|   B = |2 0 0|
    //     |2 5 8|       |0 2 0|
    //     |3 6 9|       |0 0 2|
    // AB = 2*A
    let a = Mat3::from_cols(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
    );
    let b = Mat3::from_cols(
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.0, 0.0, 2.0),
    );
    let c = a * b;
    assert_eq!(c.cols[0], Vec3::new(2.0, 4.0, 6.0));
    assert_eq!(c.cols[1], Vec3::new(8.0, 10.0, 12.0));
    assert_eq!(c.cols[2], Vec3::new(14.0, 16.0, 18.0));
}

// --- Linear algebra ---

#[test]
fn test_transpose() {
    let m = Mat3::from_cols(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
    );
    let t = m.transpose();
    assert_eq!(t.cols[0], Vec3::new(1.0, 4.0, 7.0));
    assert_eq!(t.cols[1], Vec3::new(2.0, 5.0, 8.0));
    assert_eq!(t.cols[2], Vec3::new(3.0, 6.0, 9.0));
}

#[test]
fn test_determinant() {
    // | 1 4 7 |
    // | 2 5 8 | => det = 0 (singular)
    // | 3 6 9 |
    let m = Mat3::from_cols(
        Vec3::new(1.0_f64, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
    );
    assert!((m.determinant() - 0.0).abs() < 1e-10);
}

#[test]
fn test_determinant_nonzero() {
    // | 1 0 0 |
    // | 0 2 0 | => det = 6
    // | 0 0 3 |
    let m = Mat3::from_cols(
        Vec3::new(1.0_f64, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.0, 0.0, 3.0),
    );
    assert!((m.determinant() - 6.0).abs() < 1e-10);
}

#[test]
fn test_inverse() {
    let m = Mat3::from_cols(
        Vec3::new(1.0_f64, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.0, 0.0, 4.0),
    );
    let inv = m.inverse().unwrap();
    let product = m * inv;
    let i = Mat3::<f64>::identity();
    for col in 0..3 {
        assert!((product.cols[col].x - i.cols[col].x).abs() < 1e-10);
        assert!((product.cols[col].y - i.cols[col].y).abs() < 1e-10);
        assert!((product.cols[col].z - i.cols[col].z).abs() < 1e-10);
    }
}

#[test]
fn test_inverse_general() {
    let m = Mat3::from_cols(
        Vec3::new(2.0_f64, 1.0, 1.0),
        Vec3::new(1.0, 3.0, 2.0),
        Vec3::new(1.0, 0.0, 0.0),
    );
    let inv = m.inverse().unwrap();
    let product = m * inv;
    let i = Mat3::<f64>::identity();
    for col in 0..3 {
        assert!((product.cols[col].x - i.cols[col].x).abs() < 1e-10);
        assert!((product.cols[col].y - i.cols[col].y).abs() < 1e-10);
        assert!((product.cols[col].z - i.cols[col].z).abs() < 1e-10);
    }
}

#[test]
fn test_inverse_singular() {
    let m = Mat3::from_cols(
        Vec3::new(1.0_f64, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
    );
    assert!(m.inverse().is_none());
}

// --- Traits ---

#[test]
fn test_debug() {
    let m = Mat3::from_cols(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
    );
    let s = format!("{m:?}");
    assert!(s.contains("1.0"));
    assert!(s.contains("9.0"));
}

#[test]
fn test_clone_copy_eq() {
    let a = Mat3::from_cols(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
    );
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

// --- Integer support ---

#[test]
fn test_integer_mat3() {
    let a = Mat3::from_cols(
        Vec3::new(1_i32, 2, 3),
        Vec3::new(4, 5, 6),
        Vec3::new(7, 8, 9),
    );
    let b = Mat3::from_cols(
        Vec3::new(9, 8, 7),
        Vec3::new(6, 5, 4),
        Vec3::new(3, 2, 1),
    );
    let c = a + b;
    assert_eq!(c.cols[0], Vec3::new(10, 10, 10));
}
