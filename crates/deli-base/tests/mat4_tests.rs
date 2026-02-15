use deli_base::{Mat4, Vec4};

#[test]
fn test_from_cols() {
    let c0 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let c1 = Vec4::new(5.0, 6.0, 7.0, 8.0);
    let c2 = Vec4::new(9.0, 10.0, 11.0, 12.0);
    let c3 = Vec4::new(13.0, 14.0, 15.0, 16.0);
    let m = Mat4::from_cols(c0, c1, c2, c3);
    assert_eq!(m.cols[0], c0);
    assert_eq!(m.cols[1], c1);
    assert_eq!(m.cols[2], c2);
    assert_eq!(m.cols[3], c3);
}

#[test]
fn test_identity() {
    let m = Mat4::<f64>::identity();
    assert_eq!(m.cols[0], Vec4::new(1.0, 0.0, 0.0, 0.0));
    assert_eq!(m.cols[1], Vec4::new(0.0, 1.0, 0.0, 0.0));
    assert_eq!(m.cols[2], Vec4::new(0.0, 0.0, 1.0, 0.0));
    assert_eq!(m.cols[3], Vec4::new(0.0, 0.0, 0.0, 1.0));
}

#[test]
fn test_zero() {
    let m = Mat4::<f64>::zero();
    for col in 0..4 {
        assert_eq!(m.cols[col], Vec4::new(0.0, 0.0, 0.0, 0.0));
    }
}

// --- Arithmetic ---

#[test]
fn test_add() {
    let a = Mat4::from_cols(
        Vec4::new(1.0, 2.0, 3.0, 4.0),
        Vec4::new(5.0, 6.0, 7.0, 8.0),
        Vec4::new(9.0, 10.0, 11.0, 12.0),
        Vec4::new(13.0, 14.0, 15.0, 16.0),
    );
    let b = a;
    let c = a + b;
    assert_eq!(c.cols[0], Vec4::new(2.0, 4.0, 6.0, 8.0));
    assert_eq!(c.cols[3], Vec4::new(26.0, 28.0, 30.0, 32.0));
}

#[test]
fn test_sub() {
    let a = Mat4::from_cols(
        Vec4::new(10.0, 10.0, 10.0, 10.0),
        Vec4::new(10.0, 10.0, 10.0, 10.0),
        Vec4::new(10.0, 10.0, 10.0, 10.0),
        Vec4::new(10.0, 10.0, 10.0, 10.0),
    );
    let b = Mat4::from_cols(
        Vec4::new(1.0, 2.0, 3.0, 4.0),
        Vec4::new(5.0, 6.0, 7.0, 8.0),
        Vec4::new(9.0, 10.0, 11.0, 12.0),
        Vec4::new(13.0, 14.0, 15.0, 16.0),
    );
    let c = a - b;
    assert_eq!(c.cols[0], Vec4::new(9.0, 8.0, 7.0, 6.0));
    assert_eq!(c.cols[3], Vec4::new(-3.0, -4.0, -5.0, -6.0));
}

#[test]
fn test_mul_scalar() {
    let m = Mat4::<f64>::identity();
    let r = m * 5.0;
    assert_eq!(r.cols[0], Vec4::new(5.0, 0.0, 0.0, 0.0));
    assert_eq!(r.cols[1], Vec4::new(0.0, 5.0, 0.0, 0.0));
}

#[test]
fn test_neg() {
    let m = Mat4::<f64>::identity();
    let r = -m;
    assert_eq!(r.cols[0], Vec4::new(-1.0, 0.0, 0.0, 0.0));
    assert_eq!(r.cols[1], Vec4::new(0.0, -1.0, 0.0, 0.0));
}

#[test]
fn test_mul_vec4() {
    let m = Mat4::<f64>::identity();
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(m * v, v);
}

#[test]
fn test_mul_vec4_scale() {
    // Diagonal scale matrix
    let m = Mat4::from_cols(
        Vec4::new(2.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 3.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 4.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 5.0),
    );
    let v = Vec4::new(1.0, 1.0, 1.0, 1.0);
    assert_eq!(m * v, Vec4::new(2.0, 3.0, 4.0, 5.0));
}

#[test]
fn test_mul_mat4_identity() {
    let m = Mat4::from_cols(
        Vec4::new(1.0, 2.0, 3.0, 4.0),
        Vec4::new(5.0, 6.0, 7.0, 8.0),
        Vec4::new(9.0, 10.0, 11.0, 12.0),
        Vec4::new(13.0, 14.0, 15.0, 16.0),
    );
    let i = Mat4::<f64>::identity();
    assert_eq!(i * m, m);
    assert_eq!(m * i, m);
}

#[test]
fn test_mul_mat4() {
    // 2*I * A = 2*A
    let a = Mat4::from_cols(
        Vec4::new(1.0, 2.0, 3.0, 4.0),
        Vec4::new(5.0, 6.0, 7.0, 8.0),
        Vec4::new(9.0, 10.0, 11.0, 12.0),
        Vec4::new(13.0, 14.0, 15.0, 16.0),
    );
    let two_i = Mat4::from_cols(
        Vec4::new(2.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 2.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 2.0),
    );
    let c = two_i * a;
    assert_eq!(c.cols[0], Vec4::new(2.0, 4.0, 6.0, 8.0));
    assert_eq!(c.cols[3], Vec4::new(26.0, 28.0, 30.0, 32.0));
}

// --- Linear algebra ---

#[test]
fn test_transpose() {
    let m = Mat4::from_cols(
        Vec4::new(1.0, 5.0, 9.0, 13.0),
        Vec4::new(2.0, 6.0, 10.0, 14.0),
        Vec4::new(3.0, 7.0, 11.0, 15.0),
        Vec4::new(4.0, 8.0, 12.0, 16.0),
    );
    let t = m.transpose();
    assert_eq!(t.cols[0], Vec4::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(t.cols[1], Vec4::new(5.0, 6.0, 7.0, 8.0));
    assert_eq!(t.cols[2], Vec4::new(9.0, 10.0, 11.0, 12.0));
    assert_eq!(t.cols[3], Vec4::new(13.0, 14.0, 15.0, 16.0));
}

#[test]
fn test_determinant_identity() {
    let m = Mat4::<f64>::identity();
    assert!((m.determinant() - 1.0).abs() < 1e-10);
}

#[test]
fn test_determinant_diagonal() {
    let m = Mat4::from_cols(
        Vec4::new(2.0_f64, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 3.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 4.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 5.0),
    );
    assert!((m.determinant() - 120.0).abs() < 1e-10);
}

#[test]
fn test_determinant_singular() {
    // All rows equal → det = 0
    let m = Mat4::from_cols(
        Vec4::new(1.0_f64, 1.0, 1.0, 1.0),
        Vec4::new(2.0, 2.0, 2.0, 2.0),
        Vec4::new(3.0, 3.0, 3.0, 3.0),
        Vec4::new(4.0, 4.0, 4.0, 4.0),
    );
    assert!(m.determinant().abs() < 1e-10);
}

#[test]
fn test_inverse_identity() {
    let m = Mat4::<f64>::identity();
    let inv = m.inverse().unwrap();
    assert_eq!(inv, m);
}

#[test]
fn test_inverse_diagonal() {
    let m = Mat4::from_cols(
        Vec4::new(2.0_f64, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 4.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 5.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 10.0),
    );
    let inv = m.inverse().unwrap();
    let product = m * inv;
    let i = Mat4::<f64>::identity();
    for col in 0..4 {
        assert!((product.cols[col].x - i.cols[col].x).abs() < 1e-10);
        assert!((product.cols[col].y - i.cols[col].y).abs() < 1e-10);
        assert!((product.cols[col].z - i.cols[col].z).abs() < 1e-10);
        assert!((product.cols[col].w - i.cols[col].w).abs() < 1e-10);
    }
}

#[test]
fn test_inverse_general() {
    let m = Mat4::from_cols(
        Vec4::new(1.0_f64, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(3.0, 4.0, 5.0, 1.0),
    );
    let inv = m.inverse().unwrap();
    let product = m * inv;
    let i = Mat4::<f64>::identity();
    for col in 0..4 {
        assert!((product.cols[col].x - i.cols[col].x).abs() < 1e-10);
        assert!((product.cols[col].y - i.cols[col].y).abs() < 1e-10);
        assert!((product.cols[col].z - i.cols[col].z).abs() < 1e-10);
        assert!((product.cols[col].w - i.cols[col].w).abs() < 1e-10);
    }
}

#[test]
fn test_inverse_singular() {
    let m = Mat4::from_cols(
        Vec4::new(1.0_f64, 1.0, 1.0, 1.0),
        Vec4::new(2.0, 2.0, 2.0, 2.0),
        Vec4::new(3.0, 3.0, 3.0, 3.0),
        Vec4::new(4.0, 4.0, 4.0, 4.0),
    );
    assert!(m.inverse().is_none());
}

// --- Traits ---

#[test]
fn test_debug() {
    let m = Mat4::<f64>::identity();
    let s = format!("{m:?}");
    assert!(s.contains("1.0"));
}

#[test]
fn test_clone_copy_eq() {
    let a = Mat4::<f64>::identity();
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

// --- Integer support ---

#[test]
fn test_integer_mat4() {
    let a = Mat4::from_cols(
        Vec4::new(1_i32, 2, 3, 4),
        Vec4::new(5, 6, 7, 8),
        Vec4::new(9, 10, 11, 12),
        Vec4::new(13, 14, 15, 16),
    );
    let b = a;
    let c = a + b;
    assert_eq!(c.cols[0], Vec4::new(2, 4, 6, 8));
}
