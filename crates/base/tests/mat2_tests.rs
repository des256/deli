use base::{Mat2, Vec2};

#[test]
fn test_from_cols() {
    let m = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    assert_eq!(m.cols[0], Vec2::new(1.0, 2.0));
    assert_eq!(m.cols[1], Vec2::new(3.0, 4.0));
}

#[test]
fn test_identity() {
    let m = Mat2::<f64>::identity();
    assert_eq!(m.cols[0], Vec2::new(1.0, 0.0));
    assert_eq!(m.cols[1], Vec2::new(0.0, 1.0));
}

#[test]
fn test_zero() {
    let m = Mat2::<f64>::zero();
    assert_eq!(m.cols[0], Vec2::new(0.0, 0.0));
    assert_eq!(m.cols[1], Vec2::new(0.0, 0.0));
}

// --- Arithmetic ---

#[test]
fn test_add() {
    let a = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let b = Mat2::from_cols(Vec2::new(5.0, 6.0), Vec2::new(7.0, 8.0));
    let c = a + b;
    assert_eq!(c.cols[0], Vec2::new(6.0, 8.0));
    assert_eq!(c.cols[1], Vec2::new(10.0, 12.0));
}

#[test]
fn test_sub() {
    let a = Mat2::from_cols(Vec2::new(5.0, 6.0), Vec2::new(7.0, 8.0));
    let b = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let c = a - b;
    assert_eq!(c.cols[0], Vec2::new(4.0, 4.0));
    assert_eq!(c.cols[1], Vec2::new(4.0, 4.0));
}

#[test]
fn test_mul_scalar() {
    let m = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let r = m * 2.0;
    assert_eq!(r.cols[0], Vec2::new(2.0, 4.0));
    assert_eq!(r.cols[1], Vec2::new(6.0, 8.0));
}

#[test]
fn test_neg() {
    let m = Mat2::from_cols(Vec2::new(1.0, -2.0), Vec2::new(3.0, -4.0));
    let r = -m;
    assert_eq!(r.cols[0], Vec2::new(-1.0, 2.0));
    assert_eq!(r.cols[1], Vec2::new(-3.0, 4.0));
}

#[test]
fn test_mul_vec2() {
    // | 1 3 |   | 5 |   | 1*5+3*6 |   | 23 |
    // | 2 4 | * | 6 | = | 2*5+4*6 | = | 34 |
    let m = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let v = Vec2::new(5.0, 6.0);
    let r = m * v;
    assert_eq!(r, Vec2::new(23.0, 34.0));
}

#[test]
fn test_mul_mat2() {
    // A = |1 3|  B = |5 7|
    //     |2 4|      |6 8|
    // AB = |1*5+3*6  1*7+3*8|   |23 31|
    //      |2*5+4*6  2*7+4*8| = |34 46|
    let a = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let b = Mat2::from_cols(Vec2::new(5.0, 6.0), Vec2::new(7.0, 8.0));
    let c = a * b;
    assert_eq!(c.cols[0], Vec2::new(23.0, 34.0));
    assert_eq!(c.cols[1], Vec2::new(31.0, 46.0));
}

#[test]
fn test_identity_mul() {
    let m = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let i = Mat2::<f64>::identity();
    assert_eq!(i * m, m);
    assert_eq!(m * i, m);
}

// --- Linear algebra ---

#[test]
fn test_transpose() {
    // | 1 3 |^T = | 1 2 |
    // | 2 4 |     | 3 4 |
    let m = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let t = m.transpose();
    assert_eq!(t.cols[0], Vec2::new(1.0, 3.0));
    assert_eq!(t.cols[1], Vec2::new(2.0, 4.0));
}

#[test]
fn test_determinant() {
    // | 1 3 | => det = 1*4 - 3*2 = -2
    // | 2 4 |
    let m = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    assert_eq!(m.determinant(), -2.0);
}

#[test]
fn test_inverse() {
    let m = Mat2::from_cols(Vec2::new(1.0_f64, 2.0), Vec2::new(3.0, 4.0));
    let inv = m.inverse().unwrap();
    let product = m * inv;
    let i = Mat2::<f64>::identity();
    for col in 0..2 {
        assert!((product.cols[col].x - i.cols[col].x).abs() < 1e-10);
        assert!((product.cols[col].y - i.cols[col].y).abs() < 1e-10);
    }
}

#[test]
fn test_inverse_singular() {
    // Singular matrix (cols are linearly dependent)
    let m = Mat2::from_cols(Vec2::new(1.0_f64, 2.0), Vec2::new(2.0, 4.0));
    assert!(m.inverse().is_none());
}

// --- Traits ---

#[test]
fn test_debug() {
    let m = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let s = format!("{m:?}");
    assert!(s.contains("1.0"));
    assert!(s.contains("4.0"));
}

#[test]
fn test_clone_copy() {
    let a = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_partial_eq() {
    let a = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let b = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let c = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 5.0));
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// --- Integer support ---

#[test]
fn test_integer_mat2() {
    let a = Mat2::from_cols(Vec2::new(1_i32, 2), Vec2::new(3, 4));
    let b = Mat2::from_cols(Vec2::new(5, 6), Vec2::new(7, 8));
    let c = a + b;
    assert_eq!(c.cols[0], Vec2::new(6, 8));
    assert_eq!(c.cols[1], Vec2::new(10, 12));
}
