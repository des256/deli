use deli_base::{Rect, Vec2};

// --- Construction ---

#[test]
fn test_new() {
    let r = Rect::new(Vec2::new(1.0_f64, 2.0), Vec2::new(3.0, 4.0));
    assert_eq!(r.origin, Vec2::new(1.0, 2.0));
    assert_eq!(r.size, Vec2::new(3.0, 4.0));
}

#[test]
fn test_from_min_max() {
    let r = Rect::<f64>::from_min_max(Vec2::new(1.0, 2.0), Vec2::new(4.0, 6.0));
    assert_eq!(r.origin, Vec2::new(1.0, 2.0));
    assert_eq!(r.size, Vec2::new(3.0, 4.0));
}

#[test]
fn test_zero() {
    let r = Rect::<f64>::zero();
    assert_eq!(r.origin, Vec2::new(0.0, 0.0));
    assert_eq!(r.size, Vec2::new(0.0, 0.0));
}

// --- Accessors ---

#[test]
fn test_min_max() {
    let r = Rect::new(Vec2::new(1.0_f64, 2.0), Vec2::new(3.0, 4.0));
    assert_eq!(r.min(), Vec2::new(1.0, 2.0));
    assert_eq!(r.max(), Vec2::new(4.0, 6.0));
}

#[test]
fn test_center() {
    let r = Rect::new(Vec2::new(0.0_f64, 0.0), Vec2::new(10.0, 6.0));
    let c = r.center();
    assert!((c.x - 5.0).abs() < 1e-10);
    assert!((c.y - 3.0).abs() < 1e-10);
}

#[test]
fn test_area() {
    let r = Rect::new(Vec2::new(0.0_f64, 0.0), Vec2::new(5.0, 3.0));
    assert!((r.area() - 15.0).abs() < 1e-10);
}

// --- Contains ---

#[test]
fn test_contains_point_inside() {
    let r = Rect::new(Vec2::new(0.0_f64, 0.0), Vec2::new(10.0, 10.0));
    assert!(r.contains_point(Vec2::new(5.0, 5.0)));
}

#[test]
fn test_contains_point_on_min_edge() {
    let r = Rect::new(Vec2::new(0.0_f64, 0.0), Vec2::new(10.0, 10.0));
    assert!(r.contains_point(Vec2::new(0.0, 0.0)));
}

#[test]
fn test_contains_point_on_max_edge() {
    let r = Rect::new(Vec2::new(0.0_f64, 0.0), Vec2::new(10.0, 10.0));
    // max edge is exclusive
    assert!(!r.contains_point(Vec2::new(10.0, 10.0)));
}

#[test]
fn test_contains_point_outside() {
    let r = Rect::new(Vec2::new(0.0_f64, 0.0), Vec2::new(10.0, 10.0));
    assert!(!r.contains_point(Vec2::new(11.0, 5.0)));
    assert!(!r.contains_point(Vec2::new(-1.0, 5.0)));
}

#[test]
fn test_contains_rect_fully_inside() {
    let outer = Rect::new(Vec2::new(0.0_f64, 0.0), Vec2::new(10.0, 10.0));
    let inner = Rect::new(Vec2::new(2.0, 2.0), Vec2::new(3.0, 3.0));
    assert!(outer.contains_rect(inner));
}

#[test]
fn test_contains_rect_partial_overlap() {
    let outer = Rect::new(Vec2::new(0.0_f64, 0.0), Vec2::new(10.0, 10.0));
    let partial = Rect::new(Vec2::new(5.0, 5.0), Vec2::new(10.0, 10.0));
    assert!(!outer.contains_rect(partial));
}

// --- Intersects ---

#[test]
fn test_intersects_overlapping() {
    let a = Rect::new(Vec2::new(0.0_f64, 0.0), Vec2::new(10.0, 10.0));
    let b = Rect::new(Vec2::new(5.0, 5.0), Vec2::new(10.0, 10.0));
    assert!(a.intersects(b));
    assert!(b.intersects(a));
}

#[test]
fn test_intersects_no_overlap() {
    let a = Rect::new(Vec2::new(0.0_f64, 0.0), Vec2::new(5.0, 5.0));
    let b = Rect::new(Vec2::new(6.0, 6.0), Vec2::new(5.0, 5.0));
    assert!(!a.intersects(b));
}

#[test]
fn test_intersects_touching_edge() {
    // Touching at edge but not overlapping
    let a = Rect::new(Vec2::new(0.0_f64, 0.0), Vec2::new(5.0, 5.0));
    let b = Rect::new(Vec2::new(5.0, 0.0), Vec2::new(5.0, 5.0));
    assert!(!a.intersects(b));
}

// --- Intersection ---

#[test]
fn test_intersection_overlapping() {
    let a = Rect::new(Vec2::new(0.0_f64, 0.0), Vec2::new(10.0, 10.0));
    let b = Rect::new(Vec2::new(5.0, 5.0), Vec2::new(10.0, 10.0));
    let i = a.intersection(b).unwrap();
    assert_eq!(i.origin, Vec2::new(5.0, 5.0));
    assert_eq!(i.size, Vec2::new(5.0, 5.0));
}

#[test]
fn test_intersection_no_overlap() {
    let a = Rect::new(Vec2::new(0.0_f64, 0.0), Vec2::new(5.0, 5.0));
    let b = Rect::new(Vec2::new(6.0, 6.0), Vec2::new(5.0, 5.0));
    assert!(a.intersection(b).is_none());
}

// --- Union (bounding rect) ---

#[test]
fn test_union() {
    let a = Rect::new(Vec2::new(0.0_f64, 0.0), Vec2::new(5.0, 5.0));
    let b = Rect::new(Vec2::new(3.0, 3.0), Vec2::new(7.0, 7.0));
    let u = a.union(b);
    assert_eq!(u.origin, Vec2::new(0.0, 0.0));
    assert_eq!(u.size, Vec2::new(10.0, 10.0));
}

#[test]
fn test_union_disjoint() {
    let a = Rect::new(Vec2::new(0.0_f64, 0.0), Vec2::new(2.0, 2.0));
    let b = Rect::new(Vec2::new(8.0, 8.0), Vec2::new(2.0, 2.0));
    let u = a.union(b);
    assert_eq!(u.origin, Vec2::new(0.0, 0.0));
    assert_eq!(u.size, Vec2::new(10.0, 10.0));
}

// --- Traits ---

#[test]
fn test_debug() {
    let r = Rect::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let s = format!("{r:?}");
    assert!(s.contains("origin"));
    assert!(s.contains("size"));
}

#[test]
fn test_clone_copy() {
    let a = Rect::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_partial_eq() {
    let a = Rect::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let b = Rect::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let c = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(3.0, 4.0));
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_default_is_zero() {
    let r = Rect::<f64>::default();
    assert_eq!(r.origin, Vec2::new(0.0, 0.0));
    assert_eq!(r.size, Vec2::new(0.0, 0.0));
}

// --- Integer support ---

#[test]
fn test_integer_rect() {
    let r = Rect::new(Vec2::new(1_i32, 2), Vec2::new(3, 4));
    assert_eq!(r.min(), Vec2::new(1, 2));
    assert_eq!(r.max(), Vec2::new(4, 6));
    assert_eq!(r.area(), 12);
}

#[test]
fn test_integer_contains_point() {
    let r = Rect::new(Vec2::new(0_i32, 0), Vec2::new(10, 10));
    assert!(r.contains_point(Vec2::new(5, 5)));
    assert!(!r.contains_point(Vec2::new(10, 10)));
}
