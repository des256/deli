use codec::{Codec, DecodeError};

#[derive(Debug, PartialEq, Codec)]
struct Point {
    x: f32,
    y: f32,
}

#[derive(Debug, PartialEq, Codec)]
struct Empty;

#[derive(Debug, PartialEq, Codec)]
struct Named {
    id: u32,
    label: String,
    active: bool,
}

#[derive(Debug, PartialEq, Codec)]
struct Nested {
    origin: Point,
    name: String,
}

// --- Round-trip tests ---

#[test]
fn test_derive_simple_struct() {
    let p = Point { x: 1.5, y: -3.0 };
    let bytes = p.to_bytes();
    let decoded = Point::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, p);
}

#[test]
fn test_derive_empty_struct() {
    let e = Empty;
    let bytes = e.to_bytes();
    assert!(bytes.is_empty());
    let decoded = Empty::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, e);
}

#[test]
fn test_derive_mixed_fields() {
    let n = Named {
        id: 42,
        label: "hello".to_string(),
        active: true,
    };
    let bytes = n.to_bytes();
    let decoded = Named::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, n);
}

#[test]
fn test_derive_nested_struct() {
    let n = Nested {
        origin: Point { x: 1.0, y: 2.0 },
        name: "origin".to_string(),
    };
    let bytes = n.to_bytes();
    let decoded = Nested::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, n);
}

#[test]
fn test_derive_encoding_is_sequential() {
    let p = Point { x: 1.0, y: 2.0 };
    let mut buf = Vec::new();
    p.encode(&mut buf);

    // Should be f32 LE for x, then f32 LE for y
    let mut pos = 0;
    let x = f32::decode(&buf, &mut pos).unwrap();
    let y = f32::decode(&buf, &mut pos).unwrap();
    assert_eq!(x, 1.0);
    assert_eq!(y, 2.0);
    assert_eq!(pos, buf.len());
}

#[test]
fn test_derive_truncated_buffer() {
    let p = Point { x: 1.0, y: 2.0 };
    let bytes = p.to_bytes();
    // Chop off last byte
    let truncated = &bytes[..bytes.len() - 1];
    let mut pos = 0;
    assert!(matches!(
        Point::decode(truncated, &mut pos),
        Err(DecodeError::UnexpectedEof)
    ));
}

#[test]
fn test_derive_multiple_in_sequence() {
    let mut buf = Vec::new();
    let p1 = Point { x: 1.0, y: 2.0 };
    let p2 = Point { x: 3.0, y: 4.0 };
    p1.encode(&mut buf);
    p2.encode(&mut buf);

    let mut pos = 0;
    let d1 = Point::decode(&buf, &mut pos).unwrap();
    let d2 = Point::decode(&buf, &mut pos).unwrap();
    assert_eq!(d1, p1);
    assert_eq!(d2, p2);
    assert_eq!(pos, buf.len());
}
