use codec::{Codec, DecodeError};

#[derive(Debug, PartialEq, Codec)]
enum Color {
    Red,
    Green,
    Blue,
}

#[derive(Debug, PartialEq, Codec)]
enum Shape {
    Circle { radius: f32 },
    Rectangle { width: f32, height: f32 },
    Point,
}

#[derive(Debug, PartialEq, Codec)]
enum Value {
    Int(i64),
    Float(f64),
    Text(String),
    Nothing,
}

#[derive(Debug, PartialEq, Codec)]
struct Drawing {
    shape: Shape,
    color: Color,
    label: String,
}

// --- Unit variants ---

#[test]
fn test_unit_variants() {
    let bytes = Color::Red.to_bytes();
    assert_eq!(Color::from_bytes(&bytes).unwrap(), Color::Red);

    let bytes = Color::Green.to_bytes();
    assert_eq!(Color::from_bytes(&bytes).unwrap(), Color::Green);

    let bytes = Color::Blue.to_bytes();
    assert_eq!(Color::from_bytes(&bytes).unwrap(), Color::Blue);
}

#[test]
fn test_unit_variant_encoding() {
    // Unit variants should be just a u32 discriminant
    let mut buf = Vec::new();
    Color::Red.encode(&mut buf);
    assert_eq!(buf, vec![0, 0, 0, 0]); // variant 0

    buf.clear();
    Color::Blue.encode(&mut buf);
    assert_eq!(buf, vec![2, 0, 0, 0]); // variant 2
}

// --- Named field variants ---

#[test]
fn test_named_variant_circle() {
    let s = Shape::Circle { radius: 5.0 };
    let bytes = s.to_bytes();
    let decoded = Shape::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, s);
}

#[test]
fn test_named_variant_rectangle() {
    let s = Shape::Rectangle {
        width: 10.0,
        height: 20.0,
    };
    let bytes = s.to_bytes();
    let decoded = Shape::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, s);
}

#[test]
fn test_named_variant_point() {
    let s = Shape::Point;
    let bytes = s.to_bytes();
    let decoded = Shape::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, s);
}

// --- Tuple variants ---

#[test]
fn test_tuple_variant_int() {
    let v = Value::Int(-42);
    let bytes = v.to_bytes();
    let decoded = Value::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, v);
}

#[test]
fn test_tuple_variant_text() {
    let v = Value::Text("hello world".to_string());
    let bytes = v.to_bytes();
    let decoded = Value::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, v);
}

#[test]
fn test_tuple_variant_nothing() {
    let v = Value::Nothing;
    let bytes = v.to_bytes();
    let decoded = Value::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, v);
}

// --- Nested in struct ---

#[test]
fn test_enum_in_struct() {
    let d = Drawing {
        shape: Shape::Circle { radius: 3.0 },
        color: Color::Green,
        label: "my circle".to_string(),
    };
    let bytes = d.to_bytes();
    let decoded = Drawing::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, d);
}

// --- Sequential ---

#[test]
fn test_sequential_enum_values() {
    let mut buf = Vec::new();
    Value::Int(1).encode(&mut buf);
    Value::Float(2.5).encode(&mut buf);
    Value::Text("hi".to_string()).encode(&mut buf);
    Value::Nothing.encode(&mut buf);

    let mut pos = 0;
    assert_eq!(Value::decode(&buf, &mut pos).unwrap(), Value::Int(1));
    assert_eq!(Value::decode(&buf, &mut pos).unwrap(), Value::Float(2.5));
    assert_eq!(
        Value::decode(&buf, &mut pos).unwrap(),
        Value::Text("hi".to_string())
    );
    assert_eq!(Value::decode(&buf, &mut pos).unwrap(), Value::Nothing);
    assert_eq!(pos, buf.len());
}

// --- Error cases ---

#[test]
fn test_invalid_discriminant() {
    // Color has 3 variants (0,1,2). Discriminant 99 is invalid.
    let mut buf = Vec::new();
    99u32.encode(&mut buf);
    let mut pos = 0;
    assert!(matches!(
        Color::decode(&buf, &mut pos),
        Err(DecodeError::InvalidVariant(99))
    ));
}

#[test]
fn test_truncated_variant_data() {
    // Encode Circle discriminant (0) but truncate the radius
    let mut buf = Vec::new();
    0u32.encode(&mut buf);
    buf.push(0); // only 1 byte of f32 instead of 4
    let mut pos = 0;
    assert!(matches!(
        Shape::decode(&buf, &mut pos),
        Err(DecodeError::UnexpectedEof)
    ));
}
