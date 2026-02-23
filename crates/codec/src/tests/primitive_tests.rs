use codec::{Codec, DecodeError};

// --- Helper ---

fn round_trip<T: Codec + PartialEq + std::fmt::Debug>(value: &T) {
    let mut buf = Vec::new();
    value.encode(&mut buf);
    let mut pos = 0;
    let decoded = T::decode(&buf, &mut pos).unwrap();
    assert_eq!(&decoded, value);
    assert_eq!(pos, buf.len(), "all bytes should be consumed");
}

// --- bool ---

#[test]
fn test_bool_true() {
    round_trip(&true);
}

#[test]
fn test_bool_false() {
    round_trip(&false);
}

#[test]
fn test_bool_encoding() {
    let mut buf = Vec::new();
    true.encode(&mut buf);
    assert_eq!(buf, vec![1]);

    buf.clear();
    false.encode(&mut buf);
    assert_eq!(buf, vec![0]);
}

// --- u8 ---

#[test]
fn test_u8() {
    round_trip(&0u8);
    round_trip(&127u8);
    round_trip(&255u8);
}

#[test]
fn test_u8_encoding() {
    let mut buf = Vec::new();
    42u8.encode(&mut buf);
    assert_eq!(buf, vec![42]);
}

// --- u16 ---

#[test]
fn test_u16() {
    round_trip(&0u16);
    round_trip(&256u16);
    round_trip(&u16::MAX);
}

#[test]
fn test_u16_little_endian() {
    let mut buf = Vec::new();
    0x0102u16.encode(&mut buf);
    assert_eq!(buf, vec![0x02, 0x01]); // little-endian
}

// --- u32 ---

#[test]
fn test_u32() {
    round_trip(&0u32);
    round_trip(&1_000_000u32);
    round_trip(&u32::MAX);
}

// --- u64 ---

#[test]
fn test_u64() {
    round_trip(&0u64);
    round_trip(&u64::MAX);
}

// --- i8 ---

#[test]
fn test_i8() {
    round_trip(&0i8);
    round_trip(&-1i8);
    round_trip(&i8::MIN);
    round_trip(&i8::MAX);
}

// --- i16 ---

#[test]
fn test_i16() {
    round_trip(&0i16);
    round_trip(&-1i16);
    round_trip(&i16::MIN);
    round_trip(&i16::MAX);
}

// --- i32 ---

#[test]
fn test_i32() {
    round_trip(&0i32);
    round_trip(&-1i32);
    round_trip(&i32::MIN);
    round_trip(&i32::MAX);
}

// --- i64 ---

#[test]
fn test_i64() {
    round_trip(&0i64);
    round_trip(&-1i64);
    round_trip(&i64::MIN);
    round_trip(&i64::MAX);
}

// --- f32 ---

#[test]
fn test_f32() {
    round_trip(&0.0f32);
    round_trip(&1.5f32);
    round_trip(&-3.14f32);
    round_trip(&f32::INFINITY);
    round_trip(&f32::NEG_INFINITY);
}

#[test]
fn test_f32_nan() {
    let mut buf = Vec::new();
    f32::NAN.encode(&mut buf);
    let mut pos = 0;
    let decoded = f32::decode(&buf, &mut pos).unwrap();
    assert!(decoded.is_nan());
}

// --- f64 ---

#[test]
fn test_f64() {
    round_trip(&0.0f64);
    round_trip(&1.5f64);
    round_trip(&-3.14f64);
    round_trip(&f64::INFINITY);
    round_trip(&f64::NEG_INFINITY);
}

#[test]
fn test_f64_nan() {
    let mut buf = Vec::new();
    f64::NAN.encode(&mut buf);
    let mut pos = 0;
    let decoded = f64::decode(&buf, &mut pos).unwrap();
    assert!(decoded.is_nan());
}

// --- String ---

#[test]
fn test_string_empty() {
    round_trip(&String::new());
}

#[test]
fn test_string_ascii() {
    round_trip(&"hello".to_string());
}

#[test]
fn test_string_unicode() {
    round_trip(&"日本語テスト".to_string());
}

#[test]
fn test_string_encoding() {
    let mut buf = Vec::new();
    "hi".to_string().encode(&mut buf);
    // 4-byte LE length (2) + 2 bytes of "hi"
    assert_eq!(buf, vec![2, 0, 0, 0, b'h', b'i']);
}

// --- Multiple values in sequence ---

#[test]
fn test_sequential_encode_decode() {
    let mut buf = Vec::new();
    42u32.encode(&mut buf);
    true.encode(&mut buf);
    "test".to_string().encode(&mut buf);
    3.14f64.encode(&mut buf);

    let mut pos = 0;
    assert_eq!(u32::decode(&buf, &mut pos).unwrap(), 42);
    assert_eq!(bool::decode(&buf, &mut pos).unwrap(), true);
    assert_eq!(String::decode(&buf, &mut pos).unwrap(), "test");
    assert_eq!(f64::decode(&buf, &mut pos).unwrap(), 3.14);
    assert_eq!(pos, buf.len());
}

// --- Error cases ---

#[test]
fn test_decode_empty_buffer() {
    let buf = vec![];
    let mut pos = 0;
    assert!(matches!(
        u32::decode(&buf, &mut pos),
        Err(DecodeError::UnexpectedEof)
    ));
}

#[test]
fn test_decode_truncated() {
    let buf = vec![1, 2]; // only 2 bytes, u32 needs 4
    let mut pos = 0;
    assert!(matches!(
        u32::decode(&buf, &mut pos),
        Err(DecodeError::UnexpectedEof)
    ));
}

#[test]
fn test_decode_string_truncated_length() {
    let buf = vec![10, 0, 0, 0, b'h', b'i']; // claims 10 bytes, only 2 available
    let mut pos = 0;
    assert!(matches!(
        String::decode(&buf, &mut pos),
        Err(DecodeError::UnexpectedEof)
    ));
}

#[test]
fn test_decode_string_invalid_utf8() {
    let mut buf = Vec::new();
    3u32.encode(&mut buf); // length = 3
    buf.extend_from_slice(&[0xFF, 0xFE, 0xFD]); // invalid UTF-8
    let mut pos = 0;
    assert!(matches!(
        String::decode(&buf, &mut pos),
        Err(DecodeError::InvalidUtf8)
    ));
}

#[test]
fn test_decode_bool_invalid() {
    let buf = vec![2]; // not 0 or 1
    let mut pos = 0;
    assert!(matches!(
        bool::decode(&buf, &mut pos),
        Err(DecodeError::InvalidBool(_))
    ));
}

// --- Vec<T: Codec> ---

#[test]
fn test_vec_u8_empty() {
    round_trip(&Vec::<u8>::new());
}

#[test]
fn test_vec_u8_simple() {
    round_trip(&vec![1u8, 2, 3, 4, 5]);
}

#[test]
fn test_vec_u32_simple() {
    round_trip(&vec![100u32, 200, 300]);
}

#[test]
fn test_vec_encoding() {
    let mut buf = Vec::new();
    vec![10u8, 20, 30].encode(&mut buf);
    // 4-byte LE length (3) + 3 bytes (each u8 is 1 byte)
    assert_eq!(buf, vec![3, 0, 0, 0, 10, 20, 30]);
}

#[test]
fn test_vec_decode_truncated_length() {
    let buf = vec![2, 0]; // only 2 bytes, u32 needs 4
    let mut pos = 0;
    assert!(matches!(
        Vec::<u8>::decode(&buf, &mut pos),
        Err(DecodeError::UnexpectedEof)
    ));
}

#[test]
fn test_vec_decode_truncated_data() {
    let buf = vec![5, 0, 0, 0, 1, 2]; // claims 5 elements, only 2 available
    let mut pos = 0;
    assert!(matches!(
        Vec::<u8>::decode(&buf, &mut pos),
        Err(DecodeError::UnexpectedEof)
    ));
}
