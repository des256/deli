use deli_base::Tensor;
use deli_image::{Image, ImageError};

#[test]
fn test_image_u8_helpers() {
    let data: Vec<u8> = (0..24).collect();
    let tensor = Tensor::new(vec![2, 3, 4], data).unwrap();
    let image = Image::U8(tensor);

    assert_eq!(image.shape(), &[2, 3, 4]);
    assert_eq!(image.height(), 2);
    assert_eq!(image.width(), 3);
    assert_eq!(image.channels(), 4);
}

#[test]
fn test_image_u16_helpers() {
    let data: Vec<u16> = (0..24).collect();
    let tensor = Tensor::new(vec![2, 3, 4], data).unwrap();
    let image = Image::U16(tensor);

    assert_eq!(image.shape(), &[2, 3, 4]);
    assert_eq!(image.height(), 2);
    assert_eq!(image.width(), 3);
    assert_eq!(image.channels(), 4);
}

#[test]
fn test_image_f32_helpers() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let tensor = Tensor::new(vec![2, 3, 4], data).unwrap();
    let image = Image::F32(tensor);

    assert_eq!(image.shape(), &[2, 3, 4]);
    assert_eq!(image.height(), 2);
    assert_eq!(image.width(), 3);
    assert_eq!(image.channels(), 4);
}

#[test]
fn test_image_error_from_image_error() {
    let img_err = crates_image::ImageError::Unsupported(
        crates_image::error::UnsupportedError::from_format_and_kind(
            crates_image::error::ImageFormatHint::Unknown,
            crates_image::error::UnsupportedErrorKind::Format(
                crates_image::error::ImageFormatHint::Unknown,
            ),
        ),
    );

    let err: ImageError = img_err.into();
    let err_str = format!("{}", err);
    assert!(err_str.contains("decode error"));
}

#[test]
fn test_image_error_from_tensor_error() {
    let tensor_err = deli_base::TensorError::ShapeMismatch {
        expected: 10,
        got: 5,
    };

    let err: ImageError = tensor_err.into();
    let err_str = format!("{}", err);
    assert!(err_str.contains("tensor error"));
}

#[test]
fn test_image_error_display() {
    let err = ImageError::Decode("test error".to_string());
    assert_eq!(format!("{}", err), "decode error: test error");

    let err = ImageError::Tensor(deli_base::TensorError::ShapeOverflow);
    assert!(format!("{}", err).contains("tensor error"));
}
