use deli_video::convert::yuyv_to_rgb;

#[test]
fn test_yuyv_to_rgb_gray() {
    // Y=128, U=128, V=128 (mid-gray, no chroma)
    let yuyv = vec![128, 128, 128, 128];
    let rgb = yuyv_to_rgb(&yuyv, 2, 1).expect("valid input");

    assert_eq!(rgb.len(), 6); // 2 pixels * 3 channels
    assert_eq!(rgb, vec![128, 128, 128, 128, 128, 128]);
}

#[test]
fn test_yuyv_to_rgb_white() {
    // Y=255, U=128, V=128 -> pure white
    let yuyv = vec![255, 128, 255, 128];
    let rgb = yuyv_to_rgb(&yuyv, 2, 1).expect("valid input");

    assert_eq!(rgb, vec![255, 255, 255, 255, 255, 255]);
}

#[test]
fn test_yuyv_to_rgb_black() {
    // Y=0, U=128, V=128 -> pure black
    let yuyv = vec![0, 128, 0, 128];
    let rgb = yuyv_to_rgb(&yuyv, 2, 1).expect("valid input");

    assert_eq!(rgb, vec![0, 0, 0, 0, 0, 0]);
}

#[test]
fn test_yuyv_to_rgb_multiline() {
    // 2x2 image: 4 pixels, 2 YUYV macro-pixels
    let yuyv = vec![
        128, 128, 128, 128, // Row 0: gray, gray
        255, 128, 0, 128, // Row 1: white, black
    ];
    let rgb = yuyv_to_rgb(&yuyv, 2, 2).expect("valid input");

    assert_eq!(rgb.len(), 12); // 4 pixels * 3 channels
    // Row 0: gray, gray
    assert_eq!(&rgb[0..6], &[128, 128, 128, 128, 128, 128]);
    // Row 1: white, black
    assert_eq!(&rgb[6..12], &[255, 255, 255, 0, 0, 0]);
}

#[test]
fn test_yuyv_to_rgb_short_input_returns_none() {
    // 2x2 image expects 8 bytes, only provide 4
    let yuyv = vec![128, 128, 128, 128];
    assert!(yuyv_to_rgb(&yuyv, 2, 2).is_none());
}

#[test]
fn test_yuyv_to_rgb_empty_input() {
    assert!(yuyv_to_rgb(&[], 2, 1).is_none());
}

#[test]
fn test_yuyv_to_rgb_zero_dimensions() {
    // 0x0 requires 0 bytes, empty data is valid
    let rgb = yuyv_to_rgb(&[], 0, 0).expect("valid for 0x0");
    assert!(rgb.is_empty());
}
