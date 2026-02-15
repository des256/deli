/// Convert HWC RGB buffer to packed ARGB u32 for minifb
fn rgb_to_argb(buf: &[u8], width: usize, height: usize) -> Vec<u32> {
    let mut argb = Vec::with_capacity(width * height);
    for i in 0..width * height {
        let idx = i * 3;
        let r = buf[idx] as u32;
        let g = buf[idx + 1] as u32;
        let b = buf[idx + 2] as u32;
        argb.push((r << 16) | (g << 8) | b);
    }
    argb
}

#[test]
fn test_rgb_to_argb_single_pixel() {
    // Red pixel: R=255, G=0, B=0 → 0x00FF0000
    let buf = [255, 0, 0];
    let result = rgb_to_argb(&buf, 1, 1);
    assert_eq!(result, vec![0x00FF0000]);
}

#[test]
fn test_rgb_to_argb_white_pixel() {
    // White pixel: R=255, G=255, B=255 → 0x00FFFFFF
    let buf = [255, 255, 255];
    let result = rgb_to_argb(&buf, 1, 1);
    assert_eq!(result, vec![0x00FFFFFF]);
}

#[test]
fn test_rgb_to_argb_2x1() {
    // Two pixels: red, blue
    let buf = [255, 0, 0, 0, 0, 255];
    let result = rgb_to_argb(&buf, 2, 1);
    assert_eq!(result, vec![0x00FF0000, 0x000000FF]);
}
