/// Converts YUYV (YUV 4:2:2) pixel data to RGB.
///
/// YUYV packs as `[Y0, U, Y1, V, ...]` — each pair of pixels shares U and V.
/// Converts to RGB using BT.601 coefficients:
/// - R = Y + 1.402 * (V - 128)
/// - G = Y - 0.344 * (U - 128) - 0.714 * (V - 128)
/// - B = Y + 1.772 * (U - 128)
///
/// Returns RGB data as `[R, G, B, R, G, B, ...]` with 3 bytes per pixel.
///
/// # Errors
///
/// Returns `None` if the input data length is less than `width * height * 2` bytes
/// (the expected size for YUYV at the given dimensions).
pub fn yuyv_to_rgb(data: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let pixel_count = (width as usize) * (height as usize);
    let expected_len = pixel_count * 2;
    if data.len() < expected_len {
        return None;
    }

    let mut rgb = Vec::with_capacity(pixel_count * 3);

    // YUYV has 2 bytes per pixel (4 bytes for 2 pixels: Y0 U Y1 V)
    for chunk in data[..expected_len].chunks_exact(4) {
        let y0 = chunk[0] as f32;
        let u = chunk[1] as f32;
        let y1 = chunk[2] as f32;
        let v = chunk[3] as f32;

        // BT.601 conversion for pixel 0
        let r0 = (y0 + 1.402 * (v - 128.0)).clamp(0.0, 255.0) as u8;
        let g0 = (y0 - 0.344 * (u - 128.0) - 0.714 * (v - 128.0)).clamp(0.0, 255.0) as u8;
        let b0 = (y0 + 1.772 * (u - 128.0)).clamp(0.0, 255.0) as u8;

        // BT.601 conversion for pixel 1
        let r1 = (y1 + 1.402 * (v - 128.0)).clamp(0.0, 255.0) as u8;
        let g1 = (y1 - 0.344 * (u - 128.0) - 0.714 * (v - 128.0)).clamp(0.0, 255.0) as u8;
        let b1 = (y1 + 1.772 * (u - 128.0)).clamp(0.0, 255.0) as u8;

        rgb.extend_from_slice(&[r0, g0, b0, r1, g1, b1]);
    }

    Some(rgb)
}
