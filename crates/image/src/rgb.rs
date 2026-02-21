use crate::*;
use base::Vec2;

pub fn yuyv_to_rgb(size: Vec2<usize>, data: &[u8]) -> Vec<u8> {
    let pixel_count = size.x * size.y;
    let mut rgb = Vec::with_capacity(pixel_count * 3);

    for chunk in data.chunks_exact(4) {
        let (r0, g0, b0) = yuv_to_rgb(chunk[0], chunk[1], chunk[3]);
        let (r1, g1, b1) = yuv_to_rgb(chunk[2], chunk[1], chunk[3]);
        rgb.extend_from_slice(&[r0, g0, b0, r1, g1, b1]);
    }

    rgb
}

pub fn srggb10p_to_rgb(size: Vec2<usize>, data: &[u8]) -> Vec<u8> {
    let width = size.x;
    let height = size.y;
    if height < 2 || width < 2 {
        return vec![0u8; width * height * 3];
    }

    let stride = data.len() / height;
    let mut rgb = vec![0u8; width * height * 3];

    for y in (0..height - 1).step_by(2) {
        for x in (0..width - 1).step_by(2) {
            let pos = x % 4;
            let top = y * stride + (x / 4) * 5;
            let bot = top + stride;
            let top_lo = data[top + 4] as u32;
            let bot_lo = data[bot + 4] as u32;
            let r = (((data[top + pos] as u32) << 2 | ((top_lo >> (pos * 2)) & 0x03)) >> 2) as u8;
            let gr = (((data[top + pos + 1] as u32) << 2 | ((top_lo >> (pos * 2 + 2)) & 0x03)) >> 2)
                as u8;
            let gb = (((data[bot + pos] as u32) << 2 | ((bot_lo >> (pos * 2)) & 0x03)) >> 2) as u8;
            let b = (((data[bot + pos + 1] as u32) << 2 | ((bot_lo >> (pos * 2 + 2)) & 0x03)) >> 2)
                as u8;
            let g = ((gr as u16 + gb as u16) / 2) as u8;

            let i = (y * width + x) * 3;
            let j = i + width * 3;
            rgb[i] = r;
            rgb[i + 1] = g;
            rgb[i + 2] = b;
            rgb[i + 3] = r;
            rgb[i + 4] = gr;
            rgb[i + 5] = b;
            rgb[j] = r;
            rgb[j + 1] = gb;
            rgb[j + 2] = b;
            rgb[j + 3] = r;
            rgb[j + 4] = g;
            rgb[j + 5] = b;
        }
    }

    rgb
}

pub fn yu12_to_rgb(size: Vec2<usize>, data: &[u8]) -> Vec<u8> {
    let width = size.x;
    let height = size.y;
    let y_len = width * height;
    let uv_w = width / 2;

    let y_plane = &data[..y_len];
    let u_plane = &data[y_len..];
    let v_offset = uv_w * (height / 2);
    let v_plane = &data[y_len + v_offset..];

    let mut rgb = Vec::with_capacity(y_len * 3);

    for row in 0..height {
        for col in 0..width {
            let y = y_plane[row * width + col];
            let u = u_plane[(row / 2) * uv_w + col / 2];
            let v = v_plane[(row / 2) * uv_w + col / 2];
            let (r, g, b) = yuv_to_rgb(y, u, v);
            rgb.extend_from_slice(&[r, g, b]);
        }
    }

    rgb
}

pub fn jpeg_to_rgb(image: &Image) -> Result<Image, ImageError> {
    image.format.ensure_format(PixelFormat::Jpeg)?;
    let decoded = crates_image::load_from_memory(&image.data)
        .map_err(|e| ImageError::Decode(format!("Failed to decode JPEG: {}", e)))?;

    let rgb_image = decoded.to_rgb8();
    let (width, height) = rgb_image.dimensions();
    let size = Vec2::new(width as usize, height as usize);
    let data = rgb_image.into_raw();

    Ok(Image::new(size, data, PixelFormat::Rgb8))
}
