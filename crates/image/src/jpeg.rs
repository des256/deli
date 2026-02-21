use crate::*;
use base::Vec2;
use crates_image::ImageEncoder;

pub fn rgb_to_jpeg(size: Vec2<usize>, data: &[u8], quality: u8) -> Vec<u8> {
    let mut buffer = Vec::new();
    let encoder = crates_image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buffer, quality);
    encoder
        .write_image(
            data,
            size.x as u32,
            size.y as u32,
            crates_image::ExtendedColorType::Rgb8,
        )
        .expect("JPEG encoding failed");
    buffer
}

pub fn argb_to_jpeg(size: Vec2<usize>, data: &[u8], quality: u8) -> Vec<u8> {
    let rgb: Vec<u8> = data
        .chunks_exact(4)
        .flat_map(|c| [c[1], c[2], c[3]])
        .collect();
    rgb_to_jpeg(size, &rgb, quality)
}

pub fn yuyv_to_jpeg(size: Vec2<usize>, data: &[u8], quality: u8) -> Vec<u8> {
    let rgb = yuyv_to_rgb(size, data);
    rgb_to_jpeg(size, &rgb, quality)
}

pub fn srggb10p_to_jpeg(size: Vec2<usize>, data: &[u8], quality: u8) -> Vec<u8> {
    let rgb = srggb10p_to_rgb(size, data);
    rgb_to_jpeg(size, &rgb, quality)
}

pub fn yu12_to_jpeg(size: Vec2<usize>, data: &[u8], quality: u8) -> Vec<u8> {
    let rgb = yu12_to_rgb(size, data);
    rgb_to_jpeg(size, &rgb, quality)
}
