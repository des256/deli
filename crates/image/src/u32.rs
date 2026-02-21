use crate::*;
use base::Vec2;

fn pack_u32(r: u8, g: u8, b: u8) -> u32 {
    0xFF00_0000 | (r as u32) << 16 | (g as u32) << 8 | b as u32
}

pub fn rgb_to_u32(size: Vec2<usize>, data: &[u8]) -> Vec<u32> {
    let pixel_count = size.x * size.y;
    let mut buf = Vec::with_capacity(pixel_count);
    for chunk in data.chunks_exact(3) {
        buf.push(pack_u32(chunk[0], chunk[1], chunk[2]));
    }
    buf
}

pub fn argb_to_u32(size: Vec2<usize>, data: &[u8]) -> Vec<u32> {
    let pixel_count = size.x * size.y;
    let mut buf = Vec::with_capacity(pixel_count);
    for chunk in data.chunks_exact(4) {
        buf.push(
            (chunk[0] as u32) << 24
                | (chunk[1] as u32) << 16
                | (chunk[2] as u32) << 8
                | chunk[3] as u32,
        );
    }
    buf
}

pub fn yuyv_to_u32(size: Vec2<usize>, data: &[u8]) -> Vec<u32> {
    let pixel_count = size.x * size.y;
    let mut buf = Vec::with_capacity(pixel_count);
    for chunk in data.chunks_exact(4) {
        let (r0, g0, b0) = yuv_to_rgb(chunk[0], chunk[1], chunk[3]);
        let (r1, g1, b1) = yuv_to_rgb(chunk[2], chunk[1], chunk[3]);
        buf.push(pack_u32(r0, g0, b0));
        buf.push(pack_u32(r1, g1, b1));
    }
    buf
}

pub fn srggb10p_to_u32(size: Vec2<usize>, data: &[u8]) -> Vec<u32> {
    let width = size.x;
    let height = size.y;
    if height < 2 || width < 2 {
        return vec![0u32; width * height];
    }

    let stride = data.len() / height;
    let mut buf = vec![0u32; width * height];

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

            let i = y * width + x;
            let j = i + width;
            buf[i] = pack_u32(r, g, b);
            buf[i + 1] = pack_u32(r, gr, b);
            buf[j] = pack_u32(r, gb, b);
            buf[j + 1] = pack_u32(r, g, b);
        }
    }

    buf
}

pub fn yu12_to_u32(size: Vec2<usize>, data: &[u8]) -> Vec<u32> {
    let width = size.x;
    let height = size.y;
    let y_len = width * height;
    let uv_w = width / 2;

    let y_plane = &data[..y_len];
    let u_plane = &data[y_len..];
    let v_offset = uv_w * (height / 2);
    let v_plane = &data[y_len + v_offset..];

    let mut buf = Vec::with_capacity(y_len);

    for row in 0..height {
        for col in 0..width {
            let y = y_plane[row * width + col];
            let u = u_plane[(row / 2) * uv_w + col / 2];
            let v = v_plane[(row / 2) * uv_w + col / 2];
            let (r, g, b) = yuv_to_rgb(y, u, v);
            buf.push(pack_u32(r, g, b));
        }
    }

    buf
}

pub fn jpeg_to_u32(data: &[u8]) -> Vec<u32> {
    let decoded = crates_image::load_from_memory(data).expect("JPEG decoding failed");
    let rgb = decoded.to_rgb8();
    let mut buf = Vec::with_capacity((rgb.width() * rgb.height()) as usize);
    for chunk in rgb.as_raw().chunks_exact(3) {
        buf.push(pack_u32(chunk[0], chunk[1], chunk[2]));
    }
    buf
}
