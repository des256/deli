use {crate::*, base::Vec2};

pub fn rgb_to_argb(size: Vec2<usize>, data: &[u8]) -> Vec<u8> {
    let pixel_count = size.x * size.y;
    let mut argb = Vec::with_capacity(pixel_count * 4);

    for chunk in data.chunks_exact(3) {
        argb.push(0xFF); // A
        argb.push(chunk[0]); // R
        argb.push(chunk[1]); // G
        argb.push(chunk[2]); // B
    }

    argb
}

pub fn yuyv_to_argb(size: Vec2<usize>, data: &[u8]) -> Vec<u8> {
    let pixel_count = size.x * size.y;
    let mut argb = Vec::with_capacity(pixel_count * 4);

    for chunk in data.chunks_exact(4) {
        let (r0, g0, b0) = yuv_to_rgb(chunk[0], chunk[1], chunk[3]);
        let (r1, g1, b1) = yuv_to_rgb(chunk[2], chunk[1], chunk[3]);
        argb.extend_from_slice(&[0xFF, r0, g0, b0, 0xFF, r1, g1, b1]);
    }

    argb
}

pub fn srggb10p_to_argb(size: Vec2<usize>, data: &[u8]) -> Vec<u8> {
    let width = size.x;
    let height = size.y;
    if height < 2 || width < 2 {
        return vec![0u8; width * height * 4];
    }

    let stride = data.len() / height;
    let mut argb = vec![0u8; width * height * 4];

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

            let i = (y * width + x) * 4;
            let j = i + width * 4;
            argb[i] = 0xFF;
            argb[i + 1] = r;
            argb[i + 2] = g;
            argb[i + 3] = b;
            argb[i + 4] = 0xFF;
            argb[i + 5] = r;
            argb[i + 6] = gr;
            argb[i + 7] = b;
            argb[j] = 0xFF;
            argb[j + 1] = r;
            argb[j + 2] = gb;
            argb[j + 3] = b;
            argb[j + 4] = 0xFF;
            argb[j + 5] = r;
            argb[j + 6] = g;
            argb[j + 7] = b;
        }
    }

    argb
}

pub fn yu12_to_argb(size: Vec2<usize>, data: &[u8]) -> Vec<u8> {
    let width = size.x;
    let height = size.y;
    let y_len = width * height;
    let uv_w = width / 2;

    let y_plane = &data[..y_len];
    let u_plane = &data[y_len..];
    let v_offset = uv_w * (height / 2);
    let v_plane = &data[y_len + v_offset..];

    let mut argb = Vec::with_capacity(y_len * 4);

    for row in 0..height {
        for col in 0..width {
            let y = y_plane[row * width + col];
            let u = u_plane[(row / 2) * uv_w + col / 2];
            let v = v_plane[(row / 2) * uv_w + col / 2];
            let (r, g, b) = yuv_to_rgb(y, u, v);
            argb.extend_from_slice(&[0xFF, r, g, b]);
        }
    }

    argb
}
