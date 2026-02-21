use {
    base::log,
    image::Image,
    minifb::{Key, Window, WindowOptions},
    video::{VideoData, VideoFrame, VideoIn},
};

pub async fn frame_to_argb(frame: &VideoFrame) -> Vec<u32> {
    match &frame.data {
        VideoData::Yuyv(data) => {
            let pixel_count = frame.size.x * frame.size.y;

            let mut argb = Vec::with_capacity(pixel_count);

            // YUYV has 2 bytes per pixel (4 bytes for 2 pixels: Y0 U Y1 V)
            for chunk in data[..pixel_count * 2].chunks_exact(4) {
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

                argb.extend_from_slice(&[
                    0xFF000000 | (r0 as u32) << 16 | (g0 as u32) << 8 | (b0 as u32),
                    0xFF000000 | (r1 as u32) << 16 | (g1 as u32) << 8 | (b1 as u32),
                ]);
            }

            argb
        }
        VideoData::Jpeg(data) => match image::decode_image(&data).await {
            Ok(Image::U8(tensor)) => {
                let mut argb = Vec::with_capacity(tensor.shape[0] * tensor.shape[1]);
                for pixel in tensor.data.chunks_exact(3) {
                    argb.push(
                        0xFF000000
                            | (pixel[0] as u32) << 16
                            | (pixel[1] as u32) << 8
                            | (pixel[2] as u32),
                    );
                }
                argb
            }
            Ok(_) => {
                log::error!("Unexpected pixel format from JPEG decode");
                return Vec::new();
            }
            Err(error) => {
                log::error!("Failed to decode JPEG: {}", error);
                return Vec::new();
            }
        },
        VideoData::Srggb10p(data) => {
            let width = frame.size.x;
            let height = frame.size.y;
            let stride = data.len() / height;
            let mut argb = vec![0u32; width * height];

            // Single-pass: unpack + demosaic over 2x2 RGGB blocks
            // x is always even, so x and x+1 share the same 5-byte pack group
            for y in (0..height - 1).step_by(2) {
                for x in (0..width - 1).step_by(2) {
                    let pos = x % 4;
                    let top = y * stride + (x / 4) * 5;
                    let bot = top + stride;
                    let top_lo = data[top + 4] as u32;
                    let bot_lo = data[bot + 4] as u32;
                    let r = (data[top + pos] as u32) << 2 | ((top_lo >> (pos * 2)) & 0x03);
                    let gr = (data[top + pos + 1] as u32) << 2 | ((top_lo >> (pos * 2 + 2)) & 0x03);
                    let gb = (data[bot + pos] as u32) << 2 | ((bot_lo >> (pos * 2)) & 0x03);
                    let b = (data[bot + pos + 1] as u32) << 2 | ((bot_lo >> (pos * 2 + 2)) & 0x03);
                    let g = (gr + gb) / 2;

                    let r8 = r >> 2;
                    let g8 = g >> 2;
                    let gr8 = gr >> 2;
                    let gb8 = gb >> 2;
                    let b8 = b >> 2;

                    let i = y * width + x;
                    argb[i] = 0xFF000000 | (r8 << 16) | (g8 << 8) | b8;
                    argb[i + 1] = 0xFF000000 | (r8 << 16) | (gr8 << 8) | b8;
                    argb[i + width] = 0xFF000000 | (r8 << 16) | (gb8 << 8) | b8;
                    argb[i + width + 1] = 0xFF000000 | (r8 << 16) | (g8 << 8) | b8;
                }
            }

            argb
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    // Initialize camera
    let mut videoin = VideoIn::open(None).await?;
    let size = videoin.size();
    log::info!("resolution: {}x{}", size.x, size.y);
    let format = videoin.format();
    log::info!("format: {:?}", format);
    let frame_rate = videoin.frame_rate();
    log::info!("frame rate: {}", frame_rate);

    // Create display window
    let mut window = Window::new(
        "Camera View - ESC to exit",
        size.x,
        size.y,
        WindowOptions::default(),
    )?;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame = videoin.capture().await?;
        let argb = frame_to_argb(&frame).await;
        window.update_with_buffer(&argb, frame.size.x, frame.size.y)?;
    }

    Ok(())
}
