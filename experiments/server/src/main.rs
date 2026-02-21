use {
    base::{Vec2, log},
    com::WsServer,
    image::{Image, ImageError, encode_jpeg},
    server::{Language, ToMonitor},
    video::{VideoData, VideoIn},
};

const DEFAULT_ADDR: &str = "0.0.0.0:5090";

async fn yuyv_to_jpeg(size: Vec2<usize>, data: &[u8], quality: u8) -> Result<Vec<u8>, ImageError> {
    let pixel_count = size.x * size.y;
    let mut rgb = Vec::with_capacity(pixel_count * 3);

    for chunk in data[..pixel_count * 2].chunks_exact(4) {
        let y0 = chunk[0] as f32;
        let u = chunk[1] as f32;
        let y1 = chunk[2] as f32;
        let v = chunk[3] as f32;

        let r0 = (y0 + 1.402 * (v - 128.0)).clamp(0.0, 255.0) as u8;
        let g0 = (y0 - 0.344 * (u - 128.0) - 0.714 * (v - 128.0)).clamp(0.0, 255.0) as u8;
        let b0 = (y0 + 1.772 * (u - 128.0)).clamp(0.0, 255.0) as u8;

        let r1 = (y1 + 1.402 * (v - 128.0)).clamp(0.0, 255.0) as u8;
        let g1 = (y1 - 0.344 * (u - 128.0) - 0.714 * (v - 128.0)).clamp(0.0, 255.0) as u8;
        let b1 = (y1 + 1.772 * (u - 128.0)).clamp(0.0, 255.0) as u8;

        rgb.extend_from_slice(&[r0, g0, b0, r1, g1, b1]);
    }

    let tensor = base::Tensor::new(vec![size.y, size.x, 3], rgb)?;
    encode_jpeg(Image::U8(tensor), quality).await
}

async fn srggb10p_to_jpeg(
    size: Vec2<usize>,
    data: &[u8],
    quality: u8,
) -> Result<Vec<u8>, ImageError> {
    let width = size.x;
    let height = size.y;
    let stride = data.len() / height;
    let mut rgb = vec![0u8; width * height * 3];

    for y in (0..height - 1).step_by(2) {
        for x in (0..width - 1).step_by(2) {
            let pos = x % 4;
            let top = y * stride + (x / 4) * 5;
            let bot = top + stride;
            let top_lo = data[top + 4] as u32;
            let bot_lo = data[bot + 4] as u32;
            let r = ((data[top + pos] as u32) << 2 | ((top_lo >> (pos * 2)) & 0x03)) >> 2;
            let gr = ((data[top + pos + 1] as u32) << 2 | ((top_lo >> (pos * 2 + 2)) & 0x03)) >> 2;
            let gb = ((data[bot + pos] as u32) << 2 | ((bot_lo >> (pos * 2)) & 0x03)) >> 2;
            let b = ((data[bot + pos + 1] as u32) << 2 | ((bot_lo >> (pos * 2 + 2)) & 0x03)) >> 2;
            let g = ((gr + gb) / 2) as u8;
            let (r, gr, gb, b) = (r as u8, gr as u8, gb as u8, b as u8);

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

    let tensor = base::Tensor::new(vec![height, width, 3], rgb)?;
    encode_jpeg(Image::U8(tensor), quality).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    log::info!("websocket server: {}", DEFAULT_ADDR);

    // Open default video input
    let mut videoin = VideoIn::open(None).await?;
    let size = videoin.size();
    log::info!("resolution: {}x{}", size.x, size.y);
    let format = videoin.format();
    log::info!("format: {:?}", format);
    let frame_rate = videoin.frame_rate();
    log::info!("frame rate: {}", frame_rate);

    // Bind WebSocket server
    let server = WsServer::<ToMonitor>::bind(DEFAULT_ADDR).await?;

    let mut prev_client_count = 0;

    loop {
        // Capture frame
        let frame = videoin.capture().await?;

        // convert to JPEG if needed
        let jpeg = match frame.data {
            VideoData::Jpeg(data) => data,
            VideoData::Yuyv(data) => yuyv_to_jpeg(frame.size, &data, 80).await?,
            VideoData::Srggb10p(data) => srggb10p_to_jpeg(frame.size, &data, 80).await?,
        };

        // Broadcast frame to monitor
        server.send(&ToMonitor::VideoJpeg(jpeg)).await?;

        // Send initial settings when client count changes
        let client_count = server.client_count().await;
        if client_count != prev_client_count {
            log::info!("Connected clients: {}", client_count);
            // Send Settings after video frame (not before) to avoid race with client listener setup
            if client_count > 0 {
                server
                    .send(&ToMonitor::Settings {
                        language: Language::EnglishUs,
                    })
                    .await?;
            }
            prev_client_count = client_count;
        }
    }
}
