use {
    base::{Vec2, log},
    com::WsServer,
    image::{PixelFormat, argb_to_jpeg, rgb_to_jpeg, srggb10p_to_jpeg, yu12_to_jpeg, yuyv_to_jpeg},
    server::{Language, ToMonitor},
    video::{VideoIn, VideoInConfig},
};

#[cfg(feature = "realsense")]
use video::realsense::RealsenseConfig;
#[cfg(feature = "rpicam")]
use video::rpicam::RpiCamConfig;
#[cfg(feature = "v4l2")]
use video::v4l2::V4l2Config;

const DEFAULT_ADDR: &str = "0.0.0.0:5090";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    log::info!("websocket server: {}", DEFAULT_ADDR);

    // Open video input
    #[cfg(feature = "rpicam")]
    let mut videoin = VideoIn::open(Some(VideoInConfig::RpiCam(RpiCamConfig {
        size: Some(Vec2::new(640, 480)),
        frame_rate: Some(30.0),
        ..Default::default()
    })))
    .await?;
    #[cfg(feature = "realsense")]
    let mut videoin = VideoIn::open(Some(VideoInConfig::Realsense(RealsenseConfig {
        color: Some(Vec2::new(640, 480)),
        depth: Some(Vec2::new(640, 480)),
        frame_rate: Some(30.0),
        ..Default::default()
    })))
    .await?;
    #[cfg(all(not(feature = "rpicam"), not(feature = "realsense"), feature = "v4l2"))]
    let mut videoin = VideoIn::open(Some(VideoInConfig::V4l2(V4l2Config {
        size: Some(Vec2::new(640, 480)),
        frame_rate: Some(30.0),
        ..Default::default()
    })))
    .await?;
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

        // convert color to JPEG
        let size = frame.color.size;
        let data = &frame.color.data;

        #[cfg(feature = "realsense")]
        let jpeg = if let Some(ref depth) = frame.depth {
            depth_to_jpeg(depth.size, &depth.data, 80)
        } else {
            color_to_jpeg(size, data, frame.color.format, 80)
        };
        #[cfg(not(feature = "realsense"))]
        let jpeg = color_to_jpeg(size, data, frame.color.format, 80);

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

fn color_to_jpeg(size: Vec2<usize>, data: &[u8], format: PixelFormat, quality: u8) -> Vec<u8> {
    match format {
        PixelFormat::Jpeg => data.to_vec(),
        PixelFormat::Yuyv => yuyv_to_jpeg(size, data, quality),
        PixelFormat::Srggb10p => srggb10p_to_jpeg(size, data, quality),
        PixelFormat::Yu12 => yu12_to_jpeg(size, data, quality),
        PixelFormat::Rgb8 => rgb_to_jpeg(size, data, quality),
        PixelFormat::Argb8 => argb_to_jpeg(size, data, quality),
    }
}

#[cfg(feature = "realsense")]
fn depth_to_jpeg(size: Vec2<usize>, data: &[u16], quality: u8) -> Vec<u8> {
    let max_depth = data.iter().copied().filter(|&d| d > 0).max().unwrap_or(1);

    let rgb: Vec<u8> = data
        .iter()
        .flat_map(|&d| {
            let d = d.saturating_sub(200);
            if d == 0 {
                return [0, 0, 0];
            }
            let max = max_depth.saturating_sub(200).max(1);
            let t = (d as f32).ln() / (max as f32).ln();
            depth_rainbow(t)
        })
        .collect();

    rgb_to_jpeg(size, &rgb, quality)
}

/// Maps a normalized depth value (0.0 = near, 1.0 = far) to yellow->grey->blue.
#[cfg(feature = "realsense")]
fn depth_rainbow(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        let f = t * 2.0; // 0..1 within first half
        let r = 1.0 - f * 0.5;   // 1.0 -> 0.5
        let g = 1.0 - f * 0.5;   // 1.0 -> 0.5
        let b = f * 0.5;         // 0.0 -> 0.5
        (r, g, b)
    } else {
        let f = (t - 0.5) * 2.0; // 0..1 within second half
        let r = 0.5 - f * 0.5;   // 0.5 -> 0.0
        let g = 0.5 - f * 0.5;   // 0.5 -> 0.0
        let b = 0.5 + f * 0.5;   // 0.5 -> 1.0
        (r, g, b)
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}
