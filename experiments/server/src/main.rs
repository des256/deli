use {
    base::{Vec2, log},
    com::WsServer,
    image::{PixelFormat, argb_to_jpeg, rgb_to_jpeg, srggb10p_to_jpeg, yu12_to_jpeg, yuyv_to_jpeg},
    server::{Language, ToMonitor},
    video::{VideoIn, VideoInConfig},
};

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
        frame_rate: Some(60.0),
        ..Default::default()
    })))
    .await?;
    #[cfg(all(not(feature = "rpicam"), feature = "v4l2"))]
    let mut videoin = VideoIn::open(Some(VideoInConfig::V4l2(V4l2Config {
        size: Some(Vec2::new(640, 480)),
        frame_rate: Some(60.0),
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

        // convert to JPEG if needed
        let size = frame.image.size;
        let data = &frame.image.data;
        let jpeg = match frame.image.format {
            PixelFormat::Jpeg => data.clone(),
            PixelFormat::Yuyv => yuyv_to_jpeg(size, data, 80),
            PixelFormat::Srggb10p => srggb10p_to_jpeg(size, data, 80),
            PixelFormat::Yu12 => yu12_to_jpeg(size, data, 80),
            PixelFormat::Rgb8 => rgb_to_jpeg(size, data, 80),
            PixelFormat::Argb8 => argb_to_jpeg(size, data, 80),
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
