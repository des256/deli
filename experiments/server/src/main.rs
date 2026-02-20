use {
    base::log,
    com::WsServer,
    image::{Image, encode_jpeg},
    server::{Language, ToMonitor},
    video::{VideoData, VideoIn},
};

const DEFAULT_ADDR: &str = "0.0.0.0:5090";

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
        let jpeg = match frame.data {
            VideoData::Jpeg(data) => data,
            VideoData::Yuyv(tensor) => encode_jpeg(Image::U8(tensor), 80).await?,
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
