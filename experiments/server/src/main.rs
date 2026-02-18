use base::log;
use com::WsServer;
use futures_util::StreamExt;
use image::{Image, encode_jpeg};
use server::{Language, ToMonitor};
use video::{CameraConfig, RPiCamera, VideoData};

const DEFAULT_ADDR: &str = "0.0.0.0:5090";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    log::info!("Server Experiment - Camera Data Broadcaster");
    log::info!("Binding to: {}", DEFAULT_ADDR);

    // Open default camera
    let config = CameraConfig::default().with_width(640).with_height(480);
    let mut camera = RPiCamera::new(config)?;
    log::info!("Camera opened: 640x480");

    // Bind WebSocket server
    let server = WsServer::<ToMonitor>::bind(DEFAULT_ADDR).await?;
    log::info!("Listening on {}", server.local_addr());

    let mut prev_client_count = 0;

    loop {
        // Capture frame
        let frame = camera.next().await.unwrap()?;
        let jpeg = match frame.data {
            VideoData::Jpeg(data) => data,
            VideoData::Rgb(tensor) => encode_jpeg(Image::U8(tensor), 80).await?,
        };

        // Broadcast frame to monitor
        server.send(&ToMonitor::VideoJpeg(jpeg)).await?;

        // Send initial settings when client count changes
        let client_count = server.client_count().await;
        if client_count != prev_client_count {
            log::info!("Connected clients: {}", client_count);
            // Send Settings after video frame (not before) to avoid race with client listener setup
            if client_count > 0 {
                server.send(&ToMonitor::Settings { language: Language::EnglishUs }).await?;
            }
            prev_client_count = client_count;
        }
    }
}
