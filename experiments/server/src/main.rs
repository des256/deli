use deli_base::log;
use deli_video::{CameraConfig, RPiCamera, VideoData};
use futures_util::StreamExt;
use deli_com::WsServer;
use deli_image::{encode_jpeg, Image};
use server::Data;

const DEFAULT_ADDR: &str = "0.0.0.0:5090";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    deli_base::init_stdout_logger();

    log::info!("Server Experiment - Camera Data Broadcaster");
    log::info!("Binding to: {}", DEFAULT_ADDR);

    // Open default camera
    let config = CameraConfig::default().with_width(640).with_height(480);
    let mut camera = RPiCamera::new(config)?;
    log::info!("Camera opened: 640x480");

    // Bind WebSocket server
    let server = WsServer::<Data>::bind(DEFAULT_ADDR).await?;
    log::info!("Listening on {}", server.local_addr());

    let mut prev_client_count = 0;
    let mut value: i32 = 0;

    loop {
        // Capture frame
        let frame = camera.next().await.unwrap()?;
        let jpeg = match frame.data {
            VideoData::Jpeg(data) => data,
            VideoData::Rgb(tensor) => encode_jpeg(Image::U8(tensor), 80).await?,
        };

        // Broadcast Data with frame
        let data = Data::new(value, true, jpeg);
        server.send(&data).await?;

        value = value.wrapping_add(1);

        // Log client count changes
        let client_count = server.client_count().await;
        if client_count != prev_client_count {
            log::info!("Connected clients: {}", client_count);
            prev_client_count = client_count;
        }
    }
}
