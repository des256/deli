use camera_viewer::Frame;
use deli_base::log;
use deli_camera::{Camera, CameraConfig, Frame as CameraFrame, V4l2Camera};
use deli_com::Server;
use deli_image::DecodedImage;

const DEFAULT_ADDR: &str = "0.0.0.0:9920";

/// Decode a camera Frame into an RGB tensor.
async fn frame_to_rgb(frame: CameraFrame) -> Result<deli_base::Tensor<u8>, Box<dyn std::error::Error>> {
    match frame {
        CameraFrame::Rgb(tensor) => Ok(tensor),
        CameraFrame::Jpeg(data) => match deli_image::decode_image(&data).await? {
            DecodedImage::U8(tensor) => Ok(tensor),
            _ => Err("Unexpected pixel format from JPEG decode".into()),
        },
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    deli_base::init_stdout_logger();

    // Parse address from args or use default
    let addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_ADDR.to_string());

    log::info!("Camera Broadcaster");
    log::info!("Binding to: {}", addr);

    // Open camera
    let config = CameraConfig::default().with_width(640).with_height(480);
    let mut camera = V4l2Camera::new(config)?;
    log::info!("Camera opened: 640x480");

    // Bind sender server
    let sender = Server::<Frame>::bind(&addr).await?;
    log::info!("Listening on {}", addr);

    let mut prev_client_count = 0;

    loop {
        // Capture and decode frame
        let tensor = frame_to_rgb(camera.recv().await?).await?;

        // Extract dimensions from tensor shape [H, W, 3]
        if tensor.shape.len() != 3 || tensor.shape[2] != 3 {
            log::warn!("Expected [H, W, 3] frame shape, got {:?}", tensor.shape);
            continue;
        }

        let height = tensor.shape[0] as u32;
        let width = tensor.shape[1] as u32;

        // Create Frame and broadcast
        let frame = Frame::new(width, height, tensor.data);
        sender.send(&frame).await?;

        // Log client count changes
        let client_count = sender.client_count().await;
        if client_count != prev_client_count {
            log::info!("Connected clients: {}", client_count);
            prev_client_count = client_count;
        }
    }
}
