use deli_base::log;
use deli_com::WsServer;
use server::Data;

const DEFAULT_ADDR: &str = "127.0.0.1:5090";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    deli_base::init_stdout_logger();

    log::info!("Server Experiment - Data Broadcaster");
    log::info!("Binding to: {}", DEFAULT_ADDR);

    // Bind WebSocket server
    let server = WsServer::<Data>::bind(DEFAULT_ADDR).await?;
    log::info!("Listening on {}", server.local_addr());

    let mut prev_client_count = 0;
    let mut value: i32 = 0;

    // Broadcast interval (1 second)
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));

    loop {
        interval.tick().await;

        // Create Data with incrementing value
        let data = Data::new(value, true);
        server.send(&data).await?;

        // Increment value
        value = value.wrapping_add(1);

        // Log client count changes
        let client_count = server.client_count().await;
        if client_count != prev_client_count {
            log::info!("Connected clients: {}", client_count);
            prev_client_count = client_count;
        }

        if client_count > 0 {
            log::debug!("Broadcasted: value={}, flag=true", value.wrapping_sub(1));
        }
    }
}
