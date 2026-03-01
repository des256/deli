use {crate::*, base::*, image::*, realsense_sys as sys, tokio::sync::mpsc as tokio_mpsc};

const CHANNEL_CAPACITY: usize = 4;
const CAPTURE_TIMEOUT_MS: u32 = 5000;

fn check_rs2_error(err: *mut sys::rs2_error, context: &str) {
    if err.is_null() {
        return;
    }
    unsafe {
        let msg = std::ffi::CStr::from_ptr(sys::rs2_get_error_message(err))
            .to_string_lossy()
            .into_owned();
        sys::rs2_free_error(err);
        log_fatal!("{}: {}", context, msg); // TODO: maybe fix later
    }
}

#[derive(Debug, Clone, Default)]
pub struct RealsenseConfig {
    pub index: Option<usize>,
    pub color: Option<Vec2<usize>>,
    pub depth: Option<Vec2<usize>>,
    pub ir: Option<Vec2<usize>>,
    pub frame_rate: Option<f32>,
}

pub struct RealsenseListener {
    frame_rx: tokio_mpsc::Receiver<VideoFrame>,
    size: Vec2<usize>,
}

pub fn create(config: Option<RealsenseConfig>) -> Result<RealsenseListener, VideoError> {
    let (frame_tx, frame_rx) = tokio_mpsc::channel::<VideoFrame>(CHANNEL_CAPACITY);
    let config = config.unwrap_or_default();
    let device_index = config.index.unwrap_or(0) as i32;
    let frame_rate = config.frame_rate.unwrap_or(30.0) as i32;
    let size = config.color.unwrap_or(Vec2::new(640, 480));
    std::thread::spawn(move || {
        let mut err = std::ptr::null_mut::<sys::rs2_error>();
        let context = unsafe { sys::rs2_create_context(sys::RS2_API_VERSION as i32, &mut err) };
        check_rs2_error(err, "failed to create realsense context");
        let device_list = unsafe { sys::rs2_query_devices(context, &mut err) };
        check_rs2_error(err, "failed to query devices");
        let device_count = unsafe { sys::rs2_get_device_count(device_list, &mut err) };
        check_rs2_error(err, "failed to get device count");
        if device_count == 0 {
            unsafe { sys::rs2_delete_device_list(device_list) };
            unsafe { sys::rs2_delete_context(context) };
            log_fatal!("No RealSense devices found");
        }
        if device_index >= device_count {
            unsafe { sys::rs2_delete_device_list(device_list) };
            unsafe { sys::rs2_delete_context(context) };
            log_fatal!(
                "RealSense device index {} out of range (found {} devices)",
                device_index,
                device_count
            );
        }
        let device = unsafe { sys::rs2_create_device(device_list, device_index, &mut err) };
        check_rs2_error(err, "failed to create device");
        unsafe { sys::rs2_hardware_reset(device, &mut err) };
        check_rs2_error(err, "failed to hardware reset device");
        unsafe { sys::rs2_delete_device(device) };
        unsafe { sys::rs2_delete_device_list(device_list) };
        std::thread::sleep(std::time::Duration::from_secs(3));
        err = std::ptr::null_mut();
        let device_list = unsafe { sys::rs2_query_devices(context, &mut err) };
        check_rs2_error(err, "failed to query devices after reset");
        let device = unsafe { sys::rs2_create_device(device_list, device_index, &mut err) };
        check_rs2_error(err, "failed to create device after reset");
        let rs2_pipeline = unsafe { sys::rs2_create_pipeline(context, &mut err) };
        check_rs2_error(err, "failed to create pipeline");
        let rs2_config = unsafe { sys::rs2_create_config(&mut err) };
        check_rs2_error(err, "failed to create config");
        if let Some(_) = config.color {
            unsafe {
                sys::rs2_config_enable_stream(
                    rs2_config,
                    sys::rs2_stream_RS2_STREAM_COLOR,
                    -1,
                    size.x as i32,
                    size.y as i32,
                    sys::rs2_format_RS2_FORMAT_RGB8,
                    frame_rate,
                    &mut err,
                )
            };
            check_rs2_error(err, "failed to enable color stream");
        }
        if let Some(_) = config.depth {
            unsafe {
                sys::rs2_config_enable_stream(
                    rs2_config,
                    sys::rs2_stream_RS2_STREAM_DEPTH,
                    -1,
                    size.x as i32,
                    size.y as i32,
                    sys::rs2_format_RS2_FORMAT_Z16,
                    frame_rate,
                    &mut err,
                )
            };
            check_rs2_error(err, "failed to enable depth stream");
        }
        if let Some(ir_size) = config.ir {
            for index in [1, 2] {
                unsafe {
                    sys::rs2_config_enable_stream(
                        rs2_config,
                        sys::rs2_stream_RS2_STREAM_INFRARED,
                        index,
                        ir_size.x as i32,
                        ir_size.y as i32,
                        sys::rs2_format_RS2_FORMAT_Y8,
                        frame_rate,
                        &mut err,
                    )
                };
                check_rs2_error(err, "failed to enable IR stream");
            }
        }
        let profile =
            unsafe { sys::rs2_pipeline_start_with_config(rs2_pipeline, rs2_config, &mut err) };
        if !err.is_null() {
            unsafe { sys::rs2_delete_config(rs2_config) };
            unsafe { sys::rs2_delete_pipeline(rs2_pipeline) };
            unsafe { sys::rs2_delete_context(context) };
            log_fatal!("failed to start pipeline");
        }
        unsafe { sys::rs2_delete_config(rs2_config) };
        unsafe { sys::rs2_delete_device(device) };
        unsafe { sys::rs2_delete_device_list(device_list) };
        unsafe { sys::rs2_delete_pipeline_profile(profile) };
        let mut err = std::ptr::null_mut::<sys::rs2_error>();
        let composite = unsafe {
            sys::rs2_pipeline_wait_for_frames(rs2_pipeline, CAPTURE_TIMEOUT_MS, &mut err)
        };
        check_rs2_error(err, "failed to wait for frames");
        let mut err = std::ptr::null_mut::<sys::rs2_error>();
        let count = unsafe { sys::rs2_embedded_frames_count(composite, &mut err) };
        if !err.is_null() {
            unsafe { sys::rs2_free_error(err) };
            log_fatal!("failed to get embedded frame count");
        }
        let mut color: Option<Image> = None;
        let mut depth: Option<DepthImage> = None;
        let mut left: Option<IrImage> = None;
        let mut right: Option<IrImage> = None;
        for i in 0..count {
            err = std::ptr::null_mut();
            let frame = unsafe { sys::rs2_extract_frame(composite, i, &mut err) };
            if !err.is_null() {
                unsafe { sys::rs2_free_error(err) };
                continue;
            }
            err = std::ptr::null_mut();
            let mut stream: sys::rs2_stream = 0;
            let mut format: sys::rs2_format = 0;
            let mut index: i32 = 0;
            let mut _uid: i32 = 0;
            let mut _fps: i32 = 0;
            let profile = unsafe { sys::rs2_get_frame_stream_profile(frame, &mut err) };
            if !err.is_null() {
                unsafe { sys::rs2_free_error(err) };
                unsafe { sys::rs2_release_frame(frame) };
                continue;
            }
            err = std::ptr::null_mut();
            unsafe {
                sys::rs2_get_stream_profile_data(
                    profile,
                    &mut stream,
                    &mut format,
                    &mut index,
                    &mut _uid,
                    &mut _fps,
                    &mut err,
                )
            };
            if !err.is_null() {
                unsafe { sys::rs2_free_error(err) };
                unsafe { sys::rs2_release_frame(frame) };
                continue;
            }
            err = std::ptr::null_mut();
            let width = unsafe { sys::rs2_get_frame_width(frame, &mut err) as usize };
            err = std::ptr::null_mut();
            let height = unsafe { sys::rs2_get_frame_height(frame, &mut err) as usize };
            err = std::ptr::null_mut();
            let data_size = unsafe { sys::rs2_get_frame_data_size(frame, &mut err) as usize };
            err = std::ptr::null_mut();
            let data_ptr = unsafe { sys::rs2_get_frame_data(frame, &mut err) as *const u8 };
            let size = Vec2::new(width, height);
            match stream {
                sys::rs2_stream_RS2_STREAM_COLOR => {
                    let data = unsafe { std::slice::from_raw_parts(data_ptr, data_size).to_vec() };
                    color = Some(Image::new(size, data, PixelFormat::Rgb8));
                }
                sys::rs2_stream_RS2_STREAM_DEPTH => {
                    let u16_ptr = data_ptr as *const u16;
                    let u16_len = data_size / 2;
                    let data = unsafe { std::slice::from_raw_parts(u16_ptr, u16_len).to_vec() };
                    depth = Some(DepthImage { size, data });
                }
                sys::rs2_stream_RS2_STREAM_INFRARED => {
                    let data = unsafe { std::slice::from_raw_parts(data_ptr, data_size).to_vec() };
                    match index {
                        1 => left = Some(IrImage { size, data }),
                        2 => right = Some(IrImage { size, data }),
                        _ => {}
                    }
                }
                _ => {}
            }
            unsafe { sys::rs2_release_frame(frame) };
        }
        unsafe { sys::rs2_release_frame(composite) };
        if let Err(error) = frame_tx.blocking_send(VideoFrame {
            color: color.unwrap(),
            depth,
            left,
            right,
        }) {
            log_error!("failed to send video frame: {}", error);
        }
    });

    Ok(RealsenseListener { frame_rx, size })
}

// TODO: shutdown logic
/*
        unsafe {
            if !self.pipeline.is_null() {
                let mut err = std::ptr::null_mut::<sys::rs2_error>();
                sys::rs2_pipeline_stop(self.pipeline, &mut err);
                if !err.is_null() {
                    sys::rs2_free_error(err);
                }
                sys::rs2_delete_pipeline(self.pipeline);
                self.pipeline = std::ptr::null_mut();
            }
            if !self.context.is_null() {
                sys::rs2_delete_context(self.context);
                self.context = std::ptr::null_mut();
            }
        }
    }
*/

impl RealsenseListener {
    pub fn size(&self) -> Vec2<usize> {
        self.size
    }

    pub async fn recv(&mut self) -> Option<VideoFrame> {
        self.frame_rx.recv().await
    }

    pub fn try_recv(&mut self) -> Option<VideoFrame> {
        match self.frame_rx.try_recv() {
            Ok(frame) => Some(frame),
            _ => None,
        }
    }
}
