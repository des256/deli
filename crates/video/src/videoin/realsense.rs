use {
    crate::*,
    base::*,
    image::{Image, PixelFormat},
    realsense_sys as sys,
};

const CAPTURE_TIMEOUT_MS: u32 = 5000;

#[derive(Debug, Clone, Default)]
pub struct RealsenseConfig {
    pub index: Option<usize>,
    pub color: Option<Vec2<usize>>,
    pub depth: Option<Vec2<usize>>,
    pub ir: Option<Vec2<usize>>,
    pub frame_rate: Option<f32>,
}

pub(crate) struct Realsense {
    pipeline: *mut sys::rs2_pipeline,
    context: *mut sys::rs2_context,
    has_depth: bool,
    has_ir: bool,
}

unsafe impl Send for Realsense {}

impl Realsense {
    pub fn new() -> Self {
        Self {
            pipeline: std::ptr::null_mut(),
            context: std::ptr::null_mut(),
            has_depth: false,
            has_ir: false,
        }
    }

    fn is_open(&self) -> bool {
        !self.pipeline.is_null()
    }
}

impl VideoInDevice for Realsense {
    fn open(&mut self, config: &VideoInConfig) -> Result<VideoInConfig, VideoError> {
        self.close();

        #[allow(irrefutable_let_patterns)]
        let config = if let VideoInConfig::Realsense(config) = config {
            config
        } else {
            return Err(VideoError::Device(
                "Realsense::open requires VideoInConfig::Realsense".to_string(),
            ));
        };

        let device_index = config.index.unwrap_or(0) as i32;
        let frame_rate = config.frame_rate.unwrap_or(30.0) as i32;
        let color_size = config.color.unwrap_or(Vec2::new(640, 480));

        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();

            // Create context
            let ctx = sys::rs2_create_context(sys::RS2_API_VERSION as i32, &mut err);
            check_rs2_error(err, "Failed to create context")?;

            // Query devices
            let device_list = sys::rs2_query_devices(ctx, &mut err);
            check_rs2_error(err, "Failed to query devices")?;

            let device_count = sys::rs2_get_device_count(device_list, &mut err);
            check_rs2_error(err, "Failed to get device count")?;

            if device_count == 0 {
                sys::rs2_delete_device_list(device_list);
                sys::rs2_delete_context(ctx);
                return Err(VideoError::Device("No RealSense devices found".to_string()));
            }

            if device_index >= device_count {
                sys::rs2_delete_device_list(device_list);
                sys::rs2_delete_context(ctx);
                return Err(VideoError::Device(format!(
                    "RealSense device index {} out of range (found {} devices)",
                    device_index, device_count,
                )));
            }

            let device = sys::rs2_create_device(device_list, device_index, &mut err);
            check_rs2_error(err, "Failed to create device")?;

            // Hardware reset (needed on Jetson/embedded)
            sys::rs2_hardware_reset(device, &mut err);
            check_rs2_error(err, "Failed to hardware reset device")?;

            sys::rs2_delete_device(device);
            sys::rs2_delete_device_list(device_list);
            std::thread::sleep(std::time::Duration::from_secs(3));

            // Re-query devices after reset
            err = std::ptr::null_mut();
            let device_list = sys::rs2_query_devices(ctx, &mut err);
            check_rs2_error(err, "Failed to query devices after reset")?;

            let device = sys::rs2_create_device(device_list, device_index, &mut err);
            check_rs2_error(err, "Failed to create device after reset")?;

            // Create pipeline
            let pipeline = sys::rs2_create_pipeline(ctx, &mut err);
            check_rs2_error(err, "Failed to create pipeline")?;

            // Create config with explicit stream selection
            let rs_config = sys::rs2_create_config(&mut err);
            check_rs2_error(err, "Failed to create config")?;

            // Enable color stream
            sys::rs2_config_enable_stream(
                rs_config,
                sys::rs2_stream_RS2_STREAM_COLOR,
                -1, // any index
                color_size.x as i32,
                color_size.y as i32,
                sys::rs2_format_RS2_FORMAT_RGB8,
                frame_rate,
                &mut err,
            );
            check_rs2_error(err, "Failed to enable color stream")?;

            // Enable depth stream if requested
            if let Some(depth_size) = config.depth {
                sys::rs2_config_enable_stream(
                    rs_config,
                    sys::rs2_stream_RS2_STREAM_DEPTH,
                    -1,
                    depth_size.x as i32,
                    depth_size.y as i32,
                    sys::rs2_format_RS2_FORMAT_Z16,
                    frame_rate,
                    &mut err,
                );
                check_rs2_error(err, "Failed to enable depth stream")?;
            }

            // Enable IR streams if requested
            if let Some(ir_size) = config.ir {
                for index in [1, 2] {
                    sys::rs2_config_enable_stream(
                        rs_config,
                        sys::rs2_stream_RS2_STREAM_INFRARED,
                        index,
                        ir_size.x as i32,
                        ir_size.y as i32,
                        sys::rs2_format_RS2_FORMAT_Y8,
                        frame_rate,
                        &mut err,
                    );
                    check_rs2_error(err, "Failed to enable IR stream")?;
                }
            }

            let profile = sys::rs2_pipeline_start_with_config(pipeline, rs_config, &mut err);
            if !err.is_null() {
                sys::rs2_delete_config(rs_config);
                return check_rs2_error(err, "Failed to start pipeline").map(|_| unreachable!());
            }
            sys::rs2_delete_config(rs_config);

            // Log device info
            log_device_info(device);

            // Clean up setup objects (pipeline and context stay alive)
            sys::rs2_delete_device(device);
            sys::rs2_delete_device_list(device_list);
            sys::rs2_delete_pipeline_profile(profile);

            self.pipeline = pipeline;
            self.context = ctx;
        }

        self.has_depth = config.depth.is_some();
        self.has_ir = config.ir.is_some();

        Ok(VideoInConfig::Realsense(RealsenseConfig {
            index: Some(device_index as usize),
            color: Some(color_size),
            depth: config.depth,
            ir: config.ir,
            frame_rate: Some(frame_rate as f32),
        }))
    }

    fn close(&mut self) {
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

    fn blocking_capture(&mut self) -> Result<VideoFrame, VideoError> {
        if !self.is_open() {
            return Err(VideoError::Stream("Pipeline not started".to_string()));
        }

        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();

            // Use the BLOCKING rs2_pipeline_wait_for_frames
            let composite =
                sys::rs2_pipeline_wait_for_frames(self.pipeline, CAPTURE_TIMEOUT_MS, &mut err);
            check_rs2_error(err, "Failed to wait for frames")?;

            let result = extract_video_frame(composite, self.has_depth, self.has_ir);
            sys::rs2_release_frame(composite);
            result
        }
    }
}

impl Drop for Realsense {
    fn drop(&mut self) {
        self.close();
    }
}

fn check_rs2_error(err: *mut sys::rs2_error, context: &str) -> Result<(), VideoError> {
    if err.is_null() {
        return Ok(());
    }
    unsafe {
        let msg = std::ffi::CStr::from_ptr(sys::rs2_get_error_message(err))
            .to_string_lossy()
            .into_owned();
        sys::rs2_free_error(err);
        Err(VideoError::Device(format!("{context}: {msg}")))
    }
}

unsafe fn log_device_info(dev: *mut sys::rs2_device) {
    unsafe {
        let mut err = std::ptr::null_mut::<sys::rs2_error>();

        let name =
            sys::rs2_get_device_info(dev, sys::rs2_camera_info_RS2_CAMERA_INFO_NAME, &mut err);
        let name_str = if err.is_null() && !name.is_null() {
            std::ffi::CStr::from_ptr(name).to_string_lossy()
        } else {
            if !err.is_null() {
                sys::rs2_free_error(err);
                err = std::ptr::null_mut();
            }
            "?".into()
        };

        let fw = sys::rs2_get_device_info(
            dev,
            sys::rs2_camera_info_RS2_CAMERA_INFO_FIRMWARE_VERSION,
            &mut err,
        );
        let fw_str = if err.is_null() && !fw.is_null() {
            std::ffi::CStr::from_ptr(fw).to_string_lossy()
        } else {
            if !err.is_null() {
                sys::rs2_free_error(err);
                err = std::ptr::null_mut();
            }
            "?".into()
        };

        let usb = sys::rs2_get_device_info(
            dev,
            sys::rs2_camera_info_RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR,
            &mut err,
        );
        let usb_str = if err.is_null() && !usb.is_null() {
            std::ffi::CStr::from_ptr(usb).to_string_lossy()
        } else {
            if !err.is_null() {
                sys::rs2_free_error(err);
            }
            "?".into()
        };

        log_info!("realsense: device={name_str}, fw={fw_str}, usb={usb_str}");
    }
}

unsafe fn extract_video_frame(
    composite: *mut sys::rs2_frame,
    has_depth: bool,
    has_ir: bool,
) -> Result<VideoFrame, VideoError> {
    unsafe {
        let mut err = std::ptr::null_mut::<sys::rs2_error>();
        let count = sys::rs2_embedded_frames_count(composite, &mut err);
        if !err.is_null() {
            sys::rs2_free_error(err);
            return Err(VideoError::Stream(
                "Failed to get embedded frame count".to_string(),
            ));
        }

        let mut color: Option<Image> = None;
        let mut depth: Option<DepthImage> = None;
        let mut left: Option<IrImage> = None;
        let mut right: Option<IrImage> = None;

        for i in 0..count {
            err = std::ptr::null_mut();
            let frame = sys::rs2_extract_frame(composite, i, &mut err);
            if !err.is_null() {
                sys::rs2_free_error(err);
                continue;
            }

            err = std::ptr::null_mut();
            let mut stream: sys::rs2_stream = 0;
            let mut format: sys::rs2_format = 0;
            let mut index: i32 = 0;
            let mut _uid: i32 = 0;
            let mut _fps: i32 = 0;
            let profile = sys::rs2_get_frame_stream_profile(frame, &mut err);
            if !err.is_null() {
                sys::rs2_free_error(err);
                sys::rs2_release_frame(frame);
                continue;
            }
            err = std::ptr::null_mut();
            sys::rs2_get_stream_profile_data(
                profile,
                &mut stream,
                &mut format,
                &mut index,
                &mut _uid,
                &mut _fps,
                &mut err,
            );
            if !err.is_null() {
                sys::rs2_free_error(err);
                sys::rs2_release_frame(frame);
                continue;
            }

            err = std::ptr::null_mut();
            let w = sys::rs2_get_frame_width(frame, &mut err) as usize;
            err = std::ptr::null_mut();
            let h = sys::rs2_get_frame_height(frame, &mut err) as usize;
            err = std::ptr::null_mut();
            let data_size = sys::rs2_get_frame_data_size(frame, &mut err) as usize;
            err = std::ptr::null_mut();
            let data_ptr = sys::rs2_get_frame_data(frame, &mut err) as *const u8;

            let size = Vec2::new(w, h);

            match stream {
                s if s == sys::rs2_stream_RS2_STREAM_COLOR => {
                    let data = std::slice::from_raw_parts(data_ptr, data_size).to_vec();
                    color = Some(Image::new(size, data, PixelFormat::Rgb8));
                }
                s if s == sys::rs2_stream_RS2_STREAM_DEPTH && has_depth => {
                    let u16_ptr = data_ptr as *const u16;
                    let u16_len = data_size / 2;
                    let data = std::slice::from_raw_parts(u16_ptr, u16_len).to_vec();
                    depth = Some(DepthImage { size, data });
                }
                s if s == sys::rs2_stream_RS2_STREAM_INFRARED && has_ir => {
                    let data = std::slice::from_raw_parts(data_ptr, data_size).to_vec();
                    match index {
                        1 => left = Some(IrImage { size, data }),
                        2 => right = Some(IrImage { size, data }),
                        _ => {}
                    }
                }
                _ => {}
            }

            sys::rs2_release_frame(frame);
        }

        let color =
            color.ok_or_else(|| VideoError::Stream("No color frame in composite".to_string()))?;

        Ok(VideoFrame {
            color,
            depth,
            left,
            right,
        })
    }
}
