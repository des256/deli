use video::{VideoData, VideoFrame};

#[cfg(feature = "v4l2")]
#[tokio::test]
async fn test_v4l2_camera_implements_stream() {
    fn assert_stream<T: Stream<Item = Result<VideoFrame, CameraError>>>() {}
    assert_stream::<video::V4l2Camera>();
}

#[tokio::test]
async fn test_video_frame_rgb_variant() {
    let tensor = base::Tensor::new(vec![2, 2, 3], vec![0u8; 12]).unwrap();
    let frame = VideoFrame {
        data: VideoData::Rgb(tensor),
        width: 2,
        height: 2,
    };
    match frame.data {
        VideoData::Rgb(t) => assert_eq!(t.shape, vec![2, 2, 3]),
        VideoData::Jpeg(_) => panic!("Expected VideoData::Rgb"),
    }
}

#[tokio::test]
async fn test_video_frame_jpeg_variant() {
    let frame = VideoFrame {
        data: VideoData::Jpeg(vec![0xFF, 0xD8, 0xFF, 0xE0]),
        width: 640,
        height: 480,
    };
    match frame.data {
        VideoData::Jpeg(data) => {
            assert_eq!(data.len(), 4);
            assert_eq!(data[0], 0xFF);
        }
        VideoData::Rgb(_) => panic!("Expected VideoData::Jpeg"),
    }
}
