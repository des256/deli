use deli_video::{CameraError, VideoFrame};
use futures_core::Stream;

#[tokio::test]
async fn test_v4l2_camera_implements_stream() {
    fn assert_stream<T: Stream<Item = Result<VideoFrame, CameraError>>>() {}
    assert_stream::<deli_video::V4l2Camera>();
}

#[tokio::test]
async fn test_video_frame_rgb_variant() {
    let tensor = deli_base::Tensor::new(vec![2, 2, 3], vec![0u8; 12]).unwrap();
    let frame = VideoFrame::Rgb(tensor);
    match frame {
        VideoFrame::Rgb(t) => assert_eq!(t.shape, vec![2, 2, 3]),
        VideoFrame::Jpeg(_) => panic!("Expected VideoFrame::Rgb"),
    }
}

#[tokio::test]
async fn test_video_frame_jpeg_variant() {
    let frame = VideoFrame::Jpeg(vec![0xFF, 0xD8, 0xFF, 0xE0]);
    match frame {
        VideoFrame::Jpeg(data) => {
            assert_eq!(data.len(), 4);
            assert_eq!(data[0], 0xFF);
        }
        VideoFrame::Rgb(_) => panic!("Expected VideoFrame::Jpeg"),
    }
}
