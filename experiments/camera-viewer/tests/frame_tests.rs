use camera_viewer::Frame;
use codec::Codec;

fn round_trip(frame: &Frame) {
    let mut buf = Vec::new();
    frame.encode(&mut buf);

    let mut pos = 0;
    let decoded = Frame::decode(&buf, &mut pos).unwrap();

    assert_eq!(decoded.width(), frame.width());
    assert_eq!(decoded.height(), frame.height());
    assert_eq!(decoded.data(), frame.data());
    assert_eq!(pos, buf.len(), "all bytes should be consumed");
}

#[test]
fn test_frame_roundtrip_empty() {
    let frame = Frame::new(640, 480, vec![]);
    round_trip(&frame);
}

#[test]
fn test_frame_roundtrip_with_data() {
    let frame = Frame::new(2, 2, vec![255, 0, 128, 64, 32, 16, 8, 4, 2, 1, 100, 200]);
    round_trip(&frame);
}
