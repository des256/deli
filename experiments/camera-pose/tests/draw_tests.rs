use deli_infer::{Keypoint, KeypointIndex, PoseDetection};
use deli_math::{Rect, Vec2};

mod draw {
    include!("../src/draw.rs");
}

use draw::*;

#[test]
fn test_draw_line_horizontal() {
    let mut buf = vec![0u8; 10 * 5 * 3]; // 10x5 RGB image
    let white = [255, 255, 255];

    draw_line(&mut buf, 10, 5, 1, 2, 8, 2, white);

    // Check pixels at y=2, x=1..8 are white
    for x in 1..=8 {
        let idx = (2 * 10 + x) * 3;
        assert_eq!(buf[idx..idx+3], white, "Pixel at ({}, 2) should be white", x);
    }

    // Check pixel outside line is black
    assert_eq!(&buf[0..3], [0, 0, 0], "Pixel at (0, 0) should be black");
}

#[test]
fn test_draw_line_vertical() {
    let mut buf = vec![0u8; 5 * 10 * 3]; // 5x10 RGB image
    let red = [255, 0, 0];

    draw_line(&mut buf, 5, 10, 2, 1, 2, 8, red);

    // Check pixels at x=2, y=1..8 are red
    for y in 1..=8 {
        let idx = (y * 5 + 2) * 3;
        assert_eq!(buf[idx..idx+3], red, "Pixel at (2, {}) should be red", y);
    }
}

#[test]
fn test_draw_line_clips_to_bounds() {
    let mut buf = vec![0u8; 10 * 10 * 3];
    let white = [255, 255, 255];

    // Line goes out of bounds — should clip
    draw_line(&mut buf, 10, 10, -5, 5, 15, 5, white);

    // Should only draw from x=0 to x=9 at y=5
    for x in 0..10 {
        let idx = (5 * 10 + x) * 3;
        assert_eq!(buf[idx..idx+3], white, "Pixel at ({}, 5) should be white", x);
    }
}

#[test]
fn test_draw_filled_circle_basic() {
    let mut buf = vec![0u8; 20 * 20 * 3];
    let green = [0, 255, 0];

    draw_filled_circle(&mut buf, 20, 20, 10, 10, 3, green);

    // Check center pixel is green
    let idx = (10 * 20 + 10) * 3;
    assert_eq!(buf[idx..idx+3], green, "Center pixel should be green");

    // Check a pixel inside radius is green
    let idx = (10 * 20 + 12) * 3; // (12, 10) is inside radius 3
    assert_eq!(buf[idx..idx+3], green, "Pixel at (12, 10) should be green");

    // Check a pixel outside radius is black
    let idx = (10 * 20 + 15) * 3; // (15, 10) is outside radius 3
    assert_eq!(&buf[idx..idx+3], [0, 0, 0], "Pixel at (15, 10) should be black");
}

#[test]
fn test_draw_filled_circle_clips() {
    let mut buf = vec![0u8; 10 * 10 * 3];
    let blue = [0, 0, 255];

    // Circle center at (1, 1) with radius 5 — most of it out of bounds
    draw_filled_circle(&mut buf, 10, 10, 1, 1, 5, blue);

    // Should only fill visible pixels, no panic
    let idx = (1 * 10 + 1) * 3;
    assert_eq!(buf[idx..idx+3], blue, "Center should be blue");
}

#[test]
fn test_rgb_to_argb() {
    let rgb = vec![
        255, 0, 0,   // Red
        0, 255, 0,   // Green
        0, 0, 255,   // Blue
        128, 128, 128, // Gray
    ];

    let argb = rgb_to_argb(&rgb, 2, 2);

    assert_eq!(argb.len(), 4);
    assert_eq!(argb[0], 0x00FF0000); // Red as ARGB
    assert_eq!(argb[1], 0x0000FF00); // Green as ARGB
    assert_eq!(argb[2], 0x000000FF); // Blue as ARGB
    assert_eq!(argb[3], 0x00808080); // Gray as ARGB
}

#[test]
fn test_draw_skeleton_basic() {
    let mut buf = vec![0u8; 100 * 100 * 3];

    // Create a simple detection with two keypoints above threshold
    let mut keypoints = [Keypoint { position: Vec2::new(0.0, 0.0), confidence: 0.0 }; 17];

    // Set nose and left eye above threshold
    keypoints[KeypointIndex::Nose as usize] = Keypoint {
        position: Vec2::new(50.0, 50.0),
        confidence: 0.8
    };
    keypoints[KeypointIndex::LeftEye as usize] = Keypoint {
        position: Vec2::new(45.0, 45.0),
        confidence: 0.9
    };

    let detection = PoseDetection {
        bbox: Rect::new(Vec2::new(40.0, 40.0), Vec2::new(20.0, 20.0)),
        confidence: 0.95,
        keypoints,
    };

    draw_skeleton(&mut buf, 100, 100, &detection, 0.3);

    // Verify nose keypoint was drawn (center should have color)
    let idx = (50 * 100 + 50) * 3;
    assert_ne!(buf[idx..idx+3], [0, 0, 0], "Nose keypoint should be drawn");

    // Verify left eye keypoint was drawn
    let idx = (45 * 100 + 45) * 3;
    assert_ne!(buf[idx..idx+3], [0, 0, 0], "Left eye keypoint should be drawn");
}

#[test]
fn test_draw_skeleton_filters_low_confidence() {
    let mut buf = vec![0u8; 100 * 100 * 3];

    let mut keypoints = [Keypoint { position: Vec2::new(0.0, 0.0), confidence: 0.0 }; 17];

    // Set a keypoint below threshold
    keypoints[KeypointIndex::Nose as usize] = Keypoint {
        position: Vec2::new(50.0, 50.0),
        confidence: 0.2  // Below 0.3 threshold
    };

    let detection = PoseDetection {
        bbox: Rect::new(Vec2::new(40.0, 40.0), Vec2::new(20.0, 20.0)),
        confidence: 0.95,
        keypoints,
    };

    draw_skeleton(&mut buf, 100, 100, &detection, 0.3);

    // Verify nose was NOT drawn (confidence too low)
    let idx = (50 * 100 + 50) * 3;
    assert_eq!(buf[idx..idx+3], [0, 0, 0], "Low confidence keypoint should not be drawn");
}
