use deli_infer::{KeypointIndex, PoseDetection};

/// Draw a line using Bresenham's algorithm with clipping
pub fn draw_line(
    buf: &mut [u8],
    width: usize,
    height: usize,
    mut x0: i32,
    mut y0: i32,
    mut x1: i32,
    mut y1: i32,
    color: [u8; 3],
) {
    // Cohen-Sutherland line clipping to bounds
    loop {
        let outcode0 = compute_outcode(x0, y0, width as i32, height as i32);
        let outcode1 = compute_outcode(x1, y1, width as i32, height as i32);

        if (outcode0 | outcode1) == 0 {
            // Both points inside — proceed to draw
            break;
        } else if (outcode0 & outcode1) != 0 {
            // Both points outside same edge — line completely clipped
            return;
        } else {
            // Line crosses bounds — clip it
            let outcode = if outcode0 != 0 { outcode0 } else { outcode1 };
            let (x, y) = clip_point(x0, y0, x1, y1, outcode, width as i32, height as i32);

            if outcode == outcode0 {
                x0 = x;
                y0 = y;
            } else {
                x1 = x;
                y1 = y;
            }
        }
    }

    // Bresenham line drawing
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;

    loop {
        set_pixel(buf, width, x0 as usize, y0 as usize, color);

        if x0 == x1 && y0 == y1 {
            break;
        }

        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x0 += sx;
        }
        if e2 < dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Draw a filled circle with clipping
pub fn draw_filled_circle(
    buf: &mut [u8],
    width: usize,
    height: usize,
    cx: i32,
    cy: i32,
    radius: i32,
    color: [u8; 3],
) {
    let r2 = radius * radius;

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= r2 {
                let x = cx + dx;
                let y = cy + dy;

                // Clip to bounds
                if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                    set_pixel(buf, width, x as usize, y as usize, color);
                }
            }
        }
    }
}

/// Draw COCO skeleton on an RGB buffer
pub fn draw_skeleton(
    buf: &mut [u8],
    width: usize,
    height: usize,
    detection: &PoseDetection,
    kp_threshold: f32,
) {
    use KeypointIndex::*;

    // Define skeleton connections (COCO 17-keypoint skeleton)
    let connections = [
        // Face (cyan)
        (Nose, LeftEye, [0, 255, 255]),
        (Nose, RightEye, [0, 255, 255]),
        (LeftEye, LeftEar, [0, 255, 255]),
        (RightEye, RightEar, [0, 255, 255]),
        // Torso (green)
        (LeftShoulder, RightShoulder, [0, 255, 0]),
        (LeftShoulder, LeftHip, [0, 255, 0]),
        (RightShoulder, RightHip, [0, 255, 0]),
        (LeftHip, RightHip, [0, 255, 0]),
        // Arms (yellow)
        (LeftShoulder, LeftElbow, [255, 255, 0]),
        (RightShoulder, RightElbow, [255, 255, 0]),
        (LeftElbow, LeftWrist, [255, 255, 0]),
        (RightElbow, RightWrist, [255, 255, 0]),
        // Legs (magenta)
        (LeftHip, LeftKnee, [255, 0, 255]),
        (RightHip, RightKnee, [255, 0, 255]),
        (LeftKnee, LeftAnkle, [255, 0, 255]),
        (RightKnee, RightAnkle, [255, 0, 255]),
        // Neck connections
        (Nose, LeftShoulder, [255, 255, 255]),
        (Nose, RightShoulder, [255, 255, 255]),
    ];

    // Draw skeleton lines
    for (kp1, kp2, color) in &connections {
        let pt1 = detection.keypoint(*kp1);
        let pt2 = detection.keypoint(*kp2);

        if pt1.confidence >= kp_threshold && pt2.confidence >= kp_threshold {
            draw_line(
                buf,
                width,
                height,
                pt1.position.x as i32,
                pt1.position.y as i32,
                pt2.position.x as i32,
                pt2.position.y as i32,
                *color,
            );
        }
    }

    // Draw keypoint dots
    for kp in &detection.keypoints {
        if kp.confidence >= kp_threshold {
            draw_filled_circle(
                buf,
                width,
                height,
                kp.position.x as i32,
                kp.position.y as i32,
                3,
                [255, 255, 255],
            );
        }
    }
}

/// Convert HWC RGB buffer to packed ARGB u32 for minifb
pub fn rgb_to_argb(buf: &[u8], width: usize, height: usize) -> Vec<u32> {
    let mut argb = Vec::with_capacity(width * height);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let r = buf[idx] as u32;
            let g = buf[idx + 1] as u32;
            let b = buf[idx + 2] as u32;

            // Pack as 0x00RRGGBB
            argb.push((r << 16) | (g << 8) | b);
        }
    }

    argb
}

// Helper functions

fn set_pixel(buf: &mut [u8], width: usize, x: usize, y: usize, color: [u8; 3]) {
    let idx = (y * width + x) * 3;
    buf[idx] = color[0];
    buf[idx + 1] = color[1];
    buf[idx + 2] = color[2];
}

// Cohen-Sutherland clipping helpers
const INSIDE: u8 = 0; // 0000
const LEFT: u8 = 1;   // 0001
const RIGHT: u8 = 2;  // 0010
const BOTTOM: u8 = 4; // 0100
const TOP: u8 = 8;    // 1000

fn compute_outcode(x: i32, y: i32, width: i32, height: i32) -> u8 {
    let mut code = INSIDE;
    if x < 0 {
        code |= LEFT;
    } else if x >= width {
        code |= RIGHT;
    }
    if y < 0 {
        code |= TOP;
    } else if y >= height {
        code |= BOTTOM;
    }
    code
}

fn clip_point(
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    outcode: u8,
    width: i32,
    height: i32,
) -> (i32, i32) {
    let dx = x1 - x0;
    let dy = y1 - y0;

    if outcode & TOP != 0 {
        // Point is above y=0
        let x = x0 + dx * (0 - y0) / dy;
        (x, 0)
    } else if outcode & BOTTOM != 0 {
        // Point is below y=height-1
        let x = x0 + dx * (height - 1 - y0) / dy;
        (x, height - 1)
    } else if outcode & LEFT != 0 {
        // Point is left of x=0
        let y = y0 + dy * (0 - x0) / dx;
        (0, y)
    } else {
        // Point is right of x=width-1
        let y = y0 + dy * (width - 1 - x0) / dx;
        (width - 1, y)
    }
}
