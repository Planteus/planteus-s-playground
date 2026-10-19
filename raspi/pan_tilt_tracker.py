#!/usr/bin/env python3
"""
Pan-tilt camera object tracker for Raspberry Pi 2B.

Hardware assumptions:
  - Pan servo  -> GPIO 18 (BCM)
  - Tilt servo -> GPIO 24 (BCM)
  - Camera: NoIR camera module (via picamera2 / libcamera stack)
  - pigpiod running (sudo systemctl enable pigpiod --now)

Tracking method: HSV color threshold -> largest contour centroid.
This is deliberately lightweight so it runs at a usable frame rate on a Pi 2B.
Swap out `find_target()` later for something fancier once this loop feels smooth.
"""

import time
import cv2
import numpy as np
import pigpio

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PAN_PIN = 18
TILT_PIN = 24

# Servo pulse widths in microseconds (typical hobby servo range; adjust to your servo's spec sheet)
SERVO_MIN_US = 500
SERVO_MAX_US = 2500
SERVO_CENTER_US = 1500

# Soft-limit the range of motion so the mechanism can't slam into its stops
PAN_MIN_DEG, PAN_MAX_DEG = 20, 160
TILT_MIN_DEG, TILT_MAX_DEG = 20, 160

# Frame size — keep this small on a Pi 2B. 320x240 is a good starting point.
FRAME_W, FRAME_H = 320, 240

# Control tuning
DEADZONE_PX = 15          # ignore offsets smaller than this (prevents jitter/hunting)
KP_PAN = 0.03             # proportional gain, pixels-of-error -> degrees-of-move
KP_TILT = 0.03
KD_PAN = 0.02             # derivative gain -- reacts to closing speed, damps overshoot
KD_TILT = 0.02
MAX_STEP_DEG = 3.0        # clamp how far a servo can move in a single frame (smooths motion)

# Exponential smoothing on the detected centroid (0-1). Lower = smoother but
# laggier; higher = snappier but noisier. Small/distant targets have fewer
# pixels contributing to their centroid, so their raw position is noisier
# frame to frame -- this smooths that out before it ever reaches the
# controller, which is most of what was causing side-to-side wiggling.
SMOOTHING_ALPHA = 0.4

# Click-to-select-target tolerances: how far to pad the sampled H/S/V outward
# when building a tracking range from a click. Hue is tight (colors close in
# hue but different lighting are still the same object); saturation/value get
# more slack since those shift a lot with lighting/shadow.
CLICK_SAMPLE_RADIUS = 3   # sample a (2*r+1)x(2*r+1) patch around the click, not just 1px
CLICK_HUE_TOLERANCE = 12
CLICK_SAT_TOLERANCE = 80
CLICK_VAL_TOLERANCE = 80

# Flip to -1 if that axis moves the wrong way for your servo horn orientation.
# Tilt is inverted on this rig.
PAN_INVERT = 1
TILT_INVERT = -1

# HSV color range to track. Currently set for the yellow teapot, which renders
# as cyan on this sensor -- based on a click-sample reading of (H=101, S=245,
# V=183). Doesn't wrap around the hue circle, so one range is enough. Padding
# added around the measured value for lighting/angle variation.
HSV_LOWER1 = np.array([85, 120, 80])
HSV_UPPER1 = np.array([116, 255, 255])
# Second range unused for a non-wrapping color -- set to never match.
HSV_LOWER2 = np.array([180, 255, 255])
HSV_UPPER2 = np.array([180, 255, 255])
MIN_CONTOUR_AREA = 200    # ignore tiny noise blobs

# Ignore this top fraction of the frame when matching -- the ceiling/roof was
# getting picked up because it happens to render a similar false-color to the
# shirt under this sensor. A person's shirt is very unlikely to appear right
# at the top edge of frame, so this cheaply excludes ceiling false-positives
# without having to fight the color thresholds. Raise it if the roof still
# creeps in at the edges; lower it (or set to 0) if it starts excluding
# legitimate high-angle tracking.
ROI_TOP_FRACTION = 0.15

# ---------------------------------------------------------------------------
# Preview / debug display
# ---------------------------------------------------------------------------
# You're accessing the Pi over VNC, so a normal cv2 window works fine here --
# VNC gives you a full desktop session, no extra streaming server needed.
SHOW_PREVIEW = True
DEBUG_COLOR = (0, 255, 0)          # green overlay lines/text
DEBUG_COLOR_NO_TARGET = (0, 0, 255)  # red when nothing is found

# ---------------------------------------------------------------------------
# Servo helper
# ---------------------------------------------------------------------------

class Servo:
    def __init__(self, pi, pin, min_deg, max_deg, start_deg=90):
        self.pi = pi
        self.pin = pin
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.angle = start_deg
        self._write(self.angle)

    def _deg_to_us(self, deg):
        span = SERVO_MAX_US - SERVO_MIN_US
        return int(SERVO_MIN_US + (deg / 180.0) * span)

    def _write(self, deg):
        deg = max(self.min_deg, min(self.max_deg, deg))
        self.angle = deg
        self.pi.set_servo_pulsewidth(self.pin, self._deg_to_us(deg))

    def nudge(self, delta_deg):
        self._write(self.angle + delta_deg)

    def stop(self):
        # 0 pulsewidth = stop sending pulses (many servos, saves a bit of power/heat)
        self.pi.set_servo_pulsewidth(self.pin, 0)


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------

def find_target(frame_bgr):
    """Return ((cx, cy) or None, mask). Mask is returned too so it can be
    shown in a debug window -- seeing the mask is the fastest way to tell
    whether your HSV range is matching the right thing."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, HSV_LOWER1, HSV_UPPER1)
    mask2 = cv2.inRange(hsv, HSV_LOWER2, HSV_UPPER2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    if ROI_TOP_FRACTION > 0:
        cutoff_row = int(mask.shape[0] * ROI_TOP_FRACTION)
        mask[:cutoff_row, :] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None, mask

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


def compute_hsv_bounds(h, s, v):
    """Build (lower1, upper1, lower2, upper2) HSV bounds around a sampled
    color, handling hue wraparound (colors near 0/180, like red) by splitting
    into two ranges. The second range is a never-match sentinel when no
    wraparound is needed, matching the convention used elsewhere in this file."""
    s_lower = max(0, int(s) - CLICK_SAT_TOLERANCE)
    v_lower = max(0, int(v) - CLICK_VAL_TOLERANCE)
    h_low = int(h) - CLICK_HUE_TOLERANCE
    h_high = int(h) + CLICK_HUE_TOLERANCE

    never_match_lower = np.array([180, 255, 255])
    never_match_upper = np.array([180, 255, 255])

    if h_low < 0:
        lower1 = np.array([0, s_lower, v_lower])
        upper1 = np.array([h_high, 255, 255])
        lower2 = np.array([180 + h_low, s_lower, v_lower])
        upper2 = np.array([179, 255, 255])
    elif h_high > 179:
        lower1 = np.array([h_low, s_lower, v_lower])
        upper1 = np.array([179, 255, 255])
        lower2 = np.array([0, s_lower, v_lower])
        upper2 = np.array([h_high - 180, 255, 255])
    else:
        lower1 = np.array([h_low, s_lower, v_lower])
        upper1 = np.array([h_high, 255, 255])
        lower2 = never_match_lower
        upper2 = never_match_upper

    return lower1, upper1, lower2, upper2


# Shared with the mouse callback so it can read the most recent frame.
# We keep the RAW, unconverted frame here -- the click handler converts it
# the same way the main pipeline does, so the color it picks matches exactly
# what find_target() will see.
_latest_raw_frame = None

# Set by the click handler when a new target color is picked; main() checks
# this each loop to reset position smoothing (a new target shouldn't inherit
# the old target's smoothed position).
_color_just_changed = False


def on_mouse_click(event, x, y, flags, userdata):
    """Click any object in the 'tracker' preview window to start tracking its
    color. Samples a small patch (not just one pixel) around the click for a
    more stable reading, then builds HSV bounds around it automatically."""
    global HSV_LOWER1, HSV_UPPER1, HSV_LOWER2, HSV_UPPER2, _color_just_changed

    if event != cv2.EVENT_LBUTTONDOWN or _latest_raw_frame is None:
        return

    h_img, w_img = _latest_raw_frame.shape[:2]
    r = CLICK_SAMPLE_RADIUS
    y0, y1 = max(0, y - r), min(h_img, y + r + 1)
    x0, x1 = max(0, x - r), min(w_img, x + r + 1)
    patch = _latest_raw_frame[y0:y1, x0:x1]

    # Same conversion the main pipeline uses (see main()), applied here so the
    # sampled color matches what find_target() actually computes.
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    h = int(np.median(hsv_patch[:, :, 0]))
    s = int(np.median(hsv_patch[:, :, 1]))
    v = int(np.median(hsv_patch[:, :, 2]))

    HSV_LOWER1, HSV_UPPER1, HSV_LOWER2, HSV_UPPER2 = compute_hsv_bounds(h, s, v)
    _color_just_changed = True

    print(f"\n[new target] clicked ({x},{y}) -> sampled HSV = ({h}, {s}, {v})")
    print(f"  now tracking range1 {HSV_LOWER1.tolist()} - {HSV_UPPER1.tolist()}")
    if not np.array_equal(HSV_LOWER2, np.array([180, 255, 255])):
        print(f"  plus wraparound range2 {HSV_LOWER2.tolist()} - {HSV_UPPER2.tolist()}")


def draw_debug_overlay(frame_bgr, target, center, err_x, err_y, pan_angle, tilt_angle, status):
    """Draw a crosshair, deadzone box, tracked-target circle, and status text
    onto a copy of the frame so you can see exactly what the loop is doing."""
    out = frame_bgr.copy()
    cx0, cy0 = center

    # Crosshair at frame center
    cv2.drawMarker(out, (cx0, cy0), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)

    # Deadzone box around center
    cv2.rectangle(
        out,
        (cx0 - DEADZONE_PX, cy0 - DEADZONE_PX),
        (cx0 + DEADZONE_PX, cy0 + DEADZONE_PX),
        (255, 255, 0), 1,
    )

    color = DEBUG_COLOR if target is not None else DEBUG_COLOR_NO_TARGET

    if target is not None:
        tx, ty = target
        cv2.circle(out, (tx, ty), 12, color, 2)
        cv2.line(out, (cx0, cy0), (tx, ty), color, 1)

    lines = [
        f"target: {target if target else 'NONE'}",
        f"err_x={err_x:+d}  err_y={err_y:+d}" if target else "err: n/a",
        f"pan={pan_angle:.1f} deg  tilt={tilt_angle:.1f} deg",
        status,
    ]
    for i, line in enumerate(lines):
        y = 16 + i * 16
        cv2.putText(out, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("Could not connect to pigpiod. Is it running? (sudo systemctl start pigpiod)")

    pan = Servo(pi, PAN_PIN, PAN_MIN_DEG, PAN_MAX_DEG)
    tilt = Servo(pi, TILT_PIN, TILT_MIN_DEG, TILT_MAX_DEG)

    # Using picamera2 (current Raspberry Pi OS camera stack). If you're on the
    # legacy stack, swap this block for `picamera` + a VideoCapture-style read loop.
    from picamera2 import Picamera2
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)  # let auto-exposure/AWB settle

    center_x, center_y = FRAME_W // 2, FRAME_H // 2

    # Position smoothing and PD-control state, carried across loop iterations.
    smoothed_cx, smoothed_cy = None, None
    prev_err_x, prev_err_y = 0, 0

    if SHOW_PREVIEW:
        cv2.namedWindow("tracker")
        cv2.setMouseCallback("tracker", on_mouse_click)

    try:
        while True:
            frame = picam2.capture_array()
            # Camera is mounted upside down (cabling reasons) -- correct with a
            # single 180-degree rotation, applied before anything else touches
            # the frame so tracking, the mask, and the preview are all correct.
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            global _latest_raw_frame
            _latest_raw_frame = frame

            # Convert to the channel order find_target()/the click handler expect.
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            target, mask = find_target(frame_bgr)
            err_x = err_y = 0
            status = "idle"

            global _color_just_changed
            if _color_just_changed:
                smoothed_cx, smoothed_cy = None, None
                prev_err_x, prev_err_y = 0, 0
                _color_just_changed = False

            if target is not None:
                raw_cx, raw_cy = target

                if smoothed_cx is None:
                    smoothed_cx, smoothed_cy = raw_cx, raw_cy
                else:
                    smoothed_cx = SMOOTHING_ALPHA * raw_cx + (1 - SMOOTHING_ALPHA) * smoothed_cx
                    smoothed_cy = SMOOTHING_ALPHA * raw_cy + (1 - SMOOTHING_ALPHA) * smoothed_cy

                err_x = int(smoothed_cx - center_x)
                err_y = int(smoothed_cy - center_y)

                moved_pan = moved_tilt = False

                if abs(err_x) > DEADZONE_PX:
                    d_x = err_x - prev_err_x
                    step = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, -(err_x * KP_PAN + d_x * KD_PAN) * PAN_INVERT))
                    pan.nudge(step)
                    moved_pan = True

                if abs(err_y) > DEADZONE_PX:
                    d_y = err_y - prev_err_y
                    step = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, -(err_y * KP_TILT + d_y * KD_TILT) * TILT_INVERT))
                    tilt.nudge(step)
                    moved_tilt = True

                prev_err_x, prev_err_y = err_x, err_y

                if moved_pan or moved_tilt:
                    status = f"moving (pan={'Y' if moved_pan else 'n'} tilt={'Y' if moved_tilt else 'n'})"
                else:
                    status = "target inside deadzone -- holding still (this is expected, not stuck)"
            else:
                smoothed_cx, smoothed_cy = None, None
                prev_err_x, prev_err_y = 0, 0
                status = "no target found -- click the object you want to track"

            if SHOW_PREVIEW:
                debug_frame = draw_debug_overlay(
                    frame_bgr, target, (center_x, center_y), err_x, err_y,
                    pan.angle, tilt.angle, status,
                )
                cv2.imshow("tracker", debug_frame)
                cv2.imshow("mask (white = matched)", mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        pan.stop()
        tilt.stop()
        pi.stop()
        picam2.stop()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
