import time
import yaml
import cv2
from camera_openmv import OpenMVCamera
from calib import compute_homography, plane_to_pixel
from grid import generate_grid_points
from angle_map import AngleMapper
from laser_detect import detect_laser_center
from pid import PID
from serial_io import SerialSender


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ClickCollector:
    def __init__(self):
        self.points = []
        self.done = False

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            if len(self.points) >= 4:
                self.done = True


def collect_points(frame, title):
    collector = ClickCollector()
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, collector.on_mouse)
    while not collector.done:
        disp = frame.copy()
        for p in collector.points:
            cv2.circle(disp, p, 4, (0, 255, 0), -1)
        cv2.imshow(title, disp)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cv2.destroyWindow(title)
    return collector.points


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def main():
    cfg = load_config("pc_app/config.yaml")

    cam_cfg = cfg["app"]
    plane_cfg = cfg["plane"]
    angle_cfg = cfg["angles"]
    pid_cfg = cfg["pid"]

    cam = OpenMVCamera(
        index=cam_cfg["camera_index"],
        width=cam_cfg["camera_width"],
        height=cam_cfg["camera_height"],
        fps=cam_cfg["camera_fps"],
    )
    cam.open()

    sender = SerialSender(
        cam_cfg["serial_port"], cam_cfg["serial_baud"], cam_cfg["send_rate_hz"]
    )
    sender.open()

    ret, frame = cam.read()
    if not ret:
        print("Failed to capture initial frame.")
        return

    print("Click 4 corners of the target plane (clockwise).")
    img_points = collect_points(frame, "Calibration - Plane Corners")
    if len(img_points) != 4:
        print("Calibration aborted.")
        return

    H = compute_homography(img_points, plane_cfg["width_mm"], plane_cfg["height_mm"])

    # Angle map calibration (4 points)
    angle_map = AngleMapper()
    for i, p in enumerate(img_points):
        px = int(p[0])
        py = int(p[1])
        print(
            f"Point {i+1}/4. Aim laser to this corner and input yaw,pitch degrees."
        )
        user_in = input("Yaw,Pitch (deg): ").strip()
        yaw_s, pitch_s = user_in.split(",")
        angle_map.add_point(
            (0.0, 0.0) if i == 0 else (
                plane_cfg["width_mm"], 0.0
            ) if i == 1 else (
                plane_cfg["width_mm"], plane_cfg["height_mm"]
            ) if i == 2 else (
                0.0, plane_cfg["height_mm"]
            ),
            (float(yaw_s), float(pitch_s)),
        )
    angle_map.fit()

    grid_pts = generate_grid_points(
        plane_cfg["width_mm"], plane_cfg["height_mm"], plane_cfg["grid_spacing_mm"]
    )

    pid_x = PID(pid_cfg["kp"], pid_cfg["ki"], pid_cfg["kd"], pid_cfg["integral_limit"])
    pid_y = PID(pid_cfg["kp"], pid_cfg["ki"], pid_cfg["kd"], pid_cfg["integral_limit"])

    for idx, plane_xy in enumerate(grid_pts):
        target_pixel = plane_to_pixel(H, plane_xy)
        if target_pixel is None:
            continue

        yaw, pitch = angle_map.map(plane_xy)
        yaw = clamp(yaw, angle_cfg["yaw_min_deg"], angle_cfg["yaw_max_deg"])
        pitch = clamp(pitch, angle_cfg["pitch_min_deg"], angle_cfg["pitch_max_deg"])

        sender.send_angles(yaw, pitch)
        time.sleep(0.1)

        pid_x.reset()
        pid_y.reset()
        last_time = time.time()

        for _ in range(cam_cfg["max_pid_iters"]):
            ret, frame = cam.read()
            if not ret:
                break

            laser_center, mask = detect_laser_center(
                frame,
                color_space=cam_cfg["laser_color_space"],
                hsv_lower=cam_cfg["laser_hsv_lower"],
                hsv_upper=cam_cfg["laser_hsv_upper"],
            )

            if laser_center is None:
                continue

            ex = target_pixel[0] - laser_center[0]
            ey = target_pixel[1] - laser_center[1]
            err = (ex**2 + ey**2) ** 0.5
            if err <= cam_cfg["pixel_error_thresh"]:
                break

            now = time.time()
            dt = now - last_time
            last_time = now

            yaw += pid_x.update(ex, dt)
            pitch += pid_y.update(ey, dt)
            yaw = clamp(yaw, angle_cfg["yaw_min_deg"], angle_cfg["yaw_max_deg"])
            pitch = clamp(pitch, angle_cfg["pitch_min_deg"], angle_cfg["pitch_max_deg"])
            sender.send_angles(yaw, pitch)

            if cam_cfg["debug_draw"]:
                disp = frame.copy()
                cv2.circle(disp, (int(target_pixel[0]), int(target_pixel[1])), 5, (0, 255, 0), 2)
                cv2.circle(disp, (int(laser_center[0]), int(laser_center[1])), 5, (0, 0, 255), 2)
                cv2.imshow("Tracking", disp)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        print(f"Grid {idx+1}/{len(grid_pts)} done.")

    sender.close()
    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
