import cv2
import numpy as np


def detect_laser_center(frame_bgr, color_space="HSV", hsv_lower=None, hsv_upper=None):
    if color_space.upper() == "HSV":
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array(hsv_lower if hsv_lower is not None else [0, 0, 240], dtype=np.uint8)
        upper = np.array(hsv_upper if hsv_upper is not None else [180, 40, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 3:
        return None, mask

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, mask
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask
