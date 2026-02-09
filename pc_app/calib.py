import numpy as np


def compute_homography(image_points, width_mm, height_mm):
    if len(image_points) != 4:
        raise ValueError("need 4 image points for homography")
    # target plane corners in mm: (0,0), (W,0), (W,H), (0,H)
    plane_points = np.array(
        [
            [0.0, 0.0],
            [width_mm, 0.0],
            [width_mm, height_mm],
            [0.0, height_mm],
        ],
        dtype=np.float32,
    )
    img_pts = np.array(image_points, dtype=np.float32)
    H, _ = cv2_find_homography(img_pts, plane_points)
    if H is None:
        raise RuntimeError("homography failed")
    return H


def cv2_find_homography(src_pts, dst_pts):
    # local import so calib.py stays lightweight
    import cv2

    H, mask = cv2.findHomography(src_pts, dst_pts, method=0)
    return H, mask


def pixel_to_plane(H, pixel_xy):
    x, y = pixel_xy
    vec = np.array([x, y, 1.0], dtype=np.float64)
    out = H.dot(vec)
    if out[2] == 0:
        return None
    return (out[0] / out[2], out[1] / out[2])


def plane_to_pixel(H, plane_xy):
    # inverse transform
    import numpy as np

    Hinv = np.linalg.inv(H)
    x, y = plane_xy
    vec = np.array([x, y, 1.0], dtype=np.float64)
    out = Hinv.dot(vec)
    if out[2] == 0:
        return None
    return (out[0] / out[2], out[1] / out[2])
