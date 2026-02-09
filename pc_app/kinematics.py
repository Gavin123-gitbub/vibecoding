import math


def board_to_global(xb_mm, yb_mm, oq_mm):
    # Gimbal global coordinates: xg fixed to OQ, yg depends on xb, zg depends on yb
    xg = oq_mm
    yg = -xb_mm
    zg = yb_mm
    return xg, yg, zg


def inverse_kinematics(oq_mm, ol_mm, xb_mm, yb_mm):
    xg, yg, zg = board_to_global(xb_mm, yb_mm, oq_mm)
    alpha = math.degrees(math.atan2(yg, xg))  # yaw

    theta = math.atan2(zg, xg)
    op3 = math.hypot(xg, zg)
    if op3 == 0:
        return None
    ratio = ol_mm / op3
    if ratio > 1:
        ratio = 1
    if ratio < -1:
        ratio = -1
    lam = math.acos(ratio)
    beta = math.degrees((math.pi / 2.0) - (theta + lam))  # pitch
    return alpha, beta


def plane_to_board(plane_xy, width_mm, height_mm, origin_mode="center"):
    x, y = plane_xy
    if origin_mode == "center":
        xb = x - width_mm / 2.0
        yb = y - height_mm / 2.0
    else:
        xb = x
        yb = y
    return xb, yb
