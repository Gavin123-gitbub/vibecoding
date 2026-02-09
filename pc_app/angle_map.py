import numpy as np


class AngleMapper:
    def __init__(self):
        self.plane_points = []
        self.angle_points = []
        self.A = None  # 2x3 affine-ish map

    def add_point(self, plane_xy, angles_yaw_pitch):
        self.plane_points.append(plane_xy)
        self.angle_points.append(angles_yaw_pitch)

    def is_ready(self):
        return len(self.plane_points) >= 4

    def fit(self):
        if len(self.plane_points) < 4:
            raise ValueError("need >=4 points to fit angle map")
        # fit linear model: [yaw, pitch] = M * [x, y, 1]
        X = []
        Y = []
        for (x, y), (yaw, pitch) in zip(self.plane_points, self.angle_points):
            X.append([x, y, 1.0])
            Y.append([yaw, pitch])
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        # least squares
        M, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        self.A = M.T  # 2x3

    def map(self, plane_xy):
        if self.A is None:
            raise RuntimeError("angle map not fitted")
        x, y = plane_xy
        vec = np.array([x, y, 1.0], dtype=np.float64)
        out = self.A.dot(vec)
        return (float(out[0]), float(out[1]))
