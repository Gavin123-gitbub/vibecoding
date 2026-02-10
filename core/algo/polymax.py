import numpy as np
from scipy import linalg


class PolyMaxLSCF:
    """
    PolyMax / LSCF (Right Matrix Fraction Description) 实现
    模型: H(ω) = B(ω) * A(ω)^{-1}
    A(z) = 1 + a1 z^-1 + ... + am z^-m
    B(z) = b0 + b1 z^-1 + ... + bm z^-m
    """

    def __init__(self, max_order: int = 64):
        self.max_order = max_order

    @staticmethod
    def _ensure_frf_shape(frf):
        frf = np.asarray(frf)
        if frf.ndim == 1:
            frf = frf[:, None]
        return frf

    @staticmethod
    def _estimate_dt(freq_axis):
        fmax = np.max(freq_axis)
        if fmax <= 0:
            raise ValueError("invalid frequency axis")
        fs = 2.0 * fmax
        return 1.0 / fs

    def fit(self, frf: np.ndarray, freq_axis: np.ndarray, order: int):
        """
        计算单阶 PolyMax 参数
        返回: poles_z (Z域), poles_s (连续域), a, b
        """
        if order < 1:
            raise ValueError("order must be >= 1")

        frf = self._ensure_frf_shape(frf)
        n_freq, n_ch = frf.shape
        freq_axis = np.asarray(freq_axis)
        if freq_axis.shape[0] != n_freq:
            raise ValueError("freq_axis length mismatch")

        dt = self._estimate_dt(freq_axis)
        omega = 2.0 * np.pi * freq_axis
        z = np.exp(-1j * omega * dt)  # Z^-1 基函数

        # 线性化: H(ω) * (1 + a1 z + ... + am z^m) - (b0 + b1 z + ... + bm z^m) = 0
        # 将未知量堆叠为 x = [a1..am, b0..bm]
        # 对每个频点构建方程: [H*z^1 ... H*z^m | -z^0 ... -z^m] x = -H
        rows = n_freq * n_ch
        cols = order + (order + 1)
        J = np.zeros((rows, cols), dtype=np.complex128)
        y = np.zeros((rows, 1), dtype=np.complex128)

        row = 0
        for c in range(n_ch):
            Hc = frf[:, c]
            for i in range(n_freq):
                zi = z[i]
                # H*z^k (k=1..m)
                for k in range(1, order + 1):
                    J[row, k - 1] = Hc[i] * (zi ** k)
                # -z^k (k=0..m)
                for k in range(0, order + 1):
                    J[row, order + k] = -(zi ** k)
                y[row, 0] = -Hc[i]
                row += 1

        # 最小二乘解
        x, _, _, _ = linalg.lstsq(J, y)
        a = np.concatenate(([1.0], x[:order].flatten()))
        b = x[order:].flatten()

        # 极点提取: A(z) = 1 + a1 z^-1 + ... + am z^-m
        # 转换为多项式: z^m + a1 z^{m-1} + ... + am = 0
        poly = np.concatenate(([1.0], x[:order].flatten()))
        poles_z = np.roots(poly)
        poles_s = np.log(poles_z) / dt

        return poles_z, poles_s, a, b

    def sweep_orders(self, frf: np.ndarray, freq_axis: np.ndarray, orders=range(2, 65)):
        """
        稳态图数据: 扫描阶数并收集极点
        返回:
          - raw: list of dict {order, poles_z, poles_s}
          - points: list of {order, freq, damping}
        """
        raw = []
        points = []
        for order in orders:
            if order < 1 or order > self.max_order:
                continue
            poles_z, poles_s, _, _ = self.fit(frf, freq_axis, order)
            raw.append({"order": order, "poles_z": poles_z, "poles_s": poles_s})
            points.extend(poles_to_stability_points(poles_s, order))
        return {"raw": raw, "points": points}

    def sweep_and_cluster(self, frf: np.ndarray, freq_axis: np.ndarray, orders=range(2, 65), freq_tol=0.02, min_count=3):
        """
        扫描阶数 + 极点聚类筛选
        返回:
          - raw: 原始极点
          - points: 稳态图点集
          - stable: 聚类后的稳定模态
        """
        out = self.sweep_orders(frf, freq_axis, orders)
        stable = cluster_stability_points(out["points"], freq_tol=freq_tol, min_count=min_count)
        out["stable"] = stable
        return out


def poles_to_stability_points(poles_s, order):
    """
    极点转稳态图点集:
    freq = |mu| / 2π
    damping = -Re(mu) / |mu|
    """
    points = []
    for mu in poles_s:
        wn = np.abs(mu)
        if wn == 0:
            continue
        freq = wn / (2 * np.pi)
        damping = -np.real(mu) / wn
        points.append({"order": order, "freq": float(freq), "damping": float(damping)})
    return points


def merge_multi_ref_poles(poles_list, freq_tol=0.02):
    """
    多参考极点合并: 频率相近的极点合并
    """
    merged = []
    for poles in poles_list:
        for p in poles:
            freq = p["freq"]
            found = False
            for m in merged:
                if abs(m["freq"] - freq) / (m["freq"] + 1e-12) < freq_tol:
                    m["freq"] = (m["freq"] + freq) / 2.0
                    m["damping"] = (m["damping"] + p["damping"]) / 2.0
                    found = True
                    break
            if not found:
                merged.append({"freq": freq, "damping": p["damping"], "order": p.get("order")})
    return merged


def cluster_stability_points(points, freq_tol=0.02, min_count=3):
    """
    极点聚类 + 稳态筛选:
    - 频率相近聚类
    - 过滤出现次数 < min_count 的点
    """
    clusters = []
    for p in points:
        placed = False
        for c in clusters:
            if abs(c["freq"] - p["freq"]) / (c["freq"] + 1e-12) < freq_tol:
                c["freqs"].append(p["freq"])
                c["damps"].append(p["damping"])
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"freq": p["freq"], "freqs": [p["freq"]], "damps": [p["damping"]], "count": 1})

    stable = []
    for c in clusters:
        if c["count"] >= min_count:
            stable.append(
                {
                    "freq": float(np.mean(c["freqs"])),
                    "damping": float(np.mean(c["damps"])),
                    "count": c["count"],
                }
            )
    return stable
