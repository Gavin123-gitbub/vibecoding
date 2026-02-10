import numpy as np
from scipy import linalg, signal


class EFDD_Solver:
    """
    增强频域分解法 (Enhanced FDD)
    输入: data [channels, samples], fs
    输出: 模态频率、阻尼、模态向量等
    """

    def __init__(
        self,
        mac_threshold: float = 0.9,
        nperseg: int = 2048,
        peak_prominence: float = 0.05,
        peak_distance_hz: float = 0.5,
    ):
        self.mac_threshold = mac_threshold
        self.nperseg = nperseg
        self.peak_prominence = peak_prominence
        self.peak_distance_hz = peak_distance_hz

    @staticmethod
    def _mac(u: np.ndarray, v: np.ndarray) -> float:
        """
        MAC (Modal Assurance Criterion):
        MAC = |u^H v|^2 / (u^H u * v^H v)
        """
        num = np.abs(np.vdot(u, v)) ** 2
        den = (np.vdot(u, u).real * np.vdot(v, v).real) + 1e-12
        return float(num / den)

    def solve(self, data: np.ndarray, fs: float):
        if data.ndim != 2:
            raise ValueError("data must be [channels, samples]")
        ch, n = data.shape
        nperseg = min(self.nperseg, n)

        # 1) 计算 CSD 矩阵: 对每对通道求 Gxy(f)
        freqs = None
        G = None
        for i in range(ch):
            for j in range(ch):
                f, Pxy = signal.csd(
                    data[i],
                    data[j],
                    fs=fs,
                    nperseg=nperseg,
                    detrend="constant",
                    scaling="density",
                )
                if freqs is None:
                    freqs = f
                    G = np.zeros((ch, ch, len(freqs)), dtype=np.complex128)
                G[i, j, :] = Pxy

        # 2) 对每个频点做 SVD: G = U S U^H
        s1 = np.zeros(len(freqs))
        u1 = np.zeros((ch, len(freqs)), dtype=np.complex128)
        for k in range(len(freqs)):
            U, S, _ = linalg.svd(G[:, :, k], full_matrices=False)
            s1[k] = S[0].real
            u1[:, k] = U[:, 0]

        # 3) 峰值拾取: 使用 find_peaks
        df = freqs[1] - freqs[0]
        min_distance = max(1, int(self.peak_distance_hz / max(df, 1e-12)))
        peaks, _ = signal.find_peaks(
            s1,
            prominence=self.peak_prominence * np.max(s1),
            distance=min_distance,
        )

        results = []
        for pk in peaks:
            u_peak = u1[:, pk]
            # 4) MAC 滤波: 仅保留与峰值模态向量相似的频带
            mac_vals = np.array([self._mac(u_peak, u1[:, k]) for k in range(len(freqs))])
            mask = mac_vals >= self.mac_threshold

            # 筛选单一模态频带 (只保留 S1*mask)
            s_filtered = s1 * mask.astype(float)

            # 5) IFFT -> 时域自由衰减 (自相关近似)
            # 构造对称谱以 IFFT
            sdof_spec = s_filtered
            time_signal = np.fft.irfft(sdof_spec, n=nperseg)

            # 6) 阻尼识别: 对数衰减法
            # 使用 Hilbert 包络 |x_h(t)|，log 线性拟合
            analytic = signal.hilbert(time_signal)
            envelope = np.abs(analytic) + 1e-12
            t = np.arange(len(envelope)) / fs
            # 只取前半段以减少噪声影响
            m = max(10, len(t) // 2)
            log_env = np.log(envelope[:m])
            A = np.vstack([t[:m], np.ones(m)]).T
            slope, _ = np.linalg.lstsq(A, log_env, rcond=None)[0]

            # 阻尼比 xi = -slope / wn
            f_peak = freqs[pk]
            wn = 2 * np.pi * f_peak
            xi = max(0.0, -slope / (wn + 1e-12))

            # 细化频率: 取该模态频带内的最大能量频率
            if np.any(mask):
                f_refined = freqs[np.argmax(s_filtered)]
            else:
                f_refined = f_peak

            results.append(
                {
                    "f_peak": f_peak,
                    "f_refined": f_refined,
                    "damping": xi,
                    "mode_shape": u_peak,
                    "mac_band": mask,
                }
            )

        return results

    def solve_multi_ref(self, data: np.ndarray, fs: float, ref_indices):
        """
        多参考通道 EFDD: 对每个参考通道独立求解并合并结果
        """
        all_results = []
        for ref_idx in ref_indices:
            # 通过加权方式融合参考通道: 将数据重排使 ref_idx 优先
            data_re = np.vstack([data[ref_idx], np.delete(data, ref_idx, axis=0)])
            res = self.solve(data_re, fs)
            all_results.extend(res)
        return all_results


class SSI_COV_Solver:
    """
    协方差驱动随机子空间法 (SSI-COV)
    """

    def __init__(self, orders=range(2, 50), block_rows=20):
        self.orders = list(orders)
        self.block_rows = block_rows

    @staticmethod
    def _compute_covariances(data: np.ndarray, max_lag: int):
        """
        计算 R_i = E[y_{k+i} y_k^T]
        data: [channels, samples]
        返回列表 R[0..max_lag]
        """
        ch, n = data.shape
        R = []
        for lag in range(max_lag + 1):
            Y1 = data[:, lag:]
            Y0 = data[:, : n - lag]
            R.append((Y1 @ Y0.T) / (n - lag))
        return R

    @staticmethod
    def _build_toeplitz(R, i):
        """
        构建 Toeplitz 块矩阵 T_{1|i}
        T = [R1 R2 ... Ri
             R2 R3 ... R{i+1}
             ...
             Ri R{i+1} ... R{2i-1}]
        """
        ch = R[0].shape[0]
        T = np.zeros((ch * i, ch * i))
        for r in range(i):
            for c in range(i):
                T[r * ch : (r + 1) * ch, c * ch : (c + 1) * ch] = R[r + c + 1]
        return T

    def solve(self, data: np.ndarray, fs: float):
        if data.ndim != 2:
            raise ValueError("data must be [channels, samples]")
        ch, n = data.shape
        i = self.block_rows
        R = self._compute_covariances(data, max_lag=2 * i)
        T = self._build_toeplitz(R, i)

        # SVD: T = U S V^T
        U, S, Vt = linalg.svd(T, full_matrices=False)

        results = []
        dt = 1.0 / fs
        for order in self.orders:
            if order <= 0 or order > len(S):
                continue
            U1 = U[:, :order]
            S1 = np.diag(np.sqrt(S[:order]))

            # 扩展可观测矩阵 O_i = U1 * S1
            Oi = U1 @ S1
            # 分块: Oi(1..i-1) 和 Oi(2..i)
            O1 = Oi[:-ch, :]
            O2 = Oi[ch:, :]

            # 最小二乘求 A
            A = linalg.lstsq(O1, O2)[0]
            C = Oi[:ch, :]

            # 离散 -> 连续
            eigvals, eigvecs = linalg.eig(A)
            mu = np.log(eigvals) / dt
            wn = np.abs(mu)
            freq = wn / (2 * np.pi)
            damping = -np.real(mu) / (wn + 1e-12)
            mode_shapes = C @ eigvecs

            results.append(
                {
                    "order": order,
                    "freq": freq,
                    "damping": damping,
                    "mode_shape": mode_shapes,
                }
            )

        return results

    def solve_multi_ref(self, data: np.ndarray, fs: float, ref_indices):
        """
        多参考通道 SSI-COV: 对每个参考通道独立求解并合并结果
        """
        all_results = []
        for ref_idx in ref_indices:
            data_re = np.vstack([data[ref_idx], np.delete(data, ref_idx, axis=0)])
            all_results.extend(self.solve(data_re, fs))
        return all_results


class RDT_ERA_Solver:
    """
    随机减量 (RDT) + ERA 组合
    """

    def __init__(self, threshold=1.5, length=2048, order=10, block_rows=20):
        self.threshold = threshold
        self.length = length
        self.order = order
        self.block_rows = block_rows

    def _rdt(self, data: np.ndarray, ref_idx: int):
        ref = data[ref_idx]
        sigma = np.std(ref)
        thr = self.threshold * sigma
        crossings = np.where((ref[:-1] < thr) & (ref[1:] >= thr))[0]

        segments = []
        for idx in crossings:
            if idx + self.length <= data.shape[1]:
                segments.append(data[:, idx : idx + self.length])
        if not segments:
            raise RuntimeError("RDT: no valid triggers found")
        return np.mean(np.stack(segments, axis=0), axis=0)

    def _era(self, irf: np.ndarray, fs: float):
        ch, L = irf.shape
        m = self.block_rows
        m = min(m, L // 2 - 1)
        if m <= 1:
            raise ValueError("insufficient length for ERA")

        # 构建 Hankel 矩阵 H0, H1
        H0 = np.zeros((ch * m, m))
        H1 = np.zeros((ch * m, m))
        for i in range(m):
            for j in range(m):
                H0[i * ch : (i + 1) * ch, j] = irf[:, i + j]
                H1[i * ch : (i + 1) * ch, j] = irf[:, i + j + 1]

        U, S, Vt = linalg.svd(H0, full_matrices=False)
        r = min(self.order, len(S))
        U1 = U[:, :r]
        S1 = np.diag(np.sqrt(S[:r]))
        V1 = Vt[:r, :].T

        # Ahat = S^{-1/2} U^T H1 V S^{-1/2}
        S_inv = linalg.inv(S1)
        Ahat = S_inv @ (U1.T @ H1 @ V1) @ S_inv
        Chat = U1[:ch, :] @ S1

        eigvals, eigvecs = linalg.eig(Ahat)
        dt = 1.0 / fs
        mu = np.log(eigvals) / dt
        wn = np.abs(mu)
        freq = wn / (2 * np.pi)
        damping = -np.real(mu) / (wn + 1e-12)
        mode_shapes = Chat @ eigvecs
        return freq, damping, mode_shapes

    def solve(self, data: np.ndarray, fs: float, ref_idx: int = 0):
        irf = self._rdt(data, ref_idx)
        freq, damping, mode_shapes = self._era(irf, fs)
        return {
            "freq": freq,
            "damping": damping,
            "mode_shape": mode_shapes,
            "irf": irf,
        }


def stability_filter(results, freq_tol=0.02, damp_tol=0.05, mac_tol=0.9):
    """
    稳态图判据:
    - 频率相对变化 < freq_tol
    - 阻尼相对变化 < damp_tol
    - MAC > mac_tol
    """
    stable = []
    for i in range(1, len(results)):
        prev = results[i - 1]
        curr = results[i]
        freq_prev = np.atleast_1d(prev["freq"])
        freq_curr = np.atleast_1d(curr["freq"])
        damp_prev = np.atleast_1d(prev["damping"])
        damp_curr = np.atleast_1d(curr["damping"])
        mode_prev = prev["mode_shape"]
        mode_curr = curr["mode_shape"]

        for k in range(min(len(freq_prev), len(freq_curr))):
            f1, f2 = freq_prev[k], freq_curr[k]
            d1, d2 = damp_prev[k], damp_curr[k]
            if f1 == 0:
                continue
            if abs(f2 - f1) / abs(f1) > freq_tol:
                continue
            if abs(d2 - d1) / (abs(d1) + 1e-12) > damp_tol:
                continue
            # MAC 计算
            u = mode_prev[:, k]
            v = mode_curr[:, k]
            mac = np.abs(np.vdot(u, v)) ** 2 / ((np.vdot(u, u).real * np.vdot(v, v).real) + 1e-12)
            if mac < mac_tol:
                continue
            stable.append((f2, d2, mac))
    return stable


def run_all_solvers(data: np.ndarray, fs: float):
    """
    示例: 依次调用 EFDD / SSI-COV / RDT-ERA
    """
    efdd = EFDD_Solver()
    ssi = SSI_COV_Solver(orders=range(2, 20), block_rows=20)
    rdt = RDT_ERA_Solver(threshold=1.5, length=2048, order=10, block_rows=20)

    efdd_res = efdd.solve(data, fs)
    ssi_res = ssi.solve(data, fs)
    rdt_res = rdt.solve(data, fs, ref_idx=0)

    return {
        "efdd": efdd_res,
        "ssi": ssi_res,
        "rdt": rdt_res,
    }
