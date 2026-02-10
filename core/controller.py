from typing import Dict, List, Optional

import numpy as np

from core.algo_oma import EFDD_Solver, SSI_COV_Solver, build_stability_plot_data, stability_filter
from core.algo.polymax import PolyMaxLSCF


class OMAController:
    """
    统一控制入口:
    - Quick Mode: EFDD, 取前5峰值
    - Pro Mode: SSI / PolyMax, 输出稳态图数据
    """

    def __init__(self):
        self.efdd = EFDD_Solver()
        self.ssi = SSI_COV_Solver(orders=range(2, 50), block_rows=20)
        self.poly = PolyMaxLSCF(max_order=64)

    def quick_mode(self, data: np.ndarray, fs: float, top_n: int = 5):
        res = self.efdd.solve(data, fs)
        res = sorted(res, key=lambda r: r["f_peak"], reverse=False)[:top_n]
        return res

    def pro_mode_ssi(self, data: np.ndarray, fs: float):
        res = self.ssi.solve(data, fs)
        points = build_stability_plot_data(res)
        stable = stability_filter(res)
        return {"raw": res, "points": points, "stable": stable}

    def pro_mode_polymax(self, frf: np.ndarray, freq_axis: np.ndarray):
        out = self.poly.sweep_and_cluster(frf, freq_axis, orders=range(2, 65))
        return out

    def auto_params(self, data: np.ndarray, fs: float):
        """
        自适应参数估计:
        - 频率范围: [0, fs/2]
        - 阶数建议: 2 ~ min(64, channels*10)
        """
        ch = data.shape[0]
        return {"fmin": 0.0, "fmax": fs / 2.0, "order_min": 2, "order_max": min(64, ch * 10)}
