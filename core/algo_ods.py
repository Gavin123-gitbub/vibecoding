import numpy as np


class TimeODS:
    """
    ODS (Operating Deflection Shapes) - 时域
    直接取某一时刻的数据作为形状向量
    """

    @staticmethod
    def extract(data: np.ndarray, t_index: int):
        """
        data: [channels, samples]
        t_index: 采样点索引
        返回: shape_vector [channels]
        """
        if data.ndim != 2:
            raise ValueError("data must be [channels, samples]")
        return data[:, t_index]


class FreqODS:
    """
    ODS - 频域
    提取指定频率的幅值/相位向量
    """

    @staticmethod
    def extract(data: np.ndarray, fs: float, target_freq: float):
        """
        data: [channels, samples]
        返回: shape_vector (复数) [channels]
        """
        if data.ndim != 2:
            raise ValueError("data must be [channels, samples]")
        n = data.shape[1]
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        idx = int(np.argmin(np.abs(freqs - target_freq)))
        spectrum = np.fft.rfft(data, axis=1)
        return spectrum[:, idx]
