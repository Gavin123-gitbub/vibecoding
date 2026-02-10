from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ModalDataSet:
    geometry: Dict[str, List]
    frf_data: Optional[np.ndarray]
    freq_axis: Optional[np.ndarray]
    coherence: Optional[np.ndarray]


class FRFEngine:
    @staticmethod
    def compute_frf(time_data: np.ndarray, ref_channel_index: int, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        H1 estimator: FRF = Gxy / Gxx
        Gxy = X * conj(Y)
        Gxx = X * conj(X)
        Coherence = |Gxy|^2 / (Gxx * Gyy)
        """
        if time_data.ndim != 2:
            raise ValueError("time_data must be 2D array [n_samples, n_channels]")
        n_samples, n_channels = time_data.shape
        if not (0 <= ref_channel_index < n_channels):
            raise ValueError("ref_channel_index out of range")

        x = time_data[:, ref_channel_index]
        X = np.fft.rfft(x)
        Y_all = np.fft.rfft(time_data, axis=0)

        Gxx = X * np.conj(X)
        Gxy = X[:, None] * np.conj(Y_all)
        Gyy = Y_all * np.conj(Y_all)

        eps = 1e-12
        frf = Gxy / (Gxx[:, None] + eps)
        coherence = (np.abs(Gxy) ** 2) / ((Gxx[:, None] * Gyy) + eps)
        freq_axis = np.fft.rfftfreq(n_samples, d=1.0 / fs)

        return frf, freq_axis, coherence
