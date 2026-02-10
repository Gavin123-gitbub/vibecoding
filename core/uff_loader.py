from typing import Dict, List

import numpy as np

from core.data_manager import ModalDataSet


class UFFLoader:
    @staticmethod
    def load(path: str) -> ModalDataSet:
        try:
            import pyuff
        except ImportError as exc:
            raise RuntimeError("pyuff is required to load UFF/UNV files") from exc

        uff_file = pyuff.UFF(path)
        nodes = []
        lines = []
        frf_data = None
        freq_axis = None
        coherence = None

        for dataset in uff_file.read_sets():
            dtype = dataset.get("type", None)
            if dtype == 15:
                nodes.append(
                    {
                        "id": int(dataset["node"]),
                        "x": float(dataset["x"]),
                        "y": float(dataset["y"]),
                        "z": float(dataset["z"]),
                        "desc": dataset.get("desc", ""),
                    }
                )
            elif dtype == 82:
                lines.append((int(dataset["node1"]), int(dataset["node2"])))
            elif dtype == 58:
                # Data: frequency response typically
                if frf_data is None:
                    frf_data = []
                data = np.asarray(dataset["data"], dtype=float)
                frf_data.append(data)
                if freq_axis is None and "x" in dataset:
                    freq_axis = np.asarray(dataset["x"], dtype=float)

        if frf_data is not None:
            frf_data = np.stack(frf_data, axis=1)

        geometry = {"nodes": nodes, "lines": lines}
        return ModalDataSet(geometry=geometry, frf_data=frf_data, freq_axis=freq_axis, coherence=coherence)
