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


class UFFWriter:
    @staticmethod
    def export_modal(path: str, geometry: dict, freqs: list, dampings: list, mode_shapes: np.ndarray):
        """
        简化导出:
        - Dataset 15: 节点
        - Dataset 82: 线
        - Dataset 58: 模态参数 (示意)
        """
        import pyuff

        uff = pyuff.UFF()
        # Nodes (Dataset 15)
        for node in geometry.get("nodes", []):
            uff.write_sets(
                {
                    "type": 15,
                    "node": node["id"],
                    "x": node["x"],
                    "y": node["y"],
                    "z": node["z"],
                    "desc": node.get("desc", ""),
                }
            )
        # Lines (Dataset 82)
        for line in geometry.get("lines", []):
            uff.write_sets({"type": 82, "node1": line[0], "node2": line[1]})
        # Modal params (Dataset 58) - 仅示例
        for i, f in enumerate(freqs):
            d = dampings[i] if i < len(dampings) else 0.0
            uff.write_sets({"type": 58, "data": [f, d]})

        uff.write(path)
