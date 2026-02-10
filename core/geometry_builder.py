from typing import Dict, List


class GeometryBuilder:
    def __init__(self):
        self.nodes: Dict[int, Dict] = {}
        self.lines: List[tuple] = []

    def add_node(self, node_id: int, x: float, y: float, z: float, description: str = "") -> None:
        self.nodes[int(node_id)] = {"id": int(node_id), "x": float(x), "y": float(y), "z": float(z), "desc": description}

    def add_line(self, node1: int, node2: int) -> None:
        self.lines.append((int(node1), int(node2)))

    def import_from_dataframe(self, df) -> None:
        required = ["Node ID", "X (m)", "Y (m)", "Z (m)"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"missing column: {col}")
        for _, row in df.iterrows():
            desc = row["Description"] if "Description" in df.columns else ""
            self.add_node(row["Node ID"], row["X (m)"], row["Y (m)"], row["Z (m)"], desc)

    def export_geometry(self):
        nodes = list(self.nodes.values())
        return {"nodes": nodes, "lines": self.lines}
