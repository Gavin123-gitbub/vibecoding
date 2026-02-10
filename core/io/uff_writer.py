from datetime import datetime
from typing import List

import math


class UFFWriter:
    """
    纯 Python UFF/UNV 写入器（简化版）
    - Dataset 15: 节点
    - Dataset 82: 连线
    - Dataset 58: 函数数据（时域/频域）
    """

    def __init__(self, fh):
        self.fh = fh

    def write_header(self, dataset_type: int):
        self.fh.write(f"{-1:6d}\n")
        self.fh.write(f"{dataset_type:6d}\n")

    def write_footer(self):
        self.fh.write(f"{-1:6d}\n")

    def write_nodes(self, nodes):
        """
        nodes: list of dict {id,x,y,z,color(optional)}
        """
        self.write_header(15)
        for node in nodes:
            nid = int(node["id"])
            color = int(node.get("color", 0))
            x = float(node["x"])
            y = float(node["y"])
            z = float(node["z"])
            # 每个节点 4 行，简化为:
            # line1: node_id, color
            # line2: x y z
            # line3: blank
            # line4: blank
            self.fh.write(f"{nid:10d}{color:10d}\n")
            self.fh.write(f"{x:20.11E}{y:20.11E}{z:20.11E}\n")
            self.fh.write(f"{'':80s}\n")
            self.fh.write(f"{'':80s}\n")
        self.write_footer()

    def write_lines(self, lines):
        """
        lines: list of (node1, node2)
        """
        self.write_header(82)
        # 简化单 trace
        trace_num = 1
        nodes = []
        for n1, n2 in lines:
            nodes.extend([int(n1), int(n2)])
        if not nodes:
            nodes = []
        n_nodes = len(nodes)
        self.fh.write(f"{trace_num:10d}{n_nodes:10d}{0:10d}\n")
        self.fh.write(f"{'TRACE':80s}\n")
        for i in range(0, n_nodes, 8):
            chunk = nodes[i : i + 8]
            self.fh.write("".join([f"{n:10d}" for n in chunk]) + "\n")
        self.write_footer()

    def write_data(self, func_type, ref_node, resp_node, data_array, x_axis_values):
        """
        Dataset 58 (Function Data)
        func_type: 1 time, 4 frequency
        data_array: real or complex
        x_axis_values: x axis
        """
        data_array = list(data_array)
        x_axis_values = list(x_axis_values)
        is_complex = any(isinstance(v, complex) for v in data_array)

        self.write_header(58)
        # Line1-5: 描述信息
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.fh.write(f"{'UFF Writer':<80s}\n")
        self.fh.write(f"{'Generated':<80s}\n")
        self.fh.write(f"{now:<80s}\n")
        self.fh.write(f"{'':80s}\n")
        self.fh.write(f"{'':80s}\n")

        # Line6: Function Type
        self.fh.write(f"{func_type:5d}{0:10d}{0:5d}{0:10d} {'RESP':>10s}{resp_node:10d}{3:4d} {'REF':>10s}{ref_node:10d}{3:4d}\n")

        # Line7: data type / points
        ord_data_type = 5 if is_complex else 2
        num_pts = len(data_array)
        self.fh.write(f"{ord_data_type:10d}{num_pts:10d}{1:10d}{x_axis_values[0]:13.5E}{(x_axis_values[1]-x_axis_values[0]) if num_pts>1 else 0.0:13.5E}{0.0:13.5E}\n")

        # Line8-12: axis info
        self.fh.write(f"{0:10d}{0:5d}{0:5d}{0:5d} {'X':<20s} {'':<20s}\n")
        self.fh.write(f"{0:10d}{0:5d}{0:5d}{0:5d} {'Y':<20s} {'':<20s}\n")
        self.fh.write(f"{0:10d}{0:5d}{0:5d}{0:5d} {'':<20s} {'':<20s}\n")
        self.fh.write(f"{0:10d}{0:5d}{0:5d}{0:5d} {'':<20s} {'':<20s}\n")

        # Line13: padding
        self.fh.write(f"{'':80s}\n")

        # Data section
        if not is_complex:
            for i in range(0, num_pts, 6):
                chunk = data_array[i : i + 6]
                self.fh.write("".join([f"{v:13.5E}" for v in chunk]) + "\n")
        else:
            # 每行 3 对实/虚
            for i in range(0, num_pts, 3):
                chunk = data_array[i : i + 3]
                line = ""
                for v in chunk:
                    line += f"{v.real:13.5E}{v.imag:13.5E}"
                self.fh.write(line + "\n")

        self.write_footer()

    @staticmethod
    def export_simulation(filepath, geometry_data, time_data, fs):
        """
        geometry_data: {nodes, lines}
        time_data: {time, reference, response}
        """
        with open(filepath, "w") as f:
            writer = UFFWriter(f)
            writer.write_nodes(geometry_data["nodes"])
            writer.write_lines(geometry_data.get("lines", []))

            time = time_data["time"]
            ref = time_data["reference"]
            resp = time_data["response"]

            # reference
            writer.write_data(1, ref_node=1, resp_node=1, data_array=ref, x_axis_values=time)
            # responses
            for i, r in enumerate(resp, start=1):
                writer.write_data(1, ref_node=1, resp_node=i, data_array=r, x_axis_values=time)
