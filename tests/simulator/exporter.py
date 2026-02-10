import csv
from pathlib import Path

import numpy as np


def save_to_csv(folder, data):
    """
    data dict expected:
      - time: [N_samples]
      - force: [N_inputs, N_samples] or None
      - response: [N_nodes, N_samples]
      - geometry: [N_nodes, 3]
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    time = np.asarray(data["time"])
    response = np.asarray(data["response"])
    force = data.get("force", None)
    if force is not None:
        force = np.asarray(force)
        ref = force[0]
    else:
        ref = np.zeros_like(time)

    # timeseries CSV
    csv_path = folder / "test_ema.csv"
    header = ["Time", "Reference"]
    header += [f"Node_{i+1}" for i in range(response.shape[0])]
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(time)):
            row = [time[i], ref[i]] + list(response[:, i])
            writer.writerow(row)

    # geometry CSV
    geom_path = folder / "geometry.csv"
    geom = np.asarray(data["geometry"])
    with geom_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Node ID", "X (m)", "Y (m)", "Z (m)"])
        for i, (x, y, z) in enumerate(geom, start=1):
            writer.writerow([i, x, y, z])

    return csv_path, geom_path


def save_to_uff(filename, data):
    """
    使用 pyuff 导出:
      - Dataset 15: nodes
      - Dataset 82: lines (simple chain)
      - Dataset 58: time response (func_type=1)
    """
    import pyuff

    filename = Path(filename)
    time = np.asarray(data["time"])
    response = np.asarray(data["response"])
    force = data.get("force", None)
    geom = np.asarray(data["geometry"])

    uff = pyuff.UFF()

    # Nodes
    for i, (x, y, z) in enumerate(geom, start=1):
        uff.write_sets({"type": 15, "node": i, "x": x, "y": y, "z": z, "desc": f"Node_{i}"})

    # Lines (simple chain)
    for i in range(1, len(geom)):
        uff.write_sets({"type": 82, "node1": i, "node2": i + 1})

    # Function data (Dataset 58)
    dt = time[1] - time[0]
    for i in range(response.shape[0]):
        uff.write_sets(
            {
                "type": 58,
                "func_type": 1,  # time domain response
                "abscissa_spacing": 1,
                "num_pts": len(time),
                "abscissa": time,
                "ord_data": response[i],
                "rsp_node": i + 1,
                "ref_node": 1,
                "data": response[i],
                "x": time,
            }
        )

    # Reference force as a separate dataset if available
    if force is not None:
        ref = np.asarray(force)[0]
        uff.write_sets(
            {
                "type": 58,
                "func_type": 1,
                "abscissa_spacing": 1,
                "num_pts": len(time),
                "abscissa": time,
                "ord_data": ref,
                "rsp_node": 1,
                "ref_node": 1,
                "data": ref,
                "x": time,
            }
        )

    uff.write(str(filename))
    return filename
