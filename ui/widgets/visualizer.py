import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget


class GeometryView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((0, 0, 0))
        self.view.setCameraPosition(distance=5.0)
        layout.addWidget(self.view)

        grid = gl.GLGridItem()
        grid.setSize(10, 10)
        grid.setSpacing(0.5, 0.5)
        grid.translate(0, 0, -0.001)
        self.view.addItem(grid)

        self.scatter = gl.GLScatterPlotItem()
        self.view.addItem(self.scatter)

        self.lines = gl.GLLinePlotItem()
        self.view.addItem(self.lines)

    def set_points(self, points):
        """
        points: list of dicts or tuples
          dict: {x,y,z,type}, type in {"Reference","Response"}
          tuple: (x,y,z,is_ref)
        """
        if not points:
            self.scatter.setData(pos=np.zeros((0, 3)))
            self.lines.setData(pos=np.zeros((0, 3)))
            return

        pos = []
        colors = []
        for p in points:
            if isinstance(p, dict):
                x, y, z = p["x"], p["y"], p["z"]
                is_ref = str(p.get("type", "")).lower() == "reference"
            else:
                x, y, z, is_ref = p
            pos.append([x, y, z])
            colors.append([1.0, 0.0, 0.33, 1.0] if is_ref else [1.0, 1.0, 1.0, 1.0])

        pos = np.asarray(pos, dtype=float)
        colors = np.asarray(colors, dtype=float)
        self.scatter.setData(pos=pos, color=colors, size=6.0, pxMode=False)

        if len(pos) >= 2:
            self.lines.setData(pos=pos, color=(0.0, 0.94, 1.0, 0.6), width=1.0, mode="line_strip")
        else:
            self.lines.setData(pos=pos, color=(0.0, 0.94, 1.0, 0.6), width=1.0, mode="line_strip")

    def animate_mode_shape(self, shape_vector, freq_hz, base_points, scale=1.0):
        """
        使用正弦规律对 Z 轴进行动画更新
        shape_vector: [n_nodes] or [n_nodes,]
        base_points: list of dicts with x,y,z
        """
        if not base_points:
            return
        shape = np.asarray(shape_vector).flatten()
        # 归一化幅值，避免过大导致爆炸
        if np.max(np.abs(shape)) > 0:
            shape = shape / np.max(np.abs(shape))
        shape = shape * scale
        if len(shape) != len(base_points):
            return

        from PySide6.QtCore import QTimer

        if hasattr(self, "_timer") and self._timer:
            self._timer.stop()

        self._t = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(30)

        def update():
            self._t += 0.03
            pos = []
            colors = []
            for i, p in enumerate(base_points):
                z = p["z"] + shape[i] * np.sin(2 * np.pi * freq_hz * self._t)
                pos.append([p["x"], p["y"], z])
                colors.append([1.0, 1.0, 1.0, 1.0])
            pos = np.asarray(pos, dtype=float)
            self.scatter.setData(pos=pos, color=np.asarray(colors), size=6.0, pxMode=False)
            self.lines.setData(pos=pos, color=(0.0, 0.94, 1.0, 0.6), width=1.0, mode="line_strip")

        self._timer.timeout.connect(update)
        self._timer.start()


class SignalView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("k")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Amplitude")
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.getPlotItem().setMenuEnabled(False)
        layout.addWidget(self.plot)

        self.curve = self.plot.plot(pen=pg.mkPen("#00F0FF", width=2))

    def set_signal(self, t, y):
        if t is None or y is None:
            self.curve.setData([], [])
            return
        self.curve.setData(t, y)
