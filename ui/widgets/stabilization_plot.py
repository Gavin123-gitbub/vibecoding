import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QWidget, QSpinBox, QDoubleSpinBox


class StabilizationPlot(QDialog):
    """
    稳态图交互窗口
    - 背景灰色奇异值曲线
    - 前景极点 (绿=稳定, 红=不稳定)
    - 支持框选极点并标记物理模态
    """

    modes_selected = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("稳态图 (Stabilization Diagram)")
        self.resize(900, 600)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 参数输入区
        param_row = QHBoxLayout()
        self.order_min = QSpinBox()
        self.order_min.setRange(1, 200)
        self.order_min.setPrefix("Min Order: ")
        self.order_max = QSpinBox()
        self.order_max.setRange(1, 200)
        self.order_max.setPrefix("Max Order: ")
        self.fmin = QDoubleSpinBox()
        self.fmin.setRange(0.0, 1e6)
        self.fmin.setPrefix("Fmin: ")
        self.fmax = QDoubleSpinBox()
        self.fmax.setRange(0.0, 1e6)
        self.fmax.setPrefix("Fmax: ")
        param_row.addWidget(self.order_min)
        param_row.addWidget(self.order_max)
        param_row.addWidget(self.fmin)
        param_row.addWidget(self.fmax)
        layout.addLayout(param_row)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("#1E1E1E")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Order / Mode Index")
        layout.addWidget(self.plot)

        self.svd_curve = self.plot.plot(pen=pg.mkPen((150, 150, 150), width=1))
        self.stable_scatter = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(0, 255, 0, 180), size=8)
        self.unstable_scatter = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(255, 0, 0, 150), size=6)
        self.plot.addItem(self.stable_scatter)
        self.plot.addItem(self.unstable_scatter)

        # selection box
        self._selection = pg.RectROI([0, 0], [1, 1], pen=pg.mkPen("#00F0FF"))
        self._selection.hide()
        self.plot.addItem(self._selection)

        btn_layout = QHBoxLayout()
        self.select_btn = QPushButton("框选模式")
        self.confirm_btn = QPushButton("确认选择")
        btn_layout.addWidget(self.select_btn)
        btn_layout.addWidget(self.confirm_btn)
        layout.addLayout(btn_layout)

        self.select_btn.clicked.connect(self._toggle_select)
        self.confirm_btn.clicked.connect(self._confirm_selection)

        self._points = []  # list of dicts {freq, order, stable}

    def set_params(self, params: dict):
        self.order_min.setValue(params.get("order_min", 2))
        self.order_max.setValue(params.get("order_max", 50))
        self.fmin.setValue(params.get("fmin", 0.0))
        self.fmax.setValue(params.get("fmax", 1000.0))
        self._apply_filters()

    def set_svd_curve(self, freq_axis, s1):
        self.svd_curve.setData(freq_axis, s1)

    def set_poles(self, points, stable_mask=None):
        """
        points: list of dict {freq, order}
        stable_mask: list/array of bool
        """
        self._points = points
        freqs = np.array([p["freq"] for p in points])
        orders = np.array([p.get("order", 0) for p in points])
        if stable_mask is None:
            stable_mask = np.zeros(len(points), dtype=bool)
        stable_mask = np.asarray(stable_mask, dtype=bool)

        self._stable_mask = stable_mask
        self._apply_filters()

    def _apply_filters(self):
        if not self._points:
            return
        freqs = np.array([p["freq"] for p in self._points])
        orders = np.array([p.get("order", 0) for p in self._points])
        stable_mask = getattr(self, "_stable_mask", np.zeros(len(self._points), dtype=bool))

        fmin = self.fmin.value()
        fmax = self.fmax.value()
        mask = (freqs >= fmin) & (freqs <= fmax)

        self.stable_scatter.setData(freqs[mask & stable_mask], orders[mask & stable_mask])
        self.unstable_scatter.setData(freqs[mask & ~stable_mask], orders[mask & ~stable_mask])

    def _toggle_select(self):
        if self._selection.isVisible():
            self._selection.hide()
        else:
            self._selection.setPos([0, 0])
            self._selection.setSize([10, 10])
            self._selection.show()

    def _confirm_selection(self):
        if not self._selection.isVisible():
            return
        rect: QRectF = self._selection.parentBounds()
        x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
        selected = []
        for p in self._points:
            if x0 <= p["freq"] <= x1 and y0 <= p.get("order", 0) <= y1:
                selected.append(p)
        self.modes_selected.emit(selected)
