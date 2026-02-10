from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QSplitter,
    QLabel,
    QFrame,
    QToolBar,
    QStatusBar,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QInputDialog,
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt

import numpy as np

from core.data_manager import ModalDataSet, FRFEngine
from core.geometry_builder import GeometryBuilder
from core.uff_loader import UFFLoader
from core.controller import OMAController
from core.algo_ods import TimeODS, FreqODS
from ui.widgets.geo_editor import GeoEditor
from ui.widgets.channel_mapper import ChannelMapper
from ui.widgets.stabilization_plot import StabilizationPlot
from ui.widgets.visualizer import GeometryView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("扫描式激光测振仪后处理软件")
        self.resize(1400, 900)
        self.controller = OMAController()

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self.setStatusBar(QStatusBar(self))
        self._anim_scale = 1.0

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("文件")
        view_menu = menu_bar.addMenu("视图")
        help_menu = menu_bar.addMenu("帮助")

        open_uff = QAction("打开 UFF/UNV", self)
        open_uff.triggered.connect(self._open_uff)
        file_menu.addAction(open_uff)

        open_csv = QAction("打开 CSV/TXT", self)
        open_csv.triggered.connect(self._open_csv)
        file_menu.addAction(open_csv)

        file_menu.addSeparator()
        quick_mode = QAction("Quick Mode (EFDD)", self)
        quick_mode.triggered.connect(self._run_quick_mode)
        file_menu.addAction(quick_mode)

        pro_mode = QAction("Pro Mode (SSI/PolyMax)", self)
        pro_mode.triggered.connect(self._run_pro_mode)
        file_menu.addAction(pro_mode)

        export_report = QAction("导出报告", self)
        export_report.triggered.connect(self._export_report)
        file_menu.addAction(export_report)

        file_menu.addAction(QAction("导出", self))
        file_menu.addSeparator()
        file_menu.addAction(QAction("退出", self))

        view_menu.addAction(QAction("重置布局", self))
        help_menu.addAction(QAction("关于", self))

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addAction(QAction("导入", self))
        toolbar.addAction(QAction("处理", self))
        toolbar.addAction(QAction("导出", self))

    def _build_central(self) -> None:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal, root)

        left_panel = self._build_left_panel()
        center_panel = self._build_center_panel()
        right_panel = self._build_right_panel()

        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)

        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("leftPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("项目 / 数据树")
        title.setObjectName("panelTitle")

        tree = QTreeWidget()
        tree.setHeaderHidden(True)
        root = QTreeWidgetItem(["示例项目"])
        root.addChild(QTreeWidgetItem(["扫描数据"]))
        root.addChild(QTreeWidgetItem(["模态分析"]))
        tree.addTopLevelItem(root)
        tree.expandAll()

        layout.addWidget(title)
        layout.addWidget(tree)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("centerPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("3D 几何 / 波形显示区")
        title.setObjectName("panelTitle")

        self.geometry_view = GeometryView(panel)

        layout.addWidget(title)
        layout.addWidget(self.geometry_view, 1)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("rightPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("参数 / 属性面板")
        title.setObjectName("panelTitle")

        from PySide6.QtWidgets import QSlider

        anim_label = QLabel("动画幅值")
        anim_slider = QSlider(Qt.Horizontal)
        anim_slider.setMinimum(1)
        anim_slider.setMaximum(100)
        anim_slider.setValue(10)
        anim_slider.valueChanged.connect(lambda v: setattr(self, "_anim_scale", v / 10.0))

        placeholder = QLabel("参数控件占位")
        placeholder.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        placeholder.setObjectName("rightPlaceholder")

        layout.addWidget(title)
        layout.addWidget(anim_label)
        layout.addWidget(anim_slider)
        layout.addWidget(placeholder, 1)
        return panel

    def _open_uff(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开 UFF/UNV", "", "UFF/UNV (*.uff *.unv);;All (*.*)")
        if not path:
            return
        dataset = UFFLoader.load(path)
        self._update_geometry_view(dataset.geometry)
        self._last_geometry = dataset.geometry

    def _open_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开 CSV/TXT", "", "Data (*.csv *.txt);;All (*.*)")
        if not path:
            return
        time_data = self._load_time_data(path)
        if time_data is None:
            return

        geo = self._collect_geometry()
        if geo is None:
            return

        mappings = self._collect_channel_mapping(time_data.shape[1])
        if mappings is None:
            return

        ref_index = next((i for i, m in enumerate(mappings) if m["type"] == "Reference"), 0)
        fs, ok = QInputDialog.getDouble(self, "采样率", "输入采样率 (Hz):", 1000.0, 1.0, 1e6, 2)
        if not ok:
            return

        frf, freq, coh = FRFEngine.compute_frf(time_data, ref_index, fs)
        dataset = ModalDataSet(geometry=geo, frf_data=frf, freq_axis=freq, coherence=coh)
        self._update_geometry_view(dataset.geometry)
        self._last_time_data = time_data
        self._last_fs = fs
        self._last_frf = frf
        self._last_freq = freq
        self._last_geometry = geo

    def _load_time_data(self, path: str):
        try:
            return np.loadtxt(path, delimiter=",")
        except Exception:
            try:
                return np.loadtxt(path)
            except Exception:
                return None

    def _collect_geometry(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("节点坐标输入")
        layout = QVBoxLayout(dialog)
        editor = GeoEditor(dialog)
        layout.addWidget(editor)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        layout.addWidget(buttons)
        geometry = {"nodes": [], "lines": []}

        def on_update(geo):
            nonlocal geometry
            geometry = geo
            self._update_geometry_view(geo)

        editor.geometry_updated.connect(on_update)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        if dialog.exec() == QDialog.Accepted:
            return geometry
        return None

    def _collect_channel_mapping(self, n_channels: int):
        channels = [f"CH{i+1}" for i in range(n_channels)]
        dialog = QDialog(self)
        dialog.setWindowTitle("通道映射")
        layout = QVBoxLayout(dialog)
        mapper = ChannelMapper(channels, dialog)
        layout.addWidget(mapper)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        if dialog.exec() == QDialog.Accepted:
            return mapper.get_mappings()
        return None

    def _update_geometry_view(self, geometry: dict):
        nodes = geometry.get("nodes", [])
        self.geometry_view.set_points(nodes)

    def _run_quick_mode(self):
        if not hasattr(self, "_last_time_data"):
            return
        res = self.controller.quick_mode(self._last_time_data, self._last_fs, top_n=5)
        txt = "Quick Mode 结果:\n"
        for r in res:
            txt += f"f={r['f_peak']:.2f} Hz, damping={r['damping']:.4f}\n"
        # 简化展示：用几何视图标题提示
        self.setStatusTip(txt)
        self._last_modes = {"freq": res[0]["f_peak"], "damping": res[0]["damping"], "shape": None} if res else None

    def _run_pro_mode(self):
        if not hasattr(self, "_last_time_data"):
            return
        mode, ok = QInputDialog.getItem(
            self,
            "Pro Mode",
            "选择算法:",
            ["SSI", "PolyMax"],
            0,
            False,
        )
        if not ok:
            return
        if mode == "SSI":
            out = self.controller.pro_mode_ssi(self._last_time_data, self._last_fs)
        else:
            if not hasattr(self, "_last_frf"):
                return
            # PolyMax 多通道处理: 对每个通道独立求解并合并
            poles_list = []
            for ch in range(self._last_frf.shape[1]):
                poly_out = self.controller.pro_mode_polymax(self._last_frf[:, ch], self._last_freq)
                poles_list.append(poly_out["points"])
            out = {
                "points": [],
                "stable": [],
            }
            # 合并多参考极点
            from core.algo.polymax import merge_multi_ref_poles, cluster_stability_points
            merged = merge_multi_ref_poles([poles for poles in poles_list])
            stable = cluster_stability_points(merged, freq_tol=0.02, min_count=3)
            out["points"] = merged
            out["stable"] = stable
        dialog = StabilizationPlot(self)

        # 预填自适应参数
        params = self.controller.auto_params(self._last_time_data, self._last_fs)
        dialog.set_params(params)
        # 暂时用占位曲线显示奇异值 (SVD of data covariance)
        s = np.linalg.svd(self._last_time_data, compute_uv=False)
        freq_axis = np.linspace(0, self._last_fs / 2, len(s))
        dialog.set_svd_curve(freq_axis, s)

        # 稳态极点显示
        points = out["points"]
        stable_mask = np.zeros(len(points), dtype=bool)
        stable_pts = set([(p[0], p[1]) for p in out["stable"]])
        for i, p in enumerate(points):
            if (p["freq"], p["damp"]) in stable_pts:
                stable_mask[i] = True
        dialog.set_poles(points, stable_mask)

        def on_modes_selected(selected):
            if not selected:
                return
            # 取第一个模态用于动画
            freq = selected[0]["freq"]
            damping = selected[0].get("damping", 0.0)
            # 简化: 使用频域ODS 提取
            shape = FreqODS.extract(self._last_time_data, self._last_fs, freq)
            if hasattr(self, "_last_geometry"):
                self.geometry_view.animate_mode_shape(shape.real, freq, self._last_geometry["nodes"], scale=self._anim_scale)
            # 保存结果用于导出
            self._last_modes = {"freq": freq, "damping": damping, "shape": shape}

        dialog.modes_selected.connect(on_modes_selected)
        dialog.exec()

    def _export_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出报告", "", "CSV (*.csv);;UFF (*.uff)")
        if not path:
            return
        rows = []
        if hasattr(self, "_last_modes"):
            rows.append([self._last_modes["freq"], self._last_modes.get("damping", 0.0)])
        elif hasattr(self, "_last_time_data"):
            res = self.controller.quick_mode(self._last_time_data, self._last_fs, top_n=5)
            for r in res:
                rows.append([r["f_peak"], r["damping"]])
        if path.endswith(".csv"):
            import csv
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Frequency", "Damping"])
                writer.writerows(rows)
        elif path.endswith(".uff"):
            try:
                from core.uff_loader import UFFWriter
                freqs = [r[0] for r in rows]
                damps = [r[1] for r in rows]
                geom = getattr(self, "_last_geometry", {"nodes": [], "lines": []})
                UFFWriter.export_modal(path, geom, freqs, damps, np.array([]))
            except Exception:
                pass
