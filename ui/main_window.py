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
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("扫描式激光测振仪后处理软件")
        self.resize(1400, 900)

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self.setStatusBar(QStatusBar(self))

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("文件")
        view_menu = menu_bar.addMenu("视图")
        help_menu = menu_bar.addMenu("帮助")

        file_menu.addAction(QAction("新建项目", self))
        file_menu.addAction(QAction("打开项目", self))
        file_menu.addSeparator()
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

        placeholder = QLabel("显示区 (待接入渲染引擎)")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setObjectName("centerPlaceholder")
        placeholder.setMinimumHeight(400)

        layout.addWidget(title)
        layout.addWidget(placeholder, 1)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("rightPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("参数 / 属性面板")
        title.setObjectName("panelTitle")

        placeholder = QLabel("参数控件占位")
        placeholder.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        placeholder.setObjectName("rightPlaceholder")

        layout.addWidget(title)
        layout.addWidget(placeholder, 1)
        return panel
