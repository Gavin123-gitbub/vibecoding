from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem


class GeoEditor(QTableWidget):
    geometry_updated = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(0, 5, parent)
        self.setHorizontalHeaderLabels(["Node ID", "X (m)", "Y (m)", "Z (m)", "Description"])
        self.itemChanged.connect(self._on_item_changed)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.matches(QKeyEvent.StandardKey.Paste):
            self._paste_from_clipboard()
            return
        super().keyPressEvent(event)

    def _paste_from_clipboard(self) -> None:
        clipboard = self.clipboard()
        text = clipboard.text()
        if not text:
            return
        rows = [line for line in text.splitlines() if line.strip()]
        start_row = self.currentRow()
        start_col = self.currentColumn()
        if start_row < 0:
            start_row = self.rowCount()
        if start_col < 0:
            start_col = 0

        self.blockSignals(True)
        for r_idx, line in enumerate(rows):
            cols = line.split("\t")
            row = start_row + r_idx
            if row >= self.rowCount():
                self.insertRow(self.rowCount())
            for c_idx, value in enumerate(cols):
                col = start_col + c_idx
                if col >= self.columnCount():
                    continue
                item = QTableWidgetItem(value.strip())
                self.setItem(row, col, item)
        self.blockSignals(False)
        self._emit_geometry()

    def _on_item_changed(self, _item):
        self._emit_geometry()

    def _emit_geometry(self):
        nodes = []
        for row in range(self.rowCount()):
            node_id_item = self.item(row, 0)
            if not node_id_item or not node_id_item.text().strip():
                continue
            try:
                node_id = int(node_id_item.text())
                x = float(self.item(row, 1).text()) if self.item(row, 1) else 0.0
                y = float(self.item(row, 2).text()) if self.item(row, 2) else 0.0
                z = float(self.item(row, 3).text()) if self.item(row, 3) else 0.0
                desc = self.item(row, 4).text() if self.item(row, 4) else ""
            except ValueError:
                continue
            nodes.append({"id": node_id, "x": x, "y": y, "z": z, "desc": desc})
        self.geometry_updated.emit({"nodes": nodes, "lines": []})
