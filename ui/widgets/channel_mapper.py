from PySide6.QtWidgets import QComboBox, QLineEdit, QTableWidget, QTableWidgetItem


class ChannelMapper(QTableWidget):
    def __init__(self, channels, parent=None):
        super().__init__(len(channels), 3, parent)
        self.setHorizontalHeaderLabels(["Channel", "Type", "Node ID"])
        for row, name in enumerate(channels):
            self.setItem(row, 0, QTableWidgetItem(str(name)))
            combo = QComboBox()
            combo.addItems(["Reference", "Response"])
            self.setCellWidget(row, 1, combo)
            self.setCellWidget(row, 2, QLineEdit())

    def get_mappings(self):
        mappings = []
        for row in range(self.rowCount()):
            chan = self.item(row, 0).text()
            combo = self.cellWidget(row, 1)
            node_edit = self.cellWidget(row, 2)
            node_id = node_edit.text().strip()
            mappings.append(
                {
                    "channel": chan,
                    "type": combo.currentText(),
                    "node_id": int(node_id) if node_id.isdigit() else None,
                }
            )
        return mappings
