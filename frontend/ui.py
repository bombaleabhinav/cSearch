import os
import subprocess
from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout,
    QLineEdit, QListWidget, QListWidgetItem, 
    QGraphicsDropShadowEffect, QLabel, QFrame, 
    QPushButton, QProgressBar, QDialog, QFileDialog, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt5.QtGui import QColor
import backend

class WorkerThread(QThread):
    finished = pyqtSignal(object)
    
    def __init__(self, target_func, *args, **kwargs):
        super().__init__()
        self.target_func = target_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        result = self.target_func(*self.args, **self.kwargs)
        self.finished.emit(result)

class SettingsDialog(QDialog):
    ingest_started = pyqtSignal()
    ingest_finished = pyqtSignal(bool, str)

    def __init__(self, parent=None, hotkey_enabled=True):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(400, 250)
        self.hotkey_enabled = hotkey_enabled
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.hotkey_check = QCheckBox("Enable Global Hotkey (Ctrl+Space)")
        self.hotkey_check.setChecked(self.hotkey_enabled)
        layout.addWidget(self.hotkey_check)
        
        layout.addSpacing(10)
        layout.addWidget(QLabel("Index Folder Path:"))
        
        path_box = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("C:\\path\\to\\folder")
        path_box.addWidget(self.path_edit)
        
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse)
        path_box.addWidget(btn_browse)
        layout.addLayout(path_box)
        
        self.btn_ingest = QPushButton("Start Ingestion")
        self.btn_ingest.clicked.connect(self.start_ingest)
        layout.addWidget(self.btn_ingest)
        
        self.progress = QProgressBar()
        self.progress.setRange(0, 0) # Indeterminate
        self.progress.hide()
        layout.addWidget(self.progress)
        
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        layout.addStretch()

    def browse(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.path_edit.setText(dir_path)

    def start_ingest(self):
        path = self.path_edit.text().strip()
        if not path: return
        
        self.progress.show()
        self.btn_ingest.setEnabled(False)
        self.status_label.setText("Ingesting files...")
        
        self.thread = WorkerThread(backend.ingest_directory, path)
        self.thread.finished.connect(self.on_ingest_done)
        self.thread.start()

    def on_ingest_done(self, result):
        success, msg = result
        self.progress.hide()
        self.btn_ingest.setEnabled(True)
        self.status_label.setText("Done: " + msg)
        self.ingest_finished.emit(success, msg)

class LauncherWindow(QMainWindow):
    settings_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(650, 450)
        
        # Debounce timer
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.search)
        
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        self.container = QFrame(central)
        self.container.setObjectName("Container")
        self.container.setGeometry(10, 10, 630, 430)
        
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(15, 15, 15, 15)
        
        header = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Search files...")
        self.input.textChanged.connect(lambda: self.timer.start(200))
        header.addWidget(self.input)
        
        self.loader = QProgressBar()
        self.loader.setFixedSize(30, 5)
        self.loader.setRange(0, 0)
        self.loader.hide()
        header.addWidget(self.loader)
        
        btn_set = QPushButton("⚙")
        btn_set.setFixedSize(30, 30)
        btn_set.clicked.connect(self.settings_requested.emit)
        header.addWidget(btn_set)
        layout.addLayout(header)
        
        self.results = QListWidget()
        self.results.itemActivated.connect(self.open_file)
        layout.addWidget(self.results)
        
        self.container.setStyleSheet("""
            QFrame#Container {
                background-color: rgba(30, 30, 30, 160); /* frosted glass */
                border-radius: 14px;

                /* soft glass border */
                border: 1px solid rgba(255, 255, 255, 50);
            }

            QLineEdit {
                background-color: rgba(255, 255, 255, 20);
                border: 1px solid rgba(255, 255, 255, 40);
                border-radius: 10px;

                color: white;
                font-size: 20px;
                padding: 8px 12px;
            }

            QLineEdit:focus {
                background-color: rgba(255, 255, 255, 30);
                border: 1px solid rgba(0, 122, 204, 180);
            }

            QListWidget {
                background: transparent;
                border: none;
                color: #ddd;
            }

            QListWidget::item {
                padding: 12px;
                border-radius: 8px;
            }

            QListWidget::item:hover {
                background-color: rgba(255, 255, 255, 25);
            }

            QListWidget::item:selected {
                background-color: rgba(0, 122, 204, 150);
                color: white;
            }
            """)


        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 5)
        self.container.setGraphicsEffect(shadow)

    def search(self):
        query = self.input.text().strip()
        if not query:
            self.results.clear()
            return
            
        self.loader.show()
        self.thread = WorkerThread(backend.search_files, query)
        self.thread.finished.connect(self.on_search_results)
        self.thread.start()

    def on_search_results(self, files):
        self.loader.hide()
        self.results.clear()
        for f in files:
            # Safely extract path from dictionary or use the value itself
            if isinstance(f, dict):
                path = f.get('file_path') or f.get('path') or f.get('filepath')
            else:
                path = f
            
            # Ensure path is a string before calling os.path operations
            if not isinstance(path, str):
                continue
                
            item = QListWidgetItem(f"{os.path.basename(path)}\n{path}")
            item.setData(Qt.UserRole, path)
            self.results.addItem(item)
        if self.results.count() > 0:
            self.results.setCurrentRow(0)

    def open_file(self, item):
        path = item.data(Qt.UserRole)
        if path:
            subprocess.Popen(['explorer.exe', '/select,', os.path.normpath(path)])
            self.hide()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.hide()
        elif event.key() == (Qt.Key_Return or Qt.Key_Enter):
            if self.results.currentItem():
                self.open_file(self.results.currentItem())
        elif event.key() == Qt.Key_Down:
            self.results.setCurrentRow(self.results.currentRow() + 1)
        elif event.key() == Qt.Key_Up:
            self.results.setCurrentRow(self.results.currentRow() - 1)
