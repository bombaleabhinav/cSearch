import sys
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction, QStyle
from PyQt5.QtCore import QObject, pyqtSignal
from pynput import keyboard
from ui import LauncherWindow, SettingsDialog

class HotkeyListener(QObject):
    triggered = pyqtSignal()
    
    def __init__(self, combo="<ctrl>+<space>"):
        super().__init__()
        self.combo = combo
        self.active = True
        self.listener = keyboard.GlobalHotKeys({self.combo: self.on_press})
        self.listener.start()

    def on_press(self):
        if self.active:
            self.triggered.emit()

    def set_enabled(self, enabled):
        self.active = enabled

class SpotlightApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        
        self.launcher = LauncherWindow()
        self.settings = SettingsDialog(hotkey_enabled=True)
        
        self.hotkey = HotkeyListener()
        self.hotkey.triggered.connect(self.toggle_launcher)
        
        self.launcher.settings_requested.connect(self.show_settings)
        self.settings.hotkey_check.toggled.connect(self.hotkey.set_enabled)
        
        self.setup_tray()

    def setup_tray(self):
        self.tray = QSystemTrayIcon(self.app.style().standardIcon(QStyle.SP_ComputerIcon))
        self.tray.setToolTip("Spotlight Search")
        
        menu = QMenu()
        act_open = QAction("Open Search", menu)
        act_open.triggered.connect(self.show_launcher)
        
        act_settings = QAction("Settings", menu)
        act_settings.triggered.connect(self.show_settings)
        
        act_quit = QAction("Quit", menu)
        act_quit.triggered.connect(sys.exit)
        
        menu.addActions([act_open, act_settings])
        menu.addSeparator()
        menu.addAction(act_quit)
        
        self.tray.setContextMenu(menu)
        self.tray.show()

    def toggle_launcher(self):
        if self.launcher.isVisible():
            self.launcher.hide()
        else:
            self.show_launcher()

    def show_launcher(self):
        # Center every time to handle screen changes
        screen = self.app.primaryScreen().geometry()
        self.launcher.move((screen.width() - self.launcher.width()) // 2, 
                          (screen.height() - self.launcher.height()) // 2 - 100)
        self.launcher.show()
        self.launcher.activateWindow()
        self.launcher.input.setFocus()

    def show_settings(self):
        self.settings.show()
        self.settings.activateWindow()

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    SpotlightApp().run()
