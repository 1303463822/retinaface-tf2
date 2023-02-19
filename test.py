import os
import sys

from PyQt5.QtWidgets import QApplication
from QCandyUi import CandyWindow
from GUI import MyWindow

if __name__ == "__main__":
    Window = QApplication(sys.argv)

    w = MyWindow()
    w.ui = CandyWindow.createWindow(w.ui, "blueDeep")
    w.ui.show()

    Window.exec()
