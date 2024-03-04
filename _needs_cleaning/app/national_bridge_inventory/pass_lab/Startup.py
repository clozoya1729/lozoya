"""
Author: Christian Lozoya, 2017
"""


import sys
from PyQt5 import QtWidgets
from Interface.MainMenu import MainMenu


class Main:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        MainMenu(self.app, "PASS")


Main()
