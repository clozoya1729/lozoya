"""
Author: Christian Lozoya, 2017
"""

from Interface.SubMenu import SubMenu
from Utilities.DataHandler import *


class ReportMenu(SubMenu):
    def apply(self):
        self.update_preferences()
        self.top.destroy()

    def load_preferences(self):
        with open(RSP, "r") as reportSettings:
            for i, line in enumerate(reportSettings):
                if type(self.settings[i]) == bool:
                    self.settings[i] = True if line.strip() == 'True' else False
                else:
                    self.settings[2].set(True if line.strip() == 'True' else False)

    def update_preferences(self):
        if self.title == "Report Settings":
            with open(RSP, "w") as reportSettings:
                for i in range(self.settings.__len__()):
                    writer(reportSettings, self.settings, i)

        with open(temp_RSP, "w") as reportSettings:
            for i in range(self.settings.__len__()):
                writer(reportSettings, self.settings, i)

def writer(file, info, i):
    try:
        file.write(str(info[i].get()) + "\n")
    except:
        file.write(str(info[i]) + "\n")