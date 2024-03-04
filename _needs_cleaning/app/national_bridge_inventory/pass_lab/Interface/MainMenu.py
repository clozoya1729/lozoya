"""
Author: Christian Lozoya, 2017
"""

from Interface.SubMenu import *


class MainMenu(QtWidgets.QMainWindow):
    def __init__(self, app, title):
        super(MainMenu, self).__init__()
        self.setStyleSheet(WINDOW_STYLE)
        width = WIDTH / 2
        height = HEIGHT / 2
        xpos = (WIDTH - width) / 2
        ypos = (HEIGHT - height) / 2
        self.setGeometry(xpos, ypos, width, height)
        self.setWindowTitle(title)
        self.setWindowIcon(QtGui.QIcon(WINDOW_ICON))
        self.databaseMenu = DatabaseMenu(self, 'Database', DATABASE_ICON)
        self.searchMenu = SearchMenu(self, 'Search', SEARCH_ICON)
        self.geoSearchMenu = GeoSearchMenu(self, 'Geographic Search', GEO_SEARCH_ICON)
        self.machineLearningMenu = MachineLearningMenu(self, 'Machine Learning', MACHINE_LEARNING_ICON)

        self.data_toolbar()
        self.show()
        sys.exit(app.exec_())


    def data_toolbar(self):
        self.dataToolbar = QtWidgets.QToolBar(self)
        self.dataToolbar.setStyleSheet(TOOLBAR_STYLE)
        self.dataToolbar.setIconSize(QtCore.QSize(100, 100))
        self.dataToolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        icon = QtGui.QIcon(DATABASE_ICON)
        databaseAction = QtWidgets.QAction(icon, DATABASE_DESCRIPTION, self)
        databaseAction.setShortcut('Ctrl+D')
        databaseAction.triggered.connect(self.databaseMenu.show)
        databaseAction.setIconText("Database")
        self.dataToolbar.addAction(databaseAction)

        icon = QtGui.QIcon(SEARCH_ICON)
        searchAction = QtWidgets.QAction(icon, SEARCH_DESCRIPTION, self)
        searchAction.setShortcut('Ctrl+R')
        searchAction.triggered.connect(self.searchMenu.show)
        searchAction.setIconText("Search Engine")
        self.dataToolbar.addAction(searchAction)

        icon = QtGui.QIcon(GEO_SEARCH_ICON)
        geoSearchAction = QtWidgets.QAction(icon, GEO_SEARCH_DESCRIPTION, self)
        geoSearchAction.setShortcut('Ctrl+G')
        geoSearchAction.triggered.connect(self.geoSearchMenu.show)
        geoSearchAction.setIconText("Geographic Search")
        self.dataToolbar.addAction(geoSearchAction)

        icon = QtGui.QIcon(MACHINE_LEARNING_ICON)
        machineLearningAction = QtWidgets.QAction(icon, MACHINE_LEARNING_DESCRIPTION, self)
        machineLearningAction.setShortcut('Ctrl+M')
        machineLearningAction.triggered.connect(self.machineLearningMenu.show)
        machineLearningAction.setIconText("Machine Learning")
        self.dataToolbar.addAction(machineLearningAction)

        self.addToolBar(self.dataToolbar)