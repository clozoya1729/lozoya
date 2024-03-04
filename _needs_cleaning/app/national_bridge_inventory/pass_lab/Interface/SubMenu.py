"""
Author: Christian Lozoya, 2017
"""

from Utilities.MenuFunctions import *


class SubMenu(QtWidgets.QWidget):
    def __init__(self, parent, title, icon, standard=True):
        self.parent = parent
        super(SubMenu, self).__init__()
        self.setStyleSheet(WINDOW_STYLE + WIDGET_STYLE)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle(title)
        self.icon = QtGui.QIcon(icon)
        self.setWindowIcon(self.icon)
        self.resize(350, 400)
        self.hbox = QtWidgets.QHBoxLayout(self)
        self.topleft = make_frame(parent)
        self.topLeftLayout = make_grid(self.topleft)
        self.topright = make_frame(parent)
        self.topRightLayout = make_grid(self.topright)
        self.splitter1 = make_splitter(style='h', widgets=(self.topleft, self.topright))
        self.bottom = make_frame(parent)
        self.bottomLayout = make_grid(self.bottom)
        self.splitter2 = make_splitter(style='v', widgets=(self.splitter1, self.bottom))
        if standard:
            self.hbox.addWidget(self.splitter2)
        self.setLayout(self.hbox)
        self.topleft.setMinimumSize(WIDTH / 6, HEIGHT / 8)
        set_widget_size(self.topright, WIDTH / 6, HEIGHT / 8)
        self.bottom.setMinimumSize(WIDTH / 4, HEIGHT / 4)

        menubar = QtWidgets.QMenuBar()
        menubar.setStyleSheet(MENUBAR_STYLE)
        self.file_dropdown(menubar)

    def file_dropdown(self, menubar):
        fileMenu = menubar.addMenu('&File')

        openFileAction = QtWidgets.QAction(QtGui.QIcon(OPEN_FILE_ICON), '&Open File', self)
        openFileAction.setShortcut('Ctrl+O')
        openFileAction.setStatusTip(OPEN_FILE_DESCRIPTION)
        openFileAction.triggered.connect(lambda: open_files(self))
        fileMenu.addAction(openFileAction)

        saveFileAction = QtWidgets.QAction(QtGui.QIcon(SAVE_FILE_ICON), '&Save File', self)
        saveFileAction.setShortcut('Ctrl+S')
        saveFileAction.setStatusTip(SAVE_FILE_DESCRIPTION)
        saveFileAction.triggered.connect(lambda: save_file(self))

        fileMenu.addAction(saveFileAction)


class DatabaseMenu(SubMenu):
    def __init__(self, parent, title, icon):
        SubMenu.__init__(self, parent, title, icon)
        top_left_splitter(self, open=open_folder, save=save_file)
        self.runButton = make_button(self, command=lambda: save_file(self), text='Run')
        self.topRightLayout.addWidget(self.runButton, 4, 0)


from Search.Search import *


class SearchMenu(SubMenu):
    def __init__(self, parent, title, icon):
        SubMenu.__init__(self, parent, title, icon, standard=False)
        top_left_splitter(self, open=open_folder, save=open_folder, saveIcon=SAVE_FOLDER_ICON)

        """TOP RIGHT"""
        # TODO EXPORT AS TEXT, EXCEL, OR PDF checkbuttons
        self.txtLabel = make_label(self, 'txt:')
        self.topRightLayout.addWidget(self.txtLabel, 1, 0)

        self.txtCheck = make_button(self, type='check')
        self.topRightLayout.addWidget(self.txtCheck, 1, 1)

        self.csvLabel = make_label(self, 'csv:')
        self.topRightLayout.addWidget(self.csvLabel, 1, 2)

        self.csvCheck = make_button(self, type='check')
        self.topRightLayout.addWidget(self.csvCheck, 1, 3)

        self.excelLabel = make_label(self, 'Excel:')
        self.topRightLayout.addWidget(self.excelLabel, 2, 0)

        self.excelCheck = QtWidgets.QCheckBox(self)
        self.topRightLayout.addWidget(self.excelCheck, 2, 1)

        self.pdfLabel = make_label(self, 'pdf:')
        self.topRightLayout.addWidget(self.pdfLabel, 2, 2)

        self.pdfCheck = QtWidgets.QCheckBox(self)
        self.topRightLayout.addWidget(self.pdfCheck, 2, 3)

        # TODO UNION OR INTERSECTION SEARCH radiobutton
        self.unionLabel = make_label(self, 'Union:')
        self.topRightLayout.addWidget(self.unionLabel, 3, 0)

        self.unionRadio = make_button(self, type='radio')
        self.topRightLayout.addWidget(self.unionRadio, 3, 1)

        self.intersectionLabel = make_label(self, 'Intersection:')
        self.topRightLayout.addWidget(self.intersectionLabel, 3, 2)

        self.intersectionRadio = make_button(self, type='radio')
        self.topRightLayout.addWidget(self.intersectionRadio, 3, 3)

        # RUN BUTTON
        self.runButton = make_button(self, command=self.run, text='Run')
        self.topRightLayout.addWidget(self.runButton, 4, 0)

        """BOTTOM"""
        widget = QtWidgets.QWidget()
        widget.setStyleSheet(WIDGET_STYLE)
        self.bottom = make_grid(self)
        self.entries = {}
        for i, header in enumerate(HEADERS, start=1):
            self.entries[header] = make_entry(parent)
            label = make_label(self, header + ':')
            self.bottom.addWidget(label, i, 0)
            self.bottom.addWidget(self.entries[header], i, 1)

        widget.setLayout(self.bottom)
        scroll = make_scroll_area(widget)

        self.bottomLayout = make_grid(scroll)
        self.splitter2 = make_splitter(style='v', widgets=(self.splitter1, scroll))

        self.clearButton = make_button(self, command=lambda: clear(self.entries), text='Clear')
        self.splitter2.addWidget(self.clearButton)

        self.topleft.setMinimumSize(WIDTH / 6, HEIGHT / 8)
        set_widget_size(self.topright, WIDTH / 6, HEIGHT / 8)
        scroll.setMinimumSize(WIDTH / 4, HEIGHT / 4)

        self.hbox.addWidget(self.splitter2)
        self.setLayout(self.hbox)

    def run(self):
        dtype = {entry: str for entry in self.entries if entry != FOLDER}
        columns = [entry for entry in self.entries if entry not in (FILE, FOLDER) and self.entries[entry].text() is not '']

        if self.readLabel.text():
            dir = self.readLabel.text()
        else:
            dir = DATABASE

        folder = self.entries[FOLDER].text()
        filesList = set_files(state_code_state(), self.entries[FILE].text())

        with open(TEMP_SAVE, 'w') as saveFile:
            for i in range(self.entries.__len__()):
                saveFile.write(self.entries[FEATURES[i]].text() + "\n")

        for subdir, dirs, files in os.walk(dir):
            for file in files:
                for f in filesList:
                    if f in file and folder in subdir:
                        if os.stat(os.path.join(subdir, file)).st_size != 0:
                            search(entries=self.entries, file=file, dir=subdir[-2:], dtype=dtype, columns=columns)

    def save_query(self):
        path = save_file(self, fileExt=QRY_FILES)
        try:
            if path[-4:] != QRY_EXT:
                path = path + QRY_EXT
            with open(path, "w") as saveFile:
                for i in range(HEADERS.__len__()):
                    saveFile.write(self.entries[FEATURES[i]].get() + "\n")
        except:
            return

    def load_query(self):
        try:
            path = open_files(fileExt=QRY_FILES)
            with open(path, "r") as loadFile:
                for i in range(self.entries.__len__()):
                    self.entries[FEATURES[i]].delete(0, self.entries[FEATURES[i]].get().__len__())
                for i, line in enumerate(loadFile):
                    if line.strip():
                        self.entries[FEATURES[i]].insert(i, line.strip())
        except:
            return


from Search.GeographicSearch import *


class GeoSearchMenu(SubMenu):
    def __init__(self, parent, title, icon):
        SubMenu.__init__(self, parent, title, icon)
        top_left_splitter(self, open=open_folder, save=save_file, fileExt=KML_FILES)

        """TOP RIGHT"""
        # UNITS SELECTION
        self.units = UNITS[0]

        self.unitsLabel = make_label(self, 'Units:')
        self.topRightLayout.addWidget(self.unitsLabel, 0, 0)

        self.unitsCombo = make_combo(self, UNITS, self.change_units)
        self.topRightLayout.addWidget(self.unitsCombo, 0, 1)

        # RADIUS, LATITUDE, LONGITUDE ENTRY
        self.radiusLabel = make_label(self, 'Radius:')
        self.topRightLayout.addWidget(self.radiusLabel, 1, 0)

        self.rad = make_entry(self, width=40)
        self.topRightLayout.addWidget(self.rad, 1, 1)

        self.lonLabel = make_label(self, 'Longitude:')
        self.topRightLayout.addWidget(self.lonLabel, 2, 0)

        self.lon = make_entry(self, width=40)
        self.topRightLayout.addWidget(self.lon, 2, 1)

        self.latLabel = make_label(self, 'Latitude:')
        self.topRightLayout.addWidget(self.latLabel, 3, 0)

        self.lat = make_entry(self, width=40)
        self.topRightLayout.addWidget(self.lat, 3, 1)

        self.runButton = make_button(self, command=lambda: self.run_geo_search(), text='Run')
        self.topRightLayout.addWidget(self.runButton, 4, 0)

        # BOTTOM
        self.mapa = folium.Map(tiles='Stamen Terrain')
        self.mapa.save(BASE_MAP)

        self.mapContainer = QtWebEngineWidgets.QWebEngineView()
        self.mapContainer.load(QtCore.QUrl.fromLocalFile(BASE_MAP))
        self.bottomLayout.addWidget(self.mapContainer)

    def change_units(self, units):
        self.units = units

    def run_geo_search(self):
        self.mapa = folium.Map(tiles='Stamen Terrain')
        indices, coordinates = gather_coordinates(self.readLabel.text())
        indices, coordinates = interpret_query(
            centroid=(int(self.lat.text()), int(self.lon.text())), indices=indices,
            coordinates=coordinates, radius=int(self.rad.text()), units=self.units)

        create_kml(indices, coordinates, self.saveLabel.text())
        for point in range(len(coordinates)):
            folium.Marker(coordinates[point], popup=indices[point]).add_to(self.mapa)
        self.mapa.save(BASE_MAP)
        self.mapContainer.load(QtCore.QUrl.fromLocalFile(BASE_MAP))


class MachineLearningMenu(SubMenu):
    def __init__(self, parent, title, icon):
        SubMenu.__init__(self, parent, title, icon)
        top_left_splitter(self, open=open_folder, save=save_file)

        """TOP RIGHT"""
        # TODO EXPORT AS TEXT, EXCEL, PDF
        self.svmLabel = make_label(self, 'SVM:')
        self.svmCheck = QtWidgets.QCheckBox(self)
        self.topRightLayout.addWidget(self.svmLabel, 1, 0)
        self.topRightLayout.addWidget(self.svmCheck, 1, 1)

        self.kMeansLabel = make_label(self, 'K-Means:')
        self.kMeansCheck = QtWidgets.QCheckBox(self)
        self.topRightLayout.addWidget(self.kMeansLabel, 1, 2)
        self.topRightLayout.addWidget(self.kMeansCheck, 1, 3)

        self.mCMCLabel = make_label(self, 'MCMC:')
        self.mCMCCheck = QtWidgets.QCheckBox(self)
        self.topRightLayout.addWidget(self.mCMCLabel, 2, 0)
        self.topRightLayout.addWidget(self.mCMCCheck, 2, 1)

        self.nNLabel = make_label(self, 'Neural Network:')
        self.excelCheck = QtWidgets.QCheckBox(self)
        self.topRightLayout.addWidget(self.nNLabel, 2, 2)
        self.topRightLayout.addWidget(self.excelCheck, 2, 3)

        # UNION OR INTERSECTION SEARCH
        self.fullLabel = make_label(self, 'Full:')
        self.fullRadio = QtWidgets.QRadioButton(self)
        self.topRightLayout.addWidget(self.fullLabel, 3, 0)
        self.topRightLayout.addWidget(self.fullRadio, 3, 1)

        self.cleanLabel = make_label(self, 'Clean:')
        self.cleanRadio = QtWidgets.QRadioButton(self)
        self.topRightLayout.addWidget(self.cleanLabel, 3, 2)
        self.topRightLayout.addWidget(self.cleanRadio, 3, 3)
        self.runButton = make_button(self, command=lambda: save_file(self), text='Run')
        self.topRightLayout.addWidget(self.runButton, 4, 0)

        self.graphContainer = QtWebEngineWidgets.QWebEngineView()
        self.graphContainer.load(
            QtCore.QUrl.fromLocalFile(BASE_GRAPH))
        self.bottomLayout.addWidget(self.graphContainer)