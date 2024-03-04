import json
import os
import random
import sys
import time

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import (FigureCanvas)
from matplotlib.figure import Figure
from scipy import signal

import lozoya.data
import lozoya.file
import lozoya.gui
import lozoya.ml

matplotlib.use("TkAgg")


class LogMenu(submenu.SubMenu):
    def __init__(self, parent, title, icon):
        submenu.SubMenu.__init__(
            self,
            parent,
            title,
            icon,
        )
        self.automaticUpdate = False
        self.timeout = 0
        self.dirSelectorGrid = self.parent.menuBuilder.make_group_box(
            self._layout,
            **widgetkwargs.log.logDirsGrid,
        )
        self.indexGrid = self.parent.menuBuilder.make_group_box(
            self._layout,
            **widgetkwargs.log.logsGrid,
        )
        self.dictKeysGrid = self.parent.menuBuilder.make_group_box(
            self._layout,
            **widgetkwargs.log.dictKeysGrid,
        )
        self.contentsGrid = self.parent.menuBuilder.make_group_box(
            self._layout,
            **widgetkwargs.log.contentsGrid,
        )
        self.logReaderGrid = self.parent.menuBuilder.make_group_box(
            self._layout,
            **widgetkwargs.log.logReaderGrid,
        )
        self.toolsGrid = self.parent.menuBuilder.make_group_box(
            self.logReaderGrid.layout(),
            **widgetkwargs.log.toolsGrid,
        )
        self.dirSelector = self.parent.menuBuilder.make_list_widget(
            command=self.update_indices,
            layout=self.dirSelectorGrid.layout(),
        )
        self.indexSelector = self.parent.menuBuilder.make_list_widget(
            command=self.update_keys,
            layout=self.indexGrid.layout(),
        )
        self.dictKeysSelector = self.parent.menuBuilder.make_list_widget(
            command=self.update_contents,
            layout=self.dictKeysGrid.layout(),
        )
        self.contentsSelector = self.parent.menuBuilder.make_list_widget(
            command=self.update_log,
            layout=self.contentsGrid.layout(),
        )
        self.logReaderText = self.parent.menuBuilder.make_table(
            layout=self.logReaderGrid.layout(),
            multiSelect=True,
            row=0,
            column=1,
        )
        self.logReaderText.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.exportButton = self.parent.menuBuilder.make_button(
            self,
            command=self.export_log,
            layout=self.toolsGrid.layout(),
            **widgetkwargs.log.exportButton
        )
        self.refreshButton = self.parent.menuBuilder.make_button(
            self,
            command=self.update_log,
            layout=self.toolsGrid.layout(),
            **widgetkwargs.log.refreshButton,
        )
        self.toolsGrid.setMinimumHeight(150)
        self.toolsGrid.setMaximumHeight(150)
        self.setMinimumSize(500, 250)
        self.resize(500, 250)
        self.logReaderGrid.layout().setAlignment(self.toolsGrid, QtCore.Qt.AlignTop)
        # self.log_maintenance()
        # self.update_dir_list()
        # self.set_stylesheets()
        self.dataDook = dataprocessor.DataDude(source='scenarios')
        self.logReaderText.clear()
        self.dirSelector.insertItems(0, self.dataDook.scenarioDirs)
        self.activeScenario = self.dataDook.get_scenario(0)
        self.indexSelector.clear()
        self.indexSelector.insertItems(0, [str(n) for n, _ in enumerate(self.activeScenario.json)])
        self.j = sorted(self.activeScenario.json[0].keys())
        self.dictKeysSelector.clear()
        self.dictKeysSelector.insertItems(0, self.j)

    def new_log_file(self):
        try:
            path = os.path.join(paths.logDirs, self.parent.dataDude.date)
            filename = self.parent.dataDude.timestampHMS.replace(':', '_')
            fullpath = os.path.join(path, filename)
            with open(fullpath, 'a'):
                alertMsg = 'Created file {}.'.format(filename)
                self.update_status(alertMsg, 'success')
            return filename
        except Exception as e:
            alertMsg = 'New log file error: {}'.format(e)
            self.update_status(alertMsg, 'error')

    def new_log_dir(self):
        try:
            path = os.path.join(paths.logDirs, self.parent.dataDude.date)
            if not os.path.exists(path):
                os.mkdir(path)
            if len(os.listdir(path)) == 0:
                self.new_log_file()
        except Exception as e:
            alertMsg = 'New log directory error: {}'.format(e)
            self.update_status(alertMsg, 'error')

    def log_maintenance(self):
        try:
            _date = self.parent.dataDude.date
            date = _date.split('_')
            __old = [d for d in os.listdir(paths.logDirs) if d != 'errata']
            if len(__old) == 0:  # there are no log folders !
                self.new_log_dir()
                newfile = self.new_log_file()
                self.indexSelector.insertItems(0, [newfile])
                self.timeout = 0
                return
            _old = __old[-1]
            old = _old.split('_')
            _ = os.listdir(os.path.join(paths.logDirs, _old))[-1]
            full_ = os.path.join(paths.logDirs, _old, _)
            timeSinceModified = os.path.getmtime(full_)
            self.timeout += (self.parent.dataDude.time - timeSinceModified)
            if (int(date[1]) > int(old[1])) or ((int(date[1]) < int(old[1])) and int(date[0]) > int(old[0])):
                self.new_log_dir()
                self.dirSelector.insertItems(0, [_date])
                self.dirSelector.setCurrentRow(0)
            if self.timeout >= self.parent.config.logCreationInterval:
                newfile = self.new_log_file()
                self.indexSelector.insertItems(0, [newfile])
                self.timeout = 0
                if self.automaticUpdate:
                    self.indexSelector.item(0).setSelected(True)
                    self.indexSelector.setCurrentRow(0)
        except Exception as e:
            alertMsg = 'Log maintenance error: {}'.format(e)
            self.update_status(alertMsg, 'error')

    def update_dir_list(self):
        try:
            self.dirSelector.insertItems(0, self.dirs)
            self.dirSelector.setCurrentRow(0)
        except Exception as e:
            alertMsg = 'Update directory list error: {}'.format(e)
            self.update_status(alertMsg, 'error')

    @property
    def dirs(self):
        return [d for d in list(reversed(next(os.walk(paths.logDirs))[1])) if d != 'errata']

    @property
    def files(self):
        return self.get_files(self.activeDir)

    @property
    def activeDir(self):
        try:
            return self.dirSelector.currentItem().text()
        except Exception as e:
            alertMsg = 'Active log directory error: {}.'.format(e)
            self.update_status(alertMsg, 'error')

    @property
    def activeFile(self):
        try:
            return self.indexSelector.currentItem().text()
        except Exception as e:
            alertMsg = 'Active log file error: {}.'.format(e)
            self.update_status(alertMsg, 'error')

    def get_files(self, dir):
        return list(reversed(next(os.walk(os.path.join(paths.logDirs, dir)))[2]))

    def update_indices(self):
        try:
            index = self.dirSelector.currentRow()
            self.activeScenario = self.dataDook.get_scenario(index)
            self.indexSelector.clear()
            self.indexSelector.insertItems(0, [str(n) for n, _ in enumerate(self.activeScenario.json)])
            # self.indexSelector.setCurrentRow(0)
        except Exception as e:
            alertMsg = 'Update indices error: {}'.format(e)
            self.parent.transceiverMenu.update_status(alertMsg, 'error')

    def update_keys(self):
        try:
            index = self.indexSelector.currentRow()
            self.dictKeysSelector.clear()
            keys = sorted(self.activeScenario.json[index].keys())
            self.dictKeysSelector.insertItems(0, keys)
            # self.indexSelector.setCurrentRow(0)
        except Exception as e:
            alertMsg = 'Update keys error: {}'.format(e)
            self.parent.transceiverMenu.update_status(alertMsg, 'error')

    def update_contents(self):
        try:
            self.logReaderText.clear()
            index = self.indexSelector.currentRow()
            key = self.dictKeysSelector.currentItem().text()
            self.contentsSelector.clear()
            self.contents = self.activeScenario.json[index][key]
            print(self.contents)
            if type(self.contents) == str or type(self.contents) == int:
                self.contents = [str(self.contents)]
            if type(self.contents) == dict:
                self.update_log()
                return
            self.contentsSelector.insertItems(0, self.contents)
            # self.indexSelector.setCurrentRow(0)
        except Exception as e:
            alertMsg = 'Update contents error: {}'.format(e)
            print(self.contents)
            self.parent.transceiverMenu.update_status(alertMsg, 'error')

    def update_log(self):
        try:
            self.logReaderText.clear()
            rows = []
            for key in self.contents:
                _rows = 0
                for item in self.contents[key]:
                    _rows += 1
                rows.append(_rows)
            rows = max(rows)
            self.logReaderText.setColumnCount(len(self.contents))
            self.logReaderText.setRowCount(rows)
            for i, key in enumerate(self.contents):
                if type(self.contents[key]) == str:
                    item = QtWidgets.QTableWidgetItem(self.contents[key])
                    item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                    self.logReaderText.setItem(0, i, item)
                elif type(self.contents[key]) == list:
                    for m, key in enumerate(self.contents):
                        print(key)
                        if type(self.contents[key]) == str:
                            item = QtWidgets.QTableWidgetItem(self.contents[key])
                            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                            self.logReaderText.setItem(0, m, item)
                        else:
                            for r, item in enumerate(self.contents[key]):
                                item = QtWidgets.QTableWidgetItem(str(item))
                                item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                                self.logReaderText.setItem(m, r, item)
                else:
                    for j, item in enumerate(self.contents[key]):
                        item = QtWidgets.QTableWidgetItem(str(item))
                        item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                        self.logReaderText.setItem(i, j, item)
            # self.update_header()
            # self.logReaderText.setVerticalHeaderLabels(stamps)
            self.logReaderGrid.setTitle(self.activeScenario.fullpath)
        except Exception as e:
            alertMsg = 'Update log error: {}.'.format(str(e))
            self.update_status(alertMsg, 'error')

    def export_log(self):
        try:
            names = ['Date', 'Time'] + self.parent.config.head.split(self.parent.config.delimiter)
            readPath = os.path.join(paths.logDirs, self.activeDir, self.activeFile)
            filename = 'ofcg_{}_log_{}'.format(self.activeDir.replace(' ', '_').lower(), self.activeFile)
            savePath = '{}{}'.format(
                *QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    caption='Save File',
                    directory=os.path.join(os.getcwd(), filename),
                    filter='.xlsx',
                )
            )
            if savePath:
                self.parent.dataDude.export_data(
                    dataFrame=pd.read_csv(readPath, names=names),
                    savePath=savePath,
                    format='xlsx',
                )
                alertMsg = 'Exported log to {}'.format(savePath)
            else:
                alertMsg = status.exportCancel
            self.update_status(alertMsg, 'success')
        except Exception as e:
            alertMsg = 'Export log error: {}'.format(str(e))
            self.update_status(alertMsg, 'error')

    def update_log_reader(self, data):
        '''
        This function is called by the Transceiver listener thread when new data
        is received if automatic update is enabled in the Log Menu
        (self.automaticUpdate = True). If automatic update is disabled in the
        Log Menu (self.automaticUpdate = False), this function will not
        be called.
        :param data: (type str) The data that will be added to the log reader.
        :return: void
        '''
        try:
            if self.indexSelector.currentRow() == 0:
                # self.logReaderText.addItem(data)
                c = self.logReaderText.rowCount()
                self.logReaderText.setRowCount(c)
                _d = self._format(data.split(self.parent.config.delimiter))
                for n, d in enumerate(_d):  # remove datetime stamp
                    item = QtWidgets.QTableWidgetItem(d)
                    item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                    self.logReaderText.setItem(c, n, item)
                self.update_header()
                stamp = '{}'.format(self.parent.dataDude.timestampHMSms)
                self.logReaderText.setVerticalHeaderItem(c, QtWidgets.QTableWidgetItem(stamp))
        except Exception as e:
            print('Log reader update error: ' + str(e) + '\nData: ' + data)

    def write_to_log(self, data):
        try:
            date = self.parent.dataDude.date
            _date = date.replace('_', '/')
            time = self.parent.dataDude.timestampHMSms
            delimiter = self.parent.config.delimiter
            path = os.path.join(paths.logDirs, date, self.get_files(date)[0])
            with open(path, 'a+') as f:
                f.write('{}{}{}{}{}\n'.format(_date, delimiter, time, delimiter, data))
        except Exception as e:
            alertMsg = 'Write to log error: {}.'.format(str(e))
            self.update_status(alertMsg, 'error')

    def set_stylesheets(self):
        if self.parent.config.transceiverActive:
            autoScrollBtnStyleSheet = qss.buttonValid
        else:
            autoScrollBtnStyleSheet = qss.buttonDisabled
        exportBtnStyleSheet = qss.buttonValid
        refreshButtonStyleSheet = qss.buttonValid
        self.autoUpdateButton.setStyleSheet(qss.format_qss(autoScrollBtnStyleSheet, self.parent.palette))
        self.exportButton.setStyleSheet(qss.format_qss(exportBtnStyleSheet, self.parent.palette))
        self.refreshButton.setStyleSheet(qss.format_qss(refreshButtonStyleSheet, self.parent.palette))

    def update_header(self):
        head = self.parent.config.head.split(self.parent.config.delimiter)
        self.logReaderText.setHorizontalHeaderLabels(head)
        self.logReaderText.setColumnCount(len(head))
        self.logReaderText.resizeColumnsToContents()

    def _format(self, _list):
        try:
            dtypes = self.parent.config.dataType.split(self.parent.config.delimiter)
            _ = []
            for datum, dataType in zip(_list, dtypes):
                if dataType == 'f':
                    _.append('{:03.2f}'.format(float(datum)))
                else:
                    _.append(datum)
            return _
        except Exception as e:
            alertMsg = 'Format data error: {}.'.format(e)
            self.update_status(alertMsg, 'error')


class MainWindow(QWidget):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.layout = QVBoxLayout()
        self.layout.addWidget(MyBar(self))
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addStretch(-1)
        self.setMinimumSize(800, 400)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.pressing = False


class MainMenu(QtWidgets.QWidget):
    def __init__(self, app, title):
        super(MainMenu, self).__init__()
        self.setWindowTitle(title)
        self.setWindowIcon(QtGui.QIcon(icons.scenarioGenerator))
        self.app = app
        self.title = title
        self.pressing = False

        self.config = lozoya.file_api.TSConfigurationFile(parent=self)
        self.config.transceiverActive = False
        self.signalGenerator = signal.SignalGenerator(parent=self)
        self.palette = colors.Palette(parent=self, theme=self.config.theme, )
        self.menuBuilder = builder.MenuBuilder(parent=self)
        self.setStyleSheet(qss.format_qss(qss.window, self.palette))
        self.isPinned = False
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setStyleSheet(qss.format_qss(qss.window + qss.widget, self.palette))
        self.title = title
        self.setWindowTitle(title)

        self.hbox = QtWidgets.QGridLayout(self)
        self.hbox.addWidget(components.TSTitleBar(self, self.title), 0, 0)
        self.frame, self._layout = self.menuBuilder.subdivision(self)
        self.frameS, self._layoutS = self.menuBuilder.subdivision(self)
        self.hbox.addWidget(self.frame, 1, 0)
        self.hbox.addWidget(self.frameS, 2, 0)
        self.setLayout(self.hbox)
        self._layout.setContentsMargins(5, 0, 5, 0)
        self._layout.setSpacing(0)
        self._layoutS.setContentsMargins(5, 0, 5, 5)
        self._layoutS.setSpacing(0)
        self.hbox.setContentsMargins(0, 0, 0, 0)

        self.statusbar = self.menuBuilder.make_statusbar(self, self._layoutS, 0, 1)
        self.dataDude = datadude.DataDude(parent=self)
        self.plotDude = plotdude.Plotter(parent=self)
        self.logMenu = log.MenuLog(parent=self, **widgetkwargs.main.logMenu, )
        self.transceiverMenu = transceiver.MenuTransceiver(parent=self, **widgetkwargs.main.transceiverMenu, )
        self.plotMenu = plot.MenuPlot(parent=self, **widgetkwargs.main.plotMenu, )
        self.documentationMenu = documentation.MenuDocumentation(parent=self, **widgetkwargs.main.documentationMenu, )
        self.settingsMenu = settings.MenuSettings(parent=self, **widgetkwargs.main.settingsMenu, )

        # self.layout().addWidget(components.TitleBar(self, 'CubeSAT Controller'))
        # self.deviceDock = QtWidgets.QDockWidget('Device', self)
        # self.deviceDock.setWidget(self.deviceMenu)
        # self.deviceDock.setMaximumSize(500,500)
        # self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.deviceDock)
        # self.transceiverDock = QtWidgets.QDockWidget('Transceiver', self)
        # self.transceiverDock.setWidget(self.transceiverMenu)
        # self.transceiverDock.setMinimumSize(250,200)
        # self.transceiverDock.setMaximumSize(750,250)
        # self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.transceiverDock)
        # self.logDock = QtWidgets.QDockWidget('Log', self)
        # self.logDock.setMaximumSize(750,500)
        # self.logDock.setWidget(self.logMenu)
        # self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.logDock)
        self.toolbar = QtWidgets.QToolBar()
        self._layout.addWidget(self.toolbar, 2, 0)
        self.transceiverButton = self.menuBuilder.make_button(
            self, command=self.transceiverMenu.show, icon=icons.wifi,
            width=50, height=50, )
        self.toolbar.addWidget(self.transceiverButton)
        self.plotButton = self.menuBuilder.make_button(
            self, command=self.plotMenu.show, icon=icons.imu, width=50,
            height=50, )
        self.toolbar.addWidget(self.plotButton)
        self.logButton = self.menuBuilder.make_button(
            self, command=self.logMenu.show, icon=icons.bookShelf, width=50,
            height=50, )
        self.menu_bar()
        self.tool_bar()
        self.setContentsMargins(0, 0, 0, 0)
        self.setMinimumSize(250, 150)
        self.menuBar.setFixedHeight(22)
        self.menuBar.setStyleSheet(qss.format_qss(qss.menubar, self.palette))
        self.resize(250, 150)

        self.show()
        sys.exit(self.app.exec_())

    def menu_bar(self):
        self.menuBar = QtWidgets.QMenuBar()
        self._layout.addWidget(self.menuBar, 1, 0)
        fileMenu = self.menuBar.addMenu('File')
        settingsAction = QtWidgets.QAction('Settings', self)
        settingsAction.triggered.connect(self.settingsMenu.show)
        fileMenu.addAction(settingsAction)
        exitAction = QtWidgets.QAction('Exit', self)
        exitAction.triggered.connect(self.app.quit)
        fileMenu.addAction(exitAction)
        helpMenu = self.menuBar.addMenu('Help')
        documentationAction = QtWidgets.QAction('Documentation', self)
        documentationAction.triggered.connect(self.documentationMenu.show)
        helpMenu.addAction(documentationAction)

    def tool_bar(self):
        self.toolbar.addWidget(self.logButton)

    def update_status(self, msg, status):
        if status == 'success':
            stylesheet = qss.statusbarSuccess
        elif status == 'error':
            stylesheet = qss.statusbarError
            date = self.dataDude.date
            try:
                with open(os.path.join(paths.errorLog, date), 'a') as f:
                    timestamp = self.dataDude.timestampHMSms
                    _ = '{} {} {}\n'.format(date.replace('_', '/'), timestamp, msg)
                    print(_)
                    f.write(_)
            except:
                try:
                    os.mkdir(paths.errorLog)
                except:
                    pass
                with open(os.path.join(paths.errorLog, date), 'a') as f:
                    timestamp = self.dataDude.timestampHMSms
                    _ = '{} {} {}\n'.format(date.replace('_', '/'), timestamp, msg)
                    print(_)
                    f.write(_)
        elif status == 'alert':
            stylesheet = qss.statusbarAlert
        self.statusbar.showMessage(msg)
        self.statusbar.adjustSize()
        self.statusbar.setStyleSheet(qss.format_qss(stylesheet, self.palette))

    def update_everything(self, i, msg):
        if self.config.transceiverActive:
            self.dataDude.update_data(msg)
            if i % self.config.loggingRate == 0:
                try:
                    pass
                    self.logMenu.log_maintenance()
                    self.logMenu.write_to_log(msg)
                    # self.transceiverMenu.receivedMsg.setPlainText(msg)
                    if self.logMenu.automaticUpdate:
                        self.logMenu.update_log_reader(msg)  # time.sleep(0.05)
                except Exception as e:
                    print('Update everything error: ' + e)
        else:
            time.sleep(0.01)


class MyBar(QWidget):

    def __init__(self, parent):
        super(MyBar, self).__init__()
        self.parent = parent
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel('Transceiver')
        btn_size = 15
        self.btn_close = QPushButton("X")
        self.btn_close.clicked.connect(self.btn_close_clicked)
        self.btn_close.setFixedSize(btn_size, btn_size)
        self.btn_close.setStyleSheet("background-color: rgb(80,80,80);")
        self.btn_min = QPushButton("--")
        self.btn_min.clicked.connect(self.btn_min_clicked)
        self.btn_min.setFixedSize(btn_size, btn_size)
        self.btn_min.setStyleSheet("background-color: rgb(80,80,80);")
        self.title.setFixedHeight(btn_size)
        self.title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.btn_min)
        self.layout.addWidget(self.btn_close)
        self.title.setStyleSheet(
            """
                        background-color: rgb(80,80,80);
                        color: white;
                    """
        )
        self.setLayout(self.layout)
        self.start = QPoint(0, 0)
        self.pressing = False

    def resizeEvent(self, QResizeEvent):
        super(MyBar, self).resizeEvent(QResizeEvent)
        self.title.setFixedWidth(self.parent.width())

    def mousePressEvent(self, event):
        self.start = self.mapToGlobal(event.pos())
        self.pressing = True

    def mouseMoveEvent(self, event):
        if self.pressing:
            self.end = self.mapToGlobal(event.pos())
            self.movement = self.end - self.start
            self.parent.setGeometry(
                self.mapToGlobal(self.movement).x(),
                self.mapToGlobal(self.movement).y(),
                self.parent.width(),
                self.parent.height()
            )
            self.start = self.end

    def mouseReleaseEvent(self, QMouseEvent):
        self.pressing = False

    def btn_close_clicked(self):
        self.parent.close()

    def btn_min_clicked(self):
        self.parent.showMinimized()


class PlotMenu(submenu.SubMenu):
    def __init__(self, parent, title, icon):
        submenu.SubMenu.__init__(self, parent, title, icon, )
        try:
            self.leftFrame, self.leftLayout = self.parent.menuBuilder.subdivision(self)
            self.plotSettingsFrame, self.plotSettingsLayout = self.parent.menuBuilder.subdivision(self)
            self.plotFrame, self.velocityXPlotLayout = self.parent.menuBuilder.subdivision(self)
            self.plotSplitter = self.parent.menuBuilder.make_splitter(
                style='v',
                widgets=(self.plotSettingsFrame, self.plotFrame)
            )
            # self.plotSettings = self.parent.menuBuilder.make_group_box(
            #     parent=self.plotSettingsLayout,
            #     text='Settings',
            #     row=0,
            #     column=0,
            # )
            self.domainGroupBox = self.parent.menuBuilder.make_group_box(
                self.plotSettingsLayout.layout(),
                **widgetkwargs.plot.domainGroupBox, )
            _, self.plotDomainSlider = self.parent.menuBuilder.pair(
                self, pair='slider', text='',
                inputSettings=(0, self.parent.config.buffer - 10, self.parent.config.velocityXMinY, 1),
                command=lambda: self.update_plot_xlim(0), layout=self.domainGroupBox.layout(), row=0, column=0,
                tooltip=tooltips.domainSlider, )
            self.rangeGroupBox = self.parent.menuBuilder.make_group_box(
                self.plotSettingsLayout.layout(),
                **widgetkwargs.plot.rangeGroupBox, )
            _, self.plotMinYSlider = self.parent.menuBuilder.pair(
                self, pair='slider', text='Minimum',
                inputSettings=(-10000, 10000, self.parent.config.velocityXMinY, 1),
                command=lambda: self.update_plot_ylim(0), layout=self.rangeGroupBox.layout(), row=0, column=0,
                tooltip=tooltips.minYSlider, )
            _, self.plotMaxYSlider = self.parent.menuBuilder.pair(
                self, pair='slider', text='Maximum',
                inputSettings=(-10000, 10000, self.parent.config.velocityXMaxY, 1),
                command=lambda: self.update_plot_ylim(0), layout=self.rangeGroupBox.layout(), row=0, column=1,
                tooltip=tooltips.maxYSlider, )
            self.variablesGroupBox = self.parent.menuBuilder.make_group_box(
                self.plotSettingsLayout.layout(),
                **widgetkwargs.plot.variablesGroupBox, )
            self.yGroupBox, self.yCombo = self.parent.menuBuilder.pair(
                self, pair='combo', text='y',
                comboItems=['__timestamp__'] + self.parent.config.head.split(self.parent.config.delimiter), default=0,
                layout=self.variablesGroupBox.layout(), command=lambda: self.change_y(0), **widgetkwargs.plot.yCombo, )
            self.xGroupBox, self.xCombo = self.parent.menuBuilder.pair(
                self, pair='combo', text='x',
                comboItems=['__timestamp__'] + self.parent.config.head.split(self.parent.config.delimiter), default=3,
                layout=self.variablesGroupBox.layout(), command=lambda: self.change_x(0), **widgetkwargs.plot.xCombo, )
            # self.toolsGrid = self.parent.menuBuilder.make_group_box(
            #     self._layout,
            #     **widgetkwargs.plot.toolsGrid,
            # )
            # self.addSuplotButton = self.parent.menuBuilder.make_button(
            #     self,
            #     icon=icons.add,
            #     layout=self.toolsGrid.layout(),
            #     command=lambda: self.add_plot('0'),
            #     row=2,
            #     column=2,
            #     tooltip='Insert subplot.'
            # )
            self.plotSettingsLayout.setContentsMargins(0, 0, 0, 0)
            self.setMinimumSize(400, 400)
            self.resize(400, 400)
            self._layout.addWidget(self.plotSplitter)
            self.velocityXPlotLayout.addWidget(self.parent.plotDude.plots[0]['canvas'])
            self._layout.addWidget(self.leftFrame)
            self.update_plot_ylim(0)
        except Exception as e:
            print('Plot menu error: ' + str(e))

    def update_plot_xlim(self, var):
        self.parent.plotDude.plots[var]['ax'].set_xlim(
            xmin=self.plotDomainSlider.value(),
            xmax=self.parent.config.buffer, )
        self.parent.config.plotProperties[var]['domain'] = self.plotDomainSlider.value()
        self.parent.config.save_to_file()

    def update_plot_ylim(self, var):
        self.parent.plotDude.plots[var]['ax'].set_ylim(
            ymin=self.plotMinYSlider.value(),
            ymax=self.plotMaxYSlider.value(), )
        self.parent.config.plotProperties[var]['ymin'] = self.plotMinYSlider.value()
        self.plotMinYSlider.setRange(-10000, self.plotMaxYSlider.value() - 10)
        self.plotMaxYSlider.setRange(self.plotMinYSlider.value() + 10, 10000)
        self.parent.config.plotProperties[var]['ymax'] = self.plotMaxYSlider.value()
        self.parent.config.save_to_file()

    def change_x(self, var):
        _ = ['__timestamp__'] + self.parent.config.head.split(self.parent.config.delimiter)
        self.parent.config.plotProperties[var]['x'] = \
            (['__timestamp__'] + self.parent.config.head.split(self.parent.config.delimiter))[
                self.xCombo.currentIndex()]
        self.parent.config.save_to_file()

    def change_y(self, var):
        self.parent.config.plotProperties[var]['y'] = \
            (['__timestamp__'] + self.parent.config.head.split(self.parent.config.delimiter))[
                self.yCombo.currentIndex()]
        self.parent.config.save_to_file()

    def add_plot(self, plot):
        try:
            pass
        except Exception as e:
            print('Add plot error: ' + str(e))

    def closeEvent(self, *args, **kwargs):
        super(submenu.SubMenu, self).closeEvent(*args, **kwargs)
        for plot in self.parent.plotDude.plots:
            plot['anim'].event_source.stop()

    def showEvent(self, *args, **kwargs):
        super(submenu.SubMenu, self).showEvent(*args, **kwargs)
        for plot in self.parent.plotDude.plots:
            plot['anim'].event_source.start()


class PlotDude:
    '''
    This dude handles the plots.
    '''

    def __init__(self, parent):
        self.parent = parent
        self.plots = []
        self._x, self._y, self._z = 0, 0, 0
        self.make_plot('', 'gyro vel')

    def make_plot(self, x, y):
        n = len(self.plots)

        self.plots.append(
            self.get_plot(
                x=range(self.parent.config.buffer), y=self.parent.dataDude.get(y),
                updateFunction=self.plot_update, updateArgs=[n], plotColor=self.parent.palette.plotColors['blue'], )
        )
        self.parent.config.plotProperties.append(
            {
                'ymin': -100, 'ymax': 100, 'xmin': self.parent.config.buffer - 10, 'xmax': self.parent.config.buffer,
                'x':    x, 'y': y,
            }
        )

    def plot_update(self, i, plot):
        if not (self.parent.plotMenu.isHidden() or self.parent.plotMenu.isMinimized()):
            try:
                if self.parent.config.plotProperties[plot]['x'] == '__timestamp__':
                    xdata = range(len(self.parent.dataDude.data))
                else:
                    xdata = self.parent.dataDude.get(self.parent.config.plotProperties[plot]['x'])
                if self.parent.config.plotProperties[plot]['y'] == '__timestamp__':
                    ydata = range(len(self.parent.dataDude.data))
                else:
                    ydata = self.parent.dataDude.get(self.parent.config.plotProperties[plot]['y'])
                self.plots[plot]['plot'].set_xdata(x=xdata)
                self.plots[plot]['plot'].set_ydata(y=ydata)
            except Exception as e:
                alertMsg = 'Plot update error: {}'.format(str(e))
                self.parent.update_status(alertMsg, 'error')

    def get_plot(self, x, y, updateFunction, updateArgs, plotColor='#e27d60'):
        try:
            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            plot, = ax.plot(x, y, color='green', lw=2, zorder=3, )
            anim = animation.FuncAnimation(
                fig, updateFunction, frames=self.parent.config.frames,
                interval=self.parent.config.fps, fargs=updateArgs, )
            p = {'fig': fig, 'canvas': canvas, 'ax': ax, 'plot': plot, 'anim': anim, }
            self.format_plot(p)
            return p
        except Exception as e:
            alertMsg = 'Get plot error: {}.'.format(str(e))
            self.parent.update_status(alertMsg, 'error')

    def format_plot(self, p):
        for spine in p['ax'].spines:
            p['ax'].spines[spine].set_color('w')
        p['ax'].tick_params(colors='w')
        p['fig'].patch.set_facecolor(self.parent.palette.plotBackground)
        p['ax'].set_facecolor(self.parent.palette.plotBackground)
        p['ax'].grid()


def get_signal(q):
    s = 0
    for i in range(1000):
        s += random.randint(-10, 10) * np.sin(i * q)
    return s


def sine(q):
    i = q
    global V
    global Dsum
    global D
    global D1
    global integralVal
    Xx = np.linspace(i, i + signalFrequency, samps)
    X[:-1] = X[1:]
    X[-1] = Xx[-1]
    V[:-1] = V[1:]
    V[-1] = get_signal(i)  # np.sin(i)
    if noise:
        V[-1] += random.randint(-1000, 1000) / 1000
    if lpf:
        b, a = signal.butter(12, 2 * 0.1 / signalFrequency, btype='low')
        F1 = signal.filtfilt(b, a, V)
        lowpassGraph.set_data(X, F1)
    if integrate:
        if lpf:
            Dsum += np.trapz(F1[-2:], F1[-2:])
        else:
            Dsum += np.trapz(V[-2:], X[-2:])
        D[:-1] = D[1:]
        D[-1] = Dsum
        integralGraph.set_data(X, D)
    sinegraph.set_data(X, V)
    ax.set_xlim([X[0] - axOffset, X[-1] + signalFrequency + axOffset])


if integrate:
    for j, val in enumerate(V):
        D[j] = np.trapz(V[0:j], X[0:j])

plt.locator_params(nbins=2)
dataDude = dataprocessor.DataProcessor(source='scenarios')
for i in range(len(os.listdir(dataDude.source))):
    scenarioObject = dataDude.get_scenario(i)
    for object in scenarioObject.json:
        try:
            if not (object['dbType'] == 'ROUTE' or object['dbType'] == 'ATTACK_PLAN'):
                print(object['dbType'])
        except Exception as e:
            pass
results, requiredIDs, max = populate_results(scenarioObject, 2)

while len(requiredIDs) != 0:
    newRequiredIDs = []
    print(len(requiredIDs))
    for i, item in enumerate(requiredIDs):
        level, key, value = item.split(',')
        if key_in_results(key, value, results) == False:
            print(level, key, value)
            object = scenario.get_object_by_key(key, value, scenarioObject)
            if object:
                max = navigate_dict(newRequiredIDs, object, 1, max)
                results, newRequiredIDs = append_to_results(requiredIDs, results, scenarioObject, max)
            else:
                print('not found')
        else:
            print('Required id already in results')
    requiredIDs = newRequiredIDs

with open(os.path.join(dataDude.root, 'generated', 'test.raid'), 'w') as f:
    json.dump(results, f, )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
trainDataDir = ''
testDataDir = ''
base = os.path.join('H:\\', 'ida_ml', '10hz')
result = []
head = 'Time, ECRPositionX, ECRPositionY, ECRPositionZ, ECRVelocityX, ECRVelocityY, ECRVelocityZ, Latitude, Longitude, Altitude (HAE), Elevation, Height (AGL)'
col = ' ECRPositionX'
for dir in os.listdir(base):
    for file in os.listdir(os.path.join(base, dir)):
        split = file.split('.')
        if len(split) > 1:
            if split[1] == 'csv':
                fullpath = os.path.join(base, dir, file)
                with open(fullpath, 'r') as f:
                    _head = f.readline().strip('\n')
                    if head == _head:
                        result.append(fullpath)
print(len(result))
dfMain = pd.DataFrame(columns=head.split(','))
i = 0
for file in result:
    df = pd.read_csv(file)
    # df.loc[:, col].plot(kind='line')
    df.plot(kind='line')
    i += 1
    if i == 2:
        break
plt.show()
signalFrequency = np.pi
samps = 50
axOffset = 5
integrate = False
noise = False
lpf = True
hpf = False
fig, ax = plt.subplots(1, 1)
sinegraph, = ax.plot([], [])
lowpassGraph, = ax.plot([], [])
integralGraph, = ax.plot([], [])
integralGraph1, = ax.plot([], [])
ax.set_ylim([- 500, 500])
X = np.linspace(0, signalFrequency, samps)
V = np.zeros_like(X)
F1 = np.zeros_like(X)
D = np.zeros_like(V)
D1 = np.zeros_like(V)
Dsum = 0
integralVal = 0
anim = animation.FuncAnimation(fig, sine, interval=100)
plt.grid(True, which='both')
plt.show()
# np.random.seed(6)
# sampleRate = 10
# cutoff = 2
# noise = True
# hpf = True
# lpf = True
# integral = True
# derivative = True
# velocity = False
# displacement = False
# t = np.arange(0, sampleRate, 0.1)
# s = np.sin(t)
# #c = -np.cos(t)
# if noise:
#     s += + np.random.normal(0, 100, t.shape)
# plt.plot(t, s)
# #plt.plot(t, c)
# if hpf:
#     b, a = signal.butter(1, 2*cutoff/sampleRate, btype='high')
#     s = signal.filtfilt(b, a, s)
#     plt.plot(t, s)
# if lpf:
#     b, a = signal.butter(1, 2*cutoff/sampleRate, btype='low')
#     s = signal.filtfilt(b, a, s)
#     plt.plot(t, s)
# if velocity:
#     v = np.zeros_like(s)
#     for i, val in enumerate(s):
#         v[i] = np.trapz(s[0:i], t[0:i])
#     plt.plot(t, v)
# if displacement:
#     d = np.zeros_like(v)
#     for i, val in enumerate(v):
#         d[i] = np.trapz(v[0:i], t[0:i])
#     plt.plot(t, d)
# rms = 1
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.grid(True, which='both')
# plt.show()
ax = plt.gca()

mlDude = MLDude()
plt.show()
