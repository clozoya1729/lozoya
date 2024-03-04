import abc
import functools

import matplotlib
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWebEngineWidgets, QtWidgets
# from PyQt5.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

matplotlib.use('Qt5Agg')
borderWidth = 1
gradient0 = 'white'
gradient1 = 'grey'
fontSize = 20
background0 = '#121e26'
background1 = '#283635'
highlight0 = '#9d6556'
highlight1 = '#d2c0b2'
colorFont = '#f4efeb'
qssLayout = f"""
QWidget {{
    background: {background0};
    color: {colorFont};
}}
"""
qssInput = f"""
QWidget {{
    background: {highlight1};
    color: {background0};
}}
"""
qssCheckbox = f"""
"""
qssCombo = f"""
QComboBox::item:selected{{
}}
"""
qss = f"""
QWidget {{
    background: {background0};
    color: {colorFont};
}}
QWidget::hover {{
    color: {highlight1};
}}
"""
qss0 = f"""
QWidget::hover {{
    background: {background1};
}}
QSpinBox, QDoubleSpinBox {{
    color: {colorFont};
    font-size: {fontSize};
    background: {background0};
                }}
QSpinBox::hover, QDoubleSpinBox::hover {{
    border: {borderWidth} solid {background1};
    color: {colorFont};
    font-size: {fontSize};
    background: {highlight0};
}}
"""
matplotlib.rcParams['text.color'] = colorFont
matplotlib.rcParams['xtick.color'] = colorFont
matplotlib.rcParams['ytick.color'] = colorFont
matplotlib.rcParams['axes.labelcolor'] = colorFont


class TSApp(QtWidgets.QApplication):
    def __init__(self, name='TSApp', root='', size=(1280, 720)):
        QtWidgets.QApplication.__init__(self, [])
        self.name = name
        self.root = root
        self.window = TSMainWindow(title=name, size=size, )  # icon=QtGui.QIcon(configuration.CUBE_ICON))
        self.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
        self.window.show()


class TSFormField:
    def __init__(self, name='', callback=None, tooltip=''):
        self.name = name
        if hasattr(self, 'setToolTip'):
            self.setToolTip(tooltip)
        if hasattr(self, 'setStyleSheet'):
            self.setStyleSheet(qss)

    @abc.abstractmethod
    def get_value(self):
        pass

    # def reset(self, *args, **kwargs):  #     self.disabled = False  #     self.update_able()  #     self.set_stylesheet('')  #  # def set_disabled(self, *args, **kwargs):  #     self.disabled = True  #     self.update_able()  #     self.set_stylesheet('Disabled')  #  # def set_error(self, *args, **kwargs):  #     self.disabled = False  #     self.update_able()  #     self.set_stylesheet('Error')  #  # def set_valid(self, *args, **kwargs):  #     self.disabled = False  #     self.update_able()  #     self.set_stylesheet('Valid')  #  # def set_glowing(self, *args, **kwargs):  #     self.disabled = False  #     self.update_able()  #     self.set_stylesheet('Glow')  #  # def set_connected(self, *args, **kwargs):  #     self.disabled = True  #     self.update_able()  #     self.set_stylesheet('Connected')


class TSBrowser(QtWebEngineWidgets.QWebEngineView):
    def __init__(self, app, parent):
        super(TSBrowser, self).__init__()
        self.app = app
        self.parent = parent

    def disable_right_click(self, *args, **kwargs):
        self.setContextMenuPolicy(QtCore.Qt.NoContextMenu)

    def read_file(self, file, online=False):
        try:
            if online:
                self.load(QtCore.QUrl(file))
            else:
                self.load(QtCore.QUrl.fromLocalFile(file))
        except Exception as e:
            print(e)


class TSInputBase(TSFormField):
    def __init__(self, name='', callback=None, tooltip=''):
        TSFormField.__init__(self, name, callback, tooltip)
        if hasattr(self, 'setStyleSheet'):
            self.setStyleSheet(qssInput)


class TSInputButton(TSInputBase, QtWidgets.QPushButton):
    def __init__(self, name='', text='', callback=None):
        QtWidgets.QPushButton.__init__(self)
        self.setText(text)
        if callback:
            self._callback = callback
            self.clicked.connect(self.callback)
        TSInputBase.__init__(self, name=name, callback=callback)

    def callback(self):
        return self._callback()

    def get_value(self):
        return self.text()


class TSInputCheckbox(TSInputBase, QtWidgets.QCheckBox):
    def __init__(self, name, value=False, callback=None):
        QtWidgets.QCheckBox.__init__(self)
        if callback:
            self._callback = callback
            self.clicked.connect(self.callback)
        self.setChecked(value)
        TSInputBase.__init__(self, name=name, callback=callback, tooltip='')
        self.setStyleSheet(qssCheckbox)

    def callback(self):
        return self._callback(self.isChecked())

    def get_value(self):
        return self.isChecked()


class TSInputCombo(TSInputBase, QtWidgets.QComboBox):
    def __init__(self, options, default=0, name='', callback=None, tooltip=''):
        QtWidgets.QComboBox.__init__(self)
        self.disabled = False
        if options:
            self.update_options(options, default)
        self.adjustSize()
        if callback:
            self.activated[str].connect(callback)
        # self.setMinimumHeight(self.app.palette.minHeight)
        TSInputBase.__init__(self, name, tooltip)
        self.setStyleSheet(qssCombo)

    # def set_stylesheet(self, style):
    #     self.setStyleSheet(qss_format(vars(qss.selectionstyles)['dropdown{}'.format(style)], self.app.palette))

    def get_value(self):
        return self.currentText()

    def update_options(self, options, default=0):
        self.options = options
        self.default = default
        self.clear()
        self.addItems(self.options)
        self.setCurrentIndex(self.default)


class TSDialog(QtWidgets.QDialog):
    def __init__(self, title, message, accept, reject):
        QtWidgets.QDialog.__init__(self)
        self.setWindowTitle(title)
        buttons = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonBox = QtWidgets.QDialogButtonBox(buttons)
        self.accept_callback = accept
        self.reject_callback = reject
        self.buttonBox.accepted.connect(self.callback_accept)
        self.buttonBox.rejected.connect(self.callback_reject)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(QtWidgets.QLabel(message))
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def callback_accept(self):
        self.accept_callback()
        self.accept()

    def callback_reject(self):
        self.reject_callback()
        self.reject()


class TSInputDial(QtWidgets.QDial):
    def __init__(self, app, parent, settings, command=None, tooltip=''):
        super(TSInputDial, self).__init__(parent)
        self.app = app
        self.parent = parent
        self.setStyleSheet(qss_format(qss.sliderstyles.dial, self.app.palette))
        self.setMinimum(settings[0])
        self.setMaximum(settings[1])
        self.setValue(settings[2])
        self.setSingleStep(settings[3])
        self.setNotchesVisible(True)
        self.setWrapping(False)
        if command:
            self.valueChanged.connect(command)
        self.setToolTip(tooltip)


class TSInputSlider(TSInputBase, QtWidgets.QSlider):
    def __init__(
        self, name='', value=0, direction='h', callback=None, range=(0, 100), step=1, tooltip=''
    ):
        if direction == 'v':
            QtWidgets.QSlider.__init__(self, QtCore.Qt.Vertical)
        else:
            QtWidgets.QSlider.__init__(self, QtCore.Qt.Horizontal)
        self.setRange(range[0], range[1])
        self.setValue(value)
        self.setSingleStep(step)
        self.setTickPosition(QtWidgets.QSlider.TicksBelow)
        # self.setMinimumHeight(self.app.palette.minHeight)
        if callback:
            self._callback = callback
            self.valueChanged.connect(self.callback)
        TSInputBase.__init__(self, name=name, tooltip=tooltip)

    def callback(self):
        return self._callback(self.value())

    def get_value(self):
        return self.value()


class TSInputSpinbox(TSInputBase, QtWidgets.QSpinBox):
    def __init__(self, name='', value=0, callback=None):
        QtWidgets.QSpinBox.__init__(self)
        TSInputBase.__init__(self, name=name, tooltip='')
        if callback:
            self._callback = callback
            self.valueChanged.connect(self.callback)
        self.setValue(value)

    def callback(self):
        return self._callback(self.value())

    def get_value(self):
        return self.value()


class TSInputSpinboxDouble(TSInputBase, QtWidgets.QDoubleSpinBox):
    def __init__(self, name='', value=0, callback=None):
        QtWidgets.QDoubleSpinBox.__init__(self)
        TSInputBase.__init__(self, name=name, tooltip='')
        self.setValue(value)
        if callback:
            self._callback = callback
            self.valueChanged.connect(self.callback)

    def callback(self):
        return self._callback(self.value())

    def get_value(self):
        return self.value()


class TSFileDialog(TSFormField, QtWidgets.QFileDialog):
    def __init__(self, name=''):
        QtWidgets.QFileDialog.__init__(self)
        TSFormField.__init__(self, name)


class TSForm(QtWidgets.QDialog):
    def __init__(self, fields=tuple(), name='', horizontal=None):
        QtWidgets.QDialog.__init__(self)
        self.name = name
        self.horizontal = horizontal
        self.fields = []
        if horizontal:
            self.layout = QtWidgets.QHBoxLayout()
        else:
            self.layout = QtWidgets.QFormLayout()
        self.setLayout(self.layout)
        if fields:
            self.add_fields(fields)
        self.setStyleSheet(qssLayout)

    def add_label(self, field):
        self.layout.addRow(field)

    def add_field(self, field):
        self.fields.append(field)
        if self.horizontal:
            self.layout.addWidget(field)
        else:
            if field.name:
                self.layout.addRow(QtWidgets.QLabel(field.name), field)
            else:
                self.layout.addWidget(field)

    def add_fields(self, fields):
        for field in fields:
            self.add_field(field)

    def get_values(self):
        values = {}
        for field in self.fields:
            values[field.name] = field.get_value()
        return values


class TSLabel(QtWidgets.QLabel):
    def __init__(self, name):
        QtWidgets.QLabel.__init__(self)
        self.setText(name)
        TSFormField.__init__(self)


class TSListWidget(TSFormField, QtWidgets.QListWidget):
    def __init__(self, name, items, callback=None):
        QtWidgets.QListWidget.__init__(self)
        self.addItems(items)
        self.setCurrentRow(0)
        if callback:
            self._callback = callback
            self.itemClicked.connect(self.callback)
        # self.setFixedWidth(self.sizeHintForColumn(0) + 10)
        # self.setFixedHeight(self.minimumSizeHint().height() + 10)
        self.setFixedSize(
            self.sizeHintForColumn(0) + self.frameWidth() * 2,
            self.sizeHintForRow(0) * self.count() + 2 * self.frameWidth()
        )

        TSFormField.__init__(self, name=name)

    def callback(self):
        if self.selectedItems:
            return self._callback(self.row(self.selectedItems()[0]))


class TSLogArea(TSFormField, QtWidgets.QTextEdit):
    def __init__(self, name, fixedWidth=None, fixedHeight=None):
        QtWidgets.QTextEdit.__init__(self)
        self.setReadOnly(True)
        if fixedWidth:
            self.setFixedWidth(fixedWidth)
        if fixedHeight:
            self.setFixedHeight(fixedHeight)
        TSFormField.__init__(self, name=name)

    def update(self, text):
        currentText = self.toPlainText()
        self.setText(currentText + text + '\n')
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class TSTabWidgetSwitcher(TSForm, QtWidgets.QWidget):
    def __init__(self, widgets, name=''):
        QtWidgets.QWidget.__init__(self)
        self.stack = TSStackedWidget(name=name)
        self.widgets = widgets
        actions = []
        for widget in widgets:
            self.stack.addWidget(widget)
            action = make_action(
                widget.name, callback=functools.partial(self.stack.setCurrentWidget, widget)
            )
            actions.append(action)
        self.toolbar = make_toolbar(name, actions)
        group = make_action_group(self.toolbar, actions)
        fields = [self.toolbar, self.stack]
        TSForm.__init__(self, fields, name)

    @property
    def currentIndex(self):
        return self.toolbar.currentIndex().row()


class TSListWidgetSwitcher(TSForm, QtWidgets.QWidget):
    def __init__(self, widgets, name=''):
        QtWidgets.QWidget.__init__(self)
        self.stack = TSStackedWidget(name=name)
        self.widgets = widgets
        actions = []
        for widget in widgets:
            self.stack.addWidget(widget)
            action = make_action(
                widget.name, callback=None, )
            actions.append(action)
        self.listWidget = TSListWidget(
            name, [widget.name for widget in widgets], callback=self.stack.setCurrentIndex
        )
        fields = [self.listWidget, self.stack]
        TSForm.__init__(self, fields, name, horizontal=True)

    @property
    def currentIndex(self):
        return self.listWidget.currentIndex().row()

    @property
    def currentForm(self):
        index = self.currentIndex
        return self.widgets[index]


class TSMainWindow(QtWidgets.QMainWindow):
    def __init__(self, title, size, icon=None, centralWidget=None):
        QtWidgets.QMainWindow.__init__(self)
        self.setWindowTitle(title)
        self.setMinimumSize(size[0], size[1])
        if icon:
            self.setWindowIcon(icon)
        if centralWidget:
            self.setCentralWidget(centralWidget)


class TSMenu(QtWidgets.QWidget):
    def __init__(self, app, title, icon):
        super(
            TSMenu, self
        ).__init__()  # self.app = app  # self.isPinned = False  # self.pressing = False  # self.setWindowFlags(QtCore.Qt.FramelessWindowHint)  # self.setStyleSheet(  #     qss_format(qss.general.window + qss.general.widget, self.app.palette)  # )  # self.title = title  # self.setWindowTitle(title)  # self.icon = QtGui.QIcon(icon)  # self.setWindowIcon(self.icon)  # self.hbox = QtWidgets.QGridLayout(self)  # self.hbox.addWidget(interface.components.titlebar.TitleBar(self, self.title), 0, 0)  # self.frame, self._layout = self.app.assembler.subdivision(self)  # self.frameS, self._layoutS = self.app.assembler.subdivision(self)  # self.menubar = QtWidgets.QMenuBar()  # self.menubar.setFixedHeight(22)  # self.menubar.setStyleSheet(qss_format(qss.barstyles.menubar, self.app.palette))  # self.hbox.addWidget(self.menubar, 1, 0)  # self.hbox.addWidget(self.frame, 2, 0)  # self.hbox.addWidget(self.frameS, 3, 0)  # self.setLayout(self.hbox)  # self._layout.setContentsMargins(5, 0, 5, 0)  # self._layout.setSpacing(0)  # self._layoutS.setContentsMargins(5, 0, 5, 5)  # self._layoutS.setSpacing(0)  # self.hbox.setContentsMargins(0, 0, 0, 0)  # self.statusbar = self.app.assembler.make_statusbar(self, self._layoutS, 0, 1)  # self.statusbar.setMinimumHeight(self.app.palette.minHeight)  # self.pinButton = self.app.assembler.make_button(  #     self, layout=self._layoutS, command=self.pin, row=0, column=0, width=25, height=25, icon=configuration.pin,  #     tooltip=tooltips.pinButton  # )  # self.app = app  # self.isPinned = False  # self.pressing = False  # self.setWindowFlags(QtCore.Qt.FramelessWindowHint)  # self.setStyleSheet(  #     qss_format(  #         qss.generalstyles.window + qss.generalstyles.widget, self.app.palette  #     )  # )  # self.title = title  # self.setWindowTitle(title)  # self.icon = QtGui.QIcon(icon)  # self.setWindowIcon(self.icon)  # self.hbox = QtWidgets.QGridLayout(self)  # self.hbox.addWidget(lozoya.gui.components.titlebar.TitleBar(self, self.title), 0, 0)  # self.frame, self._layout = self.app.assembler.subdivision(self)  # self.frameS, self._layoutS = self.app.assembler.subdivision(self)  # self.menubar = MenuBar(app)  # self.hbox.addWidget(self.menubar, 1, 0)  # self.hbox.addWidget(self.frame, 2, 0)  # self.hbox.addWidget(self.frameS, 3, 0)  # self.setLayout(self.hbox)  # self._layout.setContentsMargins(5, 0, 5, 0)  # self._layout.setSpacing(0)  # self._layoutS.setContentsMargins(5, 0, 5, 5)  # self._layoutS.setSpacing(0)  # self.hbox.setContentsMargins(0, 0, 0, 0)  # self.statusbar = self.app.assembler.make_statusbar(self, self._layoutS, 0, 1)  # self.statusbar.setMinimumHeight(self.app.palette.minHeight)  # self.pinButton = self.app.assembler.make_button(  #     self, layout=self._layoutS, command=self.pin, row=0, column=0, width=25, height=25, icon=configuration.pin,  #     tooltip=tooltips.pinButton  # )  # self.app = app  # self.isPinned = False  # self.pressing = False  # self.setWindowFlags(QtCore.Qt.FramelessWindowHint)  # self.setStyleSheet(  #     qss_format(  #         qss.generalstyles.window + qss.generalstyles.widget, self.app.palette  #     )  # )  # self.title = title  # self.setWindowTitle(title)  # self.icon = QtGui.QIcon(icon)  # self.setWindowIcon(self.icon)  # self.hbox = QtWidgets.QGridLayout(self)  # self.hbox.addWidget(lozoya.gui.components.titlebar.TitleBar(self, self.title), 0, 0)  # self.frame, self._layout = self.app.assembler.subdivision(self)  # self.frameS, self._layoutS = self.app.assembler.subdivision(self)  # self.menubar = QtWidgets.QMenuBar()  # self.menubar.setFixedHeight(22)  # self.menubar.setStyleSheet(qss_format(qss.barstyles.menubar, self.app.palette))  # self.hbox.addWidget(self.menubar, 1, 0)  # self.hbox.addWidget(self.frame, 2, 0)  # self.hbox.addWidget(self.frameS, 3, 0)  # self.setLayout(self.hbox)  # self._layout.setContentsMargins(5, 0, 5, 0)  # self._layout.setSpacing(0)  # self._layoutS.setContentsMargins(5, 0, 5, 5)  # self._layoutS.setSpacing(0)  # self.hbox.setContentsMargins(0, 0, 0, 0)  # self.statusbar = self.app.assembler.make_statusbar(self, self._layoutS, 0, 1)  # self.statusbar.setMinimumHeight(self.app.palette.minHeight)  # self.pinButton = self.app.assembler.make_button(  #     self, layout=self._layoutS, command=self.pin, row=0, column=0, width=25, height=25, icon=configuration.pin,  #     tooltip=tooltips.pinButton  # )

    # def pin(self):  #     self.isPinned = not self.isPinned  #     if self.isPinned:  #         self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)  #         self.pinButton.set_connected()  #         self.update_status('Window pinned.', 'success')  #     else:  #         self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)  #         self.pinButton.reset()  #         self.update_status('Window unpinned.', 'success')  #     self.show()  #  # def pin(self, *args, **kwargs):  #     self.isPinned = not self.isPinned  #     if self.isPinned:  #         self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)  #         self.pinButton.set_connected()  #         self.update_status(*status.windowPinned)  #     else:  #         self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)  #         self.pinButton.reset()  #         self.update_status(*status.windowUnpinned)  #     self.show()  #  # def update_status(self, msg, status, e=None):  #     if e:  #         msg = msg.format(str(e))  #     try:  #         if status == 'success':  #             stylesheet = qss.statusbar.statusbarSuccess  #         elif status == 'error':  #             stylesheet = qss.statusbar.statusbarError  #             date = self.app.dataDude.date  #             try:  #                 with open(os.path.join(configuration.errorLog, date), 'a') as f:  #                     timestamp = self.app.dataDude.timestampHMSms  #                     _ = '{} {} {}\n'.format(date.replace('_', '/'), timestamp, msg)  #                     print(_)  #                     f.write(_)  #             except:  #                 try:  #                     os.mkdir(configuration.errorLog)  #                 except:  #                     pass  #                 with open(os.path.join(configuration.errorLog, date), 'a') as f:  #                     timestamp = self.app.dataDude.timestampHMSms  #                     _ = '{} {} {}\n'.format(date.replace('_', '/'), timestamp, msg)  #                     print(_)  #                     f.write(_)  #         elif status == 'alert':  #             stylesheet = qss.statusbar.statusbarAlert  #         self.statusbar.showMessage(msg)  #         self.statusbar.adjustSize()  #         self.statusbar.setStyleSheet(qss_format(stylesheet, self.app.palette))  #     except Exception as e:  #         print(e)  #  # def update_status(self, msg, s, e=None):  #     if e:  #         msg = msg.format(str(e))  #     try:  #         if s == 'error':  #             date = self.app.dataDude.date  #             try:  #                 with open(os.path.join(configuration.errorLog, date), 'a+') as f:  #                     timestamp = self.app.dataDude.timestampHMSms  #                     _ = '{} {} {}\n'.format(date.replace('_', '/'), timestamp, msg)  #                     print(_)  #                     f.write(_)  #             except:  #                 try:  #                     os.mkdir(configuration.errorLog)  #                 except:  #                     pass  #                 with open(os.path.join(configuration.errorLog, date), 'a+') as f:  #                     timestamp = self.app.dataDude.timestampHMSms  #                     _ = '{} {} {}\n'.format(date.replace('_', '/'), timestamp, msg)  #                     print(_)  #                     f.write(_)  #         self.statusbar.write(msg, s)  #     except Exception as e:  #         print(e)


class TSPlot(TSFormField, FigureCanvasQTAgg):

    def __init__(self, name='', callback=None, width=5, height=4, dpi=100, projection=None):
        fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111, projection=projection)
        FigureCanvasQTAgg.__init__(self, fig)
        TSFormField.__init__(self, name, callback)
        self.axes.set_facecolor(highlight1)
        self.figure.set_facecolor(background0)

    def clear(self):
        self.axes.cla()
        self.draw()

    def plot(self, x=None, y=None, data=None, color=None, xlabel=None, ylabel=None, clear=True):
        """
        Use x and y or data. If x and y are used, data shall not be used, and vice versa.
        """
        try:
            if clear:
                self.axes.cla()
            if (type(x) != type(None)) and (type(y) != type(None)) and (type(color) != type(None)):
                self.axes.plot(x, y, c=color)
            if (type(x) != type(None)) and (type(y) != type(None)):
                self.axes.plot(x, y)
            elif (type(data) != type(None)):
                print("??")
                self.axes.plot(data)
            self.axes.set_xlabel(xlabel)
            self.axes.set_ylabel(ylabel)
            self.draw()
        except Exception as e:
            print(e)


class TSTable(TSFormField, FigureCanvasQTAgg):
    def __init__(self, name, cellText, rowLabels, rowColors, colLabels, width=None, height=None, dpi=None):
        # fig = plt.Figure(figsize=(width, height), dpi=dpi)
        fig = plt.table(
            cellText=cellText,
            rowLabels=rowLabels,
            rowColours=rowColors,
            colLabels=colLabels,
        )
        FigureCanvasQTAgg.__init__(self, fig)
        TSFormField.__init__(self, name)


class TSStackedWidget(TSFormField, QtWidgets.QStackedWidget):
    def __init__(self, name):
        QtWidgets.QStackedWidget.__init__(self)
        TSFormField.__init__(self, name=name)


class TSText(TSFormField, QtWidgets.QLineEdit):
    def __init__(self, name='', value='', callback=None):
        QtWidgets.QLineEdit.__init__(self)
        TSFormField.__init__(self, name=name)
        self.setText(value)
        if callback:
            self._callback = callback
            self.textChanged.connect(self.callback)

    def callback(self):
        return self._callback(self.text())

    def get_value(self):
        return self.text()


class TSTextArea(TSFormField, QtWidgets.QTextEdit):
    def __init__(self, name):
        QtWidgets.QTextEdit.__init__(self)
        TSFormField.__init__(self, name=name)


class TSTextReadOnly(TSFormField, QtWidgets.QLabel):
    def __init__(self, name, text):
        QtWidgets.QLabel.__init__(self)
        self.setText(text)
        TSFormField.__init__(self, name=name)


class TSToolbar(TSFormField, QtWidgets.QToolBar):
    def __init__(self, name):
        QtWidgets.QToolBar.__init__(self, name)
        TSFormField.__init__(self, name=name)


################################################################################################
def make_action(label, callback=None, shortcut=None, icon=None, tooltip=None):
    action = QtWidgets.QAction(label, checkable=True)
    if callback:
        action.triggered.connect(callback)
    if shortcut:
        action.setShortcut(shortcut)
    if tooltip:
        action.setToolTip(tooltip)
    return action


def make_action_group(parent, actions):
    group = QtWidgets.QActionGroup(parent)
    for action in actions:
        group.addAction(action)
    return group


def make_form(fields, name=''):
    form = TSForm(fields=fields, name=name)
    return form


def make_formfield_from_class(object, attribute, value, callback):
    name = attribute  # ' '.join(re.findall('[A-Z][^A-Z]*', attribute))
    callback = lambda value: object.__setattr__(attribute, value)
    if (type(value) == bool):
        return TSInputCheckbox(name=name, callback=callback, value=value)
    if (type(value) == int):
        return TSInputSpinbox(name=name, callback=callback, value=value)
    if (type(value) == float):
        return TSInputSpinboxDouble(name=name, callback=callback, value=value)
    if (type(value) == str):
        return TSText(name=name, callback=callback, value=value)


def make_form_from_class(template: object, name: str = '', callback: callable = None):
    types = [bool, int, float, str]
    attributes = template.__dict__
    readAttributes = {}
    for key in sorted(attributes.keys()):
        attribute = attributes[key]
        if type(attribute) in types and key[0] != '_':
            readAttributes[key] = attribute
    fields = [make_formfield_from_class(template, attribute, attributes[attribute], callback) for attribute in
              readAttributes]
    form = make_form(fields, name=name)
    return form


def make_forms_from_class(templates: list):
    formsList = []
    for template in templates:
        form = make_form_from_class(template, name=template.__name__)
        formsList.append(form)
    return formsList


def make_toolbar(toolbarName, actions):
    toolbar = TSToolbar(toolbarName)
    toolbar.name = toolbarName
    toolbar.setMovable(False)
    for action in actions:
        toolbar.addAction(action)
        toolbar.addSeparator()
    actions[0].setChecked(True)
    return toolbar


def qss_format(style, element):
    pass
