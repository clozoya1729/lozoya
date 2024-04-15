import ctypes
import datetime
import os
from PyQt5 import QtGui, QtWidgets

SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
current_year = datetime.datetime.now().year - 1
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
FONT_SIZE = 9
FONT = 'Arial'
topPadding = 15
bottomPadding = 10
ROOT = os.getcwd()
BASE_MAP = os.path.join(ROOT, 'map_api', 'baseMap.html')
GRAPHS = os.path.join(ROOT, 'graph2')
BASE_GRAPH = os.path.join(GRAPHS, 'RegressionPlot.html')
CLASSIFICATION_GRAPH = os.path.join(GRAPHS, 'ClassificationPlot.html')
COUNT_GRAPH = os.path.join(GRAPHS, 'CountPlot.html')
MARKOV_CHAIN_GRAPH = os.path.join(GRAPHS, 'MarkovChainPlot2.html')
MARKOV_CHAIN_FREQUENCY_GRAPH = os.path.join(GRAPHS, 'MarkovChainFrequencyPlot.html')
MARKOV_CHAIN_FREQUENCY_CURVES_GRAPH = os.path.join(GRAPHS, 'MarkovChainFrequencyCurvesPlot.html')
MONTE_CARLO_GRAPH = os.path.join(GRAPHS, 'MonteCarloPlot.html')
REGRESSION_GRAPH = os.path.join(GRAPHS, 'RegressionPlot.html')
SEARCH_RESULTS_TABLE = os.path.join(GRAPHS, 'SearchResults2.html')
DATABASE = os.path.join(ROOT, 'Database')
ICONS = os.path.join(ROOT, 'interface7', 'InterfaceUtilities0', 'icon2')
STATES = os.path.join(DATABASE, 'Information', 'allStates.txt')
STATE_CODES = os.path.join(DATABASE, 'Information', 'allStateCodes.txt')
DATABASE_VARIABLES = os.path.join(ROOT, 'interface7', 'InterfaceUtilities0', 'label', 'Database Variables.txt')
VARIABLES = os.path.join(ROOT, 'Utilities14', 'Variables.py')
temp_RSP = os.path.join(ROOT, 'Utilities14', 'Preferences', 'Report_Preferences0.txt')
RSP = os.path.join(ROOT, 'Utilities14', 'Preferences', 'Report_Preferences0.txt')
TEMP_SAVE = os.path.join(DATABASE, 'temp.txt')
DOCUMENTATION = os.path.join(ROOT, '1', '1 sunburst.html')
UTILITIES = os.path.join(ROOT, 'Utilities14')
TRANSITION_MATRIX = os.path.join(GRAPHS, 'TransitionMatrix.html')
ROOT = os.getcwd()
REGRESSOR_GRAPH = os.path.join(GRAPHS, 'RegressorPlot.html')
DATABASE = os.path.join(ROOT, 'Database')
GRAPHS = os.path.join(ROOT, 'graph2')


def add_action(parent, command, text, icon, description, shortcut=None):
    """
    parent: QWidget
    icon2: str (file path)
    description: str
    shortcut: str
    return:
    """
    qIcon = QtGui.QIcon(icon)
    action = QtWidgets.QAction(qIcon, description, parent)
    if shortcut:
        action.setShortcut(shortcut)
    action.triggered.connect(command)
    action.setIconText(text)
    return action


def add_action(parent, command=None, text='', icon=None, description=None, shortcut=None):
    """
    parent: QWidget
    icon2: str (file path)
    description: str
    shortcut: str
    return:
    """
    qIcon = QtGui.QIcon(icon)
    action = QtWidgets.QAction(qIcon, '', parent)
    if shortcut != None:
        action.setShortcut(shortcut)
    if command != None:
        action.triggered.connect(command)
    action.setIconText(text)
    action.setToolTip(description)
    return action


def add_toolbar(parent, command=None, text='', icon=None, description=None, shortcut=None):
    """
    parent: QWidget
    icon2: str (file path)
    description: str
    shortcut: str
    return:
    """
    toolbar = QtWidgets.QToolBar(parent)
    toolbar.setStyleSheet(qss.TOOLBAR_STYLE)
    toolbar.setIconSize(QtCore.QSize(150, 150))
    toolbar.setFixedSize(400, 400)
    toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
    qIcon = QtGui.QIcon(icon)
    action = QtWidgets.QAction(qIcon, '', parent)
    if shortcut != None:
        action.setShortcut(shortcut)
    if command != None:
        action.triggered.connect(command)
    action.setIconText(text)
    action.setToolTip(description)
    toolbar.addAction(action)
    parent.addToolBar(toolbar)
    return None


def clear(entries):
    for entry in entries:
        entries[entry].setText('')


def clear(entries, FEATURES):
    for i in range(entries.__len__()):
        entries[FEATURES[i]].setText('')


def get_item(d):
    item = QtWidgets.QTableWidgetItem(d)
    item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
    return item


def insert_file_dialog(widget, container, icon, text, row, column, tooltipButton, tooltipLabel):
    label, button = make_file_dialog(
        widget=widget, icon=icon, text=text, tooltipButton=tooltipButton,
        tooltipLabel=tooltipLabel, )
    container.addWidget(button, row, column, )
    container.addWidget(label, row, column + 1, )
    return label, button


def load(self, *args, **kwargs):
    status = getattr(self.configuration.py, self.name)
    try:
        if not getattr(self.configuration.py, '{}Valid'.format(self.name[:-4])):  # remove 'Lock' from name by slicing
            status = 'disabled'
        else:
            status = 'locked' if status else 'unlocked'
    except:
        status = 'locked' if status else 'unlocked'
    self.set_style(status)
    self.set_pair_style(status)


def load_browser(browser, file, online=False):
    try:
        if online:
            browser.load(QtCore.QUrl(file))
        else:
            browser.load(QtCore.QUrl.fromLocalFile(file))
    except:
        pass
    return browser


def make_browser(layout, file, online=False):
    browser = QtWebEngineWidgets.QWebEngineView()
    if file:
        load_browser(browser, file, online)
    if layout != None:
        layout.addWidget(browser)
    return browser


def make_browser(layout, file):
    browser = QtWebEngineWidgets.QWebEngineView()
    if file:
        browser.load(QtCore.QUrl.fromLocalFile(file))
    if layout:
        layout.addWidget(browser)
    return browser


def make_browser(layout, file, online=False):
    browser = QtWebEngineWidgets.QWebEngineView()
    if file:
        load_browser(browser, file, online)
    if layout:
        layout.addWidget(browser)
    return browser


def make_button(
    parent, buttonType=None, command=None, text='', icon=None, layout=None, row=0, column=0, width=None,
    description=None
):
    if buttonType == 'radio':
        button = QtWidgets.QRadioButton(parent)
    elif buttonType == 'check':
        button = QtWidgets.QCheckBox(parent)
    else:
        if icon:
            icn = QtGui.QIcon(icon)
            button = QtWidgets.QPushButton(icn, text, parent)
            button.setIconSize(QtCore.QSize(20, 20))
        else:
            button = QtWidgets.QPushButton(text, parent)
        button.setFont(QtGui.QFont('Verdana', 9))
        button.setStyleSheet(qss.BUTTON_STYLE)
        set_widget_size(button, width if width else 25, 25)
    if command != None:
        button.clicked.connect(command)
    if description:
        button.setToolTip(description)
    if layout:
        layout.addWidget(button, row, column)
    return button


def make_button(
    parent, buttonType=None, command=None, text='', icon=None, layout=None, row=None, column=None,
    description=None
):
    if buttonType == 'radio':
        button = QtWidgets.QRadioButton(parent)
    elif buttonType == 'check':
        button = QtWidgets.QCheckBox(parent)
    else:
        if icon:
            icn = QtGui.QIcon(icon)
            button = QtWidgets.QPushButton(icn, text, parent)
            button.setIconSize(QtCore.QSize(20, 20))
        else:
            button = QtWidgets.QPushButton(text, parent)
        button.setFont(QtGui.QFont('Verdana', 9))
        button.setStyleSheet(qss.BUTTON_STYLE)
        set_widget_size(button, 25, 25)
        button.clicked.connect(command)
    if description:
        button.setToolTip(description)
    if layout:
        layout.addWidget(button, row, column)
    return button


def make_button(
    parent, buttonType=None, command=None, text='', icon=None, layout=None, row=0, column=0, width=None,
    description=None
):
    if buttonType == 'radio':
        button = QtWidgets.QRadioButton(parent)
    elif buttonType == 'check':
        button = QtWidgets.QCheckBox(parent)
    else:
        if icon:
            icn = QtGui.QIcon(icon)
            button = QtWidgets.QPushButton(icn, text, parent)
            button.setIconSize(QtCore.QSize(20, 20))
        else:
            button = QtWidgets.QPushButton(text, parent)

        button.setFont(QtGui.QFont('Verdana'))
        button.setStyleSheet(qss.BUTTON_STYLE)
        set_widget_size(button, width if width else 25, 25)
    if command != None:
        button.clicked.connect(command)
    if description:
        button.setToolTip(description)
    if layout:
        layout.addWidget(button, row, column)
    return button


def make_button(
    parent=None, buttonType=None, command=None, text='', icon=None, layout=None, row=0, column=0,
    width=None, description=None
):
    if buttonType == 'radio':
        button = QtWidgets.QRadioButton(parent)
    elif buttonType == 'check':
        button = QtWidgets.QCheckBox(parent)
    else:
        if icon:
            icn = QtGui.QIcon(icon)
            button = QtWidgets.QPushButton(icn, text, parent)
            button.setIconSize(QtCore.QSize(20, 20))
        else:
            button = QtWidgets.QPushButton(text, parent)
        button.setFont(QtGui.QFont(FONT))
        button.setStyleSheet(qss.BUTTON_STYLE)
        set_widget_size(button, width if width else 25, 25)
    if command != None:
        button.clicked.connect(command)
    if description != None:
        button.setToolTip(description)
    if layout != None:
        layout.addWidget(button, row, column)
    button.adjustSize()
    return button


def make_button(
    parent=None, buttonType=None, command=None, text='', icon=None, layout=None, row=0, column=0,
    width=None, description=None, isChecked=False, ):
    if buttonType == 'radio':
        button = QtWidgets.QRadioButton(parent)
        button.setStyleSheet(qss.RADIO_STYLE)
    elif buttonType == 'check':
        button = QtWidgets.QCheckBox(parent)
        if isChecked:
            button.setChecked(True)
        button.setStyleSheet(qss.CHECK_STYLE)
    else:
        if icon:
            icn = QtGui.QIcon(icon)
            button = QtWidgets.QPushButton(icn, text, parent)
            button.setIconSize(QtCore.QSize(20, 20))
        else:
            button = QtWidgets.QPushButton(text, parent)
        button.setStyleSheet(
            qss.BUTTON_STYLE
        )  # button.setvariables.FONT(QtGui.Qvariables.FONT(variables.FONT))  # set_widget_size(button, width if width else 25, 25)
    if command != None:
        button.clicked.connect(command)
    if description != None:
        button.setToolTip(description)
    if layout != None:
        if type(layout) == QtWidgets.QGridLayout:
            layout.addWidget(button, row, column)
        else:
            layout.addWidget(button)
    button.adjustSize()
    if width:
        button.setFixedWidth(width)
    return button


def make_button(parent, command, text='', icon=None):
    if icon:
        icn = QtGui.QIcon(icon)
        button = QtWidgets.QPushButton(icn, text, parent)
        button.setIconSize(QtCore.QSize(20, 20))
    else:
        button = QtWidgets.QPushButton(text, parent)

    button.setFont(QtGui.QFont('Verdana', 9))
    button.setStyleSheet(BUTTON_STYLE)
    menu.set_widget_size(button, 25, 25)
    button.clicked.connect(command)
    return button


def make_combo(
    parent=None, items=None, command=None, layout=None, row=0, column=0, width=40, height=20,
    description=None
):
    combo = QtWidgets.QComboBox(parent)
    set_widget_size(combo, width, height)
    combo.setStyleSheet(qss.COMBO_STYLE)
    combo.addItems(items)
    if command != None:
        combo.activated[str].connect(command)
    if description != None:
        combo.setToolTip(description)
    if layout != None:
        if type(layout) == QtWidgets.QGridLayout:
            layout.addWidget(combo, row, column)
        else:
            layout.addWidget(combo)
    combo.adjustSize()
    return combo


def make_combo(parent, items, command):
    combo = QtWidgets.QComboBox(parent)
    set_widget_size(combo, 40, 20)
    combo.setStyleSheet(qss.COMBO_STYLE)
    for item in items:
        combo.addItem(item)
    combo.activated[str].connect(command)
    return combo


def make_combo(parent, items, command, layout=None, row=None, column=None):
    combo = QtWidgets.QComboBox(parent)
    set_widget_size(combo, 40, 20)
    combo.setStyleSheet(qss.COMBO_STYLE)
    for item in items:
        combo.addItem(item)
    combo.activated[str].connect(command)
    if layout:
        layout.addWidget(combo, row, column)
    return combo


def make_combo(parent, items, command=None, layout=None, row=0, column=0, width=40, height=20, description=None):
    combo = QtWidgets.QComboBox(parent)
    set_widget_size(combo, width, height)
    combo.setStyleSheet(qss.COMBO_STYLE)
    combo.addItems(items)
    if command:
        combo.activated[str].connect(command)
    if description:
        combo.setToolTip(description)
    if layout:
        layout.addWidget(combo, row, column)
    combo.adjustSize()
    return combo


def make_combo(parent, items, command=None, layout=None, row=None, column=None, width=40, height=20, description=None):
    combo = QtWidgets.QComboBox(parent)
    set_widget_size(combo, width, height)
    combo.setStyleSheet(qss.COMBO_STYLE)
    for item in items:
        combo.addItem(item)
    if command:
        combo.activated[str].connect(command)
    if description:
        combo.setToolTip(description)
    if layout:
        layout.addWidget(combo, row, column)
    return combo


def make_dial(parent, min, max, value=None, step=1, layout=None, row=0, column=0, description=None):
    dial = QtWidgets.QDial(parent)
    dial.setStyleSheet(qss.DIAL_STYLE)
    set_widget_size(dial, 30, 30)
    dial.setMinimum(min)
    dial.setMaximum(max)
    dial.setNotchesVisible(True)
    dial.setSingleStep(step)
    dial.setWrapping(False)
    if not value:
        dial.setValue(min)
    else:
        dial.setValue(value)
    if description:
        dial.setToolTip(description)
    if layout:
        layout.addWidget(dial, row, column)
    return dial


def make_dial(parent=None, min=None, max=None, value=None, step=1, layout=None, row=0, column=0, description=None):
    dial = QtWidgets.QDial(parent)
    dial.setStyleSheet(qss.DIAL_STYLE)
    set_widget_size(dial, 30, 30)
    dial.setMinimum(min)
    dial.setMaximum(max)
    dial.setNotchesVisible(True)
    dial.setSingleStep(step)
    dial.setWrapping(False)
    if not value:
        dial.setValue(min)
    else:
        dial.setValue(value)
    if description != None:
        dial.setToolTip(description)
    if layout != None:
        layout.addWidget(dial, row, column)
    return dial


def make_dial(
    parent=None, min=None, max=None, value=None, step=1, layout=None, row=0, column=0,
    description=None
):
    dial = QtWidgets.QDial(parent)
    dial.setStyleSheet(qss.DIAL_STYLE)
    set_widget_size(dial, 30, 30)
    dial.setMinimum(min)
    dial.setMaximum(max)
    dial.setNotchesVisible(True)
    dial.setSingleStep(step)
    dial.setWrapping(False)
    if not value:
        dial.setValue(min)
    else:
        dial.setValue(value)
    if description != None:
        dial.setToolTip(description)
    if layout != None:
        if type(layout) == QtWidgets.QGridLayout:
            layout.addWidget(dial, row, column)
        else:
            layout.addWidget(dial)
    return dial


def make_dial(
    parent=None, min=None, max=None, value=None, step=1, layout=None, row=0, column=0,
    description=None, updateFunction=None
):
    dial = QtWidgets.QDial(parent)
    dial.setStyleSheet(qss.DIAL_STYLE)
    set_widget_size(dial, 30, 30)
    dial.setMinimum(min)
    dial.setMaximum(max)
    dial.setNotchesVisible(True)
    dial.setSingleStep(step)
    dial.setWrapping(False)
    if not value:
        dial.setValue(min)
    else:
        dial.setValue(value)
    if description != None:
        dial.setToolTip(description)
    if updateFunction:
        dial.valueChanged.connect(updateFunction)
    if layout != None:
        if type(layout) == QtWidgets.QGridLayout:
            layout.addWidget(dial, row, column)
        else:
            layout.addWidget(dial)
    return dial


def make_entry(parent, text='', width=150, layout=None, row=0, column=0, description=None):
    entry = QtWidgets.QLineEdit(parent)
    entry.setFont(QtGui.QFont(FONT))
    entry.setStyleSheet(qss.ENTRY_STYLE)
    set_widget_size(entry, width, 20)
    if description:
        entry.setToolTip(description)
    if layout != None:
        if type(layout) == QtWidgets.QGridLayout:
            layout.addWidget(entry, row, column)
        else:
            layout.addWidget(entry)
    return entry


def make_entry(
    parent, text='', width=150, height=30, layout=None, row=0, column=0, description=None,
    updateFunction=None, number=False, readOnly=False, area=False, stylesheet=None
):
    if area:
        entry = QtWidgets.QPlainTextEdit(parent)
        if text:
            entry.setPlainText(text)

    else:
        entry = QtWidgets.QLineEdit(parent)
        if text:
            entry.setText(text)
    # if number:
    #     entry.setValidator(QtGui.QIntValidator())
    # if updateFunction:
    #     entry.textChanged.connect(updateFunction)
    # entry.setvariables.FONT(QtGui.Qvariables.FONT(variables.FONT))
    entry.setStyleSheet(qss.ENTRY_STYLE)
    if width:
        entry.setFixedWidth(width)
    if height:
        entry.setFixedHeight(height)

    if readOnly:
        entry.setReadOnly(True)
    if description:
        entry.setToolTip(description)
    if layout != None:
        if type(layout) == QtWidgets.QGridLayout:
            layout.addWidget(entry, row, column)
        else:
            layout.addWidget(entry)
    if stylesheet:
        entry.setStyleSheet(stylesheet)
    return entry


def make_entry(parent, width=300):
    entry = QtWidgets.QLineEdit(parent)
    entry.setFont(QtGui.QFont("Verdana", 9))
    entry.setStyleSheet(qss.ENTRY_STYLE)
    set_widget_size(entry, width, 20)
    return entry


def make_entry(parent, width=300, layout=None, row=None, column=None):
    entry = QtWidgets.QLineEdit(parent)
    entry.setFont(QtGui.QFont("Verdana", 9))
    entry.setStyleSheet(qss.ENTRY_STYLE)
    set_widget_size(entry, width, 20)
    if layout:
        layout.addWidget(entry, row, column)
    return entry


def make_entry(parent, text='', width=150, layout=None, row=0, column=0, description=None):
    entry = QtWidgets.QLineEdit(parent)
    entry.setFont(QtGui.QFont(FONT))
    entry.setStyleSheet(qss.ENTRY_STYLE)
    set_widget_size(entry, width, 20)
    if description:
        entry.setToolTip(description)
    if layout:
        layout.addWidget(entry, row, column)
    return entry


def make_entry(parent, text='', width=150, layout=None, row=0, column=0, description=None):
    entry = QtWidgets.QLineEdit(parent)
    entry.setFont(QtGui.QFont("Verdana"))
    entry.setStyleSheet(qss.ENTRY_STYLE)
    set_widget_size(entry, width, 20)
    if description:
        entry.setToolTip(description)
    if layout:
        layout.addWidget(entry, row, column)
    return entry


def make_entry(parent, text='', width=300, layout=None, row=None, column=None, description=None):
    entry = QtWidgets.QLineEdit(parent)
    entry.setFont(QtGui.QFont("Verdana", 9))
    entry.setStyleSheet(qss.ENTRY_STYLE)
    set_widget_size(entry, width, 20)
    if description:
        entry.setToolTip(description)
    if layout:
        layout.addWidget(entry, row, column)
    return entry


def make_entry(parent, text='', width=150, bub4in=None, row=0, column=0, description=None, display=False, val=None):
    entry = QtWidgets.QLineEdit(parent)
    entry.setFont(QtGui.QFont(FONT))
    entry.setStyleSheet(qss.ENTRY_STYLE)
    set_widget_size(entry, width, 20)
    if description:
        entry.setToolTip(description)
    if bub4in != None:
        if type(bub4in) == QtWidgets.QGridLayout:
            bub4in.addWidget(entry, row, column)
        else:
            bub4in.addWidget(entry)
    entry.setReadOnly(display)
    if len(str(val)) > 0:
        entry.setText(str(val))
    return entry


def make_entry(parent, width=300):
    entry = QtWidgets.QLineEdit(parent)
    entry.setFont(QtGui.QFont("Verdana", 9))
    entry.setStyleSheet(ENTRY_STYLE)
    menu.set_widget_size(entry, width, 20)
    return entry


def make_file_dialog(widget, icon=None, text='', tooltipButton='', tooltipLabel=''):
    label = make_label(widget, text=text, width=200, )
    command = lambda: open_folder(widget, label=label, updateFunction=widget.rewindow, )
    button = make_button(widget, command=command, icon=icon, width=50, )
    button.setToolTip(tooltipButton)
    label.setToolTip(tooltipLabel.format(label.text()))
    return label, button


def make_form(parent=None, padding=False):
    form = QtWidgets.QFormLayout(parent)
    if padding:
        form.setContentsMargins(0, topPadding, 0, bottomPadding)
    return form


def make_frame(parent):
    frame = QtWidgets.QFrame(parent)
    frame.setStyleSheet(qss.FRAME_STYLE)
    frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
    return frame


def make_frame(parent):
    frame = QtWidgets.QFrame(parent)
    frame.setStyleSheet(qss.FRAME_STYLE)
    frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
    return frame


def make_frame(parent):
    frame = QtWidgets.QFrame(parent)
    frame.setStyleSheet(FRAME_STYLE)
    frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
    return frame


def make_form(parent=None, padding=False):
    form = QtWidgets.QFormLayout(parent)
    if padding:
        form.setContentsMargins(0, 25, 0, 0)
    return form


def make_grid(parent=None):
    grid = QtWidgets.QGridLayout(parent)
    return grid


def make_grid(parent=None, padding=False):
    grid = QtWidgets.QGridLayout(parent)
    if padding:
        grid.setContentsMargins(0, 25, 0, 25)
    return grid


def make_grid(parent):
    grid = QtWidgets.QGridLayout(parent)
    return grid


def make_group_box(parent, layout, text, row, column, width, height):
    groupBox = QtWidgets.QGroupBox(text)
    groupBox.setStyleSheet(qss.GROUP_BOX_STYLE)
    groupBox.setLayout(layout)
    parent.addWidget(groupBox, row, column)
    groupBox.setMaximumWidth(width)
    groupBox.setMaximumHeight(height)
    return groupBox.layout()


def make_group_box(parent, layout, text, row, column, width, height, border=True):
    groupBox = QtWidgets.QGroupBox(text)
    groupBox.setStyleSheet(qss.GROUP_BOX_STYLE + '' if border else 'QGroupBox {border: 0;}')
    groupBox.setLayout(layout)
    parent.addWidget(groupBox, row, column)
    groupBox.setMaximumWidth(width)
    groupBox.setMaximumHeight(height)
    return groupBox.layout()


def make_group_box(
    parent, layout, text, row, column, width=None, height=None, border=True, strictWidth=False,
    strictHeight=False, strictSize=False
):
    groupBox = QtWidgets.QGroupBox(text)
    groupBox.setStyleSheet(qss.GROUP_BOX_STYLE + '' if border else 'QGroupBox {border: 0;}')
    groupBox.setLayout(layout)
    parent.addWidget(groupBox, row, column)
    if strictWidth:
        groupBox.setFixedWidth(width)
    else:
        groupBox.setMaximumWidth(width)
    if strictHeight:
        groupBox.setFixedHeight(height)
    else:
        groupBox.setMaximumHeight(height)
    if strictSize:
        groupBox.setFixedSize(width, height)
    return groupBox.layout()


def make_grid(parent=None, padding=False):
    grid = QtWidgets.QGridLayout(parent)
    if padding:
        grid.setContentsMargins(0, topPadding, 0, bottomPadding)
    return grid


def make_int_input(widget, text, width, layout, row, column):
    _, x = QtWidgets.QInputDialog.getInt(widget, '', '')
    layout.addWidget(x, row, column)
    return _


def make_int_input(
    parent, text='', inputSettings=(0, 1, 0, 0), width=150, layout=None, row=0, column=0,
    description=None, updateFunction=None, number=False
):
    entry = QtWidgets.QSpinBox(parent)
    entry.setRange(inputSettings[0], inputSettings[1])
    # set_widget_size(entry, width, 20)
    entry.setValue(inputSettings[2])
    entry.setSingleStep(inputSettings[3])
    if width:
        entry.setFixedWidth(width)
    if updateFunction:
        entry.valueChanged.connect(updateFunction)
    if layout != None:
        if type(layout) == QtWidgets.QGridLayout:
            layout.addWidget(entry, row, column)
        else:
            layout.addWidget(entry)
    return entry


def make_label(parent, text=''):
    label = QtWidgets.QLabel(text, parent)
    label.setFont(QtGui.QFont("Verdana", 9))
    label.setStyleSheet(qss.LABEL_STYLE)
    return label


def make_label(parent, text='', layout=None, row=None, column=None):
    label = QtWidgets.QLabel(text, parent)
    if layout != None:
        layout.addWidget(label, row, column)
    label.setFont(QtGui.QFont("Verdana", 9))
    label.setStyleSheet(qss.LABEL_STYLE)
    return label


def make_label(parent, text='', layout=None, row=None, column=None, width=100, description=None):
    label = QtWidgets.QLabel(text, parent)
    set_widget_size(label, width, 20)
    if description:
        label.setToolTip(description)
    if layout != None:
        layout.addWidget(label, row, column)
    label.setFont(QtGui.QFont("Verdana", 9))
    label.setStyleSheet(qss.LABEL_STYLE)
    return label


def make_label(parent, text='', layout=None, row=0, column=0, width=None, height=None, description=None):
    label = QtWidgets.QLabel(text, parent)
    label.setWordWrap(True)
    if width and height:
        set_widget_size(label, width, height)
    else:
        label.setMaximumWidth(150)
    if description:
        label.setToolTip(description)
    if layout != None:
        if type(layout) == QtWidgets.QGridLayout:
            layout.addWidget(label, row, column)
        else:
            layout.addWidget(label)
    label.setFont(QtGui.QFont(FONT))
    label.setStyleSheet(qss.LABEL_STYLE)
    label.adjustSize()
    return label


def make_label(parent, text='', layout=None, row=0, column=0, width=None, height=None, description=None):
    label = QtWidgets.QLabel(text, parent)
    label.setWordWrap(True)
    if width and height:
        set_widget_size(label, width, height)
    elif width:
        label.setFixedWidth(width)
    else:
        label.setMaximumWidth(150)
    if description:
        label.setToolTip(description)
    if layout != None:
        if type(layout) == QtWidgets.QGridLayout:
            layout.addWidget(label, row, column)
        else:
            layout.addWidget(label)
    # label.setvariables.FONT(QtGui.Qvariables.FONT(variables.FONT))
    label.setStyleSheet(qss.LABEL_STYLE)
    label.adjustSize()
    return label


def make_label(parent, text='', layout=None, row=0, column=0, width=None, height=None, description=None):
    label = QtWidgets.QLabel(text, parent)
    label.setWordWrap(True)
    if width and height:
        set_widget_size(label, width, height)
    else:
        label.setMaximumWidth(150)
    if description:
        label.setToolTip(description)
    if layout != None:
        layout.addWidget(label, row, column)
    label.setFont(QtGui.QFont(FONT))
    label.setStyleSheet(qss.LABEL_STYLE)
    label.adjustSize()
    return label


def make_list(parent=None, items=None, command=None, layout=None, row=None, column=None, width=None):
    list = QtWidgets.QListWidget(parent)
    list.addItems(items)
    if command:
        list.currentItemChanged.connect(command)
    if layout != None:
        layout.addWidget(list, row, column)
    if width:
        list.setMaximumWidth(width)
    return list


def make_list(
    parent=None, items=None, command=None, default=None, layout=None, row=0, column=0, width=None,
    height=None, multiSelect=False
):
    list = QtWidgets.QListWidget(parent)
    list.addItems(items)
    list.setStyleSheet(qss.LIST_STYLE)
    if default != None:
        list.item(default).setSelected(True)
    if command:
        list.currentItemChanged.connect(command)
    if layout != None:
        layout.addWidget(list, row, column)
    if width:
        list.setMaximumWidth(width)
    if height:
        list.setMaximumHeight(height)
    if multiSelect:
        list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
    return list


def make_list(
    parent=None, items=None, command=None, default=None, layout=None, row=0, column=0, width=None,
    height=None, multiSelect=False
):
    list = QtWidgets.QListWidget(parent)
    list.addItems(items)
    list.setStyleSheet(qss.LIST_STYLE)
    if default != None:
        list.item(default).setSelected(True)
    if command:
        list.currentItemChanged.connect(command)
    if layout != None:
        if type(layout) == QtWidgets.QGridLayout:
            layout.addWidget(list, row, column)
        else:
            layout.addWidget(list)
    if width:
        list.setMaximumWidth(width)
    if height:
        list.setMaximumHeight(height)
    if multiSelect:
        list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
    return list


def make_pair(
    parent=None, pair=None, text=None, layout=None, row=0, column=0, labelWidth=None, labelHeight=None,
    comboItems=None, pairCommand=None, pairText=None, pairWidth=None, pairHeight=None,
    dialSettings=(None, None, None, None), description='', label=False
):
    if label:
        label = make_label(parent, text, layout, row, column, labelWidth, labelHeight, description)
        lay = layout
    else:
        grid = make_group_box(
            parent=layout, layout=make_form(padding=True), text=text, row=row, column=column,
            width=150, height=50, border=False
        )
        lay = grid
    if pair == 'check':
        pair = make_button(parent, 'check', pairCommand, '', None, lay, row, column + 1, pairWidth, description)
    if pair == 'combo':
        pair = make_combo(parent, comboItems, pairCommand, lay, row, column + 1, pairWidth, pairHeight, description)
    if pair == 'dial':
        pair = make_dial(
            parent, dialSettings[0], dialSettings[1], dialSettings[2], dialSettings[3], lay, row,
            column + 1, description
        )
    if pair == 'entry':
        pair = make_entry(parent, pairText, pairWidth, lay, row, column + 1, description)
    if pair == 'radio':
        pair = make_button(parent, 'radio', pairCommand, '', None, lay, row, column + 1, pairWidth, description)
    return None, pair


def make_pair(
    parent=None, pair=None, text=None, layout=None, row=0, column=0, labelWidth=None, labelHeight=None,
    comboItems=None, pairCommand=None, pairText=None, pairWidth=None, pairHeight=None,
    dialSettings=(None, None, None, None), description='', label=False, pairKwargs={}
):
    if label:
        label = make_label(parent, text, layout, row, column, labelWidth, labelHeight, description)
        lay = layout
    else:
        grid = make_group_box(
            layout, layout=make_form(padding=True), text=text, row=row, column=column, width=150,
            height=50, border=False
        )
        lay = grid
    if pair == 'check':
        pair = make_button(parent, 'check', **pairKwargs)
    if pair == 'combo':
        pair = make_combo(parent, comboItems, pairCommand, lay, row, column + 1, pairWidth, pairHeight, description)
    if pair == 'dial':
        pair = make_dial(
            parent, dialSettings[0], dialSettings[1], dialSettings[2], dialSettings[3], lay, row,
            column + 1, description, pairCommand
        )
    if pair == 'slider':
        pair = make_slider(
            parent, dialSettings[0], dialSettings[1], dialSettings[2], dialSettings[3], lay, row,
            column + 1, description, pairCommand, width=pairWidth, height=pairHeight
        )
    if pair == 'entry':
        pair = make_entry(parent, pairText, pairWidth, lay, row, column + 1, description, pairCommand)
    if pair == 'radio':
        pair = make_button(parent, 'radio', pairCommand, '', None, lay, row, column + 1, pairWidth, description)
    return None, pair


def make_pair(
    parent=None, pair=None, text=None, layout=None, row=0, column=0, labelWidth=None, labelHeight=None,
    comboItems=None, pairCommand=None, pairText=None, pairWidth=None, pairHeight=None,
    dialSettings=(None, None, None, None), description=''
):
    label = make_label(parent, text, layout, row, column, labelWidth, labelHeight, description)
    if pair == 'check':
        pair = make_button(parent, 'check', pairCommand, '', None, layout, row, column + 1, pairWidth, description)
    if pair == 'combo':
        pair = make_combo(parent, comboItems, pairCommand, layout, row, column + 1, pairWidth, pairHeight, description)
    if pair == 'dial':
        pair = make_dial(
            parent, dialSettings[0], dialSettings[1], dialSettings[2], dialSettings[3], layout, row,
            column + 1, description
        )
    if pair == 'entry':
        pair = make_entry(parent, pairText, pairWidth, layout, row, column + 1, description)
    if pair == 'radio':
        pair = make_button(parent, 'radio', pairCommand, '', None, layout, row, column + 1, pairWidth, description)
    return label, pair


def make_progress_bar(layout):
    p = QtWidgets.QProgressBar()
    layout.addWidget(p)
    p.setGeometry(30, 40, 200, 25)
    p.show()
    return p


def make_progress_bar(layout=None):
    p = QtWidgets.QProgressBar()
    if layout != None:
        layout.addWidget(p)
    p.setStyleSheet(qss.PROGRESS_BAR_STYLE)
    return p


def make_scroll_area(widget):
    scrollArea = QtWidgets.QScrollArea()
    scrollArea.setFrameShape(QtWidgets.QScrollArea.StyledPanel)
    scrollArea.setStyleSheet(qss.FRAME_STYLE + qss.SCROLL_STYLE)
    scrollArea.setWidget(widget)
    return scrollArea


def make_scroll_area(layout, width=None):
    widget = QtWidgets.QWidget()
    widget.setStyleSheet(qss.WIDGET_STYLE)
    if layout != None:
        widget.setLayout(layout)
    scrollArea = QtWidgets.QScrollArea()
    scrollArea.setFrameShape(QtWidgets.QScrollArea.StyledPanel)
    scrollArea.setWidgetResizable(True)
    scrollArea.setStyleSheet(qss.FRAME_STYLE + qss.SCROLL_STYLE)
    scrollArea.setWidget(widget)
    if width:
        scrollArea.setMaximumWidth(width)
    return scrollArea


def make_slider(
    parent=None, min=None, max=None, value=None, step=1, layout=None, row=0, column=0, description=None,
    updateFunction=None, width=None, height=None, inputSettings=(0, 0, 0, 0), ):
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    # slider.setStyleSheet(DIAL_STYLE)
    # set_widget_size(slider, 30, 30)
    if inputSettings:
        slider.setRange(inputSettings[0], inputSettings[1])
        slider.setSingleStep(inputSettings[3])
    else:
        slider.setMinimum(min)
        slider.setMaximum(max)
        slider.setSingleStep(step)
    slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
    if width:
        slider.setFixedWidth(width)
    if height:
        slider.setFixedHeight(height)
    # slider.setWrapping(False)
    if not value and min:
        slider.setValue(min)
    elif not value and inputSettings:
        slider.setValue(inputSettings[2])
    else:
        slider.setValue(value)
    if description != None:
        slider.setToolTip(description)
    if updateFunction:
        slider.valueChanged.connect(updateFunction)
    if layout != None:
        if type(layout) == QtWidgets.QGridLayout:
            layout.addWidget(slider, row, column)
        else:
            layout.addWidget(slider)
    return slider


def make_splitter(style, widgets):
    if style == 'h' or style == 'horizontal':
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    else:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
    for widget in widgets:
        splitter.addWidget(widget)
    splitter.setStyleSheet(qss.FRAME_STYLE + qss.SPLITTER_STYLE)
    QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    return splitter


def make_status_bar(layout, progressBar=False):
    s = QtWidgets.QStatusBar()
    if layout != None:
        layout.addWidget(s)
    if progressBar:
        p = make_progress_bar()
        s.addPermanentWidget(p)
        return s, p
    return s


def make_subdivision(parent):
    subdivision = make_frame(parent)
    layout = make_grid(subdivision)
    return subdivision, layout


def make_tab_area():
    tab = QtWidgets.QTabWidget()
    tab.setStyleSheet(qss.TAB_STYLE)
    return tab


def make_tab(parent, text, master=None, layout=None):
    tab = QtWidgets.QScrollArea()
    tab.setWidget(QtWidgets.QWidget())
    tab.setStyleSheet(qss.WIDGET_STYLE)
    tabLayout = make_grid(tab.widget())
    tab.setWidgetResizable(True)
    master.addTab(tab, text)
    return tabLayout


def menu_bar(parent):
    menubar = QtWidgets.QMenuBar(parent)
    # menubar.setStyleSheet(MENUBAR_STYLE)
    fileMenu = menubar.addMenu('&File')
    return menubar


def menu_bar(parent):
    menubar = QtWidgets.QMenuBar(parent)
    menubar.setStyleSheet(qss.MENUBAR_STYLE)
    fileMenu = menubar.addMenu('&File')
    exitAction = QtWidgets.QAction(QtGui.QIcon(qss.EXIT_ICON), 'Exit', parent)
    exitAction.setShortcut('esc')
    exitAction.setStatusTip('Exit Application')
    exitAction.triggered.connect(parent.close)
    fileMenu.addAction(exitAction)


def open_files(parent, multiple=False, fileExt='', label=None, updateFunction=None):
    if not multiple:
        paths = QtWidgets.QFileDialog.getOpenFileName(
            parent, 'Select file', os.getcwd(), fileExt + ";;" + configuration.ALL_FILES
        )
    else:
        paths = QtWidgets.QFileDialog.getOpenFileNames(parent, 'Select files', os.getcwd(), fileExt)
    if label and paths:
        label.setText(str(paths))
        label.adjustSize()
    if updateFunction:
        updateFunction()
    if paths:
        return paths


def open_folder(parent, label=None, default=os.getcwd(), updateFunction=None):
    dir = QtWidgets.QFileDialog.getExistingDirectory(parent, 'Select folder', default)
    if dir:
        label.setText(str(dir))
        label.adjustSize()
        if updateFunction:
            updateFunction()
        return dir


def remove_tabs(tabsWidget):
    for tab in range(tabsWidget.count() - 1):
        tabsWidget.removeTab(1)


def save_file(parent, fileExt='', label=None):
    paths = QtWidgets.QFileDialog.getSaveFileName(parent, 'Select file', os.getcwd(), fileExt + configuration.ALL_FILES)
    if label and paths:
        label.setText(str(paths[0]))
        label.adjustSize()
    if paths:
        return paths[0]


def savelabel_function(save, saveIcon, widget, fileExt):
    widget.saveLabel = make_entry(widget)
    widget.saveLabel.setText(ROOT)
    widget.saveLabel.setReadOnly(True)
    widget.topLeftLayout.addWidget(widget.saveLabel, 1, 1)
    if (save.__name__ == 'save_file'):
        widget.saveButton = make_button(
            widget, command=lambda: save(widget, fileExt=fileExt, label=widget.saveLabel),
            icon=saveIcon, description=configuration.SAVE_FILE_DESCRIPTION
        )
    else:
        widget.saveButton = make_button(
            widget, command=lambda: save(widget, label=widget.saveLabel), icon=saveIcon,
            description=configuration.SAVE_FOLDER_DESCRIPTION
        )
    widget.topLeftLayout.addWidget(widget.saveButton, 1, 0)


def set_connected(self, *args, **kwargs):
    super(LockButton, self).set_connected()
    self.pair.set_connected()


def set_disabled(self, *args, **kwargs):
    super(LockButton, self).set_disabled()
    self.pair.set_error()


def set_export_options(widget, tab):
    widget.exportFormatsGrid = make_group_box(
        tab, layout=make_grid(), text='Export Format', row=0, column=0, width=100,
        height=100
    )
    widget.exportFormats = make_list(items=('CSV', 'Excel', 'PDF'), layout=widget.exportFormatsGrid, multiSelect=True)

    widget.exportButton = make_button(
        text='Export', command=widget.export_results, layout=tab, row=0, column=4,
        width=50
    )


def set_export_options(widget):
    widget.exportFormatsGrid = make_group_box(
        widget.exportTab, layout=make_grid(), text='Export Format', row=0,
        column=0, width=100, height=100
    )
    widget.exportFormats = make_list(items=('CSV', 'Excel', 'PDF'), layout=widget.exportFormatsGrid, multiSelect=True)
    """widget.csvGrid = make_group_box(widget.exportTab,
                                  layout=make_grid(),
                                  text='CSV',
                                  row=0, column=0, width=45, height=45)

    widget.csvCheck = make_button(type='check',
                                layout=widget.csvGrid,
                                description=TXT_EXPORT_DESCRIPTION)

    widget.excelGrid = make_group_box(widget.exportTab,
                                    layout=make_grid(),
                                    text='Excel',
                                    row=0, column=1, width=45, height=45)

    widget.excelCheck = make_button(type='check',
                                  layout=widget.excelGrid,
                                  description=EXCEL_EXPORT_DESCRIPTION)

    widget.csvGrid = make_group_box(widget.exportTab,
                                  layout=make_grid(),
                                  text='PDF',
                                  row=0, column=2, width=45, height=45)

    widget.csvCheck = make_button(type='check',
                                layout=widget.csvGrid,
                                description=TXT_EXPORT_DESCRIPTION)"""

    widget.exportButton = make_button(
        text='Export', command=widget.export_results, layout=widget.exportTab, row=0,
        column=4, width=50
    )


def set_export_options(widget):
    widget.csvGrid = make_group_box(
        widget.exportTab, layout=make_grid(), text='CSV', row=0, column=0, width=45,
        height=45
    )
    widget.csvCheck = make_button(
        buttonType='check', layout=widget.csvGrid, description=configuration.TXT_EXPORT_DESCRIPTION
    )
    widget.excelGrid = make_group_box(
        widget.exportTab, layout=make_grid(), text='Excel', row=0, column=1, width=45,
        height=45
    )
    widget.excelCheck = make_button(
        buttonType='check', layout=widget.excelGrid, description=configuration.EXCEL_EXPORT_DESCRIPTION
    )
    widget.csvGrid = make_group_box(
        widget.exportTab, layout=make_grid(), text='PDF', row=0, column=3, width=45,
        height=45
    )
    widget.csvCheck = make_button(
        buttonType='check', layout=widget.csvGrid, description=configuration.TXT_EXPORT_DESCRIPTION
    )
    widget.exportButton = make_button(
        text='Export', command=widget.export_results, layout=widget.exportTab, row=0,
        column=4, width=50
    )


def set_export_options(widget):
    widget.exportFormatsGrid = make_group_box(
        widget.exportTab, layout=make_grid(), text='Export Format', row=0,
        column=0, width=100, height=100
    )
    widget.exportFormats = make_list(items=('CSV', 'Excel', 'PDF'), layout=widget.exportFormatsGrid, multiSelect=True)

    widget.exportButton = make_button(
        text='Export', command=widget.export_results, layout=widget.exportTab, row=0,
        column=4, width=50
    )


def set_export_options(widget, layout):
    widget.exportFormatsGrid = make_group_box(
        layout, layout=make_grid(), text='Export Format', row=0, column=0,
        width=100, height=100
    )
    widget.exportFormats = make_list(items=('CSV', 'Excel', 'PDF'), layout=widget.exportFormatsGrid, multiSelect=True)
    """widget.csvGrid = make_group_box(widget.exportTab,
                                  layout=make_grid(),
                                  text='CSV',
                                  row=0, column=0, width=45, height=45)

    widget.csvCheck = make_button(type='check',
                                layout=widget.csvGrid,
                                description=TXT_EXPORT_DESCRIPTION)

    widget.excelGrid = make_group_box(widget.exportTab,
                                    layout=make_grid(),
                                    text='Excel',
                                    row=0, column=1, width=45, height=45)

    widget.excelCheck = make_button(type='check',
                                  layout=widget.excelGrid,
                                  description=EXCEL_EXPORT_DESCRIPTION)

    widget.csvGrid = make_group_box(widget.exportTab,
                                  layout=make_grid(),
                                  text='PDF',
                                  row=0, column=2, width=45, height=45)

    widget.csvCheck = make_button(type='check',
                                layout=widget.csvGrid,
                                description=TXT_EXPORT_DESCRIPTION)"""
    widget.exportButton = make_button(
        text='Export', command=widget.export_results, layout=layout, row=0, column=4,
        width=50
    )


def set_pair_style(self, status):
    if status == 'locked':
        self.pair.set_connected()
    elif status == 'unlocked':
        self.pair.set_valid()
    elif status == 'disabled':
        self.pair.set_error()
    else:
        self.reset()


def set_glowing(self, *args, **kwargs):
    super(LockButton, self).set_glowing()
    self.pair.set_valid()


def set_models(models, tab, multiSelect, command=None, row=0, column=0):
    modelsGrid = make_group_box(tab, layout=make_grid(), text='Models', row=row, column=column, width=205, height=225)

    modelSelection = make_list(
        items=models, command=command, multiSelect=multiSelect, layout=modelsGrid, row=0,
        column=0, width=200, height=175
    )
    return modelSelection


def set_style(self, status):
    if status == 'locked':
        self.set_connected()
    elif status == 'unlocked':
        self.set_glowing()
    elif status == 'disabled':
        self.set_disabled()
    else:
        self.reset()


def set_tabs(widget, tabNames, layout, row=0, column=1):
    tabs = QtWidgets.QTabWidget()
    tabs.setStyleSheet(qss.TAB_STYLE)
    layout.addWidget(tabs, row, column)
    for name in tabNames:
        yield make_tab(widget, name, tabs)


def set_tabs(widget, tabNames, layout):
    tabs = QtWidgets.QTabWidget()
    tabs.setStyleSheet(qss.TAB_STYLE)
    layout.addWidget(tabs, 0, 1)
    for name in tabNames:
        yield make_tab(widget, name, tabs)


def set_widget_size(widget, width, height):
    widget.setMinimumSize(width, height)
    widget.setMaximumSize(width, height)


def set_widget_size(widget, width, height):
    if width and height:
        widget.setMinimumSize(width, height)
        widget.setMaximumSize(width, height)
    elif width:
        widget.setMinimumWidth(width)
        widget.setMaximumWidth(width)


def toggle(self, *args, **kwargs):
    try:
        if not self.disabled:
            status = not getattr(self.configuration.py, self.name)
            self.configuration.py.set(self.name, status)
            self.load()
            if self.callback:
                self.callback()
    except Exception as e:
        statusMsg = 'Toggle {} error: {}.'.format(self.name, e)
        self.parent.update_status(statusMsg, 'error')


def top_left_splitter(widget, open=False, save=False, text=''):
    """
    :param widget: parent widget
    :param open: method for opening file/folder
    :param fileExt: valid extensions for saving a file
    :return: void
    """
    widget.container = make_group_box(
        widget.topLeftLayout, layout=make_grid(), text=text, row=0, column=0, width=2000,
        height=150
    )
    if open:
        # FILES TO READ
        widget.readLabel, widget.readButton = make_file_dialog(
            widget, configuration.OPEN_FOLDER_ICON, configuration.DATABASE, )
        widget.container.addWidget(widget.readButton, 0, 0)
        widget.container.addWidget(widget.readLabel, 0, 1)
    if save:
        """
        FILE TO SAVE
        Add a saveLabel variable to the submenu
        """
        widget.saveLabel, widget.saveButton = make_file_dialog(
            widget, configuration.SAVE_FILE_ICON, configuration.ROOT, )
        widget.container.addWidget(widget.saveButton, 1, 0)
        widget.container.addWidget(widget.saveLabel, 1, 1)


def top_left_splitter(
    widget, open=None, openIcon=configuration.OPEN_FOLDER_ICON, save=None, saveIcon=configuration.SAVE_FILE_ICON,
    fileExt='', updateFunction=None, layout=None,
):
    if open:
        # FILES TO READ
        widget.readLabel = make_entry(widget)
        widget.readLabel.setText(DATABASE)
        widget.readLabel.setReadOnly(True)
        widget.topLeftLayout.addWidget(widget.readLabel, 0, 1)
        widget.readButton = make_button(
            widget, command=lambda: open(widget, label=widget.readLabel), icon=openIcon,
            description=configuration.OPEN_FOLDER_DESCRIPTION
        )
        widget.topLeftLayout.addWidget(widget.readButton, 0, 0)
    if save:
        # FILE TO SAVE
        savelabel_function(save, saveIcon, widget)


def top_left_splitter(
    widget, open=None, openIcon=configuration.OPEN_FOLDER_ICON, save=None, saveIcon=configuration.SAVE_FILE_ICON,
    fileExt='', updateFunction=None, layout=None,
):
    if open:
        # FILES TO READ
        widget.readLabel = make_label(widget)
        widget.topLeftLayout.addWidget(widget.readLabel, 0, 1)
        widget.readButton = make_button(widget, command=lambda: open(widget, label=widget.readLabel), icon=openIcon)
        widget.topLeftLayout.addWidget(widget.readButton, 0, 0)
    if save:
        # FILE TO SAVE
        widget.saveLabel = make_label(widget)
        widget.topLeftLayout.addWidget(widget.saveLabel, 1, 1)
        if (save.__name__ == 'save_file'):
            widget.saveButton = make_button(
                widget,
                command=lambda: save(widget, fileExt=fileExt, label=widget.saveLabel),
                icon=saveIcon
            )
        else:
            widget.saveButton = make_button(widget, command=lambda: save(widget, label=widget.saveLabel), icon=saveIcon)
        widget.topLeftLayout.addWidget(widget.saveButton, 1, 0)


def top_left_splitter(
    widget, open=None, openIcon=configuration.OPEN_FOLDER_ICON, save=None, saveIcon=configuration.SAVE_FILE_ICON,
    fileExt='', updateFunction=None, layout=None,
):
    if open:
        # FILES TO READ
        widget.readLabel = make_label(widget)
        widget.readLabel.setText(DATABASE)
        widget.topLeftLayout.addWidget(widget.readLabel, 0, 1)
        widget.readButton = make_button(widget, command=lambda: open(widget, label=widget.readLabel), icon=openIcon)
        widget.topLeftLayout.addWidget(widget.readButton, 0, 0)
    if save:
        # FILE TO SAVE
        widget.saveLabel = make_label(widget)
        widget.saveLabel.setText(ROOT)
        widget.topLeftLayout.addWidget(widget.saveLabel, 1, 1)
        if (save.__name__ == 'save_file'):
            widget.saveButton = make_button(
                widget,
                command=lambda: save(widget, fileExt=fileExt, label=widget.saveLabel),
                icon=saveIcon
            )
        else:
            widget.saveButton = make_button(widget, command=lambda: save(widget, label=widget.saveLabel), icon=saveIcon)
        widget.topLeftLayout.addWidget(widget.saveButton, 1, 0)


def top_left_splitter(
    widget, open=None, openIcon=configuration.OPEN_FOLDER_ICON, save=None, saveIcon=configuration.SAVE_FILE_ICON,
    fileExt='', updateFunction=None, layout=None,
):
    if open:
        # FILES TO READ
        widget.readLabel = make_entry(widget)
        widget.readLabel.setText(DATABASE)
        widget.readLabel.setReadOnly(True)
        widget.topLeftLayout.addWidget(widget.readLabel, 0, 1)
        widget.readButton = make_button(
            widget, command=lambda: open(
                widget, label=widget.readLabel,
                updateFunction=updateFunction
            ), icon=openIcon,
            description=configuration.OPEN_FOLDER_DESCRIPTION
        )
        widget.topLeftLayout.addWidget(widget.readButton, 0, 0)
    if save:
        # FILE TO SAVE
        savelabel_function(save, saveIcon, widget)


def top_left_splitter(
    widget, open=None, openIcon=configuration.OPEN_FOLDER_ICON, save=None, saveIcon=configuration.SAVE_FILE_ICON,
    fileExt='', updateFunction=None, layout=None,
):
    widget.container = make_group_box(layout, layout=make_grid(), text='Data', row=0, column=0, width=500, height=150)
    if open:
        # FILES TO READ
        widget.readLabel = make_entry(widget)
        widget.readLabel.setText(DATABASE)
        widget.readLabel.setReadOnly(True)
        widget.container.addWidget(widget.readLabel, 0, 1)
        widget.readButton = make_button(
            widget, command=lambda: open(
                widget, label=widget.readLabel,
                updateFunction=updateFunction
            ), icon=openIcon,
            description=configuration.OPEN_FOLDER_DESCRIPTION
        )
    if save:
        # FILE TO SAVE
        widget.saveLabel = make_entry(widget)
        widget.saveLabel.setText(ROOT)
        widget.saveLabel.setReadOnly(True)
        widget.container.addWidget(widget.saveLabel, 1, 1)
        if (save.__name__ == 'save_file'):
            widget.saveButton = make_button(
                widget,
                command=lambda: save(widget, fileExt=fileExt, label=widget.saveLabel),
                icon=saveIcon, description=configuration.SAVE_FILE_DESCRIPTION
            )
        else:
            widget.saveButton = make_button(
                widget, command=lambda: save(widget, label=widget.saveLabel), icon=saveIcon,
                description=configuration.SAVE_FOLDER_DESCRIPTION
            )
        widget.container.addWidget(widget.readButton, 0, 0)
        widget.container.addWidget(widget.saveButton, 1, 0)


class Dock:
    pass  # self.layout().addWidget(components.TitleBar(self, 'CubeSAT Controller'))  # self.deviceDock = QtWidgets.QDockWidget('Device', self)  # self.deviceDock.setWidget(self.deviceMenu)  # self.deviceDock.setMaximumSize(500,500)  # self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.deviceDock)  # self.transceiverDock = QtWidgets.QDockWidget('Transceiver', self)  # self.transceiverDock.setWidget(self.transceiverMenu)  # self.transceiverDock.setMinimumSize(250,200)  # self.transceiverDock.setMaximumSize(750,250)  # self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.transceiverDock)  # self.logDock = QtWidgets.QDockWidget('Log', self)  # self.logDock.setMaximumSize(750,500)  # self.logDock.setWidget(self.logMenu)  # self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.logDock)


class FrameCanvas(tk.Frame):
    def __init__(self, parent, menuname, f):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=menuname, font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas._tkcanvas.pack()
        self.grid(row=0, column=0, sticky="nsew")


class LockButton(qss.pushbutton.PushButton):
    def __init__(self, app, parent, name, configuration, pair, callback, layout, pos, tooltip):
        super(LockButton, self).__init__(
            app, parent, '', {
                'default': configuration.lockOpen, 'disabled': configuration.lockOpen,
                'glowing': configuration.lockOpen,
                'valid':   configuration.lockOpen, 'connected': configuration.lockClosed,
            }
        )
        self.app = app
        self.parent = parent
        self.configuration.py = configuration
        self.name = name
        self.pair = pair
        self.callback = callback
        self.setIconSize(
            QtWidgets.QtCore.QSize(int(self.app.palette.minHeight // 1.5), int(self.app.palette.minHeight // 1.5))
        )
        self.setFixedSize(self.app.palette.minHeight, self.app.palette.minHeight)
        layout.addWidget(self, pos[0], pos[1])
        self.setToolTip(tooltip)
        self.clicked.connect(self.toggle)
        self.load()


class MenuBuilder:
    def __init__(self, app):
        self.app = app

    def add_toolbar(self, parent, command=None, text='', icon=None, tooltip=None, shortcut=None, size=(400, 400)):
        """
        parent: QWidget
        icon: str (file path)
        tooltip: str
        shortcut: str
        return:
        """
        toolbar = QtWidgets.QToolBar(parent)
        toolbar.setStyleSheet(qss_format(qss.barstyles.toolbar, self.app.palette))
        toolbar.setIconSize(QtCore.QSize(150, 150))
        toolbar.setFixedSize(size[0], size[1])
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        qIcon = QtGui.QIcon(icon)
        action = QtWidgets.QAction(qIcon, '', parent)
        if shortcut:
            action.setShortcut(shortcut)
        if command:
            action.triggered.connect(command)
        action.setIconText(text)
        action.setToolTip(tooltip)
        toolbar.addAction(action)
        parent.addToolBar(toolbar)
        return toolbar

    def add_action(self, parent, command=None, text='', icon=None, tooltip=None, shortcut=None):
        """
        parent: QWidget
        icon: str (file path)
        tooltip: str
        shortcut: str
        return:
        """
        qIcon = QtGui.QIcon(icon)
        action = QtWidgets.QAction(qIcon, '', parent)
        if shortcut:
            action.setShortcut(shortcut)
        if command:
            action.triggered.connect(command)
        action.setIconText(text)
        action.setToolTip(tooltip)
        return action

    def make_scroll_area(self, layout, width=None, height=None):
        widget = QtWidgets.QWidget()
        widget.setStyleSheet(qss_format(qss.generalstyles.widget, self.app.palette))
        widget.setLayout(layout)
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setFrameShape(QtWidgets.QScrollArea.StyledPanel)
        scrollArea.setWidgetResizable(True)
        scrollArea.setStyleSheet(qss_format(qss.containerstyles.frame + qss.scroll, self.app.palette))
        scrollArea.setWidget(widget)
        self.resize_widget(scrollArea, width, height)
        return scrollArea

    def make_splitter(self, style, widgets):
        if style == 'h' or style == 'horizontal':
            splitterWidget = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        else:
            splitterWidget = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        for widget in widgets:
            splitterWidget.addWidget(widget)
        splitterWidget.setStyleSheet(
            qss_format(qss.containerstyles.frame + qss.containerstyles.splitter, self.app.palette)
        )
        QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        return splitterWidget

    def make_scroll_area_tab(self, parent, text, master=None, layout=None, stylesheet=qss.containerstyles.tab):
        tab = QtWidgets.QScrollArea()
        tab.setWidget(QtWidgets.QWidget())
        tab.setStyleSheet(qss_format(stylesheet, self.app.palette))
        tabLayout = self.make_grid(tab.widget())
        tab.setWidgetResizable(True)
        master.addTab(tab, text)
        return tabLayout

    def open_files(self, parent, multiple=False, fileExt='', label=None, command=None):
        ALL_FILES = ''
        if not multiple:
            paths = QtWidgets.QFileDialog.getOpenFileName(
                parent, 'Select file', os.getcwd(),
                fileExt + ";;" + ALL_FILES
            )
        else:
            paths = QtWidgets.QFileDialog.getOpenFileNames(parent, 'Select files', os.getcwd(), fileExt)
        if label and paths:
            label.setText(str(paths))
            label.adjustSize()
        if command:
            command()
        if paths:
            return paths

    def open_folder(self, parent, label=None, default=os.getcwd(), update_function=None):
        dir = QtWidgets.QFileDialog.getExistingDirectory(parent, 'Select folder', default)
        if dir:
            label.setText(str(dir))
            label.adjustSize()
            if update_function:
                update_function()
            return dir

    def remove_tabs(self, tabsWidget):
        for tab in range(tabsWidget.count() - 1):
            tabsWidget.removeTab(1)

    def save_file(self, parent, fileExt='', label=None):
        paths = QtWidgets.QFileDialog.getSaveFileName(parent, 'Select file', os.getcwd(), fileExt)
        if label and paths:
            label.setText(str(paths[0]))
            label.adjustSize()
        if paths:
            return paths[0]

    def set_tabs(self, widget, tabNames, layout, scroll=False):
        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet(qss_format(qss.containerstyles.tab, self.app.palette))
        # tabs.setMinimumSize(self.parent.palette.minHeight*2, self.parent.palette.minHeight)
        layout.addWidget(tabs, 0, 1)
        tabDict = {}
        for name in tabNames:
            if scroll:
                tabDict[name] = self.make_scroll_area_tab(widget, name, tabs)
            else:
                tabDict[name] = self.tab_area(widget, name, tabs)
        return tabDict

    def subdivision(self, parent):
        subdivision = self.make_frame(parent)
        layout = self.make_grid(subdivision)
        return subdivision, layout

    def tab_area(self, parent, text, master, layout=None, stylesheet=qss.containerstyles.tab):
        tab = QtWidgets.QTabWidget()
        tab.setStyleSheet(qss_format(stylesheet, self.app.palette))
        tabLayout = self.make_grid(tab)
        master.addTab(tab, text)
        return tabLayout

    def make_file_dialog(self, widget, icon=None, text='', tooltipButton='', tooltipLabel=''):
        label = self.make_label(widget, text=text, width=200, )
        command = lambda: self.open_folder(widget, label=label, update_function=widget.rewindow, )
        button = self.make_button(widget, command=command, icon=icon, width=50, )
        button.setToolTip(tooltipButton)
        label.setToolTip(tooltipLabel.format(label.text()))
        return label, button

    def insert_file_dialog(self, widget, container, icon, text, row, column, tooltipButton, tooltipLabel):
        label, button = self.make_file_dialog(
            widget=widget, icon=icon, text=text, tooltipButton=tooltipButton,
            tooltipLabel=tooltipLabel, )
        container.addWidget(button, row, column, )
        container.addWidget(label, row, column + 1, )
        return label, button

    def __init__(self, parent):
        self.parent = parent

    def add_toolbar(self, parent, command=None, text='', icon=None, tooltip=None, shortcut=None):
        """
        parent: QWidget
        icon2: str (file path)
        tooltip: str
        shortcut: str
        return:
        """
        toolbar = QtWidgets.QToolBar(parent)
        toolbar.setStyleSheet(qss.format_qss(qss.toolbar, self.parent.palette))
        toolbar.setIconSize(QtCore.QSize(150, 150))
        toolbar.setFixedSize(400, 400)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        qIcon = QtGui.QIcon(icon)
        action = QtWidgets.QAction(qIcon, '', parent)
        if shortcut:
            action.setShortcut(shortcut)
        if command:
            action.triggered.connect(command)
        action.setIconText(text)
        action.setToolTip(tooltip)
        toolbar.addAction(action)
        parent.addToolBar(toolbar)
        return None

    def add_action(self, parent, command=None, text='', icon=None, tooltip=None, shortcut=None):
        """
        parent: QWidget
        icon2: str (file path)
        tooltip: str
        shortcut: str
        return:
        """
        qIcon = QtGui.QIcon(icon)
        action = QtWidgets.QAction(qIcon, '', parent)
        if shortcut:
            action.setShortcut(shortcut)
        if command:
            action.triggered.connect(command)
        action.setIconText(text)
        action.setToolTip(tooltip)
        return action

    def load_browser(self, browser, file, online=False):
        try:
            if online:
                browser.load(QtCore.QUrl(file))
            else:
                browser.load(QtCore.QUrl.fromLocalFile(file))
        except:
            pass
        return browser

    def make_browser(self, layout, file, online=False):
        browser = QtWebEngineWidgets.QWebEngineView()
        if file:
            self.load_browser(browser, file, online)
        if layout != None:
            layout.addWidget(browser)
        return browser

    def make_button(
        self, parent=None, buttonType=None, command=None, text='', icon=None, layout=None, row=0, column=0,
        width=None, tooltip='', isChecked=False, stylesheet=qss.button, height=None
    ):
        if buttonType == 'radio':
            button = QtWidgets.QRadioButton(parent)
            button.setStyleSheet(qss.format_qss(qss.radio, self.parent.palette))
        elif buttonType == 'check':
            button = QtWidgets.QCheckBox(parent)
            if isChecked:
                button.setChecked(True)
                button.setStyleSheet(qss.format_qss(qss.checkboxChecked, self.parent.palette))
            else:
                button.setStyleSheet(qss.format_qss(qss.checkboxUnchecked, self.parent.palette))
        else:
            if icon:
                icn = QtGui.QIcon(icon)
                button = QtWidgets.QPushButton(icn, text, parent)
                button.setIconSize(QtCore.QSize(20, 20))
            else:
                button = QtWidgets.QPushButton(text, parent)
            if stylesheet:
                button.setStyleSheet(qss.format_qss(stylesheet, self.parent.palette))
        button.setToolTip(tooltip)
        # button.setvariables.FONT(QtGui.Qvariables.FONT(variables.FONT))
        if command:
            button.clicked.connect(command)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(button, row, column)
            else:
                layout.addWidget(button)
        # button.adjustSize()
        self.resize_widget(button, width, height)
        return button

    def make_combo(
        self, parent=None, items=None, command=None, layout=None, row=0, column=0, width=40, height=20,
        tooltip='', default=None
    ):
        combo = QtWidgets.QComboBox(parent)
        self.resize_widget(combo, width, height)
        combo.setStyleSheet(qss.format_qss(qss.dropdown, self.parent.palette))
        combo.addItems(items)
        if default:
            combo.setCurrentIndex(default)
        if command:
            combo.activated[str].connect(command)
        combo.setToolTip(tooltip)
        if layout:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(combo, row, column)
            else:
                layout.layout().addWidget(combo)
        combo.adjustSize()
        return combo

    def make_dial(
        self, parent=None, min=None, max=None, value=None, step=1, layout=None, row=0, column=0, tooltip='',
        command=None
    ):
        dial = QtWidgets.QDial(parent)
        dial.setStyleSheet(qss.format_qss(qss.dial, self.parent.palette))
        self.resize_widget(dial, 30, 30)
        dial.setMinimum(min)
        dial.setMaximum(max)
        dial.setNotchesVisible(True)
        dial.setSingleStep(step)
        dial.setWrapping(False)
        if not value:
            dial.setValue(min)
        else:
            dial.setValue(value)
        dial.setToolTip(tooltip)
        if command:
            dial.valueChanged.connect(command)
        if layout:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(dial, row, column)
            else:
                layout.addWidget(dial)
        return dial

    def make_entry(
        self, parent, text='', width=None, height=None, layout=None, row=0, column=0, tooltip=None,
        command=None, number=False, readOnly=False, area=False, stylesheet=qss.entry
    ):
        if area:
            entry = QtWidgets.QPlainTextEdit(parent)
            if text:
                entry.setPlainText(text)
        else:
            entry = QtWidgets.QLineEdit(parent)
            if text:
                entry.setText(text)
        if command:
            entry.textChanged.connect(command)
        # entry.setvariables.FONT(QtGui.Qvariables.FONT(variables.FONT))
        self.resize_widget(entry, width, height)
        entry.setReadOnly(readOnly)
        entry.setToolTip(tooltip)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(entry, row, column)
            else:
                layout.layout().addWidget(entry)
        if stylesheet:
            entry.setStyleSheet(qss.format_qss(stylesheet, self.parent.palette))
        return entry

    def make_form(self, parent=None, padding=False):
        form = QtWidgets.QFormLayout(parent)
        if padding:
            form.setContentsMargins(0, self.parent.palette.topPadding, 0, self.parent.palette.bottomPadding)
        return form

    def make_frame(self, parent):
        frame = QtWidgets.QFrame(parent)
        frame.setStyleSheet(qss.format_qss(qss.frame, self.parent.palette))
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        return frame

    def make_group_box(
        self, parent, layout=None, text='', row=0, column=0, width=None, height=None, border=True,
        strictWidth=False, strictHeight=False, strictSize=False
    ):
        groupBox = QtWidgets.QGroupBox(text)
        groupBox.setStyleSheet(qss.format_qss(qss.groupBox, self.parent.palette, ) + '' if border else qss.groupBoxNu)
        groupBox.setLayout(layout) if layout else groupBox.setLayout(self.make_grid())
        parent.addWidget(groupBox, row, column)
        if strictWidth or width:
            groupBox.setFixedWidth(width)

        if strictHeight or height:
            groupBox.setFixedHeight(height)

        if strictSize:
            groupBox.setFixedSize(width, height)
        return groupBox

    def make_grid(self, parent=None, padding=False):
        grid = QtWidgets.QGridLayout(parent)
        if padding:
            grid.setContentsMargins(0, topPadding, 0, bottomPadding)
        return grid

    def make_label(self, parent, text='', layout=None, row=0, column=0, width=None, height=None, tooltip=''):
        labelWidget = QtWidgets.QLabel(text, parent)
        labelWidget.setWordWrap(True)
        self.resize_widget(labelWidget, width, height)
        labelWidget.setToolTip(tooltip)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(labelWidget, row, column)
            else:
                layout.addWidget(labelWidget)

        # label.setvariables.FONT(QtGui.Qvariables.FONT(variables.FONT))
        labelWidget.setStyleSheet(qss.format_qss(qss.label, self.parent.palette))
        labelWidget.adjustSize()
        return labelWidget

    def make_list_widget(
        self, parent=None, items=None, command=None, default=None, layout=None, row=0, column=0,
        width=None, height=None, multiSelect=False
    ):
        listWidget = QtWidgets.QListWidget(parent)
        if items:
            listWidget.addItems(items)
        listWidget.setStyleSheet(qss.format_qss(qss.list, self.parent.palette))
        if default != None:
            listWidget.setCurrentRow(default)
        if command:
            listWidget.currentItemChanged.connect(command)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(listWidget, row, column)
            else:
                layout.addWidget(listWidget)
        self.resize_widget(listWidget, width, height)
        if multiSelect:
            listWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        return listWidget

    def pair(
        self, parent=None, pair=None, text=None, layout=None, row=0, column=0, labelWidth=None, labelHeight=None,
        comboItems=None, command=None, pairText=None, pairWidth=None, pairHeight=None,
        dialSettings=(None, None, None, None), tooltip='', label=False, pairKwargs={}, default=None,
        readOnly=False, pairStyleSheet=None, area=False
    ):
        if label:
            labelWidget = self.make_label(parent, text, layout, row, column, labelWidth, labelHeight, tooltip)
            lay = layout
        else:
            grid = self.make_group_box(
                layout, layout=self.make_form(padding=True), text=text, row=row, column=column,
                border=False
            )
            lay = grid
        if pair == 'check':
            pair = self.make_button(parent, 'check', **pairKwargs)
        if pair == 'combo':
            pair = self.make_combo(
                parent=parent, items=comboItems, command=command, layout=lay, row=row,
                column=column + 1, width=pairWidth, height=pairHeight, tooltip=tooltip,
                default=default
            )
        if pair == 'dial':
            pair = self.make_dial(
                parent, dialSettings[0], dialSettings[1], dialSettings[2], dialSettings[3], lay, row,
                column + 1, tooltip, command
            )
        if pair == 'slider':
            pair = self.make_slider(
                parent, inputSettings=dialSettings, layout=lay, row=row, column=column + 1,
                tooltip=tooltip, command=command, width=pairWidth, height=pairHeight,
                stylesheet=pairStyleSheet, )
        if pair == 'entry':
            pair = self.make_entry(
                parent, text=pairText, width=pairWidth, layout=lay, row=row, column=column + 1,
                tooltip=tooltip, command=command, readOnly=readOnly, stylesheet=pairStyleSheet,
                area=area, )
        if pair == 'radio':
            pair = self.make_button(parent, 'radio', command, '', None, lay, row, column + 1, pairWidth, tooltip, )
        if pair == 'numerical':
            pair = self.make_numerical_input(
                parent, inputSettings=dialSettings, layout=lay, row=row, column=column + 1,
                tooltip=tooltip, command=command, width=pairWidth, height=pairHeight,
                readOnly=readOnly, stylesheet=pairStyleSheet, )
        if label:
            return labelWidget, pair
        return grid, pair

    def progress_bar(self, layout):
        p = QtWidgets.QProgressBar()
        layout.addWidget(p)
        p.setGeometry(30, 40, 200, 25)
        p.show()
        return p

    def make_scroll_area(self, layout, width=None, height=None):
        widget = QtWidgets.QWidget()
        widget.setStyleSheet(qss.format_qss(qss.widget, self.parent.palette))
        widget.setLayout(layout)
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setFrameShape(QtWidgets.QScrollArea.StyledPanel)
        scrollArea.setWidgetResizable(True)
        scrollArea.setStyleSheet(qss.format_qss(qss.frame + qss.scroll, self.parent.palette))
        scrollArea.setWidget(widget)
        self.resize_widget(scrollArea, width, height)
        return scrollArea

    def make_splitter(self, style, widgets):
        if style == 'h' or style == 'horizontal':
            splitterWidget = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        else:
            splitterWidget = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        for widget in widgets:
            splitterWidget.addWidget(widget)

        splitterWidget.setStyleSheet(qss.format_qss(qss.frame + qss.splitter, self.parent.palette))
        QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        return splitterWidget

    def make_scroll_area_tab(self, parent, text, master=None, layout=None, stylesheet=qss.widget):
        tab = QtWidgets.QScrollArea()
        tab.setWidget(QtWidgets.QWidget())
        tab.setStyleSheet(qss.format_qss(stylesheet, self.parent.palette))
        tabLayout = self.make_grid(tab.widget())
        tab.setWidgetResizable(True)
        master.addTab(tab, text)
        return tabLayout

    def make_statusbar(self, parent, layout, row, column):
        statusbar = QtWidgets.QStatusBar(parent)
        statusbar.setStyleSheet(qss.format_qss(qss.statusbar, self.parent.palette))
        layout.addWidget(statusbar, row, column)
        return statusbar

    def menu_bar(self, parent, submenus, stylesheet=qss.menubar):
        menubar = QtWidgets.QMenuBar(parent)
        menubar.setStyleSheet(qss.format_qss(stylesheet, self.parent.palette))
        for submenu in submenus:
            menubar.addMenu(submenu)
        return menubar

    def open_files(self, parent, multiple=False, fileExt='', label=None, command=None):
        ALL_FILES = ''
        if not multiple:
            paths = QtWidgets.QFileDialog.getOpenFileName(
                parent, 'Select file', os.getcwd(),
                fileExt + ";;" + ALL_FILES
            )
        else:
            paths = QtWidgets.QFileDialog.getOpenFileNames(parent, 'Select files', os.getcwd(), fileExt)
        if label and paths:
            label.setText(str(paths))
            label.adjustSize()
        if command:
            command()
        if paths:
            return paths

    def open_folder(self, parent, label=None, default=os.getcwd(), command=None):
        dir = QtWidgets.QFileDialog.getExistingDirectory(parent, 'Select folder', default)
        if dir:
            label.setText(str(dir))
            label.adjustSize()
            if command:
                command()
            return dir

    def remove_tabs(tabsWidget):
        for tab in range(tabsWidget.count() - 1):
            tabsWidget.removeTab(1)

    def save_file(self, parent, fileExt='', label=None):
        paths = QtWidgets.QFileDialog.getSaveFileName(parent, 'Select file', os.getcwd(), fileExt)
        if label and paths:
            label.setText(str(paths[0]))
            label.adjustSize()
        if paths:
            return paths[0]

    def set_tabs(self, widget, tabNames, layout):
        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet(qss.format_qss(qss.tab, self.parent.palette))
        layout.addWidget(tabs, 0, 1)
        for name in tabNames:
            yield self.make_scroll_area_tab(widget, name, tabs)

    def subdivision(self, parent):
        subdivision = self.make_frame(parent)
        layout = self.make_grid(subdivision)
        return subdivision, layout

    def tab_area(self):
        tab = QtWidgets.QTabWidget()
        tab.setStyleSheet(qss.format_qss(qss.tab, self.parent.palette))
        return tab

    def make_file_dialog(self, widget, icon=None, text='', tooltipButton='', tooltipLabel=''):
        label = self.make_label(widget, text=text, width=200, )
        command = lambda: self.open_folder(widget, label=label, command=widget.rewindow, )
        button = self.make_button(widget, command=command, icon=icon, width=50, )
        button.setToolTip(tooltipButton)
        label.setToolTip(tooltipLabel.format(label.text()))
        return label, button

    def insert_file_dialog(self, widget, container, icon, text, row, column, tooltipButton, tooltipLabel):
        label, button = self.make_file_dialog(
            widget=widget, icon=icon, text=text, tooltipButton=tooltipButton,
            tooltipLabel=tooltipLabel, )
        container.addWidget(button, row, column, )
        container.addWidget(label, row, column + 1, )
        return label, button

    def make_numerical_input(
        self, parent, text='', inputSettings=(0, 1, 0, 0), width=150, layout=None, row=0, column=0,
        tooltip='', command=None, decimals=0, height=None, readOnly=False, stylesheet=None
    ):
        if decimals > 0:
            entry = QtWidgets.QDoubleSpinBox(parent, decimals=decimals)
        else:
            entry = QtWidgets.QSpinBox(parent)
        entry.setRange(inputSettings[0], inputSettings[1])
        entry.setValue(inputSettings[2])
        entry.setSingleStep(inputSettings[3])
        self.resize_widget(entry, width, height)
        if command:
            entry.valueChanged.connect(command)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(entry, row, column)
            else:
                layout.layout().addWidget(entry)
        entry.setToolTip(tooltip)
        entry.setReadOnly(readOnly)
        if stylesheet:
            entry.setStyleSheet(qss.format_qss(stylesheet, self.parent.palette))
        else:
            entry.setStyleSheet(qss.format_qss(qss.spin, self.parent.palette))
        return entry

    def make_slider(
        self, parent=None, layout=None, row=0, column=0, tooltip='', command=None, width=None, height=None,
        inputSettings=(0, 0, 0, 0), stylesheet=qss.slider
    ):
        sliderWidget = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sliderWidget.setRange(inputSettings[0], inputSettings[1])
        sliderWidget.setValue(inputSettings[2])
        sliderWidget.setSingleStep(inputSettings[3])
        sliderWidget.setToolTip(tooltip)
        sliderWidget.setStyleSheet(qss.format_qss(stylesheet, self.parent.palette))
        sliderWidget.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.resize_widget(sliderWidget, width, height)
        if command:
            sliderWidget.valueChanged.connect(command)
        if type(layout) == QtWidgets.QGridLayout:
            layout.addWidget(sliderWidget, row, column)
        else:
            layout.layout().addWidget(sliderWidget)
        return sliderWidget

    def resize_widget(self, widget, width, height):
        if width:
            widget.setFixedWidth(width)
        if height:
            widget.setFixedHeight(height)

    def __init__(self, app):
        self.app = app

    def add_toolbar(self, parent, command=None, text='', icon=None, tooltip=None, shortcut=None, size=(400, 400)):
        """
        parent: QWidget
        icon2: str (file path)
        tooltip: str
        shortcut: str
        return:
        """
        toolbar = QtWidgets.QToolBar(parent)
        toolbar.setStyleSheet(qss_format(qss.barstyle.toolbar, self.app.palette))
        toolbar.setIconSize(QtCore.QSize(150, 150))
        toolbar.setFixedSize(size[0], size[1])
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        qIcon = QtGui.QIcon(icon)
        action = QtWidgets.QAction(qIcon, '', parent)
        if shortcut:
            action.setShortcut(shortcut)
        if command:
            action.triggered.connect(command)
        action.setIconText(text)
        action.setToolTip(tooltip)
        toolbar.addAction(action)
        parent.addToolBar(toolbar)
        return toolbar

    def add_action(self, parent, command=None, text='', icon=None, tooltip=None, shortcut=None):
        """
        parent: QWidget
        icon2: str (file path)
        tooltip: str
        shortcut: str
        return:
        """
        qIcon = QtGui.QIcon(icon)
        action = QtWidgets.QAction(qIcon, '', parent)
        if shortcut:
            action.setShortcut(shortcut)
        if command:
            action.triggered.connect(command)
        action.setIconText(text)
        action.setToolTip(tooltip)
        return action

    def load_browser(self, browser, file, online=False):
        try:
            if online:
                browser.load(QtCore.QUrl(file))
            else:
                browser.load(QtCore.QUrl.fromLocalFile(file))
        except:
            pass
        return browser

    def make_browser(self, layout, file, online=False):
        browser = QtWebEngineWidgets.QWebEngineView()
        if file:
            self.load_browser(browser, file, online)
        if layout != None:
            layout.addWidget(browser)
        return browser

    def make_button(
        self, parent=None, buttonType=None, command=None, text='', icon=None, layout=None, row=0, column=0,
        width=None, tooltip='', isChecked=False, stylesheet=qss.btnstyle.button,
        height=None
    ):
        if buttonType == 'radio':
            button = QtWidgets.QRadioButton(parent)
            button.setStyleSheet(qss_format(qss.radio, self.app.palette))
        elif buttonType == 'check':
            button = QtWidgets.QCheckBox(parent)
            if isChecked:
                button.setChecked(True)
                button.setStyleSheet(qss_format(qss.btnstyle.checkboxChecked, self.app.palette))
            else:
                button.setStyleSheet(qss_format(qss.btnstyle.checkboxUnchecked, self.app.palette))
        else:
            button = qss.pbutton.PushButton(self.app, parent, text, icon)
            if stylesheet:
                button.setStyleSheet(qss_format(stylesheet, self.app.palette))
        button.setToolTip(tooltip)
        # button.setvariables.FONT(QtGui.Qvariables.FONT(variables.FONT))
        if command:
            button.clicked.connect(command)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(button, row, column)
            else:
                layout.layout().addWidget(button)
        # button.adjustSize()
        button.setMinimumSize(self.app.palette.minHeight, self.app.palette.minHeight)
        button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.resize_widget(button, width, height)
        return button

    def make_combo(
        self, parent=None, items=None, command=None, layout=None, row=0, column=0, width=40, height=20,
        tooltip='', default=None
    ):
        combo = QtWidgets.QComboBox(parent)
        self.resize_widget(combo, width, height)
        combo.setStyleSheet(qss_format(qss.selectionstyle.dropdown, self.app.palette))
        combo.addItems(items)
        if default:
            combo.setCurrentIndex(default)
        if command:
            combo.activated[str].connect(command)
        combo.setToolTip(tooltip)
        if layout:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(combo, row, column)
            else:
                layout.layout().addWidget(combo)
        combo.adjustSize()
        combo.setMinimumHeight(self.app.palette.minHeight)
        return combo

    def make_dial(
        self, parent=None, min=None, max=None, value=None, step=1, layout=None, row=0, column=0, tooltip='',
        command=None
    ):
        dial = QtWidgets.QDial(parent)
        dial.setStyleSheet(qss_format(qss.slider.dial, self.app.palette))
        self.resize_widget(dial, 30, 30)
        dial.setMinimum(min)
        dial.setMaximum(max)
        dial.setNotchesVisible(True)
        dial.setSingleStep(step)
        dial.setWrapping(False)
        if not value:
            dial.setValue(min)
        else:
            dial.setValue(value)
        dial.setToolTip(tooltip)
        if command:
            dial.valueChanged.connect(command)
        if layout:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(dial, row, column)
            else:
                layout.addWidget(dial)
        return dial

    def make_entry(
        self, parent, text='', width=None, height=None, layout=None, row=0, column=0, tooltip=None,
        command=None, number=False, readOnly=False, area=False, stylesheet=qss.entry.entry
    ):
        if area:
            entry = TSInputTextArea(self.app, parent)
            if text:
                entry.setPlainText(text)
        else:
            entry = TSInputTextLine(self.app, parent)
            if text:
                entry.setText(text)
        if command:
            entry.textChanged.connect(command)
        # entry.setvariables.FONT(QtGui.Qvariables.FONT(variables.FONT))
        self.resize_widget(entry, width, height)
        entry.setMinimumHeight(self.app.palette.minHeight)
        entry.setReadOnly(readOnly)
        entry.setToolTip(tooltip)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(entry, row, column)
            else:
                layout.layout().addWidget(entry)
        if stylesheet:
            entry.setStyleSheet(qss_format(stylesheet, self.app.palette))
        # entry.setMinimumHeight(40)
        return entry

    def make_form(self, parent=None, padding=False):
        form = QtWidgets.QFormLayout(parent)
        if padding:
            form.setContentsMargins(0, self.app.palette.topPadding, 0, self.app.palette.bottomPadding)
        return form

    def make_frame(self, parent):
        frame = QtWidgets.QFrame(parent)
        frame.setStyleSheet(qss_format(qss.containerstyle.frame, self.app.palette))
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        return frame

    def make_group_box(
        self, parent, layout=None, text='', row=0, column=0, width=None, height=None, border=True,
        strictWidth=False, strictHeight=False, strictSize=False, columnSpan=1, tooltip=None
    ):
        groupBox = QtWidgets.QGroupBox(text)
        groupBox.setStyleSheet(
            qss_format(
                qss.containerstyle.groupBox,
                self.app.palette, ) + '' if border else qss.containerstyle.groupBoxNu
        )
        groupBox.setLayout(layout) if layout else groupBox.setLayout(self.make_grid())
        parent.addWidget(groupBox, row, column, 1, columnSpan)
        if strictWidth or width:
            groupBox.setFixedWidth(width)
        if strictHeight or height:
            groupBox.setFixedHeight(height)
        if strictSize:
            groupBox.setFixedSize(width, height)
        if tooltip:
            groupBox.setToolTip(tooltip)
        if border:
            groupBox.layout().setContentsMargins(1, 1, 1, 1)
        else:
            groupBox.layout().setContentsMargins(1, 4, 1, 4)
        groupBox.layout().setSpacing(0)
        return groupBox

    def make_grid(self, parent=None, padding=False):
        grid = QtWidgets.QGridLayout(parent)
        if padding:
            grid.setContentsMargins(0, qss.qss.topPadding, 0, qss.qss.bottomPadding)
        return grid

    def make_label(self, parent, text='', layout=None, row=0, column=0, width=None, height=None, tooltip=''):
        labelWidget = QtWidgets.QLabel(text, parent)
        labelWidget.setWordWrap(True)
        self.resize_widget(labelWidget, width, height)
        labelWidget.setToolTip(tooltip)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(labelWidget, row, column)
            else:
                layout.addWidget(labelWidget)

        # label.setvariables.FONT(QtGui.Qvariables.FONT(variables.FONT))
        labelWidget.setStyleSheet(qss_format(qss.general.label, self.app.palette))
        labelWidget.adjustSize()
        return labelWidget

    def make_list_widget(
        self, parent=None, items=None, command=None, default=None, layout=None, row=0, column=0,
        width=None, height=None, multiSelect=False
    ):
        listWidget = QtWidgets.QListWidget(parent)
        if items:
            listWidget.addItems(items)
        listWidget.setStyleSheet(qss_format(qss.selectionstyle.list, self.app.palette))
        if default != None:
            listWidget.setCurrentRow(default)
        if command:
            listWidget.currentItemChanged.connect(command)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(listWidget, row, column)
            else:
                layout.addWidget(listWidget)
        self.resize_widget(listWidget, width, height)
        if multiSelect:
            listWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        return listWidget

    def make_table(
        self, parent=None, items=None, command=None, default=None, layout=None, row=0, column=0, width=None,
        height=None, multiSelect=False
    ):
        table = TSLogTable(self.app, parent)
        # if items:
        #     table.addItems(items)
        # if default != None:
        #     table.setCurrentRow(default)
        # if command:
        #     table.currentItemChanged.connect(command)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(table, row, column)
            else:
                layout.addWidget(table)
        # self.resize_widget(table, width, height)
        # if multiSelect:
        #     table.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        return table

    def pair(
        self, parent=None, pair=None, text=None, layout=None, row=0, column=0, labelWidth=None, labelHeight=None,
        comboItems=None, command=None, pairText=None, pairWidth=None, pairHeight=None,
        inputSettings=(None, None, None, None), tooltip='', label=False, pairKwargs={}, default=None,
        readOnly=False, pairSS=None, area=False, columnSpan=1, gbHeight=None, gbWidth=None
    ):
        if label:
            labelWidget = self.make_label(parent, text, layout, row, column, labelWidth, labelHeight, tooltip)
            lay = layout
        else:
            grid = self.make_group_box(
                layout, layout=self.make_form(padding=True), text=text, row=row, column=column,
                border=False, width=gbWidth, height=gbHeight, columnSpan=columnSpan
            )
            lay = grid
        if pair == 'check':
            pair = self.make_button(parent, 'check', layout=lay, **pairKwargs)
        if pair == 'combo':
            pair = self.make_combo(
                parent=parent, items=comboItems, command=command, layout=lay, row=row,
                column=column + 1, width=pairWidth, height=pairHeight, tooltip=tooltip,
                default=default
            )
        if pair == 'dial':
            pair = self.make_dial(
                parent, inputSettings[0], inputSettings[1], inputSettings[2], inputSettings[3], lay,
                row, column + 1, tooltip, command
            )
        if pair == 'slider':
            pair = self.make_slider(
                parent, inputSettings=inputSettings, layout=lay, row=row, column=column + 1,
                tooltip=tooltip, command=command, width=pairWidth, height=pairHeight, )
        if pair == 'entry':
            pair = self.make_entry(
                parent, text=pairText, width=pairWidth, layout=lay, height=pairHeight, row=row,
                column=column + 1, tooltip=tooltip, command=command, area=area, )
        if pair == 'radio':
            pair = self.make_button(parent, 'radio', command, '', None, lay, row, column + 1, pairWidth, tooltip, )
        if pair == 'numerical':
            pair = self.make_numerical_input(
                parent, inputSettings=inputSettings, layout=lay, row=row,
                column=column + 1, tooltip=tooltip, command=command, width=pairWidth,
                height=pairHeight, )
        if label:
            return labelWidget, pair
        return grid, pair

    def progress_bar(self, layout):
        p = QtWidgets.QProgressBar()
        layout.addWidget(p)
        p.setGeometry(30, 40, 200, 25)
        p.show()
        return p

    def make_scroll_area(self, layout, width=None, height=None):
        widget = QtWidgets.QWidget()
        widget.setStyleSheet(qss_format(qss.general.widget, self.app.palette))
        widget.setLayout(layout)
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setFrameShape(QtWidgets.QScrollArea.StyledPanel)
        scrollArea.setWidgetResizable(True)
        scrollArea.setStyleSheet(
            qss_format(
                qss.containerstyle.frame + qss.scroll, self.app.palette
            )
        )
        scrollArea.setWidget(widget)
        self.resize_widget(scrollArea, width, height)
        return scrollArea

    def make_splitter(self, style, widgets):
        if style == 'h' or style == 'horizontal':
            splitterWidget = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        else:
            splitterWidget = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        for widget in widgets:
            splitterWidget.addWidget(widget)

        splitterWidget.setStyleSheet(
            qss_format(
                qss.containerstyle.frame + qss.containerstyle.splitter,
                self.app.palette
            )
        )
        QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        return splitterWidget

    def make_scroll_area_tab(
        self, parent, text, master=None, layout=None,
        stylesheet=qss.containerstyle.tab
    ):
        tab = QtWidgets.QScrollArea()
        tab.setWidget(QtWidgets.QWidget())
        tab.setStyleSheet(qss_format(stylesheet, self.app.palette))
        tabLayout = self.make_grid(tab.widget())
        tab.setWidgetResizable(True)
        master.addTab(tab, text)
        return tabLayout

    def make_statusbar(self, parent, layout, row, column):
        statusbar = QtWidgets.QStatusBar(parent)
        statusbar.setStyleSheet(qss_format(qss.statusbar.statusbar, self.app.palette))
        layout.addWidget(statusbar, row, column)
        return statusbar

    def menu_bar(self, parent, submenus, stylesheet=qss.barstyle.menubar):
        menubar = QtWidgets.QMenuBar(parent)
        menubar.setStyleSheet(qss_format(stylesheet, self.app.palette))
        for submenu in submenus:
            menubar.addMenu(submenu)
        return menubar

    def open_files(self, parent, multiple=False, fileExt='', label=None, command=None):
        ALL_FILES = ''
        if not multiple:
            paths = QtWidgets.QFileDialog.getOpenFileName(
                parent, 'Select file', os.getcwd(),
                fileExt + ";;" + ALL_FILES
            )
        else:
            paths = QtWidgets.QFileDialog.getOpenFileNames(parent, 'Select files', os.getcwd(), fileExt)
        if label and paths:
            label.setText(str(paths))
            label.adjustSize()
        if command:
            command()
        if paths:
            return paths

    def open_folder(self, parent, label=None, default=os.getcwd(), command=None):
        dir = QtWidgets.QFileDialog.getExistingDirectory(parent, 'Select folder', default)
        if dir:
            label.setText(str(dir))
            label.adjustSize()
            if command:
                command()
            return dir

    def remove_tabs(tabsWidget):
        for tab in range(tabsWidget.count() - 1):
            tabsWidget.removeTab(1)

    def save_file(self, parent, fileExt='', label=None):
        paths = QtWidgets.QFileDialog.getSaveFileName(parent, 'Select file', os.getcwd(), fileExt)
        if label and paths:
            label.setText(str(paths[0]))
            label.adjustSize()
        if paths:
            return paths[0]

    def set_tabs(self, widget, tabNames, layout, scroll=False):
        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet(qss_format(qss.containerstyle.tab, self.app.palette))
        # tabs.setMinimumSize(self.parent.palette.minHeight*1, self.parent.palette.minHeight)
        layout.addWidget(tabs, 0, 1)
        tabDict = {}
        for name in tabNames:
            if scroll:
                tabDict[name] = self.make_scroll_area_tab(widget, name, tabs)
            else:
                tabDict[name] = self.tab_area(widget, name, tabs)
        return tabDict

    def subdivision(self, parent):
        subdivision = self.make_frame(parent)
        layout = self.make_grid(subdivision)
        return subdivision, layout

    def tab_area(self, parent, text, master, layout=None, stylesheet=qss.containerstyle.tab):
        tab = QtWidgets.QTabWidget()
        tab.setStyleSheet(qss_format(stylesheet, self.app.palette))
        tabLayout = self.make_grid(tab)
        master.addTab(tab, text)
        return tabLayout

    def make_file_dialog(self, widget, icon=None, text='', tooltipButton='', tooltipLabel=''):
        label = self.make_label(widget, text=text, width=200, )
        command = lambda: self.open_folder(widget, label=label, command=widget.rewindow, )
        button = self.make_button(widget, command=command, icon=icon, width=50, )
        button.setToolTip(tooltipButton)
        label.setToolTip(tooltipLabel.format(label.text()))
        return label, button

    def insert_file_dialog(self, widget, container, icon, text, row, column, tooltipButton, tooltipLabel):
        label, button = self.make_file_dialog(
            widget=widget, icon=icon, text=text, tooltipButton=tooltipButton,
            tooltipLabel=tooltipLabel, )
        container.addWidget(button, row, column, )
        container.addWidget(label, row, column + 1, )
        return label, button

    def make_numerical_input(
        self, parent, text='', inputSettings=(0, 1, 0, 0), width=150, layout=None, row=0, column=0,
        tooltip='', command=None, decimals=None, height=None
    ):
        if decimals:
            entry = TSInputFloat(self.app, parent, inputSettings, decimals=decimals)
        else:
            entry = TSInputInteger(self.app, parent, inputSettings)
        self.resize_widget(entry, width, height)
        if command:
            entry.valueChanged.connect(command)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(entry, row, column)
            else:
                layout.layout().addWidget(entry)
        entry.setToolTip(tooltip)
        return entry

    def make_slider(
        self, parent=None, layout=None, row=0, column=0, tooltip='', command=None, width=None, height=None,
        inputSettings=(0, 0, 0, 0)
    ):
        sliderWidget = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sliderWidget.setRange(inputSettings[0], inputSettings[1])
        sliderWidget.setValue(inputSettings[2])
        sliderWidget.setSingleStep(inputSettings[3])
        sliderWidget.setToolTip(tooltip)
        # sliderWidget.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.resize_widget(sliderWidget, width, height)
        sliderWidget.setMinimumHeight(self.app.palette.minHeight)
        if command:
            sliderWidget.valueChanged.connect(command)
        if type(layout) == QtWidgets.QGridLayout:
            layout.addWidget(sliderWidget, row, column)
        else:
            layout.layout().addWidget(sliderWidget)
        return sliderWidget

    def resize_widget(self, widget, width, height):
        if width:
            widget.setFixedWidth(width)
        if height:
            widget.setFixedHeight(height)

    def __init__(self, parent):
        self.parent = parent

    def add_toolbar(self, parent, command=None, text='', icon=None, tooltip=None, shortcut=None, size=(400, 400)):
        """
        parent: QWidget
        icon: str (file path)
        tooltip: str
        shortcut: str
        return:
        """
        toolbar = QtWidgets.QToolBar(parent)
        toolbar.setStyleSheet(qss.format_qss(qss.toolbar, self.parent.palette))
        toolbar.setIconSize(QtCore.QSize(150, 150))
        toolbar.setFixedSize(size[0], size[1])
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        qIcon = QtGui.QIcon(icon)
        action = QtWidgets.QAction(qIcon, '', parent)
        if shortcut:
            action.setShortcut(shortcut)
        if command:
            action.triggered.connect(command)
        action.setIconText(text)
        action.setToolTip(tooltip)
        toolbar.addAction(action)
        parent.addToolBar(toolbar)
        return toolbar

    def add_action(self, parent, command=None, text='', icon=None, tooltip=None, shortcut=None):
        """
        parent: QWidget
        icon: str (file path)
        tooltip: str
        shortcut: str
        return:
        """
        qIcon = QtGui.QIcon(icon)
        action = QtWidgets.QAction(qIcon, '', parent)
        if shortcut:
            action.setShortcut(shortcut)
        if command:
            action.triggered.connect(command)
        action.setIconText(text)
        action.setToolTip(tooltip)
        return action

    def load_browser(self, browser, file, online=False):
        try:
            if online:
                browser.load(QtCore.QUrl(file))
            else:
                browser.load(QtCore.QUrl.fromLocalFile(file))
        except:
            pass
        return browser

    def make_browser(self, layout, file, online=False):
        browser = QtWebEngineWidgets.QWebEngineView()
        if file:
            self.load_browser(browser, file, online)
        if layout != None:
            layout.addWidget(browser)
        return browser

    def make_button(
        self, parent=None, buttonType=None, command=None, text='', icon=None, layout=None, row=0, column=0,
        width=None, tooltip='', isChecked=False, stylesheet=qss.button, height=None
    ):
        if buttonType == 'radio':
            button = QtWidgets.QRadioButton(parent)
            button.setStyleSheet(qss.format_qss(qss.radio, self.parent.palette))
        elif buttonType == 'check':
            button = QtWidgets.QCheckBox(parent)
            if isChecked:
                button.setChecked(True)
                button.setStyleSheet(qss.format_qss(qss.checkboxChecked, self.parent.palette))
            else:
                button.setStyleSheet(qss.format_qss(qss.checkboxUnchecked, self.parent.palette))
        else:
            if icon:
                icn = QtGui.QIcon(icon)
                button = QtWidgets.QPushButton(icn, text, parent)
                button.setIconSize(QtCore.QSize(20, 20))
            else:
                button = QtWidgets.QPushButton(text, parent)
            if stylesheet:
                button.setStyleSheet(qss.format_qss(stylesheet, self.parent.palette))
        button.setToolTip(tooltip)
        # button.setvariables.FONT(QtGui.Qvariables.FONT(variables.FONT))
        if command:
            button.clicked.connect(command)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(button, row, column)
            else:
                layout.layout().addWidget(button)
        # button.adjustSize()
        button.setMinimumSize(self.parent.palette.minHeight, self.parent.palette.minHeight)
        button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.resize_widget(button, width, height)
        return button

    def make_combo(
        self, parent=None, items=None, command=None, layout=None, row=0, column=0, width=40, height=20,
        tooltip='', default=None
    ):
        combo = QtWidgets.QComboBox(parent)
        self.resize_widget(combo, width, height)
        combo.setStyleSheet(qss.format_qss(qss.dropdown, self.parent.palette))
        combo.addItems(items)
        if default:
            combo.setCurrentIndex(default)
        if command:
            combo.activated[str].connect(command)
        combo.setToolTip(tooltip)
        if layout:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(combo, row, column)
            else:
                layout.layout().addWidget(combo)
        combo.adjustSize()
        combo.setMinimumHeight(self.parent.palette.minHeight)
        return combo

    def make_dial(
        self, parent=None, min=None, max=None, value=None, step=1, layout=None, row=0, column=0, tooltip='',
        command=None
    ):
        dial = QtWidgets.QDial(parent)
        dial.setStyleSheet(qss.format_qss(qss.dial, self.parent.palette))
        self.resize_widget(dial, 30, 30)
        dial.setMinimum(min)
        dial.setMaximum(max)
        dial.setNotchesVisible(True)
        dial.setSingleStep(step)
        dial.setWrapping(False)
        if not value:
            dial.setValue(min)
        else:
            dial.setValue(value)
        dial.setToolTip(tooltip)
        if command:
            dial.valueChanged.connect(command)
        if layout:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(dial, row, column)
            else:
                layout.addWidget(dial)
        return dial

    def make_entry(
        self, parent, text='', width=None, height=None, layout=None, row=0, column=0, tooltip=None,
        command=None, number=False, readOnly=False, area=False, stylesheet=qss.entry
    ):
        if area:
            entry = QtWidgets.QPlainTextEdit(parent)
            if text:
                entry.setPlainText(text)
        else:
            entry = QtWidgets.QLineEdit(parent)
            if text:
                entry.setText(text)
        if command:
            entry.textChanged.connect(command)
        # entry.setvariables.FONT(QtGui.Qvariables.FONT(variables.FONT))
        self.resize_widget(entry, width, height)
        entry.setMinimumHeight(self.parent.palette.minHeight)
        entry.setReadOnly(readOnly)
        entry.setToolTip(tooltip)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(entry, row, column)
            else:
                layout.layout().addWidget(entry)
        if stylesheet:
            entry.setStyleSheet(qss.format_qss(stylesheet, self.parent.palette))
        # entry.setMinimumHeight(40)
        return entry

    def make_form(self, parent=None, padding=False):
        form = QtWidgets.QFormLayout(parent)
        if padding:
            form.setContentsMargins(0, self.parent.palette.topPadding, 0, self.parent.palette.bottomPadding)
        return form

    def make_frame(self, parent):
        frame = QtWidgets.QFrame(parent)
        frame.setStyleSheet(qss.format_qss(qss.frame, self.parent.palette))
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        return frame

    def make_group_box(
        self, parent, layout=None, text='', row=0, column=0, width=None, height=None, border=True,
        strictWidth=False, strictHeight=False, strictSize=False, columnSpan=1, tooltip=None
    ):
        groupBox = QtWidgets.QGroupBox(text)
        groupBox.setStyleSheet(qss.format_qss(qss.groupBox, self.parent.palette, ) + '' if border else qss.groupBoxNu)
        groupBox.setLayout(layout) if layout else groupBox.setLayout(self.make_grid())
        parent.addWidget(groupBox, row, column, 1, columnSpan)
        if strictWidth or width:
            groupBox.setFixedWidth(width)
        if strictHeight or height:
            groupBox.setFixedHeight(height)
        if strictSize:
            groupBox.setFixedSize(width, height)
        if tooltip:
            groupBox.setToolTip(tooltip)
        if border:
            groupBox.layout().setContentsMargins(1, 1, 1, 1)
        else:
            groupBox.layout().setContentsMargins(1, 4, 1, 4)
        groupBox.layout().setSpacing(0)
        return groupBox

    def make_grid(self, parent=None, padding=False):
        grid = QtWidgets.QGridLayout(parent)
        if padding:
            grid.setContentsMargins(0, topPadding, 0, bottomPadding)
        return grid

    def make_label(self, parent, text='', layout=None, row=0, column=0, width=None, height=None, tooltip=''):
        labelWidget = QtWidgets.QLabel(text, parent)
        labelWidget.setWordWrap(True)
        self.resize_widget(labelWidget, width, height)
        labelWidget.setToolTip(tooltip)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(labelWidget, row, column)
            else:
                layout.addWidget(labelWidget)

        # label.setvariables.FONT(QtGui.Qvariables.FONT(variables.FONT))
        labelWidget.setStyleSheet(qss.format_qss(qss.label, self.parent.palette))
        labelWidget.adjustSize()
        return labelWidget

    def make_list_widget(
        self, parent=None, items=None, command=None, default=None, layout=None, row=0, column=0,
        width=None, height=None, multiSelect=False
    ):
        listWidget = QtWidgets.QListWidget(parent)
        if items:
            listWidget.addItems(items)
        listWidget.setStyleSheet(qss.format_qss(qss.list, self.parent.palette))
        if default != None:
            listWidget.setCurrentRow(default)
        if command:
            listWidget.currentItemChanged.connect(command)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(listWidget, row, column)
            else:
                layout.addWidget(listWidget)
        self.resize_widget(listWidget, width, height)
        if multiSelect:
            listWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        return listWidget

    def make_table(
        self, parent=None, items=None, command=None, default=None, layout=None, row=0, column=0, width=None,
        height=None, multiSelect=False
    ):
        table = QtWidgets.QTableWidget(parent)
        # if items:
        #     table.addItems(items)
        table.setStyleSheet(qss.format_qss(qss.table, self.parent.palette))
        # if default != None:
        #     table.setCurrentRow(default)
        # if command:
        #     table.currentItemChanged.connect(command)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(table, row, column)
            else:
                layout.addWidget(table)
        # self.resize_widget(table, width, height)
        # if multiSelect:
        #     table.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        return table

    def pair(
        self, parent=None, pair=None, text=None, layout=None, row=0, column=0, labelWidth=None, labelHeight=None,
        comboItems=None, command=None, pairText=None, pairWidth=None, pairHeight=None,
        inputSettings=(None, None, None, None), tooltip='', label=False, pairKwargs={}, default=None,
        readOnly=False, pairStyleSheet=None, area=False, columnSpan=1
    ):
        if label:
            labelWidget = self.make_label(parent, text, layout, row, column, labelWidth, labelHeight, tooltip)
            lay = layout
        else:
            grid = self.make_group_box(
                layout, layout=self.make_form(padding=True), text=text, row=row, column=column,
                border=False, columnSpan=columnSpan
            )
            lay = grid
        if pair == 'check':
            pair = self.make_button(parent, 'check', layout=lay, **pairKwargs)
        if pair == 'combo':
            pair = self.make_combo(
                parent=parent, items=comboItems, command=command, layout=lay, row=row,
                column=column + 1, width=pairWidth, height=pairHeight, tooltip=tooltip,
                default=default
            )
        if pair == 'dial':
            pair = self.make_dial(
                parent, inputSettings[0], inputSettings[1], inputSettings[2], inputSettings[3], lay,
                row, column + 1, tooltip, command
            )
        if pair == 'slider':
            pair = self.make_slider(
                parent, inputSettings=inputSettings, layout=lay, row=row, column=column + 1,
                tooltip=tooltip, command=command, width=pairWidth, height=pairHeight,
                stylesheet=pairStyleSheet, )
        if pair == 'entry':
            pair = self.make_entry(
                parent, text=pairText, width=pairWidth, layout=lay, row=row, column=column + 1,
                tooltip=tooltip, command=command, readOnly=readOnly, stylesheet=pairStyleSheet,
                area=area, )
        if pair == 'radio':
            pair = self.make_button(parent, 'radio', command, '', None, lay, row, column + 1, pairWidth, tooltip, )
        if pair == 'numerical':
            pair = self.make_numerical_input(
                parent, inputSettings=inputSettings, layout=lay, row=row,
                column=column + 1, tooltip=tooltip, command=command, width=pairWidth,
                height=pairHeight, readOnly=readOnly, stylesheet=pairStyleSheet, )
        if label:
            return labelWidget, pair
        return grid, pair

    def progress_bar(self, layout):
        p = QtWidgets.QProgressBar()
        layout.addWidget(p)
        p.setGeometry(30, 40, 200, 25)
        p.show()
        return p

    def make_scroll_area(self, layout, width=None, height=None):
        widget = QtWidgets.QWidget()
        widget.setStyleSheet(qss.format_qss(qss.widget, self.parent.palette))
        widget.setLayout(layout)
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setFrameShape(QtWidgets.QScrollArea.StyledPanel)
        scrollArea.setWidgetResizable(True)
        scrollArea.setStyleSheet(qss.format_qss(qss.frame + qss.scroll, self.parent.palette))
        scrollArea.setWidget(widget)
        self.resize_widget(scrollArea, width, height)
        return scrollArea

    def make_splitter(self, style, widgets):
        if style == 'h' or style == 'horizontal':
            splitterWidget = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        else:
            splitterWidget = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        for widget in widgets:
            splitterWidget.addWidget(widget)

        splitterWidget.setStyleSheet(qss.format_qss(qss.frame + qss.splitter, self.parent.palette))
        QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        return splitterWidget

    def make_scroll_area_tab(self, parent, text, master=None, layout=None, stylesheet=qss.tab):
        tab = QtWidgets.QScrollArea()
        tab.setWidget(QtWidgets.QWidget())
        tab.setStyleSheet(qss.format_qss(stylesheet, self.parent.palette))
        tabLayout = self.make_grid(tab.widget())
        tab.setWidgetResizable(True)
        master.addTab(tab, text)
        return tabLayout

    def make_statusbar(self, parent, layout, row, column):
        statusbar = QtWidgets.QStatusBar(parent)
        statusbar.setStyleSheet(qss.format_qss(qss.statusbar, self.parent.palette))
        layout.addWidget(statusbar, row, column)
        return statusbar

    def menu_bar(self, parent, submenus, stylesheet=qss.menubar):
        menubar = QtWidgets.QMenuBar(parent)
        menubar.setStyleSheet(qss.format_qss(stylesheet, self.parent.palette))
        for submenu in submenus:
            menubar.addMenu(submenu)
        return menubar

    def open_files(self, parent, multiple=False, fileExt='', label=None, command=None):
        ALL_FILES = ''
        if not multiple:
            paths = QtWidgets.QFileDialog.getOpenFileName(
                parent, 'Select file', os.getcwd(),
                fileExt + ";;" + ALL_FILES
            )
        else:
            paths = QtWidgets.QFileDialog.getOpenFileNames(parent, 'Select files', os.getcwd(), fileExt)
        if label and paths:
            label.setText(str(paths))
            label.adjustSize()
        if command:
            command()
        if paths:
            return paths

    def open_folder(self, parent, label=None, default=os.getcwd(), command=None):
        dir = QtWidgets.QFileDialog.getExistingDirectory(parent, 'Select folder', default)
        if dir:
            label.setText(str(dir))
            label.adjustSize()
            if command:
                command()
            return dir

    def remove_tabs(tabsWidget):
        for tab in range(tabsWidget.count() - 1):
            tabsWidget.removeTab(1)

    def save_file(self, parent, fileExt='', label=None):
        paths = QtWidgets.QFileDialog.getSaveFileName(parent, 'Select file', os.getcwd(), fileExt)
        if label and paths:
            label.setText(str(paths[0]))
            label.adjustSize()
        if paths:
            return paths[0]

    def set_tabs(self, widget, tabNames, layout, scroll=False):
        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet(qss.format_qss(qss.tab, self.parent.palette))
        # tabs.setMinimumSize(self.parent.palette.minHeight*2, self.parent.palette.minHeight)
        layout.addWidget(tabs, 0, 1)
        for name in tabNames:
            if scroll:
                yield self.make_scroll_area_tab(widget, name, tabs)
            else:
                yield self.tab_area(widget, name, tabs)

    def subdivision(self, parent):
        subdivision = self.make_frame(parent)
        layout = self.make_grid(subdivision)
        return subdivision, layout

    def tab_area(self, parent, text, master, layout=None, stylesheet=qss.tab):
        tab = QtWidgets.QTabWidget()
        tab.setStyleSheet(qss.format_qss(stylesheet, self.parent.palette))
        tabLayout = self.make_grid(tab)
        master.addTab(tab, text)
        return tabLayout

    def make_file_dialog(self, widget, icon=None, text='', tooltipButton='', tooltipLabel=''):
        label = self.make_label(widget, text=text, width=200, )
        command = lambda: self.open_folder(widget, label=label, command=widget.rewindow, )
        button = self.make_button(widget, command=command, icon=icon, width=50, )
        button.setToolTip(tooltipButton)
        label.setToolTip(tooltipLabel.format(label.text()))
        return label, button

    def insert_file_dialog(self, widget, container, icon, text, row, column, tooltipButton, tooltipLabel):
        label, button = self.make_file_dialog(
            widget=widget, icon=icon, text=text, tooltipButton=tooltipButton,
            tooltipLabel=tooltipLabel, )
        container.addWidget(button, row, column, )
        container.addWidget(label, row, column + 1, )
        return label, button

    def make_numerical_input(
        self, parent, text='', inputSettings=(0, 1, 0, 0), width=150, layout=None, row=0, column=0,
        tooltip='', command=None, decimals=0, height=None, readOnly=False, stylesheet=None
    ):
        if decimals > 0:
            entry = QtWidgets.QDoubleSpinBox(parent, decimals=decimals)
        else:
            entry = QtWidgets.QSpinBox(parent)
        entry.setRange(inputSettings[0], inputSettings[1])
        entry.setValue(inputSettings[2])
        entry.setSingleStep(inputSettings[3])
        self.resize_widget(entry, width, height)
        entry.setMinimumHeight(self.parent.palette.minHeight)
        if command:
            entry.valueChanged.connect(command)
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(entry, row, column)
            else:
                layout.layout().addWidget(entry)
        entry.setToolTip(tooltip)
        entry.setReadOnly(readOnly)
        if stylesheet:
            entry.setStyleSheet(qss.format_qss(stylesheet, self.parent.palette))
        else:
            entry.setStyleSheet(qss.format_qss(qss.spin, self.parent.palette))
        return entry

    def make_slider(
        self, parent=None, layout=None, row=0, column=0, tooltip='', command=None, width=None, height=None,
        inputSettings=(0, 0, 0, 0), stylesheet=None
    ):
        if not stylesheet:
            stylesheet = qss.slider
        sliderWidget = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sliderWidget.setRange(inputSettings[0], inputSettings[1])
        sliderWidget.setValue(inputSettings[2])
        sliderWidget.setSingleStep(inputSettings[3])
        sliderWidget.setToolTip(tooltip)
        sliderWidget.setStyleSheet(qss.format_qss(stylesheet, self.parent.palette))
        # sliderWidget.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.resize_widget(sliderWidget, width, height)
        sliderWidget.setMinimumHeight(self.parent.palette.minHeight)
        if command:
            sliderWidget.valueChanged.connect(command)
        if type(layout) == QtWidgets.QGridLayout:
            layout.addWidget(sliderWidget, row, column)
        else:
            layout.layout().addWidget(sliderWidget)
        return sliderWidget

    def resize_widget(self, widget, width, height):
        if width:
            widget.setFixedWidth(width)
        if height:
            widget.setFixedHeight(height)

    def __init__(self, app):
        self.app = app

    def make_browser(self, parent, layout):
        widget = TSBrowser(app=self.app, parent=parent, )
        self.set_layout(widget, layout)
        return widget

    def make_button(
        self, parent=None, buttonType=None, command=None, text='', icon=None, layout=None, row=0, column=0,
        width=None, tooltip='', checked=False, height=None
    ):
        if buttonType == 'radio':
            button = TSInputRadioButton(self.app, parent)
        elif buttonType == 'check':
            button = TSInputCheckBox(self.app, parent, checked, tooltip, command)
        else:
            button = PushButton(self.app, parent, text, icon, tooltip, command)
        self.set_layout(button, layout, row, column)
        self.resize_widget(button, width, height)
        return button

    def make_combo(
        self, parent=None, items=None, command=None, layout=None, row=0, column=0, width=40, height=20,
        tooltip='', default=None
    ):
        widget = TSInputCombo(
            app=self.app, parent=parent, options=items, default=default, command=command,
            tooltip=tooltip, )
        self.set_layout(widget, layout, row, column)
        self.resize_widget(widget, width, height)
        return widget

    def make_dial(self, parent, layout=None, settings=(0, 100, 0, 1), row=0, column=0, command=None, tooltip=''):
        widget = TSInputDial(app=self.app, parent=parent, settings=settings, command=command, tooltip=tooltip, )
        self.resize_widget(widget, 30, 30)
        self.set_layout(widget, layout, row, column)
        return widget

    def make_entry(
        self, parent, text='', width=None, height=None, layout=None, row=0, column=0, tooltip=None,
        command=None, readOnly=False, area=False
    ):
        kwargs = dict(app=self.app, parent=parent, command=command, readOnly=readOnly, text=text, tooltip=tooltip)
        if area:
            entry = TSInputTextArea(**kwargs)
        else:
            entry = TSInputTextLine(**kwargs)
        self.resize_widget(entry, width, height)
        self.set_layout(entry, layout, row, column)
        return entry

    def make_form(self, parent=None, padding=False):
        form = QtWidgets.QFormLayout(parent)
        if padding:
            form.setContentsMargins(0, self.app.palette.topPadding, 0, self.app.palette.bottomPadding)
        return form

    def make_frame(self, parent):
        frame = QtWidgets.QFrame(parent)
        frame.setStyleSheet(qss_format(qss.containerstyles.frame, self.app.palette))
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        return frame

    def make_group_box(
        self, parent, layout=None, text='', row=0, column=0, width=None, height=None, border=True,
        strictWidth=False, strictHeight=False, strictSize=False, columnSpan=1, tooltip=None
    ):
        groupBox = QtWidgets.QGroupBox(text)
        groupBox.setStyleSheet(
            qss_format(
                qss.containerstyles.groupBox, self.app.palette, ) + '' if border else qss.containerstyles.groupBoxNu
        )
        groupBox.setLayout(layout) if layout else groupBox.setLayout(self.make_grid())
        parent.addWidget(groupBox, row, column, 1, columnSpan)
        if strictWidth or width:
            groupBox.setFixedWidth(width)
        if strictHeight or height:
            groupBox.setFixedHeight(height)
        if strictSize:
            groupBox.setFixedSize(width, height)
        if tooltip:
            groupBox.setToolTip(tooltip)
        if border:
            groupBox.layout().setContentsMargins(1, 1, 1, 1)
        else:
            groupBox.layout().setContentsMargins(1, 4, 1, 4)
        groupBox.layout().setSpacing(0)
        return groupBox

    def make_grid(self, parent=None, padding=False):
        grid = QtWidgets.QGridLayout(parent)
        if padding:
            grid.setContentsMargins(0, 1, 0, 1)
        return grid

    def make_label(self, parent, text='', layout=None, row=0, column=0, width=None, height=None, tooltip=''):
        widget = TSLabel(app=self.app, parent=parent, text=text, tooltip=tooltip, )
        self.resize_widget(widget, width, height)
        self.set_layout(widget, layout, row, column)
        return widget

    def make_list_widget(
        self, parent=None, items=None, command=None, default=0, layout=None, row=0, column=0,
        width=None, height=None, multiSelect=False, tooltip=''
    ):
        listWidget = TSListSelect(
            app=self.app, parent=parent, items=items, default=default,
            multiSelect=multiSelect, command=command, tooltip=tooltip, )
        self.set_layout(listWidget, layout, row, column)
        self.resize_widget(listWidget, width, height)
        return listWidget

    def make_log_table(self, parent=None, layout=None, row=0, column=0, tooltip=''):
        widget = TSLogTable(app=self.app, parent=parent, tooltip=tooltip, )
        self.set_layout(widget, layout, row, column)
        return widget

    def pair(
        self, parent=None, pair=None, text=None, layout=None, row=0, column=0, labelWidth=None, labelHeight=None,
        comboItems=None, command=None, pairText=None, pairWidth=None, pairHeight=None,
        settings=(None, None, None, None), tooltip='', label=False, pairKwargs={}, default=None, readOnly=False,
        pairSS=None, area=False, columnSpan=1, gbHeight=None, gbWidth=None
    ):
        if label:
            labelWidget = self.make_label(parent, text, layout, row, column, labelWidth, labelHeight, tooltip)
            lay = layout
        else:
            grid = self.make_group_box(
                layout, layout=self.make_form(padding=True), text=text, row=row, column=column,
                border=False, width=gbWidth, height=gbHeight, columnSpan=columnSpan
            )
            lay = grid
        if pair == 'check':
            pair = self.make_button(parent, 'check', layout=lay, **pairKwargs)
        if pair == 'combo':
            pair = self.make_combo(
                parent=parent, items=comboItems, command=command, layout=lay, row=row,
                column=column + 1, width=pairWidth, height=pairHeight, tooltip=tooltip,
                default=default
            )
        if pair == 'dial':
            pair = self.make_dial(parent, settings, lay, row, column + 1, tooltip, command)
        if pair == 'slider':
            pair = self.make_slider(
                parent, settings=settings, layout=lay, row=row, column=column + 1, tooltip=tooltip,
                command=command, width=pairWidth, height=pairHeight, )
        if pair == 'entry':
            pair = self.make_entry(
                parent, text=pairText, width=pairWidth, layout=lay, height=pairHeight, row=row,
                column=column + 1, tooltip=tooltip, command=command, area=area, )
        if pair == 'radio':
            pair = self.make_button(parent, 'radio', command, '', None, lay, row, column + 1, pairWidth, tooltip, )
        if pair == 'numerical':
            pair = self.make_numerical_input(
                parent, settings=settings, layout=lay, row=row, column=column + 1,
                tooltip=tooltip, command=command, width=pairWidth, height=pairHeight, )
        if label:
            return labelWidget, pair
        return grid, pair

    def make_scroll_area_tab(self, parent, text, master=None, layout=None, stylesheet=qss.containerstyles.tab):
        tab = QtWidgets.QScrollArea()
        tab.setWidget(QtWidgets.QWidget())
        tab.setStyleSheet(qss_format(stylesheet, self.app.palette))
        tabLayout = self.make_grid(tab.widget())
        tab.setWidgetResizable(True)
        master.addTab(tab, text)
        return tabLayout

    def make_statusbar(self, parent, layout, row, column):
        widget = TSStatusBar(app=self.app, parent=parent, )
        layout.addWidget(widget, row, column)
        return widget

    def menu_bar(self, parent, submenus, stylesheet=qss.barstyles.menubar):
        menubar = QtWidgets.QMenuBar(parent)
        menubar.setStyleSheet(qss_format(stylesheet, self.app.palette))
        for submenu in submenus:
            menubar.addMenu(submenu)
        return menubar

    def open_folder(self, parent, label=None, default=os.getcwd(), command=None):
        dir = QtWidgets.QFileDialog.getExistingDirectory(parent, 'Select folder', default)
        if dir:
            label.setText(str(dir))
            label.adjustSize()
            if command:
                command()
            return dir

    def set_tabs(self, widget, tabNames, layout, scroll=False):
        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet(qss_format(qss.containerstyles.tab, self.app.palette))
        # tabs.setMinimumSize(self.parent.palette.minHeight*2, self.parent.palette.minHeight)
        layout.addWidget(tabs, 0, 1)
        tabDict = {}
        for name in tabNames:
            if scroll:
                tabDict[name] = self.make_scroll_area_tab(widget, name, tabs)
            else:
                tabDict[name] = self.tab_area(widget, name, tabs)
        return tabDict

    def subdivision(self, parent):
        subdivision = self.make_frame(parent)
        layout = self.make_grid(subdivision)
        return subdivision, layout

    def tab_area(self, parent, text, master, layout=None, stylesheet=qss.containerstyles.tab):
        tab = QtWidgets.QTabWidget()
        tab.setStyleSheet(qss_format(stylesheet, self.app.palette))
        tabLayout = self.make_grid(tab)
        master.addTab(tab, text)
        return tabLayout

    def make_file_dialog(self, widget, icon=None, text='', tooltipButton='', tooltipLabel=''):
        label = self.make_label(widget, text=text, width=200, )
        command = lambda: self.open_folder(widget, label=label, command=widget.rewindow, )
        button = self.make_button(widget, command=command, icon=icon, width=50, )
        button.setToolTip(tooltipButton)
        label.setToolTip(tooltipLabel.format(label.text()))
        return label, button

    def insert_file_dialog(self, widget, container, icon, text, row, column, tooltipButton, tooltipLabel):
        label, button = self.make_file_dialog(
            widget=widget, icon=icon, text=text, tooltipButton=tooltipButton,
            tooltipLabel=tooltipLabel, )
        container.addWidget(button, row, column, )
        container.addWidget(label, row, column + 1, )
        return label, button

    def make_numerical_input(
        self, parent, settings=(0, 1, 0, 0), width=150, layout=None, row=0, column=0, tooltip='',
        command=None, readOnly=False, decimals=None, height=None
    ):
        kwargs = dict(
            app=self.app, parent=parent, settings=settings, command=command, readOnly=readOnly,
            tooltip=tooltip
        )
        if decimals:
            kwargs['decimals'] = decimals
            widget = TSInputFloat(**kwargs)
        else:
            widget = TSInputInteger(**kwargs)
        self.resize_widget(widget, width, height)
        self.set_layout(widget, layout, row, column)
        return widget

    def make_slider(
        self, parent=None, layout=None, row=0, column=0, tooltip='', command=None, width=None, height=None,
        settings=(0, 0, 0, 0), direction='h'
    ):
        widget = TSInputSlider(
            app=self.app, parent=parent, settings=settings, direction=direction, command=command,
            tooltip=tooltip, )
        self.resize_widget(widget, width, height)
        self.set_layout(widget, layout, row, column)
        return widget

    def resize_widget(self, widget, width, height):
        if width:
            widget.setFixedWidth(width)
        if height:
            widget.setFixedHeight(height)

    def set_layout(self, widget, layout, row=0, column=0):
        if layout != None:
            if type(layout) == QtWidgets.QGridLayout:
                layout.addWidget(widget, row, column)
            else:
                layout.layout().addWidget(widget)

    def __init__(self):
        """All lists must be same length
            container is a tkinter frame
            text, labels, interface7 are each a list of strings
            commands is a list of functions to execute
            cursor is the cursor icon2 over this object
            row, column, columnspan, width are each a list of integers
            sticky is one or a combination of ('n', 's', 'w', 'e')
            tearoff is a boolean
            parent is a window object"""
        self.MASTERPATH = configuration.getMasterPath()
        self.temp_r_s_p = "\BridgeDataQuery\\Utilities14\Preferences\Temporary_Report_Preferences.txt"
        self.r_s_p = "\BridgeDataQuery\\Utilities14\Preferences\Report_Preferences.txt"
        self.mc_s_p = "\BridgeDataQuery\\Utilities14\Preferences\Markov_Chain_Preferences.txt"
        self.number_of_items = 142
        self.lists = ListGenerator.ListGenerator(self)
        # self.WIDTH = user32.GetSystemMetrics(0)
        # self.HEIGHT = user32.GetSystemMetrics(1)
        self.SMALL_FONT = ("Verdana", 8)
        self.MEDIUM_FONT = ("Verdana", 10)
        self.LARGE_FONT = ("Verdana", 18)

    @staticmethod
    def button(container, text, commands, cursor, row, column, sticky):
        for index in (range(len(row))):
            ttk.Button(container, text=text[index], command=commands[index], cursor=cursor).grid(
                row=row[index],
                column=column[index],
                sticky=sticky[index]
            )

    def check_button(self, container, commands, cursor, row, column, sticky):
        self.check_buttons = []
        for i in (range(len(row))):
            checkButton = Checkbutton(container, command=commands[i], cursor=cursor)
            checkButton.grid(row=row[i], column=column[i], sticky=sticky[i])
            self.check_buttons.append(checkButton)

    @staticmethod
    def label(container, text, row, column, columnspan, sticky):
        for index, label in enumerate(range(len(row))):
            ttk.Label(container, text=text[index]).grid(
                row=row[index], column=column[index],
                columnspan=columnspan[index], sticky=sticky[index]
            )

    def entry(self, container, width, row, column, columnspan, sticky):
        self.entries = []
        for index in (range(len(row))):
            entry = ttk.Entry(container, width=width[index])
            entry.grid(row=row[index], column=column[index], columnspan=columnspan[index], sticky=sticky[index])
            self.entries.append(entry)

    @staticmethod
    def separator(container, orient, row, column, columnspan, sticky, pady=5):
        ttk.Separator(container, orient=orient).grid(
            row=row, column=column, columnspan=columnspan, sticky=sticky,
            pady=pady
        )

    @staticmethod
    def drop_down_menu(parent, menus, tearoff, labels, commands):
        menubar = Menu(parent.container)
        for i, menu in enumerate(menus):
            menu = Menu(menubar, tearoff=tearoff[i])
            index = 0
            for label in (labels[i]):
                if label == 'separator()':
                    menu.add_separator()
                else:
                    menu.add_command(label=label, command=commands[i][index])
                    index += 1
            menubar.add_cascade(label=menus[i], menu=menu)

        parent.configuration.py(menu=menubar)

    @staticmethod
    def set_active(top):
        """Disable search windows so that only the most recent window can be interacted with"""
        top.grab_set()

    def __init__(self):
        """
        All lists must be same length
        container is a tkinter frame
        text, labels, interface7 are each a list of strings
        commands is a list of functions to execute
        cursor is the cursor icon2 over this object
        row, column, columnspan, width are each a list of integers
        sticky is one or a combination of ('n', 's', 'w', 'e')
        tearoff is a boolean
        parent is a window object
        """
        self.MASTERPATH = configuration.getMasterPath()
        self.temp_rSP = "\BridgeDataQuery\\Utilities14\Preferences\Temporary_Report_Preferences.txt"
        self.rSP = "\BridgeDataQuery\\Utilities14\Preferences\Report_Preferences.txt"
        self.mcSP = "\BridgeDataQuery\\Utilities14\Preferences\Markov_Chain_Preferences.txt"
        self.number_of_items = 142
        self.lists = ListGenerator.ListGenerator(self)
        # self.WIDTH = user32.GetSystemMetrics(0)
        # self.HEIGHT = user32.GetSystemMetrics(1)
        self.SMALL_FONT = ("Verdana", 8)
        self.MEDIUM_FONT = ("Verdana", 10)
        self.LARGE_FONT = ("Verdana", 18)

    @staticmethod
    def drop_down_menu(parent, menus, tearoff, labels, commands):
        menubar = Menu(parent.container)
        for i, menu in enumerate(menus):
            menu = Menu(menubar, tearoff=tearoff[i])
            index = 0
            for label in (labels[i]):
                if label == 'separator()':
                    menu.add_separator()
                else:
                    menu.add_command(label=label, command=commands[i][index])
                    index += 1
            menubar.add_cascade(label=menus[i], menu=menu)

        parent.configuration.py(menu=menubar)

    def __init__(self):
        """
        All lists must be same length
        container is a tkinter frame
        text, labels, interface7 are each a list of strings
        commands is a list of functions to execute
        cursor is the cursor icon2 over this object
        row, column, columnspan, width are each a list of integers
        sticky is one or a combination of ('n', 's', 'w', 'e')
        tearoff is a boolean
        parent is a window object
        """
        self.number_of_items = 142
        self.lists = ListGenerator(self)
        # self.WIDTH = user32.GetSystemMetrics(0)
        # self.HEIGHT = user32.GetSystemMetrics(1)
        self.SMALL_FONT = ("Verdana", 8)
        self.MEDIUM_FONT = ("Verdana", 10)
        self.LARGE_FONT = ("Verdana", 18)

    @staticmethod
    def label(container, text, row, column, columnspan, sticky):
        for index, label in enumerate(range(len(row))):
            ttk.Label(container, text=text[index]).grid(
                row=row[index], column=column[index],
                columnspan=columnspan[index], sticky=sticky[index], pady=5
            )

    @staticmethod
    def drop_down_menu(parent, menus, tearoff, labels, commands):
        menubar = Menu(parent.container)
        for i, menu in enumerate(menus):
            menu = Menu(menubar, tearoff=tearoff[i])
            index = 0
            for label in (labels[i]):
                if label == 'separator()':
                    menu.add_separator()
                else:
                    menu.add_command(label=label, command=commands[i][index])
                    index += 1
            menubar.add_cascade(label=menus[i], menu=menu)

        parent.configuration.py(menu=menubar)

    def __init__(self):
        """
        All lists must be same length
        container is a tkinter frame
        text, labels, interface7 are each a list of strings
        commands is a list of functions to execute
        cursor is the cursor icon2 over this object
        row, column, columnspan, width are each a list of integers
        sticky is one or a combination of ('n', 's', 'w', 'e')
        tearoff is a boolean
        parent is a window object
        """
        self.number_of_items = 144

    @staticmethod
    def button(container, text, commands, cursor, row, column, sticky=None):
        for i in (range(len(row))):
            ttk.Button(container, text=text[i], command=commands[i], cursor=cursor).grid(
                row=row[i], column=column[i],
                sticky=sticky[
                    i] if sticky else None
            )

    def check_button(self, container, commands, cursor, row, column, sticky=None):
        self.checkButtons = []
        for i in (range(len(row))):
            checkButton = Checkbutton(container, command=commands[i], cursor=cursor)
            checkButton.grid(row=row[i], column=column[i], sticky=sticky[i] if sticky else None)
            self.checkButtons.append(checkButton)

    def radio_button(self, container, text, variables, values, commands, cursor, row, column, sticky=None):
        self.radioButtons = []
        for i in (range(len(row))):
            radioButton = Radiobutton(
                container, text=text[i], variable=variables[i], value=values[i],
                command=commands[i]
            )
            radioButton.grid(row=row[i], column=column[i], sticky=sticky[i] if sticky else None)
            self.radioButtons.append(radioButton)

    @staticmethod
    def label(container, text, row, column, columnspan, sticky=None):
        for i, label in enumerate(range(len(row))):
            ttk.Label(container, text=text[i]).grid(
                row=row[i], column=column[i], columnspan=columnspan[i],
                sticky=sticky[i] if sticky else None, pady=5
            )

    def entry(self, container, width, row, column, columnspan, sticky=None):
        self.entries = []
        for i in (range(len(row))):
            entry = ttk.Entry(container, width=width[i])
            entry.grid(row=row[i], column=column[i], columnspan=columnspan[i], sticky=sticky[i] if sticky else None)
            self.entries.append(entry)

    @staticmethod
    def separator(container, orient, row, column, columnspan, sticky=None, pady=5):
        ttk.Separator(container, orient=orient).grid(
            row=row, column=column, columnspan=columnspan, sticky=sticky,
            pady=pady
        )

    @staticmethod
    def drop_down_menu(parent, menus, tearoff, labels, commands):
        try:
            menubar = Menu(parent.container)
        except:
            menubar = Menu(parent)
        for i, menu in enumerate(menus):
            menu = Menu(menubar, tearoff=tearoff[i])
            index = 0
            for label in (labels[i]):
                if label == 'separator()':
                    menu.add_separator()
                else:
                    menu.add_command(label=label, command=commands[i][index])
                    index += 1
            menubar.add_cascade(label=menus[i], menu=menu)

        parent.configuration.py(menu=menubar)

    @staticmethod
    def separator(container, orient, row, column, columnspan, sticky):
        ttk.Separator(container, orient=orient).grid(row=row, column=column, columnspan=columnspan, sticky=sticky)

    @staticmethod
    def drop_down_menu(parent, menus, tearoff, labels, commands):
        menubar = Menu(parent.container)
        for i, menu in enumerate(menus):
            menu = Menu(menubar, tearoff=tearoff[i])
            index = 0
            for label in (labels[i]):
                if label == 'separator()':
                    menu.add_separator()
                else:
                    menu.add_command(label=label, command=commands[i][index])
                    index += 1
            menubar.add_cascade(label=menus[i], menu=menu)

        parent.configuration.py(menu=menubar)

    @staticmethod
    def set_active(top):
        """Disable search windows so that only the most recent window can be interacted with"""
        top.grab_set()

    """Hosts functions for simplifying the creation of menus0"""

    def __init__(self):
        self.MASTERPATH = MasterPath_Object.getMasterPath()
        self.number_of_items = 142
        self.list_generator = ListGenerator.ListGenerator(self)

        self.SMALL_FONT = ("Verdana", 8)
        self.MEDIUM_FONT = ("Verdana", 10)
        self.LARGE_FONT = ("Verdana", 12)
        self.entries = {}

    def add_button(self, container, text, commands, cursor, row, column, sticky):
        """Add a button"""
        for index, button in enumerate(range(len(row))):
            Button(container, text=text[index], command=commands[index], cursor=cursor).grid(
                row=row[index],
                column=column[index],
                sticky=sticky[index]
            )

    def add_check_button(self, container, commands, cursor, row, column, sticky):
        """Add a button"""
        for index, button in enumerate(range(len(row))):
            Button(container, command=commands[index], cursor=cursor).grid(
                row=row[index], column=column[index],
                sticky=sticky[index]
            )

    def add_label(self, container, text, row, column, columnspan, sticky):
        for index, label in enumerate(range(len(row))):
            TSLabel(container, text=text[index]).grid(
                row=row[index], column=column[index], columnspan=columnspan[index],
                sticky=sticky[index]
            )

    def add_entry(self, container, labels, width, row, column, columnspan, sticky):
        for index, label in enumerate(labels):
            entry = Entry(container, width=width[index])
            entry.grid(row=row[index], column=column[index], columnspan=columnspan[index], sticky=sticky[index])
            self.entries[label] = (entry)

    def add_separator(self, container, orient, row, column, columnspan, sticky):
        """Add a separator"""
        ttk.Separator(container, orient=orient).grid(row=row, column=column, columnspan=columnspan, sticky=sticky)

    def add_drop_down_menu(self, parent, menus, tearoff, labels, commands):
        "Add a drop-down menu"
        menubar = Menu(parent.container)
        for i, menu in enumerate(menus):
            menu = Menu(menubar, tearoff=tearoff[i])
            index = 0
            for label in (labels[i]):
                if label == 'add_separator()':
                    menu.add_separator()
                else:
                    menu.add_command(label=label, command=commands[i][index])
                    index += 1
            menubar.add_cascade(label=menus[i], menu=menu)

        parent.configuration.py(menu=menubar)

    def set_active(self, top):
        top.grab_set()

    def __init__(self):
        """All lists must be same length
            container is a tkinter frame
            text, labels, menus0 are each a list of strings
            commands is a list of functions to execute
            cursor is the cursor icon over this object
            row, column, columnspan, width are each a list of integers
            sticky is one or a combination of ('n', 's', 'w', 'e')
            tearoff is a boolean
            parent is a window object"""
        self.MASTERPATH = MasterPath_Object.getMasterPath()
        self.temp_r_s_p = "\BridgeDataQuery\\Utilities1\Preferences\Temporary_Report_Preferences.txt"
        self.r_s_p = "\BridgeDataQuery\\Utilities1\Preferences\Report_Preferences.txt"
        self.mc_s_p = "\BridgeDataQuery\\Utilities1\Preferences\Markov_Chain_Preferences.txt"
        self.number_of_items = 142
        self.lists = ListGenerator.ListGenerator(self)
        self.entry_groups = {}
        self.WIDTH = user32.GetSystemMetrics(0)
        self.HEIGHT = user32.GetSystemMetrics(1)
        self.SMALL_FONT = ("Verdana", 8)
        self.MEDIUM_FONT = ("Verdana", 10)
        self.LARGE_FONT = ("Verdana", 18)

    @staticmethod
    def button(container, text, commands, cursor, row, column, sticky):
        for index in (range(len(row))):
            ttk.Button(container, text=text[index], command=commands[index], cursor=cursor).grid(
                row=row[index],
                column=column[index],
                sticky=sticky[index]
            )

    def check_button(self, container, commands, cursor, row, column, sticky):
        self.checkButtons = []
        for i in (range(len(row))):
            checkButton = Checkbutton(container, command=commands[i], cursor=cursor)
            checkButton.grid(row=row[i], column=column[i], sticky=sticky[i])
            self.checkButtons.append(checkButton)

    def radio_button(self, container, text, variables, values, commands, cursor, row, column, sticky):
        self.radioButtons = []
        for i in (range(len(row))):
            radioButton = ttk.Radiobutton(
                container, text=text[i], variable=variables[i], value=values[i],
                command=commands[i]
            )
            radioButton.grid(row=row[i], column=column[i], sticky=sticky[i])
            self.radioButtons.append(radioButton)

    @staticmethod
    def label(container, text, row, column, columnspan, sticky, padx=0):
        for index, label in enumerate(range(len(row))):
            TSLabel(container, text=text[index], borderwidth=1).grid(
                row=row[index], column=column[index],
                columnspan=columnspan[index], sticky=sticky[index],
                padx=padx
            )

    def entry(self, name, container, labels, width, row, column, columnspan, sticky):
        self.entry_groups[name] = MenuBuilder.create_entry(container, labels, width, row, column, columnspan, sticky)
        print(self.entry_groups[name])
        return self.entry_groups[name]

    @staticmethod
    def create_entry(container, labels, width, row, column, columnspan, sticky):
        entries = pd.DataFrame(
            index=[labels[0][i] for i, r in enumerate(row)],
            columns=[labels[1][j] for j, c in enumerate(column)]
        )
        entries.fillna(value=0, inplace=True)
        for col_index, c in enumerate(range(len(column))):
            for row_index, r in enumerate(range(len(row))):
                entry = ttk.Entry(container, width=width[col_index])
                entry.grid(row=row[r], column=column[c], columnspan=columnspan[col_index], sticky=sticky[col_index])
        return entries

    @staticmethod
    def set_entry(names, data):
        for name in names:
            for datum in data:
                print(name)
                print(datum)

    @staticmethod
    def separator(container, orient, row, column, columnspan, sticky, pady=5):
        ttk.Separator(container, orient=orient).grid(
            row=row, column=column, columnspan=columnspan, sticky=sticky,
            pady=pady
        )

    @staticmethod
    def drop_down_menu(parent, menus, tearoff, labels, commands):
        menubar = Menu(parent.container)
        for i, menu in enumerate(menus):
            menu = Menu(menubar, tearoff=tearoff[i])
            index = 0
            for label in (labels[i]):
                if label == 'separator()':
                    menu.add_separator()
                else:
                    menu.add_command(label=label, command=commands[i][index])
                    index += 1
            menubar.add_cascade(label=menus[i], menu=menu)

        parent.configuration.py(menu=menubar)

    @staticmethod
    def set_active(top):
        """Disable other windows so that only the most recent window can be interacted with"""
        top.grab_set()

    @staticmethod
    def drop_down_menu(parent, menus, tearoff, labels, commands):
        menubar = Menu(parent.container)
        for i, menu in enumerate(menus):
            menu = Menu(menubar, tearoff=tearoff[i])
            index = 0
            for label in (labels[i]):
                if label == 'separator()':
                    menu.add_separator()
                else:
                    menu.add_command(label=label, command=commands[i][index])
                    index += 1
            menubar.add_cascade(label=menus[i], menu=menu)

        parent.configuration.py(menu=menubar)

    @staticmethod
    def set_active(top):
        """
        Disable search windows so that only the most recent window can be interacted with
        """
        top.grab_set()

    def __init__(self):
        """
        All lists must be same length
        container is a tkinter frame
        text, labels, interface7 are each a list of strings
        commands is a list of functions to execute
        cursor is the cursor icon2 over this object
        row, column, columnspan, width are each a list of integers
        sticky is one or a combination of ('n', 's', 'w', 'e')
        tearoff is a boolean
        parent is a window object
        """
        # TODO
        self.number_of_items = 144

    @staticmethod
    def button(container, text, commands, cursor, coords, sticky=None):
        for i in (range(len(coords))):
            ttk.Button(container, text=text[i], command=commands[i], cursor=cursor).grid(
                row=coords[i][0],
                column=coords[i][1],
                sticky=sticky[
                    i] if sticky else None
            )

    def check_button(self, container, commands, cursor, coords, sticky=None):
        self.checkButtons = []
        for i in (range(len(coords))):
            checkButton = Checkbutton(container, command=commands[i], cursor=cursor)
            checkButton.grid(row=coords[i][0], column=coords[i][1], sticky=sticky[i] if sticky else None)
            self.checkButtons.append(checkButton)

    def radio_button(self, container, text, variables, values, commands, cursor, coords, sticky=None):
        self.radioButtons = []
        for i in (range(len(coords))):
            radioButton = Radiobutton(
                container, text=text[i], variable=variables[i], value=values[i],
                command=commands[i]
            )
            radioButton.grid(row=coords[i][0], column=coords[i][1], sticky=sticky[i] if sticky else None)
            self.radioButtons.append(radioButton)

    @staticmethod
    def label(container, text, coords, columnspan, sticky=None):
        for i, label in enumerate(range(len(coords))):
            ttk.Label(container, text=text[i]).grid(
                row=coords[i][0], column=coords[i][1], columnspan=columnspan[i],
                sticky=sticky[i] if sticky else None, pady=5
            )

    @staticmethod
    def raw_label(container, text, coords, sticky=None):
        ttk.Label(container, text=text).grid(row=coords[0], column=coords[1], sticky=sticky, pady=5)

    @staticmethod
    def raw_entry(container):
        return ttk.Entry(container)

    def entry(self, container, width, coords, columnspan, sticky=None):
        self.entries = []
        for i in (range(len(coords))):
            entry = ttk.Entry(container, width=width[i])
            entry.grid(
                row=coords[i][0], column=coords[i][1], columnspan=columnspan[i],
                sticky=sticky[i] if sticky else None
            )
            self.entries.append(entry)

    @staticmethod
    def separator(container, orient, coords, columnspan, sticky=None, pady=5):
        ttk.Separator(container, orient=orient).grid(
            row=coords[0], column=coords[1], columnspan=columnspan,
            sticky=sticky, pady=pady
        )

    @staticmethod
    def drop_down_menu(parent, menus, tearoff, labels, commands):
        try:
            menubar = Menu(parent.container)
        except:
            menubar = Menu(parent)
        for i, menu in enumerate(menus):
            menu = Menu(menubar, tearoff=tearoff[i])
            index = 0
            for label in (labels[i]):
                if label == 'separator()':
                    menu.add_separator()
                else:
                    menu.add_command(label=label, command=commands[i][index])
                    index += 1
            menubar.add_cascade(label=menus[i], menu=menu)

        parent.configuration.py(menu=menubar)

    @staticmethod
    def set_active(top):
        """
        Disable search windows so that only the most recent window can be interacted with
        """
        top.grab_set()

    @staticmethod
    def partial_drop(menubar, tearoff):
        return tk.Menu(menubar, tearoff=tearoff)

    @staticmethod
    def set_drop(container, menubar):
        tk.Tk.configuration.py(container, menu=menubar)

    @staticmethod
    def top(container, title):
        top = Toplevel(container, borderwidth=5)
        top.title(title)
        return top

    @staticmethod
    def save_file(fileExt, index=4, caller=None):
        path = tk.filedialog.asksaveasfilename(filetypes=(fileExt, configuration.ALL_FILES))
        if caller:
            caller.parent.util.entries[index].delete(0, END)
            caller.parent.util.entries[index].insert(0, path)
        else:
            return path

    @staticmethod
    def open_file(fileExt):
        path = tk.filedialog.askopenfilename(filetypes=(fileExt, configuration.ALL_FILES))
        return path

    @staticmethod
    def select_folder(caller, index=0):
        path = tk.filedialog.askdirectory(parent=caller.top, initialdir="/", title='Select Folder')
        caller.parent.util.entries[index].delete(0, END)
        caller.parent.util.entries[index].insert(0, path)

    @staticmethod
    def get_var(master, type, contents):
        if type == 'bool':
            return BooleanVar(master=master, value=contents)
        elif type == 'str':
            return StringVar(master=master, value=contents)

    @staticmethod
    def list_menu(container, variable, contents, command, coords, span, sticky=None):
        listMenu = ttk.OptionMenu(container, variable, *contents, command=command)
        listMenu.grid(row=coords[0], column=coords[1], columnspan=span[1], sticky=sticky)


class Popup:
    def __init__(self, parent, title, message):
        self.parent = parent
        self.util = self.parent.util
        self.title = title
        self.message = message
        self.top = Toplevel(self.parent.container)
        self.top.minsize(250, 100)
        self.container = ttk.Frame(self.top)
        self.container.pack(side="top", fill="both", expand=True)
        self.top.wm_title(self.title)
        self.top.grab_set()

    def okay(self):
        self.top.destroy()


class PopupError(Popup):
    def populate(self):
        # label
        ttk.Label(self.container, text=self.message).grid(row=0, column=1, columnspan=3, sticky='we')
        # Separators
        ttk.Separator(self.container, orient='h').grid(row=2, column=1, columnspan=4, sticky='we')
        # Buttons
        ttk.Button(self.container, text="Okay", command=self.okay, cursor='hand2').grid(row=3, column=1, sticky='we')


class PushButton(QtWidgets.QPushButton):
    def __init__(self, app, parent, text='', _icons=None, tooltip='', command=None):
        super(PushButton, self).__init__(QtGui.QIcon(), text, parent)
        self.app = app
        self.parent = parent
        self.text = text
        self.setToolTip(tooltip)
        if _icons == None:
            self.icons = {t: None for t in ['default', 'disabled', 'glowing', 'valid', 'connected']}
        elif type(_icons) == str:
            self.icons = {t: _icons for t in ['default', 'disabled', 'glowing', 'valid', 'connected']}
        else:
            self.icons = _icons
        self.setIcon(QtGui.QIcon(self.icons['default']))
        self.setIconSize(QtCore.QSize(20, 20))
        self.reset()
        self.disabled = False
        if command:
            self.clicked.connect(command)
        self.setMinimumSize(self.app.palette.minHeight, self.app.palette.minHeight)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.adjustSize()

    def reset(self, *args, **kwargs):
        self.disabled = False
        self.setIcon(QtGui.QIcon(self.icons['default']))
        self.setStyleSheet(qss_format(qss.btnstyles.button, self.app.palette))
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def set_disabled(self, *args, **kwargs):
        self.disabled = True
        self.setIcon(QtGui.QIcon(self.icons['disabled']))
        self.setStyleSheet(qss_format(qss.btnstyles.buttonDisabled, self.app.palette))
        self.setCursor(QtGui.QCursor(QtCore.Qt.ForbiddenCursor))

    def set_valid(self, *args, **kwargs):
        self.disabled = False
        self.setIcon(QtGui.QIcon(self.icons['valid']))
        self.setStyleSheet(qss_format(qss.btnstyles.buttonValid, self.app.palette))
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def set_glowing(self, *args, **kwargs):
        self.disabled = False
        self.setIcon(QtGui.QIcon(self.icons['glowing']))
        self.setStyleSheet(qss_format(qss.btnstyles.buttonGlow, self.app.palette))
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def set_connected(self, *args, **kwargs):
        self.disabled = False
        self.setIcon(QtGui.QIcon(self.icons['connected']))
        self.setStyleSheet(qss_format(qss.btnstyles.buttonConnected, self.app.palette))
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def __init__(self, app, parent, text='', _icons=None):
        super(PushButton, self).__init__(QtGui.QIcon(), text, parent)
        self.app = app
        self.parent = parent
        self.text = text
        if _icons == None:
            self.icons = {t: None for t in ['default', 'disabled', 'glowing', 'valid', 'connected']}
        elif type(_icons) == str:
            self.icons = {t: _icons for t in ['default', 'disabled', 'glowing', 'valid', 'connected']}
        else:
            self.icons = _icons
        self.setIcon(QtGui.QIcon(self.icons['default']))
        self.setIconSize(QtCore.QSize(20, 20))
        self.disabled = False

    def reset(self):
        self.disabled = False
        self.setIcon(QtGui.QIcon(self.icons['default']))
        self.setStyleSheet(qss_format(qss.btnstyles.button, self.app.palette))

    def set_disabled(self):
        self.disabled = True
        self.setIcon(QtGui.QIcon(self.icons['disabled']))
        self.setStyleSheet(qss_format(qss.btnstyles.buttonDisabled, self.app.palette))

    def set_valid(self):
        self.disabled = False
        self.setIcon(QtGui.QIcon(self.icons['valid']))
        self.setStyleSheet(qss_format(qss.btnstyles.buttonValid, self.app.palette))

    def set_glowing(self):
        self.disabled = False
        self.setIcon(QtGui.QIcon(self.icons['glowing']))
        self.setStyleSheet(qss_format(qss.btnstyles.buttonGlow, self.app.palette))

    def set_connected(self):
        self.disabled = False
        self.setIcon(QtGui.QIcon(self.icons['connected']))
        self.setStyleSheet(qss_format(qss.btnstyles.buttonConnected, self.app.palette))


class SettingsForm(QtWidgets.QWidget):
    def __init__(self, app, parent, height):
        super(SettingsForm, self).__init__()
        self.app = app
        self.parent = parent
        self.height = height
        self.frame = self.app.assembler.make_frame(self.parent)
        self.layout = self.app.assembler.make_grid(self.frame)
        self.frame.setFixedHeight(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.hidden = True

    def toggle(self, *args, **kwargs):
        parentHeight = self.parent.frameGeometry().height()
        parentWidth = self.parent.frameGeometry().width()
        if self.hidden:
            self.parent.resize(parentWidth, parentHeight + self.height)
            self.frame.setFixedHeight(self.height)
        else:
            self.parent.resize(parentWidth, parentHeight - self.height)
            self.frame.setFixedHeight(0)
        self.hidden = not self.hidden


class SubMenu(QtGui.QWidget):
    def __init__(self, parent, title, icon):
        parent = parent
        super(SubMenu, self).__init__()
        self.setStyleSheet(qss.WINDOW_STYLE)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle(title)
        icon = QtGui.QIcon(icon)
        self.setWindowIcon(icon)
        self.resize(350, 400)

        hbox = QtGui.QHBoxLayout(self)

        topleft = QtGui.QFrame(self)
        topleft.setStyleSheet(qss.FRAME_STYLE)
        topleft.setFrameShape(QtGui.QFrame.StyledPanel)
        topleft.setMinimumWidth(150)
        topleft.setMinimumHeight(150)
        topLeftLayout = QtGui.QGridLayout(topleft)

        topright = QtGui.QFrame(self)
        topright.setStyleSheet(qss.FRAME_STYLE)
        topright.setFrameShape(QtGui.QFrame.StyledPanel)
        topright.setMinimumWidth(150)
        topright.setMinimumHeight(150)
        topRightLayout = QtGui.QGridLayout(topright)

        bottom = QtGui.QFrame(self)
        bottom.setFrameShape(QtGui.QFrame.StyledPanel)
        bottom.setStyleSheet(qss.FRAME_STYLE)
        bottom.setMinimumWidth(300)
        bottom.setMinimumHeight(40)

        splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)
        splitter1.addWidget(topleft)
        splitter1.addWidget(topright)
        splitter1.setStyleSheet(qss.SPLITTER_STYLE)

        splitter2 = QtGui.QSplitter(QtCore.Qt.Vertical)
        splitter2.addWidget(splitter1)
        splitter2.addWidget(bottom)
        splitter2.setStyleSheet(qss.SPLITTER_STYLE)

        hbox.addWidget(splitter2)
        self.setLayout(hbox)

        # UNITS SELECTION
        self.lbl = QtGui.QLabel('Units:', self)
        self.lbl.adjustSize()
        combo = QtGui.QComboBox(self)
        combo.setStyleSheet(qss.COMBO_STYLE)
        for unit in UNITS:
            combo.addItem(unit)
        self.units = UNITS[0]
        topRightLayout.addWidget(self.lbl, 0, 0)
        topRightLayout.addWidget(combo, 0, 1)
        combo.activated[str].connect(self.change_units)

        # RADIUS, LATITUDE, LONGITUDE ENTRY
        self.radiusLabel = QtGui.QLabel('Radius:', self)
        self.rad = QtGui.QLineEdit(self)
        topRightLayout.addWidget(self.radiusLabel, 1, 0)
        topRightLayout.addWidget(self.rad, 1, 1)

        self.lonLabel = QtGui.QLabel('Longitude:', self)
        self.lon = QtGui.QLineEdit(self)

        topRightLayout.addWidget(self.lonLabel, 2, 0)
        topRightLayout.addWidget(self.lon, 2, 1)

        self.latLabel = QtGui.QLabel('Latitude:', self)
        self.lat = QtGui.QLineEdit(self)

        topRightLayout.addWidget(self.latLabel, 3, 0)
        topRightLayout.addWidget(self.lat, 3, 1)

    def change_units(self, units):
        self.units = units
        print(self.rad.text())

    def onChanged(self, text):
        self.radiusLabel.setText(text)
        self.radiusLabel.adjustSize()


class SubMenu(QtWidgets.QWidget):
    def __init__(self, parent, title, icon, opensave=True, standard=True, tabbed=False):
        self.parent = parent
        self.results = None
        self.resultsTitle = ''
        super(SubMenu, self).__init__()
        self.setStyleSheet(qss.WINDOW_STYLE + qss.WIDGET_STYLE)
        # self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle(title)
        self.icon = QtGui.QIcon(icon)
        self.setWindowIcon(self.icon)
        self.hbox = QtWidgets.QHBoxLayout(self)

        if opensave:
            # self.topleft = make_frame(parent)
            # self.topLeftLayout = make_grid(self.topleft)
            self.topright = make_frame(parent)
            self.topRightLayout = make_grid(self.topright)
            # self.splitter1 = make_splitter(style='h', widgets=(self.topleft, self.topright))
            self.bottom = make_frame(parent)
            self.bottomLayout = make_grid(self.bottom)
            self.splitter2 = make_splitter(style='v', widgets=(self.topright, self.bottom))
            if standard:
                self.hbox.addWidget(self.splitter2)

        elif tabbed:
            self.tabs = QtWidgets.QTabWidget()
            self.tabs.setStyleSheet(qss.TAB_STYLE)
            self.hbox.addWidget(self.tabs)
        self.setLayout(self.hbox)

        # menu_bar(self)

    def export_results(self):
        try:
            formats = [x.text() for x in self.exportFormats.selectedItems()]
            if len(formats) == 0:
                QtWidgets.QMessageBox.critical(self, "No Selection", "Select a format.", QtWidgets.QMessageBox.Ok)
                return
            if type(self.results) != pd.DataFrame and type(self.results) != list and type(self.results) != dict:
                QtWidgets.QMessageBox.critical(
                    self, "Nothing To Export", "Run an analysis before expQtWidgets.orting.", QtWidgets.QMessageBox.Ok
                )
                return
            if type(self.results) == list:
                for i, result in enumerate(self.results):
                    dh.export_data(
                        dataFrame=result, savePath=os.path.join(self.saveLabel.text(), self.resultsTitle),
                        formats=formats, suffix=''
                    )
            elif type(self.results) == dict:
                for i, result in enumerate(self.results):
                    dh.export_data(
                        dataFrame=self.results[result],
                        savePath=os.path.join(self.saveLabel.text(), result + self.resultsTitle),
                        formats=formats, suffix=''
                    )
            else:
                dh.export_data(
                    dataFrame=self.results, savePath=os.path.join(self.saveLabel.text(), self.resultsTitle),
                    formats=formats, suffix=''
                )
        except Exception as e:
            print(e)


class VerticalScrolledFrame(Frame):
    # Use the 'interior' attribute to place widgets inside the scrollable frame
    # Construct and pack/place/grid normally
    # This frame only allows vertical scrolling
    def __init__(self, parent, *args, **kw):
        Frame.__init__(self, parent, *args, **kw)

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)  # EXPAND FALSE
        canvas = Canvas(self, bd=0, highlightthickness=0, yscrollcommand=vscrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.configuration.py(command=canvas.yview, cursor='hand2')

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Frame(canvas, bd=5)
        interior_id = canvas.create_window(0, 0, window=interior, anchor=NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.configuration.py(scrollregion="0 0 %s %s" % size)

            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.configuration.py(width=interior.winfo_reqwidth())

        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())

        canvas.bind('<Configure>', _configure_canvas)

    def __init__(self, parent, *args, **kw):
        Frame.__init__(self, parent, *args, **kw)
        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)  # EXPAND FALSE
        canvas = Canvas(self, bd=0, highlightthickness=0, yscrollcommand=vscrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.configuration.py(command=canvas.yview, cursor='hand2')
        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)
        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Frame(canvas, borderwidth=10)
        interior_id = canvas.create_window(0, 0, window=interior, anchor=NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.configuration.py(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.configuration.py(width=interior.winfo_reqwidth())

        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())

        canvas.bind('<Configure>', _configure_canvas)

    # Use the 'interior' attribute to place widgets inside the scrollable frame
    # Construct and pack/place/grid normally
    # This frame only allows vertical scrolling
    def __init__(self, parent, *args, **kw):
        Frame.__init__(self, parent, *args, **kw)

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)  # EXPAND FALSE
        canvas = Canvas(self, bd=0, highlightthickness=0, yscrollcommand=vscrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.configuration.py(command=canvas.yview, cursor='hand2')

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior, anchor=NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.configuration.py(scrollregion="0 0 %s %s" % size)

            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.configuration.py(width=interior.winfo_reqwidth())

        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())

        canvas.bind('<Configure>', _configure_canvas)


class Window:
    def __init__(self, title):
        self.app = QtGui.QApplication(sys.argv)
        self.title = title
        self.util = MenuBuilder()

    def mainloop(self, args):
        Slave(args[0], args[1])


class Window(tk.Tk):
    def __init__(self, windowName):
        """Creates the window where the application will be displayed."""
        tk.Tk.__init__(self)
        # tk.Tk.iconbitmap(self, default=None)
        self.util = c_lo_globalUtilities.c_lo_globalUtilities()
        tk.Tk.wm_title(self, window_name)
        # tk.Tk.minsize(self, width=int(self.util.WIDTH / 1), height=int(2 * self.util.HEIGHT / 3))
        # geometry("%dx%d" % (WIDTH / 1, 2 * HEIGHT / 3))
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        """self.frame = View(self.container, self, window_name)
        self.frame.grid(row=0, column=0, sticky="nsew")

    def show_frame(self):
        self.frame.tkraise()

object_oriented View(tk.Frame):
    def __init__(self, parent, controller, window_name):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=window_name, font=controller.util.LARGE_FONT)
        label.grid(pady=10, padx=10)"""

    def __init__(self, windowName):
        """Creates the window where the application will be displayed."""
        tk.Tk.__init__(self)
        # tk.Tk.iconbitmap(self, default=None)
        self.util = MenuBuilder()
        tk.Tk.wm_title(self, windowName)
        # tk.Tk.minsize(self, width=int(self.util.WIDTH / 1), height=int(2 * self.util.HEIGHT / 3))
        # self.geometry("%dx%d" % (WIDTH / 1, HEIGHT / 1))
        self.container = ttk.Frame(self, borderwidth=5, relief='groove')
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        s = ttk.Style()
        # s.theme_use('winnative')
        s.configure('.', font=('Times New Roman', 10))
        self.state('zoomed')
        # print(s.theme_names())

        """self.frame = View(self.container, self, window_name)
        self.frame.grid(row=0, column=0, sticky="nsew")

    def show_frame(self):
        self.frame.tkraise()

object_oriented View(tk.Frame):
    def __init__(self, parent, controller, window_name):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=window_name, font=controller.util.LARGE_FONT)
        label.grid(pady=10, padx=10)"""


############################################################
class TSAutoScrollbar(QtWidgets.QScrollBar):
    # a scrollbar that hides itself if it's not needed.  only
    # works if you use the grid geometry manager.
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            # grid_remove is currently missing from Tkinter!
            self.tk.call("grid", "remove", self)
        else:
            self.grid()
        Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise TclError("cannot use pack with this widget")

    def place(self, **kw):
        raise TclError("cannot use place with this widget")


class TSInputCheckBox(QtWidgets.QCheckBox):
    def __init__(self, app, parent, checked=False, tooltip='', command=None):
        super(TSInputCheckBox, self).__init__(parent)
        self.app = app
        self.parent = parent
        self.setToolTip(tooltip)
        if checked:
            self.set_checked()
        else:
            self.set_unchecked()
        if command:
            self.clicked.connect(command)
        self.setMinimumSize(self.app.palette.minHeight, self.app.palette.minHeight)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def set_unchecked(self, *args, **kwargs):
        self.setChecked(False)
        self.setStyleSheet(qss_format(qss.btnstyles.checkboxUnchecked, self.app.palette))

    def set_checked(self, *args, **kwargs):
        self.setChecked(True)
        self.setStyleSheet(qss_format(qss.btnstyles.checkboxChecked, self.app.palette))


class TSInputBox(QtWidgets.QWidget):
    def set_stylesheet(self, style):
        self.setStyleSheet(qss_format(vars(qss.entry)['{}{}'.format(self.inputType, style)], self.app.palette))

    def reset(self):
        self.disabled = False
        self.update_able()
        self.set_stylesheet('')

    def set_disabled(self):
        self.disabled = True
        self.update_able()
        self.set_stylesheet('Disabled')

    def set_error(self):
        self.disabled = False
        self.update_able()
        self.set_stylesheet('Error')

    def set_valid(self):
        self.disabled = False
        self.update_able()
        self.set_stylesheet('Valid')

    def set_glowing(self):
        self.disabled = False
        self.update_able()
        self.set_stylesheet('Glow')

    def set_connected(self):
        self.disabled = True
        self.update_able()
        self.set_stylesheet('Connected')

    def __init__(self, app, parent, inputType, readOnly=False, command=None, tooltip=''):
        super(TSInputBox, self).__init__(parent)
        self.app = app
        self.parent = parent
        self.disabled = False
        self.inputType = inputType
        self.reset()
        self.setToolTip(tooltip)
        self.setMinimumHeight(self.app.palette.minHeight)
        self.set_command(command)
        self.setReadOnly(readOnly)  # self.setvariables.FONT(QtGui.Qvariables.FONT('Arial'))

    def set_command(self, command):
        if command:
            if self.inputType == 'spin':
                self.valueChanged.connect(command)
            elif self.inputType == 'entry':
                self.textChanged.connect(command)

    def update_able(self, *args, **kwargs):
        if self.disabled:
            self.setReadOnly(True)
            self.setDisabled(True)
            if self.inputType == 'spin':
                self.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        else:
            self.setReadOnly(False)
            self.setDisabled(False)
            if self.inputType == 'spin':
                self.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

    def set_stylesheet(self, style):
        self.setStyleSheet(
            qss_format(vars(qss.entrystyles)['{}{}'.format(self.inputType, style)], self.app.palette)
        )


class TSInputFloat(QtWidgets.QDoubleSpinBox, TSInputBox):
    def __init__(self, app, parent, inputSettings, decimals=0):
        TSInputBox.__init__(self, app, parent, 'spin')
        self.setRange(inputSettings[0], inputSettings[1])
        self.setValue(inputSettings[2])
        self.setSingleStep(inputSettings[3])
        self.setDecimals(decimals)

    def __init__(self, app, parent, settings, readOnly=False, decimals=0, command=None, tooltip=''):
        TSInputBox.__init__(self, app, parent, 'spin', readOnly, command)
        self.blockSignals(True)
        self.setRange(settings[0], settings[1])
        self.setValue(settings[2])
        self.setSingleStep(settings[3])
        self.setDecimals(decimals)
        self.blockSignals(False)


class TSInputInteger(QtWidgets.QSpinBox, TSInputBox):
    def __init__(self, app, parent, inputSettings):
        TSInputBox.__init__(self, app, parent, 'spin')
        self.setRange(inputSettings[0], inputSettings[1])
        self.setValue(inputSettings[2])
        self.setSingleStep(inputSettings[3])


class TSInputRadioButton(QtWidgets.QRadioButton):
    def __init__(self, app, parent):
        super(TSInputRadioButton, self).__init__(parent)
        self.app = app
        self.parent = parent
        self.setStyleSheet(qss_format(qss.radio, self.app.palette))


class TSInputTextArea(QtWidgets.QPlainTextEdit, TSInputBox):
    def __init__(self, app, parent):
        TSInputBox.__init__(self, app, parent, 'entry')

    def set_text(self, text):
        self.setPlainText(text)


class TSInputTextLine(QtWidgets.QLineEdit, TSInputBox):
    def __init__(self, app, parent, readOnly=False, command=None, text='', tooltip=''):
        TSInputBox.__init__(self, app, parent, 'entry', readOnly, command)
        self.blockSignals(True)
        self.set_text(text)
        self.blockSignals(False)

    def set_text(self, text):
        self.setText(text)

    def __init__(self, app, parent):
        TSInputBox.__init__(self, app, parent, 'entry')


class TSLabel(QtWidgets.QLabel):
    def __init__(self, app, parent, text, tooltip=''):
        super(TSLabel, self).__init__(text, parent)
        self.app = app
        self.parent = parent
        self.setWordWrap(True)
        self.setToolTip(tooltip)
        # self.setvariables.FONT(QtGui.Qvariables.FONT('Arial'))
        self.setStyleSheet(qss_format(qss.generalstyles.label, self.app.palette))
        self.adjustSize()


class TSListSelect(QtWidgets.QListWidget):
    def __init__(self, app, parent, items=None, default=0, multiSelect=False, command=None, tooltip=''):
        super(TSListSelect, self).__init__(parent)
        self.app = app
        if items:
            self.replace_items(items, default)
        if multiSelect:
            self.setSelectionmode(QtWidgets.QAbstractItemView.MultiSelection)
        if command:
            self.currentItemChanged.connect(command)
        self.setStyleSheet(qss_format(qss.selectionstyles.list, self.app.palette))
        self.setToolTip(tooltip)

    def replace_items(self, items, default):
        self.blockSignals(True)
        self.clear()
        self.insertItems(0, items)
        self.setCurrentRow(default)
        self.blockSignals(False)


class TSLogTable(QtWidgets.QTableWidget):
    def scroll_to_bottom(self, c, item):
        self.scrollToBottom()
        self.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtTop)
        self.selectRow(c + 1)

    def set_disabled(self):
        self.setStyleSheet(qss_format(qss.selectionstyles.tableDisabled, self.app.palette))

    def set_valid(self):
        self.setStyleSheet(qss_format(qss.selectionstyles.tableValid, self.app.palette))

    def set_connected(self):
        self.setStyleSheet(qss_format(qss.tableConnected, self.app.palette))

    def __init__(self, app, parent, tooltip=''):
        super(TSLogTable, self).__init__(parent)
        self.app = app
        self.parent = parent
        self.setStyleSheet(qss_format(qss.selectionstyles.table, self.app.palette))
        self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.setToolTip(tooltip)

    def update_contents(self, *args, **kwargs):
        try:
            txt = self.app.logger.activeLogContent
            self.setColumnCount(len(txt[0]))
            self.setRowCount(len(txt))
            if txt != ['']:
                self.clear()
                for k, t in enumerate(txt):
                    s = t.split(self.app.dataConfig.delimiter)[2:]  # slice to remove datetime stamp
                    for u, _ in enumerate(self.app.logger._format(s)):
                        self.setItem(k, u, get_item(_))
        except Exception as e:
            print('Update log contents error: {}.'.format(e))

    def update_row_labels(self, *args, **kwargs):
        stamps = []
        txt = self.app.logger.activeLogContent
        self.setColumnCount(len(txt[0]))
        self.setRowCount(len(txt))
        if txt != ['']:
            for k, t in enumerate(txt):
                s = t.split(self.app.dataConfig.delimiter)
                stamps.append('{}'.format(s[1]))
        self.setVerticalHeaderLabels(stamps)

    def auto_update_refresh(self, data):
        """
        This function is called by the Transceiver listener thread when new data
        is received if automatic update is enabled in the Log Menu
        (self.app.logConfig._autoUpdate = True). If automatic update is disabled in the
        Log Menu (self.autoUpdate = False), this function will not
        be called.
        :param data: (type str) The data that will be added to the menus0 reader.
        :return: void
        """
        try:
            if self.app.logMenu.logFileSelector.currentRow() == 0:
                c = self.rowCount()
                self.setRowCount(c + 1)
                for n, d in enumerate(self.app.logger._format(data.split(self.app.dataConfig.delimiter))):
                    item = get_item(d)
                    self.setItem(c, n, item)
                    if self.app.logConfig._autoUpdate:
                        self.scroll_to_bottom(c + 1, item)
                self.setVerticalHeaderItem(c, get_item(self.app.dataDude.timestampHMSms))
                self.update_col_labels()
        except Exception as e:
            statusMsg = 'Log reader update error: ' + str(e) + '\nData: ' + data
            self.app.logMenu.update_status(statusMsg, 'error')

    def update_col_labels(self, *args, **kwargs):
        head = self.app.dataConfig.head.split(self.app.dataConfig.delimiter)
        self.setHorizontalHeaderLabels(head)
        self.setColumnCount(len(head))
        self.resizeColumnsToContents()

    def set_disabled(self, *args, **kwargs):
        self.setStyleSheet(qss_format(qss.selectionstyles.tableDisabled, self.app.palette))

    def set_valid(self, *args, **kwargs):
        self.setStyleSheet(qss_format(qss.selectionstyles.tableValid, self.app.palette))

    def set_connected(self, *args, **kwargs):
        self.setStyleSheet(qss_format(qss.selectionstyles.tableConnected, self.app.palette))


class TSMenuBar(QtWidgets.QMenuBar):
    def __init__(self, app):
        super(TSMenuBar, self).__init__()
        self.setFixedHeight(22)
        self.setStyleSheet(qss_format(qss.barstyles.menubar, app.palette))

    def add_menu(self, title, iconPath=''):
        return self.addMenu(QtGui.QIcon(iconPath), title)

    def add_entry(self, title, callback, menu, iconPath=''):
        action = QtWidgets.QAction(QtGui.QIcon(iconPath), title, self)
        action.triggered.connect(callback)
        menu.addAction(action)


class TSProgressBar(QtWidgets.QProgressBar):
    def __init__(self, app, parent):
        super(TSProgressBar, self).__init__(parent)
        self.app = app
        self.parent = parent
        self.setGeometry(30, 40, 200, 25)
        self.show()


class TSStatusBar(QtWidgets.QStatusBar):
    def __init__(self, app, parent):
        super(TSStatusBar, self).__init__(parent)
        self.app = app
        self.parent = parent
        self.reset()
        self.setSizeGripEnabled(False)

    def write(self, msg, s):
        self.showMessage(msg)
        self.adjustSize()
        self.set_stylesheet(s.lower().capitalize())

    def set_stylesheet(self, style=''):
        styleSheet = getattr(qss.statusbarstyles, 'statusbar{}'.format(style))
        self.setStyleSheet(qss_format(styleSheet, self.app.palette))

    def reset(self, *args, **kwargs):
        self.set_stylesheet()

    def set_success(self, *args, **kwargs):
        self.set_stylesheet('Success')

    def set_alert(self, *args, **kwargs):
        self.set_stylesheet('Alert')

    def set_error(self, *args, **kwargs):
        self.set_stylesheet('Error')


class TSTitleBar(QtWidgets.QWidget):
    def __init__(self, parent, title):
        super(TSTitleBar, self).__init__(parent)
        self.parent = parent
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.title = QtWidgets.QLabel(title)
        self.minBtn = QtWidgets.QPushButton("—")
        self.closeBtn = QtWidgets.QPushButton("X")
        self.minBtn.clicked.connect(self.minimize_window)
        self.closeBtn.clicked.connect(self.close_window)
        self.minBtn.setFixedSize(self.parent.app.palette.titlebarBtnSize, self.parent.app.palette.titlebarBtnSize)
        self.closeBtn.setFixedSize(self.parent.app.palette.titlebarBtnSize, self.parent.app.palette.titlebarBtnSize)
        self.minBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.closeBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.minBtn.setStyleSheet(qss.titlebarstyles.minButton)
        self.closeBtn.setStyleSheet(qss.titlebarstyles.xButton)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.minBtn)
        self.layout.addWidget(self.closeBtn)
        self.setContentsMargins(0, 0, 1, 0)
        self.setFixedHeight(self.parent.app.palette.titlebarBtnSize)
        self.setLayout(self.layout)
        self.start = QtCore.QPoint(0, 0)
        self.pressing = False
        self.setStyleSheet(qss_format(qss.titlebarstyles.titleBar, self.parent.app.palette))
        self.title.setStyleSheet(
            qss_format(qss.titlebarstyles.titleBarLabel, self.parent.app.palette)
        )

    def resizeEvent(self, QResizeEvent):
        super(TSTitleBar, self).resizeEvent(QResizeEvent)
        self.title.setFixedWidth(self.parent.width())

    def mousePressEvent(self, event):
        self.start = self.mapToGlobal(event.pos())
        self.pressing = True

    def mouseMoveEvent(self, event):
        if self.pressing:
            self.end = self.mapToGlobal(event.pos())
            self.movement = self.end - self.start
            self.parent.setGeometry(
                self.mapToGlobal(self.movement).x(), self.mapToGlobal(self.movement).y(),
                self.parent.width(), self.parent.height()
            )
            self.start = self.end

    def mouseReleaseEvent(self, QMouseEvent):
        self.pressing = False

    def close_window(self, *args, **kwargs):
        self.parent.close()

    def minimize_window(self, *args, **kwargs):
        self.parent.showMinimized()

    def __init__(self, parent, title):
        super(TSTitleBar, self).__init__()
        btn_size = 15
        self.parent = parent
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.title = QtWidgets.QLabel(title)
        self.btn_close = QtWidgets.QPushButton("X")
        self.btn_close.clicked.connect(self.btn_close_clicked)
        self.btn_close.setFixedSize(btn_size, btn_size)
        self.btn_close.setStyleSheet("background-color: rgb(90,80,80);")
        self.btn_close.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_close.setStyleSheet(qss.xButton)
        self.btn_min = QtWidgets.QPushButton("—")
        self.btn_min.clicked.connect(self.btn_min_clicked)
        self.btn_min.setFixedSize(btn_size, btn_size)
        self.btn_min.setStyleSheet("background-color: rgb(80,80,90);")
        self.btn_min.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_min.setStyleSheet(qss.minButton)
        self.title.setFixedHeight(btn_size)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.btn_min)
        self.layout.addWidget(self.btn_close)
        self.setContentsMargins(0, 0, 1, 0)
        self.setFixedHeight(15)
        self.title.setStyleSheet(
            """
                        background-color: rgb(50,50,50);
                        border-radius: 0;
                        color: white;
                    """
        )
        self.setLayout(self.layout)
        self.start = QtCore.QPoint(0, 0)
        self.pressing = False
        self.title.setStyleSheet(qss.titleBar)

    def btn_close_clicked(self):
        self.parent.close()

    def btn_min_clicked(self):
        self.parent.showMinimized()
