from Utilities.Constants import *


def make_grid(parent):
    grid = QtWidgets.QGridLayout(parent)
    return grid


def make_splitter(style, widgets):
    if style == 'h' or style == 'horizontal':
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    else:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

    for widget in widgets:
        splitter.addWidget(widget)

    splitter.setStyleSheet(FRAME_STYLE + SPLITTER_STYLE)
    QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    return splitter


def make_frame(parent):
    frame = QtWidgets.QFrame(parent)
    frame.setStyleSheet(FRAME_STYLE)
    frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
    return frame


def make_scroll_area(widget):
    scrollArea = QtWidgets.QScrollArea()
    scrollArea.setFrameShape(QtWidgets.QScrollArea.StyledPanel)
    scrollArea.setStyleSheet(FRAME_STYLE + SCROLL_STYLE)
    scrollArea.setWidget(widget)
    return scrollArea


def make_button(parent, type=None, command=None, text='', icon=None):
    if type == 'radio':
        button = QtWidgets.QRadioButton(parent)

    elif type == 'check':
        button = QtWidgets.QCheckBox(parent)

    else:
        if icon:
            icn = QtGui.QIcon(icon)
            button = QtWidgets.QPushButton(icn, text, parent)
            button.setIconSize(QtCore.QSize(20, 20))
        else:
            button = QtWidgets.QPushButton(text, parent)

        button.setFont(QtGui.QFont('Verdana', 9))
        button.setStyleSheet(BUTTON_STYLE)
        set_widget_size(button, 25, 25)
        button.clicked.connect(command)

    return button


def make_label(parent, text=''):
    label = QtWidgets.QLabel(text, parent)
    label.setFont(QtGui.QFont("Verdana", 9))
    label.setStyleSheet(LABEL_STYLE)
    return label


def make_entry(parent, width=300):
    entry = QtWidgets.QLineEdit(parent)
    entry.setFont(QtGui.QFont("Verdana", 9))
    entry.setStyleSheet(ENTRY_STYLE)
    set_widget_size(entry, width, 20)
    return entry


def make_combo(parent, items, command):
    combo = QtWidgets.QComboBox(parent)
    set_widget_size(combo, 40, 20)
    combo.setStyleSheet(COMBO_STYLE)
    for item in items:
        combo.addItem(item)
    combo.activated[str].connect(command)
    return combo


def top_left_splitter(widget, open=None, openIcon=OPEN_FOLDER_ICON, save=None, saveIcon=SAVE_FILE_ICON, fileExt=''):
    """
    :param widget: parent widget
    :param open: method for opening file/folder
    :param fileExt: valid extensions for saving a file
    :return: void
    """
    if open:
        # FILES TO READ
        widget.readLabel = make_label(widget)
        widget.topLeftLayout.addWidget(widget.readLabel, 0, 1)

        widget.readButton = make_button(widget, command=lambda: open(widget, label=widget.readLabel),
                                        icon=openIcon)
        widget.topLeftLayout.addWidget(widget.readButton, 0, 0)

    if save:
        # FILE TO SAVE
        widget.saveLabel = make_label(widget)
        widget.topLeftLayout.addWidget(widget.saveLabel, 1, 1)


        if (save.__name__ == 'save_file'):
            widget.saveButton = make_button(widget,
                                            command=lambda: save(widget, fileExt=fileExt, label=widget.saveLabel),
                                            icon=saveIcon)
        else:
            widget.saveButton = make_button(widget,
                                            command=lambda: save(widget, label=widget.saveLabel),
                                            icon=saveIcon)

        widget.topLeftLayout.addWidget(widget.saveButton, 1, 0)


def open_files(parent, multiple=False, fileExt='', label=None):
    if not multiple:
        paths = QtWidgets.QFileDialog.getOpenFileName(parent, 'Select file', os.getcwd(), fileExt + ";;" + ALL_FILES)
    else:
        paths = QtWidgets.QFileDialog.getOpenFileNames(parent, 'Select files', os.getcwd(), fileExt)
    if label and paths:
        label.setText(str(paths))
        label.adjustSize()
    if paths:
        return paths


def save_file(parent, fileExt='', label=None):
    paths = QtWidgets.QFileDialog.getSaveFileName(parent, 'Select file', os.getcwd(), fileExt + ALL_FILES)
    if label and paths:
        label.setText(str(paths[0]))
        label.adjustSize()
    if paths:
        return paths[0]


def open_folder(parent, label=None, default=os.getcwd()):
    dir = QtWidgets.QFileDialog.getExistingDirectory(parent, 'Select folder', default)
    if dir:
        label.setText(str(dir))
        label.adjustSize()
        return dir


def set_widget_size(widget, width, height):
    widget.setMinimumSize(width, height)
    widget.setMaximumSize(width, height)


def clear(entries):
    for i in range(entries.__len__()):
        entries[FEATURES[i + 0]].setText('')