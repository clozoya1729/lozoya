import configuration
import lozoya.file
import lozoya.gui


def update_path():
    global selectedFiles
    selectionToolbar.setEnabled(True)
    path = str(fileDialog.getExistingDirectory(app.window))
    statusWidget.update(84 * '-' + '\nSet working directory to: {}'.format(path))
    buttonRootDirectory.setText(path)
    selectedFiles = lozoya.file.get_files(path)
    update_list_widget(selectedFiles)
    actionSelectAll.setChecked(True)
    folderCount = lozoya.file.folder_count(path)
    fileCount = lozoya.file.file_count(path)
    size = lozoya.file.file_size_sum(selectedFiles)
    readableSize = lozoya.file.readable_size(size)
    statusWidget.update('  • {} folders\n  • {} files\n  • {}'.format(folderCount, fileCount, readableSize))


def update_list_widget(files=tuple()):
    listWidget.clear()
    if files:
        listWidget.addItems(files)
        actionToolbar.setEnabled(True)
    else:
        listWidget.addItems(['No results'])
        actionToolbar.setEnabled(False)


def update_list_widget_decorator():
    def inner(func):
        def wrapper(*args, **kwargs):
            global selectedFiles
            statusWidget.update("  → {}".format(func.__name__.replace('_', ' ').title()))
            path = buttonRootDirectory.text()
            selectedFiles = func()(path)
            update_list_widget(selectedFiles)
            statusWidget.update("    • {} results".format(len(selectedFiles)))

        return wrapper

    return inner


@update_list_widget_decorator()
def select_all():
    return lozoya.file.get_subpaths


@update_list_widget_decorator()
def select_files():
    return lozoya.file.get_files


@update_list_widget_decorator()
def select_folders():
    return lozoya.file.get_folders


@update_list_widget_decorator()
def select_empty_files():
    return lozoya.file.get_empty_files


@update_list_widget_decorator()
def select_empty_folders():
    return lozoya.file.get_empty_files


@update_list_widget_decorator()
def select_duplicate_files():
    return lozoya.file.get_empty_files


def delete_selected():
    def accept():
        global selectedFiles
        numberOfFiles = len(selectedFiles)
        size = lozoya.file.file_size_sum(selectedFiles)
        readableSize = lozoya.file.readable_size(size)
        lozoya.file.remove_paths(selectedFiles)
        statusWidget.update('Deleted {} files ({}).'.format(numberOfFiles, readableSize))
        selectedFiles = []
        update_list_widget(selectedFiles)
        actionDeleteSelected.setChecked(False)

    def reject():
        actionDeleteSelected.setChecked(False)

    message = 'Delete {} files?'.format(len(selectedFiles))
    dialog = lozoya.gui.TSDialog('Delete Selected Files', message, accept, reject)
    dialog.exec()


selectedFiles = []
windowWidth = 500
windowHeight = 500
logHeight = 100
app = lozoya.gui.TSApp(
    name=configuration.name,
    root=configuration.root,
    size=(windowWidth, windowHeight)
)
fileDialog = lozoya.gui.TSFileDialog(name='Path selection')
listWidget = lozoya.gui.TSListWidget(name='Results', items=['No results'])
buttonRootDirectory = lozoya.gui.TSInputButton(name='Working Folder', text='Browse', callback=update_path)
actionSelectAll = lozoya.gui.make_action('All', select_all)
actionSelectFiles = lozoya.gui.make_action('Files', select_files)
actionSelectFolders = lozoya.gui.make_action('Folders', select_folders)
actionSelectEmptyFiles = lozoya.gui.make_action('Empty Files', select_empty_files)
actionSelectEmptyFolders = lozoya.gui.make_action('Empty Folders', select_empty_folders)
actionSelectDuplicateFiles = lozoya.gui.make_action('Duplicate Files', select_duplicate_files)
selectionActions = [actionSelectAll, actionSelectFiles, actionSelectFolders, actionSelectEmptyFiles,
                    actionSelectEmptyFolders, actionSelectDuplicateFiles]
selectionToolbar = lozoya.gui.make_toolbar('Selection', selectionActions)
selectionToolbar.setEnabled(False)
actionDeleteSelected = lozoya.gui.make_action('Delete Selected', delete_selected)
actionActions = [actionDeleteSelected]
actionToolbar = lozoya.gui.make_toolbar('Actions', actionActions)
actionToolbar.setEnabled(False)
group = lozoya.gui.make_action_group(selectionToolbar, selectionActions)
statusWidget = lozoya.gui.TSLogArea('Log', fixedHeight=logHeight)
fields = [buttonRootDirectory, selectionToolbar, actionToolbar, listWidget, statusWidget]
form = lozoya.gui.TSForm(fields, name='configuration')
app.window.setCentralWidget(form)
app.exec()
