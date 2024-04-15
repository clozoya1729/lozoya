from Utilities.Dependencies import *
from Utilities.Variables import *

# SYSTEM
SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
HAND = 'hand2'
PTR = 'left_ptr'

# PATHS
ROOT = os.getcwd()
DATABASE = ROOT + '\Database\Years\\'
VALUES = ROOT + r'\Interface\InterfaceUtilities\Labels\parameterValues.txt'
STATES = ROOT + r'\Database\Information\allStates.txt'
STATE_CODES = ROOT + r'\Database\Information\allStateCodes.txt'
ITEM_NAMES = ROOT + r'\Interface\InterfaceUtilities\Labels\ItemNames.txt'
temp_RSP = ROOT + '\\Utilities\Preferences\Temporary_Report_Preferences.txt'
RSP = ROOT + '\\Utilities\Preferences\Report_Preferences.txt'
TEMP_SAVE = ROOT + "\Database\\temp.txt"
BASE_MAP = ROOT + r'\Graphs\Maps\baseMap.html'
BASE_GRAPH = ROOT + r'\Graphs\Tooltip.html'

def collect_headers(dir):
    """
    Create a file containing all unique
    column names in the entire database
    """
    headers = [FOLDER]
    for folder in os.listdir(dir):
        path = os.path.join(dir, folder)
        for filepath in os.listdir(path):
            try:
                with open(os.path.join(path, filepath), newline='') as file: header = next(csv.reader(file))
                for column in header:
                    if column not in headers:
                        headers.append(column)
            except:
                pass

    with open(ITEM_NAMES, "w") as file:
        for header in headers:
            file.write(header+'\n')

    return headers

HEADERS = collect_headers(DATABASE)
# ENUMERATE OF FEATURES IN DATABASE
FEATURES = {}
with open(ITEM_NAMES, "r") as f:
    for i, line in enumerate(f):
        FEATURES[i] = line.strip()

# List of items that can transition through states
tmp = []
with open(VALUES, "r") as parameterValuesFile:
    with open(ITEM_NAMES, "r") as parameterNamesFile:
        for name, value in zip(csv.reader(parameterNamesFile), (csv.reader(parameterValuesFile))):
            if name[0] != FOLDER and name[0] != FILE and "None" not in value:
                tmp.append(name[0])
LEARNABLE = [FEATURES[i] for i in FEATURES if (FEATURES[i]) in tmp]


# EXTENSIONS
CSV_EXT = '.csv'
KML_EXT = '.kml'
PDF_EXT = '.pdf'
QRY_EXT = '.qry'
TXT_EXT = '.txt'
XLSX_EXT = '.xlsx'


# EXTENSION TUPLES
ALL_FILES = 'All files (*.*)'
CSV_FILES = 'csv files (*.csv);;'
KML_FILES = 'kml files (*.kml);;'
PDF_FILES = 'pdf files (*.pdf);;'
QRY_FILES = 'Query files (*.qry);;'
TXT_FILES = 'text files (*.txt);;'
XLSX_FILES = 'xlsx files (*.xlsx);;'

# NUMERICAL
CURRENT_YEAR = datetime.datetime.now().year - 1
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
INDICATORS = ('--', '-')
NA_VALUES = ['N']
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
UNITS = ('km', 'm', 'mi', 'ft')





# DESCRIPTIONS
ICONS = ROOT +  r'\Interface\InterfaceUtilities\Icons'

WINDOW_ICON = ICONS + r'\window.png'


OPEN_FOLDER_DESCRIPTION = 'Open a folder from which to extract files.'
OPEN_FOLDER_ICON = ICONS + r'\openFolder.png'

SAVE_FOLDER_DESCRIPTION = 'Open a folder to save files in.'
SAVE_FOLDER_ICON = ICONS + r'\saveFolder.png'

OPEN_FILE_DESCRIPTION = 'Open file.'
OPEN_FILE_ICON = ICONS + r'\openFile.png'

SAVE_FILE_DESCRIPTION = 'Save file.'
SAVE_FILE_ICON = ICONS + r'\saveFile.png'

EXIT_DESCRIPTION = 'Exit application.'
EXIT_ICON = ICONS + r'\exit.png'


#Data encapsulates search, geosearch, machine learning, and sentiment analysis
DATABASE_DESCRIPTION = 'Database: Set your database and its variables.'
DATABASE_ICON =  ICONS + r'\database.png'

SEARCH_DESCRIPTION = 'Search Engine: Navigate your database and specify criteria for extracting data.'
SEARCH_ICON = ICONS + r'\search.png'

GEO_SEARCH_DESCRIPTION = 'Geographic Search: Explore your database and produce geographical information system visualizations.'
GEO_SEARCH_ICON = ICONS + r'\geoSearch.png'

MACHINE_LEARNING_DESCRIPTION = 'Machine Learning: Perform statistical analysis on your data and visualize your results.'
MACHINE_LEARNING_ICON = ICONS + r'\machineLearning.png'


BACKGROUND_IMAGE = ICONS + r'\background.png'
DOWN_ARROW_ICON = ICONS + r'downArrow.png'

BACKGROUND_COLOR_DARK = r'rgb(25,25,25)'
BACKGROUND_COLOR_LIGHT = r'rgb(49,49,49)'
BORDER_COLOR = r'rgb(70,70,70)'
FONT_COLOR = r'rgb(255,255,255)'
BUTTON_COLOR = r'rgb(55,55,55)'
BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'
LIGHT_GRADIENT = r'qlineargradient(x1: 0.4, y1: 1, x2: -0.3, y2: -0.8, stop: 0.05 rgb(32,33,30), stop: 0.9 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
DARK_GRADIENT = r'qlineargradient(x1: 0.4, y1: 1, x2: -0.3, y2: -0.8, stop: 0.05 rgb(22,23,20), stop: 0.9 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'
WIDGET_STYLE = """
            QWidget
            {
                color: """+FONT_COLOR+""";
                background-color: """+BACKGROUND_COLOR_DARK+""";
                selection-background-color:"""+BACKGROUND_COLOR_LIGHT+""";
                selection-color: """+FONT_COLOR+""";
                background-clip: border;
                border-image: none;
                outline: 0;
            }
            """
ENTRY_STYLE = """
            QWidget
            {
                background-color: """+BACKGROUND_COLOR_DARK+""";
                border: 2px """+BORDER_STYLE+""" """+BORDER_COLOR+""";
            }
            """
WINDOW_STYLE = """
            QMainWindow {
                background: """+BACKGROUND_COLOR_DARK+""";
                color: """+FONT_COLOR+""";
                background-image: url(./Interface/InterfaceUtilities/Icons/bg2.png);
                
            }
            QMainWindow::separator {
                background: """+BACKGROUND_COLOR_LIGHT+""";
                color: """+BACKGROUND_COLOR_LIGHT+""";
                width: 10px; /* when vertical */
                height: 10px; /* when horizontal */
            }
            
            QMainWindow::separator:hover {
                background: """+BACKGROUND_COLOR_LIGHT+""";
                color: black;
            }

            """
BUTTON_STYLE = """
            QPushButton {
                background: """+LIGHT_GRADIENT+""";
                color: """+FONT_COLOR+""";
                border: """+BORDER_WIDTH+""" """+BORDER_STYLE+""" """+BORDER_COLOR+""";
                border-radius: """+BORDER_RADIUS+""";
                min-width: 80px;
            }
            
            QPushButton:hover {
                background: """+DARK_GRADIENT+""";
                border: """+BORDER_WIDTH+""" """+BORDER_STYLE+""" """+BORDER_COLOR+""";
            }
            
            QPushButton:pressed {
                background: """+DARK_GRADIENT+""";
            }
            
            QPushButton:flat {
                background: """+LIGHT_GRADIENT+""";
                border: """+BORDER_COLOR+""";; /* no border for a flat push button */
            }
            
            QPushButton:default {
                background: """+LIGHT_GRADIENT+""";
                border: """+BORDER_WIDTH+""" """+BORDER_STYLE+""" """+BORDER_COLOR+""";
            }
                

            """

CHECK_STYLE = """
            QCheckBox {
                spacing: 5px;
            }
            
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }
            
            QCheckBox::indicator:unchecked {
            }
            
            QCheckBox::indicator:unchecked:hover {
            }
            
            QCheckBox::indicator:unchecked:pressed {
            }
            
            QCheckBox::indicator:checked {
            }
            
            QCheckBox::indicator:checked:hover {
            }
            
            QCheckBox::indicator:checked:pressed {
            }
            
            QCheckBox::indicator:indeterminate:hover {
            }
            
            QCheckBox::indicator:indeterminate:pressed {
            }
            """

COMBO_STYLE = """
            QComboBox {
                background: """+LIGHT_GRADIENT+""";
                color: rgb(255,255,255);
                border: 3px """+BORDER_STYLE+""" """+BORDER_COLOR+""";
                padding: 1px 1px 1px 3px;
                min-width: 6px;
            }
            
            QComboBox:!editable::hover {
                background: """+DARK_GRADIENT+"""; 
            }
                
            QComboBox:editable {
                background: """+DARK_GRADIENT+""";         
            }
                
            /* QComboBox gets the "on" state when the popup is open */
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: """+DARK_GRADIENT+""";
            }
                
            QComboBox:on { /* shift the text when the popup opens */
                padding-top: 3px;
                padding-left: 4px;
            }
                
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 10px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: """+BORDER_STYLE+""";
                border-top-right-radius: """+BORDER_RADIUS+""";
                border-bottom-right-radius: """+BORDER_RADIUS+""";
            }
                
            QComboBox::down-arrow {
                
            }
                
            QComboBox::down-arrow:on { /* shift the arrow when popup is open */
                top: 1px;
                left: 1px;
            }
            """

FRAME_STYLE = """
            QFrame {
                background: """+BACKGROUND_COLOR_DARK+""";
                background-image: url(./Interface/InterfaceUtilities/Icons/bg2.png);
            }
            """
LABEL_STYLE = """
            QLabel {
                background: """+BACKGROUND_COLOR_DARK+""";
                color: """+FONT_COLOR+""";
                border: 1px """+BORDER_STYLE+""" """+BORDER_COLOR+""";
                border-radius: """+BORDER_RADIUS+""";
                padding: 2px;
            }
            """

MENUBAR_STYLE = """
            QMenuBar {
                background: """+BACKGROUND_COLOR_LIGHT+""";
                color: """+FONT_COLOR+""";
                border: 1px solid rgb(0,0,0);
            }
    
            QMenuBar::item {
                background: """+BACKGROUND_COLOR_LIGHT+""";
                color: """+FONT_COLOR+""";
            }
    
            QMenuBar::item::selected {
                background: """+BACKGROUND_COLOR_DARK+""";
            }
    
            QMenu {
                background: """+BACKGROUND_COLOR_LIGHT+""";
                color: """+FONT_COLOR+""";
                border: 1px solid #000;           
            }
    
            QMenu::item::selected {
                background-color: """+BACKGROUND_COLOR_DARK+""";
            }
            """

SCROLL_STYLE = """
            QScrollBar:vertical {
                 background: """+BACKGROUND_COLOR_DARK+""";
                 width: 15px;
                 margin: 22px 0 22px 0;
             }
             QScrollBar::handle:vertical {
                 background: """+DARK_GRADIENT+""";
                 min-height: 20px;
             }
             QScrollBar::handle:vertical:pressed {
                 background: """+DARK_GRADIENT+""";
                 min-height: 20px;
             }
             QScrollBar::add-line:vertical {
                 background: """+LIGHT_GRADIENT+""";
                 height: 20px;
                 subcontrol-position: bottom;
                 subcontrol-origin: margin;
             }
            
             QScrollBar::sub-line:vertical {
                 background: """+DARK_GRADIENT+""";
                 height: 20px;
                 subcontrol-position: top;
                 subcontrol-origin: margin;
             }
             QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                 border: 2px solid """+DARK_GRADIENT+""";
                 width: 3px;
                 height: 3px;
                 background: black;
             }
            QScrollBar::up-arrow:vertical:pressed, QScrollBar::down-arrow:vertical:pressed {
                 border: 2px solid """+LIGHT_GRADIENT+""";
                 width: 3px;
                 height: 3px;
                 background: black;
             }
            
             QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                 background: none;
             }
             """
SPLITTER_STYLE = """
            QSplitter {
                background: """+BACKGROUND_COLOR_DARK+""";
            }
            QSplitter::handle {
                background: """+BACKGROUND_COLOR_LIGHT+""";
                color: """+FONT_COLOR+""";
            }

            QSplitter::handle:hover {
                background: white;
            }

            
            QSplitter::handle:horizontal {
                width: 8px;
            }
            
            QSplitter::handle:vertical {
                height: 8px;
            }
            
            QSplitter::handle:pressed {
                background-color: """+DARK_GRADIENT+""";
            }
            """

STATUSBAR_STYLE = """
            QStatusBar {
                background: """+BACKGROUND_COLOR_LIGHT+""";
                color: """+FONT_COLOR+""";
                border: 1px solid rgb(100,100,100);
            }
            
            QStatusBar::item {
                border: """+BORDER_WIDTH+""" """+BORDER_STYLE+""" """+BORDER_COLOR+""";
                border-radius: """+BORDER_RADIUS+""";
            
            }"""

TOOLBAR_STYLE = """ 
            QToolBar, QToolButton, QToolTip { 
                background: rgb(56,60,55);
                background: """+LIGHT_GRADIENT+""";

                color: """ + FONT_COLOR + """;
                spacing: 3px; /* spacing between items in the tool bar */
                border: 1px """+BORDER_STYLE+""" """+BORDER_COLOR+""";
                
            } 
            QToolBar {
                background-image: url(./Interface/InterfaceUtilities/Icons/bg2.png);
            }

            
            QToolButton:hover {
                background: """+DARK_GRADIENT+""";
                border: 0px;
            }
            
            QToolBar::handle {
                background: """+BACKGROUND_COLOR_LIGHT+""";
                border: 1px solid rgb(100,100,100);
            } 
            """
