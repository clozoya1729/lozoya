groupBox = """
    QGroupBox {
        border: none;
        background: {backgroundColorDark};
        font-size:{fontSize};
        margin-bottom: 0ex;
        margin-top: 3.6ex; /* leave space at the top for the title */
        padding: 0;
        margin: 0;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center; /* position at the top center */
        padding: 0 3px 0 0;   
        margin: 0;     
        font-size:{fontSize};     
    }
    """
groupBoxNu = """
    QGroupBox {
        border: none;
        margin-top: 3ex; /* leave space at the top for the title */
        font-size: {fontSize};
        padding: 0;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left; /* position at the top center */
        padding: 0 3px 0 0;   
        margin: 0;    
        font-size:{fontSize}; 
    }
"""
frame = """
    QFrame {
        background: {backgroundColorDark};
        border: none;
    }
    """
splitter = """
    QSplitter {
        background: {backgroundColorDark};
    }
    QSplitter::handle {
        background: orange;
        color: {fontColor};
    }
    QSplitter::handle:hover {
        background: white;
    }
    QSplitter::handle:horizontal {
        width: 4px;
    }
    QSplitter::handle:vertical {
        height: 2px;
    }
    QSplitter::handle:pressed {
        background-color: {yellowGradient};
    }
    """
tab = """
    QTabWidget{
        background: {backgroundColorDark};
        font-size: {fontSize};
    }
    QTabWidget::pane { /* The tab widget frame */
        background: {backgroundColorDark};
    }
    QTabWidget::tab-bar {
        left: 5px; /* move to the right by 5px */
        background: {backgroundColorDark};
    }
    /* Style the tab using the tab sub-control. Note that it reads QTabBar _not_ QTabWidget */
    QTabBar::tab {
        background: {backgroundColorDark};
        border: none;
        min-width: 26ex;
        margin: 0;
        padding: 0;
        border-bottom: 1px solid rgb(50,50,100);
    }
    QTabBar::tab:!selected:hover {
            background: {backgroundColorLight};
    }
    QTabBar::tab:selected {
        color: orange;
        margin: 0;
        padding: 0;
        background: {backgroundColorLighter};
        border-bottom: 1px solid rgb(50,50,255);
    }
    QTabBar::tab:selected:hover {
    }
    QTabBar::tab:!selected {
    }
    """
scroll = """
    QScrollBar:vertical {
        background: rgb(50,50,50);         width: 5px;
     }
     QScrollBar:horizontal {
        background: rgb(50,50,50);         height: 5px;
     }
     QScrollBar::handle:vertical {
         background: rgb(200,200,200);
         min-height: 20px;
     }
    QScrollBar::handle:horizontal {
         background: rgb(200,200,200);
         min-height: 20px;
     }
     QScrollBar::handle:vertical:pressed {
         background: rgb(255,255,255);
         min-height: 20px;
     }
    QScrollBar::handle:horizontal:pressed {
         background: rgb(255,255,255);
         min-width: 20px;
     }
     QScrollBar::add-line:vertical {
        background: rgb(50,50,50);
     }
     QScrollBar::sub-line:vertical {
        background: rgb(50,50,50);
     }
     QScrollBar::add-line:horizontal {
        background: rgb(50,50,50);
     }
     QScrollBar::sub-line:horizontal {
         background: rgb(50,50,50);
     }
     QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
            background: none;
     }
    QScrollBar::up-arrow:vertical:pressed, QScrollBar::down-arrow:vertical:pressed {
     }
     QScrollBar::add-page, QScrollBar::sub-page {
         background: none;
     }
     """

menubar = """
    QMenu {
        background: rgb(120,120,120);
        color: {fontColor};
        width: 120px;
    }
    QMenu::separator {
        height: 2px;
    }
    QMenu::item {
        background: rgb(110,110,120);
    }
    QMenu::item:selected {
        background: rgb(90,90,100);
    }
    QMenuBar {
        background: rgb(20,20,20);
        border: 1px solid rgb(0,0,0);
        color: {fontColor};
        font-size: {fontSize};
    }
    QMenuBar::item {
        background: {backgroundColorLightererer};
        color: {fontColor};
    }
    QMenuBar::item:selected {
        background: {backgroundColorLighterer};
    }
    QMenuBar::item:pressed {
        background: {backgroundColorLighter};
        color: black;
    }
    [objectName^="exportAction"] {
        background: blue;
    }
    """
toolbar = """ 
    QToolBar, QToolButton, QToolTip { 
        background: rgb(56,60,55);
        background: {lightGradient};
        color: {fontColor};
        spacing: 3px; /* spacing between items in the tool bar */
        border: {borderWidth} {borderStyle} {borderColor};
    } 
    QToolButton {
        background-image: url(./interface/image/background/background.png);
        background: {orangeGradient};
        border: 8px solid white;
        border-left-color: rgb(70,70,70);
        border-top-color: rgb(100,100,100);
        border-right-color: rgb(55,55,55);
        border-bottom-color: rgb(45,45,45);
        color: rgb(0,0,50);
        height: 72px;
        width: 72px;
        font-size: {fontSize};
        font-family: {font};
        font-weight: bold;
    }
    QToolButton:hover {
        background: {yellowGradient};
        border-left-color: rgb(255,195,25);
        border-top-color: rgb(255,230,75);
        border-right-color: rgb(230,130,50);
        border-bottom-color: rgb(190,90,35);
        margin: -2px;
    }
    QToolButton:pressed {
        background: {orangeGradientPressed};
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);
        border-bottom-color: rgb(160,60,10);
        margin: 2px;
    }
    QToolBar::handle {
        subcontrol-position: top;
        background: {backgroundColorLight};
        border: {borderWidth} {borderStyle} rgb(100,100,100);
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);
        border-bottom-color: rgb(160,60,10);
    } 
    """
window = """
    QMainWindow {
        background-clip: border;
        background-color: {backgroundColorDark};
        color: {fontColor};
        background-image: url(./interface/image/background.png);
        margin: 0;
        padding: 0;
    }
    QMainWindow::separator {
        background: {backgroundColorDark};
        color: {backgroundColorLight};
        width: 1px; /* when vertical */
        height: 1px; /* when horizontal */
    }
    QMainWindow::separator:hover {
        background: {backgroundColorLight};
        color: black;
    }
    """
tooltip = """
    QToolBar, QToolButton, QToolTip
    {
        background: rgb(56,60,55);
        background: {lightGradient};
        color: {fontColor};
        spacing: 3px; /* spacing between items in the tool bar */
        border: 1px {borderStyle} {borderColor};
    }
    """
widget = tooltip + """
     QWidget
     {
         color: {fontColor};
         background-clip: border;
         background-color: {backgroundColorDark};
         selection-background-color: {backgroundColorLight};
         selection-color: {fontColor};
         outline: 0;
         font-family: {font};
         margin: 0;
         padding: 0;
     }
     """
label = tooltip + """
        QLabel {
            background: {labelColor};
            color: {fontColor};
            border: 1px {borderStyle} {borderColor};
            border-top-left-radius: 5px;
            border-top-right-radius: 10px;
            border-bottom-right-radius: 5px;
            border-bottom-left-radius: 10px;
            padding: 2px;
            font-size: {fontSize};
        }
        """

BORDER_COLOR = r'rgb(70,70,70)'
BUTTON_COLOR = r'rgb(55,55,55)'
BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'
GROUP_BOX_STYLE = """
                QGroupBox {
                }
            """
LIST_STYLE = """
            QListView {
                background: """ + BACKGROUND_COLOR_DARK + """;
                selection-color: """ + CONTRASTED_COLOR + """;
                selection-background-color: """ + CONTRASTED_COLOR + """;
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            }
            QListView:item:selected {
                color: """ + BACKGROUND_COLOR_DARK + """;
                background: """ + CONTRASTED_COLOR + """;
            }
            QListView:item:hover {
                background: """ + BACKGROUND_COLOR_LIGHTER + """;
            }
            QListView:item:selected:hover {
                background: """ + CONTRASTED_COLOR + """;
            }
            """
TOOLTIP_STYLE = """ 
            QToolBar, QToolButton, QToolTip { 
                background: rgb(56,60,55);
                background: """ + LIGHT_GRADIENT + """;

                color: """ + FONT_COLOR + """;
                spacing: 3px; /* spacing between items in the tool bar */
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            }
            """
WIDGET_STYLE = TOOLTIP_STYLE + """
            QWidget
            {
                color: """ + FONT_COLOR + """;
                background-color: """ + BACKGROUND_COLOR_DARK + """;
                selection-background-color:""" + BACKGROUND_COLOR_LIGHT + """;
                selection-color: """ + FONT_COLOR + """;
                background-clip: border;
                border-image: none;
                outline: 0;
            }
            """
ENTRY_STYLE = """
            QWidget
            {
                background-color: """ + LABEL_COLOR + """;
                border: 2px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            }
            """
WINDOW_STYLE = """
            QMainWindow {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);

            }
            QMainWindow::separator {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + BACKGROUND_COLOR_LIGHT + """;
                width: 10px; /* when vertical */
                height: 10px; /* when horizontal */
            }
            QMainWindow::separator:hover {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: black;
            }
            """
BUTTON_STYLE = """
            QPushButton {
                background: """ + LIGHT_GRADIENT + """;
                color: """ + FONT_COLOR + """;
                border: """ + BORDER_WIDTH + """ """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;
                min-width: 80px;
                icon2-size: 500px 100px;
            }
            QPushButton:hover {
                background: """ + DARK_GRADIENT + """;
                border: """ + BORDER_WIDTH + """ """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            }
            QPushButton:pressed {
                background: """ + DARK_GRADIENT + """;
            }
            QPushButton:flat {
                background: """ + LIGHT_GRADIENT + """;
                border: """ + BORDER_COLOR + """;; /* no border for a flat push button */
            }
            QPushButton:default {
                background: """ + LIGHT_GRADIENT + """;
                border: """ + BORDER_WIDTH + """ """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            }
            QPushButton::icon2 {
                height: 100px;
                width: 100px;
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
            QComboBox QAbstractItemView {
                border: 2px solid """ + BACKGROUND_COLOR_LIGHT + """;
                selection-background-color: """ + BACKGROUND_COLOR_LIGHTER + """;
                background: """ + BACKGROUND_COLOR_DARK + """;
            }
            QComboBox {
                background: """ + LIGHT_GRADIENT + """;
                color: """ + CONTRASTED_COLOR + """;
                border: 3px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                padding: 1px 1px 1px 3px;
                min-width: 6px;
            }
            QComboBox:!editable::hover {
                background: """ + DARK_GRADIENT + """; 
            }
            QComboBox:editable {
                background: """ + DARK_GRADIENT + """;         
            }
            /* QComboBox gets the "on" state when the popup is open */
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: """ + DARK_GRADIENT + """;
            }
            QComboBox:on { /* shift the text when the popup opens */
                padding-top: 3px;
                padding-left: 4px;
                background: """ + LIGHT_GRADIENT + """;
            }
            QComboBox::drop-down {
                background: """ + LIGHT_GRADIENT + """;
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 10px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: """ + BORDER_STYLE + """;
                border-top-right-radius: """ + BORDER_RADIUS + """;
                border-bottom-right-radius: """ + BORDER_RADIUS + """;
            }
            QComboBox::down-arrow {
            }
            QComboBox::down-arrow:on { /* shift the arrow when popup is open */
                top: 1px;
                left: 1px;
            }
            """
DIAL_STYLE = """CustomDial {
                background-color: #27272B;
                color: #FFFFFF;
                qproperty-knobRadius: 5px;
                qproperty-knobMargin: 5px;
            }
            """
FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """
LABEL_STYLE = TOOLTIP_STYLE + """
            QLabel {
                background: """ + LABEL_COLOR + """;
                color: """ + FONT_COLOR + """;
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;
                padding: 2px;
            }
            """
MENUBAR_STYLE = """
            QMenuBar {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid rgb(0,0,0);
            }
            QMenuBar::item {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
            }
            QMenuBar::item::selected {
                background: """ + BACKGROUND_COLOR_DARK + """;
            }
            QMenu {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid #000;           
            }
            QMenu::item::selected {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
            }
            """
PROGRESS_BAR_STYLE = """
            QProgressBar:horizontal {
            background: """ + BACKGROUND_COLOR_DARK + """;
            border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            border-radius: """ + BORDER_RADIUS + """;
            width: 100px;
            height: 5px;
            text-align: right;
            margin-right: 10ex;
            }
            QProgressBar::chunk:horizontal {
            background: """ + CONTRASTED_COLOR + """;
            border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            border-radius: """ + BORDER_RADIUS + """;
            }
            """
SCROLL_STYLE = """
            QScrollBar:vertical {
                 background: """ + BACKGROUND_COLOR_DARK + """;
                 width: 15px;
                 margin: 22px 0 22px 0;
             }
             QScrollBar::handle:vertical {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::handle:vertical:pressed {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::add-line:vertical {
                 background: """ + LIGHT_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: bottom;
                 subcontrol-origin: margin;
             }
             QScrollBar::sub-line:vertical {
                 background: """ + DARK_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: top;
                 subcontrol-origin: margin;
             }
             QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                 border: 2px solid """ + DARK_GRADIENT + """;
                 width: 3px;
                 height: 3px;
                 background: black;
             }
            QScrollBar::up-arrow:vertical:pressed, QScrollBar::down-arrow:vertical:pressed {
                 border: 2px solid """ + LIGHT_GRADIENT + """;
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
                background: """ + BACKGROUND_COLOR_DARK + """;
            }
            QSplitter::handle {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
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
                background-color: """ + DARK_GRADIENT + """;
            }
            """
STATUSBAR_STYLE = """
            QStatusBar {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid rgb(100,100,100);
            }
            QStatusBar::item {
                border: """ + BORDER_WIDTH + """ """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;
            }
            """
TAB_STYLE = """
            QTabWidget::pane { /* The tab widget frame */
            border-top: 2px solid """ + BACKGROUND_COLOR_LIGHT + """;
            }
            QTabWidget::tab-bar {
            left: 5px; /* move to the right by 5px */
            background: """ + BACKGROUND_COLOR_LIGHT + """;
            }
            /* Style the tab using the tab sub-control. Note that it reads QTabBar _not_ QTabWidget */
            QTabBar::tab {
            background: """ + BACKGROUND_COLOR_LIGHT + """;
            border: 2px solid """ + BACKGROUND_COLOR_LIGHT + """;
            border-bottom-color: """ + BACKGROUND_COLOR_LIGHT + """; /* same as the pane color */
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            min-width: 8ex;
            padding: 2px;
            }
            QTabBar::tab:selected, QTabBar::tab:hover {
            background: """ + BACKGROUND_COLOR_DARK + """;
            }
            QTabBar::tab:selected {
            border-color: #9B9B9B;
            border-bottom-color: #C2C7CB; /* same as pane color */
            }
            QTabBar::tab:!selected {
            margin-top: 2px; /* make non-selected tabs look smaller */
            background: """ + BACKGROUND_COLOR_LIGHT + """;
            border-color: """ + BACKGROUND_COLOR_LIGHTER + """;
            }
            """

TOOLBAR_STYLE = """ 
            QToolBar, QToolButton, QToolTip { 
                background: rgb(56,60,55);
                background: """ + LIGHT_GRADIENT + """;
                color: """ + FONT_COLOR + """;
                spacing: 3px; /* spacing between items in the tool bar */
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            } 
            QToolBar {
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            QToolButton:hover {
                background: """ + DARK_GRADIENT + """;
                border: 0px;
            }
            QToolBar::handle {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                border: 1px solid rgb(100,100,100);
            } 
            """

BORDER_COLOR = r'rgb(70,70,70)'
BUTTON_COLOR = r'rgb(55,55,55)'
BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'

BACKGROUND_COLOR = r'rgb(255,255,255)'
CONTRASTED_COLOR = r'rgb(0,0,0)'
LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(222,222,222), stop: 0.5 rgb(248,250,247), stop: 0.5 rgb(253,255,252), stop: 1.0 rgb(232,232,232));'
DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(190,193,190), stop: 0.5 rgb(245,245,245), stop: 0.5 rgb(254,255,253), stop: 1.0 rgb(200,198,200));'
FONT_COLOR = r'rgb(0,0,0)'

GROUP_BOX_STYLE = """
                QGroupBox {
                }
            """

LIST_STYLE = """
            QListView {
                background: """ + BACKGROUND_COLOR + """;
                selection-color: """ + CONTRASTED_COLOR + """;
                selection-background-color: """ + CONTRASTED_COLOR + """;
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            }
            QListView:item:selected {
                color: """ + BACKGROUND_COLOR + """;
                background: """ + CONTRASTED_COLOR + """;
            }
            QListView:item:hover {
                background: """ + BACKGROUND_COLOR + """;
            }
            QListView:item:selected:hover {
                background: """ + CONTRASTED_COLOR + """;
            }
            """
TOOLTIP_STYLE = """ 
            QToolBar, QToolButton, QToolTip { 
                background: rgb(56,60,55);
                background: """ + LIGHT_GRADIENT + """;

                color: """ + FONT_COLOR + """;
                spacing: 3px; /* spacing between items in the tool bar */
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;

            } """

WIDGET_STYLE = TOOLTIP_STYLE + """
            QWidget
            {
                color: """ + FONT_COLOR + """;
                background-color: """ + BACKGROUND_COLOR + """;
                selection-background-color:""" + BACKGROUND_COLOR + """;
                selection-color: """ + FONT_COLOR + """;
                background-clip: border;
                outline: 0;
            }
            """
ENTRY_STYLE = """
            QWidget
            {
                background-color: """ + BACKGROUND_COLOR + """;
                border: 2px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            }
            """
WINDOW_STYLE = """
            QMainWindow {
                background: """ + BACKGROUND_COLOR + """;
                color: """ + FONT_COLOR + """;

            }
            QMainWindow::separator {
                background: """ + BACKGROUND_COLOR + """;
                color: """ + BACKGROUND_COLOR + """;
                width: 10px; /* when vertical */
                height: 10px; /* when horizontal */
            }

            QMainWindow::separator:hover {
                background: """ + BACKGROUND_COLOR + """;
                color: black;
            }

            """
BUTTON_STYLE = """
            QPushButton {
                background: """ + LIGHT_GRADIENT + """;
                color: """ + FONT_COLOR + """;
                border: """ + BORDER_WIDTH + """ """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;
                min-width: 80px;
                icon2-size: 500px 100px;
            }

            QPushButton:hover {
                background: """ + DARK_GRADIENT + """;
                border: """ + BORDER_WIDTH + """ """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            }

            QPushButton:pressed {
                background: """ + DARK_GRADIENT + """;
            }

            QPushButton:flat {
                background: """ + LIGHT_GRADIENT + """;
                border: """ + BORDER_COLOR + """;; /* no border for a flat push button */
            }

            QPushButton:default {
                background: """ + LIGHT_GRADIENT + """;
                border: """ + BORDER_WIDTH + """ """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            }
            QPushButton::icon2 {
                height: 100px;
                width: 100px;
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
            QComboBox QAbstractItemView {
                border: 2px solid """ + BACKGROUND_COLOR + """;
                selection-background-color: """ + BACKGROUND_COLOR + """;
                background: """ + BACKGROUND_COLOR + """;
            }

            QComboBox {
                background: """ + LIGHT_GRADIENT + """;
                color: """ + CONTRASTED_COLOR + """;
                border: 3px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                padding: 1px 1px 1px 3px;
                min-width: 6px;
            }

            QComboBox:!editable::hover {
                background: """ + DARK_GRADIENT + """; 
            }

            QComboBox:editable {
                background: """ + DARK_GRADIENT + """;         
            }

            /* QComboBox gets the "on" state when the popup is open */
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: """ + DARK_GRADIENT + """;
            }

            QComboBox:on { /* shift the text when the popup opens */
                padding-top: 3px;
                padding-left: 4px;
                background: """ + LIGHT_GRADIENT + """;
            }

            QComboBox::drop-down {
                background: """ + LIGHT_GRADIENT + """;
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 10px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: """ + BORDER_STYLE + """;
                border-top-right-radius: """ + BORDER_RADIUS + """;
                border-bottom-right-radius: """ + BORDER_RADIUS + """;
            }

            QComboBox::down-arrow {

            }

            QComboBox::down-arrow:on { /* shift the arrow when popup is open */
                top: 1px;
                left: 1px;
            }
            """

DIAL_STYLE = """CustomDial {
                background-color: #27272B;
                color: #FFFFFF;
                qproperty-knobRadius: 5px;
                qproperty-knobMargin: 5px;
            }
            """
FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR + """;
            }
            """
LABEL_STYLE = TOOLTIP_STYLE + """
            QLabel {
                background: """ + BACKGROUND_COLOR + """;
                color: """ + FONT_COLOR + """;
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;
                padding: 2px;
            }
            """

MENUBAR_STYLE = """
            QMenuBar {
                background: """ + BACKGROUND_COLOR + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid rgb(0,0,0);
            }

            QMenuBar::item {
                background: """ + BACKGROUND_COLOR + """;
                color: """ + FONT_COLOR + """;
            }

            QMenuBar::item::selected {
                background: """ + BACKGROUND_COLOR + """;
            }

            QMenu {
                background: """ + BACKGROUND_COLOR + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid #000;           
            }

            QMenu::item::selected {
                background-color: """ + BACKGROUND_COLOR + """;
            }
            """

PROGRESS_BAR_STYLE = """
            QProgressBar:horizontal {
            background: """ + BACKGROUND_COLOR + """;
            border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            border-radius: """ + BORDER_RADIUS + """;
            width: 100px;
            height: 5px;
            text-align: right;
            margin-right: 10ex;

            }
            QProgressBar::chunk:horizontal {
            background: """ + CONTRASTED_COLOR + """;
            border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            border-radius: """ + BORDER_RADIUS + """;
            }
            """

SCROLL_STYLE = """
            QScrollBar:vertical {
                 background: """ + BACKGROUND_COLOR + """;
                 width: 15px;
                 margin: 22px 0 22px 0;
             }
             QScrollBar::handle:vertical {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::handle:vertical:pressed {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::add-line:vertical {
                 background: """ + LIGHT_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: bottom;
                 subcontrol-origin: margin;
             }

             QScrollBar::sub-line:vertical {
                 background: """ + DARK_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: top;
                 subcontrol-origin: margin;
             }
             QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                 border: 2px solid """ + DARK_GRADIENT + """;
                 width: 3px;
                 height: 3px;
                 background: black;
             }
            QScrollBar::up-arrow:vertical:pressed, QScrollBar::down-arrow:vertical:pressed {
                 border: 2px solid """ + LIGHT_GRADIENT + """;
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
                background: """ + BACKGROUND_COLOR + """;
            }
            QSplitter::handle {
                background: """ + BACKGROUND_COLOR + """;
                color: """ + FONT_COLOR + """;
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
                background-color: """ + DARK_GRADIENT + """;
            }
            """

STATUSBAR_STYLE = """
            QStatusBar {
                background: """ + BACKGROUND_COLOR + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid rgb(100,100,100);
            }

            QStatusBar::item {
                border: """ + BORDER_WIDTH + """ """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;

            }"""

TAB_STYLE = """
            QTabWidget::pane { /* The tab widget frame */
            border-top: 2px solid """ + BACKGROUND_COLOR + """;
            }
            QTabWidget::tab-bar {
            left: 5px; /* move to the right by 5px */
            background: """ + BACKGROUND_COLOR + """;
            }
            /* Style the tab using the tab sub-control. Note that it reads QTabBar _not_ QTabWidget */
            QTabBar::tab {
            background: """ + BACKGROUND_COLOR + """;
            border: 2px solid """ + BACKGROUND_COLOR + """;
            border-bottom-color: """ + BACKGROUND_COLOR + """; /* same as the pane color */
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            min-width: 8ex;
            padding: 2px;
            }
            QTabBar::tab:selected, QTabBar::tab:hover {
            background: """ + BACKGROUND_COLOR + """;
            }
            QTabBar::tab:selected {
            border-color: #9B9B9B;
            border-bottom-color: #C2C7CB; /* same as pane color */
            }
            QTabBar::tab:!selected {
            margin-top: 2px; /* make non-selected tabs look smaller */
            background: """ + BACKGROUND_COLOR + """;
            border-color: """ + BACKGROUND_COLOR + """;
            }
            """

TOOLBAR_STYLE = """ 
            QToolBar, QToolButton, QToolTip { 
                background: rgb(56,60,55);
                background: """ + LIGHT_GRADIENT + """;

                color: """ + FONT_COLOR + """;
                spacing: 3px; /* spacing between items in the tool bar */
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            } 
            QToolButton:hover {
                background: """ + DARK_GRADIENT + """;
                border: 0px;
            }
            QToolBar::handle {
                background: """ + BACKGROUND_COLOR + """;
                border: 1px solid rgb(100,100,100);
            } 
            """

button = """
    QPushButton {
        background: {lightererGradient};
        color: {fontColor};
        border: {borderWidth} {borderStyle} {borderColor};
        border-radius: {borderRadius};
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton:hover {
        background: {darkGradient};
        border: {borderWidth} {borderStyle} orange;
    }
    QPushButton:pressed {
        background: rgb(200,100,50);
        color: black;
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
buttonDisconnected = """
    QPushButton {
        background: {lightererGradient};
        color: {fontColor};
        border: {borderWidth} {borderStyle} rgba(100,25,25);
        border-radius: {borderRadius};
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton:hover {
        background: yellow;
        border: {borderWidth} {borderStyle} rgba(125,200,50);
        color: black;
    }
    QPushButton:pressed {
        background: rgb(150,50,25);
        color: black;
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
buttonDisabled = """
    QPushButton {
        background: {darkGradient};
        color: {lightGradient};
        border: {borderWidth} {borderStyle} {darkGradient};
        border-radius: {borderRadius};
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton:hover {
        background: {darkGradient};
        border: {borderWidth} {borderStyle} {darkGradient};
        color: rgba(50,20,20);
    }
    QPushButton:pressed {
        background: {darkGradient};
        border: {borderWidth} {borderStyle} {darkGradient};
        color: rgba(75,25,25);
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
buttonConnected = """
    QPushButton {
        background: orange;
        color: {fontColor};
        border: {borderConnected};
        border-radius: {borderRadius};
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton::hover {
        background: yellow;
        border: {borderConnectedHover};
        color: black;
    }
    QPushButton:pressed {
        background: rgb(150,50,25);
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
buttonValid = """
    QPushButton {
        background: {lightererGradient};
        color: {fontColor};
        border: {borderValid};
        border-radius: {borderRadius};
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton:hover {
        background: yellow;
        border: {borderValidHover};
        color: darkblue;
    }
    QPushButton:pressed {
        background: orange;
        color: blue;
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
checkBox = """
    QCheckBox {
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
    }
    QCheckBox::indicator:checked {
        background: orange;
        border: {borderValid};
    }
    QCheckBox::indicator:unchecked {
        background: black;
        border: {borderWidth} {borderStyle} {borderColor};
    }
    QCheckBox::indicator:unchecked:pressed {
        background: yellow;
    }
    QCheckBox::indicator:checked:pressed {
        background: orange;
    }
    QCheckBox::indicator:indeterminate:hover {
    }
    QCheckBox::indicator:indeterminate:pressed {
        background: orange;    
    }
    QCheckBox::indicator:checked:hover {
        background: orange;
        border: {borderValidHover};
    }
    QCheckBox::indicator:unchecked:hover {
        background: black;
        border: {borderWidth} {borderStyle} orange;
    }
    """
checkboxChecked = """
    QCheckBox {
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 12px;
        height: 12px;
        background: orange;
        border: 4px {borderStyle} {darkGreenGradient};
    }
    QCheckBox::indicator:hover {
        background: orange;
        border: 4px {borderStyle} {lightGreenGradient};
    }
    QCheckBox::indicator:pressed {
        background: yellow;
        border: 4px {borderStyle} {darkGreenGradient};
    }
    """
checkboxUnchecked = """
    QCheckBox {
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 12px;
        height: 12px;
        background: black;
        border: 4px {borderStyle} {lightGradient};
    }
    QCheckBox::indicator:hover {
        background: rgb(200,100,0);
        border: 4px {borderStyle} {darkGradient};
    }
    QCheckBox::indicator:pressed {
        background: yellow;
        border: 4px {borderStyle} {lightGradient};
    }
    """
dropdown = """
    QComboBox QAbstractItemView {
        border: {borderWidth} solid {backgroundColorLight};
        selection-background-color: {backgroundColorLighter};
        background: {backgroundColorDark};
        font-size: {fontSize};
    }
    QComboBox {
        background: {lightGradient};
        color: {contrastColor};
        border: 3px {borderStyle} {borderColor};
        padding: 1px 1px 1px 3px;
        min-width: 6px;
        font-size: {fontSize};
    } 
    QComboBox:!editable::hover {
        background: {darkGradient}; 
        border: {borderWidth} {borderStyle} orange;
    }
    QComboBox:editable {
        background: {darkGradient};         
    }
    /* QComboBox gets the "on" state when the popup is open */
    QComboBox:!editable:on, QComboBox::drop-down:editable:on {
        background: {darkGradient};
    }
    QComboBox:on { /* shift the text when the popup opens */
        padding-top: 3px;
        padding-left: 4px;
        background: {lightGradient};
    }
    QComboBox::drop-down {
        background: {lightGradient};
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 10px;
        border-left-width: 1px;
        border-left-color: darkgray;
        border-left-style: {borderStyle};
        border-top-right-radius: {borderRadius};
        border-bottom-right-radius: {borderRadius};
    }
    QComboBox::down-arrow {
    }
    QComboBox::down-arrow:on { /* shift the arrow when popup is open */
        top: 1px;
        left: 1px;
    }
    """
dial = """CustomDial {
        background-color: #27272B;
        color: #FFFFFF;
        qproperty-knobRadius: 5;
        qproperty-knobMargin: 5;
    }
    """
entry = """
    QWidget
    {
        background-color: {labelColor};
        border: {borderWidth} {borderStyle} {borderColor};
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
        font-family: {font};
    }
    QPlainTextEdit
    {
        font-family: {font};
        font-size: {fontSize};
    }
    """
entryGlow = """
    QPlainTextEdit::hover
    {
        border: {borderWidth} {borderStyle} yellow;
        font-family: {font};
        font-size: {fontSize};
    }
    QPlainTextEdit
    {
        background-color: {labelColor};
        border: {borderWidth} {borderStyle} orange;
        font-family: {font};
        font-size: {fontSize}
        margin: 0px;
        padding: 0px;
    }
    """
entryConnected = """
    QLineEdit::hover
    {
        background-color: {entryBGConnectedHover};
        border: {borderConnectedHover};
        font-family: {font};
        font-size: {fontSize};
    }
    QLineEdit
    {
        background-color: {entryBGConnected};
        border: {borderConnected};
        font-family: {font};
        font-size: {fontSize};
        margin: 0px;
        padding: 0px;
    }
    QPlainTextEdit::hover
    {
        background-color: {entryBGConnectedHover};
        border: {borderConnectedHover};
        color: white;
        font-family: {font};
        font-size: {fontSize};
    }
    QPlainTextEdit
    {
        background-color: {entryBGConnected};
        border: {borderConnected};
        color: white;
        font-family: {font};
        font-size: {fontSize};
        margin: 0px;
        padding: 0px;
    }
    """
entryDisabled = """
    QLineEdit::hover
    {
        background: rgba(20,10,10,1);
        border: {borderWidth} {borderStyle} {darkGradient};
        font-family: {font};
        font-size: {fontSize};
    }
    QLineEdit
    {
        background: rgba(40,25,25,1);
        border: {borderWidth} {borderStyle} {darkGradient};
        font-family: {font};
        font-size: {fontSize};
        margin: 0px;
        padding: 0px;
    }
    QPlainTextEdit::hover
    {
        background: rgba(40,20,20,1);
        border: {borderWidth} {borderStyle} {darkGradient};
        color: rgba(50,25,25,1);
        font-family: {font};
        font-size: {fontSize};
        margin: 0px;
        padding: 0px;
    }
    QPlainTextEdit
    {
        background: rgba(40,25,25,1);
        border: {borderWidth} {borderStyle} {darkGradient};
        color: rgba(50,25,25,1);
        font-family: {font};
        font-size: {fontSize};
    }
    """
entryDisconnected = """
    QLineEdit::hover
    {
        background: rgba(255,0,0,1);
        border: {borderDisconnectedHover};
        font-family: {font};
        font-size: {fontSize};
    }
    QLineEdit
    {
        background: rgba(255,0,0,0.01);
        border: {borderDisconnected};
        font-family: {font};
        font-size: {fontSize};
        margin: 0px;
        padding: 0px;
    }
    QPlainTextEdit::hover
    {
        background: rgba(20,10,10,1);
        border: {borderDisconnectedHover};
        color: rgba(50,25,25,1);
        font-family: {font};
        font-size: {fontSize};
        margin: 0px;
        padding: 0px;
    }
    QPlainTextEdit
    {
        background: rgba(35,20,20,1);
        border: {borderDisconnected};
        color: rgba(50,25,25,1);
        font-family: {font};
        font-size: {fontSize};
    }
    """
entryValid = """
    QLineEdit::hover
    {
        background-color: {entryBGValidHover};
        border: {borderValidHover};
        font-family: {font};
        font-size: {fontSize};
    }
    QLineEdit
    {
        background-color: {entryBGValid};
        border: {borderValid};
        font-family: {font};
        font-size: {fontSize};
        margin: 0px;
        padding: 0px;
    }
    QPlainTextEdit::hover
    {
        background-color: {entryBGValidHover};
        border: {borderValidHover};
        font-family: {font};
        font-size: {fontSize};
    }
    QPlainTextEdit
    {
        background-color: {entryBGValid};
        border: {borderValid};
        color: white;
        font-family: {font};
        font-size: {fontSize};
        margin: 0px;
        padding: 0px;
    }
    """
entryError = """
    QLineEdit::hover
    {
        background-color: rgba(255,0,0,0.1);
        border: {borderErrorHover};
        font-family: {font};
        font-size: {fontSize};
    }
    QLineEdit
    {
        background-color: rgba(255,0,0,0.05);
        border: {borderError};
        font-family: {font};
        font-size: {fontSize};
        padding: 0px;
        margin: 0px;
    }
    """
groupBox = """
    QGroupBox {
        font-size:{fontSize};
        background: rgba(0,0,0,0.25);
        border: 4px solid rgba(50,50,50,1);
        border-radius: 10px;
        margin-top: 2.6ex; /* leave space at the top for the title */
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center; /* position at the top center */
        padding: 0 3px;        
    }
    """
groupBoxNu = """
    QGroupBox {
        border: 2px solid rgba(100,100,100,0.05);
        border-radius: 5px;
        margin-top: 3ex; /* leave space at the top for the title */
        background: rgba(0,0,0,0.1);
        font-size: 10pt;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left; /* position at the top center */
        padding: 0 3px;        
    }
"""
list = """
    QListView {
        background: {backgroundColorDark};
        selection-color: {contrastColor};
        selection-background-color: {contrastColor};
        border: 1px {borderStyle} {borderColor};
        font-size:{fontSize}
    }
    QListView:item:selected {
        color: {backgroundColorDark};
        background: {contrastColor};
    }
    QListView:item:hover {
        background: {backgroundColorLighter};
    }
    QListView:item:selected:hover {
        background: {contrastColor};
    }
    """
tooltip = """
    QToolBar, QToolButton, QToolTip
    {
        background: rgb(56,60,55);
        background: {lightGradient};
        color: {fontColor};
        spacing: 3px; /* spacing between items in the tool bar */
        border: 1px {borderStyle} {borderColor};
    }
    """
frame = """
    QFrame {
        background: {backgroundColorDark};
        background-image: url(./interface7/image/background.png);
    }
    """
label = tooltip + """
        QLabel {
            background: {labelColor};
            color: {fontColor};
            border: 1px {borderStyle} {borderColor};
            border-radius: {borderRadius};
            padding: 2px;
            font-size: {fontSize};
        }
        """
menubar = """
    QMenuBar {
        background: {backgroundColorLightererer};
        color: {fontColor};
        border: 1px solid rgb(0,0,0);
    }
    QMenuBar::item {
        background: {backgroundColorLighter};
        color: {fontColor};
    }
    QMenuBar::item::hover {
        background-color: {backgroundColorDark};
    }
    QMenu {
        background: {backgroundColorLight};
        color: {fontColor};
        border: 1px solid #000;           
    }
    QMenu::item::selected {
        background-color: {backgroundColorLight};
    }
    """
radio = """
    QRadio:default {
    }
    """
scroll = """
    QScrollBar:vertical {
         background: {backgroundColorDark};
         width: 15px;
         margin: 22px 0 22px 0;
     }
     QScrollBar::handle:vertical {
         background: {darkGradient};
         min-height: 20px;
     }
     QScrollBar::handle:vertical:pressed {
         background: {darkGradient};
         min-height: 20px;
     }
     QScrollBar::add-line:vertical {
         background: {lightGradient};
         height: 20px;
         subcontrol-position: bottom;
         subcontrol-origin: margin;
     }
     QScrollBar::sub-line:vertical {
         background: {darkGradient};
         height: 20px;
         subcontrol-position: top;
         subcontrol-origin: margin;
     }
     QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
         border: {borderWidth} solid {darkGradient};
         width: 3px;
         height: 3px;
         background: black;
     }
    QScrollBar::up-arrow:vertical:pressed, QScrollBar::down-arrow:vertical:pressed {
         border: {borderWidth} solid {lightGradient};
         width: 3px;
         height: 3px;
         background: black;
     }
     QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
         background: none;
     }
     """
slider = """
    QSlider::handle {
        background: rgb(150,50,25);
        border: {borderWidth} {borderStyle} {lightGradient};
    }
    QSlider::handle:hover {
        background: rgb(255,100,50);
        border: {borderWidth} {borderStyle} {darkGradient};
    }
    QSlider::handle:pressed {
        background: yellow;
        border: {borderWidth} {borderStyle} {darkGradient};
    }
    QSlider::sub-page {
        background: {darkGradient};
        border: {borderWidth} {borderStyle}  {lightGradient};
    }
    QSlider::add-page {
        border: {borderWidth} {borderStyle}  {lightGradient};
    }
    QSlider {
        border: {borderWidth} {borderStyle} {darkGradient};
    }
    """
splitter = """
    QSplitter {
        background: {backgroundColorDark};
    }
    QSplitter::handle {
        background: orange;
        color: {fontColor};
    }
    QSplitter::handle:hover {
        background: white;
    }
    QSplitter::handle:horizontal {
        width: 4px;
    }
    QSplitter::handle:vertical {
        height: 2px;
    }
    QSplitter::handle:pressed {
        background-color: yellow;
    }
    """
spin = """
    QSpinBox, QDoubleSpinBox {
        border: {borderWidth} solid {lighterGradient};
        color: white;
        font-size: {fontSize};
    }
    QSpinBox::hover, QDoubleSpinBox::hover {
        border: {borderWidth} solid {lightererGradient};
        color: white;
        font-size: {fontSize};
    }
    """
spinConnected = """
    QSpinBox, QDoubleSpinBox {
        background-color: {entryBGConnected};
        border: {borderConnected};
        color: white;
        font-size: {fontSize};
    }
    QSpinBox::hover, QDoubleSpinBox::hover {
        background-color: {entryBGConnectedHover};
        border: {borderConnectedHover};
        color: white;
        font-size: {fontSize};
    }
    QSpinBox::up-arrow, QSpinBox::down-arrow {
        background: {darkGradient};
    }
    """
spinValid = """
    QSpinBox, QDoubleSpinBox {
        background-color: {entryBGValid};
        border: {borderValid};
        color: white;
        font-size: {fontSize};
    }
    QSpinBox::hover, QDoubleSpinBox::hover {
        background-color: {entryBGValid};
        border: {borderValidHover};
        color: white;
        font-size: {fontSize};
    }
    """
statusbar = """
    QStatusBar {
        border: {borderWidth} {borderStyle} rgb(100,100,100);
        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
        font-weight: {fontWeight};
    }
    QStatusBar::hover {
        border: {borderWidth} {borderStyle} rgb(125,125,125);
        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} {borderColor};
        border-radius: {borderRadius};
    }
    """
statusbarAlert = """
    QStatusBar {
        background: rgba(75,75,25,0.5);
        border: {borderAlert};
        color: {fontColor};
        font-size: {fontSize};
        font-family: {font};
        font-weight: {fontWeight};
    }
    QStatusBar::hover {
        background: rgba(100,100,25,0.5);
        border: {borderAlertHover};
        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} yellow;
        border-radius: {borderRadius};
    }
    """
statusbarError = """
    QStatusBar {
        background: rgba(75,0,0,0.25);
        border: {borderError};
        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
        font-weight: {fontWeight};
    }
    QStatusBar::hover {
        background: rgba(75,0,0,0.25);
        border: {borderErrorHover};
        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} red;
        border-radius: {borderRadius};
    }
    """
statusbarSuccess = """
    QStatusBar {
        background: rgba(0,75,0,0.5);
        border: {borderValid};
        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
        font-weight: {fontWeight};
    }
    QStatusBar::hover {
        background: rgba(0,100,0,0.5);
        border: {borderValidHover};
        color: {fontColor};
        font-size: {fontSize};
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} green;
        border-radius: {borderRadius};
    }
    """
tab = """
    QTabWidget::pane { /* The tab widget frame */
    border-top: 2px solid {backgroundColorLight};
    }
    QTabWidget::tab-bar {
    left: 5px; /* move to the right by 5px */
    background: {backgroundColorLight};
    }
    /* Style the tab using the tab sub-control. Note that it reads QTabBar _not_ QTabWidget */
    QTabBar::tab {
    background: {backgroundColorLight};
    border: {borderWidth} solid {backgroundColorLight};
    border-bottom-color: {backgroundColorLight}; /* same as the pane color */
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    min-width: 8ex;
    padding: 2px;
    }
    QTabBar::tab:selected, QTabBar::tab:hover {
    background: {backgroundColorDark};
    }
    QTabBar::tab:selected {
    border-color: #9B9B9B;
    border-bottom-color: #C2C7CB; /* same as pane color */
    }
    QTabBar::tab:!selected {
    margin-top: 2px; /* make non-selected tabs look smaller */
    background: {backgroundColorLight};
    border-color: {backgroundColorLighter};
    }
    """
toolbar = """ 
    QToolBar, QToolButton, QToolTip { 
        background: rgb(56,60,55);
        background: {lightGradient};
        color: {fontColor};
        spacing: 3px; /* spacing between items in the tool bar */
        border: {borderWidth} {borderStyle} {borderColor};
    } 
    QToolButton {
        background-image: url(./interface7/image/background.png);
        font-size: {fontSize};
        font-family: {font};
    }
    QToolButton:hover {
        background: {darkGradient};
        border: 0px;
    }
    QToolBar::handle {
        background: {backgroundColorLight};
        border: {borderWidth} {borderStyle} rgb(100,100,100);
    } 
    """
widget = tooltip + """
         QWidget
         {
             color: {fontColor};
             background-color: {backgroundColorDark};
             selection-background-color: {backgroundColorLight};
             selection-color: {fontColor};
             background-clip: border;
             border-image: none;
             outline: 0;
             font-family: {font};
         }
         """
window = """
    QMainWindow {
        background: {backgroundColorDark};
        color: {fontColor};
        background-image: url(./interface7/image/background.png);
    }
    QMainWindow::separator {
        background: {backgroundColorDark};
        color: {backgroundColorLight};
        width: 10px; /* when vertical */
        height: 10px; /* when horizontal */
    }
    QMainWindow::separator:hover {
        background: {backgroundColorLight};
        color: black;
    }
    """

import re

scroll = """
    QScrollBar:vertical {
        background: rgb(50,50,50);         width: 5px;
     }
     QScrollBar:horizontal {
        background: rgb(50,50,50);         height: 5px;
     }
     QScrollBar::handle:vertical {
         background: rgb(200,200,200);
         min-height: 20px;
     }
    QScrollBar::handle:horizontal {
         background: rgb(200,200,200);
         min-height: 20px;
     }
     QScrollBar::handle:vertical:pressed {
         background: rgb(255,255,255);
         min-height: 20px;
     }
    QScrollBar::handle:horizontal:pressed {
         background: rgb(255,255,255);
         min-width: 20px;
     }
     QScrollBar::add-line:vertical {
        background: rgb(50,50,50);
     }
     QScrollBar::sub-line:vertical {
        background: rgb(50,50,50);
     }
     QScrollBar::add-line:horizontal {
        background: rgb(50,50,50);
     }
     QScrollBar::sub-line:horizontal {
         background: rgb(50,50,50);
     }
     QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
            background: none;
     }
    QScrollBar::up-arrow:vertical:pressed, QScrollBar::down-arrow:vertical:pressed {
     }
     QScrollBar::add-page, QScrollBar::sub-page {
         background: none;
     }
     """
button = """
    QPushButton {
        background: {lightererGradient};
        color: {fontColor};
        border: {borderWidth} {borderStyle} {borderColor};
        border-left-color: rgb(70,70,70);
        border-top-color: rgb(100,100,100);
        border-right-color: rgb(55,55,55);
        border-bottom-color: rgb(45,45,45);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton:hover {
        background: {darkGradient};
        border: {borderWidth} {borderStyle} orange;
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);
        border-bottom-color: rgb(160,60,10);
    }
    QPushButton:pressed {
        background: {orangeGradientPressed};
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);
        border-bottom-color: rgb(160,60,10);
        color: black;
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
buttonGlow = """
    QPushButton {
        background: {lightererGradient};
        color: {fontColor};
        border: {borderWidth} {borderStyle} orange;
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);
        border-bottom-color: rgb(160,60,10);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton:hover {
        background: {darkGradient};
        border: {borderWidth} {borderStyle} rgba(125,200,50);
        border-left-color: rgb(255,195,25);
        border-top-color: rgb(255,230,75);
        border-right-color: rgb(230,130,50);
        border-bottom-color: rgb(190,90,35);
    }
    QPushButton:pressed {
        background: {orangeGradientPressed};
        color: black;
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
buttonDisconnected = """
    QPushButton {
        background: {lightererGradient};
        color: {fontColor};
        border: {borderWidth} {borderStyle} rgba(100,25,25);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton:hover {
        background: {yellowGradient};
        border: {borderWidth} {borderStyle} rgba(125,200,50);
        color: black;
    }
    QPushButton:pressed {
        background: rgb(150,50,25);
        color: black;
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
buttonDisabled = """
    QPushButton {
        background: {darkGradient};
        color: {lightGradient};
        border: {borderWidth} {borderStyle} {darkGradient};
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        border-radius: {borderRadius};
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton:hover {
        background: {darkGradient};
        border: {borderWidth} {borderStyle} {darkGradient};
        color: rgba(50,20,20);
    }
    QPushButton:pressed {
        background: {darkGradient};
        border: {borderWidth} {borderStyle} {darkGradient};
        color: rgba(75,25,25);
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
buttonConnected = """
    QPushButton {
        background: {orangeGradient};
        color: {fontColor};
        border: {borderConnected};
        border-left-color: rgb(20,20,180);
        border-top-color: rgb(20,20,255);
        border-right-color: rgb(20,20,150);
        border-bottom-color: rgb(20,20,120);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton::hover {
        background: {yellowGradient};
        border: {borderConnectedHover};
        border-left-color: rgb(35,70,225);
        border-top-color: rgb(50,100,255);
        border-right-color: rgb(30,60,175);
        border-bottom-color: rgb(25,55,150);
        color: black;
    }
    QPushButton:pressed {
        background: {orangeGradientPressed};
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
buttonValid = """
    QPushButton {
        background: {lightererGradient};
        border: {borderValid};
        border-left-color: rgb(0,120,0);
        border-top-color: rgb(0,150,0);
        border-right-color: rgb(0,90,0);
        border-bottom-color: rgb(0,70,0);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: {fontColor};
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton:hover {
        background: {yellowGradient};
        border: {borderValidHover};
        border-left-color: rgb(0,150,0);
        border-top-color: rgb(0,200,0);
        border-right-color: rgb(0,120,0);
        border-bottom-color: rgb(0,90,0);
        color: darkblue;
    }
    QPushButton:pressed {
        background: {orangeGradientPressed};
        color: blue;
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
checkBox = """
    QCheckBox {
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
    }
    QCheckBox::indicator:checked {
        background: {orangeGradient};
        border: {borderValid};
    }
    QCheckBox::indicator:unchecked {
        background: black;
        border: {borderWidth} {borderStyle} {borderColor};
    }
    QCheckBox::indicator:unchecked:pressed {
        background: {yellowGradient};
    }
    QCheckBox::indicator:checked:pressed {
        background: {orangeGradientPressed};
    }
    QCheckBox::indicator:indeterminate:hover {
    }
    QCheckBox::indicator:indeterminate:pressed {
        background: {orangeGradientPressed};
    }
    QCheckBox::indicator:checked:hover {
        background: {orangeGradientPressed};
        border: {borderValidHover};
    }
    QCheckBox::indicator:unchecked:hover {
        background: black;
        border: {borderWidth} {borderStyle} orange;
    }
    """
checkboxChecked = """
    QCheckBox {
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        background: orange;
        border: 4px {borderStyle} {darkGreenGradient};
        border-left-color: rgb(0,120,0);
        border-top-color: rgb(0,150,0);
        border-right-color: rgb(0,90,0);
        border-bottom-color: rgb(0,70,0);
        border-top-left-radius: 4px;
        border-top-right-radius: 7px;
        border-bottom-right-radius: 4px;
        border-bottom-left-radius: 7px;
    }
    QCheckBox::indicator:hover {
        background: {yellowGradient};
        border: 4px {borderStyle} {lightGreenGradient};
        border-left-color: rgb(0,150,0);
        border-top-color: rgb(0,200,0);
        border-right-color: rgb(0,120,0);
        border-bottom-color: rgb(0,90,0);
    }
    QCheckBox::indicator:pressed {
        background: {orangeGradientPressed};
        border: 4px {borderStyle} {darkGreenGradient};
        border-left-color: rgb(0,120,0);
        border-top-color: rgb(0,150,0);
        border-right-color: rgb(0,90,0);
        border-bottom-color: rgb(0,70,0);
    }
    """
checkboxUnchecked = """
    QCheckBox {
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        background: black;
        border: 4px {borderStyle} {lightGradient};
        border-left-color: rgb(70,70,70);
        border-top-color: rgb(100,100,100);
        border-right-color: rgb(55,55,55);
        border-bottom-color: rgb(45,45,45);
        border-top-left-radius: 4px;
        border-top-right-radius: 7px;
        border-bottom-right-radius: 4px;
        border-bottom-left-radius: 7px;
    }
    QCheckBox::indicator:hover {
        background: black;
        border: 4px {borderStyle} {darkGradient};
        border-left-color: rgb(255,195,25);
        border-top-color: rgb(255,230,75);
        border-right-color: rgb(230,130,50);
        border-bottom-color: rgb(190,90,35);
    }
    QCheckBox::indicator:pressed {
        background: {orangeGradientPressed};
    }
    """
dropdown = """
    QComboBox QAbstractItemView {
        border: {borderWidth} {borderStyle} {backgroundColorLight};
        background-clip: border;
        border-left-color: rgb(255,195,25);
        border-top-color: rgb(255,230,75);
        border-right-color: rgb(230,130,50);
        border-bottom-color: rgb(190,90,35);
        border-top-left-radius: 10px;
        border-top-right-radius: 5px;
        border-bottom-right-radius: 10px;
        border-bottom-left-radius: 5px;
        selection-background-color: {backgroundColorLighter};
        background: {backgroundColorDark};
        font-size: {fontSize};
        color: rgb(200,200,200);
    }
    QComboBox {
        background-clip: border;
        background: {lightGradient};
        color: {contrastColor};
        border: 3px {borderStyle} {borderColor};
        border-left-color: rgb(70,70,70);
        border-top-color: rgb(100,100,100);
        border-right-color: rgb(55,55,55);
        border-bottom-color: rgb(45,45,45);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        padding: 1px 1px 1px 3px;
        min-width: 6px;
        font-size: {fontSize};
    } 
    QComboBox:!editable::hover {
        background: {darkGradient}; 
        background-clip: border;
        border: {borderWidth} {borderStyle} orange;
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);
        border-bottom-color: rgb(160,60,10);
    }
    QComboBox:editable {
        background: {darkGradient};    
        background-clip: margin;   
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;  
    }
    /* QComboBox gets the "on" state when the popup is open */
    QComboBox:!editable:on, QComboBox::drop-down:editable:on {
        background: {lighterGradient};
        background-clip: border;
        border-left-color: rgb(255,195,25);
        border-top-color: rgb(255,230,75);
        border-right-color: rgb(230,130,50);
        border-bottom-color: rgb(190,90,35);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: rgb(150,150,150);
    }
    QComboBox:on { /* shift the text when the popup opens */
        padding-top: 3px;
        padding-left: 4px;
        background: {lightGradient};
        background-clip: border;
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
    }
    QComboBox::drop-down {
        background-clip: border;
        subcontrol-origin: border;
        subcontrol-position: top right;
    }
    QComboBox::drop-down:hover {
        background-clip: border;
        subcontrol-origin: border;
        subcontrol-position: top right;
    }
    QComboBox::down-arrow {
    }
    QComboBox::down-arrow:on { /* shift the arrow when popup is open */
    }
    """
dropdownDisabled = """
    QComboBox {
        background-clip: border;
        background: {lightGradient};
        color: {contrastColor};
        border: {borderWidth} {borderStyle} {darkGradient};
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        padding: 1px 1px 1px 3px;
        min-width: 6px;
        font-size: {fontSize};
    } 
    QComboBox:editable {
        background: {darkGradient};    
        background-clip: margin;   
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;  
    }
    /* QComboBox gets the "on" state when the popup is open */
    QComboBox:!editable:on, QComboBox::drop-down:editable:on {
        background: {lighterGradient};
        background-clip: border;
        border-left-color: rgb(255,195,25);
        border-top-color: rgb(255,230,75);
        border-right-color: rgb(230,130,50);
        border-bottom-color: rgb(190,90,35);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: rgb(150,150,150);
    }
    QComboBox:on { /* shift the text when the popup opens */
        padding-top: 3px;
        padding-left: 4px;
        background: {lightGradient};
        background-clip: border;
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
    }
    QComboBox::drop-down {
        background: rgba(0,0,0,0);
        background-clip: border;
        subcontrol-origin: border;
        subcontrol-position: top right;
    }
    QComboBox::down-arrow {
        background: rgba(0,0,0,0);
    }
    QComboBox::down-arrow:on { /* shift the arrow when popup is open */
    }
    """
dial = """CustomDial {
        background-color: #27272B;
        color: #FFFFFF;
        qproperty-knobRadius: 5;
        qproperty-knobMargin: 5;
    }
    """
entry = scroll + """
    QWidget
    {
        background-color: {labelColor};
        border: {borderWidth} {borderStyle} {borderColor};
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
        font-family: {font};
    }
    QLineEdit, QPlainTextEdit
    {
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        font-family: {font};
        font-size: {fontSize};
    }
    """
entryGlow = scroll + """
    QLineEdit::hover, QPlainTextEdit::hover
    {
        border: {borderWidth} {borderStyle} yellow;
        border-left-color: rgb(255,195,25);
        border-top-color: rgb(255,230,75);
        border-right-color: rgb(230,130,50);
        border-bottom-color: rgb(190,90,35);
        font-family: {font};
        font-size: {fontSize};
    }
    QLineEdit, QPlainTextEdit
    {
        background-color: {labelColor};
        border: {borderWidth} {borderStyle} orange;
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);
        border-bottom-color: rgb(160,60,10);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        font-family: {font};
        font-size: {fontSize}
        margin: 0px;
        padding: 0px;
    }
    """
entryConnected = scroll + """
    QLineEdit::hover, QPlainTextEdit::hover
    {
        background: rgba(0,0,10,0.5);
        /*border: {borderConnectedHover};
        border-left-color: rgb(35,70,225);
        border-top-color: rgb(50,100,255);
        border-right-color: rgb(30,60,175);*/
        border-bottom: {borderWidth} solid rgb(25,55,150);
        color: white;
        font-family: {font};
        font-size: {fontSize};
    }
    QLineEdit, QPlainTextEdit
    {
        background: rgba(0,0,10,0.5);
        /*border: {borderConnected};
        border-left-color: rgb(20,20,180);
        border-top-color: rgb(20,20,255);
        border-right-color: rgb(20,20,150);*/
        border-bottom: {borderWidth} solid rgb(20,20,120);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: white;
        font-family: {font};
        font-size: {fontSize};
        margin: 0;     
        padding: 0;
    }
    """
entryDisabled = scroll + """
    QLineEdit::hover, QPlainTextEdit::hover
    {
        background: rgba(40,20,20,1);
        border: {borderWidth} {borderStyle} {darkGradient};
        color: rgba(50,25,25,1);
        font-family: {font};
        font-size: {fontSize};
        margin: 0px;
        padding: 0px;
    }
    QLineEdit, QPlainTextEdit
    {
        background: rgba(40,25,25,1);
        border: {borderWidth} {borderStyle} {darkGradient};
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: rgba(50,25,25,1);
        font-family: {font};
        font-size: {fontSize};
    }
    """
entryDisconnected = scroll + """
    QLineEdit::hover, QPlainTextEdit::hover
    {
        background: rgba(20,10,10,1);
        /*border: {borderDisconnectedHover};*/
        border-bottom: {borderWidth} solid rgba(75,45,45,1);
        color: rgba(50,25,25,1);
        font-family: {font};
        font-size: {fontSize};
        margin: 0px;
        padding: 0px;
    }
    QLineEdit, QPlainTextEdit
    {
        background: rgba(35,20,20,1);
        /*border: {borderDisconnected};*/
        border-bottom: {borderWidth} solid rgba(50,25,25,1);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: rgba(50,25,25,1);
        font-family: {font};
        font-size: {fontSize};
    }
    """
entryValid = scroll + """
    QLineEdit::hover, QPlainTextEdit::hover
    {
        background: rgba(0,20,0,0.5);
        /*border: {borderValidHover};
        border-left-color: rgb(0,150,0);
        border-top-color: rgb(0,200,0);
        border-right-color: rgb(0,120,0);*/
        border-bottom: {borderWidth} solid rgb(0,90,0);
        font-family: {font};
        font-size: {fontSize};
    }
    QLineEdit, QPlainTextEdit
    {
        background: rgba(0,10,0,0.5);
        /*border: {borderValid};
        border-left-color: rgb(0,120,0);
        border-top-color: rgb(0,150,0);
        border-right-color: rgb(0,90,0);*/
        border-bottom: {borderWidth} solid rgb(0,70,0);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: white;
        font-family: {font};
        font-size: {fontSize};
        margin: 0px;
        padding: 0px;
    }
    """
entryError = scroll + """
    QLineEdit::hover, QPlainTextEdit::hover
    {
        background: rgba(20,0,0,0.5);
        /*border: {borderErrorHover};
        border-left-color: rgb(175,0,0);
        border-top-color: rgb(225,20,20);
        border-right-color: rgb(125,0,0);*/
        border-bottom: {borderWidth} solid rgb(100,0,0);
        font-family: {font};
        font-size: {fontSize};
    }
    QLineEdit, QPlainTextEdit
    {
        background: rgba(10,0,0,0.5);
        /*border: {borderError};
        border-left-color: rgb(125,0,0);
        border-top-color: rgb(175,20,20);
        border-right-color: rgb(100,0,0);*/
        border-bottom: {borderWidth} solid rgb(75,0,0);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        font-family: {font};
        font-size: {fontSize};
        padding: 0px;
        margin: 0px;
    }
    """
groupBox = """
    QGroupBox {
        font-size:{fontSize};
        background: rgba(0,0,0,0.25);
        /*border: {borderWidth} {borderStyle} rgba(50,50,50,1);
        border-left-color: rgb(70,70,70);
        border-top-color: rgb(100,100,100);
        border-right-color: rgb(55,55,55);*/
        border-bottom: {borderWidth} solid rgb(45,45,45);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        margin-bottom: 0ex;
        margin-top: 3.6ex; /* leave space at the top for the title */
        padding: 0;
        margin: 0;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center; /* position at the top center */
        padding: 0 3px 0 0;   
        margin: 0;          
    }
    """
groupBoxNu = """
    QGroupBox {
        /*border: {borderWidth} {borderStyle} rgba(100,100,100,0.05);
        border-left-color: rgb(70,70,70);
        border-top-color: rgb(100,100,100);
        border-right-color: rgb(55,55,55);
        border-bottom: 2px solid rgb(45,45,45);*/
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        margin-top: 3ex; /* leave space at the top for the title */
        background: rgba(0,0,0,0.2);
        font-size: {fontSize};
        padding: 0;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left; /* position at the top center */
        padding: 0 3px 0 0;   
        margin: 0;     
    }
"""
list = scroll + """
    QListView {
        background: rgba(10,7,7,0.1);
        /*border: {borderWidth} {borderStyle} {borderColor};
        border-left-color: rgb(70,70,70);
        border-top-color: rgb(100,100,100);
        border-right-color: rgb(55,55,55);*/
        border-bottom: {borderWidth} solid rgb(45,45,45);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        selection-color: {contrastColor};
        selection-background-color: {contrastColor};
        font-size:{fontSize};
    }
    QListView:item{
        background: rgb(20,20,20);
        background-image: url(./interface/img/dark-leather.png);
        border: 1px outset rgb(30,30,30);
    }
    QListView:item:selected {
        color: {backgroundColorDark};
        background: {contrastColor};
    }
    QListView:item:hover {
        background: {backgroundColorLighter};
    }
    QListView:item:selected:hover {
        background: {contrastColor};
    }

    """
tooltip = """
    QToolBar, QToolButton, QToolTip
    {
        background: rgb(56,60,55);
        background: {lightGradient};
        color: {fontColor};
        spacing: 3px; /* spacing between items in the tool bar */
        border: 1px {borderStyle} {borderColor};
    }
    """
frame = """
    QFrame {
        background: {backgroundColorDark};
        background-image: url(./interface/img/background.png);
    }
    """
label = tooltip + """
        QLabel {
            background: {labelColor};
            color: {fontColor};
            border: 1px {borderStyle} {borderColor};
            border-top-left-radius: 5px;
            border-top-right-radius: 10px;
            border-bottom-right-radius: 5px;
            border-bottom-left-radius: 10px;
            padding: 2px;
            font-size: {fontSize};
        }
        """
menubar = """
    QMenuBar {
        background: {backgroundColorLightererer};
        color: {fontColor};
        border: 1px solid rgb(0,0,0);
    }
    QMenuBar::item {
        background: {backgroundColorLighter};
        color: {fontColor};
    }
    QMenuBar::item::hover {
        background-color: {backgroundColorDark};
    }
    QMenu {
        background: {backgroundColorLight};
        color: {fontColor};
        border: 1px solid #000;           
    }
    QMenu::item::selected {
        background-color: {backgroundColorLight};
    }
    """
radio = """
    QRadio:default {
    }
    """
slider = """
    QSlider::handle {
        background: rgb(255,100,50);
        border: {borderWidth} {borderStyle} {lightGradient};
        border-left-color: rgb(70,70,70);
        border-top-color: rgb(100,100,100);
        border-right-color: rgb(55,55,55);
        border-bottom-color: rgb(45,45,45);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
    }
    QSlider::handle:hover {
        background: {yellowGradient};
    }
    QSlider::handle:pressed {
        background: rgb(150,50,25);
    }
    QSlider::sub-page {
    }
    QSlider::add-page {
    }
    QSlider::groove {
    }
    QSlider {
        border: {borderWidth} {borderStyle} {darkGradient};
        border-left-color: rgb(70,70,70);
        border-top-color: rgb(100,100,100);
        border-right-color: rgb(55,55,55);
        border-bottom-color: rgb(45,45,45);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
    }
    QSlider:hover {
        border: {borderWidth} {borderStyle} {darkGradient};
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);
        border-bottom-color: rgb(160,60,10);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
    }
    QSlider::sub-page:hover {       
    }
    QSlider::add-page:hover {
    }

    """
splitter = """
    QSplitter {
        background: {backgroundColorDark};
    }
    QSplitter::handle {
        background: orange;
        color: {fontColor};
    }
    QSplitter::handle:hover {
        background: white;
    }
    QSplitter::handle:horizontal {
        width: 4px;
    }
    QSplitter::handle:vertical {
        height: 2px;
    }
    QSplitter::handle:pressed {
        background-color: {yellowGradient};
    }
    """
spin = """
    QSpinBox, QDoubleSpinBox {
        background-color: {lightGradient};
        /*border: {borderWidth} {borderStyle} {lighterGradient};
        border-left-color: rgb(70,70,70);
        border-top-color: rgb(100,100,100);
        border-right-color: rgb(55,55,55);*/
        border-bottom: {borderWidth} solid rgb(45,45,45);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: white;
        font-size: {fontSize};
    }
    QSpinBox::hover, QDoubleSpinBox::hover {
        /*border: {borderWidth} {borderStyle} {lightererGradient};
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);*/
        border-bottom: {borderWidth} solid rgb(160,60,10);
        color: white;
        font-size: {fontSize};
    }
    QSpinBox::up-arrow:pressed, QSpinBox::down-arrow:pressed {
    }
    QSpinBox::up-arrow, QSpinBox::down-arrow {
    }
    QSpinBox::up-button, QSpinBox::down-button {
    }
    QSpinBox::down-arrow:disabled, QSpinBox::down-arrow:off, QSpinBox::up-arrow:disabled, QSpinBox::up-arrow:off { /* off state when value is max */
        background-color: red;
    }
    """
spinConnected = """
    QSpinBox, QDoubleSpinBox {
        background: rgba(0,0,10,0.5);
        /*border: {borderConnected};
        border-left-color: rgb(20,20,180);
        border-top-color: rgb(20,20,255);
        border-right-color: rgb(20,20,150);*/
        border-bottom: {borderWidth} solid rgb(20,20,120);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: white;
        font-size: {fontSize};
    }
    QSpinBox::hover, QDoubleSpinBox::hover {
        background: rgba(0,0,20,0.5);
        /*border: {borderConnectedHover};
        border-left-color: rgb(35,70,225);
        border-top-color: rgb(50,100,255);
        border-right-color: rgb(30,60,175);*/
        border-bottom-color: 4px solid rgb(25,55,150);
        color: white;
        font-size: {fontSize};
    }
    QSpinBox::up-arrow, QSpinBox::down-arrow {
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
    }
    """
spinValid = """
    QSpinBox, QDoubleSpinBox {
        background-color: {entryBGValid};
        border: {borderValid};
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: white;
        font-size: {fontSize};
    }
    QSpinBox::hover, QDoubleSpinBox::hover {
        background-color: {entryBGValid};
        border: {borderValidHover};
        color: white;
        font-size: {fontSize};
    }
    QSpinBox::up-arrow, QSpinBox::down-arrow {
    }
    """
statusbar = """
    QStatusBar {
        /*border: {borderWidth} {borderStyle} rgb(100,100,100);
        border-left-color: rgb(70,70,70);
        border-top-color: rgb(100,100,100);
        border-right-color: rgb(55,55,55);*/
        border-bottom: {borderWidth} solid rgb(45,45,45);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
        font-weight: {fontWeight};
    }
    QStatusBar::hover {
        /*border: {borderWidth} {borderStyle} rgb(125,125,125);
        border-left-color: rgb(90,90,90);
        border-top-color: rgb(140,140,140);
        border-right-color: rgb(65,65,65);*/
        border-bottom: {borderWidth} solid rgb(55,55,55);
        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} {borderColor};
        border-radius: {borderRadius};
    }
    """
statusbarAlert = """
    QStatusBar {
        background: rgba(25,25,0,0.5);
        /*border: {borderAlert};
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);*/
        border-bottom: {borderWidth} solid rgb(160,60,10);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: rgb(255,255,150);
        font-size: {fontSize};
        font-family: {font};
        font-weight: {fontWeight};
    }
    QStatusBar::hover {
        background: rgba(40,40,0,0.5);
        /*border: {borderAlertHover};
        border-left-color: rgb(255,195,25);
        border-top-color: rgb(255,230,75);
        border-right-color: rgb(230,130,50);*/
        border-bottom: {borderWidth} solid rgb(190,90,35);
        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} yellow;
        border-radius: {borderRadius};
    }
    """
statusbarError = """
    QStatusBar {
        background: rgba(75,0,0,0.25);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        /*border: {borderError};
        border-left-color: rgb(125,0,0);
        border-top-color: rgb(175,20,20);
        border-right-color: rgb(100,0,0);*/
        border-bottom: {borderWidth} solid rgb(80,0,0);

        color: rgb(255,5,50);
        font-family: {font};
        font-size: {fontSize};
        font-weight: {fontWeight};
    }
    QStatusBar::hover {
        background: rgba(75,0,0,0.25);
        /*border: {borderErrorHover};
        border-left-color: rgb(175,0,0);
        border-top-color: rgb(225,20,20);
        border-right-color: rgb(125,0,0);*/
        border-bottom: {borderWidth} solid rgb(100,0,0);

        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} red;
        border-radius: {borderRadius};
    }
    """
statusbarSuccess = """
    QStatusBar {
        background: rgba(0,10,0,0.5);
        /*border: {borderValid};
        border-left-color: rgb(0,120,0);
        border-top-color: rgb(0,150,0);
        border-right-color: rgb(0,90,0);*/
        border-bottom: {borderWidth} solid rgb(0,70,0);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: rgb(100,255,100);
        font-family: {font};
        font-size: {fontSize};
        font-weight: {fontWeight};
    }
    QStatusBar::hover {
        background: rgba(0,20,0,0.5);
        /*border: {borderValidHover};
        border-left-color: rgb(0,150,0);
        border-top-color: rgb(0,200,0);
        border-right-color: rgb(0,120,0);*/
        border-bottom: {borderWidth} solid rgb(0,90,0);
        color: {fontColor};
        font-size: {fontSize};
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} green;
        border-radius: {borderRadius};
    }
    """
tab = """
    QTabWidget{
        font-size: 8pt;
    }
    QTabWidget::pane { /* The tab widget frame */
        border-top: 2px solid {backgroundColorLight};
    }
    QTabWidget::tab-bar {
        left: 5px; /* move to the right by 5px */
        background: {backgroundColorLight};
    }
    /* Style the tab using the tab sub-control. Note that it reads QTabBar _not_ QTabWidget */
    QTabBar::tab {
        background: {backgroundColorLight};
        border: {borderWidth} {borderStyle} {backgroundColorLight};
        border-bottom-color: {backgroundColorLight}; /* same as the pane color */
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        min-width: 14ex;
        padding: 2px;
    }
    QTabBar::tab:!selected:hover {
        background: {backgroundColorDark};
        border-left-color: rgb(150,150,125);
        border-top-color: rgb(225,225,200);
        border-right-color: rgb(125,125,100);
        border-bottom-color: rgb(100,100,75);
    }
    QTabBar::tab:selected {
        background: rgb(25,25,75);
        border: 5px {borderStyle} orange;
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);
        border-bottom-color: rgb(160,60,10);
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    QTabBar::tab:selected:hover {
        background: rgb(0,0,75);
        border: 5px {borderStyle} orange;
        border-left-color: rgb(255,195,25);
        border-top-color: rgb(255,230,75);
        border-right-color: rgb(230,130,50);
        border-bottom-color: rgb(190,90,35);
    }
    QTabBar::tab:!selected {
        margin-top: 2px; /* make non-selected tabs look smaller */
        background: {borderWidth} {borderStyle} rgb(10,10,10);
        border-left-color: rgb(100,100,100);
        border-top-color: rgb(175,175,175);
        border-right-color: rgb(75,75,75);
        border-bottom-color: rgb(50,50,50);
    }
    """
table = scroll + """
    QTableCornerButton::section, QHeaderView::section{
        background: {lightGradient};
        background-image: url(./interface/img/dark-leather.png);
        border: 2px outset rgb(40,40,40);
    }
    QTableView {
        border-bottom: {borderWidth} solid rgb(45,45,45);
        gridline-color: rgb(30,30,30);
        color: white;
        background: rgba(10,7,7,0.1);
    }
    QTableView::item{
        background: rgb(10,10,10);
        background-image: url(./interface/img/dark-leather.png);
        border-bottom: 1px solid rgb(60,60,60);
    }
    QTableView::item:selected {
        background: rgb(150,150,150);
        gridline-color: black;
        border: 1px outset white;
        color: black;
    }
    """
toolbar = """ 
    QToolBar, QToolButton, QToolTip { 
        background: rgb(56,60,55);
        background: {lightGradient};
        color: {fontColor};
        spacing: 3px; /* spacing between items in the tool bar */
        border: {borderWidth} {borderStyle} {borderColor};
    } 
    QToolButton {
        background-image: url(./interface/img/background.png);
        background: {orangeGradient};
        border: 8px solid white;
        border-left-color: rgb(70,70,70);
        border-top-color: rgb(100,100,100);
        border-right-color: rgb(55,55,55);
        border-bottom-color: rgb(45,45,45);
        color: rgb(0,0,50);
        height: 72px;
        width: 72px;
        font-size: 8pt;
        font-family: {font};
        font-weight: bold;
    }
    QToolButton:hover {
        background: {yellowGradient};
        border-left-color: rgb(255,195,25);
        border-top-color: rgb(255,230,75);
        border-right-color: rgb(230,130,50);
        border-bottom-color: rgb(190,90,35);
        margin: -2px;
    }
    QToolButton:pressed {
        background: {orangeGradientPressed};
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);
        border-bottom-color: rgb(160,60,10);
        margin: 2px;
    }
    QToolBar::handle {
        subcontrol-position: top;
        background: {backgroundColorLight};
        border: {borderWidth} {borderStyle} rgb(100,100,100);
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);
        border-bottom-color: rgb(160,60,10);
    } 
    """
widget = tooltip + """
         QWidget
         {
             color: {fontColor};
             background-clip: border;
             background-color: rgba(255,255,255,0.1);
             border-top-left-radius: 5px;
             border-top-right-radius: 10px;
             border-bottom-right-radius: 5px;
             border-bottom-left-radius: 10px;
             selection-background-color: {backgroundColorLight};
             selection-color: {fontColor};
             outline: 0;
             font-family: {font};
             margin: 0;
             padding: 0;
             background-image: url(./interface/img/dark-leather.png);
         }
         """
window = """
    QMainWindow {
        background: {backgroundColorDark};
        background-clip: border;
        color: {fontColor};
        background-image: url(./interface/img/background.png);
         margin: 0;
         padding: 0;
    }
    QMainWindow::separator {
        background: {backgroundColorDark};
        color: {backgroundColorLight};
        width: 10px; /* when vertical */
        height: 10px; /* when horizontal */
    }
    QMainWindow::separator:hover {
        background: {backgroundColorLight};
        color: black;
    }
    """
minButton = """
    /*QPushButton {
        background: rgb(35,35,100);
        border: 1px solid rgb(100,100,255);
        min-width: 25px;
    }
    QPushButton::hover {
        background: rgb(35,35,150);
        border: 1px solid rgb(125,125,255);
    }
    QPushButton::pressed {
        background: rgb(15,15,80);
        border: 1px solid rgb(75,75,200);
    }*/
    QPushButton {
        background: none;
        border-top-left-radius: 5px;
        border-top-right-radius: 3px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 3px;
        /*border: 1px solid rgb(50,50,255);*/
        min-width: 25px;
        font-size: 6pt;
    }
    QPushButton::hover {
        background: rgb(35,35,50,0.1);
        border-bottom: 1px solid rgb(75,75,255);
    }
    QPushButton::pressed {
        background: rgb(30,30,45);
        border-top: 2px solid rgba(0,0,0,0);
        border-bottom: 1px solid rgb(25,25,200);
    }
    """
xButton = """
    /*QPushButton {
        background: rgb(100,35,35);
        border: 1px solid rgb(255,100,100);
        min-width: 25px;
    }
    QPushButton::hover {
        background: rgb(150,35,35);
        border: 1px solid rgba(255,125,125,0.1);
    }
    QPushButton::pressed {
        background: rgb(80,15,15);
        border: 1px solid rgb(200,75,75);
    }*/
    QPushButton {
        background: none;
        border-top-left-radius: 5px;
        border-top-right-radius: 3px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 3px;
        /*border: 1px solid rgb(255,50,50);*/
        min-width: 25px;
        height:5px;
        font-size: 6pt;
    }
    QPushButton::hover {
        background: rgba(50,35,35,0.5);
        border-bottom: 1px solid rgb(255,75,75);
    }
    QPushButton::pressed {
        background: rgb(45,30,30);
        border-top: 2px solid rgba(0,0,0,0);
        border-bottom: 1px solid rgb(200,25,25);
    }
    """
titleBar = """
    QLabel {
        background: {backgroundColorDark};
        background-image: url(./interface/img/background.png);
        border-top: 1px solid rgba(50,55,52,1);
        border-bottom: 1px outset rgb(125,50,0);
        border-top-left-radius: 0px;
        border-top-right-radius: 0px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 5px;
        font-size: 8pt;
        color: {fontColor};
    }

    """


def format_qss(s, palette):
    try:
        matches = re.findall('{.+?}', s)
        result = '' + s
        for m in matches:
            result = result.replace(m, getattr(palette, m[1:-1]))
        return result
    except Exception as e:
        alertMsg = 'QSS formatting error: {}'.format(str(e))
        if str(e) == 'expected string or bytes-like object':
            alertMsg = '{} but got {} type object {}'.format(alertMsg, type(s), s)
        palette.parent.update_status(alertMsg, 'error')


def format_qss(s, palette):
    try:
        matches = re.findall('{.+?}', s)
        result = '' + s
        for m in matches:
            result = result.replace(m, getattr(palette, m[1:-1]))
        return result
    except Exception as e:
        if str(e) == 'expected string or bytes-like object':
            print('QSS formatting error: {} but got {} type object {}'.format(str(e), type(s), s))
        else:
            print('QSS formatting error: ' + str(e))


from Utilities19.Colors import *

BORDER_COLOR = r'rgb(70,70,70)'
BUTTON_COLOR = r'rgb(55,55,55)'
BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'

GROUP_BOX_STYLE = """
                QGroupBox {

                }
            """
WINDOW_STYLE = """
            QMainWindow {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);

            }
            QMainWindow::separator {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + BACKGROUND_COLOR_LIGHT + """;
                width: 10px; /* when vertical */
                height: 10px; /* when horizontal */
            }

            QMainWindow::separator:hover {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: black;
            }

            """
BUTTON_STYLE = """
            QPushButton {
                background: """ + LIGHT_GRADIENT + """;
                color: """ + FONT_COLOR + """;
                border: """ + BORDER_WIDTH + """ """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;
                min-width: 80px;
            }

            QPushButton:hover {
                background: """ + DARK_GRADIENT + """;
                border: """ + BORDER_WIDTH + """ """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            }

            QPushButton:pressed {
                background: """ + DARK_GRADIENT + """;
            }

            QPushButton:flat {
                background: """ + LIGHT_GRADIENT + """;
                border: """ + BORDER_COLOR + """;; /* no border for a flat push button */
            }

            QPushButton:default {
                background: """ + LIGHT_GRADIENT + """;
                border: """ + BORDER_WIDTH + """ """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
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
            QComboBox QAbstractItemView {
                border: 2px solid """ + BACKGROUND_COLOR_LIGHT + """;
                selection-background-color: """ + BACKGROUND_COLOR_LIGHTER + """;
                background: """ + BACKGROUND_COLOR_DARK + """;
            }

            QComboBox {
                background: """ + LIGHT_GRADIENT + """;
                color: """ + CONTRASTED_COLOR + """;
                border: 3px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                padding: 1px 1px 1px 3px;
                min-width: 6px;
            }

            QComboBox:!editable::hover {
                background: """ + DARK_GRADIENT + """; 
            }

            QComboBox:editable {
                background: """ + DARK_GRADIENT + """;         
            }

            /* QComboBox gets the "on" state when the popup is open */
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: """ + DARK_GRADIENT + """;
            }

            QComboBox:on { /* shift the text when the popup opens */
                padding-top: 3px;
                padding-left: 4px;
                background: """ + LIGHT_GRADIENT + """;
            }

            QComboBox::drop-down {
                background: """ + LIGHT_GRADIENT + """;
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 10px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: """ + BORDER_STYLE + """;
                border-top-right-radius: """ + BORDER_RADIUS + """;
                border-bottom-right-radius: """ + BORDER_RADIUS + """;
            }

            QComboBox::down-arrow {

            }

            QComboBox::down-arrow:on { /* shift the arrow when popup is open */
                top: 1px;
                left: 1px;
            }
            """

DIAL_STYLE = """CustomDial {
                background-color: #27272B;
                color: #FFFFFF;
                qproperty-knobRadius: 5;
                qproperty-knobMargin: 5;
            }
            """
FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """
MENUBAR_STYLE = """
            QMenuBar {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid rgb(0,0,0);
            }

            QMenuBar::item {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
            }

            QMenuBar::item::selected {
                background: """ + BACKGROUND_COLOR_DARK + """;
            }

            QMenu {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid #000;           
            }

            QMenu::item::selected {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
            }
            """

SCROLL_STYLE = """
            QScrollBar:vertical {
                 background: """ + BACKGROUND_COLOR_DARK + """;
                 width: 15px;
                 margin: 22px 0 22px 0;
             }
             QScrollBar::handle:vertical {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::handle:vertical:pressed {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::add-line:vertical {
                 background: """ + LIGHT_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: bottom;
                 subcontrol-origin: margin;
             }

             QScrollBar::sub-line:vertical {
                 background: """ + DARK_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: top;
                 subcontrol-origin: margin;
             }
             QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                 border: 2px solid """ + DARK_GRADIENT + """;
                 width: 3px;
                 height: 3px;
                 background: black;
             }
            QScrollBar::up-arrow:vertical:pressed, QScrollBar::down-arrow:vertical:pressed {
                 border: 2px solid """ + LIGHT_GRADIENT + """;
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
                background: """ + BACKGROUND_COLOR_DARK + """;
            }
            QSplitter::handle {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
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
                background-color: """ + DARK_GRADIENT + """;
            }
            """

STATUSBAR_STYLE = """
            QStatusBar {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid rgb(100,100,100);
            }

            QStatusBar::item {
                border: """ + BORDER_WIDTH + """ """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;

            }"""

TOOLBAR_STYLE = """ 
            QToolBar, QToolButton, QToolTip { 
                background: rgb(56,60,55);
                background: """ + LIGHT_GRADIENT + """;

                color: """ + FONT_COLOR + """;
                spacing: 3px; /* spacing between items in the tool bar */
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;

            } 
            QToolBar {
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }


            QToolButton:hover {
                background: """ + DARK_GRADIENT + """;
                border: 0px;
            }

            QToolBar::handle {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                border: 1px solid rgb(100,100,100);
            } 
            """

BACKGROUND_COLOR_DARK = r'rgb(25,25,25)'
BACKGROUND_COLOR_LIGHT = r'rgb(49,49,49)'
BORDER_COLOR = r'rgb(70,70,70)'
FONT_COLOR = r'rgb(255,255,255)'
BUTTON_COLOR = r'rgb(55,55,55)'
BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'
LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(32,33,30), stop: 0.5 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(22,23,20), stop: 0.5 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'
ENTRY_STYLE = """
            QWidget
            {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
                border: 2px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
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
                background: """ + LIGHT_GRADIENT + """;
                color: rgb(255,255,255);
                border: 3px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                padding: 1px 1px 1px 3px;
                min-width: 6px;
            }

            QComboBox:!editable::hover {
                background: """ + DARK_GRADIENT + """; 
            }

            QComboBox:editable {
                background: """ + DARK_GRADIENT + """;         
            }

            /* QComboBox gets the "on" state when the popup is open */
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: """ + DARK_GRADIENT + """;
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
                border-left-style: """ + BORDER_STYLE + """;
                border-top-right-radius: """ + BORDER_RADIUS + """;
                border-bottom-right-radius: """ + BORDER_RADIUS + """;
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
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """
LABEL_STYLE = TOOLTIP_STYLE + """
            QLabel {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;
                padding: 2px;
            }
            """

from Utilities.Colors import *

BORDER_COLOR = r'rgb(70,70,70)'
BUTTON_COLOR = r'rgb(55,55,55)'
BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'

GROUP_BOX_STYLE = """
                QGroupBox {
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

DIAL_STYLE = """CustomDial {
                background-color: #27272B;
                color: #FFFFFF;
                qproperty-knobRadius: 5px;
                qproperty-knobMargin: 5px;
            }
            """
FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """
PROGRESS_BAR_STYLE = """
            QProgressBar:horizontal {
            background: """ + BACKGROUND_COLOR_DARK + """;
            border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            border-radius: """ + BORDER_RADIUS + """;
            width: 100px;
            height: 5px;
            text-align: right;
            margin-right: 10ex;

            }
            QProgressBar::chunk:horizontal {
            background: """ + CONTRASTED_COLOR + """;
            border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            border-radius: """ + BORDER_RADIUS + """;
            }
            """

BACKGROUND_COLOR_DARK = r'rgb(25,25,25)'
BACKGROUND_COLOR_LIGHT = r'rgb(49,49,49)'
BACKGROUND_COLOR_LIGHTER = r'rgb(79,79,79)'

BORDER_COLOR = r'rgb(70,70,70)'
FONT_COLOR = r'rgb(255,255,255)'
BUTTON_COLOR = r'rgb(55,55,55)'
BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'
LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(32,33,30), stop: 0.5 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(22,23,20), stop: 0.5 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'
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
DIAL_STYLE = """CustomDial {
                background-color: #27272B;
                color: #FFFFFF;
                qproperty-knobRadius: 5;
                qproperty-knobMargin: 5;
            }
            """
FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """

from Utilities.Colors import *

BORDER_COLOR = r'rgb(70,70,70)'
BUTTON_COLOR = r'rgb(55,55,55)'
BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'

GROUP_BOX_STYLE = """
                QGroupBox {
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

DIAL_STYLE = """CustomDial {
                background-color: #27272B;
                color: #FFFFFF;
                qproperty-knobRadius: 5;
                qproperty-knobMargin: 5;
            }
            """
FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """

from Utilities23.Colors import *

BORDER_COLOR = r'rgb(70,70,70)'
BUTTON_COLOR = r'rgb(55,55,55)'
BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'

GROUP_BOX_STYLE = """
                QGroupBox {
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

DIAL_STYLE = """CustomDial {
                background-color: #27272B;
                color: #FFFFFF;
                qproperty-knobRadius: 5;
                qproperty-knobMargin: 5;
            }
            """
FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """
from Utilities22.Colors import *

BORDER_COLOR = r'rgb(70,70,70)'
BUTTON_COLOR = r'rgb(55,55,55)'
BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'

GROUP_BOX_STYLE = """
                QGroupBox {
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

DIAL_STYLE = """CustomDial {
                background-color: #27272B;
                color: #FFFFFF;
                qproperty-knobRadius: 5;
                qproperty-knobMargin: 5;
            }
            """
FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """
BORDER_COLOR = r'rgb(70,70,70)'
FONT_COLOR = r'rgb(255,255,255)'
BUTTON_COLOR = r'rgb(55,55,55)'
BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'
LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(32,33,30), stop: 0.5 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(22,23,20), stop: 0.5 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'
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

DIAL_STYLE = """CustomDial {
                background-color: #27272B;
                color: #FFFFFF;
                qproperty-knobRadius: 5;
                qproperty-knobMargin: 5;
            }
            """
FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """

BACKGROUND_COLOR_DARK = r'rgb(25,25,25)'
BACKGROUND_COLOR_LIGHT = r'rgb(49,49,49)'
BORDER_COLOR = r'rgb(70,70,70)'
FONT_COLOR = r'rgb(255,255,255)'
BUTTON_COLOR = r'rgb(55,55,55)'
BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'
LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(32,33,30), stop: 0.5 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(22,23,20), stop: 0.5 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'
WIDGET_STYLE = """
            QWidget
            {
                color: """ + FONT_COLOR + """;
                background-color: """ + BACKGROUND_COLOR_DARK + """;
                selection-background-color:""" + BACKGROUND_COLOR_LIGHT + """;
                selection-color: """ + FONT_COLOR + """;
                background-clip: border;
                border-image: none;
                outline: 0;
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

FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """
LABEL_STYLE = """
            QLabel {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;
                padding: 2px;
            }
            """

from Utilities24.Colors import *

BORDER_COLOR = r'rgb(70,70,70)'
FONT_COLOR = r'rgb(255,255,255)'
BUTTON_COLOR = r'rgb(55,55,55)'
BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'
LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(32,33,30), stop: 0.5 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(22,23,20), stop: 0.5 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'
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

DIAL_STYLE = """CustomDial {
                background-color: #27272B;
                color: #FFFFFF;
                qproperty-knobRadius: 5;
                qproperty-knobMargin: 5;
            }
            """
FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """
import os
import sys

sys.path.append(os.path.abspath('../../LoParDataSoftware/'))
from LoParGeneralVariables.Colors import *

BORDER_COLOR = r'rgb(70,70,70)'
BUTTON_COLOR = r'rgb(55,55,55)'
BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'

GROUP_BOX_STYLE = """
                QGroupBox {
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

DIAL_STYLE = """CustomDial {
                background-color: #27272B;
                color: #FFFFFF;
                qproperty-knobRadius: 5px;
                qproperty-knobMargin: 5px;
            }
            """
FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """

sizeGrip = """
    QSizeGrip {
        border: 1px solid white;
        border-radius: 0;
        image: url(./interface/image/icon/expand-diagonal-1);
        width: 16px;
        height: 16px;
    }
    """
statusbar = sizeGrip + """
    QStatusBar {
        /*border: {borderWidth} {borderStyle} rgb(100,100,100);
        border-left-color: rgb(70,70,70);
        border-top-color: rgb(100,100,100);
        border-right-color: rgb(55,55,55);*/
        border-bottom: {borderWidth} solid rgb(45,45,45);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
        font-weight: {fontWeight};
    }
    QStatusBar::hover {
        /*border: {borderWidth} {borderStyle} rgb(125,125,125);
        border-left-color: rgb(90,90,90);
        border-top-color: rgb(140,140,140);
        border-right-color: rgb(65,65,65);*/
        border-bottom: {borderWidth} solid rgb(55,55,55);
        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} {borderColor};
        border-radius: {borderRadius};
    }
    """
statusbarAlert = sizeGrip + """
    QStatusBar {
        background: rgba(25,25,0,0.5);
        /*border: {borderAlert};
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);*/
        border-bottom: {borderWidth} solid rgb(160,60,10);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: rgb(255,255,150);
        font-size: {fontSize};
        font-family: {font};
        font-weight: {fontWeight};
    }
    QStatusBar::hover {
        background: rgba(40,40,0,0.5);
        /*border: {borderAlertHover};
        border-left-color: rgb(255,195,25);
        border-top-color: rgb(255,230,75);
        border-right-color: rgb(230,130,50);*/
        border-bottom: {borderWidth} solid rgb(190,90,35);
        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} yellow;
        border-radius: {borderRadius};
    }
    """
statusbarError = sizeGrip + """
    QStatusBar {
        background: rgba(75,0,0,0.25);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        /*border: {borderError};
        border-left-color: rgb(125,0,0);
        border-top-color: rgb(175,20,20);
        border-right-color: rgb(100,0,0);*/
        border-bottom: {borderWidth} solid rgb(80,0,0);

        color: rgb(255,5,50);
        font-family: {font};
        font-size: {fontSize};
        font-weight: {fontWeight};
    }
    QStatusBar::hover {
        background: rgba(75,0,0,0.25);
        /*border: {borderErrorHover};
        border-left-color: rgb(175,0,0);
        border-top-color: rgb(225,20,20);
        border-right-color: rgb(125,0,0);*/
        border-bottom: {borderWidth} solid rgb(100,0,0);

        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} red;
        border-radius: {borderRadius};
    }
    """
statusbarSuccess = sizeGrip + """
    QStatusBar {
        background: rgba(0,10,0,0.5);
        /*border: {borderValid};
        border-left-color: rgb(0,120,0);
        border-top-color: rgb(0,150,0);
        border-right-color: rgb(0,90,0);*/
        border-bottom: {borderWidth} solid rgb(0,70,0);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: rgb(100,255,100);
        font-family: {font};
        font-size: {fontSize};
        font-weight: {fontWeight};
    }
    QStatusBar::hover {
        background: rgba(0,20,0,0.5);
        /*border: {borderValidHover};
        border-left-color: rgb(0,150,0);
        border-top-color: rgb(0,200,0);
        border-right-color: rgb(0,120,0);*/
        border-bottom: {borderWidth} solid rgb(0,90,0);
        color: {fontColor};
        font-size: {fontSize};
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} green;
        border-radius: {borderRadius};
    }
    """
sizeGrip = """
    QSizeGrip {
        border: 1px solid white;
        border-radius: 0;
        image: url(./interface/image/icon/expand-diagonal-1);
        width: 16px;
        height: 16px;
    }
    """
statusbar = sizeGrip + """
    QStatusBar {
        /*border: {borderWidth} {borderStyle} rgb(100,100,100);
        border-left-color: rgb(70,70,70);
        border-top-color: rgb(100,100,100);
        border-right-color: rgb(55,55,55);*/
        border-bottom: {borderWidth} solid rgb(45,45,45);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
        font-weight: {fontWeight};
    }
    QStatusBar::hover {
        /*border: {borderWidth} {borderStyle} rgb(125,125,125);
        border-left-color: rgb(90,90,90);
        border-top-color: rgb(140,140,140);
        border-right-color: rgb(65,65,65);*/
        border-bottom: {borderWidth} solid rgb(55,55,55);
        color: {fontColor};
        font-family: {font};
        font-size: {fontSize};
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} {borderColor};
        border-radius: {borderRadius};
    }
    """
statusbarAlert = statusbar + """
    QStatusBar {
        background: rgba(25,25,0,0.5);
        border-bottom: {borderWidth} solid rgb(160,60,10);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: rgb(255,255,150);
    }
    QStatusBar::hover {
        background: rgba(40,40,0,0.5);
        border-bottom: {borderWidth} solid rgb(190,90,35);
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} yellow;
        border-radius: {borderRadius};
    }
    """
statusbarError = statusbar + """
    QStatusBar {
        background: rgba(75,0,0,0.25);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        border-bottom: {borderWidth} solid rgb(80,0,0);
        color: rgb(255,5,50);
    }
    QStatusBar::hover {
        background: rgba(75,0,0,0.25);
        border-bottom: {borderWidth} solid rgb(100,0,0);
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} red;
        border-radius: {borderRadius};
    }
    """
statusbarSuccess = statusbar + """
    QStatusBar {
        background: rgba(0,10,0,0.5);
        border-bottom: {borderWidth} solid rgb(0,70,0);
        border-top-left-radius: 5px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 5px;
        border-bottom-left-radius: 10px;
        color: rgb(100,255,100);
    }
    QStatusBar::hover {
        background: rgba(0,20,0,0.5);
        border-bottom: {borderWidth} solid rgb(0,90,0);
    }
    QStatusBar::item {
        border: {borderWidth} {borderStyle} green;
        border-radius: {borderRadius};
    }
    """
settingsButton = """
    QPushButton {
        background: rgb(150,180,170);
        border-left:none;border-top:none;border-right:none;
        border-bottom: 1px solid rgb(50,50,50);
        min-width: 25px;
        font-size: {fontSize};
    }
    QPushButton::hover {
        background: rgb(75,75,75);
        border-bottom: 1px solid rgb(75,75,75);
    }
    QPushButton::pressed {
        background: rgb(50,50,50);
        border-top: 2px solid rgba(0,0,0,0);
        border-bottom: 1px solid rgb(25,25,25);
    }
    """

minButton = """
    /*QPushButton {
        background: rgb(35,35,100);
        border: 1px solid rgb(100,100,255);
        min-width: 25px;
    }
    QPushButton::hover {
        background: rgb(35,35,150);
        border: 1px solid rgb(125,125,255);
    }
    QPushButton::pressed {
        background: rgb(15,15,80);
        border: 1px solid rgb(75,75,200);
    }*/
    QPushButton {
        background: rgba(0,0,0,0);
        border-left:none;border-top:none;border-right:none;
        border-bottom: 1px solid rgb(50,50,50);
        min-width: 25px;
        font-size: {fontSize};
        height:5px;
        margin-bottom: -1px;
    }
    QPushButton::hover {
        background: rgb(75,75,75);
        border-bottom: 1px solid rgb(75,75,75);
    }
    QPushButton::pressed {
        background: rgb(50,50,50);
        border-top: 2px solid rgba(0,0,0,0);
        border-bottom: 1px solid rgb(25,25,25);
    }
    """
xButton = """
    /*QPushButton {
        background: rgb(100,35,35);
        border: 1px solid rgb(255,100,100);
        min-width: 25px;
    }
    QPushButton::hover {
        background: rgb(150,35,35);
        border: 1px solid rgba(255,125,125,0.1);
    }
    QPushButton::pressed {
        background: rgb(80,15,15);
        border: 1px solid rgb(200,75,75);
    }*/
    QPushButton {
        background: rgba(0,0,0,0);
        border-left:none;border-top:none;border-right:none;
        border-bottom: 1px solid rgb(255,50,50);
        min-width: 25px;
        height:5px;
        font-size: {fontSize};
        margin-bottom: -1px;
    }
    QPushButton::hover {
        background: rgb(100,0,0);
        border-bottom: 1px solid rgb(255,75,75);
        margin-bottom: -1px;
    }
    QPushButton::pressed {
        background: rgb(50,0,0);
        border-top: 2px solid rgba(0,0,0,0);
        border-bottom: 1px solid rgb(255,0,0);
        margin-bottom: -1px;
    }
    """
titleBar = """
    QWidget{
        background: {backgroundColorDark};
        background-image: url(./interface/image/background/background.png);
        border-top: 1px solid rgba(50,55,52,1);
        border-bottom: 1px outset {utepBlue};
        padding-top: 0;
        margin-top: 0;
    }
    """
titleBarLabel = """
    QLabel {
        border-radius: 0;
        color: rgb(150,170,180);
        font-family: {font};
        font-size: {fontSize};
    }
    """

button = """
    QPushButton {
        background: {lightererGradient};
        color: {fontColor};
        border: {borderDefault};
        border-radius: {borderRadius};
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton:hover {
        background: {darkGradient};
        border: {borderWidth} {borderStyle} orange;
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);
        border-bottom-color: rgb(160,60,10);
    }
    QPushButton:pressed {
        background: {orangeGradientPressed};
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);
        border-bottom-color: rgb(160,60,10);
        color: black;
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
buttonDisabled = button + """
    QPushButton {
        background: {darkGradient};
        color: {lightGradient};
        border: {borderDisabled};
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton:hover {
        background: {darkGradient};
        border: {borderDisabledHover};
        color: rgba(50,20,20);
    }
    QPushButton:pressed {
        background: {darkGradient};
        border: {borderWidth} {borderStyle} {darkGradient};
        color: rgba(75,25,25);
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
buttonConnected = button + """
    QPushButton {
        background: {orangeGradient};
        color: {fontColor};
        border: {borderConnected};
        border-left-color: rgb(20,20,180);
        border-top-color: rgb(20,20,255);
        border-right-color: rgb(20,20,150);
        border-bottom-color: rgb(20,20,120);
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton::hover {
        background: {yellowGradient};
        border: {borderConnectedHover};
        border-left-color: rgb(35,70,225);
        border-top-color: rgb(50,100,255);
        border-right-color: rgb(30,60,175);
        border-bottom-color: rgb(25,55,150);
        color: black;
    }
    QPushButton:pressed {
        background: {orangeGradientPressed};
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
buttonGlow = button + """
    QPushButton {
        background: {lightererGradient};
        color: {fontColor};
        border: {borderWidth} {borderStyle} orange;
        border-left-color: rgb(255,165,0);
        border-top-color: rgb(255,200,50);
        border-right-color: rgb(200,100,25);
        border-bottom-color: rgb(160,60,10);
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton:hover {
        background: {darkGradient};
        border: {borderWidth} {borderStyle} rgba(125,200,50);
        border-left-color: rgb(255,195,25);
        border-top-color: rgb(255,230,75);
        border-right-color: rgb(230,130,50);
        border-bottom-color: rgb(190,90,35);
    }
    QPushButton:pressed {
        background: {orangeGradientPressed};
        color: black;
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
buttonValid = button + """
    QPushButton {
        background: {lightererGradient};
        border: {borderValid};
        border-left-color: rgb(0,120,0);
        border-top-color: rgb(0,150,0);
        border-right-color: rgb(0,90,0);
        border-bottom-color: rgb(0,70,0);
        color: {fontColor};
        padding: 0px;
        margin: 0px;
        font-size: {fontSize};
    }
    QPushButton:hover {
        background: {yellowGradient};
        border: {borderValidHover};
        border-left-color: rgb(0,150,0);
        border-top-color: rgb(0,200,0);
        border-right-color: rgb(0,120,0);
        border-bottom-color: rgb(0,90,0);
        color: darkblue;
    }
    QPushButton:pressed {
        background: {orangeGradientPressed};
        color: blue;
    }
    QPushButton:flat {
        background: {lightGradient};
        border: {borderColor}; /* no border for a flat push button */
    }
    QPushButton:default {
        background: {lightGradient};
        border: {borderWidth} {borderStyle} {borderColor};
    }
    """
checkboxUnchecked = """
    QCheckBox {
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        background: black;
        border: 4px {borderStyle} {lightGradient};
        border-left-color: rgb(70,70,70);
        border-top-color: rgb(100,100,100);
        border-right-color: rgb(55,55,55);
        border-bottom-color: rgb(45,45,45);
    }
    QCheckBox::indicator:hover {
        background: black;
        border: 4px {borderStyle} {darkGradient};
        border-left-color: rgb(255,195,25);
        border-top-color: rgb(255,230,75);
        border-right-color: rgb(230,130,50);
        border-bottom-color: rgb(190,90,35);
    }
    QCheckBox::indicator:pressed {
        background: {orangeGradientPressed};
    }
    """
checkboxChecked = """
    QCheckBox {
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        background: orange;
        border: 4px {borderStyle} {darkGreenGradient};
        border-left-color: rgb(0,120,0);
        border-top-color: rgb(0,150,0);
        border-right-color: rgb(0,90,0);
        border-bottom-color: rgb(0,70,0);
    }
    QCheckBox::indicator:hover {
        background: {yellowGradient};
        border: 4px {borderStyle} {lightGreenGradient};
        border-left-color: rgb(0,150,0);
        border-top-color: rgb(0,200,0);
        border-right-color: rgb(0,120,0);
        border-bottom-color: rgb(0,90,0);
    }
    QCheckBox::indicator:pressed {
        background: {orangeGradientPressed};
        border: 4px {borderStyle} {darkGreenGradient};
        border-left-color: rgb(0,120,0);
        border-top-color: rgb(0,150,0);
        border-right-color: rgb(0,90,0);
        border-bottom-color: rgb(0,70,0);
    }
    """
checkBox = """
    QCheckBox {
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
    }
    QCheckBox::indicator:checked {
        background: {orangeGradient};
        border: {borderValid};
    }
    QCheckBox::indicator:unchecked {
        background: black;
        border: {borderWidth} {borderStyle} {borderColor};
    }
    QCheckBox::indicator:unchecked:pressed {
        background: {yellowGradient};
    }
    QCheckBox::indicator:checked:pressed {
        background: {orangeGradientPressed};
    }
    QCheckBox::indicator:indeterminate:hover {
    }
    QCheckBox::indicator:indeterminate:pressed {
        background: {orangeGradientPressed};
    }
    QCheckBox::indicator:checked:hover {
        background: {orangeGradientPressed};
        border: {borderValidHover};
    }
    QCheckBox::indicator:unchecked:hover {
        background: black;
        border: {borderWidth} {borderStyle} orange;
    }
    """
from lozoya import scroll

dropdown = """
    QComboBox QAbstractItemView {
        border: {borderDefault};
        background-clip: border;
        selection-background-color: {backgroundColorLighter};
        background: {backgroundColorLight};    
        font-size: {fontSize};
        color: rgb(200,200,200);
    }
    QComboBox {
        background-clip: border;
        background: {backgroundColorLight};    
        color: {contrastColor};
        border: {borderDefault};
        padding: 1px 1px 1px 3px;
        min-width: 6px;
        font-size: {fontSize};
    } 
    QComboBox:!editable::hover {
        background: {lighterGradient};
        background-clip: border;
        border: {borderGlowHover};
    }
    QComboBox:editable {
        background-clip: margin;   
    }
    /* QComboBox gets the "on" state when the popup is open */
    QComboBox:!editable:on, QComboBox::drop-down:editable:on {
        background: {lighterGradient};
        background-clip: border;
        color: rgb(150,150,150);
    }
    QComboBox:on { /* shift the text when the popup opens */
        padding-top: 3px;
        padding-left: 4px;
        background-clip: border;
    }
    QComboBox::drop-down {
        background-clip: border;
        subcontrol-origin: border;
        subcontrol-position: top right;
    }
    QComboBox::drop-down:hover {
        background-clip: border;
        subcontrol-origin: border;
        subcontrol-position: top right;
    }
    QComboBox::down-arrow {
    }
    QComboBox::down-arrow:on { /* shift the arrow when popup is open */
    }
    """
dropdownConnected = """
    QComboBox {
        background-clip: border;
        background: {lightGradient};
        color: {contrastColor};
        border: {borderDisabled};
        padding: 1px 1px 1px 3px;
        min-width: 6px;
        font-size: {fontSize};
    } 
    QComboBox:editable {
        background: {darkGradient};    
        background-clip: margin;   
    }
    /* QComboBox gets the "on" state when the popup is open */
    QComboBox:!editable:on, QComboBox::drop-down:editable:on {
        background: {lighterGradient};
        background-clip: border;
        color: rgb(150,150,150);
    }
    QComboBox:on { /* shift the text when the popup opens */
        padding-top: 3px;
        padding-left: 4px;
        background: {lightGradient};
        background-clip: border;
    }
    QComboBox::drop-down {
        background: rgba(0,0,0,0);
        background-clip: border;
        subcontrol-origin: border;
        subcontrol-position: top right;
    }
    QComboBox::down-arrow {
        background: rgba(0,0,0,0);
    }
    QComboBox::down-arrow:on { /* shift the arrow when popup is open */
    }
    """
dropdownDisabled = """
    QComboBox {
        background-clip: border;
        background: {lightGradient};
        color: {contrastColor};
        border: {borderDisabled};
        padding: 1px 1px 1px 3px;
        min-width: 6px;
        font-size: {fontSize};
    } 
    QComboBox:editable {
        background: {darkGradient};    
        background-clip: margin;   
    }
    /* QComboBox gets the "on" state when the popup is open */
    QComboBox:!editable:on, QComboBox::drop-down:editable:on {
        background: {lighterGradient};
        background-clip: border;
        color: rgb(150,150,150);
    }
    QComboBox:on { /* shift the text when the popup opens */
        padding-top: 3px;
        padding-left: 4px;
        background: {lightGradient};
        background-clip: border;
    }
    QComboBox::drop-down {
        background: rgba(0,0,0,0);
        background-clip: border;
        subcontrol-origin: border;
        subcontrol-position: top right;
    }
    QComboBox::down-arrow {
        background: rgba(0,0,0,0);
    }
    QComboBox::down-arrow:on { /* shift the arrow when popup is open */
    }
    """
dropdownValid = """
    QComboBox {
        background-clip: border;
        background: {lightGradient};
        color: {contrastColor};
        border: {borderDisabled};
        padding: 1px 1px 1px 3px;
        min-width: 6px;
        font-size: {fontSize};
    } 
    QComboBox:editable {
        background: {darkGradient};    
        background-clip: margin;   
    }
    /* QComboBox gets the "on" state when the popup is open */
    QComboBox:!editable:on, QComboBox::drop-down:editable:on {
        background: {lighterGradient};
        background-clip: border;
        color: rgb(150,150,150);
    }
    QComboBox:on { /* shift the text when the popup opens */
        padding-top: 3px;
        padding-left: 4px;
        background: {lightGradient};
        background-clip: border;
    }
    QComboBox::drop-down {
        background: rgba(0,0,0,0);
        background-clip: border;
        subcontrol-origin: border;
        subcontrol-position: top right;
    }
    QComboBox::down-arrow {
        background: rgba(0,0,0,0);
    }
    QComboBox::down-arrow:on { /* shift the arrow when popup is open */
    }
    """
list = scroll + """
    QListView {
        background: rgba(10,7,7,0.1);
        border: {borderDefault};
        selection-color: {contrastColor};
        selection-background-color: {contrastColor};
        font-size:{fontSize};
    }
    QListView:item{
        background: rgb(20,20,20);
        border: 1px outset rgb(30,30,30);
    }
    QListView:item:selected {
        background: rgb(110,110,120);
    }
    QListView:item:hover {
        background: rgb(80,80,100);
    }
    QListView:item:selected:hover {
        background: rgb(80,80,100);
        color: white;
    }

    """
table = scroll + """
    QTableCornerButton::section, QHeaderView::section{
        background: {backgroundColorDark};
        border: {borderWidth} {borderStyle} rgb(60,60,60);
    }
    QTableView {
        border: {borderWidth} solid rgb(60,60,60);
        gridline-color: rgb(30,30,30);
        color: white;
        background: rgba(10,7,7,0.1);
    }
    QTableView::item{
        background: rgb(10,10,10);
        border-bottom: 1px solid rgb(60,60,60);
    }
    QTableView::item:selected {
        background: rgb(150,150,150);
        gridline-color: black;
        border: 1px outset white;
        color: black;
    }
    """
tableDisabled = table + """
    QTableView {
        border: {borderDisabledFull};
    }
    """
tableValid = table + """
    QTableView {
        border: {borderValidFull};
    }
    QTableCornerButton::section {
        border: {borderValidFull};
        background: green;
    }
    QTableView {
        gridline-color: rgb(30,100,30);
        color: white;
    }
    QTableView::item{
        background: rgb(0,0,0);
        border-bottom: 1px solid green;
    }
    """
tableConnected = table + """
    QTableView {
        border: {borderConnectedFull};
    }
    QTableCornerButton::section {
        border: {borderConnectedFull};
        background: {utepBlue};
    }
    QTableView {
        border: {borderWidth} solid rgb(60,60,60);
        gridline-color: rgb(30,30,100);
        color: white;
    }
    QTableView::item{
        background: rgb(0,0,0);
        border-right: 1px solid {utepBlue};
    }
    """
dial = """CustomDial {
        background-color: #27272B;
        color: #FFFFFF;
        qproperty-knobRadius: 5;
        qproperty-knobMargin: 5;
    }
    """
slider = """
    QSlider::handle {
        background: rgb(255,100,50);
        border: {borderWidth} {borderStyle} {lightGradient};
    }
    QSlider::handle:hover {
        background: {yellowGradient};
    }
    QSlider::handle:pressed {
        background: rgb(150,50,25);
    }
    QSlider::sub-page {
    }
    QSlider::add-page {
    }
    QSlider::groove {
    }
    QSlider {
        border: {borderWidth} {borderStyle} {darkGradient};
    }
    QSlider:hover {
        border: {borderWidth} {borderStyle} {darkGradient};
    }
    QSlider::sub-page:hover {       
    }
    QSlider::add-page:hover {
    }

    """
from lozoya import scroll

entry = scroll + """
    QWidget
    {
        background-color: {labelColor};
        border: {borderWidth} {borderStyle} {borderColor};
        font-family: {font};
        font-size: {fontSize};
        padding: 0px;
        margin: 0px;
    }
    QLineEdit, QPlainTextEdit
    {
        border: {borderDefault};
        font-family: {font};
        font-size: {fontSize};
    }
    """
entryGlow = scroll + entry + """
    QLineEdit::hover, QPlainTextEdit::hover
    {
        border: {borderGlowHover};
    }
    QLineEdit, QPlainTextEdit
    {
        background-color: {labelColor};
        border: {borderGlow};
    }
    """
entryConnected = scroll + entry + """
    QLineEdit::hover, QPlainTextEdit::hover
    {
        background: rgba(0,0,10,0.5);
        border: {borderConnectedHover};
    }
    QLineEdit, QPlainTextEdit
    {
        background: rgba(0,0,10,0.5);
        border: {borderConnected};
    }
    """
entryDisabled = scroll + entry + """
    QLineEdit::hover, QPlainTextEdit::hover
    {
        background: rgba(40,20,20,1);
        border: {borderDisabledHover};
        color: rgba(50,25,25,1);
    }
    QLineEdit, QPlainTextEdit
    {
        background: rgba(40,25,25,1);
        border: {borderDisabled};
        color: rgba(50,25,25,1);
    }
    """
entryValid = scroll + entry + """
    QLineEdit::hover, QPlainTextEdit::hover
    {
        background: rgba(0,20,0,0.5);
        border: {borderValidHover};
    }
    QLineEdit, QPlainTextEdit
    {
        background: rgba(0,10,0,0.5);
        border: {borderValid};
    }
    """
entryError = scroll + entry + """
    QLineEdit::hover, QPlainTextEdit::hover
    {
        background: rgba(20,0,0,0.5);
        border: {borderErrorHover};
    }
    QLineEdit, QPlainTextEdit
    {
        background: rgba(10,0,0,0.5);
        border: {borderError};
    }
    """
spin = """
    QSpinBox, QDoubleSpinBox {
        background-color: {lightGradient};
        border: {borderDefault};
        color: white;
        font-size: {fontSize};
    }
    QSpinBox::hover, QDoubleSpinBox::hover {
        border: {borderDefaultHover};
        color: white;
        font-size: {fontSize};
    }
    QSpinBox::up-arrow, QSpinBox::down-arrow {
    }
    QSpinBox::up-arrow:pressed, QSpinBox::down-arrow:pressed {
    }
    QSpinBox::up-arrow, QSpinBox::down-arrow {
    }
    QSpinBox::up-button, QSpinBox::down-button {
    }
    QSpinBox::down-arrow:disabled, QSpinBox::down-arrow:off, QSpinBox::up-arrow:disabled, QSpinBox::up-arrow:off { /* off state when value is max */
    }
    """
spinConnected = spin + """
    QSpinBox, QDoubleSpinBox {
        background: rgba(0,0,10,0.5);
        border: {borderConnected};
    }
    QSpinBox::hover, QDoubleSpinBox::hover {
        background: rgba(0,0,20,0.5);
        border: {borderConnectedHover};
    }
    """
spinDisabled = spin + """
    QSpinBox, QDoubleSpinBox {
        background-color: {entryBGValid};
        border: {borderValid};
    }
    QSpinBox::hover, QDoubleSpinBox::hover {
        background-color: {entryBGValid};
        border: {borderValidHover};
    }
    """
spinError = spin + """
    QSpinBox, QDoubleSpinBox {
        background-color: {entryBGValid};
        border: {borderValid};
    }
    QSpinBox::hover, QDoubleSpinBox::hover {
        background-color: {entryBGValid};
        border: {borderValidHover};
    }
    """
spinGlow = spin + """
    QSpinBox, QDoubleSpinBox {
        background-color: {entryBGValid};
        border: {borderValid};
    }
    QSpinBox::hover, QDoubleSpinBox::hover {
        background-color: {entryBGValid};
        border: {borderValidHover};
    }
    """
spinValid = spin + """
    QSpinBox, QDoubleSpinBox {
        background-color: {entryBGValid};
        border: {borderValid};
    }
    QSpinBox::hover, QDoubleSpinBox::hover {
        background-color: {entryBGValid};
        border: {borderValidHover};
    }
    """
radio = """
    QRadio:default {
    }
    """
from interface.colors import *

borderColor = r'rgb(70,70,70)'
buttonColor = r'rgb(55,55,55)'
borderRadius = r'3px'
borderWidth = r'3px'
borderStyle = r'double'
fontSize = '12pt'

GROUP_BOX_STYLE = """
                QGroupBox {
                    font-size:""" + fontSize + """
                }
            """

LIST_STYLE = """
            QListView {
                background: """ + BACKGROUND_COLOR_DARK + """;
                selection-color: """ + CONTRASTED_COLOR + """;
                selection-background-color: """ + CONTRASTED_COLOR + """;
                border: 1px """ + borderStyle + """ """ + borderColor + """;
                font-size:""" + fontSize + """
            }
            QListView:item:selected {
                color: """ + BACKGROUND_COLOR_DARK + """;
                background: """ + CONTRASTED_COLOR + """;
            }
            QListView:item:hover {
                background: """ + BACKGROUND_COLOR_LIGHTER + """;
            }
            QListView:item:selected:hover {
                background: """ + CONTRASTED_COLOR + """;
            }
            """
TOOLTIP_STYLE = """ 
            QToolBar, QToolButton, QToolTip { 
                background: rgb(56,60,55);
                background: """ + LIGHT_GRADIENT + """;

                color: """ + FONT_COLOR + """;
                spacing: 3px; /* spacing between items in the tool bar */
                border: 1px """ + borderStyle + """ """ + borderColor + """;

            } """

WIDGET_STYLE = TOOLTIP_STYLE + """
            QWidget
            {
                color: """ + FONT_COLOR + """;
                background-color: """ + BACKGROUND_COLOR_DARK + """;
                selection-background-color:""" + BACKGROUND_COLOR_LIGHT + """;
                selection-color: """ + FONT_COLOR + """;
                background-clip: border;
                border-image: none;
                outline: 0;
            }
            """
ENTRY_STYLE = """
            QWidget
            {
                background-color: """ + LABEL_COLOR + """;
                border: 2px """ + borderStyle + """ """ + borderColor + """;
                padding: 0px;
                margin: 0px;
                font-size: """ + fontSize + """;
            }
            QPlainTextEdit
            {
                font-size: """ + fontSize + """;
            }
            """
GLOW_ENTRY_STYLE = """
            QPlainTextEdit::hover
            {
                border: 2px """ + borderStyle + """ """ + 'yellow' + """;
                font-size: """ + fontSize + """;
            }
            QPlainTextEdit
            {
                background-color: """ + LABEL_COLOR + """;
                border: 2px """ + borderStyle + """ """ + 'orange' + """;
                padding: 0px;
                margin: 0px;
                font-size: """ + fontSize + """;
            }
            """
WINDOW_STYLE = """
            QMainWindow {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;
                background-image: url(./interface7/img/background.png);

            }
            QMainWindow::separator {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + BACKGROUND_COLOR_LIGHT + """;
                width: 10px; /* when vertical */
                height: 10px; /* when horizontal */
            }

            QMainWindow::separator:hover {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: black;
            }

            """
BUTTON_STYLE = """
            QPushButton {
                background: """ + LIGHTER_GRADIENT + """;
                color: """ + FONT_COLOR + """;
                border: """ + borderWidth + """ """ + borderStyle + """ """ + borderColor + """;
                border-radius: """ + borderRadius + """;
                min-width: 20px;
                padding: 0px;
                margin: 0px;
                font-size: """ + fontSize + """;
            }

            QPushButton:hover {
                background: """ + DARK_GRADIENT + """;
                border: """ + borderWidth + """ """ + borderStyle + """ """ + 'orange' + """;
            }

            QPushButton:pressed {
                background: yellow;
                color: black;
            }

            QPushButton:flat {
                background: """ + LIGHT_GRADIENT + """;
                border: """ + borderColor + """;; /* no border for a flat push button */
            }

            QPushButton:default {
                background: """ + LIGHT_GRADIENT + """;
                border: """ + borderWidth + """ """ + borderStyle + """ """ + borderColor + """;
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

            QCheckBox::indicator:checked {
                background: orange;
                border: 2px """ + borderStyle + """ """ + borderColor + """;
            }

            QCheckBox::indicator:unchecked {
                background: white;
                border: 2px """ + borderStyle + """ """ + borderColor + """;
            }

            QCheckBox::indicator:unchecked:hover {
            }

            QCheckBox::indicator:unchecked:pressed {
                background: yellow;
            }

            QCheckBox::indicator:checked:hover {
            }

            QCheckBox::indicator:checked:pressed {
                background: yellow;
            }

            QCheckBox::indicator:indeterminate:hover {
            }

            QCheckBox::indicator:indeterminate:pressed {
            }

            QCheckBox::indicator:checked:hover {
                background: yellow;
                border: 2px """ + borderStyle + """ """ + 'orange' + """;
            }

            QCheckBox::indicator:unchecked:hover {
                border: 2px """ + borderStyle + """ """ + 'orange' + """;
            }
            """

COMBO_STYLE = """
            QComboBox QAbstractItemView {
                border: 2px solid """ + BACKGROUND_COLOR_LIGHT + """;
                selection-background-color: """ + BACKGROUND_COLOR_LIGHTER + """;
                background: """ + BACKGROUND_COLOR_DARK + """;
            }

            QComboBox {
                background: """ + LIGHT_GRADIENT + """;
                color: """ + CONTRASTED_COLOR + """;
                border: 3px """ + borderStyle + """ """ + borderColor + """;
                padding: 1px 1px 1px 3px;
                min-width: 6px;
            }

            QComboBox:!editable::hover {
                background: """ + DARK_GRADIENT + """; 
            }

            QComboBox:editable {
                background: """ + DARK_GRADIENT + """;         
            }

            /* QComboBox gets the "on" state when the popup is open */
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: """ + DARK_GRADIENT + """;
            }

            QComboBox:on { /* shift the text when the popup opens */
                padding-top: 3px;
                padding-left: 4px;
                background: """ + LIGHT_GRADIENT + """;
            }

            QComboBox::drop-down {
                background: """ + LIGHT_GRADIENT + """;
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 10px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: """ + borderStyle + """;
                border-top-right-radius: """ + borderRadius + """;
                border-bottom-right-radius: """ + borderRadius + """;
            }

            QComboBox::down-arrow {

            }

            QComboBox::down-arrow:on { /* shift the arrow when popup is open */
                top: 1px;
                left: 1px;
            }
            """

DIAL_STYLE = """CustomDial {
                background-color: #27272B;
                color: #FFFFFF;
                qproperty-knobRadius: 5;
                qproperty-knobMargin: 5;
            }
            """
FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/img/background.png);
            }
            """
LABEL_STYLE = TOOLTIP_STYLE + """
            QLabel {
                background: """ + LABEL_COLOR + """;
                color: """ + FONT_COLOR + """;
                border: 1px """ + borderStyle + """ """ + borderColor + """;
                border-radius: """ + borderRadius + """;
                padding: 2px;
                font-size: """ + fontSize + """;
            }
            """

MENUBAR_STYLE = """
            QMenuBar {
                background: """ + BACKGROUND_COLOR_LIGHTERER + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid rgb(0,0,0);
            }

            QMenuBar::item {
                background: """ + BACKGROUND_COLOR_LIGHTER + """;
                color: """ + FONT_COLOR + """;
            }

            QMenuBar::item::hover {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
            }

            QMenu {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid #000;           
            }

            QMenu::item::selected {
                background-color: """ + BACKGROUND_COLOR_LIGHT + """;
            }
            """

SCROLL_STYLE = """
            QScrollBar:vertical {
                 background: """ + BACKGROUND_COLOR_DARK + """;
                 width: 15px;
                 margin: 22px 0 22px 0;
             }
             QScrollBar::handle:vertical {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::handle:vertical:pressed {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::add-line:vertical {
                 background: """ + LIGHT_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: bottom;
                 subcontrol-origin: margin;
             }

             QScrollBar::sub-line:vertical {
                 background: """ + DARK_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: top;
                 subcontrol-origin: margin;
             }
             QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                 border: 2px solid """ + DARK_GRADIENT + """;
                 width: 3px;
                 height: 3px;
                 background: black;
             }
            QScrollBar::up-arrow:vertical:pressed, QScrollBar::down-arrow:vertical:pressed {
                 border: 2px solid """ + LIGHT_GRADIENT + """;
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
                background: """ + BACKGROUND_COLOR_DARK + """;
            }
            QSplitter::handle {
                background: """ + 'orange' + """;
                color: """ + FONT_COLOR + """;
            }

            QSplitter::handle:hover {
                background: white;
            }


            QSplitter::handle:horizontal {
                width: 4px;
            }

            QSplitter::handle:vertical {
                height: 2px;
            }

            QSplitter::handle:pressed {
                background-color: """ + 'yellow' + """;
            }
            """

TAB_STYLE = """
            QTabWidget::pane { /* The tab widget frame */
            border-top: 2px solid """ + BACKGROUND_COLOR_LIGHT + """;
            }
            QTabWidget::tab-bar {
            left: 5px; /* move to the right by 5px */
            background: """ + BACKGROUND_COLOR_LIGHT + """;
            }
            /* Style the tab using the tab sub-control. Note that it reads QTabBar _not_ QTabWidget */
            QTabBar::tab {
            background: """ + BACKGROUND_COLOR_LIGHT + """;
            border: 2px solid """ + BACKGROUND_COLOR_LIGHT + """;
            border-bottom-color: """ + BACKGROUND_COLOR_LIGHT + """; /* same as the pane color */
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            min-width: 8ex;
            padding: 2px;
            }
            QTabBar::tab:selected, QTabBar::tab:hover {
            background: """ + BACKGROUND_COLOR_DARK + """;
            }
            QTabBar::tab:selected {
            border-color: #9B9B9B;
            border-bottom-color: #C2C7CB; /* same as pane color */
            }
            QTabBar::tab:!selected {
            margin-top: 2px; /* make non-selected tabs look smaller */
            background: """ + BACKGROUND_COLOR_LIGHT + """;
            border-color: """ + BACKGROUND_COLOR_LIGHTER + """;
            }
            """

TOOLBAR_STYLE = """ 
            QToolBar, QToolButton, QToolTip { 
                background: rgb(56,60,55);
                background: """ + LIGHT_GRADIENT + """;

                color: """ + FONT_COLOR + """;
                spacing: 3px; /* spacing between items in the tool bar */
                border: 1px """ + borderStyle + """ """ + borderColor + """;

            } 
            QToolButton {
                background-image: url(./interface7/img/background.png);
                font-size: 18pt;
            }

            QToolButton:hover {
                background: """ + DARK_GRADIENT + """;
                border: 0px;
            }

            QToolBar::handle {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                border: 1px solid rgb(100,100,100);
            } 
            """

PINNED_BUTTON_STYLE = """
            QPushButton {
                background: """ + 'orange' + """;
                color: """ + FONT_COLOR + """;
                border: """ + borderWidth + """ """ + borderStyle + """ """ + buttonColor + """;
                border-radius: """ + borderRadius + """;
                min-width: 20px;
                padding: 0px;
                margin: 0px;
            }

            QPushButton:hover {
                background: """ + 'yellow' + """;
                border: """ + borderWidth + """ """ + borderStyle + """ """ + 'orange' + """;
            }

            QPushButton:pressed {
                background: """ + DARK_GRADIENT + """;
            }

            QPushButton:flat {
                background: """ + LIGHT_GRADIENT + """;
                border: """ + borderColor + """;; /* no border for a flat push button */
            }

            QPushButton:default {
                background: """ + LIGHT_GRADIENT + """;
                border: """ + borderWidth + """ """ + borderStyle + """ """ + borderColor + """;
            }
            """
RADIO_STYLE = """
            QRadio:default {
            }
"""

import re
import ctypes

from matplotlib import colors as mcolors

'''COLOR_SET = ('peru',
             'plum',
             'powderblue',
             'moccasin',
             #'mintcream',
             #'mistyrose',
             #'turquoise',
             'pink',
             'violet',
             'cyan')'''
'''COLOR_SET = (
'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk',
'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue',
'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey',
'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender',
'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue',
'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin',
'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen',
'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red',
'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue',
'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato',
'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen')'''
BACKGROUND_COLOR_DARK = r'rgb(25,25,25)'
BACKGROUND_COLOR_LIGHT = r'rgb(49,49,49)'
BACKGROUND_COLOR_LIGHTER = r'rgb(79,79,79)'
BACKGROUND_COLOR_LIGHTERER = r'rgb(92,92,92)'
COLOR_SET = (
    '#ff0000', '#0000ff', '#ffa500', '#30d5c8', '#7cfc00', '#ffff00', '#ffc0cb', '#9370db', '#ee82ee', '#00ffff')
plotColors = {
    'red':    '#e27d60',
    'blue':   '#557a95',
    'orange': '#e8a87c',
    'purple': '#c38d9e',
    'green':  '#afd275',
    'pink':   '#e7717d',
}
theme = 'dark'


def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.1)


def _format(s, palette):
    try:
        matches = re.findall('{.+?}', s)
        result = '' + s
        for m in matches:
            result = result.replace(m, getattr(palette, m[1:-1]))
        return result
    except Exception as e:
        statusMsg = 'QSS formatting error: {}'.format(str(e))
        if str(e) == 'expected string or bytes-like object':
            statusMsg = '{} but got {} type object {}'.format(statusMsg, type(s), s)
        print(statusMsg)


class Palette:
    def __init__(self, app):
        self.app = app
        self.titlebarBtnSize = 20
        self.gradientTemplate = r'qlineargradient(x1: 0.4, y1: 1, x2: -0.3, y2: -0.8, stop: 0.05 {}, stop: 0.9 {}, stop: 0.5 {}, stop: 1.0 {});'
        self.borderColor = r'rgb(70,70,70)'
        self.buttonColor = r'rgb(55,55,55)'
        self.borderRadius = r'5px'
        self.borderWidth = r'1px'
        self.borderWidthThick = r'2px'
        self.borderStyle = r'double'
        self.borderStyle = r'solid'
        self.fontWeight = '100'
        self.fontColor = 'rgb(100,100,100)'
        self.minHeight = 20
        self.user32 = ctypes.windll.user32
        self.width = self.user32.GetSystemMetrics(0)
        self.height = self.user32.GetSystemMetrics(1)
        self.hand = 'hand2'
        self.ptr = 'left_ptr'
        self.font = 'Arial'
        self.fontSize = '10pt'
        self.smallFont = (self.font, 8)
        self.mediumFont = (self.font, 10)
        self.largeFont = (self.font, 12)
        self.entryBGValid = 'rgba(0,255,0,0.05)'
        self.entryBGValidHover = 'rgba(0,255,0,0.1)'
        self.entryBGConnected = 'rgba(0,0,255,0.05)'
        self.entryBGConnectedHover = 'rgba(0,0,255,0.1)'
        self.topPadding = 15
        self.bottomPadding = 10
        self.ridder = 'border-left: none;border-top: none;border-right: none'
        self.standardBorder = '{0} {1} {2}'.format(self.borderWidth, self.borderStyle, '{}')
        self.cutBorder = '{0} {1} {2}; {3}'.format(self.borderWidth, self.borderStyle, '{}', self.ridder)
        _fontSize = 9
        self.plotCanvas = r'0.1'
        self.plotBackground = r'0.1'
        self.backgroundColorDarker = r'rgb(15,15,15)'
        self.backgroundColorDark = r'rgb(25,25,25)'
        self.backgroundColorLight = r'rgb(49,49,49)'
        self.backgroundColorLighter = r'rgb(79,79,79)'
        self.backgroundColorLighterer = r'rgb(92,92,92)'
        self.backgroundColorLightererer = r'rgb(120,120,120)'
        self.labelColor = r'rgb(25,25,25)'
        self.GRAPH_BACKGROUND_COLOR = r'rgb(92,92,92)'
        self.contrastColor = r'white'
        self.lightestGradient = self.gradientTemplate.format(
            'rgb(80,80,80)',
            'rgb(120,120,120)',
            'rgb(110,110,110)',
            'rgb(90,90,90)',
        )
        self.lightererGradient = self.gradientTemplate.format(
            'rgb(80,80,80)',
            'rgb(120,120,120)',
            'rgb(110,110,110)',
            'rgb(90,90,90)',
        )
        self.lighterGradient = self.gradientTemplate.format(
            'rgb(40,40,40)',
            'rgb(80,80,80)',
            'rgb(70,70,70)',
            'rgb(50,50,50)',
        )
        self.lightGradient = self.gradientTemplate.format(
            'rgb(32,33,30)',
            'rgb(56,58,55)',
            'rgb(61,63,60)',
            'rgb(46,50,48)',
        )
        self.darkGradient = self.gradientTemplate.format(
            'rgb(22,23,20)',
            'rgb(90,90,90)',
            'rgb(114,115,113)',
            'rgb(46,50,48)',
        )
        self.darkGreenGradient = self.gradientTemplate.format(
            'rgb(0,90,0)',
            'rgb(0,255,0)',
            'rgb(0,115,0)',
            'rgb(0,255,48)',
        )
        self.lightGreenGradient = self.gradientTemplate.format(
            'rgb(0,125,0)',
            'rgb(0,255,0)',
            'rgb(0,200,0)',
            'rgb(0,120,48)',
        )
        self.darkRedGradient = self.gradientTemplate.format(
            'rgb(23,0,0)',
            'rgb(75,0,0)',
            'rgb(90,0,0)',
            'rgb(50,0,0)',
        )
        self.darkBlueGradient = self.gradientTemplate.format(
            'rgb(0,0,23)',
            'rgb(0,0,75)',
            'rgb(0,0,90)',
            'rgb(0,0,50)',
        )
        self.orangeGradient = self.gradientTemplate.format(
            'rgb(255,100,0)',
            'rgb(25,10,10)',
            'rgb(255,225,100)',
            'rgb(0,0,0)',
        )
        self.orangeBlueGradient = self.gradientTemplate.format(
            'darkblue',
            'rgb(25,10,10)',
            'orange',
            'rgb(0,0,0)',
        )
        self.yellowGradient = self.gradientTemplate.format(
            'orange',
            'rgb(25,10,10)',
            'yellow',
            'rgb(0,0,0)',
        )
        self.orangeGradientPressed = self.gradientTemplate.format(
            'rgb(200,25,0)',
            'rgb(25,10,10)',
            'rgb(255,225,100)',
            'rgb(0,0,0)',
        )
        self.fontColor = r'rgb(255,255,255)'
        self.plotColors = {
            'red':    '#e27d60',
            'blue':   '#557a95',
            'orange': '#e8a87c',
            'purple': '#c38d9e',
            'green':  '#afd275',
            'pink':   '#e7717d',
        }
        self.thrusterColor = 'white'
        self.borderAlert = self.cutBorder.format('rgba(175,175,25,1)')
        self.borderAlertHover = self.cutBorder.format('rgba(200,200,50,1)')
        self.borderDefault = self.cutBorder.format('rgb(0,0,0)')
        self.borderDefaultHover = self.cutBorder.format('rgb(0,0,0)')
        self.borderGlow = self.cutBorder.format('orange')
        self.borderGlowHover = self.cutBorder.format('yellow')
        self.borderConnected = self.cutBorder.format('rgb(20,20,150)')
        self.borderConnectedHover = self.cutBorder.format('rgb(50,75,255)')
        self.borderDisabled = self.cutBorder.format(self.darkGradient)
        self.borderDisabledHover = self.cutBorder.format(self.darkGradient)
        self.borderError = self.cutBorder.format('rgba(125,0,0,1)')
        self.borderErrorHover = self.cutBorder.format('rgba(170,0,0,1)')
        self.borderValid = self.cutBorder.format('rgba(0,75,0,1)')
        self.borderValidHover = self.cutBorder.format('rgba(0,100,0,1)')
        self.borderAlertFull = self.standardBorder.format('rgba(175,175,25,1)')
        self.borderAlertHoverFull = self.standardBorder.format('rgba(200,200,50,1)')
        self.borderDefaultFull = self.standardBorder.format('rgb(0,0,0)')
        self.borderDefaultHoverFull = self.standardBorder.format('rgb(0,0,0)')
        self.borderGlowFull = self.standardBorder.format('orange')
        self.borderGlowHoverFull = self.standardBorder.format('yellow')
        self.borderConnectedFull = self.standardBorder.format('rgb(20,20,150)')
        self.borderConnectedHoverFull = self.standardBorder.format('rgb(50,75,255)')
        self.borderDisabledFull = self.standardBorder.format(self.darkGradient)
        self.borderDisabledHoverFull = self.standardBorder.format(self.darkGradient)
        self.borderErrorFull = self.standardBorder.format('rgba(125,0,0,1)')
        self.borderErrorHoverFull = self.standardBorder.format('rgba(170,0,0,1)')
        self.borderValidFull = self.standardBorder.format('rgba(0,75,0,1)')
        self.borderValidHoverFull = self.standardBorder.format('rgba(0,100,0,1)')
        self.utepBlue = 'rgb(25,75,125)'


class Palette:
    def __init__(self, app):
        self.app = app
        self.titlebarBtnSize = 20
        self.gradientTemplate = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 {}, stop: 0.5 {}, stop: 0.5 {}, stop: 1.0 {});'
        self.borderColor = r'rgb(70,70,70)'
        self.buttonColor = r'rgb(55,55,55)'
        self.borderRadius = r'5px'
        self.borderWidth = r'1px'
        self.borderWidthThick = r'2px'
        self.borderStyle = r'double'
        self.borderStyle = r'solid'
        self.fontWeight = '100'
        self.fontColor = 'rgb(100,100,100)'
        self.minHeight = 20
        self.user32 = ctypes.windll.user32
        self.width = self.user32.GetSystemMetrics(0)
        self.height = self.user32.GetSystemMetrics(1)
        self.hand = 'hand2'
        self.ptr = 'left_ptr'
        self.font = 'Arial'
        self.fontSize = '10pt'
        self.smallFont = (self.font, 8)
        self.mediumFont = (self.font, 10)
        self.largeFont = (self.font, 12)
        self.entryBGValid = 'rgba(0,255,0,0.05)'
        self.entryBGValidHover = 'rgba(0,255,0,0.1)'
        self.entryBGConnected = 'rgba(0,0,255,0.05)'
        self.entryBGConnectedHover = 'rgba(0,0,255,0.1)'
        self.topPadding = 15
        self.bottomPadding = 10
        self.ridder = 'border-left: none;border-top: none;border-right: none'
        self.standardBorder = '{0} {1} {2}'.format(self.borderWidth, self.borderStyle, '{}')
        self.cutBorder = '{0} {1} {2}; {3}'.format(self.borderWidth, self.borderStyle, '{}', self.ridder)
        _fontSize = 9
        self.plotCanvas = r'0.1'
        self.plotBackground = r'0.1'
        self.backgroundColorDarker = r'rgb(15,15,15)'
        self.backgroundColorDark = r'rgb(25,25,25)'
        self.backgroundColorLight = r'rgb(49,49,49)'
        self.backgroundColorLighter = r'rgb(79,79,79)'
        self.backgroundColorLighterer = r'rgb(92,92,92)'
        self.backgroundColorLightererer = r'rgb(120,120,120)'
        self.labelColor = r'rgb(25,25,25)'
        self.GRAPH_BACKGROUND_COLOR = r'rgb(92,92,92)'
        self.contrastColor = r'white'
        self.lightestGradient = self.gradientTemplate.format(
            'rgb(80,80,80)', 'rgb(120,120,120)', 'rgb(110,110,110)',
            'rgb(90,90,90)', )
        self.lightererGradient = self.gradientTemplate.format(
            'rgb(80,80,80)', 'rgb(120,120,120)', 'rgb(110,110,110)',
            'rgb(90,90,90)', )
        self.lighterGradient = self.gradientTemplate.format(
            'rgb(40,40,40)', 'rgb(80,80,80)', 'rgb(70,70,70)',
            'rgb(50,50,50)', )
        self.lightGradient = self.gradientTemplate.format(
            'rgb(32,33,30)', 'rgb(56,58,55)', 'rgb(61,63,60)',
            'rgb(46,50,48)', )
        self.darkGradient = self.gradientTemplate.format(
            'rgb(22,23,20)', 'rgb(90,90,90)', 'rgb(114,115,113)',
            'rgb(46,50,48)', )
        self.darkGreenGradient = self.gradientTemplate.format(
            'rgb(0,90,0)', 'rgb(0,255,0)', 'rgb(0,115,0)',
            'rgb(0,255,48)', )
        self.lightGreenGradient = self.gradientTemplate.format(
            'rgb(0,125,0)', 'rgb(0,255,0)', 'rgb(0,200,0)',
            'rgb(0,120,48)', )
        self.darkRedGradient = self.gradientTemplate.format(
            'rgb(23,0,0)', 'rgb(75,0,0)', 'rgb(90,0,0)',
            'rgb(50,0,0)', )
        self.darkBlueGradient = self.gradientTemplate.format(
            'rgb(0,0,23)', 'rgb(0,0,75)', 'rgb(0,0,90)',
            'rgb(0,0,50)', )
        self.orangeGradient = self.gradientTemplate.format(
            'rgb(255,100,0)', 'rgb(25,10,10)', 'rgb(255,225,100)',
            'rgb(0,0,0)', )
        self.orangeBlueGradient = self.gradientTemplate.format('darkblue', 'rgb(25,10,10)', 'orange', 'rgb(0,0,0)', )
        self.yellowGradient = self.gradientTemplate.format('orange', 'rgb(25,10,10)', 'yellow', 'rgb(0,0,0)', )
        self.orangeGradientPressed = self.gradientTemplate.format(
            'rgb(200,25,0)', 'rgb(25,10,10)', 'rgb(255,225,100)',
            'rgb(0,0,0)', )
        self.fontColor = r'rgb(255,255,255)'
        self.plotColors = {
            'red':   '#e27d60', 'blue': '#557a95', 'orange': '#e8a87c', 'purple': '#c38d9e',
            'green': '#afd275', 'pink': '#e7717d',
        }
        self.thrusterColor = 'white'
        self.borderAlert = self.cutBorder.format('rgba(175,175,25,1)')
        self.borderAlertHover = self.cutBorder.format('rgba(200,200,50,1)')
        self.borderDefault = self.cutBorder.format('rgb(0,0,0)')
        self.borderDefaultHover = self.cutBorder.format('rgb(0,0,0)')
        self.borderGlow = self.cutBorder.format('orange')
        self.borderGlowHover = self.cutBorder.format('yellow')
        self.borderConnected = self.cutBorder.format('rgb(20,20,150)')
        self.borderConnectedHover = self.cutBorder.format('rgb(50,75,255)')
        self.borderDisabled = self.cutBorder.format(self.darkGradient)
        self.borderDisabledHover = self.cutBorder.format(self.darkGradient)
        self.borderError = self.cutBorder.format('rgba(125,0,0,1)')
        self.borderErrorHover = self.cutBorder.format('rgba(170,0,0,1)')
        self.borderValid = self.cutBorder.format('rgba(0,75,0,1)')
        self.borderValidHover = self.cutBorder.format('rgba(0,100,0,1)')
        self.borderAlertFull = self.standardBorder.format('rgba(175,175,25,1)')
        self.borderAlertHoverFull = self.standardBorder.format('rgba(200,200,50,1)')
        self.borderDefaultFull = self.standardBorder.format('rgb(0,0,0)')
        self.borderDefaultHoverFull = self.standardBorder.format('rgb(0,0,0)')
        self.borderGlowFull = self.standardBorder.format('orange')
        self.borderGlowHoverFull = self.standardBorder.format('yellow')
        self.borderConnectedFull = self.standardBorder.format('rgb(20,20,150)')
        self.borderConnectedHoverFull = self.standardBorder.format('rgb(50,75,255)')
        self.borderDisabledFull = self.standardBorder.format(self.darkGradient)
        self.borderDisabledHoverFull = self.standardBorder.format(self.darkGradient)
        self.borderErrorFull = self.standardBorder.format('rgba(125,0,0,1)')
        self.borderErrorHoverFull = self.standardBorder.format('rgba(170,0,0,1)')
        self.borderValidFull = self.standardBorder.format('rgba(0,75,0,1)')
        self.borderValidHoverFull = self.standardBorder.format('rgba(0,100,0,1)')
        self.utepBlue = 'rgb(25,75,125)'


class Palette:
    def set_theme(self, theme):
        self.theme = theme
        self.gradientTemplate = r'qlineargradient(x1: 0.4, y1: 1, x2: -0.3, y2: -0.8, stop: 0.05 {}, stop: 0.9 {}, stop: 0.5 {}, stop: 1.0 {});'
        self.borderColor = r'rgb(70,70,70)'
        self.buttonColor = r'rgb(55,55,55)'
        self.borderRadius = r'3px'
        self.borderWidth = r'2px'
        self.borderWidthThick = r'4px'
        self.borderStyle = r'double'
        self.borderStyle = r'solid'
        self.fontSize = '9pt'
        self.fontWeight = '100'
        self.fontColor = 'rgb(100,100,100)'
        self.minHeight = 25
        self.user32 = ctypes.windll.user32
        self.width = self.user32.GetSystemMetrics(0)
        self.height = self.user32.GetSystemMetrics(1)
        self.hand = 'hand2'
        self.ptr = 'left_ptr'
        self.smallFont = ('Verdana', 8)
        self.mediumFont = ('Verdana', 10)
        self.largeFont = ('Verdana', 12)
        self.entryBGValid = 'rgba(0,255,0,0.05)'
        self.entryBGValidHover = 'rgba(0,255,0,0.1)'
        self.entryBGConnected = 'rgba(0,0,255,0.05)'
        self.entryBGConnectedHover = 'rgba(0,0,255,0.1)'
        self.font = 'Arial'
        self.topPadding = 15
        self.bottomPadding = 10
        self.borderConnected = '{} {} rgb(20,20,150)'.format(self.borderWidth, self.borderStyle)
        self.borderConnectedHover = '{} {} rgb(50,75,255)'.format(self.borderWidth, self.borderStyle)
        self.borderValid = '{} {} rgba(0,75,0,1)'.format(self.borderWidth, self.borderStyle)
        self.borderValidHover = '{} {} rgba(0,100,0,1)'.format(self.borderWidth, self.borderStyle)
        self.borderDisconnected = '{} {} rgba(50,25,25,1)'.format(self.borderWidth, self.borderStyle)
        self.borderDisconnectedHover = '{} {} rgba(75, 45, 45, 1)'.format(self.borderWidth, self.borderStyle)
        self.borderError = '{} {} rgba(125,0,0,1)'.format(self.borderWidth, self.borderStyle)
        self.borderErrorHover = '{} {} rgba(170,0,0,1)'.format(self.borderWidth, self.borderStyle)
        self.borderAlert = '{} {} rgba(175,175,25,1)'.format(self.borderWidth, self.borderStyle)
        self.borderAlertHover = '{} {} rgba(200,200,50,1)'.format(self.borderWidth, self.borderStyle)

        _fontSize = 9
        if self.theme == 'dark':
            self.plotCanvas = r'0.1'
            self.plotBackground = r'0.1'
            self.backgroundColorDark = r'rgb(25,25,25)'
            self.backgroundColorLight = r'rgb(49,49,49)'
            self.backgroundColorLighter = r'rgb(79,79,79)'
            self.backgroundColorLightererer = r'rgb(92,92,92)'
            self.labelColor = r'rgb(25,25,25)'
            self.GRAPH_BACKGROUND_COLOR = r'rgb(92,92,92)'
            self.contrastColor = r'white'
            self.lightestGradient = self.gradientTemplate.format(
                'rgb(80,80,80)', 'rgb(120,120,120)',
                'rgb(110,110,110)', 'rgb(90,90,90)', )
            self.lightererGradient = self.gradientTemplate.format(
                'rgb(80,80,80)', 'rgb(120,120,120)',
                'rgb(110,110,110)', 'rgb(90,90,90)', )
            self.lighterGradient = self.gradientTemplate.format(
                'rgb(40,40,40)', 'rgb(80,80,80)', 'rgb(70,70,70)',
                'rgb(50,50,50)', )
            self.lightGradient = self.gradientTemplate.format(
                'rgb(32,33,30)', 'rgb(56,58,55)', 'rgb(61,63,60)',
                'rgb(46,50,48)', )
            self.darkGradient = self.gradientTemplate.format(
                'rgb(22,23,20)', 'rgb(90,90,90)', 'rgb(114,115,113)',
                'rgb(46,50,48)', )
            self.darkGreenGradient = self.gradientTemplate.format(
                'rgb(0,90,0)', 'rgb(0,255,0)', 'rgb(0,115,0)',
                'rgb(0,255,48)', )
            self.lightGreenGradient = self.gradientTemplate.format(
                'rgb(0,125,0)', 'rgb(0,255,0)', 'rgb(0,200,0)',
                'rgb(0,120,48)', )
            self.darkRedGradient = self.gradientTemplate.format(
                'rgb(23,0,0)', 'rgb(75,0,0)', 'rgb(90,0,0)',
                'rgb(50,0,0)', )
            self.darkBlueGradient = self.gradientTemplate.format(
                'rgb(0,0,23)', 'rgb(0,0,75)', 'rgb(0,0,90)',
                'rgb(0,0,50)', )
            self.orangeGradient = self.gradientTemplate.format(
                'rgb(255,100,0)', 'rgb(25,10,10)', 'rgb(255,225,100)',
                'rgb(0,0,0)', )
            self.orangeBlueGradient = self.gradientTemplate.format(
                'darkblue', 'rgb(25,10,10)', 'orange',
                'rgb(0,0,0)', )
            self.yellowGradient = self.gradientTemplate.format('orange', 'rgb(25,10,10)', 'yellow', 'rgb(0,0,0)', )
            self.orangeGradientPressed = self.gradientTemplate.format(
                'rgb(200,25,0)', 'rgb(25,10,10)',
                'rgb(255,225,100)', 'rgb(0,0,0)', )
            self.fontColor = r'rgb(255,255,255)'

        elif self.theme == 'default':
            self.backgroundColorDark = r'rgb(252,252,252)'
            self.backgroundColorLight = r'rgb(255,255,255)'
            self.backgroundColorLighter = r'rgb(240,240,240)'
            self.backgroundColorLightererer = r'rgb(255,255,255)'
            self.labelColor = r'rgb(255,255,255)'
            self.GRAPH_BACKGROUND_COLOR = r'rgb(255,255,255)'
            self.contrastColor = r'rgb(0,0,0)'
            self.lightGradient = r'qlineargradient(x1: 0.4, y1: 1, x2: -0.3, y2: -0.8, stop: 0.05 rgb(222,222,222), stop: 0.9 rgb(248,250,247), stop: 0.5 rgb(253,255,252), stop: 1.0 rgb(232,232,232));'
            self.darkGradient = r'qlineargradient(x1: 0.4, y1: 1, x2: -0.3, y2: -0.8, stop: 0.05 rgb(190,193,190), stop: 0.9 rgb(245,245,245), stop: 0.5 rgb(254,255,253), stop: 1.0 rgb(200,198,200));'
            self.fontColor = r'rgb(0,0,0)'

        elif self.theme == 'pink':
            self.backgroundColorDark = r'rgb(153,204,204)'
            self.backgroundColorLight = r'rgb(225,255,255)'
            self.backgroundColorLighter = r'rgb(240,240,240)'
            self.backgroundColorLightererer = r'rgb(235,248,249)'
            self.labelColor = r'rgb(255,255,255)'
            self.GRAPH_BACKGROUND_COLOR = r'rgb(255,255,255)'
            self.contrastColor = r'black'
            self.lightGradient = r'qlineargradient(x1: 0.4, y1: 1, x2: -0.3, y2: -0.8, stop: 0.05 rgb(222,222,222), stop: 0.9 rgb(248,250,247), stop: 0.5 rgb(253,255,252), stop: 1.0 rgb(232,232,232));'
            self.darkGradient = r'qlineargradient(x1: 0.4, y1: 1, x2: -0.3, y2: -0.8, stop: 0.05 rgb(190,193,190), stop: 0.9 rgb(245,245,245), stop: 0.5 rgb(254,255,253), stop: 1.0 rgb(200,198,200));'
            self.fontColor = r'rgb(0,0,0)'

        self.plotColors = {
            'red':   '#e27d60', 'blue': '#557a95', 'orange': '#e8a87c', 'purple': '#c38d9e',
            'green': '#afd275', 'pink': '#e7717d',
        }
        self.thrusterColor = 'white'


class Palette:
    def __init__(self, parent, theme):
        self.parent = parent
        self.set_theme(theme)

    def set_theme(self, theme):
        self.theme = theme
        self.borderColor = r'rgb(70,70,70)'
        self.buttonColor = r'rgb(55,55,55)'
        self.borderRadius = r'3px'
        self.borderWidth = r'3px'
        self.borderStyle = r'double'
        self.borderStyle = r'solid'
        self.fontSize = '12pt'
        self.fontWeight = '100'
        self.fontColor = 'white'
        self.user32 = ctypes.windll.user32
        self.width = self.user32.GetSystemMetrics(0)
        self.height = self.user32.GetSystemMetrics(1)
        self.hand = 'hand2'
        self.ptr = 'left_ptr'
        self.smallFont = ('Verdana', 8)
        self.mediumFont = ('Verdana', 10)
        self.largeFont = ('Verdana', 12)
        self.entryBGValid = 'rgba(0,255,0,0.05)'
        self.entryBGValidHover = 'rgba(0,255,0,0.1)'
        self.entryBGConnected = 'rgba(0,0,255,0.05)'
        self.entryBGConnectedHover = 'rgba(0,0,255,0.1)'
        self.font = 'Arial'
        self.topPadding = 15
        self.bottomPadding = 10
        self.borderConnected = '{} {} rgb(10,10,150)'.format(self.borderWidth, self.borderStyle)
        self.borderConnectedHover = '{} {} rgb(50,50,255)'.format(self.borderWidth, self.borderStyle)
        self.borderValid = '{} {} rgba(0,75,0,1)'.format(self.borderWidth, self.borderStyle)
        self.borderValidHover = '{} {} rgba(0,100,0,1)'.format(self.borderWidth, self.borderStyle)
        self.borderDisconnected = '{} {} rgba(50,25,25,1)'.format(self.borderWidth, self.borderStyle)
        self.borderDisconnectedHover = '{} {} rgba(75, 45, 45, 1)'.format(self.borderWidth, self.borderStyle)
        self.borderError = '{} {} rgba(125,0,0,1)'.format(self.borderWidth, self.borderStyle)
        self.borderErrorHover = '{} {} rgba(170,0,0,1)'.format(self.borderWidth, self.borderStyle)
        self.borderAlert = '{} {} rgba(175,175,25,1)'.format(self.borderWidth, self.borderStyle)
        self.borderAlertHover = '{} {} rgba(200,200,50,1)'.format(self.borderWidth, self.borderStyle)

        _fontSize = 9
        if self.theme == 'dark':
            self.plotCanvas = r'0.1'
            self.plotBackground = r'0.1'
            self.backgroundColorDark = r'rgb(25,25,25)'
            self.backgroundColorLight = r'rgb(49,49,49)'
            self.backgroundColorLighter = r'rgb(79,79,79)'
            self.backgroundColorLightererer = r'rgb(92,92,92)'
            self.labelColor = r'rgb(25,25,25)'
            self.GRAPH_BACKGROUND_COLOR = r'rgb(92,92,92)'
            self.contrastColor = r'white'
            self.lightestGradient = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(80,80,80), stop: 0.5 rgb(120,120,120), stop: 0.5 rgb(110,110,110), stop: 1.0 rgb(90,90,90));'
            self.lightererGradient = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(80,80,80), stop: 0.5 rgb(120,120,120), stop: 0.5 rgb(110,110,110), stop: 1.0 rgb(90,90,90));'
            self.lighterGradient = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(40,40,40), stop: 0.5 rgb(80,80,80), stop: 0.5 rgb(70,70,70), stop: 1.0 rgb(50,50,50));'
            self.lightGradient = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(32,33,30), stop: 0.5 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
            self.darkGradient = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(22,23,20), stop: 0.5 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'
            self.fontColor = r'rgb(255,255,255)'
            self.darkGreenGradient = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(0,90,0), stop: 0.5 rgb(0,255,0), stop: 0.5 rgb(0,115,0), stop: 1.0 rgb(0,255,0));'
            self.lightGreenGradient = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(0,125,0), stop: 0.5 rgb(0,255,0), stop: 0.5 rgb(0,200,0), stop: 1.0 rgb(0,120,0));'
            self.darkRedGradient = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(23,0,0), stop: 0.5 rgb(75,0,0), stop: 0.5 rgb(90,0,0), stop: 1.0 rgb(50,0,0));'
            self.darkBlueGradient = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(0,0,23), stop: 0.5 rgb(0,0,75), stop: 0.5 rgb(0,0,90), stop: 1.0 rgb(0,0,50));'

        elif self.theme == 'default':
            self.backgroundColorDark = r'rgb(252,252,252)'
            self.backgroundColorLight = r'rgb(255,255,255)'
            self.backgroundColorLighter = r'rgb(240,240,240)'
            self.backgroundColorLightererer = r'rgb(255,255,255)'
            self.labelColor = r'rgb(255,255,255)'
            self.GRAPH_BACKGROUND_COLOR = r'rgb(255,255,255)'
            self.contrastColor = r'rgb(0,0,0)'
            self.lightGradient = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(222,222,222), stop: 0.5 rgb(248,250,247), stop: 0.5 rgb(253,255,252), stop: 1.0 rgb(232,232,232));'
            self.darkGradient = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(190,193,190), stop: 0.5 rgb(245,245,245), stop: 0.5 rgb(254,255,253), stop: 1.0 rgb(200,198,200));'
            self.fontColor = r'rgb(0,0,0)'

        elif self.theme == 'pink':
            self.backgroundColorDark = r'rgb(153,204,204)'
            self.backgroundColorLight = r'rgb(225,255,255)'
            self.backgroundColorLighter = r'rgb(240,240,240)'
            self.backgroundColorLightererer = r'rgb(235,248,249)'
            self.labelColor = r'rgb(255,255,255)'
            self.GRAPH_BACKGROUND_COLOR = r'rgb(255,255,255)'
            self.contrastColor = r'black'
            self.lightGradient = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(222,222,222), stop: 0.5 rgb(248,250,247), stop: 0.5 rgb(253,255,252), stop: 1.0 rgb(232,232,232));'
            self.darkGradient = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(190,193,190), stop: 0.5 rgb(245,245,245), stop: 0.5 rgb(254,255,253), stop: 1.0 rgb(200,198,200));'
            self.fontColor = r'rgb(0,0,0)'

        self.plotColors = {
            'red':   '#e27d60', 'blue': '#557a95', 'orange': '#e8a87c', 'purple': '#c38d9e',
            'green': '#afd275', 'pink': '#e7717d',
        }
        self.thrusterColor = 'white'


if theme == 'dark':
    BACKGROUND_COLOR_DARK = r'rgb(25,25,25)'
    BACKGROUND_COLOR_LIGHT = r'rgb(49,49,49)'
    BACKGROUND_COLOR_LIGHTER = r'rgb(79,79,79)'
    BACKGROUND_COLOR_LIGHTERER = r'rgb(92,92,92)'
    LABEL_COLOR = r'rgb(25,25,25)'
    GRAPH_BACKGROUND_COLOR = r'rgb(92,92,92)'
    CONTRASTED_COLOR = r'white'
    LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(32,33,30), stop: 0.5 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
    DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(22,23,20), stop: 0.5 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'
    FONT_COLOR = r'rgb(255,255,255)'
elif theme == 'default':
    BACKGROUND_COLOR_DARK = r'rgb(252,252,252)'
    BACKGROUND_COLOR_LIGHT = r'rgb(255,255,255)'
    BACKGROUND_COLOR_LIGHTER = r'rgb(240,240,240)'
    BACKGROUND_COLOR_LIGHTERER = r'rgb(255,255,255)'
    LABEL_COLOR = r'rgb(255,255,255)'
    GRAPH_BACKGROUND_COLOR = r'rgb(255,255,255)'
    CONTRASTED_COLOR = r'rgb(0,0,0)'
    LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(222,222,222), stop: 0.5 rgb(248,250,247), stop: 0.5 rgb(253,255,252), stop: 1.0 rgb(232,232,232));'
    DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(190,193,190), stop: 0.5 rgb(245,245,245), stop: 0.5 rgb(254,255,253), stop: 1.0 rgb(200,198,200));'
    FONT_COLOR = r'rgb(0,0,0)'
elif theme == 'pink':
    BACKGROUND_COLOR_DARK = r'rgb(153,204,204)'
    BACKGROUND_COLOR_LIGHT = r'rgb(225,255,255)'
    BACKGROUND_COLOR_LIGHTER = r'rgb(240,240,240)'
    BACKGROUND_COLOR_LIGHTERER = r'rgb(235,248,249)'
    LABEL_COLOR = r'rgb(255,255,255)'
    GRAPH_BACKGROUND_COLOR = r'rgb(255,255,255)'
    CONTRASTED_COLOR = r'black'
    LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(222,222,222), stop: 0.5 rgb(248,250,247), stop: 0.5 rgb(253,255,252), stop: 1.0 rgb(232,232,232));'
    DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(190,193,190), stop: 0.5 rgb(245,245,245), stop: 0.5 rgb(254,255,253), stop: 1.0 rgb(200,198,200));'
    FONT_COLOR = r'rgb(0,0,0)'
if theme == 'dark':
    CONTRASTED_COLOR = r'rgb(255,255,255)'
    LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(32,33,30), stop: 0.5 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
    DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(22,23,20), stop: 0.5 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'
    FONT_COLOR = r'rgb(255,255,255)'
elif theme == 'pink':
    CONTRASTED_COLOR = r'rgb(0,0,0)'
    LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(222,222,222), stop: 0.5 rgb(248,250,247), stop: 0.5 rgb(253,255,252), stop: 1.0 rgb(232,232,232));'
    DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(190,193,190), stop: 0.5 rgb(245,245,245), stop: 0.5 rgb(254,255,253), stop: 1.0 rgb(200,198,200));'
    FONT_COLOR = r'rgb(0,0,0)'
if theme == 'dark':
    PLOT_BACKGROUND = r'0.1'
    BACKGROUND_COLOR_DARK = r'rgb(25,25,25)'
    BACKGROUND_COLOR_LIGHT = r'rgb(49,49,49)'
    BACKGROUND_COLOR_LIGHTER = r'rgb(79,79,79)'
    BACKGROUND_COLOR_LIGHTERER = r'rgb(92,92,92)'
    LABEL_COLOR = r'rgb(25,25,25)'
    GRAPH_BACKGROUND_COLOR = r'rgb(92,92,92)'
    CONTRASTED_COLOR = r'white'
    LIGHTER_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.3, y2: -0.5, stop: 0.05 rgb(40,40,40), stop: 0.5 rgb(80,80,80), stop: 0.5 rgb(70,70,70), stop: 1.0 rgb(50,50,50));'
    LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.3, y2: -0.5, stop: 0.05 rgb(32,33,30), stop: 0.5 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
    DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.3, y2: -0.5, stop: 0.05 rgb(22,23,20), stop: 0.5 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'
    FONT_COLOR = r'rgb(255,255,255)'
elif theme == 'default':
    BACKGROUND_COLOR_DARK = r'rgb(252,252,252)'
    BACKGROUND_COLOR_LIGHT = r'rgb(255,255,255)'
    BACKGROUND_COLOR_LIGHTER = r'rgb(240,240,240)'
    BACKGROUND_COLOR_LIGHTERER = r'rgb(255,255,255)'
    LABEL_COLOR = r'rgb(255,255,255)'
    GRAPH_BACKGROUND_COLOR = r'rgb(255,255,255)'
    CONTRASTED_COLOR = r'rgb(0,0,0)'
    LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.3, y2: -0.5, stop: 0.05 rgb(222,222,222), stop: 0.5 rgb(248,250,247), stop: 0.5 rgb(253,255,252), stop: 1.0 rgb(232,232,232));'
    DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.3, y2: -0.5, stop: 0.05 rgb(190,193,190), stop: 0.5 rgb(245,245,245), stop: 0.5 rgb(254,255,253), stop: 1.0 rgb(200,198,200));'
    FONT_COLOR = r'rgb(0,0,0)'
elif theme == 'pink':
    BACKGROUND_COLOR_DARK = r'rgb(153,204,204)'
    BACKGROUND_COLOR_LIGHT = r'rgb(225,255,255)'
    BACKGROUND_COLOR_LIGHTER = r'rgb(240,240,240)'
    BACKGROUND_COLOR_LIGHTERER = r'rgb(235,248,249)'
    LABEL_COLOR = r'rgb(255,255,255)'
    GRAPH_BACKGROUND_COLOR = r'rgb(255,255,255)'
    CONTRASTED_COLOR = r'black'
    LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.3, y2: -0.5, stop: 0.05 rgb(222,222,222), stop: 0.5 rgb(248,250,247), stop: 0.5 rgb(253,255,252), stop: 1.0 rgb(232,232,232));'
    DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.3, y2: -0.5, stop: 0.05 rgb(190,193,190), stop: 0.5 rgb(245,245,245), stop: 0.5 rgb(254,255,253), stop: 1.0 rgb(200,198,200));'
    FONT_COLOR = r'rgb(0,0,0)'
