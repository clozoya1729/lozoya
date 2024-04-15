root = r'E:\github2\test\cubesat'
name = 'CubeSat'
thrusterColor = 'white'


class ConfigData:
    def __init__(self):
        self.delimiter = ','
        self.delimiterValid = True
        self.delimiterLock = True
        self.nVars = 1
        self.nVarsLock = False
        self.buffer = 10
        self.dataType = 't,f,f'
        self.dataTypeValid = True
        self.dataTypeLock = True
        self.head = ''
        self.headValid = False
        self.headLock = False
        self.units = ''
        self.unitsValid = False
        self.unitsLock = False
        self._bufferMax = 100


class ConfigDevice:
    def __init__(self):
        self.renderAxesQuiver = False
        self.renderCubeQuiver = True
        self.renderWireframe = False
        self.renderFaces = True
        self.renderThrusters = True
        self.renderAV = False
        self.x = 0
        self.y = 0
        self.z = 0
        self.xIndex = 0
        self.yIndex = 0
        self.zIndex = 0
        self.units = 'Degrees'
        self.unitsIndex = 0


class ConfigGenerator:
    def __init__(self):
        self.methods = 't,l,l'
        self.methodsValid = True
        self.methodsLock = True
        self.reliability = 100
        self.reliabilityLock = True
        self.scale = 1
        self.simulationEnabled = False


class ConfigLog:
    def __init__(self):
        self.autoUpdate = False
        self.loggingRate = 1000
        self.logCreationInterval = 60


class ConfigPlot:
    def __init__(self):
        self.dimensions = (1, 1, 1)
        self.plots = []
        self.randomAV = True
        self.fps = 1
        self.frames = 0
        self.ms = 10
        self.thrusterColor = 'black'
        self.reach = 1
        self.lims = (-self.reach, self.reach)
        self.lenLim = 20
        self.scale = 5
        self.wireframeWidth = 3
        self.axesQuiverWidth = 2
        self.cubeQuiverWidth = 5


class ConfigTransceiver:
    def __init__(self):
        self.baudRateIndex = 0
        self.baudRate = 9600
        self.portLock = False
        self.port = 'COM1'
        self.portIndex = 0
        self.bandIndex = 0
        self.band = 'Longwave'
        self.transmissionFrequency = 2
        self._transceiverActive = False


configurators = [
    ConfigData,
    ConfigDevice,
    ConfigGenerator,
    ConfigLog,
    ConfigPlot,
    ConfigTransceiver
]

# from utility.paths import IMG, ICONS
# import os
# # DATABASE
# ENCODING = None
# SEPARATOR = None
# LOG_SEPARATOR = ','
# HEADERS = []
#
# # STYLE (QSS)
# FONT_SIZE = 9
# FONT = 'Arial'
# topPadding = 15
# bottomPadding = 10
#
# # DEVICE
# ofcgTransmissionFrequency = 400
# ip = "129.108.152.32"
#
# # PLOT
# fps = 60
# dimensions = (1, 1, 1)
# reach = 1
# lims = (-reach, reach)
# ms = 10
#
# root = os.getcwd()
# # DATABASE
# db = os.path.join(root, 'database')
# configuration.py = os.path.join(root, 'configuration.py')
# saved = os.path.join(db, configuration.py, 'configuration.py')
# dataConfig = os.path.join(saved, 'data')
# deviceConfig = os.path.join(saved, 'device')
# encryptionConfig = os.path.join(saved, 'encryption')
# filterConfig = os.path.join(saved, 'filters')
# generatorConfig = os.path.join(saved, 'generator1')
# logConfig = os.path.join(saved, 'log')
# plotConfig = os.path.join(saved, 'plot0')
# transceiverConfig = os.path.join(saved, 'transceiver')
# transformConfig = os.path.join(saved, 'transforms')
# configPaths = {
#     'data'      : dataConfig, 'device': deviceConfig, 'encryption': encryptionConfig, 'filters': filterConfig,
#     'generator1': generatorConfig, 'log': logConfig, 'plot0': plotConfig, 'transceiver': transceiverConfig,
#     'transforms': transformConfig,
# }
# # INTERFACE
# interface = os.path.join(root, 'interface7')
# plots = os.path.join(interface, '../plots')
# img = os.path.join(interface, 'image')
# icons = os.path.join(img, 'icon2')
# bgs = os.path.join(img, 'background')
# # LOGGING
# logDirs = os.path.join(root, 'log')
# errorLog = os.path.join(logDirs, 'error_log')
# # UTILITY
# util = os.path.join(root, 'Utilities14')
# docs = os.path.join(root, '1', 'index.html')
# book = os.path.join(icons, 'book-close')
# help = os.path.join(icons, 'question-help-circle')
# cog = os.path.join(icons, 'cog')
# deviceExchange = os.path.join(icons, 'connect-device-exchange')
# medicalCross = os.path.join(icons, 'medical-app-smartphone')
# pin = os.path.join(icons, 'pin')
# battery = os.path.join(icons, 'charging-battery-full-1')
# sensor = os.path.join(icons, 'equipment-pressure-measure')
# downArrow = os.path.join(icons, 'download-arrow-1')
# refreshButton = os.path.join(icons, 'button-refresh-arrows')
# exportButton = os.path.join(icons, 'drawer-send')
# lockClosed = os.path.join(icons, 'lock-1')
# lockOpen = os.path.join(icons, 'lock-unlock')
# add = os.path.join(icons, 'add-circle-bold')
# connected = os.path.join(icons, 'flash-1')
# disconnected = os.path.join(icons, 'flash-off')
# resize = os.path.join(icons, 'expand-diagonal-2')
# bgImg = os.path.join(bgs, 'background.png')
# cSETRLogo = os.path.join(icons, 'csetr-logo.png')
# exit = os.path.join(icons, 'logout-1.png')
# openFolder = os.path.join(icons, 'openFolder.png')
# saveFile = os.path.join(icons, 'saveFile.png')
# settings = os.path.join(icons, 'navigation-menu')
# cube = os.path.join(icons, 'shape-cube')
# wifi = os.path.join(icons, 'wifi')
# bookShelf = os.path.join(icons, 'book-library-shelf')
# signal = os.path.join(icons, 'graph2-stats0')
# sendMsg = os.path.join(icons, 'send-email-2')
# # DATABASE
# DATABASE = os.path.join(root, 'database')
# USER_PREFERENCE = os.path.join(DATABASE, 'user_preference')
# # INTERFACE
# INTERFACE = os.path.join(root, 'interface7')
# PLOTS = os.path.join(INTERFACE, '../plots')
# IMG = os.path.join(INTERFACE, 'image')
# ICONS = os.path.join(IMG, 'icon2')
# # PLOTS
# VELOCITY_PLOT = os.path.join(PLOTS, 'VelocityPlot.html')
# ACCELERATION_PLOT = os.path.join(PLOTS, 'AccelerationPlot.html')
# EPSCURRENT_PLOT = os.path.join(PLOTS, 'EPSCurrentPlot.html')
# EPSVOLTAGE_PLOT = os.path.join(PLOTS, 'EPSVoltagePlot.html')
# TEMPERATURE_PLOT = os.path.join(PLOTS, 'TemperaturePlot.html')
# PRESSURE_PLOT = os.path.join(PLOTS, 'PressurePlot.html')
# # LOGGING
# logDirs = os.path.join(root, 'log')
# # UTILITY
# UTILITY = os.path.join(root, 'Utilities14')
# DOCUMENTATION = os.path.join(root, '1', 'index.html')
#
# root = os.getcwd()
#
# # DATABASE
# db = os.path.join(root, 'data')
# configuration.py = os.path.join(root, 'config0')
# saved = os.path.join(db, configuration.py, 'saved')
# dataConfig = os.path.join(saved, 'data')
# deviceConfig = os.path.join(saved, 'device')
# encryptionConfig = os.path.join(saved, 'encryption')
# filterConfig = os.path.join(saved, 'filters')
# generatorConfig = os.path.join(saved, 'generator')
# logConfig = os.path.join(saved, 'log')
# plotConfig = os.path.join(saved, 'plot')
# transceiverConfig = os.path.join(saved, 'transceiver')
# transformConfig = os.path.join(saved, 'transforms')
# configPaths = {
#     'data'      : dataConfig, 'device': deviceConfig, 'encryption': encryptionConfig, 'filters': filterConfig,
#     'generator' : generatorConfig, 'log': logConfig, 'plot': plotConfig, 'transceiver': transceiverConfig,
#     'transforms': transformConfig,
# }
#
# # INTERFACE
# interface = os.path.join(root, '../interface0')
# plots = os.path.join(interface, 'plots')
# img = os.path.join(interface, 'img')
# icons = os.path.join(img, 'icons')
# bgs = os.path.join(img, 'background')
#
# # LOGGING
# logDirs = os.path.join(root, 'logs0')
# errorLog = os.path.join(logDirs, 'error_log')
#
# # UTILITY
# util = os.path.join(root, 'utility0')
# docs = os.path.join(root, 'documentation2', 'index.html')
# book = os.path.join(icons, 'book-close')
# help = os.path.join(icons, 'question-help-circle')
# cog = os.path.join(icons, 'cog')
# deviceExchange = os.path.join(icons, 'connect-device-exchange')
# medicalCross = os.path.join(icons, 'medical-app-smartphone')
# pin = os.path.join(icons, 'pin')
# battery = os.path.join(icons, 'charging-battery-full-1')
# sensor = os.path.join(icons, 'equipment-pressure-measure')
# downArrow = os.path.join(icons, 'download-arrow-1')
# refreshButton = os.path.join(icons, 'button-refresh-arrows')
# exportButton = os.path.join(icons, 'drawer-send')
# lockClosed = os.path.join(icons, 'lock-1')
# lockOpen = os.path.join(icons, 'lock-unlock')
# add = os.path.join(icons, 'add-circle-bold')
# connected = os.path.join(icons, 'flash-1')
# disconnected = os.path.join(icons, 'flash-off')
# resize = os.path.join(icons, 'expand-diagonal-3')
# bgImg = os.path.join(bgs, 'background.png')
# cSETRLogo = os.path.join(icons, 'csetr-logo.png')
# exit = os.path.join(icons, 'logout-2.png')
# openFolder = os.path.join(icons, 'openFolder.png')
# saveFile = os.path.join(icons, 'saveFile.png')
# settings = os.path.join(icons, 'navigation-menu')
# cube = os.path.join(icons, 'shape-cube')
# wifi = os.path.join(icons, 'wifi')
# bookShelf = os.path.join(icons, 'book-library-shelf')
# signal = os.path.join(icons, 'graph-stats')
# sendMsg = os.path.join(icons, 'send-email-3')
#
# ROOT = os.getcwd()
#
# # INTERFACE
# INTERFACE = os.path.join(ROOT, 'interface7')
# PLOTS = os.path.join(INTERFACE, 'plots')
# IMG = os.path.join(INTERFACE, 'img')
# ICONS = os.path.join(IMG, 'icons')
# LABELS = os.path.join(INTERFACE, 'labels', 'headers.txt')
# DATABASE_VARIABLES = os.path.join(INTERFACE, 'labels', 'database_variables.txt')
#
# # PLOTS
# BASE_GRAPH = os.path.join(PLOTS, 'RegressionPlot.html')
# REGRESSION_PLOT = os.path.join(PLOTS, 'RegressionPlot.html')
# SEARCH_RESULTS_TABLE = os.path.join(PLOTS, 'SearchResults2.html')
# VELOCITY_PLOT = os.path.join(PLOTS, 'VelocityPlot.html')
# ACCELERATION_PLOT = os.path.join(PLOTS, 'AccelerationPlot.html')
# EPSCURRENT_PLOT = os.path.join(PLOTS, 'EPSCurrentPlot.html')
# EPSVOLTAGE_PLOT = os.path.join(PLOTS, 'EPSVoltagePlot.html')
# TEMPERATURE_PLOT = os.path.join(PLOTS, 'TemperaturePlot.html')
# PRESSURE_PLOT = os.path.join(PLOTS, 'PressurePlot.html')
#
# # LOGGING
# LOG_DIRS = os.path.join(ROOT, 'logs')
# IMU_LOG = os.path.join(LOG_DIRS, 'IMU Status')
# EPS_LOG = os.path.join(LOG_DIRS, 'EPS Status')
# SENSOR_LOG = os.path.join(LOG_DIRS, 'Sensor Status')
#
# # UTILITY
# UTILITY = os.path.join(ROOT, 'Utilities14')
# VARIABLES = os.path.join(UTILITY, 'configuration.py')
# tempSETTINGS = os.path.join(UTILITY, 'Preferences', 'temp_settings.txt')
# SETTINGS = os.path.join(UTILITY, 'Preferences', 'settings.txt')
#
# DATABASE = os.path.join(ROOT, 'database')
# TEMP_SAVE = os.path.join(DATABASE, 'temp.txt')
#
# DOCUMENTATION = os.path.join(ROOT, '1', 'index.html')
#
# # SHORTCUTS
# device = 'Ctrl+D'
# transceiver = 'Ctr+T'
# plot = 'Ctrl+P'
# log = 'Ctrl+L'
# settings = 'Ctrl+S'
#
# # ICONS
# COLD_GAS_ICON = os.path.join(ICONS, 'cold-gas.png')
# BACKGROUND_IMG = os.path.join(IMG, 'background.png')
# WINDOW_IMG = os.path.join(IMG, 'csetr-logo.png')
# CUBE_ICON = os.path.join(ICONS, '12-Design', '06-Shapes', '512w', 'shape-cube')
# DOWN_ARROW_ICON = os.path.join(IMG, 'downArrow.png')
# EXIT_ICON = os.path.join(IMG, 'exit.png')
# OPEN_FILE_ICON = os.path.join(IMG, 'openFile.png')
# OPEN_FOLDER_ICON = os.path.join(IMG, 'openFolder.png')
# SAVE_FILE_ICON = os.path.join(IMG, 'saveFile.png')
# SAVE_FOLDER_ICON = os.path.join(IMG, 'saveFolder.png')
# DOCUMENTATION_ICON = os.path.join(ICONS, '11-Content', '02-Books', '48w', 'book-close')
# TRANSCEIVER_ICON = os.path.join(ICONS, '05-Internet-Networks-Servers', '02-Wifi', '512w', 'wifi')
# HELP_ICON = os.path.join(ICONS, '01-Interface18 Essential', '16-Help5', '512w', 'question-help-circle')
# DEVICE_ICON = os.path.join(ICONS, '20-Phones-Mobile-Devices', '15-Connect-Devices', '512w', 'connect-device-exchange')
# OBC_ICON = os.path.join(ICONS, '35-Health-Beauty', '08-Medical-Apps', '512w', 'medical-app-smartphone')
# IMU_ICON = os.path.join(ICONS, '01-Interface18 Essential', '17-stats0', '512w', 'graph-stats0')
# PIN_ICON = os.path.join(ICONS, '01-Interface18 Essential', '31-Pin', '96w', 'pin')
# SETTINGS_ICON = os.path.join(ICONS, '01-Interface18 Essential', '03-Menu', '48w', 'navigation-menu-vertical')
# LOG_ICON = os.path.join(ICONS, '11-Content', '02-Books', '512w', 'book-library-shelf')
# EPS_ICON = os.path.join(ICONS, '20-Phones-Mobile-Devices', '17-Charging-Battery', '512w', 'charging-battery-full-1')
# SENSOR_ICON = os.path.join(ICONS, '30-Tools-Construction', '08-Equipment', '512w', 'equipment-pressure-measure')
#
# # STATUS MESSAGES
# # LOG
# logNotLinked = 'Selected interface7 is not linked to transceiver.'
# autoUpdateDisabled = ''
# # SUCCESS
# createdFile = 'Created file {}.', 'success'
# enabledTransceiver = 'Transceiver has been enabled.', 'success'
# disconnectedPort = 'Disconnected from port {}.', 'success'
# connectedPort = 'Connected to port {}.', 'success'
# exportLog = 'Exported log to {}.', 'success'
# autoUpdateOn = 'Enabled automatic update.', 'success'
# autoUpdateOff = 'Disabled automatic update.', 'success'
# # ALERT
# disabledTransceiver = 'Transceiver has been disabled.', 'alert'
# notFullyLoaded = 'Must connect all parameters in settings menu before enabling transceiver', 'alert'
# emptyDelimiter = 'Delimiter cannot be empty.', 'alert'
# exitLinkedLog = 'Disabled automatic Update. Selected interface7 is not linked to transceiver.', 'alert'
# exportLogCancel = 'Canceled export.', 'alert'
# # ERROR
# createLogDirError = 'Create log directory error: {}', 'error'
# createLogFileError = 'Create log file error: {}', 'error'
# writeLogError = 'Create log directory error: {}', 'error'
# transmissionError = 'Transmission error: {}.', 'error'
# generateError = 'Generate number error: {}.', 'error'
# simulateError = 'Simulate signal error: {}.', 'error'
# connectionError = 'Connection error: {}.', 'error'
# connectPortError = 'Connect port error: {}.', 'error'
# updateEncryptionError = 'Update encryption error: {}.', 'error'
# setPortError = 'Set port error: {}.', 'error'
# encryptionError = 'Encryption error: {}.', 'error'
# xorError = 'XOR(0) encryption error: {}.', 'error'
# exportLogError = 'Export log error: {}.', 'error'
# formatDataError = 'Format data error: {}.', 'error'
# logMaintenanceError = 'Log maintenance error: {}.', 'error'
# updateLogListError = 'Update log list error: {}.', 'error'
# updateDireListError = 'Update directory list error: {}.', 'error'
# refreshLogError = 'Refresh log error: {}.', 'error'
# toggleAutomaticUpdateError = 'Toggle automatic update error: {}.', 'error'
#
# # LOG
# logNotLinked = 'Selected menus0 is not linked to transceiver.'
# autoUpdateDisabled = ''
#
# # SUCCESS
# createdFile = 'Created file {}.', 'success'
# enabledTransceiver = 'Transceiver has been enabled.', 'success'
# disconnectedPort = 'Disconnected from port {}.', 'success'
# connectedPort = 'Connected to port {}.', 'success'
# exportLog = 'Exported log to {}.', 'success'
# autoUpdateOn = 'Enabled automatic update.', 'success'
# autoUpdateOff = 'Disabled automatic update.', 'success'
# windowPinned = 'Window pinned.', 'success'
#
# # ALERT
# disabledTransceiver = 'Transceiver has been disabled.', 'alert'
# notFullyLoaded = 'Must connect all parameters in settings menu before enabling transceiver', 'alert'
# emptyDelimiter = 'Delimiter cannot be empty.', 'alert'
# exitLinkedLog = 'Disabled automatic Update. Selected menus0 is not linked to transceiver.', 'alert'
# exportLogCancel = 'Canceled export.', 'alert'
# windowUnpinned = 'Window unpinned.', 'alert'
#
# # ERROR
# createLogDirError = 'Create log directory error: {}', 'error'
# createLogFileError = 'Create log file error: {}', 'error'
# writeLogError = 'Create log directory error: {}', 'error'
# transmissionError = 'Transmission error: {}.', 'error'
# generateError = 'Generate number error: {}.', 'error'
# simulateError = 'Simulate signal error: {}.', 'error'
# connectionError = 'Connection error: {}.', 'error'
# connectPortError = 'Connect port error: {}.', 'error'
# updateEncryptionError = 'Update encryption error: {}.', 'error'
# setPortError = 'Set port error: {}.', 'error'
# encryptionError = 'Encryption error: {}.', 'error'
# xorEncryptionError = 'XOR(0) encryption error: {}.', 'error'
# exportLogError = 'Export log error: {}.', 'error'
# formatDataError = 'Format data error: {}.', 'error'
# logMaintenanceError = 'Log maintenance error: {}.', 'error'
# updateLogListError = 'Update log list error: {}.', 'error'
# updateDirListError = 'Update directory list error: {}.', 'error'
# refreshLogError = 'Refresh log error: {}.', 'error'
# toggleAutomaticUpdateError = 'Toggle automatic update error: {}.', 'error'
# getDataError = 'Get data error: {}.', 'error'
# dataExportError = 'Data export error: {}.', 'error'
# updateDataError = 'Data update error: {}.', 'error'
# initDataDudeError = 'Init DataDude error: {}.', 'error'
# integrationError = 'Integration error: {}.', 'error'
# differentiationError = 'Differentiation error: {}.', 'error'
# fftError = 'Fast Fourier Transform error: {}.', 'error'
# initPlotDudeError = 'Init PlotDude error: {}.', 'error'
# addPlotError = 'Add plot error: {}.', 'error'
# initAnimated2DPlotError = 'Init animated 2D plot error: {}.', 'error'
# plotUpdateError = 'Plot update error: {}.', 'error'
# updateAnimationError = 'Update animation error: {}.', 'error'
# renderCubeError = 'Render cube error: {}.', 'error'
# initDeviceMenuError = 'Init device menu error: {}.', 'error'
# formatDevicePlot = 'Format device plot error: {}.', 'error'
#
# # TOOLTIPS
# # DEVICE MENU
# renderFaces = 'Toggle whether the cube\n' \
#               'faces are rendered.'
# renderWireframe = 'Toggle whether the cube\n' \
#                   'wireframe is rendered.'
# renderThrusters = 'Toggle whether the thrusters\n' \
#                   'are rendered.'
# renderCubeQuiver = 'Toggle whether the cube\n' \
#                    'quiver is rendered.'
# renderAxesQuiver = 'Toggle whether the axes\n' \
#                    'quiver is rendered.'
# pinButton = 'Pin/unpin window.\n' \
#             'Pinned window will remain\n' \
#             'above unpinned windows and\n' \
#             'search applications.'
# posZCombo = 'Select a variable from the\n' \
#             'data sent by the CubeSAT to\n' \
#             'use for plotting the position\n' \
#             'of the CubeSAT.'
#
# # PLOT
# plotRange = 'Left slider sets minimum y-value.\n' \
#             'Right slider sets maximum y-value.'
# minYSlider = 'Sets minimum y-value.'
# maxYSlider = 'Sets maximum y-value.'
# plotBuffer = 'Set the number of samples.'
#
# # TRANSCEIVER
# bandCombo = 'Set the frequency band.'
# baudRateCombo = 'Set the baud rate.'
# encryptionCombo = 'Set which encryption method to use.'
# frequencySlider = 'Set the transmission frequency.'
# msgSend = 'Type a message for the CubeSAT.\n' \
#           'Click the Send button to send it.'
# msgReceive = 'Messages received from the CubeSAT appear here.'
# msgSendButton = 'Click to send message.'
# dataTypeEntry = 'Enter the data type of each variable\n' \
#                 'in the data that the CubeSAT will send\n' \
#                 'as a comma-separated list.\n' \
#                 'd = Integer\n' \
#                 'f = Decimal\n' \
#                 'h = Hexadecimal\n' \
#                 's = String\n' \
#                 't = Time'
# dataTypeLockBtn = 'Toggle the data type text box\n' \
#                   'between editable and read-only.'
# headEntry = 'Enter the name of each variable\n' \
#             'in the data that the CubeSAT will send\n' \
#             'as a comma-separated list.'
# headLockBtn = 'Toggle the head text box\n' \
#               'between editable and read-only.'
# # LOG MENU
# autoUpdateButton = 'Set whether to auto-scroll the\n' \
#                    'log reader when new data is received.'
# exportButton = 'Export data to Excel (.xlsx).'
# refreshButton = 'Refresh the log reader with\n' \
#                 'data received since last refresh.'
#
# # DEVICE MENU
# renderFaces = 'Toggle whether the cube\n' \
#               'faces are rendered.'
# renderWireframe = 'Toggle whether the cube\n' \
#                   'wireframe is rendered.'
# renderThrusters = 'Toggle whether the thrusters\n' \
#                   'are rendered.'
# renderCubeQuiver = 'Toggle whether the cube\n' \
#                    'quiver is rendered.'
# renderAxesQuiver = 'Toggle whether the axes\n' \
#                    'quiver is rendered.'
# renderAV = 'Toggle whether the angular\n' \
#            'velocity vector is rendered\n' \
#            'on the device plot0.'
# pinButton = 'Pin/unpin window.\n' \
#             'Pinned window will remain\n' \
#             'above unpinned windows and\n' \
#             'search applications.'
# posCombo = 'Select a variable from the\n' \
#            'data sent by the CubeSAT to\n' \
#            'use for plotting the position\n' \
#            'of the CubeSAT.'
# # PLOT
# plotRange = 'Left slider sets minimum y-value.\n' \
#             'Right slider sets maximum y-value.'
# minYSlider = 'Sets minimum y-value.'
# maxYSlider = 'Sets maximum y-value.'
# domainSlider = 'Set the minimim x-value.'
# # TRANSCEIVER
# bandCombo = 'Set the frequency band.'
# baudRateCombo = 'Set the baud rate.'
# encryptionCombo = 'Set which encryption method to use.'
# frequencySlider = 'Set the transmission frequency.'
# msgSend = 'Type a message for the CubeSAT.\n' \
#           'Click the Send button to send it.'
# msgReceive = 'Messages received from the CubeSAT appear here.'
# msgSendButton = 'Click to send message.'
# nVarsInput = 'Enter the number of variables the\n' \
#              'CubeSAT will be sending to the transceiver.'
# nVarsLkBtn = 'Toggle the number of variables input box\n' \
#              'between editable and read-only.'
# delimiterEntry = 'Enter the delimiter the CubeSAT will use\n' \
#                  'to format the data it sends to the transceiver.'
# delimiterLkBtn = 'Toggle the head text box\n' \
#                  'between editable and read-only.'
# dataTypeEntry = 'Enter the data type of each variable\n' \
#                 'in the data that the CubeSAT will send\n' \
#                 'as a list separated by the delimiter.\n' \
#                 'd = Integer\n' \
#                 'f = Decimal\n' \
#                 'h = Hexadecimal\n' \
#                 's = String\n' \
#                 't = Time'
# dataTypeLkBtn = 'Toggle the data type text box\n' \
#                 'between editable and read-only.'
# headEntry = 'Enter the name of each variable\n' \
#             'in the data that the CubeSAT will send\n' \
#             'as a list separated by the delimiter.'
# headLkBtn = 'Toggle the head text box\n' \
#             'between editable and read-only.'
# unitsEntry = 'Enter the unit of each variable\n' \
#              'in the data that the CubeSAT will send\n' \
#              'as a list separated by the delimiter.'
# unitsLkBtn = 'Toggle the units text box\n' \
#              'between editable and read-only.'
# buffer = 'Set the number of samples.'
# # LOG MENU
# autoUpdateButton = 'Set whether to auto-scroll the\n' \
#                    'interface7 reader when new data is received.'
# exportButton = 'Export data to Excel (.xlsx).'
# refreshButton = 'Refresh the interface7 reader with\n' \
#                 'data received since last refresh.'
# loggingRate = 'Enter the number of samples to\n' \
#               'be skipped by the logger.'
# logCreationInterval = 'Enter the number of seconds that\n' \
#                       'a interface7 will be written to before a\n' \
#                       'new file is created.'
#
# # DEVICE MENU
# renderFaces = 'Toggle whether the cube\n' \
#               'faces are rendered.'
# renderWireframe = 'Toggle whether the cube\n' \
#                   'wireframe is rendered.'
# renderThrusters = 'Toggle whether the thrusters\n' \
#                   'are rendered.'
# renderCubeQuiver = 'Toggle whether the cube\n' \
#                    'quiver is rendered.'
# renderAxesQuiver = 'Toggle whether the axes\n' \
#                    'quiver is rendered.'
# renderAV = 'Toggle whether the angular\n' \
#            'velocity vector is rendered\n' \
#            'on the device plot.'
# pinButton = 'Pin/unpin window.\n' \
#             'Pinned window will remain\n' \
#             'above unpinned windows and\n' \
#             'other applications.'
# posCombo = 'Select a variable from the\n' \
#            'data sent by the CubeSAT to\n' \
#            'use for plotting the position\n' \
#            'of the CubeSAT.'
#
# # PLOT
# plotRange = 'Left slider sets minimum y-value.\n' \
#             'Right slider sets maximum y-value.'
# minYSlider = 'Sets minimum y-value.'
# maxYSlider = 'Sets maximum y-value.'
# domainSlider = 'Set the minimim x-value.'
#
# # TRANSCEIVER
# bandCombo = 'Set the frequency band.'
# baudRateCombo = 'Set the baud rate.'
# encryptionCombo = 'Set which encryption method to use.'
# frequencySlider = 'Set the transmission frequency.'
# msgSend = 'Type a message for the CubeSAT.\n' \
#           'Click the Send button to send it.'
# msgReceive = 'Messages received from the CubeSAT appear here.'
# msgSendButton = 'Click to send message.'
# nVarsInput = 'Enter the number of variables the\n' \
#              'CubeSAT will be sending to the transceiver.'
# nVarsLkBtn = 'Toggle the number of variables input box\n' \
#              'between editable and read-only.'
# delimiterEntry = 'Enter the delimiter the CubeSAT will use\n' \
#                  'to format the data it sends to the transceiver.'
# delimiterLkBtn = 'Toggle the head text box\n' \
#                  'between editable and read-only.'
# dataTypeEntry = 'Enter the data type of each variable\n' \
#                 'in the data that the CubeSAT will send\n' \
#                 'as a list separated by the delimiter.\n' \
#                 'd = Integer\n' \
#                 'f = Decimal\n' \
#                 'h = Hexadecimal\n' \
#                 's = String\n' \
#                 't = Time'
# dataTypeLkBtn = 'Toggle the data type text box\n' \
#                 'between editable and read-only.'
# headEntry = 'Enter the name of each variable\n' \
#             'in the data that the CubeSAT will send\n' \
#             'as a list separated by the delimiter.'
# headLkBtn = 'Toggle the head text box\n' \
#             'between editable and read-only.'
# unitsEntry = 'Enter the unit of each variable\n' \
#              'in the data that the CubeSAT will send\n' \
#              'as a list separated by the delimiter.'
# unitsLkBtn = 'Toggle the units text box\n' \
#              'between editable and read-only.'
# buffer = 'Set the number of samples.'
# # LOG MENU
# autoUpdateButton = 'Set whether to auto-scroll the\n' \
#                    'menus0 reader when new data is received.'
# exportButton = 'Export data to Excel (.xlsx).'
# refreshButton = 'Refresh the menus0 reader with\n' \
#                 'data received since last refresh.'
# loggingRate = 'Enter the number of samples to\n' \
#               'be skipped by the logger.'
# logCreationInterval = 'Enter the number of seconds that\n' \
#                       'a menus0 will be written to before a\n' \
#                       'new file is created.'
# logDirSelector = 'Select date.'
# logFileSelector = 'Select file.'
# logReader = 'Log data.'
#
# # MAIN MENU
# CLASSIFICATION_MENU_DESCRIPTION = ''
# DATABASE_MENU_DESCRIPTION = ''
# DOCUMENTATION_MENU_DESCRIPTION = ''
# REGRESSION_MENU_DESCRIPTION = ''
# SEARCH_MENU_DESCRIPTION = ''
# SIMULATION_MENU_DESCRIPTION = ''
#
# # GENERAL
# OPEN_FILE_DESCRIPTION = ''
# OPEN_FOLDER_DESCRIPTION = ''
# SAVE_FILE_DESCRIPTION = ''
# SAVE_FOLDER_DESCRIPTION = ''
#
# # DATABASE MENU
# DATABASE_ENCODING_DESCRIPTION = ''
# DATABASE_INDEX_DESCRIPTION = ''
# DATABASE_LONGITUDE_DESCRIPTION = ''
# DATABASE_LATITUDE_DESCRIPTION = ''
#
# # EXPORTING
# EXPORT_EXCEL_DESCRIPTION = ''
# EXPORT_CSV_DESCRIPTION = ''
#
# # MACHINE LEARNING
# AUTOMATIC_TUNING_DESCRIPTION = ''
#
# # REGRESSION MENU
# INDEPENDENT_VARIABLE_DESCRIPTION = ''
# DEPENDENT_VARIABLE_DESCRIPTION = ''
# REGRESSION_RUN_DESCRIPTION = ''
#
# # SEARCH MENU
# SEARCH_RUN_DESCRIPTION = ''
#
# # SIMULATION MENU
# INDEPENDENT_VARIABLE_DESCRIPTION = ''
# DEPENDENT_VARIABLE_DESCRIPTION = ''
# MARKOV_CHAIN_INITIAL_STATE_DESCRIPTION = ''
# LATIN_HYPERCUBE_ITERATIONS_DESCRIPTION = ''
# MONTE_CARLO_ITERATIONS_DESCRIPTION = ''
#
# SAVE_BUTTON_HOVER_TEXT = 'Select directory where logs will be saved.'
# SAVE_LABEL_HOVER_TEXT = 'Logs are being saved to {}'
#
# # DEVICE MENU
# renderFaces = 'Toggle whether the cube\n' \
#               'faces are rendered.'
# renderWireframe = 'Toggle whether the cube\n' \
#                   'wireframe is rendered.'
# renderThrusters = 'Toggle whether the thrusters\n' \
#                   'are rendered.'
# renderCubeQuiver = 'Toggle whether the cube\n' \
#                    'quiver is rendered.'
# renderAxesQuiver = 'Toggle whether the axes\n' \
#                    'quiver is rendered.'
# pinButton = 'Pin/unpin window.\n' \
#             'Pinned window will remain\n' \
#             'above unpinned windows and\n' \
#             'other applications.'
# posZCombo = 'Select a variable from the\n' \
#             'data sent by the CubeSAT to\n' \
#             'use for plotting the position\n' \
#             'of the CubeSAT.'
#
# # PLOT
# plotRange = 'Left slider sets minimum y-value.\n' \
#             'Right slider sets maximum y-value.'
# minYSlider = 'Sets minimum y-value.'
# maxYSlider = 'Sets maximum y-value.'
# domainSlider = 'Set the minimim x-value.'
#
# # TRANSCEIVER
# bandCombo = 'Set the frequency band.'
# baudRateCombo = 'Set the baud rate.'
# encryptionCombo = 'Set which encryption method to use.'
# frequencySlider = 'Set the transmission frequency.'
# msgSend = 'Type a message for the CubeSAT.\n' \
#           'Click the Send button to send it.'
# msgReceive = 'Messages received from the CubeSAT appear here.'
# msgSendButton = 'Click to send message.'
# nVarsInput = 'Enter the number of variables the\n' \
#              'CubeSAT will be sending to the transceiver.'
# nVarsLockBtn = 'Toggle the number of variables input box\n' \
#                'between editable and read-only.'
# delimiterEntry = 'Enter the delimiter the CubeSAT will use\n' \
#                  'to format the data it sends to the transceiver.'
# delimiterLockBtn = 'Toggle the head text box\n' \
#                    'between editable and read-only.'
# dataTypeEntry = 'Enter the data type of each variable\n' \
#                 'in the data that the CubeSAT will send\n' \
#                 'as a list separated by the delimiter.\n' \
#                 'd = Integer\n' \
#                 'f = Decimal\n' \
#                 'h = Hexadecimal\n' \
#                 's = String\n' \
#                 't = Time'
# dataTypeLockBtn = 'Toggle the data type text box\n' \
#                   'between editable and read-only.'
# headEntry = 'Enter the name of each variable\n' \
#             'in the data that the CubeSAT will send\n' \
#             'as a list separated by the delimiter.'
# headLockBtn = 'Toggle the head text box\n' \
#               'between editable and read-only.'
# unitsEntry = 'Enter the unit of each variable\n' \
#              'in the data that the CubeSAT will send\n' \
#              'as a list separated by the delimiter.'
# unitsLockBtn = 'Toggle the units text box\n' \
#                'between editable and read-only.'
# buffer = 'Set the number of samples.'
# # LOG MENU
# autoUpdateButton = 'Set whether to auto-scroll the\n' \
#                    'log reader when new data is received.'
# exportButton = 'Export data to Excel (.xlsx).'
# refreshButton = 'Refresh the log reader with\n' \
#                 'data received since last refresh.'

# from interface import icons
# from utility import paths, shortcuts
# from utility.interface import tooltips
#
#
# class device:
#     posXCombo = dict(text='x', tooltip=tooltips.posZCombo, )
#     posYCombo = dict(text='y', tooltip=tooltips.posZCombo, )
#     posZCombo = dict(text='z', tooltip=tooltips.posZCombo, )
#     deviceContainer = dict(text='Device', row=0, column=0, )
#     deviceSettings = dict(text='Settings', height=100, row=0, column=0, )
#     renderWireframeCheckBox = dict(text='Wireframe', label=True, row=0, column=2, tooltip=tooltips.renderWireframe, )
#     renderThrustersCheckBox = dict(text='Thrusters', label=True, row=0, column=4, tooltip=tooltips.renderThrusters, )
#     renderAxesQuiverCheckBox = dict(text='Axes Quiver', label=True, row=1, column=0,
#                                     tooltip=tooltips.renderAxesQuiver, )
#     renderCubeQuiverCheckBox = dict(label=True, row=1, column=2, tooltip=tooltips.renderCubeQuiver, )
#
#
# class log:
#     # LABEL
#     activeLogLabel = dict(row=0, column=2, width=250, )
#     # BUTTON
#     autoUpdateButton = dict(row=2, column=0, width=25, icon=icons.downArrow, tooltip=tooltips.autoUpdateButton, )
#     exportButton = dict(row=0, column=0, width=25, icon=icons.exportButton, tooltip=tooltips.exportButton)
#     refreshButton = dict(row=1, column=0, width=25, icon=icons.refreshButton, tooltip=tooltips.refreshButton, )
#     # GRID
#     logDirsGrid = dict(text='Log Directory', row=1, column=0, width=150, )
#     logsGrid = dict(text='Log', row=1, column=1, width=150, )
#     logReaderGrid = dict(text='', row=1, column=2, )
#     toolsGrid = dict(text='', row=1, column=3, )
#
#
# class main:
#     deviceToolBar = dict(text='Device', icon=icons.cube, shortcut=shortcuts.device, )
#     transceiverToolbar = dict(icon=icons.wifi, text='Transceiver', shortcut=shortcuts.transceiver, )
#     plotToolbar = dict(icon=icons.imu, text='Plot', shortcut=shortcuts.plot, )
#     logToolbar = dict(icon=icons.bookShelf, text='Log', shortcut=shortcuts.log, )
#     deviceMenu = dict(title='Device', icon=icons.cube, )
#     documentationMenu = dict(title='Documentation', icon=icons.book, )
#     logMenu = dict(title='Log', icon=icons.bookShelf, )
#     plotMenu = dict(title='Plot', icon=icons.imu, )
#     settingsMenu = dict(title='Settings', icon=icons.settings, )
#     transceiverMenu = dict(title='Transceiver', icon=icons.wifi, )
#
#
# class plot:
#     toolsGrid = dict(text='', row=4, column=3, )
#     yCombo = dict(text='y', row=2, label=True, column=0, )
#
#
# class sensor:
#     pressureContainer = dict(text='Pressure', row=0, column=0, )
#     temperatureContainer = dict(text='Temperature', row=0, column=0, )
#     pressureSettings = dict(text='Settings', height=100, row=0, column=0, )
#     temperatureSettings = dict(text='Settings', height=100, row=0, column=0, )
#     pressureLimYLabel = dict(text='Range', width=60, row=0, column=0, tooltip=tooltips.plotRange, )
#     pressureMinYEntry = dict(width=50, row=0, column=1, tooltip=tooltips.minYSlider, )
#     pressureMaxYEntry = dict(width=50, row=0, column=2, tooltip=tooltips.maxYSlider, )
#     pressureBuffLabel = dict(text='Buffer', width=60, row=1, column=0, tooltip=tooltips.plotBuffer, )
#     pressureBuffEntry = dict(width=50, row=1, column=1, tooltip=tooltips.plotBuffer, )
#     temperatureLimYLabel = dict(text='Range', width=60, row=0, column=0, tooltip=tooltips.plotRange, )
#     temperatureMinYEntry = dict(width=50, row=0, column=1, tooltip=tooltips.minYSlider, )
#     temperatureMaxYEntry = dict(width=50, row=0, column=2, tooltip=tooltips.maxYSlider, )
#     temperatureBuffLabel = dict(text='Buffer', width=60, row=1, column=0, tooltip=tooltips.plotBuffer, )
#     temperatureBuffEntry = dict(width=50, row=1, column=1, tooltip=tooltips.plotBuffer, )
#
#
# class settings:
#     # DATA
#     dataTypeEntry = dict(text='Data Types', row=0, column=0,  # paidWidth=100,
#                          tooltip=tooltips.dataTypeEntry, )
#     dataTypeLockBtn = dict(width=25, row=0, column=1, tooltip=tooltips.dataTypeLockBtn, )
#     headEntry = dict(text='Column Names', row=1, column=0,  # paidWidth=100,
#                      tooltip=tooltips.headEntry, )
#     headLockBtn = dict(width=25, row=0, column=1, tooltip=tooltips.headLockBtn, )
#     # LOG
#     logSettingsGroupBox = dict(text='Log', row=2, column=0, )
#     loggingRateSpinBox = dict(text='Logging Rate', row=0, column=0, )
#     logCreationIntervalSpinBox = dict(text='Log Creation Interval (Seconds)', row=0, column=1, )
#     # DEVICE
#     deviceSettingsGroupBox = dict(text='Device', row=1, column=0, )
#     # TRANSCEIVER
#     transceiverSettingsGroupBox = dict(text='Transceiver', row=3, column=0, )
#     encryptionCombo = dict(text='Encryption', row=0, column=0, label=True,  # paidWidth=100,
#                            tooltip=tooltips.encryptionCombo, )
#     baudRateCombo = dict(text='Baud Rate (Hz)', row=1, column=0, label=True,  # paidWidth=100,
#                          tooltip=tooltips.baudRateCombo)
#     bandCombo = dict(text='Frequency Band', row=2, column=0, label=True,  # pairWidth=120,
#                      tooltip=tooltips.bandCombo, )
#     rfLabel = dict(row=3, column=0,  # pairWidth=100,
#                    tooltip=tooltips.frequencySlider, )
#     rfSlider = dict(row=3, column=1,  # pairWidth=100,
#                     tooltip=tooltips.frequencySlider, )
#     rfSpinBox = dict(row=3, column=2, decimals=2, width=65, tooltip=tooltips.frequencySlider, )
#     portEntry = dict(text='Port', label=True, row=4, column=0, tooltip='',  # TODO
#                      )
#     portLockBtn = dict(width=25, row=4, column=2, tooltip='',  # TODO
#                        )
#
#
# class transceiver:
#     # Group Box
#     transmissionsGroupBox = dict(text='Transmission', row=0, column=0,  # width=600,
#                                  # height=170,
#                                  )
#     cmdEntry = dict(text='Command', area=True,  # width=700,
#                     # height=55,
#                     row=0, column=0, tooltip=tooltips.msgSend, )
#     sendCmdBtn = dict(text='Send', row=0, column=1, tooltip=tooltips.msgSendButton, )
#     transceiverActiveBtn = dict(width=40, row=0, column=1, tooltip='',  # TODO
#                                 )
#     receivedMsg = dict(text='Received', area=True,  # width=700,
#                        # height=55,
#                        row=1, column=0, readOnly=True, tooltip=tooltips.msgReceive, )
#
#
# class device:
#     xCombo = dict(gbHeight=40, row=0, column=0, text='x', tooltip=tooltips.posCombo, )
#     yCombo = dict(text='y', row=0, column=1, gbHeight=40, tooltip=tooltips.posCombo, )
#     zCombo = dict(text='z', row=0, column=2, gbHeight=40, tooltip=tooltips.posCombo, )
#     units = dict(text='Units', row=0, column=3, gbHeight=40, tooltip='units', )
#     coordinateDataGroupBox = dict(height=54, text='Coordinate Data', row=0, column=0, border=False, )
#
#
# class log:
#     # GRID
#     logDirsGrid = dict(border=False, column=0, row=1, text='Log Directory', width=90, )
#     logsGrid = dict(border=False, column=1, row=1, text='Log', width=80, )
#     logReaderGrid = dict(border=False, column=2, row=1, text='', )
#
#
# class main:
#     mainMenu = dict(title='cSETR CubeSAT Controller', icon=paths.cube, )
#     deviceToolBar = dict(text='Device', icon=paths.cube, shortcut=shortcuts.device, )
#     transceiverToolbar = dict(icon=paths.wifi, text='Transceiver', shortcut=shortcuts.transceiver, )
#     plotToolbar = dict(icon=paths.signal, text='Plot', shortcut=shortcuts.plot, )
#     logToolbar = dict(icon=paths.bookShelf, text='Log', shortcut=shortcuts.log, )
#     deviceMenu = dict(title='Device', icon=paths.cube, )
#     documentationMenu = dict(title='Documentation', icon=paths.book, )
#     logMenu = dict(title='Log', icon=paths.bookShelf, )
#     plotMenu = dict(title='Plot', icon=paths.signal, )
#     settingsMenu = dict(title='Settings', icon=paths.settings, )
#     transceiverMenu = dict(title='Transceiver', icon=paths.wifi, )
#     deviceButton = dict(icon=paths.cube, width=50, height=50, )
#     transceiverButton = dict(icon=paths.wifi, width=50, height=50, )
#     plotButton = dict(icon=paths.signal, width=50, height=50, )
#     logButton = dict(icon=paths.bookShelf, width=50, height=50, )
#
#
# class plot:
#     toolsGrid = dict(column=3, row=4, text='', )
#     domainGroupBox = dict(border=False, column=0, row=0, text='Domain', )
#     rangeGroupBox = dict(border=False, column=1, row=0, text='Range', )
#     variablesGroupBox = dict(border=False, column=2, row=0, text='Variables', )
#     xCombo = dict(column=0, row=0, )
#     yCombo = dict(column=1, row=0, )
#
#
# class settings:
#     # DATA
#     delimiterEntry = dict(text='Delimiter', row=0, column=0, columnSpan=1, gbWidth=70, gbHeight=40,
#                           tooltip=tooltips.dataTypeEntry, )
#     delimiterLkBtn = dict(width=25, row=0, column=1, tooltip=tooltips.dataTypeLkBtn, )
#     nVarsInput = dict(text='Variables', row=0, column=1, columnSpan=1, gbWidth=80, gbHeight=40,
#                       tooltip=tooltips.nVarsInput, )
#     nVarsLkBtn = dict(width=25, row=0, column=1, tooltip=tooltips.nVarsLkBtn, )
#     bufferSlider = dict(text='Buffer', row=0, column=2, columnSpan=2, gbHeight=40, tooltip=tooltips.buffer, )
#     dataTypeEntry = dict(text='Data Types', row=1, column=0, columnSpan=4, gbHeight=40,
#                          tooltip=tooltips.dataTypeEntry, )
#     dataTypeLkBtn = dict(width=25, row=0, column=1, tooltip=tooltips.dataTypeLkBtn, )
#     headEntry = dict(text='Column Names', row=2, column=0, columnSpan=4, gbHeight=40, tooltip=tooltips.headEntry, )
#     headLkBtn = dict(width=25, row=0, column=1, tooltip=tooltips.headLkBtn, )
#     unitsEntry = dict(text='Units', row=3, column=0, columnSpan=4, gbHeight=40, tooltip=tooltips.unitsEntry, )
#     unitsLkBtn = dict(width=25, row=0, column=1, tooltip=tooltips.unitsLkBtn, )
#     # DEVICE
#     deviceSettingsGroupBox = dict(text='Device', row=0, column=0, )
#     renderAxesQuiverCheckBox = dict(text='Axes Quiver', row=0, column=0, tooltip=tooltips.renderAxesQuiver, )
#     renderCubeQuiverCheckBox = dict(row=0, column=1, tooltip=tooltips.renderCubeQuiver, )
#     renderWireframeCheckBox = dict(text='Wireframe', row=0, column=2, tooltip=tooltips.renderWireframe, )
#     renderFacesCheckBox = dict(text='Faces', row=0, column=3, tooltip=tooltips.renderFaces, )
#     renderThrustersCheckBox = dict(text='Thrusters', row=0, column=4, tooltip=tooltips.renderThrusters, )
#     renderAVCheckBox = dict(text='Angular Velocity', row=0, column=5, tooltip=tooltips.renderAV, )
#     # GENERATOR
#     enableSimulationCheckBox = dict(text='Enable Simulation', row=0, column=0, columnSpan=4, gbHeight=40,
#                                     tooltip='Enable simulation.', )
#     methodsEntry = dict(text='Method', row=1, column=0, columnSpan=1, gbHeight=40, tooltip=tooltips.dataTypeEntry, )
#     methodLkBtn = dict(width=25, row=0, column=1, tooltip=tooltips.dataTypeLkBtn, )
#     reliabilityInput = dict(text='Reliability', row=1, column=1, columnSpan=1, gbHeight=40, gbWidth=70,
#                             tooltip=tooltips.dataTypeEntry, )
#     reliabilityLkBtn = dict(width=25, row=0, column=1, tooltip=tooltips.dataTypeLkBtn, )
#     # LOG
#     logSettingsGroupBox = dict(text='Log', row=0, column=0, )
#     loggingRateSpinBox = dict(text='Rate (Samples)', row=0, column=0, tooltip=tooltips.loggingRate, )
#     logCreationIntervalSpinBox = dict(text='Creation Interval (Seconds)', row=0, column=1,
#                                       tooltip=tooltips.logCreationInterval, )
#     # TRANSCEIVER
#     transceiverSettingsGroupBox = dict(text='Transceiver', row=0, column=0, )
#     encryptionCombo = dict(text='Encryption', row=0, column=0,  # paidWidth=100,
#                            tooltip=tooltips.encryptionCombo, )
#     baudRateCombo = dict(text='Baud Rate (Hz)', row=0, column=1,  # paidWidth=100,
#                          tooltip=tooltips.baudRateCombo)
#     bandCombo = dict(text='Frequency Band', row=1, column=0,  # pairWidth=120,
#                      tooltip=tooltips.bandCombo, )
#     rfSlider = dict(row=1, column=1,  # pairWidth=100,
#                     tooltip=tooltips.frequencySlider, )
#     rfSpinBox = dict(row=0, column=1, decimals=2, width=65, tooltip=tooltips.frequencySlider, )
#     portEntry = dict(text='Port', row=0, column=2, tooltip='',  # TODO
#                      )
#     portLkBtn = dict(width=25, row=0, column=1, tooltip='',  # TODO
#                      )
#
#
# class transceiver:
#     # Group Box
#     transmissionsGroupBox = dict(text='Transmission', row=0, column=0,  # width=600,
#                                  # height=170,
#                                  )
#     cmdEntry = dict(text='Command', area=True,  # width=700,
#                     # height=55,
#                     row=1, column=0, tooltip=tooltips.msgSend, )
#     sendCmdBtn = dict(icon=paths.sendMsg, row=0, column=1, tooltip=tooltips.msgSendButton, )
#     transceiverActiveBtn = dict(row=0, column=1, tooltip='',  # TODO
#                                 )
#     receivedMsg = dict(text='Received', area=True,  # width=700,
#                        # height=55,
#                        row=2, column=0, readOnly=True, tooltip=tooltips.msgReceive, )
#
#
# from utility import paths, tooltips
#
#
# class device:
#     xCombo = dict(gbHeight=40, row=0, column=0, text='x', tooltip=tooltips.posCombo, )
#     yCombo = dict(text='y', row=0, column=1, gbHeight=40, tooltip=tooltips.posCombo, )
#     zCombo = dict(text='z', row=0, column=2, gbHeight=40, tooltip=tooltips.posCombo, )
#     coordinateDataGroupBox = dict(height=54, text='Coordinate Data', row=0, column=0, border=False, )
#
#
# class log:
#     # LABEL
#     activeLogLabel = dict(row=0, column=2, width=250, )
#     # BUTTON
#     autoUpdateButton = dict(row=2, column=0, width=25, icon=paths.downArrow, tooltip=tooltips.autoUpdateButton, )
#     exportButton = dict(row=0, column=0, width=25, icon=paths.exportButton, tooltip=tooltips.exportButton)
#     refreshButton = dict(row=1, column=0, width=25, icon=paths.refreshButton, tooltip=tooltips.refreshButton, )
#     # GRID
#     logDirsGrid = dict(text='Log Directory', row=1, column=0, width=90, border=False, )
#     logsGrid = dict(border=False, text='Log', row=1, column=1, width=80, )
#     toolsGrid = dict(text='', row=0, column=3, )
#     logReaderGrid = dict(border=False, text='', row=1, column=3, )
#
#
# class plot:
#     toolsGrid = dict(text='', row=4, column=3, )
#     domainGroupBox = dict(text='Domain', row=0, column=0, border=False, )
#     rangeGroupBox = dict(text='Range', row=0, column=1, border=False, )
#     variablesGroupBox = dict(text='Variables', row=0, column=2, border=False, )
#     xCombo = dict(row=0, column=0, )
#     yCombo = dict(row=0, column=1, )
#
#
# class settings:
#     # DATA
#     delimiterEntry = dict(text='Delimiter', row=0, column=0, columnSpan=1, gbWidth=70, gbHeight=40,
#                           tooltip=tooltips.dataTypeEntry, )
#     delimiterLkBtn = dict(width=25, row=0, column=1, tooltip=tooltips.dataTypeLkBtn, )
#     nVarsInput = dict(text='Variables', row=0, column=1, columnSpan=1, gbWidth=80, gbHeight=40,
#                       tooltip=tooltips.nVarsInput, )
#     nVarsLkBtn = dict(width=25, row=0, column=1, tooltip=tooltips.nVarsLkBtn, )
#     bufferSlider = dict(text='Buffer', row=0, column=2, columnSpan=2, gbHeight=40, tooltip=tooltips.buffer, )
#     dataTypeEntry = dict(text='Data Types', row=1, column=0, columnSpan=4, gbHeight=40,
#                          tooltip=tooltips.dataTypeEntry, )
#     dataTypeLkBtn = dict(width=25, row=0, column=1, tooltip=tooltips.dataTypeLkBtn, )
#     headEntry = dict(text='Column Names', row=2, column=0, columnSpan=4, gbHeight=40, tooltip=tooltips.headEntry, )
#     headLkBtn = dict(width=25, row=0, column=1, tooltip=tooltips.headLkBtn, )
#     unitsEntry = dict(text='Units', row=3, column=0, columnSpan=4, gbHeight=40, tooltip=tooltips.unitsEntry, )
#     unitsLkBtn = dict(width=25, row=0, column=1, tooltip=tooltips.unitsLkBtn, )
#     # DEVICE
#     deviceSettingsGroupBox = dict(text='Device', row=0, column=0, )
#     renderAxesQuiverCheckBox = dict(text='Axes Quiver', row=0, column=0, tooltip=tooltips.renderAxesQuiver, )
#     renderCubeQuiverCheckBox = dict(row=0, column=1, tooltip=tooltips.renderCubeQuiver, )
#     renderWireframeCheckBox = dict(text='Wireframe', row=0, column=2, tooltip=tooltips.renderWireframe, )
#     renderFacesCheckBox = dict(text='Faces', row=0, column=3, tooltip=tooltips.renderFaces, )
#     renderThrustersCheckBox = dict(text='Thrusters', row=0, column=4, tooltip=tooltips.renderThrusters, )
#     renderAVCheckBox = dict(text='Angular Velocity', row=0, column=5, tooltip=tooltips.renderAV, )
#     # GENERATOR
#     methodsEntry = dict(text='Method', row=0, column=0, columnSpan=4, gbHeight=40, tooltip=tooltips.dataTypeEntry, )
#     methodLkBtn = dict(width=25, row=0, column=1, tooltip=tooltips.dataTypeLkBtn, )
#     reliabilityInput = dict(text='Reliability', row=1, column=0, columnSpan=1, gbHeight=40, gbWidth=70,
#                             tooltip=tooltips.dataTypeEntry, )
#     reliabilityLkBtn = dict(width=25, row=0, column=1, tooltip=tooltips.dataTypeLkBtn, )
#     # LOG
#     logSettingsGroupBox = dict(text='Log', row=0, column=0, )
#     loggingRateSpinBox = dict(text='Rate (Samples)', row=0, column=0, tooltip=tooltips.loggingRate, )
#     logCreationIntervalSpinBox = dict(text='Creation Interval (Seconds)', row=0, column=1,
#                                       tooltip=tooltips.logCreationInterval, )
#     # TRANSCEIVER
#     transceiverSettingsGroupBox = dict(text='Transceiver', row=0, column=0, )
#     encryptionCombo = dict(text='Encryption', row=0, column=0,  # paidWidth=100,
#                            tooltip=tooltips.encryptionCombo, )
#     baudRateCombo = dict(text='Baud Rate (Hz)', row=0, column=1,  # paidWidth=100,
#                          tooltip=tooltips.baudRateCombo)
#     bandCombo = dict(text='Frequency Band', row=1, column=0,  # pairWidth=120,
#                      tooltip=tooltips.bandCombo, )
#     rfSlider = dict(row=1, column=1,  # pairWidth=100,
#                     tooltip=tooltips.frequencySlider, )
#     rfSpinBox = dict(row=0, column=1, decimals=2, width=65, tooltip=tooltips.frequencySlider, )
#     portEntry = dict(text='Port', row=0, column=2, tooltip='',  # TODO
#                      )
#     portLkBtn = dict(width=25, row=0, column=1, tooltip='',  # TODO
#                      )
#
#
# class device:
#     posXCombo = dict(text='x', tooltip=tooltips.posZCombo, )
#     posYCombo = dict(text='y', tooltip=tooltips.posZCombo, )
#     posZCombo = dict(text='z', tooltip=tooltips.posZCombo, )
#     coordinateDataGroupBox = dict(text='Coordinate Data', row=0, column=0, border=False, )
#
#
# class log:
#     # BUTTON
#     autoUpdateButton = dict(row=2, column=0, width=25, icon=icons.downArrow, tooltip=tooltips.autoUpdateButton, )
#     exportButton = dict(row=0, column=0, width=25, icon=icons.exportButton, tooltip=tooltips.exportButton)
#     refreshButton = dict(row=1, column=0, width=25, icon=icons.refreshButton, tooltip=tooltips.refreshButton, )
#     # GRID
#     logDirsGrid = dict(text='Scenario', row=1, column=0, border=False, )
#     logsGrid = dict(border=False, text='Index', row=1, column=2, width=40, )
#     dictKeysGrid = dict(border=False, text='Keys', row=1, column=3, )
#     contentsGrid = dict(border=False, text='Contents', row=1, column=4, )
#     toolsGrid = dict(text='', row=0, column=0, )
#     logReaderGrid = dict(border=False, text='', row=1, column=10, )
#
#
# class settings:
#     # DATA
#     delimiterEntry = dict(text='Delimiter', row=0, column=0,  # paidWidth=100,
#                           tooltip=tooltips.dataTypeEntry, )
#     delimiterLockBtn = dict(width=25, row=0, column=1, tooltip=tooltips.dataTypeLockBtn, )
#     nVarsInput = dict(text='Number of Variables', row=0, column=1,  # paidWidth=100,
#                       tooltip=tooltips.nVarsInput, )
#     nVarsLockBtn = dict(width=25, row=0, column=1, tooltip=tooltips.nVarsLockBtn, )
#     dataTypeEntry = dict(text='Data Types', row=1, column=0,  # paidWidth=100,
#                          tooltip=tooltips.dataTypeEntry, )
#     dataTypeLockBtn = dict(width=25, row=0, column=1, tooltip=tooltips.dataTypeLockBtn, )
#     headEntry = dict(text='Column Names', row=2, column=0,  # paidWidth=100,
#                      tooltip=tooltips.headEntry, )
#     headLockBtn = dict(width=25, row=0, column=1, tooltip=tooltips.headLockBtn, )
#     unitsEntry = dict(text='Units', row=3, column=0,  # paidWidth=100,
#                       tooltip=tooltips.unitsEntry, )
#     unitsLockBtn = dict(width=25, row=0, column=1, tooltip=tooltips.unitsLockBtn, )
#     # LOG
#     logSettingsGroupBox = dict(text='Log', row=2, column=0, )
#     loggingRateSpinBox = dict(text='Logging Rate', row=0, column=0, )
#     logCreationIntervalSpinBox = dict(text='Log Creation Interval (Seconds)', row=1, column=0, )
#     # DEVICE
#     deviceSettingsGroupBox = dict(text='Device', row=0, column=1, )
#     renderAxesQuiverCheckBox = dict(text='Axes Quiver', row=0, column=0, tooltip=tooltips.renderAxesQuiver, )
#     renderCubeQuiverCheckBox = dict(row=0, column=1, tooltip=tooltips.renderCubeQuiver, )
#     renderWireframeCheckBox = dict(text='Wireframe', row=1, column=0, tooltip=tooltips.renderWireframe, )
#     renderFacesCheckBox = dict(text='Faces', row=1, column=1, tooltip=tooltips.renderFaces, )
#     renderThrustersCheckBox = dict(text='Thrusters', row=2, column=0, tooltip=tooltips.renderThrusters, )
#     # TRANSCEIVER
#     transceiverSettingsGroupBox = dict(text='Transceiver', row=3, column=0, )
#     encryptionCombo = dict(text='Encryption', row=0, column=0,  # paidWidth=100,
#                            tooltip=tooltips.encryptionCombo, )
#     baudRateCombo = dict(text='Baud Rate (Hz)', row=1, column=0,  # paidWidth=100,
#                          tooltip=tooltips.baudRateCombo)
#     bandCombo = dict(text='Frequency Band', row=2, column=0,  # pairWidth=120,
#                      tooltip=tooltips.bandCombo, )
#     rfSlider = dict(row=3, column=0,  # pairWidth=100,
#                     tooltip=tooltips.frequencySlider, )
#     rfSpinBox = dict(row=0, column=1, decimals=2, width=65, tooltip=tooltips.frequencySlider, )
#     portEntry = dict(text='Port', row=4, column=0, tooltip='',  # TODO
#                      )
#     portLockBtn = dict(width=25, row=0, column=1, tooltip='',  # TODO
#                        )


########################################################
import json
import os

import pandas as pd

name = 'IDA ML'
root = r'E:\github2\test\ida'


class MLDude:
    def __init__(self):
        self.basepath = os.path.join(os.getcwd(), '10hz')
        self.dirs = os.listdir(self.basepath)
        self.files = []
        self.columnNames = []
        self.directory()
        self.file()

    def directory(self):
        for dir in self.dirs:
            fullpath = os.path.join(self.basepath, dir)
            files = os.listdir(fullpath)
            for file in files:
                r = file.split('.')
                if len(r) > 1:
                    if r[1] == 'csv':
                        filepath = os.path.join(fullpath, file)
                        self.files.append(filepath)
        print(self.files)
        n = 0

    def file(self):
        for i, file in enumerate(self.files):
            df = pd.read_csv(file, nrows=2)
            col = ' Height (AGL)'
            if col in df.columns:
                try:
                    df = pd.read_csv(file)
                    pNan = df[col].isna().sum() / len(df)
                    if pNan < 0.1:
                        df[col].plot()
                        break
                    n += 1
                except Exception as e:
                    print(e)


class Scenario:
    def __init__(self, fullpath):
        '''
        Scenario object contains the name of a scenario and the path to the corresponding RAID file.
        The path is passed through the fullpath argument.
        The RAID file is read and stored in the _json variable.
        The _json variable is a list.
        Each  item in the list is an entry in the database contained in the RAID file.
        An entry may be a ROUTE or ATTACK_PLAN type and contains several variables.
        :param fullpath: A string of text containing the full path to a RAID file.
        '''
        self.fullpath = fullpath
        self.name = os.path.basename(os.path.normpath(fullpath))
        self.json = self._set_json()

    def _set_json(self):
        '''
        Load the data into the _json variable
        :return:
        '''
        with open(self.fullpath, 'r') as f:
            _j = json.load(f)
        return _j


def get_object_by_key(key, value, scenario):
    '''
    Return an object from a scenario if the key and value match the provided key and value.
    :param key:
    :param value:
    :param scenario:
    :return:
    '''
    for object in scenario.json:
        if key in object:
            if object[key] == value:
                return object
