import matplotlib

import configuration
import lozoya.communication
import lozoya.data
import lozoya.decorators
import lozoya.file
import lozoya.gui
import lozoya.math
import lozoya.plot

try:
    from matplotlib.backends.backend_qt5agg import (FigureCanvas)
except:
    from matplotlib.backends.backend_qt5agg import FigureCanvasAgg as FigureCanvas
matplotlib.use('TkAgg')


class App(lozoya.gui.TSApp):
    def __init__(self, name, root):
        lozoya.gui.TSApp.__init__(self, name, root)
        self.make_configuration_objects()
        self.make_computation_objects()
        self.make_forms()
        self.widgetSwitcher = lozoya.gui.TSTabWidgetSwitcher(widgets=self.forms)
        self.window.setCentralWidget(
            self.widgetSwitcher
        )  # self.tool_bars()  # self.menu_bar()  # self.status_bar()  # self.initialize_style()  # self.dataConfig = configuration.py.DataConfig(app=self.app)  # self.deviceConfig = configuration.py.DeviceConfig(app=self.app)  # self.deviceMenu = device.DeviceMenu(app=self.app, **app.cubesat.widgetkwargs.main.deviceMenu)  # self.deviceMenu = device.DeviceMenu(parent=self, title='Device', icon=icons.CUBE_ICON, )  # self.deviceMenu = device.DeviceMenu(parent=self, **app.cubesat.widgetkwargs.main.deviceMenu, )  # self.documentationMenu = documentation.DocumentationMenu(  #         parent=self, title='Documentation5', icon=icons.DOCUMENTATION_ICON, )  # self.documentationMenu = documentation.DocumentationMenu(app=self.app, **app.cubesat.widgetkwargs.main.documentationMenu)  # self.epsMenu = eps.EPSMenu(parent=self, title='EPS Status', icon=icons.EPS_ICON, )  # self.generatorConfig = configuration.py.GeneratorConfig(app=self.app)  # self.imuMenu = imu.IMUMenu(parent=self, title='IMU Status', icon=icons.IMU_ICON, )  # self.logConfig = configuration.py.LogConfig(app=self.app)  # self.logMenu = app.log_tool.log.LogMenu2(app=self.app, **app.cubesat.widgetkwargs.main.logMenu)  # self.logMenu = app.log_tool.log.LogMenu2(app=self.app, title='Log', icon=icons.LOG_ICON, )  # self.logMenu.update_dir_list()  # self.mainMenu = main.MainMenu(self, **app.cubesat.widgetkwargs.main.mainMenu)  # self.plotConfig = configuration.py.PlotConfig(app=self.app)  # self.plotMenu = plot.PlotMenu(app=self.app, **app.cubesat.widgetkwargs.main.plotMenu)  # self.preferences = preference.Preference(parent=self)  # self.settingsMenu = settings.SettingsMenu(app=self.app, **app.cubesat.widgetkwargs.main.settingsMenu)  # self.settingsMenu = settings.SettingsMenu(parent=self, title='Settings', icon=icons.SETTINGS_ICON, )  # self.settingsMenu = settings.SettingsMenu(parent=self, **app.cubesat.widgetkwargs.main.settingsMenu, )  # self.transceiverConfig._transceiverActive = False  # self.transceiverConfig = configuration.py.TransceiverConfig(app=self.app)  # self.transceiverMenu = transceiver.TransceiverMenu(app=self.app, **app.cubesat.widgetkwargs.main.transceiverMenu)  # self.transceiverMenu = transceiver.TransceiverMenu(  #         parent=self, title='Transceiver', icon=icons.TRANSCEIVER_ICON, )  # self.client = client.Client(app=self.app)  # self.comPort = comport.ComPort(app=self.app)  # self.controller = control.Controller(app=self.app)  # self.cubeSat = satellite.CubeSat(app=self.app, thrusters=thruster.configurations['1z4'])  # self.dataDude = datadude.DataDude(app=self.app)  # self.dataDude = datadude.DataDude(10, 'var,', ',', 'f')  # self.logger = logger.Logger(app=self.app)  # self.plotDude = lozoya.plot_api.PlotDude2(parent=self)  # self.server = server.Server(app=self.app)  # self.signalGenerator = generator.SignalGenerator(app=self.app)  # self.signalGenerator = generator.SignalGenerator(dataType=1, delimiter=1, methods=1, reliability=1)  # self.transceiver = Transceiver(app=self.app)  # self.preferences.transceiverActive = False  # self.transformer = transforms.Transformer(app=self.app)  # self.transceiver.start_listener(self.update_everything)  # self.transceiver.start_listener()  # self.setStyleSheet(qss.WINDOW_STYLE)  # self.setStyleSheet(qss.format_qss(qss.window, self.palette))  # self.setStyleSheet(  #         gui.qss._format(  #                 lozoya.gui.styles.generalstyles.window + lozoya.gui.styles.generalstyles.widget, self.app.palette  #         )  # )  # self.setStyleSheet(  #         qss._format(interface.styles.general.window + interface.styles.general.widget, self.app.palette)  # )  # self.showMaximized()  # sys.exit(self.app.exec_())

    def make_computation_objects(self):
        # self.cubesat = CubeSat()
        # self.controller = Controller()
        # self.dataer = DataDude()
        # self.logger = Logger()
        # self.plotter = Plotter()
        pass

    def make_configuration_objects(self):
        make = lozoya.file.make_configuration_file_from_class
        root = self.root
        self.configData = make(
            template=configuration.ConfigData,
            path=f'{root}\data.config',
        )
        self.configDevice = make(
            template=configuration.ConfigDevice,
            path=f'{root}\device.config',
        )
        self.configGenerator = make(
            template=configuration.ConfigGenerator,
            path=f'{root}\generator.config',
        )
        self.configLog = make(
            template=configuration.ConfigLog,
            path=f'{root}\log.config',
        )
        self.configPlot = make(
            template=configuration.ConfigPlot,
            path=f'{root}\plot.config',
        )
        self.configTransceiver = make(
            template=configuration.ConfigTransceiver,
            path=f'{root}\\transceiver.config',
        )

    def make_forms(self):
        make = lozoya.gui.make_form_from_class
        self.formConfigData = make(
            template=self.configData, name='Data'
        )
        self.formConfigDevice = make(
            template=self.configDevice, name='Device'
        )
        self.formConfigGenerator = make(
            template=self.configGenerator, name='Generator'
        )
        self.formConfigLog = make(
            template=self.configLog, name='Log'
        )
        self.formConfigPlot = make(
            template=self.configPlot, name='Plot'
        )
        self.formConfigTransceiver = make(
            template=self.configTransceiver, name='Transceiver'
        )
        self.forms = [self.formConfigData, self.formConfigDevice, self.formConfigGenerator, self.formConfigLog,
                      self.formConfigPlot, self.formConfigTransceiver, ]

    # def update_status(self, msg, stylesheet):  #     self.statusbar.showMessage(msg)  #     self.statusbar.adjustSize()  #     self.statusbar.setStyleSheet(qss.format_qss(stylesheet, self.palette))

    # def _update_status(self, f):  #     def _(*args, **kwargs):  #         try:  #             return f(*args, **kwargs)  #         except Exception as e:  #             statusMsg = '{}.'.format(e)  #             self.update_status(statusMsg, 'error')

    # def status_bar(self):  #     self.statusbar = self.statusBar()  # statusBar = QtWidgets.QStatusBar()  # self.statusbar.setStyleSheet(qss.format_qss(qss.statusbar, self.palette))  # statusBar.setStyleSheet(qss.STATUSBAR_STYLE)  # self.statusBar.showMessage('hi')  #  # def menu_bar(self):  #     menubar = self.menuBar()  #     # menubar.setStyleSheet(qss.MENUBAR_STYLE) # qss.format_qss(qss.menubar, self.palette)  #     fileMenu = menubar.addMenu('&File')  #     helpMenu = menubar.addMenu('&Help')  #     openFileAction = lozoya.gui.make_action(  #         'Open File',  # callback=lambda: open_files(menu),  #         shortcut='Ctrl+O', icon=None,  # QtGui.QIcon(icon.OPEN_FILE_ICON)  #     )  #     settingsAction = lozoya.gui.make_action(  #         'Settings',  # callback=self.settingsMenu.show,  #         shortcut='Ctrl+S', icon=None,  # QtGui.QIcon(icons.SETTINGS_ICON),  #         tooltip='Settings Option Menu configuration.Configuration, etc.', )  #     exitAction = lozoya.gui.make_action(  #         'Exit', callback=self.close, shortcut='Ctrl+esc', icon=None,  # QtGui.QIcon(icons.EXIT_ICON),  #         tooltip='Exit Application'  #     )  #     documentationAction = lozoya.gui.make_action(  #         'Documentation',  # callback = self.documentationMenu.show  #         shortcut='Ctrl+D', icon=None,  # QtGui.QIcon(icons.DOCUMENTATION_ICON)  #         tooltip='Documentation for how to use the app.', )  #     fileMenu.addAction(openFileAction)  #     fileMenu.addAction(settingsAction)  #     fileMenu.addAction(exitAction)  #     helpMenu.addAction(  #         documentationAction  #     )  # self.menubarItem = self.menubar.addMenu(QtGui.QIcon(utility.configuration.settings), '')  # settingsAction = QtWidgets.QAction(QtGui.QIcon(utility.configuration.cog), 'Settings', self)  # settingsAction.triggered.connect(self.app.settingsMenu.show)  # self.menubarItem.addAction(settingsAction)  # documentationAction = QtWidgets.QAction(QtGui.QIcon(utility.configuration.book), 'Documentation', self)  # documentationAction.triggered.connect(self.app.documentationMenu.show)  # self.menubarItem.addAction(documentationAction)  # settingsAction.triggered.connect(self.app.settingsMenu.show)  # self.menubarItem.addSeparator()  # exitAction = QtWidgets.QAction(QtGui.QIcon(utility.configuration.exit), 'Exit', self)  # exitAction.triggered.connect(self.app.quit)  # self.menubarItem.addAction(exitAction)  # pass  # menubarFile = self.menubar.add_menu('File', utility.configuration.settings)  # self.menubar.add_entry(  #         title='Save', callback=self.app.settingsMenu.show, menu=menubarFile, iconPath=utility.configuration.cog  # )  # self.menubar.add_entry(  #         title='Open', callback=self.app.settingsMenu.show, menu=menubarFile, iconPath=utility.configuration.cog  # )  # menubarFile.addSeparator()  # self.menubar.add_entry(title='Exit', callback=self.app.quit, menu=menubarFile, iconPath=utility.configuration.exit)  # menubarEdit = self.menubar.add_menu('Edit')  # self.menubar.add_entry(  #         title='Settings', callback=self.app.settingsMenu.show, menu=menubarEdit, iconPath=utility.configuration.cog  # )  # menubarHelp = self.menubar.add_menu('Help')  # self.menubar.add_entry(  #         'Documentation', self.app.documentationMenu.show, menubarHelp, iconPath=utility.configuration.book  # )  #  # def tool_bars(self):  #     self.device = lozoya.gui.make_action(  #         callback=None,  # self.deviceMenu.show,  #         label='Device', icon=None,  # icons.CUBE_ICON,  #         shortcut='Ctrl+D', )  #     self.epsMonitor = lozoya.gui.make_action(  #         callback=None,  # self.epsMenu.show,  #         icon=None,  # icons.EPS_ICON,  #         label='EPS', shortcut='Ctrl+E', )  #     self.imuMonitor = lozoya.gui.make_action(  #         callback=None,  # self.imuMenu.show,  #         icon=None,  # icons.IMU_ICON,  #         label='IMU', shortcut='Ctrl+I', )  #     self.log = lozoya.gui.make_action(  #         callback=None,  # self.logMenu.show,  #         icon=None,  # icons.LOG_ICON,  #         label='Log', shortcut='Ctrl+L', )  #     self.transceiver = lozoya.gui.make_action(  #         callback=None,  # self.transceiverMenu.show,  #         icon=None,  # icons.TRANSCEIVER_ICON,  #         label='Transceiver', shortcut='Ctrl+T', )  #     # self.deviceToolbar = self.menuBuilder.make_action(  #     #         callback=self.deviceMenu.show,  #     #         **app.cubesat.widgetkwargs.main.deviceToolBar,  #     # )  #     # self.transceiverToolbar = self.menuBuilder.make_action(  #     #         callback=self.transceiverMenu.show,  #     #         **app.cubesat.widgetkwargs.main.transceiverToolbar,  #     # )  #     # self.plotToolbar = self.menuBuilder.make_action(  #     #         callback=self.plotMenu.show,  #     #         **app.cubesat.widgetkwargs.main.plotToolbar,  #     # )  #     # self.logToolbar = self.menuBuilder.make_action(  #     #         callback=self.logMenu.show,  #     #         **app.cubesat.widgetkwargs.main.logToolbar,  #     # )  #     # self.transceiverButton = lozoya.gui.make_button(  #     #         callback=self.app.transceiverMenu.show,  #     #         **app.cubesat.widgetkwargs.main.transceiverButton,  #     # )  #     # self.logButton = lozoya.gui.make_button(  #     #         callback=self.app.logMenu.show,  #     #         **app.cubesat.widgetkwargs.main.logButton,  #     # )  #     # self.deviceButton = lozoya.gui.make_button(  #     #         callback=self.app.deviceMenu.show,  #     #         **app.cubesat.widgetkwargs.main.deviceButton,  #     # )  #     # self.plotButton = lozoya.gui.make_button(  #     #         callback=self.app.plotMenu.show,  #     #         **app.cubesat.widgetkwargs.main.plotButton,  #     # )  #     actions = [self.device, self.epsMonitor, self.imuMonitor, self.log, self.transceiver, ]  #     self.toolbar = lozoya.gui.make_toolbar('Toolbar', actions)  #     group = lozoya.gui.make_action_group(self.toolbar, actions)  #     self.form.add_field(self.toolbar)

    # def initialize_style(self):  #     self.setContentsMargins(0, 0, 0, 0)  #     self.setFixedSize(300, 150)  #     self.setStyleSheet(  #             lozoya.gui.qss._format(  #                     lozoya.gui.styles.generalstyles.window + lozoya.gui.styles.generalstyles.widget, self.app.palette  #             )  #     )

    # def update_status(self, menu, msg, status, e=None):  #     try:  #         getattr(self, menu).update_status(msg, status, e)  #     except Exception as e:  #         print('Failed update status: {}.'.format(e))  #         print('{}: {}.'.format(status, msg))

    # @property  # def fullyLoaded(self):  #     return self.dataConfig.delimiterLock and self.dataConfig.nVarsLock and self.dataConfig.dataTypeLock and self.dataConfig.headLock and self.dataConfig.unitsLock and self.transceiverConfig.portConnected

    # def update_everything(self, i, msg):  #     time.sleep(0)  #     if self.transceiverConfig._transceiverActive or self.generatorConfig.simulationEnabled:  #         self.dataDude.update_data(msg)  #         if i % self.logConfig.loggingRate == 0:  #             try:  #                 self.logger.log_maintenance()  #                 self.logger.write_to_log(msg)  #                 self.transceiverMenu.receivedMsg.setPlainText(msg)  #                 if self.logConfig._autoUpdate:  #                     self.logMenu.logReaderTxt.auto_update_refresh(msg)  #             except Exception as e:  #                 print('Update everything error: ' + str(e))


# class CubeSat(lozoya.math_api.Cube):
#     d = {'-2': 'lb', '-1': 'rf', '-0': 'lf', '0': 'rb', '1': 'lb', '2': 'rb'}
#     activeThruster = None
#     activeThrusterCounter = 0
#     activeThrusterCounterMaximum = 10
#     _x, _y, _z = 0, 0, 0
#     x, y, z = 0, 0, 0
#     dx, dy, dz = 0, 0, 0
#     d = {'-2': 'lb', '-1': 'rf', '-0': 'lf', '0': 'rb', '1': 'lb', '2': 'rb'}
#     activeThruster = None
#     activeThrusterCounter = 0
#     activeThrusterCounterMaximum = 10
#
#     def __init__(self, app, dimension=1, centroid=lozoya.math_api.origin, mass=1, thrusters=None, randomAV=False):
#         super(CubeSat, self).__init__(app, dimension, centroid, mass, randomAV, )
#         self.facecolors = [cc('g'), cc('g'), cc('b'), cc('b'), cc('y'), cc('y'), ]
#         self.thrusters = {thruster.label: thruster for thruster in thrusters}
#
#     def apply_thrust(self, x, y, z, thruster='lb'):
#         self.activeThruster = thruster
#         # x, y, z = self.thrusters[thruster].thrustDirection
#         self._angularVelocity.x += x
#         self._angularVelocity.y += y
#         self._angularVelocity.z += z
#
#     def decide(self, *args, **kwargs):
#         avc = self.avc
#         if avc != (0, 0, 0):
#             setsd = [abs(_) for _ in avc]
#             criticalIndex = setsd.index(max(setsd))
#             criticalValue = avc[criticalIndex]
#             decision = str('{}{}'.format('-' if criticalValue < 0 else '', criticalIndex))
#             self.apply_thrust(self.d[decision])
#
#     def device_update(self, x, y, z):
#         try:
#             if self.app.deviceConfig.units == 'Degrees':
#                 conversion = -1  # Degrees
#             else:
#                 conversion = -0.0174533  # Radians
#             self.x, self.y, self.z = x, y, z
#             self.x = conversion * self.x
#             self.y = conversion * self.y
#             self.z = conversion * self.z
#             self.dx = self.x - self._x
#             self.dy = self.y - self._y
#             self.dz = self.z - self._z
#             self.app.cubeSat.set_avc(self.dx, self.dy, self.dz)
#             self.app.cubeSat.rotate()
#             self._x = self.x
#             self._y = self.y
#             self._z = self.z
#         except Exception as e:
#             statusMsg = 'Kinematic calculations error:  ' + str(e)
#             self.app.update_status('plot', statusMsg, 'error')
#
#     def print_info(self):
#         print(self.activeThruster, self._angularVelocity['x'], self._angularVelocity['y'], self._angularVelocity['z'], )
#
#     def rotate(self, *args, **kwargs):
#         ox, oy, oz = self.centroid
#         aX, aY, aZ = self.avc
#         for name, prop in [('_vertices', self._vertices)]:
#             results = {}
#             for k in prop:
#                 c = prop[k]
#                 px, py, pz = c.as_list()
#                 qx, qy = lozoya.math_api.rotate_point(aZ, ox, oy, px, py)  # rotate about z-axis
#                 qy, qz = lozoya.math_api.rotate_point(aX, oy, oz, qy, pz)  # rotate about x-axis
#                 qx, qz = lozoya.math_api.rotate_point(aY, ox, oz, qx, qz)  # rotate about y-axis
#                 results[k] = lozoya.math_api.Coordinate(qx, qy, qz)
#             setattr(self, name, results)
#         for name, prop in [('thrusters', self.thrusters)]:
#             for k in prop:
#                 c = prop[k].location
#                 px, py, pz = c.as_list()
#                 qx, qy = lozoya.math_api.rotate_point(aZ, ox, oy, px, py)  # rotate about z-axis
#                 qy, qz = lozoya.math_api.rotate_point(aX, oy, oz, qy, pz)  # rotate about x-axis
#                 qx, qz = lozoya.math_api.rotate_point(aY, ox, oz, qx, qz)  # rotate about y-axis
#                 prop[k].location = lozoya.math_api.Coordinate(qx, qy, qz)
#
#         for i, k in enumerate(self.quiverP):
#             px, py, pz = k
#             qx, qy = lozoya.math_api.rotate_point(-aZ, ox, oy, px, py)  # rotate about z-axis
#             qy, qz = lozoya.math_api.rotate_point(-aX, oy, oz, qy, pz)  # rotate about x-axis
#             qx, qz = lozoya.math_api.rotate_point(-aY, ox, oz, qx, qz)  # rotate about y-axis
#             self.quiverP[i] = [qx, qy, qz]
#         if self.activeThruster != None:
#             self.activeThrusterCounter += 1
#             if self.activeThrusterCounter > self.activeThrusterCounterMaximum:
#                 self.activeThrusterCounter = 0
#                 self.activeThruster = None
#
#     def what(self, old, new):
#         if new != old:
#             return new, new - old
#         return new, new
#
#     @property
#     def atc(self, *args, **kwargs):
#         return self.activeThrusterCounter
#
#     @property
#     def atcm(self, *args, **kwargs):
#         return self.activeThrusterCounterMaximum


# class Controller:
#     def __init__(self):
#         self.timer = 0
#         self.timeout = 0
#         self.tagged = False
#         self.cubesat = self.app.cubeSAT
#         self.commands = {
#             'q': lambda: self.cubesat.apply_thrust('lf'),
#             'w': lambda: self.cubesat.apply_thrust('rf'),
#             'e': lambda: self.cubesat.apply_thrust('lb'),
#             'r': lambda: self.cubesat.apply_thrust('rb'),
#             'i': self.cubesat.decide,
#         }
#
#     def check_press(self, *args, **kwargs):
#         msg = 'C0000'
#         if self.timeout == 0:
#             for command in self.commands:
#                 if keyboard.is_pressed(command):
#                     self.timeout = 15
#                     self.commands[command]()
#             if keyboard.is_pressed('['):
#                 # TODO: r1 -> r2, r1=-r2, rf=0
#                 self.tagged = True
#             if self.tagged:
#                 if self.timer <= 3:
#                     self.cubesat.apply_thrust('lb')
#                 else:
#                     self.cubesat.apply_thrust('rb')
#             if self.timer > 6:
#                 self.tagged = False
#                 self.timer = 0
#             if keyboard.is_pressed('x'):
#                 msg = self.button_x()
#             if keyboard.is_pressed('r'):
#                 msg = self.button_r()
#             if keyboard.is_pressed('t'):
#                 msg = self.button_t()
#         self.timer += 1
#         if self.timer > self.timeout and not self.tagged:
#             self.timer = 0
#             self.timeout = 0
#         return msg
#
#     def button_x(self, *args, **kwargs):
#         self.app.deviceConfig.update('renderAV', not self.app.deviceConfig.renderAV)
#         return "X0000"
#
#     def button_r(self, *args, **kwargs):
#         return "CC180"
#
#     def button_t(self, *args, **kwargs):
#         return "CW005"


# class DataDude:
#     """
#     This dude shall check the size of each log file periodically
#     If the size of a log file has increased, it shall update the dataframe for that file
#     Interface18 objects shall update when the dataframe is updated
#     """
#
#     def __init__(self, parent):
#         self.parent = parent
#         date = datetime.now().strftime('%m_%d_%Y')
#         _imuLog = os.path.join(configuration.IMU_LOG, date)
#         _epsLog = os.path.join(configuration.EPS_LOG, date)
#         _sensorLog = os.path.join(configuration.SENSOR_LOG, date)
#         self.imuData = pd.read_csv(_imuLog)
#         self.epsData = pd.read_csv(_epsLog)
#         self.sensorData = pd.read_csv(_sensorLog)
#
#     @property
#     def timestamps(self):
#         return self.imuData.iloc[:, 0]
#
#     @property
#     def accelerationX(self):
#         return self.imuData.iloc[:, 2]
#
#     @property
#     def accelerationY(self):
#         return self.imuData.iloc[:, 2]
#
#     @property
#     def accelerationZ(self):
#         return self.imuData.iloc[:, 3]
#
#     @property
#     def epsCurrent(self):
#         return self.epsData.iloc[:, 1]
#
#     @property
#     def epsVoltage(self):
#         return self.epsData.iloc[:, 2]
#
#     @property
#     def pressure(self):
#         return self.sensorData.iloc[:, 3]
#
#     @property
#     def temperature(self):
#         return self.sensorData.iloc[:, 4]
#
#     @property
#     def velocityX(self):
#         return self.imuData.iloc[:, 1]
#
#     @property
#     def velocityY(self):
#         return self.imuData.iloc[:, 2]
#
#     @property
#     def velocityZ(self):
#         return self.imuData.iloc[:, 3]


# class DevicePlot:
#     @lozoya.decorators.catch_error('mainMenu', *configuration.initDeviceMenuError)
#     def __init__(self, app):
#         self.app = app
#         self.fig = Figure()
#         self.canvas = FigureCanvas(self.fig)
#         self.ax = self.fig.add_subplot(111, projection='3d')
#         self.axesQuiverVector = np.array(
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], ]
#         )
#         self.avText = self.ax.text(-1, -1, -1, '', color='white')
#         self.faces = {
#             'up'   : self.ax.plot([], [], [], lw=self.app.plotConfig.wireframeWidth)[0],
#             'down' : self.ax.plot([], [], [], lw=self.app.plotConfig.wireframeWidth)[0],
#             'left' : self.ax.plot([], [], [], lw=self.app.plotConfig.wireframeWidth)[0],
#             'right': self.ax.plot([], [], [], lw=self.app.plotConfig.wireframeWidth)[0],
#             'back' : self.ax.plot([], [], [], lw=self.app.plotConfig.wireframeWidth)[0],
#             'front': self.ax.plot([], [], [], lw=self.app.plotConfig.wireframeWidth)[0],
#         }
#         self.thrusterPoints = {
#             'rf': self.ax.plot([], [], [], self.app.palette.thrusterColor, marker='$rf$', ms=self.app.plotConfig.ms)[0],
#             'rb': self.ax.plot([], [], [], self.app.palette.thrusterColor, marker='$rb$', ms=self.app.plotConfig.ms)[0],
#             'lf': self.ax.plot([], [], [], self.app.palette.thrusterColor, marker='$lf$', ms=self.app.plotConfig.ms)[0],
#             'lb': self.ax.plot([], [], [], self.app.palette.thrusterColor, marker='$lb$', ms=self.app.plotConfig.ms)[0],
#         }
#         self.poly = Poly3DCollection(
#             [], facecolors=self.app.cubeSat.facecolors, )
#         # ax.add_collection3d(poly, zs=range(6))
#         self.axesQuiverNegative = self.ax.quiver([], [], [], [], [], [])
#         self.axesQuiverPositive = self.ax.quiver([], [], [], [], [], [])
#         self.cubeQuiverNegative = self.ax.quiver([], [], [], [], [], [])
#         self.cubeQuiverPositive = self.ax.quiver([], [], [], [], [], [])
#         self._x, self._y, self._z = 0, 0, 0
#         if self.app.deviceConfig.renderAxesQuiver == False:
#             self.remove_axes_quiver()
#         if self.app.deviceConfig.renderCubeQuiver == False:
#             self.remove_cube_quiver()
#         # if self.parent.config0.renderFaces == False:
#         #     self.remove_cube_faces()
#         if self.app.deviceConfig.renderWireframe == False:
#             self.remove_wireframe()
#         self.anim = animation.FuncAnimation(
#             self.fig, self.update_animation, frames=self.app.plotConfig.frames, interval=self.app.plotConfig.fps, )
#         self.format_cube_plot()
#
#     def draw_av(self, *args, **kwargs):
#         x, y, z = self.app.cubeSat.avc
#         self.avText.set_text('$({0}, {1}, {2}) {3}$'.format(round(x, 3), round(y, 3), round(z, 3), 'rad/s'))
#
#     def remove_av(self, *args, **kwargs):
#         self.avText.set_text('')
#
#     def reset_thruster_img(self, thruster):
#         self.thrusterPoints[thruster].set_color(self.app.plotConfig.thrusterColor)
#         self.thrusterPoints[thruster].set_marker('${}$'.format(thruster))
#         self.thrusterPoints[thruster].set_markersize(self.app.plotConfig.ms)
#
#     def fire_thruster_img(self, thruster, pos):
#         self.thrusterPoints[thruster].set_color('r')
#         if pos > 0:
#             shape = "^"
#         else:
#             shape = "v"
#         self.thrusterPoints[thruster].set_marker(shape)
#         self.thrusterPoints[thruster].set_markersize(self.app.plotConfig.ms * 2)
#
#     def remove_axes_quiver(self, *args, **kwargs):
#         self.axesQuiverNegative.remove()
#         self.axesQuiverPositive.remove()
#
#     def draw_cube_faces(self, *args, **kwargs):
#         self.poly.set_verts(self.app.cubeSat.verts)
#
#     def remove_cube_faces(self, *args, **kwargs):
#         self.poly.set_verts([])
#
#     def remove_cube_quiver(self, *args, **kwargs):
#         self.cubeQuiverNegative.remove()
#         self.cubeQuiverPositive.remove()
#
#     def draw_axes_quiver(self, *args, **kwargs):
#         self.axesQuiverNegative = self.ax.quiver(
#             *-self.axesQuiverVector, arrow_length_ratio=0.1, color='yellow', lw=self.app.plotConfig.axesQuiverWidth,
#             linestyle=(0, (2, 5)), )
#         self.axesQuiverPositive = self.ax.quiver(
#             *self.axesQuiverVector, arrow_length_ratio=0.1, color='green', lw=self.app.plotConfig.axesQuiverWidth,
#             linestyle=(0, (2, 5)), )
#
#     def draw_cube_quiver(self, *args, **kwargs):
#         self.cubeQuiverNegative = self.ax.quiver(
#             *self.app.cubeSat.quiverVector / 5, arrow_length_ratio=0.15, color='blue',
#             lw=self.app.plotConfig.cubeQuiverWidth, zorder=100, )
#         self.cubeQuiverPositive = self.ax.quiver(
#             *-self.app.cubeSat.quiverVector / 5, arrow_length_ratio=0.15, color='red',
#             lw=self.app.plotConfig.cubeQuiverWidth, zorder=100, )
#
#     def draw_thrusters(self, ):
#         for thruster in self.app.cubeSat.thrusters:
#             x, y, z = self.app.cubeSat.thrusters[thruster].location.as_list()
#             self.thrusterPoints[thruster].set_data(x, y)
#             self.thrusterPoints[thruster].set_3d_properties(z)
#
#     def hide_thrusters(self, *args, **kwargs):
#         try:
#             for thruster in self.app.cubeSat.thrusters:
#                 self.thrusterPoints[thruster].set_color((0, 0, 0, 0))
#         except Exception as e:
#             print('Hide thruster error: ' + str(e))
#
#     def show_thrusters(self, *args, **kwargs):
#         for thruster in self.app.cubeSat.thrusters:
#             self.thrusterPoints[thruster].set_color(self.app.palette.thrusterColor)
#
#     def draw_wireframe(self, *args, **kwargs):
#         for face in self.faces:
#             x, y, z = self.app.cubeSat.wf(face)
#             self.faces[face].set_data(x, y)
#             self.faces[face].set_3d_properties(z)
#
#     def remove_wireframe(self, *args, **kwargs):
#         for face in self.faces:
#             _0 = np.array([0])
#             self.faces[face].set_data(_0, _0)
#             self.faces[face].set_3d_properties(_0)
#
#     def unused_rendering_code(self, *args, **kwargs):
#         if self.app.cubeSat.atc == self.app.cubeSat.atcm:
#             self.reset_thruster_img(self.app.cubeSat.activeThruster)
#         if (self.app.cubeSat.activeThruster != None) and (self.app.cubeSat.atc == 1):
#             for thruster in self.thrusterPoints:
#                 self.reset_thruster_img(thruster)
#             self.fire_thruster_img(
#                 self.app.cubeSat.activeThruster, self.app.cubeSat.thrusters[self.app.cubeSat.activeThruster].location.z
#             )
#
#     def wott(self, axis):
#         a = getattr(self.app.deviceConfig, axis)
#         if a not in [None, 'None', '']:
#             return float(self.app.dataDude.get(a)[self.app.dataConfig.buffer])
#         return 0
#
#     @lozoya.decorators.catch_error('deviceMenu', *configuration.formatDevicePlot)
#     def format_cube_plot(self, *args, **kwargs):
#         for spine in self.ax.spines:
#             self.ax.spines[spine].set_color('w')
#         self.fig.set_facecolor('black')
#         self.ax.tick_params(colors='w')
#         self.ax.patch.set_facecolor('black')
#         self.ax.set_facecolor('black')
#         self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#         self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
#         self.ax.zaxis.set_major_locator(MaxNLocator(integer=True))
#         self.ax.set_xlim3d(self.app.plotConfig.lims)
#         self.ax.set_xlabel('x', color='w')
#         self.ax.set_ylim3d(self.app.plotConfig.lims)
#         self.ax.set_ylabel('y', color='w')
#         self.ax.set_zlim3d(self.app.plotConfig.lims)
#         self.ax.set_zlabel('z', color='w')
#         self.ax.w_xaxis.set_pane_color((1, 1, 1, 0.1))
#         self.ax.w_yaxis.set_pane_color((1, 1, 1, 0.1))
#         self.ax.w_zaxis.set_pane_color((1, 1, 1, 0.1))
#
#     @lozoya.decorators.catch_error('deviceMenu', *configuration.renderCubeError)
#     def render_cube(self, *args, **kwargs):
#         if self.app.deviceConfig.renderWireframe:
#             self.draw_wireframe()
#         if self.app.deviceConfig.renderThrusters:
#             self.draw_thrusters()
#         if self.app.deviceConfig.renderAxesQuiver:
#             self.remove_axes_quiver()
#             self.draw_axes_quiver()
#         if self.app.deviceConfig.renderCubeQuiver:
#             self.remove_cube_quiver()
#             self.draw_cube_quiver()
#         if self.app.deviceConfig.renderFaces:
#             self.draw_cube_faces()
#         if self.app.deviceConfig.renderAV:
#             self.draw_av()
#
#     @lozoya.decorators.catch_error('deviceMenu', *configuration.updateAnimationError)
#     def update_animation(self, i):
#         condition = self.app.transceiverConfig._transceiverActive or self.app.generatorConfig.simulationEnabled
#         if not self.app.deviceMenu.isHidden() and condition:
#             # msg = self.parent.controller.check_press()
#             x = self.wott('x')
#             y = self.wott('y')
#             z = self.wott('z')
#             self.app.cubeSat.device_update(x, y, z)
#             # self.unused_rendering_code()
#             self.render_cube()  # time.sleep(0.006)


# class Logger:
#     def __init__(self, folder, file):
#         self.timeout = 0
#         self.activeDir = ''
#         self.activeLog = ''
#         self.logFolder = folder
#         self.logFile = file
#
#     def get_files(self, dir):
#         return list(reversed(next(os.walk(os.path.join(self.logFolder, dir)))[2]))
#
#     @lozoya.decorators.catch_error('logMenu', *configuration.createLogFileError)
#     def create_log_file(self, *args, **kwargs):
#         path = os.path.join(self.logFolder, self.app.dataDude.date)
#         filename = self.app.dataDude.timestampHMS.replace(':', '_')
#         with open(os.path.join(path, filename), 'a'):
#             self.app.logMenu.update_status(*configuration.createdFile, filename)
#         return filename
#
#     @lozoya.decorators.catch_error('logMenu', *configuration.createLogDirError)
#     def create_log_folder(self, *args, **kwargs):
#         path = os.path.join(self.logFolder, self.app.dataDude.date)
#         if not os.path.exists(path):
#             os.mkdir(path)
#         if len(os.listdir(path)) == 0:
#             self.create_log_file()
#
#     @lozoya.decorators.catch_error('logMenu', *configuration.writeLogError)
#     def write_to_log(self, data):
#         date = self.app.dataDude.date
#         _date = date.replace('_', '/')
#         t = self.app.dataDude.timestampHMSms
#         delim = self.app.dataConfig.delimiter
#         path = os.path.join(configuration.logDirs, date, self.get_files(date)[0])
#         with open(path, 'a+') as f:
#             f.write('{}{}{}{}{}\n'.format(_date, delim, t, delim, data))
#
#     @lozoya.decorators.catch_error('logMenu', *configuration.exportLogError)
#     def export_log(self, *args, **kwargs):
#         names = ['Date', 'Time'] + self.app.dataConfig.head.split(self.app.dataConfig.delimiter)
#         readPath = os.path.join(configuration.logDirs, self.activeDir, self.activeLog)
#         filename = 'ofcg_{}_log_{}'.format(self.activeDir.replace(' ', '_').lower(), self.activeLog)
#         savePath = '{}{}'.format(
#             *QtWidgets.QFileDialog.getSaveFileName(
#                 self.app.logMenu, caption='Save File', directory=os.path.join(os.getcwd(), filename), filter='.xlsx', )
#         )
#         if savePath:
#             self.app.dataDude.export_data(
#                 dataFrame=pd.read_csv(readPath, names=names), savePath=savePath, format='xlsx', )
#             self.app.logMenu.update_status(*configuration.exportLog, savePath)
#         else:
#             self.app.logMenu.update_status(*configuration.exportLogCancel)
#
#     @lozoya.decorators.catch_error('logMenu', *configuration.formatDataError)
#     def _format(self, _list):
#         dtypes = self.app.dataConfig.dataType.split(self.app.dataConfig.delimiter)
#         _ = []
#         for datum, dataType in zip(_list, dtypes):
#             if dataType == 'f':
#                 _.append('{:03.2f}'.format(float(datum)))
#             else:
#                 _.append(datum)
#         return _
#
#     @lozoya.decorators.catch_error('logMenu', *configuration.logMaintenanceError)
#     def log_maintenance(self, *args, **kwargs):
#         _date = self.app.dataDude.date
#         date = _date.split('_')
#         __old = [d for d in os.listdir(configuration.logDirs) if d != 'error_log']
#         if len(__old) == 0:  # there are no log folders !
#             self.create_log_folder()
#             newfile = self.create_log_file()
#             self.app.logMenu.logFileSelector.insertItems(0, [newfile])
#             self.timeout = 0
#             return
#         _old = __old[-1]
#         old = _old.split('_')
#         _ = os.listdir(os.path.join(configuration.logDirs, _old))[-1]
#         full_ = os.path.join(configuration.logDirs, _old, _)
#         timeSinceModified = os.path.getmtime(full_)
#         self.timeout += (self.app.dataDude.time - timeSinceModified)
#         if (int(date[1]) > int(old[1])) or ((int(date[1]) < int(old[1])) and int(date[0]) > int(old[0])):
#             self.create_log_folder()
#             self.app.logMenu.update_dir_list()
#         if self.timeout >= self.app.logConfig.logCreationInterval:
#             self.create_log_file()
#             self.app.logMenu.update_log_list(self.app.logMenu.logFileSelector.currentRow())
#             self.timeout = 0
#             if self.app.logConfig._autoUpdate:
#                 self.app.logMenu.logFileSelector.item(0).setSelected(True)
#                 self.app.logMenu.logFileSelector.setCurrentRow(0)
#
#     @property
#     def activeIsLatest(self, *args, **kwargs):
#         if self.app.logMenu.logDirSelector.currentRow() == 0 and self.app.logMenu.logFileSelector.currentRow() == 0:
#             return True
#         return False
#
#     @property
#     def activeLogContent(self):
#         with open(self.activeLogPath, 'r') as f:
#             txt = f.read().split('\n')
#         for i, line in enumerate(txt):  # remove empty rows
#             if line == '':
#                 del txt[i]
#         return txt
#
#     @property
#     def activeLogContent(self, *args, **kwargs):
#         with open(self.activeLogPath, 'r') as f:
#             txt = f.read().split('\n')
#         return [t for t in txt if t != '']
#
#     @property
#     def activeLogPath(self, *args, **kwargs):
#         return os.path.join(configuration.logDirs, self.activeDir, self.activeLog)
#
#     @property
#     def activeLogTitle(self, *args, **kwargs):
#         return '{} {}'.format(self.activeDir.replace('_', '/'), self.activeLog.replace('_', ':'))
#
#     @property
#     def dirs(self, *args, **kwargs):
#         return [d for d in list(reversed(sorted(next(os.walk(configuration.logDirs))[1]))) if
#                 d != 'error_log']
#
#     @property
#     def files(self, *args, **kwargs):
#         return list(reversed(sorted(self.get_files(self.activeDir))))


# class Plotter:
#     def __init__(self, parent):
#         self.buffer = 100
#         self.parent = parent
#         self.frames = None
#         A = -self.buffer
#         B = None
#         self._x, self._y, self._z = 0, 0, 0
#         _range = range(len(self.parent.dataDude.timestamps[A:B]))
#         # IMU
#         self.velocityX = self.get_plot(
#             x=_range, y=self.parent.dataDude.velocityX[A:B], updateFunction=self.velocityx_update,
#             plotColor=colors.plotColors['blue'], )
#
#         self.accelerationX = self.get_plot(
#             x=_range, y=self.parent.dataDude.accelerationX[A:B], updateFunction=self.accelerationx_update,
#             plotColor=colors.plotColors['green'], )
#
#         # OBC
#         self.epsCurrent = self.get_plot(
#             x=_range, y=self.parent.dataDude.epsCurrent[A:B], updateFunction=self.epscurrent_update,
#             plotColor=colors.plotColors['orange'], )
#
#         self.epsVoltage = self.get_plot(
#             x=_range, y=self.parent.dataDude.epsVoltage[A:B], updateFunction=self.epsvoltage_update,
#             plotColor=colors.plotColors['red'], )
#
#         self.pressure = self.get_plot(
#             x=_range, y=self.parent.dataDude.pressure[A:B], updateFunction=self.pressure_update,
#             plotColor=colors.plotColors['purple'], )
#
#         self.temperature = self.get_plot(
#             x=_range, y=self.parent.dataDude.temperature[A:B], updateFunction=self.temperature_update,
#             plotColor=colors.plotColors['pink'], )
#
#         self.device = self.get_device_plot(updateFunction=self.update_animation, )
#
#     def accelerationx_update(self, i):
#         m = i % len(self.parent.dataDude.timestamps)
#         if m > self.buffer:
#             chunk = self.parent.dataDude.accelerationX[m - self.buffer: m]
#             self.accelerationX['plot0'].set_ydata(chunk)
#
#     def device_update(self, *args):
#         msg, timeMS = args
#         try:
#             t = int(str(timeMS)[-4:-2])
#             msg = str(msg.split(r"b'")[1])
#             msg = str(msg.split(r'\r')[0])
#             self.x, self.y, self.z = [round(float(_)) for _ in msg.split(',')]
#             self.x, self.y = 0, 0
#             self.x = -0.0174533 * self.x
#             self.y = -0.0174533 * self.y
#             self.z = -0.0174533 * self.z
#             if self.x != self._x:
#                 dx = self.x - self._x
#                 self._x = self.x
#                 self.x = dx
#                 self.parent.cubeSat.set_avc(self.x, self.y, self.z)
#                 print(self.parent.cubeSat.avc)
#                 self.parent.cubeSat.rotate()
#                 self.parent.cubeSat.set_avc(0, 0, 0)
#             else:
#                 self._x = self.x
#
#             if self.y != self._y:
#                 dy = self.y - self._y
#                 self._y = self.y
#                 self.y = dy
#                 self.parent.cubeSat.set_avc(self.x, self.y, self.z)
#                 print(self.parent.cubeSat.avc)
#                 self.parent.cubeSat.rotate()
#                 self.parent.cubeSat.set_avc(0, 0, 0)
#             else:
#                 self._y = self.y
#
#             if self.z != self._z:
#                 dz = self.z - self._z
#                 self._z = self.z
#                 self.z = dz
#                 self.parent.cubeSat.set_avc(self.x, self.y, self.z)
#                 print(self.parent.cubeSat.avc)
#                 self.parent.cubeSat.rotate()
#                 self.parent.cubeSat.set_avc(0, 0, 0)
#             else:
#                 self._z = self.z
#             if self.parent.configuration.displayAV:
#                 orientation = self.parent.cubeSat.avc
#                 self.graphics['avText'].set_text("Angular Velocity: {}".format(orientation))
#         except Exception as e:
#             print(e)
#
#     def draw_axes_quiver(self):
#         self.graphics['axesQuiverNegative'] = self.graphics['ax'].quiver(
#             *-self.axesQuiverVector, arrow_length_ratio=0.1, color='yellow', lw=0.25, linestyle='--', )
#         self.graphics['axesQuiverPositive'] = self.graphics['ax'].quiver(
#             *self.axesQuiverVector, arrow_length_ratio=0.1, color='green', lw=0.25, linestyle='--', )
#
#     def draw_cube_quiver(self):
#         self.graphics['cubeQuiverNegative'] = self.graphics['ax'].quiver(
#             *self.parent.cubeSat.cubeQuiverVector, arrow_length_ratio=0.15, color='blue', lw=0.5, zorder=100, )
#         self.graphics['cubeQuiverPositive'] = self.graphics['ax'].quiver(
#             *-self.parent.cubeSat.cubeQuiverVector, arrow_length_ratio=0.15, color='red', lw=0.5, zorder=100, )
#
#     def draw_thrusters(self, ):
#         for thruster in self.parent.cubeSat.thrusters:
#             x, y, z = self.parent.cubeSat.thrusters[thruster].location.as_list()
#             self.thrusterPoints[thruster].set_data(x, y)
#             self.thrusterPoints[thruster].set_3d_properties(z)
#
#     def draw_wireframe(self):
#         for face in self.graphics['faces']:
#             x, y, z = self.parent.cubeSat.wf(face)
#             self.graphics['faces'][face].set_data(x, y)
#             self.graphics['faces'][face].set_3d_properties(z)
#
#     def epscurrent_update(self, i):
#         m = i % len(self.parent.dataDude.timestamps)
#         if m > self.buffer:
#             chunk = self.parent.dataDude.epsCurrent[m - self.buffer: m]
#             self.epsCurrent['plot0'].set_ydata(chunk)
#
#     def epsvoltage_update(self, i):
#         m = i % len(self.parent.dataDude.timestamps)
#         if m > self.buffer:
#             chunk = self.parent.dataDude.epsVoltage[m - self.buffer: m]
#             self.epsVoltage['plot0'].set_ydata(chunk)
#
#     def fire_thruster_img(self, thruster, pos):
#         self.thrusterPoints[thruster].set_color('r')
#         if pos > 0:
#             shape = "^"
#         else:
#             shape = "v"
#         self.thrusterPoints[thruster].set_marker(shape)
#         self.thrusterPoints[thruster].set_markersize(self.parent.configuration.ms * 2)
#
#     def format_cube_plot(self, p):
#         for spine in p['ax'].spines:
#             p['ax'].spines[spine].set_color('w')
#         p['ax'].tick_params(colors='w')
#         p['fig'].patch.set_facecolor(colors.PLOT_BACKGROUND)
#         p['ax'].set_facecolor(colors.PLOT_BACKGROUND)
#         p['ax'].xaxis.set_major_locator(MaxNLocator(integer=True))
#         p['ax'].yaxis.set_major_locator(MaxNLocator(integer=True))
#         p['ax'].zaxis.set_major_locator(MaxNLocator(integer=True))
#         p['ax'].set_xlim3d(variables.lims)
#         p['ax'].set_xlabel('x', color='w')
#         p['ax'].set_ylim3d(variables.lims)
#         p['ax'].set_ylabel('y', color='w')
#         p['ax'].set_zlim3d(variables.lims)
#         p['ax'].set_zlabel('z', color='w')
#         # p['ax'].set_title('OF-CG Attitude Control', color='w')
#         p['ax'].w_xaxis.set_pane_color((1, 1, 1, 0.1))
#         p['ax'].w_yaxis.set_pane_color((1, 1, 1, 0.1))
#         p['ax'].w_zaxis.set_pane_color((1, 1, 1, 0.1))
#
#     def format_plot(self, p):
#         for spine in p['ax'].spines:
#             p['ax'].spines[spine].set_color('w')
#         p['ax'].tick_params(colors='w')
#         p['fig'].patch.set_facecolor(colors.PLOT_BACKGROUND)
#         p['ax'].set_facecolor(colors.PLOT_BACKGROUND)
#         for i in range(0, 100, 10):
#             p['ax'].axvline(x=i, color='0.2' if not i % 20 else '0.15', ls='-' if not i % 20 else '--', )
#         for i in range(-1000, 1000, 25):
#             p['ax'].axhline(y=i, color='0.2' if not i % 50 else '0.15', ls='-' if not i % 50 else '--', )
#
#     def pressure_update(self, i):
#         m = i % len(self.parent.dataDude.timestamps)
#         if m > self.buffer:
#             chunk = self.parent.dataDude.pressure[m - self.buffer: m]
#             self.pressure['plot0'].set_ydata(chunk)
#
#     def get_device_plot(self, updateFunction):
#         fig = Figure()
#         canvas = FigureCanvas(fig)
#         ax = fig.add_subplot(111, projection='3d')
#         self.axesQuiverVector = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], ])
#
#         uFace, = ax.plot([], [], [], lw=0.25, )
#         dFace, = ax.plot([], [], [], lw=0.25, )
#         lFace, = ax.plot([], [], [], lw=0.25, )
#         rFace, = ax.plot([], [], [], lw=0.25, )
#         fFace, = ax.plot([], [], [], lw=0.25, )
#         bFace, = ax.plot([], [], [], lw=0.25, )
#         avText = ax.text(-1, -1, -1, '', color='white')
#         thrusterRF, = ax.plot([], [], [], colors.thrusterColor, marker='$rf$', markersize=variables.ms, )
#         thrusterRB, = ax.plot([], [], [], colors.thrusterColor, marker='$rb$', markersize=variables.ms, )
#         thrusterLF, = ax.plot([], [], [], colors.thrusterColor, marker='$lf$', markersize=variables.ms, )
#         thrusterLB, = ax.plot([], [], [], colors.thrusterColor, marker='$lb$', markersize=variables.ms, )
#         axesQuiverPositive = ax.quiver(
#             *self.axesQuiverVector, arrow_length_ratio=0.1, color='green', lw=0.25, linestyle='--', )
#         axesQuiverNegative = ax.quiver(
#             *-self.axesQuiverVector, arrow_length_ratio=0.1, color='yellow', lw=0.25, linestyle='--', )
#
#         cubeQuiverPositive = ax.quiver(
#             *self.parent.cubeSat.cubeQuiverVector, arrow_length_ratio=0.15, color='blue', lw=1, zorder=100, )
#         cubeQuiverNegative = ax.quiver(
#             *-self.parent.cubeSat.cubeQuiverVector, arrow_length_ratio=0.15, color='red', lw=1, zorder=100, )
#         faces = {'up': uFace, 'down': dFace, 'left': lFace, 'right': rFace, 'back': bFace, 'front': fFace, }
#         self.thrusterPoints = {'rf': thrusterRF, 'rb': thrusterRB, 'lf': thrusterLF, 'lb': thrusterLB, }
#
#         poly = Poly3DCollection([], facecolors=self.parent.cubeSat.facecolors, )
#         ax.add_collection3d(poly, zs=range(6))
#         anim = animation.FuncAnimation(
#             fig, updateFunction, frames=self.frames, interval=self.parent.configuration.fps, )
#         p = {'fig': fig, 'canvas': canvas, 'ax': ax, 'anim': anim, }
#         self.format_cube_plot(p)
#         self.graphics = {
#             'ax'                : ax, 'faces': faces, 'poly': poly, 'avText': avText,
#             'axesQuiverNegative': axesQuiverNegative, 'axesQuiverPositive': axesQuiverPositive,
#             'cubeQuiverNegative': cubeQuiverNegative, 'cubeQuiverPositive': cubeQuiverPositive,
#         }
#         self._x, self._y, self._z = 0, 0, 0
#         return p
#
#     def get_plot(self, x, y, updateFunction, plotColor='#e27d60'):
#         fig = Figure()
#         canvas = FigureCanvas(fig)
#         ax = fig.add_subplot(111)
#         plot, = ax.plot(x, y, color=plotColor, zorder=3, )
#         anim = animation.FuncAnimation(fig, updateFunction, frames=self.frames, interval=self.parent.configuration.fps)
#         p = {'fig': fig, 'canvas': canvas, 'ax': ax, 'plot0': plot, 'anim': anim, }
#         self.format_plot(p)
#         return p
#
#     def reset_thruster_img(self, thruster):
#         self.thrusterPoints[thruster].set_color(self.parent.configuration.thrusterColor)
#         self.thrusterPoints[thruster].set_marker('${}$'.format(thruster))
#         self.thrusterPoints[thruster].set_markersize(self.parent.configuration.ms)
#
#     def remove_axes_quiver(self):
#         self.graphics['axesQuiverNegative'].remove()
#         self.graphics['axesQuiverPositive'].remove()
#
#     def remove_cube_quiver(self):
#         self.graphics['cubeQuiverNegative'].remove()
#         self.graphics['cubeQuiverPositive'].remove()
#
#     def temperature_update(self, i):
#         m = i % len(self.parent.dataDude.timestamps)
#         if m > self.buffer:
#             chunk = self.parent.dataDude.temperature[m - self.buffer: m]
#             self.temperature['plot0'].set_ydata(chunk)
#
#     def velocityx_update(self, i):
#         m = i % len(self.parent.dataDude.timestamps)
#         if m > self.buffer:
#             chunk = self.parent.dataDude.velocityX[m - self.buffer: m]
#             self.velocityX['plot0'].set_ydata(chunk)
#
#     def update_animation(self, i):
#         try:
#             msg = self.parent.controller.check_press(self.parent.cubeSat, self.parent.configuration, self.graphics, )
#             # self.device_update('', '')
#             if self.parent.cubeSat.atc == self.parent.cubeSat.atcm:
#                 self.reset_thruster_img(self.parent.cubeSat.activeThruster)
#             if (self.parent.cubeSat.activeThruster != None) and (self.parent.cubeSat.atc == 1):
#                 for thruster in self.thrusterPoints:
#                     self.reset_thruster_img(thruster)
#                 self.fire_thruster_img(
#                     self.parent.cubeSat.activeThruster,
#                     self.parent.cubeSat.thrusters[self.parent.cubeSat.activeThruster].location.z
#                 )
#             self.parent.cubeSat.rotate()
#             if self.parent.configuration.displayWireframe:
#                 self.draw_wireframe()
#             if self.parent.configuration.displayThrusters:
#                 self.draw_thrusters()
#             if self.parent.configuration.displayAxesQuiver:
#                 self.remove_axes_quiver()
#                 self.draw_axes_quiver()
#             if self.parent.configuration.displayCubeQuiver:
#                 self.remove_cube_quiver()
#                 self.draw_cube_quiver()
#             if self.parent.configuration.displayAV:
#                 orientation = self.parent.cubeSat.avc
#                 self.graphics['avText'].set_text("Angular Velocity: {}".format(orientation))
#             if self.parent.configuration.displayFaces:
#                 self.graphics['poly'].set_verts(self.parent.cubeSat.verts)
#         except Exception as e:
#             print(e)


# class Thruster:
#     def __init__(self, name, location=lozoya.math_api.origin):
#         self.active = False
#         self.name = name
#         self.initialLocation = lozoya.math_api.Coordinate(*location)
#         self.location = lozoya.math_api.Coordinate(*location)
#         self.thrustForce = 1 / 10
#
#     @property
#     def thrustDirection(self, *args, **kwargs):
#         f = self.face
#         if f == 'lfc':
#             return 0, 0, -self.thrustForce
#         if f == 'rbc':
#             return 0, 0, -self.thrustForce
#         if f == 'lbc':
#             return 0, 0, self.thrustForce
#         if f == 'rfc':
#             return 0, 0, self.thrustForce
#         if f == 'lfd':
#             return self.thrustForce / 2, -self.thrustForce / 2, 0
#         if f == 'rbd':
#             return -self.thrustForce / 2, self.thrustForce / 2, 0
#         if f == 'lbd':
#             return -self.thrustForce / 2, -self.thrustForce / 2, 0
#         if f == 'rfd':
#             return self.thrustForce / 2, self.thrustForce / 2, 0
#         return 0, 0, 0
#
#     @property
#     def face(self, *args, **kwargs):
#         x, y, z = self.initialLocation.as_list()
#         if x > 0:
#             cx = 'r'
#         elif x < 0:
#             cx = 'l'
#         else:
#             cx = 'c'
#         if y > 0:
#             cy = 'f'
#         elif y < 0:
#             cy = 'b'
#         else:
#             cy = 'c'
#         if z > 0:
#             cz = 'u'
#         elif z < 0:
#             cz = 'd'
#         else:
#             cz = 'c'
#         return '{}{}{}'.format(cx, cy, cz)


# class Transceiver(QtCore.QObject):
#     def __init__(self, app, debug=False):
#         super(Transceiver, self).__init__()
#         self.app = app
#         self.debug = debug
#         self.bands = [lozoya.communication_api.Longwave(), lozoya.communication_api.AMRadio(),
#                       lozoya.communication_api.Shortwave(), lozoya.communication_api.VHFLow(),
#                       lozoya.communication_api.FMRadio(), lozoya.communication_api.VHFHigh(), lozoya.communication_api.UHF(),
#                       lozoya.communication_api.SBand(), lozoya.communication_api.XBand(), ]
#         self.baudRates = ['9600', '115200']
#
#     def start_listener(self, callbackFunc):
#         self.listener = threading.Thread(name='transceiver', target=self.transceiver_thread, args=(), kwargs={}, )
#         self.listener.daemon = True
#         self.listener.start()
#
#     def transceiver_thread(self):
#         """
#         This function is an infinite loop that awaits for a signal from the cubesat.
#         When the signal is received, it will send the data in the signal to the Logger.
#         The Logger will parse the data and write each type of data to the corresponding log.
#         Afterwards, the data is sent to the DataDude who will update its pandas DataFrames.
#         The data is displayed in the transceiver 'Received' text area, overwritting any previous display.
#         If automatic updating is enabled in the Log menu, the Log Reader Panel will be updated.
#         """
#         for i in count(0):
#             try:
#                 if self.app.transceiverConfig._transceiverActive:
#                     msg = self.app.signalGenerator.simulate_signal(i)
#                     self.app.update_everything(i, msg)
#                 time.sleep(0.005)
#             except Exception as e:
#                 print('Transceiver thread error: ' + str(e))
#
#     @property
#     def bandNames(self):
#         return [band.label for band in self.bands]
#
#     def transmit(self, msg):
#         try:
#             if self.app.transceiverConfig._transceiverActive:
#                 if not msg:
#                     return
#                 eMsg = msg
#                 if eMsg:
#                     self.app.client.send_message(eMsg)
#                 else:
#                     self.app.client.send_message(msg)
#                 self.echo(e, msg, eMsg)
#             else:
#                 self.app.update_status('Must enable transceiver to send messages.', 'alert')
#         except Exception as e:
#             self.app.transceiverMenu.update_status(*configuration.transmissionError, e)
#
#     def echo(self, e, msg, encrypted=None):
#         statusMsg = 'Sent unencrypted: {}'.format(msg)
#         self.app.transceiverMenu.update_status(statusMsg, 'success')

# class MenuPlot(lozoya.gui.TSMenu):
#     def __init__(self, app, title, icon):
#         lozoya.gui.TSMenu.__init__(self, app, title, icon, )
#         try:
#             self.settingsForm = settingsform.SettingsForm(self.app, self, height=54)
#             self._layout.addWidget(self.settingsForm.frame)
#             self.plotSettings = plotsettings.SettingsPlot(self.app, self, self.settingsForm.layout)
#             self.plotFrame, self.plotLayout = lozoya.gui.subdivision(self)
#             self._layout.addWidget(self.plotFrame)
#             self.setMinimumSize(500, 500)
#             self.resize(500, 500)
#             self.plotLayout.addWidget(self.app.plotDude.plots[0].canvas)
#             # self.setMaximumSize(750, 750)
#             self.plotSettings.update_plot_xlim(0)
#             self.plotSettings.update_plot_ylim(0)
#             self.plotSettings.change_y(0)
#             self.set_menubar()
#             self.leftFrame, self.leftLayout = lozoya.gui.subdivision(self)
#             self.plotSettingsFrame, self.plotSettingsLayout = lozoya.gui.subdivision(self)
#             self.plotFrame, self.velocityXPlotLayout = lozoya.gui.subdivision(self)
#             self.plotSplitter = lozoya.gui.make_splitter(
#                 style='v', widgets=(self.plotSettingsFrame, self.plotFrame)
#             )
#             self.plotContainer = lozoya.gui.make_group_box(
#                 parent=self.leftLayout, text='Velocity', row=0, column=0, )
#             self.plotSettings = lozoya.gui.make_group_box(
#                 parent=self.plotSettingsLayout, text='Settings', height=100, row=0, column=0, )
#             self.plotDomainLabel = lozoya.gui.make_label(
#                 self, text='Domain', layout=self.plotSettings.layout(), width=60, row=0, column=0,
#                 tooltip=tooltips.plotRange, )
#             self.plotDomainSlider = lozoya.gui.make_slider(
#                 self, inputSettings=(0, self.parent.preferences.buffer - 10, self.parent.preferences.velocityXMinY, 1),
#                 command=lambda: self.update_plot_xlim(0), layout=self.plotSettings.layout(), row=0, column=1,
#                 tooltip=tooltips.minYSlider, )
#             self.plotRangeLabel = lozoya.gui.make_label(
#                 self, text='Range', layout=self.plotSettings.layout(), width=60, row=1, column=0,
#                 tooltip=tooltips.plotRange, )
#             self.plotMinYSlider = lozoya.gui.make_slider(
#                 self, inputSettings=(-10000, 10000, self.parent.preferences.velocityXMinY, 1),
#                 command=lambda: self.update_plot_ylim(0), layout=self.plotSettings.layout(), row=1, column=1,
#                 tooltip=tooltips.minYSlider, )
#             self.plotMaxYSlider = lozoya.gui.make_slider(
#                 self, inputSettings=(-10000, 10000, self.parent.preferences.velocityXMaxY, 1),
#                 command=lambda: self.update_plot_ylim(0), layout=self.plotSettings.layout(), row=1, column=2,
#                 tooltip=tooltips.maxYSlider, )
#             _, self.yCombo = lozoya.gui.pair(
#                 self, pair='combo', comboItems=self.parent.preferences.head.split(','), default=3,
#                 layout=self.plotSettings.layout(), command=lambda: self.change_y(0),
#                 **app.cubesat.widgetkwargs.plot.yCombo, )
#             # self.toolsGrid = lozoya.gui.make_group_box(
#             #     self._layout,
#             #     **app.cubesat.widgetkwargs.plot0.toolsGrid,
#             # )
#             # self.addSuplotButton = lozoya.gui.make_button(
#             #     self,
#             #     icon2=icon2.add,
#             #     layout=self.toolsGrid.layout(),
#             #     command=lambda: self.add_plot('0'),
#             #     row=1,
#             #     column=1,
#             #     tooltip='Insert subplot.'
#             # )
#             self.plotContainer.layout().addWidget(self.plotSplitter)
#             self.plotSettingsFrame.setMaximumHeight(120)
#             self.velocityXPlotLayout.addWidget(self.parent.plotDude.plots[0]['canvas'])
#             self._layout.addWidget(self.leftFrame)
#             self.update_plot_ylim(0)
#             self.settingsForm = settingsform.SettingsForm(self.app, self, height=54)
#             self._layout.addWidget(self.settingsForm.frame)
#             self.plotSettings = plotsettings.SettingsPlot(self.app, self, self.settingsForm.layout)
#             self.menuFrame = lozoya.gui.make_frame(self)
#             self.menuContainer = lozoya.gui.make_grid(self.menuFrame)
#             self.plotFrame, self.plotLayout = lozoya.gui.subdivision(self)
#             self.menuContainer.addWidget(self.plotFrame)
#             self._layout.addWidget(self.menuFrame)
#             self.setMinimumSize(500, 500)
#             self.resize(500, 500)
#             self.plotLayout.addWidget(self.app.plotDude.plots[0].canvas)
#             # self.setMaximumSize(750, 750)
#             self.plotSettings.update_plot_xlim(0)
#             self.plotSettings.update_plot_ylim(0)
#             self.plotSettings.change_y(0)
#             self.set_menubar()
#         except Exception as e:
#             print('Plot menu error: ' + str(e))
#
#     def add_plot(self, plot):
#         try:
#             pass
#         except Exception as e:
#             print('Add plot0 error: ' + str(e))
#
#     def change_y(self, var):
#         self.parent.preferences.plotProperties[var]['y'] = self.parent.preferences.head.split(',')[
#             self.yCombo.currentIndex()]
#
#     def closeEvent(self, *args, **kwargs):
#         super(lozoya.gui.TSMenu, self).closeEvent(*args, **kwargs)
#         for plot in self.app.plotDude.plots:
#             plot.anim.event_source.stop()
#
#     def set_menubar(self, *args, **kwargs):
#         settingsIcon = None  # QtGui.QIcon(configuration.cog)
#         settingsAction = lozoya.gui.make_action(
#             'Settings', callback=self.settingsForm.toggle, icon=settingsIcon, )
#         self.menubar.addAction(settingsAction)
#
#     def showEvent(self, *args, **kwargs):
#         super(lozoya.gui.TSMenu, self).showEvent(*args, **kwargs)
#         for plot in self.app.plotDude.plots:
#             plot.anim.event_source.start()
#
#     def update_plot_xlim(self, var):
#         self.parent.plotDude.plots[var]['ax'].set_xlim(
#             xmin=self.plotDomainSlider.value(), xmax=self.parent.preferences.buffer, )
#         self.parent.preferences.plotProperties[var]['domain'] = self.plotDomainSlider.value()
#         self.parent.preferences.save_to_file()
#
#     def update_plot_ylim(self, var):
#         self.parent.plotDude.plots[var]['ax'].set_ylim(
#             ymin=self.plotMinYSlider.value(), ymax=self.plotMaxYSlider.value(), )
#         self.parent.preferences.plotProperties[var]['ymin'] = self.plotMinYSlider.value()
#         self.parent.preferences.plotProperties[var]['ymax'] = self.plotMaxYSlider.value()
#         self.parent.preferences.save_to_file()
#
#
# class MenuDevice(lozoya.gui.TSMenu):
#     def __init__(self, app, title, icon):
#         try:
#             lozoya.gui.TSMenu.__init__(self, app, title, icon, )
#             self.settingsForm = settingsform.SettingsForm(self.app, self, height=75)
#             self._layout.addWidget(self.settingsForm.frame)
#             tabNames = ['Coordinates', 'Rendering']
#             self.tabs = lozoya.gui.set_tabs(self, tabNames, self.settingsForm.layout, )
#             for tab in self.tabs:
#                 self.tabs[tab].layout().setContentsMargins(5, 5, 5, 0)
#             self.deviceCoordinateSettings = devicecoordinatesettings.SettingsDevice(
#                 self.app, self, self.tabs['Coordinates'].layout(), )
#             self.deviceRenderSettings = devicerendersettings.SettingsDeviceRender(
#                 self.app, self, self.tabs['Rendering'].layout(), )
#             self.devicePlotFrame, self.devicePlotLayout = lozoya.gui.subdivision(self)
#             # self.deviceSplitter = lozoya.gui.make_splitter(
#             #     style='v',
#             #     widgets=(self.settingsForm.frame, self.devicePlotFrame)
#             # )
#             self._layout.addWidget(self.devicePlotFrame)
#             self.setMinimumSize(400, 400)
#             self.resize(400, 400)
#             self.devicePlotLayout.addWidget(self.app.plotDude.devicePlot.canvas)
#             self.set_menubar()
#             self.load_menu()
#         except Exception as e:
#             statusMsg = 'Init device menu error: {}.'.format(str(e))
#             self.update_status(statusMsg, 'error')
#         submenu.SubMenu.__init__(self, parent, title, icon, )
#         self.deviceSettingsFrame, self.deviceSettingsLayout = lozoya.gui.make_subdivision(self)
#         self.devicePlotFrame, self.devicePlotLayout = lozoya.gui.make_subdivision(self)
#         self.deviceSplitter = lozoya.gui.make_splitter(
#             style='v', widgets=(self.deviceSettingsFrame, self.devicePlotFrame)
#         )
#         self.deviceContainer = lozoya.gui.make_group_box(
#             parent=self._layout, layout=lozoya.gui.make_grid(), text='Device', row=0, column=0, )
#         self.deviceSettings = lozoya.gui.make_group_box(
#             parent=self.deviceSettingsLayout, layout=lozoya.gui.make_grid(), text='Settings', height=100, row=0,
#             column=0, )
#         _, self.displayFacesCheckBox = lozoya.gui.make_pair(
#             self, pair='check', text='Faces', label=True, row=0, column=0, layout=self.deviceSettings, pairKwargs={
#                 'command'  : self.update_display_faces, 'description': 'Toggle whether the cube faces are rendered.',
#                 'isChecked': True,  # TODO read this from settings file
#                 'layout'   : self.deviceSettings, 'row': 0, 'column': 1,
#             }, )
#         _, self.displayWireframeCheckBox = lozoya.gui.make_pair(
#             self, pair='check', text='Wireframe', label=True, row=0, column=2, layout=self.deviceSettings, pairKwargs={
#                 'command'    : self.update_display_wireframe,
#                 'description': 'Toggle whether the cube wireframe is rendered.', 'isChecked': True,
#                 # TODO read this from settings file
#                 'layout'     : self.deviceSettings, 'row': 0, 'column': 3,
#             }, )
#         _, self.displayThrustersCheckBox = lozoya.gui.make_pair(
#             self, pair='check', text='Thrusters', label=True, row=0, column=4, layout=self.deviceSettings, pairKwargs={
#                 'command'    : self.update_display_thrusters,
#                 'description': 'Toggle whether the cube thrusters are rendered.', 'isChecked': True,
#                 # TODO read this from settings file
#                 'layout'     : self.deviceSettings, 'row': 0, 'column': 5,
#             }, )
#         _, self.displayAxesQuiverCheckBox = lozoya.gui.make_pair(
#             self, pair='check', text='Axes Quiver', label=True, row=1, column=0, layout=self.deviceSettings,
#             pairKwargs={
#                 'command'    : self.update_display_axes_quiver,
#                 'description': 'Toggle whether the axes quiver is rendered.', 'isChecked': True,
#                 # TODO read this from settings file
#                 'layout'     : self.deviceSettings, 'row': 1, 'column': 1,
#             }, )
#         _, self.displayCubeQuiverCheckBox = lozoya.gui.make_pair(
#             self, pair='check', text='Cube Quiver', label=True, row=1, column=2, layout=self.deviceSettings,
#             pairKwargs={
#                 'command'    : self.update_display_cube_quiver,
#                 'description': 'Toggle whether the cube quiver is rendered.', 'isChecked': True,
#                 # TODO read this from settings file
#                 'layout'     : self.deviceSettings, 'row': 1, 'column': 3,
#             }, )
#         self.deviceContainer.addWidget(self.deviceSplitter)
#         self.deviceSettingsFrame.setMaximumHeight(120)
#         self.devicePlotLayout.addWidget(self.parent.plotDude.device['canvas'])
#         self.setLayout(self.hbox)
#         try:
#             lozoya.gui.TSMenu.__init__(self, app, title, icon, )
#             self.settingsForm = settingsform.SettingsForm(self.app, self, height=75)
#             self._layout.addWidget(self.settingsForm.frame)
#             tabNames = ['Coordinates', 'Rendering']
#             self.tabs = lozoya.gui.set_tabs(self, tabNames, self.settingsForm.layout, )
#             for tab in self.tabs:
#                 self.tabs[tab].layout().setContentsMargins(5, 5, 5, 0)
#             self.deviceCoordinateSettings = devicecoordinatesettings.SettingsDevice(
#                 self.app, self, self.tabs['Coordinates'].layout(), )
#             self.deviceRenderSettings = devicerendersettings.SettingsDeviceRender(
#                 self.app, self, self.tabs['Rendering'].layout(), )
#             self.menuFrame = lozoya.gui.make_frame(self)
#             self.menuContainer = lozoya.gui.make_grid(self.menuFrame)
#             self.devicePlotFrame, self.devicePlotLayout = lozoya.gui.subdivision(self)
#             self.menuContainer.addWidget(self.devicePlotFrame)
#             self._layout.addWidget(self.menuFrame)
#             self.setMinimumSize(400, 400)
#             self.resize(400, 400)
#             self.devicePlotLayout.addWidget(self.app.plotDude.devicePlot.canvas)
#             self.set_menubar()
#             self.load_menu()
#         except Exception as e:
#             statusMsg = 'Init device menu error: {}.'.format(str(e))
#             self.update_status(statusMsg, 'error')
#         self.deviceSettingsFrame, self.deviceSettingsLayout = lozoya.gui.subdivision(self)
#         self.devicePlotFrame, self.devicePlotLayout = lozoya.gui.subdivision(self)
#         self.deviceSplitter = lozoya.gui.make_splitter(
#             style='v', widgets=(self.deviceSettingsFrame, self.devicePlotFrame)
#         )
#         self.deviceSettings = lozoya.gui.make_group_box(
#             parent=self.deviceSettingsLayout, **app.cubesat.widgetkwargs.device.deviceSettings
#         )
#         _, self.posXCombo = lozoya.gui.pair(
#             self, pair='combo', comboItems=['None'] + self.parent.preferences.head.split(','),
#             default=self.parent.preferences.posXIndex, layout=self.deviceSettings.layout(),
#             command=lambda: self.update_pos_x(), row=0, column=0, **app.cubesat.widgetkwargs.device.posXCombo, )
#         _, self.posYCombo = lozoya.gui.pair(
#             self, pair='combo', comboItems=['None'] + self.parent.preferences.head.split(','),
#             default=self.parent.preferences.posYIndex, layout=self.deviceSettings.layout(),
#             command=lambda: self.update_pos_y(), row=0, column=1, **app.cubesat.widgetkwargs.device.posYCombo, )
#         _, self.posZCombo = lozoya.gui.pair(
#             self, pair='combo', comboItems=['None'] + self.parent.preferences.head.split(','),
#             default=self.parent.preferences.posZIndex, layout=self.deviceSettings.layout(),
#             command=lambda: self.update_pos_z(), row=0, column=2, **app.cubesat.widgetkwargs.device.posZCombo, )
#         self._layout.addWidget(self.deviceSplitter)
#         self.deviceSettingsFrame.setMaximumHeight(120)
#         self.devicePlotLayout.addWidget(self.parent.plotDude.device['canvas'])
#
#     def closeEvent(self, *args, **kwargs):
#         super(lozoya.gui.TSMenu, self).closeEvent(*args, **kwargs)
#         self.app.plotDude.device['anim'].event_source.stop()
#
#     def load_menu(self):
#         head = ['None'] + self.app.dataDude.get_plottable()
#         if self.app.fullyLoaded:
#             for axis in ['x', 'y', 'z']:
#                 combo = '{}Combo'.format(axis)
#                 getattr(self.deviceCoordinateSettings, combo).clear()
#                 getattr(self.deviceCoordinateSettings, combo).addItems(head)
#                 getattr(self.deviceCoordinateSettings, combo).setCurrentIndex(
#                     getattr(self.app.deviceConfig, '{}Index'.format(axis))
#                 )
#                 getattr(self.deviceCoordinateSettings, combo).setStyleSheet(
#                     lozoya.gui.qss._format(lozoya.gui.styles.selectionstyles.dropdown, self.app.palette)
#                 )
#
#     def reset_menu(self, *args, **kwargs):
#         if self.app.fullyLoaded:
#             head = ['None'] + self.app.dataDude.get_plottable()
#             for axis in ['x', 'y', 'z']:
#                 combo = '{}Combo'.format(axis)
#                 getattr(self.deviceCoordinateSettings, combo).clear()
#                 getattr(self.deviceCoordinateSettings, combo).addItems(head)
#                 getattr(self.deviceCoordinateSettings, combo).setCurrentIndex(0)
#                 getattr(self.deviceCoordinateSettings, combo).setStyleSheet(
#                     lozoya.gui.qss._format(lozoya.gui.styles.selectionstyles.dropdown, self.app.palette)
#                 )
#                 self.app.deviceConfig.update(axis, head[0])
#                 self.app.deviceConfig.update('{}Index'.format(axis), 0)
#         else:
#             for axis in ['x', 'y', 'z']:
#                 _axis = axis + 'Combo'
#                 getattr(self.deviceCoordinateSettings, _axis).clear()
#                 getattr(self.deviceCoordinateSettings, _axis).setStyleSheet(
#                     lozoya.gui.qss._format(lozoya.gui.styles.selectionstyles.dropdownDisabled, self.app.palette)
#                 )
#                 getattr(self, _axis).clear()
#                 getattr(self, _axis).setStyleSheet(
#                     qss._format(interface.styles.selectionstyles.dropdownDisabled, self.app.palette)
#                 )
#                 self.app.deviceConfig.update(axis, None)
#                 self.app.deviceConfig.update('{}Index'.format(axis), None)
#
#     def set_menubar(self, *args, **kwargs):
#         settingsAction = lozoya.gui.make_action(
#             'Settings', callback=self.settingsForm.toggle,  # icon=QtGui.QIcon(configuration.cog),
#         )
#         self.menubar.addAction(settingsAction)
#
#     def showEvent(self, *args, **kwargs):
#         super(lozoya.gui.TSMenu, self).showEvent(*args, **kwargs)
#         self.app.plotDude.device['anim'].event_source.start()
#
#     def update_display_faces(self):
#         if self.parent.preferences.displayFaces == True:
#             self.parent.preferences.displayFaces = False
#             self.parent.plotDude.graphics['poly'].set_verts([])
#         else:
#             self.parent.preferences.displayFaces = True
#
#     def update_display_axes_quiver(self):
#         if self.parent.preferences.displayAxesQuiver == True:
#             self.parent.preferences.displayAxesQuiver = False
#             self.parent.plotDude.remove_axes_quiver()
#
#         else:
#             self.parent.preferences.displayAxesQuiver = True
#             self.parent.plotDude.draw_axes_quiver()
#
#     def update_display_cube_quiver(self):
#         if self.parent.preferences.displayCubeQuiver == True:
#             self.parent.preferences.displayCubeQuiver = False
#             self.parent.plotDude.remove_cube_quiver()
#
#         else:
#             self.parent.preferences.displayCubeQuiver = True
#             self.parent.plotDude.draw_cube_quiver()
#
#     def update_display_wireframe(self):
#         if self.parent.preferences.displayWireframe:
#             self.parent.preferences.displayWireframe = False
#             for face in self.parent.plotDude.graphics['faces']:
#                 self.parent.plotDude.graphics['faces'][face].set_data(0, 0)
#                 self.parent.plotDude.graphics['faces'][face].set_3d_properties(0)
#         else:
#             self.parent.preferences.displayWireframe = True
#
#     def update_display_thrusters(self):
#         try:
#             if self.parent.preferences.displayThrusters:
#                 self.parent.preferences.displayThrusters = False
#                 for thruster in self.parent.cubeSat.thrusters:
#                     self.parent.plotDude.thrusterPoints[thruster].set_color((0, 0, 0, 0))
#             else:
#                 self.parent.preferences.displayThrusters = True
#                 for thruster in self.parent.cubeSat.thrusters:
#                     self.parent.plotDude.thrusterPoints[thruster].set_color(colors.thrusterColor)
#         except Exception as e:
#             print(e)
#
#     def update_pos_x(self):
#         self.parent.preferences.posX = (['None'] + self.parent.preferences.head.split(','))[
#             self.posXCombo.currentIndex()]
#         self.parent.preferences.posXIndex = self.posXCombo.currentIndex()
#         self.parent.preferences.save_to_file()
#
#     def update_pos_y(self):
#         self.parent.preferences.posY = (['None'] + self.parent.preferences.head.split(','))[
#             self.posYCombo.currentIndex()]
#         self.parent.preferences.posYIndex = self.posYCombo.currentIndex()
#         self.parent.preferences.save_to_file()
#
#     def update_pos_z(self):
#         self.parent.preferences.posZ = (['None'] + self.parent.preferences.head.split(','))[
#             self.posZCombo.currentIndex()]
#         self.parent.preferences.posZIndex = self.posZCombo.currentIndex()
#         self.parent.preferences.save_to_file()
#
#
# class MenuDocumentation(lozoya.gui.TSMenu):
#     def __init__(self, app, title, icon):
#         lozoya.gui.TSMenu.__init__(self, app, title, icon, )
#         self.setMinimumSize(300, 500)
#         self.setMaximumSize(780, 535)
#         self.resize(300, 500)
#         self.documentationContainer = lozoya.gui.make_browser(
#             layout=self._layout, file=configuration.docs, )
#         self.documentationContainer.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
#         # self.documentationTab = lozoya.gui.make_scrollarea_tab(
#         #     self,
#         #     text='Documentation5',
#         #     master=self.tabs,
#         # )
#         self.documentationContainer = lozoya.gui.make_browser(
#             layout=self.hbox, file=configuration.DOCUMENTATION, )  # layout=self._layout ?
#         self.setMinimumSize(
#             constants.WIDTH // 3, constants.HEIGHT // 2
#         )  # self.tutorialsTab = lozoya.gui.make_scrollarea_tab(  #     self,  #     text='Tutorials',  #     master=self.tabs,  # )  # self.tutorialsContainer = lozoya.gui.make_browser(  #     layout=self.tutorialsTab,  #     file=None,  # )
#         self.menuFrame = lozoya.gui.make_frame(self)
#         self.menuContainer = lozoya.gui.make_grid(self.menuFrame)
#         self.docBrowser = lozoya.gui.make_browser(parent=self, layout=self.menuContainer, )
#         self.docBrowser.read_file(configuration.docs)
#         self.docBrowser.disable_right_click()
#         self.menuContainer.addWidget(self.docBrowser)
#         self._layout.addWidget(self.menuFrame)
#         # self.documentationTab = lozoya.gui.make_scrollarea_tab(
#         #     self,
#         #     text='1',
#         #     master=self.tabs,
#         # )
#         self.documentationContainer.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
#         self.setMinimumSize(
#             self.parent.palette.width // 3, self.parent.palette.height // 2
#         )  # self.tutorialsTab = lozoya.gui.make_scrollarea_tab(  #     self,  #     text='Tutorials',  #     master=self.tabs,  # )  # self.tutorialsContainer = lozoya.gui.make_browser(  #     layout=self.tutorialsTab,  #     file=None,  # )
#
#
# class MenuLog(lozoya.gui.TSMenu):
#     def __init__(self, app, title, icon):
#         lozoya.gui.TSMenu.__init__(self, app, title, icon, )
#         self.settingsForm = SettingsForm(self.app, self, height=54)
#         self._layout.addWidget(self.settingsForm.frame)
#         self.logSettings = SettingsLog(self.app, self, self.settingsForm.layout, )
#         self.menuFrame = lozoya.gui.make_frame(self)
#         self.menuContainer = lozoya.gui.make_grid(self.menuFrame)
#         self.logDirSelectorGrid = lozoya.gui.make_group_box(
#             self.menuContainer, **app.cubesat.widgetkwargs.log.logDirsGrid
#         )
#         self.logFileSelectorGrid = lozoya.gui.make_group_box(
#             self.menuContainer, **app.cubesat.widgetkwargs.log.logsGrid
#         )
#         self.logReaderGrid = lozoya.gui.make_group_box(
#             self.menuContainer, **app.cubesat.widgetkwargs.log.logReaderGrid
#         )
#         self.logDirSelector = lozoya.gui.make_list_widget(
#             command=lambda: self.update_log_list(), layout=self.logDirSelectorGrid.layout(),
#             tooltip=tooltips.logDirSelector, )
#         self.logFileSelector = lozoya.gui.make_list_widget(
#             command=self.refresh, layout=self.logFileSelectorGrid.layout(), tooltip=tooltips.logFileSelector, )
#         self.logReaderTxt = lozoya.gui.make_log_table(
#             layout=self.logReaderGrid.layout(), tooltip=tooltips.logReader
#         )
#         self._layout.addWidget(self.menuFrame)
#         self.setMinimumSize(500, 250)
#         self.resize(500, 250)
#         self.frame.setMinimumWidth(740)
#         self.setMinimumWidth(810)
#         self._logDir = ''
#         self._log = ''
#         self.make_log_browser()
#         self.make_log_reader()
#         self.set_stylesheets()
#         self.set_menubar()
#         # try:
#         #     self.frame.deleteLater()
#         #     self._layout.deleteLater()
#         # except:
#         #     pass
#         self.setLayout(self.hbox)
#
#     def export_results(self):
#         try:
#             names = self.get_header()
#             _pathC = os.path.join(configuration.LOG_DIRS, self._logDir, self._log)
#             _pathB = 'ofcg_{}_log_{}'.format(self._logDir.replace(' ', '_').lower(), self._log)
#             _pathA = QtWidgets.QFileDialog.getSaveFileName(
#                 self, caption='Save File', directory=os.path.join(os.getcwd(), _pathB), filter='.xlsx', )
#             path = '{}{}'.format(_pathA[0], _pathA[1])
#             print(path)
#             lozoya.data_api.export_data(dataFrame=pd.read_csv(_pathC, names=names), savePath=path, format='xlsx', )
#         except Exception as e:
#             print(e)
#
#     def format_(self, txt):
#         try:
#             newSep = '      '
#             for i, t in enumerate(txt):
#                 m = t.split(variables.LOG_SEPARATOR)
#                 m = [r(m[j]) for j, r in enumerate(types[self._logDir])]
#                 u = sizes2[self._logDir]
#                 z = u.format(*m)
#                 z = z.replace(variables.LOG_SEPARATOR, newSep)
#                 txt[i] = z
#         except Exception as e:
#             print(e)
#         return txt
#
#     def get_header(self):
#         path = os.path.join(configuration.DATABASE, '{}_header'.format(self._logDir[:3].lower()))
#         with open(path, 'r') as f:
#             header = f.readline().split(',')
#         return header
#
#     def make_log_browser(self):
#         self._logDirs = next(os.walk(configuration.LOG_DIRS))[1]
#         self.logDirsGrid = lozoya.gui.make_group_box(
#             self._layout, layout=lozoya.gui.make_grid(), text='Log Directory', row=1, column=0, width=150, )
#
#         self.logsGrid = lozoya.gui.make_group_box(
#             self._layout, layout=lozoya.gui.make_grid(), text='Log', row=1, column=1, width=150, )
#
#         self.logDirs = lozoya.gui.make_list(
#             items=self._logDirs, command=lambda: self.update_log_list(), layout=self.logDirsGrid, multiSelect=False, )
#         self.logDirs.setCurrentRow(0)
#
#         firstLogDir = os.path.join(configuration.LOG_DIRS, self._logDirs[0])
#         self._logs = next(os.walk(firstLogDir))[2]
#         self.logs = lozoya.gui.make_list(
#             items=self._logs, command=lambda: self.update_log(), layout=self.logsGrid, multiSelect=False, )
#         self.logs.setCurrentRow(0)
#
#     def make_log_reader(self):
#         logPath = os.path.join(configuration.LOG_DIRS, self._logDir, self._log)
#         with open(logPath, 'r') as f:
#             txt = f.read()
#             txt = txt.split('\n')
#         txt = self.format_(txt)
#         _file = '{0} Log: {1}'.format(self.logDirs.currentItem().text(), self._log)
#         label = logLabels[self._logDir]
#         self.exportButton = lozoya.gui.make_button(
#             self, text='Export', command=self.export_results, layout=self._layout, row=0, column=1, width=50, )
#         self.logLabel = lozoya.gui.make_label(self, text=_file, layout=self._layout, row=0, column=2, width=250, )
#         self.logViewerGrid = lozoya.gui.make_group_box(
#             self._layout, layout=lozoya.gui.make_grid(), text=label, row=1, column=2, )
#         self.logViewerText = lozoya.gui.make_list(items=txt, layout=self.logViewerGrid, multiSelect=True, )
#
#     def set_menubar(self, *args, **kwargs):
#         settingsAction = QtWidgets.QAction(QtGui.QIcon(configuration.cog), 'Settings', self)
#         refreshAction = QtWidgets.QAction(QtGui.QIcon(configuration.refreshButton), 'Refresh', self)
#         exportAction = QtWidgets.QAction(QtGui.QIcon(configuration.exportButton), 'Export', self)
#         autoUpdateAction = QtWidgets.QAction(QtGui.QIcon(configuration.downArrow), 'Auto Update', self)
#         settingsAction.triggered.connect(self.settingsForm.toggle)
#         refreshAction.triggered.connect(self.refresh)
#         exportAction.triggered.connect(self.app.logger.export_log)
#         autoUpdateAction.triggered.connect(self.toggle_auto_scroll)
#         self.menubar.addAction(settingsAction)
#         self.menubar.addAction(refreshAction)
#         self.menubar.addAction(exportAction)
#         self.menubar.addAction(autoUpdateAction)
#
#     def set_stylesheets(self, *args, **kwargs):
#         if self.app.transceiverConfig._transceiverActive or self.app.generatorConfig.simulationEnabled:
#             if self.app.logConfig._autoUpdate:
#                 self.update_status(*configuration.autoUpdateOn)
#                 self.logReaderTxt.set_connected()
#             else:
#                 self.update_status(*configuration.autoUpdateOff)
#                 self.logReaderTxt.set_valid()
#         else:
#             self.logReaderTxt.set_disabled()
#
#     def update_log_list(self):
#         self._logDir = self.logDirs.currentItem().text()
#         logDir = os.path.join(configuration.LOG_DIRS, self._logDir)
#         self._logs = next(os.walk(logDir))[2]
#         self.logs = lozoya.gui.make_list(
#             items=self._logs, command=lambda: self.update_log(), layout=self.logsGrid, multiSelect=False, )
#         self.logs.setCurrentRow(0)
#
#     def update_log(self):
#         self._log = self.logs.currentItem().text()
#         self.make_log_reader()
#
#     @lozoya.decorators.catch_error('logMenu', *configuration.refreshLogError)
#     def refresh(self, *args, **kwargs):
#         self.app.logger.activeLog = self.logFileSelector.currentItem().text().replace(':', '_')
#         condition1 = self.app.generatorConfig.simulationEnabled
#         condition2 = self.app.transceiverConfig._transceiverActive and self.app.logConfig._autoUpdate and not self.app.logger.activeIsLatest
#         if condition1 or condition2:
#             self.update_status(*configuration.exitLinkedLog)
#             self.app.logConfig._autoUpdate = False
#         self.set_stylesheets()
#         if self.logFileSelector.currentItem():
#             logPath = self.app.logger.activeLogPath
#             try:
#                 self.logReaderTxt.update_contents()
#                 self.logReaderTxt.update_row_labels()
#                 self.logReaderTxt.update_col_labels()
#             except Exception as e:
#                 print(e)
#                 with open(logPath, 'a+') as f:
#                     self.update_status('{}. Created log at {}'.format(str(e), logPath), 'success')
#         self.logReaderGrid.setTitle(self.app.logger.activeLogTitle)
#
#     @lozoya.decorators.catch_error('logMenu', *configuration.toggleAutomaticUpdateError)
#     def toggle_auto_scroll(self, *args, **kwargs):
#         if self.app.transceiverConfig._transceiverActive or self.app.generatorConfig.simulationEnabled:
#             if not self.app.logger.activeIsLatest:
#                 self.logFileSelector.setCurrentRow(0)
#             self.app.logConfig.update('_autoUpdate', not self.app.logConfig._autoUpdate)
#             self.set_stylesheets()
#
#     @lozoya.decorators.catch_error('logMenu', *configuration.updateDirListError)
#     def update_dir_list(self, *args, **kwargs):
#         self.logDirSelector.replace_items(items=[d.replace('_', '/') for d in self.app.logger.dirs], default=0)
#         self.update_log_list()
#
#     @lozoya.decorators.catch_error('logMenu', *configuration.updateLogListError)
#     def update_log_list(self, row=0):
#         self.app.logger.activeDir = self.logDirSelector.currentItem().text().replace('/', '_')
#         self.logFileSelector.replace_items(items=[d.replace('_', ':') for d in self.app.logger.files], default=row)
#         self.refresh()
#
#
# class MenuEPS(lozoya.gui.TSMenu):
#     def __init__(self, parent, title, icon):
#         submenu.SubMenu.__init__(
#             self, parent, title, icon, )
#         self.leftFrame, self.leftLayout = menus.make_subdivision(self)
#         self.rightFrame, self.rightLayout = menus.make_subdivision(self)
#         self.splitter = menus.make_splitter(
#             style='h', widgets=(self.leftFrame, self.rightFrame)
#         )
#         self.epsCurrentSettingsFrame, self.epsCurrentSettingsLayout = menus.make_subdivision(self)
#         self.epsCurrentPlotFrame, self.epsCurrentPlotLayout = menus.make_subdivision(self)
#         self.epsCurrentSplitter = menus.make_splitter(
#             style='v', widgets=(self.epsCurrentSettingsFrame, self.epsCurrentPlotFrame)
#         )
#
#         self.epsVoltageSettingsFrame, self.epsVoltageSettingsLayout = menus.make_subdivision(self)
#         self.epsVoltagePlotFrame, self.epsVoltagePlotLayout = menus.make_subdivision(self)
#         self.epsVoltageSplitter = menus.make_splitter(
#             style='v', widgets=(self.epsVoltageSettingsFrame, self.epsVoltagePlotFrame)
#         )
#         self.epsCurrentContainer = menus.make_group_box(
#             parent=self.leftLayout, layout=menus.make_grid(), text='Current', row=0, column=0, )
#         self.epsVoltageContainer = menus.make_group_box(
#             parent=self.rightLayout, layout=menus.make_grid(), text='Voltage', row=0, column=0, )
#         self.epsCurrentSettings = menus.make_group_box(
#             parent=self.epsCurrentSettingsLayout, layout=menus.make_grid(), text='Settings', height=100, row=0,
#             column=0, )
#         self.epsVoltageSettings = menus.make_group_box(
#             parent=self.epsVoltageSettingsLayout, layout=menus.make_grid(), text='Settings', height=100, row=0,
#             column=0, )
#         self.epsCurrentLimYLabel = menus.make_label(
#             self, text='Range', layout=self.epsCurrentSettings, width=60, row=0, column=0, )
#         self.epsCurrentMinYEntry = menus.make_slider(
#             self, inputSettings=(-1000, 1000, -100, 10), updateFunction=self.update_epsCurrent_ylim,
#             layout=self.epsCurrentSettings, width=50, row=0, column=1, )
#         self.epsCurrentMaxYEntry = menus.make_slider(
#             self, inputSettings=(-1000, 1000, 100, 10), updateFunction=self.update_epsCurrent_ylim,
#             layout=self.epsCurrentSettings, width=50, row=0, column=2, )
#         self.epsCurrentBuffLabel = menus.make_label(
#             self, text='Buffer', layout=self.epsCurrentSettings, width=60, row=1, column=0, )
#         self.epsCurrentBuffEntry = menus.make_slider(
#             self, inputSettings=(10, 1000, 100, 10), layout=self.epsCurrentSettings, width=50, row=1, column=1, )
#         self.epsVoltageLimYLabel = menus.make_label(
#             self, text='Range', layout=self.epsVoltageSettings, width=60, row=0, column=0, )
#         self.epsVoltageMinYEntry = menus.make_slider(
#             self, inputSettings=(-1000, 1000, -100, 10), updateFunction=self.update_epsVoltage_ylim,
#             layout=self.epsVoltageSettings, width=50, row=0, column=1, )
#         self.epsVoltageMaxYEntry = menus.make_slider(
#             self, inputSettings=(-1000, 1000, 100, 10), updateFunction=self.update_epsVoltage_ylim,
#             layout=self.epsVoltageSettings, width=50, row=0, column=2, )
#         self.epsVoltageBuffLabel = menus.make_label(
#             self, text='Buffer', layout=self.epsVoltageSettings, width=60, row=1, column=0, )
#         self.epsVoltageBuffEntry = menus.make_slider(
#             self, inputSettings=(10, 100, 10, 10), layout=self.epsVoltageSettings, width=50, row=1, column=1, )
#         self.epsCurrentContainer.addWidget(self.epsCurrentSplitter)
#         self.epsVoltageContainer.addWidget(self.epsVoltageSplitter)
#         self.epsCurrentSettingsFrame.setMaximumHeight(120)
#         self.epsVoltageSettingsFrame.setMaximumHeight(120)
#         self.epsCurrentPlotLayout.addWidget(self.parent.plotDude.epsCurrent['canvas'])
#         self.epsVoltagePlotLayout.addWidget(self.parent.plotDude.epsVoltage['canvas'])
#         self._layout.addWidget(self.splitter)
#         self.setLayout(self.hbox)
#         self.update_epsCurrent_ylim()
#         self.update_epsVoltage_ylim()
#
#     def update_epsCurrent_ylim(self):
#         self.parent.plotDude.epsCurrent['ax'].set_ylim(
#             ymin=self.epsCurrentMinYEntry.value(), ymax=self.epsCurrentMaxYEntry.value(), )
#
#     def update_epsVoltage_ylim(self):
#         self.parent.plotDude.epsVoltage['ax'].set_ylim(
#             ymin=self.epsVoltageMinYEntry.value(), ymax=self.epsVoltageMaxYEntry.value(), )
#
#
# class MenuIMU(lozoya.gui.TSMenu):
#     def __init__(self, parent, title, icon):
#         submenu.SubMenu.__init__(
#             self, parent, title, icon, )
#         self.leftFrame, self.leftLayout = menus.make_subdivision(self)
#         self.rightFrame, self.rightLayout = menus.make_subdivision(self)
#         self.splitter = menus.make_splitter(
#             style='h', widgets=(self.leftFrame, self.rightFrame)
#         )
#         self.velocityXSettingsFrame, self.velocityXSettingsLayout = menus.make_subdivision(self)
#         self.velocityXPlotFrame, self.velocityXPlotLayout = menus.make_subdivision(self)
#         self.velocityXSplitter = menus.make_splitter(
#             style='v', widgets=(self.velocityXSettingsFrame, self.velocityXPlotFrame)
#         )
#
#         self.accelerationXSettingsFrame, self.accelerationXSettingsLayout = menus.make_subdivision(self)
#         self.accelerationXPlotFrame, self.accelerationXPlotLayout = menus.make_subdivision(self)
#         self.accelerationXSplitter = menus.make_splitter(
#             style='v', widgets=(self.accelerationXSettingsFrame, self.accelerationXPlotFrame)
#         )
#         self.velocityXContainer = menus.make_group_box(
#             parent=self.leftLayout, layout=menus.make_grid(), text='Velocity', row=0, column=0, )
#         self.accelerationXContainer = menus.make_group_box(
#             parent=self.rightLayout, layout=menus.make_grid(), text='Acceleration', row=0, column=0, )
#         self.velocityXSettings = menus.make_group_box(
#             parent=self.velocityXSettingsLayout, layout=menus.make_grid(), text='Settings', height=100, row=0,
#             column=0, )
#         self.accelerationXSettings = menus.make_group_box(
#             parent=self.accelerationXSettingsLayout, layout=menus.make_grid(), text='Settings', height=100, row=0,
#             column=0, )
#         self.velocityXLimYLabel = menus.make_label(
#             self, text='Range', layout=self.velocityXSettings, width=60, row=0, column=0, )
#         self.velocityXMinYEntry = menus.make_slider(
#             self, inputSettings=(-1000, 1000, -100, 10), updateFunction=self.update_velocityX_ylim,
#             layout=self.velocityXSettings, width=50, row=0, column=1, )
#         self.velocityXMaxYEntry = menus.make_slider(
#             self, inputSettings=(-1000, 1000, 100, 10), updateFunction=self.update_velocityX_ylim,
#             layout=self.velocityXSettings, width=50, row=0, column=2, )
#         self.velocityXBuffLabel = menus.make_label(
#             self, text='Buffer', layout=self.velocityXSettings, width=60, row=1, column=0, )
#         self.velocityXBuffEntry = menus.make_slider(
#             self, inputSettings=(10, 1000, 100, 10), layout=self.velocityXSettings, width=50, row=1, column=1, )
#         self.accelerationXLimYLabel = menus.make_label(
#             self, text='Range', layout=self.accelerationXSettings, width=60, row=0, column=0, )
#         self.accelerationXMinYEntry = menus.make_slider(
#             self, inputSettings=(-1000, 1000, -100, 10), updateFunction=self.update_accelerationX_ylim,
#             layout=self.accelerationXSettings, width=50, row=0, column=1, )
#         self.accelerationXMaxYEntry = menus.make_slider(
#             self, inputSettings=(-1000, 1000, 100, 10), updateFunction=self.update_accelerationX_ylim,
#             layout=self.accelerationXSettings, width=50, row=0, column=2, )
#         self.accelerationXBuffLabel = menus.make_label(
#             self, text='Buffer', layout=self.accelerationXSettings, width=60, row=1, column=0, )
#         self.accelerationXBuffEntry = menus.make_slider(
#             self, inputSettings=(10, 100, 10, 10), layout=self.accelerationXSettings, width=50, row=1, column=1, )
#         self.velocityXContainer.addWidget(self.velocityXSplitter)
#         self.accelerationXContainer.addWidget(self.accelerationXSplitter)
#         self.velocityXSettingsFrame.setMaximumHeight(120)
#         self.accelerationXSettingsFrame.setMaximumHeight(120)
#         self.velocityXPlotLayout.addWidget(self.parent.plotDude.velocityX['canvas'])
#         self.accelerationXPlotLayout.addWidget(self.parent.plotDude.accelerationX['canvas'])
#         self._layout.addWidget(self.splitter)
#         self.setLayout(self.hbox)
#         self.update_velocityX_ylim()
#         self.update_accelerationX_ylim()
#
#     def update_velocityX_ylim(self):
#         self.parent.plotDude.velocityX['ax'].set_ylim(
#             ymin=self.velocityXMinYEntry.value(), ymax=self.velocityXMaxYEntry.value(), )
#
#     def update_accelerationX_ylim(self):
#         self.parent.plotDude.accelerationX['ax'].set_ylim(
#             ymin=self.accelerationXMinYEntry.value(), ymax=self.accelerationXMaxYEntry.value(), )
#
#
# class MenuSensor(lozoya.gui.TSMenu):
#     def __init__(self, parent, title, icon):
#         submenu.SubMenu.__init__(
#             self, parent, title, icon, )
#         self.leftFrame, self.leftLayout = menus.make_subdivision(self)
#         self.rightFrame, self.rightLayout = menus.make_subdivision(self)
#         self.splitter = menus.make_splitter(
#             style='h', widgets=(self.leftFrame, self.rightFrame)
#         )
#         self.pressureSettingsFrame, self.pressureSettingsLayout = menus.make_subdivision(self)
#         self.pressurePlotFrame, self.pressurePlotLayout = menus.make_subdivision(self)
#         self.pressureSplitter = menus.make_splitter(
#             style='v', widgets=(self.pressureSettingsFrame, self.pressurePlotFrame)
#         )
#
#         self.temperatureSettingsFrame, self.temperatureSettingsLayout = menus.make_subdivision(self)
#         self.temperaturePlotFrame, self.temperaturePlotLayout = menus.make_subdivision(self)
#         self.temperatureSplitter = menus.make_splitter(
#             style='v', widgets=(self.temperatureSettingsFrame, self.temperaturePlotFrame)
#         )
#         self.pressureContainer = menus.make_group_box(
#             parent=self.leftLayout, layout=menus.make_grid(), text='Pressure', row=0, column=0, )
#         self.temperatureContainer = menus.make_group_box(
#             parent=self.rightLayout, layout=menus.make_grid(), text='Temperature', row=0, column=0, )
#         self.pressureSettings = menus.make_group_box(
#             parent=self.pressureSettingsLayout, layout=menus.make_grid(), text='Settings', height=100, row=0,
#             column=0, )
#         self.temperatureSettings = menus.make_group_box(
#             parent=self.temperatureSettingsLayout, layout=menus.make_grid(), text='Settings', height=100, row=0,
#             column=0, )
#         self.pressureLimYLabel = menus.make_label(
#             self, text='Range', layout=self.pressureSettings, width=60, row=0, column=0, )
#         self.pressureMinYEntry = menus.make_slider(
#             self, inputSettings=(-1000, 1000, -100, 10), updateFunction=self.update_pressure_ylim,
#             layout=self.pressureSettings, width=50, row=0, column=1, )
#         self.pressureMaxYEntry = menus.make_slider(
#             self, inputSettings=(-1000, 1000, 100, 10), updateFunction=self.update_pressure_ylim,
#             layout=self.pressureSettings, width=50, row=0, column=2, )
#         self.pressureBuffLabel = menus.make_label(
#             self, text='Buffer', layout=self.pressureSettings, width=60, row=1, column=0, )
#         self.pressureBuffEntry = menus.make_slider(
#             self, inputSettings=(10, 1000, 100, 10), layout=self.pressureSettings, width=50, row=1, column=1, )
#         self.temperatureLimYLabel = menus.make_label(
#             self, text='Range', layout=self.temperatureSettings, width=60, row=0, column=0, )
#         self.temperatureMinYEntry = menus.make_slider(
#             self, inputSettings=(-1000, 1000, -100, 10), updateFunction=self.update_temperature_ylim,
#             layout=self.temperatureSettings, width=50, row=0, column=1, )
#         self.temperatureMaxYEntry = menus.make_slider(
#             self, inputSettings=(-1000, 1000, 100, 10), updateFunction=self.update_temperature_ylim,
#             layout=self.temperatureSettings, width=50, row=0, column=2, )
#         self.temperatureBuffLabel = menus.make_label(
#             self, text='Buffer', layout=self.temperatureSettings, width=60, row=1, column=0, )
#         self.temperatureBuffEntry = menus.make_slider(
#             self, inputSettings=(10, 100, 10, 10), layout=self.temperatureSettings, width=50, row=1, column=1, )
#         self.pressureContainer.addWidget(self.pressureSplitter)
#         self.temperatureContainer.addWidget(self.temperatureSplitter)
#         self.pressureSettingsFrame.setMaximumHeight(120)
#         self.temperatureSettingsFrame.setMaximumHeight(120)
#         self.pressurePlotLayout.addWidget(self.parent.plotDude.pressure['canvas'])
#         self.temperaturePlotLayout.addWidget(self.parent.plotDude.temperature['canvas'])
#         self._layout.addWidget(self.splitter)
#         self.setLayout(self.hbox)
#         self.update_pressure_ylim()
#         self.update_temperature_ylim()
#
#     def update_pressure_ylim(self):
#         self.parent.plotDude.pressure['ax'].set_ylim(
#             ymin=self.pressureMinYEntry.value(), ymax=self.pressureMaxYEntry.value(), )
#
#     def update_temperature_ylim(self):
#         self.parent.plotDude.temperature['ax'].set_ylim(
#             ymin=self.temperatureMinYEntry.value(), ymax=self.temperatureMaxYEntry.value(), )
#
#
# class MenuSettings(lozoya.gui.TSMenu):
#     def __init__(self, app, title, icon):
#         lozoya.gui.TSMenu.__init__(
#             self, app, title, icon, )
#         self.setMinimumWidth(425)
#         self.setFixedHeight(250)
#         tabNames = ['']
#         self.tabs = lozoya.gui.set_tabs(self, tabNames, self._layout, )
#         for tab in self.tabs:
#             self.tabs[tab].layout().setContentsMargins(5, 5, 5, 0)
#         submenu.SubMenu.__init__(
#             self, parent, title, icon, )
#         self.transmissionGroupBox = lozoya.gui.make_group_box(
#             self._layout, layout=lozoya.gui.make_grid(), text='Transmission', row=0, column=0, width=350, height=170, )
#         _, self.displayAVCheckBox = lozoya.gui.make_pair(
#             self, pair='check', text='Display Faces', layout=self.transmissionGroupBox, row=4, column=0,
#             description='Toggle whether the cube faces are rendered.', label=True,
#             pairCommand=self.change_frequency_dial, comboItems=['True', 'False']
#         )
#         try:
#             self.bottom.deleteLater()
#             self.bottomLayout.deleteLater()
#             self.scrollArea.deleteLater()
#             self.splitter2.deleteLater()
#         except:
#             pass
#         self.bottom = lozoya.gui.make_grid(self)
#         self.cmdEntryLabel = lozoya.gui.make_label(
#             self, text='Command:', layout=self.transmissionGroupBox, width=60, row=2, column=0, )
#         self.cmdEntry = lozoya.gui.make_label(
#             self, layout=self.transmissionGroupBox, row=2, column=1, )
#         self.sendCmdBtn = lozoya.gui.make_button(
#             self, command=self.transmit, text='Send', layout=self.transmissionGroupBox, width=50, row=2, column=2, )
#         self.receivedMsgLabel = lozoya.gui.make_label(
#             self, text='Received:', layout=self.transmissionGroupBox, width=60, row=3, column=0, )
#         self.receivedMsg = lozoya.gui.make_label(
#             self, layout=self.transmissionGroupBox, row=3, column=1, )
#         self.setLayout(self.hbox)
#         submenu.SubMenu.__init__(
#             self, parent, title, icon, )
#         self.dataSettingsGroupBox = lozoya.gui.make_group_box(
#             self._layout, text='Data', row=0, column=0, )
#         self.dataTypeEntryGroupBox, self.dataTypeEntry = lozoya.gui.pair(
#             self, pair='entry', pairText=self.parent.preferences.dataType, command=self.update_data_type,
#             layout=self.dataSettingsGroupBox.layout(), readOnly=self.parent.preferences.dataTypeLock,
#             **app.cubesat.widgetkwargs.settings.dataTypeEntry, )
#         self.dataTypeLockBtn = lozoya.gui.make_button(
#             self, command=self.lock_data_type,
#             icon=icons.lockClosed if self.parent.preferences.dataTypeLock else icons.lockOpen,
#             layout=self.dataTypeEntryGroupBox.layout(), **app.cubesat.widgetkwargs.settings.dataTypeLockBtn, )
#         self.headEntryGroupBox, self.headEntry = lozoya.gui.pair(
#             self, pair='entry', pairText=self.parent.preferences.head, command=self.update_head,
#             layout=self.dataSettingsGroupBox.layout(), readOnly=self.parent.preferences.headLock,
#             **app.cubesat.widgetkwargs.settings.headEntry, )
#         self.headLockBtn = lozoya.gui.make_button(
#             self, command=self.lock_head, icon=icons.lockClosed if self.parent.preferences.headLock else icons.lockOpen,
#             layout=self.headEntryGroupBox.layout(), **app.cubesat.widgetkwargs.settings.headLockBtn, )
#         self.bufferGroupBox, self.bufferLabel = lozoya.gui.pair(
#             self, pair='slider', text='Buffer', layout=self.dataSettingsGroupBox.layout(), row=3, column=0,
#             dialSettings=(10, 1000, self.parent.preferences.buffer, 10), command=self.update_buffer,
#             pairStyleSheet=qss.slider, tooltip=tooltips.plotBuffer, )
#         # self.bufferSlider = lozoya.gui.make_slider(
#         #     self,
#         #     inputSettings=(10, 1000, self.parent.preferences.buffer, 10),
#         #     layout=self.dataSettingsGroupBox.layout(),
#         #     row=2,
#         #     column=1,
#         #     command=self.update_buffer,
#         #     tooltip=tooltips.plotBuffer,
#         # )
#         self.update_data_type()
#         self.update_head()
#         self.deviceSettingsGroupBox = lozoya.gui.make_group_box(
#             self._layout, **app.cubesat.widgetkwargs.settings.deviceSettingsGroupBox, )
#         _, self.renderFacesCheckBox = lozoya.gui.pair(
#             self, pair='check', text='Faces', label=True, row=0, column=0, layout=self.deviceSettingsGroupBox.layout(),
#             tooltip=tooltips.renderFaces, pairKwargs={
#                 'command': self.update_render_faces, 'isChecked': self.parent.preferences.renderFaces,
#                 'layout' : self.deviceSettingsGroupBox.layout(), 'row': 0, 'column': 1, 'tooltip': tooltips.renderFaces,
#             }, )
#         _, self.renderWireframeCheckBox = lozoya.gui.pair(
#             self, pair='check', layout=self.deviceSettingsGroupBox.layout(), pairKwargs={
#                 'command': self.update_render_wireframe, 'isChecked': self.parent.preferences.renderWireframe,
#                 'layout' : self.deviceSettingsGroupBox.layout(), 'row': 0, 'column': 3,
#                 'tooltip': tooltips.renderWireframe,
#             }, **app.cubesat.widgetkwargs.device.renderWireframeCheckBox
#         )
#         _, self.renderThrustersCheckBox = lozoya.gui.pair(
#             self, pair='check', layout=self.deviceSettingsGroupBox.layout(), pairKwargs={
#                 'command': self.update_render_thrusters, 'isChecked': self.parent.preferences.renderThrusters,
#                 'layout' : self.deviceSettingsGroupBox.layout(), 'row': 0, 'column': 5,
#                 'tooltip': tooltips.renderThrusters,
#             }, **app.cubesat.widgetkwargs.device.renderThrustersCheckBox, )
#         _, self.renderAxesQuiverCheckBox = lozoya.gui.pair(
#             self, pair='check', layout=self.deviceSettingsGroupBox.layout(), pairKwargs={
#                 'command': self.update_render_axes_quiver, 'isChecked': self.parent.preferences.renderAxesQuiver,
#                 'layout' : self.deviceSettingsGroupBox.layout(), 'row': 1, 'column': 1,
#                 'tooltip': tooltips.renderAxesQuiver,
#             }, **app.cubesat.widgetkwargs.device.renderAxesQuiverCheckBox, )
#         _, self.renderCubeQuiverCheckBox = lozoya.gui.pair(
#             self, pair='check', text='Cube Quiver', layout=self.deviceSettingsGroupBox.layout(), pairKwargs={
#                 'command': self.update_render_cube_quiver, 'isChecked': self.parent.preferences.renderCubeQuiver,
#                 'layout' : self.deviceSettingsGroupBox.layout(), 'row': 1, 'column': 3,
#                 'tooltip': tooltips.renderCubeQuiver,
#             }, **app.cubesat.widgetkwargs.device.renderCubeQuiverCheckBox, )
#         self.logSettingsGroupBox = lozoya.gui.make_group_box(
#             self._layout, **app.cubesat.widgetkwargs.settings.logSettingsGroupBox, )
#         _, self.loggingRateSpinBox = lozoya.gui.pair(
#             self, pair='numerical', dialSettings=(10, 1000, self.parent.preferences.loggingRate, 1,),
#             command=self.update_logging_rate, layout=self.logSettingsGroupBox.layout(),
#             **app.cubesat.widgetkwargs.settings.loggingRateSpinBox, )
#         _, self.logCreationIntervalSpinBox = lozoya.gui.pair(
#             self, pair='numerical', dialSettings=(10, 600, self.parent.preferences.logCreationInterval, 1,),
#             command=self.update_log_creation_interval, layout=self.logSettingsGroupBox.layout(),
#             **app.cubesat.widgetkwargs.settings.logCreationIntervalSpinBox, )
#
#         b = self.parent.transceiver.bands[self.parent.preferences.bandIndex]
#
#         self.transceiverSettingsGroupBox = lozoya.gui.make_group_box(
#             self._layout, **app.cubesat.widgetkwargs.settings.transceiverSettingsGroupBox, )
#         _, self.baudRateCombo = lozoya.gui.pair(
#             self, pair='combo', comboItems=self.parent.transceiver.baudRates,
#             default=self.parent.preferences.baudRateIndex, command=self.update_baud_rate,
#             layout=self.transceiverSettingsGroupBox.layout(), **app.cubesat.widgetkwargs.settings.baudRateCombo, )
#         _, self.bandCombo = lozoya.gui.pair(
#             self, pair='combo', comboItems=self.parent.transceiver.bandNames, default=self.parent.preferences.bandIndex,
#             layout=self.transceiverSettingsGroupBox.layout(), command=self.update_band,
#             **app.cubesat.widgetkwargs.settings.bandCombo, )
#         self.rfLabel = lozoya.gui.make_label(
#             self, text='Radio Frequency ({})'.format(b.units), layout=self.transceiverSettingsGroupBox.layout(),
#             **app.cubesat.widgetkwargs.settings.rfLabel, )
#         self.rfSlider = lozoya.gui.make_slider(
#             self, inputSettings=(b.minimum * 100, b.maximum * 100,  # TODO use band conversion
#                                  self.parent.preferences.transmissionFrequency * 100, 0.01,),
#             command=self.update_frequency_slider, layout=self.transceiverSettingsGroupBox.layout(),
#             **app.cubesat.widgetkwargs.settings.rfSlider, )
#         self.rfSpinBox = lozoya.gui.make_numerical_input(
#             self, inputSettings=(b.minimum, b.maximum, self.parent.preferences.transmissionFrequency, 0.01,),
#             command=self.update_frequency_entry, layout=self.transceiverSettingsGroupBox.layout(), stylesheet=qss.spin,
#             **app.cubesat.widgetkwargs.settings.rfSpinBox, )
#         _, self.portEntry = lozoya.gui.pair(
#             self, pair='numerical', dialSettings=(0, 9999, self.parent.preferences.port, 1),
#             pairText=self.parent.preferences.head, layout=self.transceiverSettingsGroupBox.layout(),
#             readOnly=self.parent.preferences.portConnected, command=self.set_port,
#             pairStyleSheet=qss.spinConnected if self.parent.preferences.portConnected else qss.spin,
#             **app.cubesat.widgetkwargs.settings.portEntry, )
#         self.portLockBtn = lozoya.gui.make_button(
#             self, command=self.connect_port,
#             icon=icons.connected if self.parent.preferences.portConnected else icons.disconnected,
#             layout=self.transceiverSettingsGroupBox.layout(),
#             stylesheet=qss.buttonConnected if self.parent.preferences.portConnected else qss.buttonDisconnected,
#             **app.cubesat.widgetkwargs.settings.portLockBtn, )
#         self.set_stylesheets()
#
#     def transmit(self):
#         cmd = self.cmdEntry.toPlainText()
#         if not cmd:
#             QtWidgets.QMessageBox.critical(
#                 self, "Error", "Did not transmit.", QtWidgets.QMessageBox.Ok, )
#         else:
#             print('sent: {}'.format(cmd))
#
#     def change_frequency_dial(self):
#         pass
#
#     def update_data_type(self):
#         txt = self.dataTypeEntry.text().strip('\n')
#         for _ in txt.split(','):
#             if _ not in ['d', 'f', 'h', 's', 't']:  # int, float, hex, str, time
#                 self.dataTypeEntry.setStyleSheet(qss.format_qss(qss.entryError, self.parent.palette))
#                 self.dataTypeLockBtn.setStyleSheet(qss.format_qss(qss.buttonDisabled, self.parent.palette))
#                 self.parent.preferences.dataTypeValid = False
#                 return
#         if txt == '':
#             self.dataTypeEntry.setStyleSheet(qss.format_qss(qss.entry, self.parent.palette))
#         else:
#             if self.parent.preferences.dataTypeLock:
#                 self.dataTypeEntry.setStyleSheet(qss.format_qss(qss.entryConnected, self.parent.palette))
#             else:
#                 self.dataTypeEntry.setStyleSheet(qss.format_qss(qss.entryValid, self.parent.palette))
#                 self.dataTypeLockBtn.setStyleSheet(qss.format_qss(qss.buttonValid, self.parent.palette))
#                 self.parent.preferences.dataTypeValid = True
#         self.parent.preferences.dataType = txt
#         self.parent.preferences.save_to_file()
#
#     # DATA
#     def set_stylesheets(self):
#         if self.parent.preferences.headValid and self.parent.preferences.headLock:
#             headLockBtnStyleSheet = qss.buttonConnected
#         elif self.parent.preferences.headValid:
#             headLockBtnStyleSheet = qss.buttonValid
#         else:
#             headLockBtnStyleSheet = qss.buttonDisabled
#         if self.parent.preferences.dataTypeValid and self.parent.preferences.dataTypeLock:
#             dataTypeLockBtnStyleSheet = qss.buttonConnected
#         elif self.parent.preferences.headValid:
#             dataTypeLockBtnStyleSheet = qss.buttonValid
#         else:
#             dataTypeLockBtnStyleSheet = qss.buttonDisabled
#         self.headLockBtn.setStyleSheet(qss.format_qss(headLockBtnStyleSheet, self.parent.palette))
#         self.dataTypeLockBtn.setStyleSheet(qss.format_qss(dataTypeLockBtnStyleSheet, self.parent.palette))
#
#     def update_head(self):
#         txt = self.headEntry.text().strip('\n')
#         dtypes = self.dataTypeEntry.text().strip('\n')
#         if len(txt.split(',')) != len(dtypes.split(',')):
#             self.headEntry.setStyleSheet(qss.format_qss(qss.entryError, self.parent.palette))
#             self.headLockBtn.setStyleSheet(qss.format_qss(qss.buttonDisabled, self.parent.palette))
#             self.parent.preferences.headValid = False
#             return
#         if txt == '':
#             self.headEntry.setStyleSheet(qss.format_qss(qss.entry, self.parent.palette))
#         else:
#             if self.parent.preferences.headLock:
#                 self.headEntry.setStyleSheet(qss.format_qss(qss.entryConnected, self.parent.palette))
#             else:
#                 self.headEntry.setStyleSheet(qss.format_qss(qss.entryValid, self.parent.palette))
#                 self.headLockBtn.setStyleSheet(qss.format_qss(qss.buttonValid, self.parent.palette))
#                 self.parent.preferences.headValid = True
#         self.parent.preferences.head = txt
#         self.parent.preferences.save_to_file()
#
#     def lock_data_type(self):
#         if self.parent.preferences.dataTypeValid:
#             lock = not self.parent.preferences.dataTypeLock
#             self.parent.preferences.dataTypeLock = lock
#             self.dataTypeEntry.setReadOnly(lock)
#             if lock:
#                 self.dataTypeLockBtn.setStyleSheet(qss.format_qss(qss.buttonConnected, self.parent.palette))
#                 self.dataTypeLockBtn.setIcon(QtGui.QIcon(icons.lockClosed))
#                 self.dataTypeEntry.setStyleSheet(qss.format_qss(qss.entryConnected, self.parent.palette))
#             else:
#                 self.dataTypeLockBtn.setStyleSheet(qss.format_qss(qss.buttonValid, self.parent.palette))
#                 self.dataTypeLockBtn.setIcon(QtGui.QIcon(icons.lockOpen))
#                 self.dataTypeEntry.setStyleSheet(qss.format_qss(qss.entryValid, self.parent.palette))
#             self.parent.preferences.save_to_file()
#
#     def lock_head(self):
#         if self.parent.preferences.headValid:
#             lock = not self.parent.preferences.headLock
#             self.parent.preferences.headLock = lock
#             self.headEntry.setReadOnly(lock)
#             if lock:
#                 self.headLockBtn.setStyleSheet(qss.format_qss(qss.buttonConnected, self.parent.palette))
#                 self.headLockBtn.setIcon(QtGui.QIcon(icons.lockClosed))
#                 self.headEntry.setStyleSheet(qss.format_qss(qss.entryConnected, self.parent.palette))
#             else:
#                 self.headLockBtn.setStyleSheet(qss.format_qss(qss.buttonValid, self.parent.palette))
#                 self.headLockBtn.setIcon(QtGui.QIcon(icons.lockOpen))
#                 self.headEntry.setStyleSheet(qss.format_qss(qss.entryValid, self.parent.palette))
#             self.parent.preferences.save_to_file()
#
#     def update_buffer(self):
#         self.parent.preferences.buffer = self.bufferSlider.value()
#         # TODO will be a value inside a plot0 data structure:
#         self.parent.plotMenu.plotDomainSlider.setRange(0, self.parent.preferences.buffer - 10)
#         for plot in self.parent.plotDude.plots:
#             plot['ax'].set_xlim(
#                 self.parent.plotMenu.plotDomainSlider.value(), self.parent.preferences.buffer, )
#         self.update_status('Buffer: {} samples.'.format(self.bufferSlider.value()), qss.statusbarSuccess)
#         self.parent.preferences.save_to_file()
#
#     def update_render_faces(self):
#         try:
#             self.parent.preferences.renderFaces = not self.parent.preferences.renderFaces
#             if self.parent.preferences.renderFaces:
#                 self.parent.plotDude.draw_cube_faces()
#                 self.update_status('Cube face rendering enabled.', qss.statusbarSuccess)
#                 self.renderFacesCheckBox.setStyleSheet(qss.format_qss(qss.checkboxChecked, self.parent.palette))
#             else:
#                 self.parent.plotDude.remove_cube_faces()
#                 self.update_status('Cube face rendering disabled.', qss.statusbarSuccess)
#                 self.renderFacesCheckBox.setStyleSheet(qss.format_qss(qss.checkboxUnchecked, self.parent.palette))
#             self.parent.preferences.save_to_file()
#         except Exception as e:
#             alertMsg = 'Update render cube faces error: {}'.format(str(e))
#             print(alertMsg)
#             self.update_status(alertMsg, qss.statusbarError)
#
#     # DEVICE
#     def update_render_axes_quiver(self):
#         try:
#             self.parent.preferences.renderAxesQuiver = not self.parent.preferences.renderAxesQuiver
#             if self.parent.preferences.renderAxesQuiver:
#                 self.parent.plotDude.draw_axes_quiver()
#                 self.update_status('Axes quiver rendering enabled.', qss.statusbarSuccess)
#                 self.renderAxesQuiverCheckBox.setStyleSheet(qss.format_qss(qss.checkboxChecked, self.parent.palette))
#             else:
#                 self.parent.plotDude.remove_axes_quiver()
#                 self.update_status('Axes quiver rendering disabled.', qss.statusbarSuccess)
#                 self.renderAxesQuiverCheckBox.setStyleSheet(qss.format_qss(qss.checkboxUnchecked, self.parent.palette))
#             self.parent.preferences.save_to_file()
#         except Exception as e:
#             alertMsg = 'Update render axes quiver error: {}'.format(str(e))
#             print(alertMsg)
#             self.update_status(alertMsg, qss.statusbarError)
#
#     def update_render_cube_quiver(self):
#         try:
#             self.parent.preferences.renderCubeQuiver = not self.parent.preferences.renderCubeQuiver
#             if self.parent.preferences.renderCubeQuiver:
#                 self.parent.plotDude.draw_cube_quiver()
#                 self.update_status('Cube quiver rendering enabled.', qss.statusbarSuccess)
#                 self.renderCubeQuiverCheckBox.setStyleSheet(qss.format_qss(qss.checkboxChecked, self.parent.palette))
#             else:
#                 self.parent.plotDude.remove_cube_quiver()
#                 self.update_status('Cube quiver rendering disabled.', qss.statusbarSuccess)
#                 self.renderCubeQuiverCheckBox.setStyleSheet(qss.format_qss(qss.checkboxUnchecked, self.parent.palette))
#             self.parent.preferences.save_to_file()
#         except Exception as e:
#             alertMsg = 'Update render cube quiver error: {}'.format(str(e))
#             print(alertMsg)
#             self.update_status(alertMsg, qss.statusbarError)
#
#     def update_render_wireframe(self):
#         try:
#             self.parent.preferences.renderWireframe = not self.parent.preferences.renderWireframe
#             if self.parent.preferences.renderWireframe:
#                 self.parent.plotDude.draw_wireframe()
#                 self.update_status('Cube wireframe rendering enabled.', qss.statusbarSuccess)
#                 self.renderWireframeCheckBox.setStyleSheet(qss.format_qss(qss.checkboxChecked, self.parent.palette))
#             else:
#                 self.parent.plotDude.remove_wireframe()
#                 self.update_status('Cube wireframe rendering disabled.', qss.statusbarSuccess)
#                 self.renderWireframeCheckBox.setStyleSheet(qss.format_qss(qss.checkboxUnchecked, self.parent.palette))
#             self.parent.preferences.save_to_file()
#         except Exception as e:
#             alertMsg = 'Update render wireframe error: {}'.format(str(e))
#             print(alertMsg)
#             self.update_status(alertMsg, qss.statusbarError)
#
#     def update_render_thrusters(self):
#         try:
#             self.parent.preferences.renderThrusters = not self.parent.preferences.renderThrusters
#             if self.parent.preferences.renderThrusters:
#                 self.parent.plotDude.show_thrusters()
#                 self.update_status('Thruster rendering enabled.', qss.statusbarSuccess)
#                 self.renderThrustersCheckBox.setStyleSheet(qss.format_qss(qss.checkboxChecked, self.parent.palette))
#             else:
#                 self.parent.plotDude.hide_thrusters()
#                 self.update_status('Thruster rendering disabled.', qss.statusbarSuccess)
#                 self.renderThrustersCheckBox.setStyleSheet(qss.format_qss(qss.checkboxUnchecked, self.parent.palette))
#             self.parent.preferences.save_to_file()
#         except Exception as e:
#             alertMsg = 'Update render thrusters error: {}'.format(str(e))
#             print(alertMsg)
#             self.update_status(alertMsg, qss.statusbarError)
#
#     def update_baud_rate(self):
#         index = self.baudRateCombo.currentIndex()
#         self.parent.preferences.baudRate = int(self.parent.transceiver.baudRates[index])
#         self.parent.preferences.baudRateIndex = index
#         self.update_status('Baud rate: {} Hz.'.format(self.parent.preferences.baudRate), qss.statusbarSuccess)
#         self.parent.preferences.save_to_file()
#
#     # TRANSCEIVER
#     def update_band(self):
#         index = self.bandCombo.currentIndex()
#         self.parent.preferences.band = self.parent.transceiver.bandNames[index]
#         self.parent.preferences.bandIndex = index
#         b = self.parent.transceiver.bands[index]
#         _f = self.parent.preferences.transmissionFrequency
#         if _f >= b.minimum and _f <= b.maximum:
#             f = _f
#         else:
#             f = b.minimum
#         self.parent.preferences.transmissionFrequency = f
#         try:
#             self.rfLabel.setText('Radio Frequency ({})'.format(b.units))
#             self.rfSlider.setRange(b.minimum * 100, b.maximum * 100)  # TODO use band conversion
#             self.rfSlider.setValue(f)
#             self.rfSpinBox.setRange(b.minimum, b.maximum)
#             self.rfSpinBox.setValue(f)
#             self.parent.preferences.save_to_file()
#             frequency = self.rfSpinBox.value()
#             units = self.parent.transceiver.bands[self.parent.preferences.bandIndex].units
#             self.update_status(
#                 'Set {} band. Radio Frequency: {} {}'.format(b.label, frequency, units), qss.statusbarSuccess
#             )
#         except Exception as e:
#             print('Update band error: ' + str(e))
#
#     def update_frequency_slider(self):
#         try:
#             v = self.rfSlider.value() / 100  # TODO use band conversion
#             self.rfSpinBox.setValue(v)
#             self.parent.preferences.transmissionFrequency = v
#             self.parent.preferences.save_to_file()
#             frequency = self.rfSpinBox.value()
#             units = self.parent.transceiver.bands[self.parent.preferences.bandIndex].units
#             self.update_status('Radio Frequency: {} {}.'.format(frequency, units), qss.statusbarSuccess)
#         except Exception as e:
#             print('Update frequency error slider: ' + str(e))
#
#     def update_frequency_entry(self):
#         try:
#             v = self.rfSpinBox.value()
#             self.rfSlider.setValue(v * 100)  # TODO use band conversion
#             self.parent.preferences.transmissionFrequency = v
#             self.parent.preferences.save_to_file()
#             frequency = self.rfSpinBox.value()
#             units = self.parent.transceiver.bands[self.parent.preferences.bandIndex].units
#             self.update_status('Radio Frequency: {} {}.'.format(frequency, units), qss.statusbarSuccess)
#         except Exception as e:
#             print('Update frequency error entry: ' + str(e))
#
#     def connect_port(self):
#         try:
#             connected = not self.parent.preferences.portConnected
#             self.parent.preferences.portConnected = connected
#             self.portEntry.setReadOnly(connected)
#             self.parent.transceiverMenu.set_stylesheets()
#             self.parent.logMenu.set_stylesheets()
#             if connected:
#                 # TODO try to connect
#                 self.portLockBtn.setIcon(QtGui.QIcon(icons.connected))
#                 self.portLockBtn.setStyleSheet(qss.format_qss(qss.buttonConnected, self.parent.palette))
#                 self.portEntry.setStyleSheet(qss.format_qss(qss.spinConnected, self.parent.palette))
#                 alertMsg = 'Connected to port {}.'.format(self.parent.preferences.port)
#                 self.update_status(alertMsg, qss.statusbarSuccess)
#             else:
#                 # TODO confirm disconnected
#                 self.parent.preferences.transceiverActive = False
#                 self.portLockBtn.setIcon(QtGui.QIcon(icons.disconnected))
#                 self.portLockBtn.setStyleSheet(qss.format_qss(qss.buttonDisconnected, self.parent.palette))
#                 self.portEntry.setStyleSheet(qss.format_qss(qss.spin, self.parent.palette))
#                 alertMsg = 'Disconnected from port {}.'.format(self.parent.preferences.port)
#                 self.update_status(alertMsg, qss.statusbarSuccess)
#             self.parent.preferences.save_to_file()
#         except Exception as e:
#             alertMsg = 'Connect port error: {}'.format(str(e))
#             print(alertMsg)
#             self.update_status(alertMsg, qss.statusbarError)
#
#     def set_port(self):
#         try:
#             self.parent.preferences.port = self.portEntry.value()
#             self.parent.preferences.save_to_file()
#         except Exception as e:
#             alertMsg = 'Set port error: {}.'.format(e)
#             print(alertMsg)
#             self.update_status(alertMsg, qss.statusbarError)
#
#     def update_logging_rate(self):
#         try:
#             self.parent.preferences.loggingRate = self.loggingRateSpinBox.value()
#             self.parent.preferences.save_to_file()
#             self.update_status('Logging rate: {}.'.format(self.parent.preferences.loggingRate), qss.statusbarSuccess)
#         except Exception as e:
#             alertMsg = 'Update logging rate error: {}.'.format(e)
#             print(alertMsg)
#             self.update_status(alertMsg, qss.statusbarError)
#
#     def update_log_creation_interval(self):
#         try:
#             self.parent.preferences.logCreationInterval = self.logCreationIntervalSpinBox.value()
#             self.update_status(
#                 'Log creation interval: {} seconds.'.format(self.parent.preferences.logCreationInterval),
#                 qss.statusbarSuccess
#             )
#             self.parent.preferences.save_to_file()
#         except Exception as e:
#             alertMsg = 'Update log creation interval.'.format(e)
#             print(alertMsg)
#             self.update_status(alertMsg, qss.statusbarError)
#
#
# class MenuTransceiver(lozoya.gui.TSMenu):
#     def __init__(self, app, title, icon):
#         lozoya.gui.TSMenu.__init__(self, app, title, icon, )
#         self.settingsForm = settingsform.SettingsForm(self.app, self, height=200)
#         self._layout.addWidget(self.settingsForm.frame)
#         tabNames = ['Data', 'Generator', 'Radio']
#         self.tabs = lozoya.gui.set_tabs(self, tabNames, self.settingsForm.layout, )
#         for tab in self.tabs:
#             self.tabs[tab].layout().setContentsMargins(5, 5, 5, 0)
#         self.dataSettings = datasettings.SettingsData(self.app, self, self.tabs['Data'].layout())
#         self.generatorSettings = generatorsettings.SettingsGenerator(self.app, self, self.tabs['Generator'].layout(), )
#         self.transceiverSettings = radiosettings.TransceiverSettings(self.app, self, self.tabs['Radio'].layout(), )
#         self.menuFrame = lozoya.gui.make_frame(self)
#         self.menuContainer = lozoya.gui.make_grid(self.menuFrame)
#         self.cmdGroupBox, self.cmdEntry = lozoya.gui.pair(
#             parent=self, pair='entry', layout=self.menuContainer, **app.cubesat.widgetkwargs.transceiver.cmdEntry, )
#         self.sendCmdBtn = lozoya.gui.make_button(
#             parent=self, layout=self.cmdGroupBox.layout(), command=lambda: self.app.transceiver.transmit(
#                 self.cmdEntry.toPlainText()
#             ), **app.cubesat.widgetkwargs.transceiver.sendCmdBtn, )
#         self.receivedGroupBox, self.receivedMsg = lozoya.gui.pair(
#             parent=self, pair='entry', layout=self.menuContainer, **app.cubesat.widgetkwargs.transceiver.receivedMsg, )
#         condition = self.app.transceiverConfig._transceiverActive or self.app.generatorConfig.simulationEnabled
#         self._transceiverActiveBtn = lozoya.gui.make_button(
#             parent=self, layout=self.receivedGroupBox.layout(), command=self.toggle_transceiver,
#             icon=configuration.connected if condition else configuration.disconnected,
#             **app.cubesat.widgetkwargs.transceiver.transceiverActiveBtn, )
#         self._layout.addWidget(self.menuFrame)
#         self.setMinimumSize(400, 200)
#         self.resize(400, 200)
#         self.set_stylesheets()
#         self.set_menubar()
#         submenu.SubMenu.__init__(self, parent, title, icon, )
#         # self.frame.setFixedSize(740, 190)
#         # self.setFixedSize(810, 215)
#         self.frame.setFixedHeight(210)
#         self.setFixedHeight(230)
#         # self.frame.setMaximumWidth(740)
#         self.setMaximumWidth(860)
#         self.leftFrame, self.leftLayout = lozoya.gui.make_subdivision(self)
#         self.rightFrame, self.rightLayout = lozoya.gui.make_subdivision(self)
#
#         self.splitter = lozoya.gui.make_splitter(style='h', widgets=(self.leftFrame, self.rightFrame))
#         self.transmissionGroupBox = lozoya.gui.make_group_box(
#             self.leftLayout, layout=lozoya.gui.make_grid(), text='Transmission', row=0, column=0, width=400,
#             height=170, )
#         self.transmissionSettingsGroupBox = lozoya.gui.make_group_box(
#             self.rightLayout, layout=lozoya.gui.make_grid(), text='Settings', row=0, column=0, width=350, height=170, )
#         _, self.frequencyHzSlider = lozoya.gui.make_pair(
#             self, pair='slider', dialSettings=(0, 1e6, 1, 1), text='Frequency (Hz)',
#             layout=self.transmissionSettingsGroupBox, row=2, column=0, description='', label=True,
#             pairCommand=self.change_frequency_dial, pairWidth=100, )
#         self.frequencyHzEntry = lozoya.gui.make_int_input(
#             self, text='', inputSettings=(0, 1e6, 1, 1), layout=self.transmissionSettingsGroupBox, row=2, column=2,
#             number=True, updateFunction=self.change_frequency_entry, width=65, )
#
#         self.cmdEntryLabel = lozoya.gui.make_label(
#             self, text='Command:', layout=self.transmissionGroupBox, width=95, row=0, column=0, )
#         self.cmdEntry = lozoya.gui.make_entry(
#             self, area=True, width=200, height=55, layout=self.transmissionGroupBox, row=0, column=1,
#             stylesheet=qss.GLOW_ENTRY_STYLE
#         )
#         self.sendCmdBtn = lozoya.gui.make_button(
#             self, command=self.transmit, text='Send', layout=self.transmissionGroupBox, width=40, row=0, column=2, )
#         self.receivedMsgLabel = lozoya.gui.make_label(
#             self, text='Received:', layout=self.transmissionGroupBox, width=95, row=1, column=0, )
#         self.receivedMsg = lozoya.gui.make_entry(
#             self, area=True, width=200, height=55, layout=self.transmissionGroupBox, row=1, column=1, readOnly=True, )
#         self._layout.addWidget(self.splitter)
#         self.setLayout(self.hbox)
#         lozoya.gui.TSMenu.__init__(self, app, title, icon, )
#         self.settingsForm = settingsform.SettingsForm(self.app, self, height=200)
#         self._layout.addWidget(self.settingsForm.frame)
#         tabNames = ['Data', 'Generator', 'Radio']
#         self.tabs = lozoya.gui.set_tabs(self, tabNames, self.settingsForm.layout, )
#         for tab in self.tabs:
#             self.tabs[tab].layout().setContentsMargins(5, 5, 5, 0)
#         self.dataSettings = datasettings.SettingsData(self.app, self, self.tabs['Data'].layout())
#         self.generatorSettings = generatorsettings.SettingsGenerator(self.app, self, self.tabs['Generator'].layout(), )
#         self.transceiverSettings = transceiversettings.TransceiverSettings(
#             self.app, self, self.tabs['Radio'].layout(), )
#         self.cmdGroupBox, self.cmdEntry = lozoya.gui.pair(
#             parent=self, pair='entry', layout=self._layout, **app.cubesat.widgetkwargs.transceiver.cmdEntry, )
#         self.sendCmdBtn = lozoya.gui.make_button(
#             parent=self, layout=self.cmdGroupBox.layout(), command=lambda: self.app.transceiver.transmit(
#                 self.cmdEntry.toPlainText()
#             ), **app.cubesat.widgetkwargs.transceiver.sendCmdBtn, )
#         self.receivedGroupBox, self.receivedMsg = lozoya.gui.pair(
#             parent=self, pair='entry', layout=self._layout, **app.cubesat.widgetkwargs.transceiver.receivedMsg, )
#         self._transceiverActiveBtn = lozoya.gui.make_button(
#             parent=self, layout=self.receivedGroupBox.layout(), command=self.toggle_transceiver,
#             icon=configuration.connected if self.app.transceiverConfig._transceiverActive else configuration.disconnected,
#             **app.cubesat.widgetkwargs.transceiver.transceiverActiveBtn, )
#         self.setMinimumSize(400, 200)
#         self.resize(400, 200)
#         self.set_stylesheets()
#         self.set_menubar()
#         submenu.SubMenu.__init__(self, parent, title, icon, )
#         self.cmdGroupBox, self.cmdEntry = lozoya.gui.pair(
#             self, pair='entry', layout=self._layout, **app.cubesat.widgetkwargs.transceiver.cmdEntry, )
#         self.sendCmdBtn = lozoya.gui.make_button(
#             self, layout=self.cmdGroupBox.layout(), command=self.transmit,
#             **app.cubesat.widgetkwargs.transceiver.sendCmdBtn, )
#         self.receivedGroupBox, self.receivedMsg = lozoya.gui.pair(
#             self, pair='entry', layout=self._layout, **app.cubesat.widgetkwargs.transceiver.receivedMsg, )
#         self.transceiverActiveBtn = lozoya.gui.make_button(
#             self, layout=self.receivedGroupBox.layout(), command=self.toggle_transceiver,
#             icon=icons.connected if self.parent.configuration.py.transceiverActive else icons.disconnected,
#             **app.cubesat.widgetkwargs.transceiver.transceiverActiveBtn, )
#         self.setMinimumSize(325, 150)
#         self.resize(400, 150)
#         self.set_stylesheets()
#         submenu.SubMenu.__init__(self, parent, title, icon, )
#
#         self.cmdGroupBox, self.cmdEntry = lozoya.gui.pair(
#             self, pair='entry', layout=self._layout, **app.cubesat.widgetkwargs.transceiver.cmdEntry, )
#         self.sendCmdBtn = lozoya.gui.make_button(
#             self, layout=self.cmdGroupBox.layout(), command=self.transmit,
#             **app.cubesat.widgetkwargs.transceiver.sendCmdBtn, )
#         self.receivedGroupBox, self.receivedMsg = lozoya.gui.pair(
#             self, pair='entry', layout=self._layout, **app.cubesat.widgetkwargs.transceiver.receivedMsg, )
#         self.transceiverActiveBtn = lozoya.gui.make_button(
#             self, layout=self.receivedGroupBox.layout(), command=self.toggle_transceiver,
#             icon=icons.connected if self.parent.preferences.transceiverActive else icons.disconnected,
#             **app.cubesat.widgetkwargs.transceiver.transceiverActiveBtn, )
#
#         self.set_stylesheets()
#
#     def change_frequency_dial(self):
#         try:
#             self.frequencyHzEntry.setValue(self.frequencyHzSlider.value())
#         except Exception as e:
#             print(e)
#
#     def change_frequency_entry(self):
#         self.frequencyHzSlider.setValue(self.frequencyHzEntry.value())
#
#     def set_menubar(self, *args, **kwargs):
#         settingsAction = QtWidgets.QAction(QtGui.QIcon(configuration.cog), 'Settings', self)
#         settingsAction.triggered.connect(self.settingsForm.toggle)
#         self.menubar.addAction(settingsAction)
#
#     def set_stylesheets(self, *args, **kwargs):
#         if self.parent.preferences.portConnected and self.parent.preferences.transceiverActive:
#             receivedEntryStyleSheet = qss.entryConnected
#             sendCmdBtnStyleSheet = qss.buttonValid
#             sentEntryStyleSheet = qss.entryValid
#             transceiverActiveButtonStyleSheet = qss.buttonConnected
#             transceiverActiveButtonIcon = icons.connected
#         elif self.parent.preferences.portConnected:
#             receivedEntryStyleSheet = qss.entryDisconnected
#             sendCmdBtnStyleSheet = qss.buttonDisabled
#             sentEntryStyleSheet = qss.entryDisconnected
#             transceiverActiveButtonStyleSheet = qss.buttonDisconnected
#             transceiverActiveButtonIcon = icons.disconnected
#         else:
#             receivedEntryStyleSheet = qss.entryDisabled
#             sendCmdBtnStyleSheet = qss.buttonDisabled
#             sentEntryStyleSheet = qss.entryDisabled
#             transceiverActiveButtonStyleSheet = qss.buttonDisabled
#             transceiverActiveButtonIcon = icons.disconnected
#         self.receivedMsg.setStyleSheet(qss.format_qss(receivedEntryStyleSheet, self.parent.palette))
#         self.cmdEntry.setStyleSheet(qss.format_qss(sentEntryStyleSheet, self.parent.palette))
#         self.sendCmdBtn.setStyleSheet(qss.format_qss(sendCmdBtnStyleSheet, self.parent.palette))
#         self.transceiverActiveBtn.setStyleSheet(qss.format_qss(transceiverActiveButtonStyleSheet, self.parent.palette))
#         self.transceiverActiveBtn.setIcon(QtGui.QIcon(transceiverActiveButtonIcon))
#         self.cmdEntry.setReadOnly(
#             not (self.parent.preferences.portConnected and self.parent.preferences.transceiverActive)
#         )
#         return receivedEntryStyleSheet, sentEntryStyleSheet, transceiverActiveButtonStyleSheet
#         if self.app.fullyLoaded and self.app.transceiverConfig._transceiverActive:
#             self.sendCmdBtn.set_valid()
#             self.cmdEntry.set_valid()
#             self.receivedMsg.set_connected()
#             self._transceiverActiveBtn.set_connected()
#         elif self.app.fullyLoaded:
#             self.sendCmdBtn.set_disabled()
#             self.cmdEntry.set_disabled()
#             self.receivedMsg.set_disabled()
#             self._transceiverActiveBtn.set_glowing()
#         else:
#             self.sendCmdBtn.set_disabled()
#             self.cmdEntry.set_disabled()
#             self.receivedMsg.set_disabled()
#             self._transceiverActiveBtn.set_disabled()
#             self.app.logConfig._autoUpdate = False
#         self.app.logMenu.set_stylesheets()
#         if self.app.fullyLoaded and self.app.transceiverConfig._transceiverActive:
#             self.sendCmdBtn.set_valid()
#             self.cmdEntry.set_valid()
#             self.receivedMsg.set_connected()
#             self._transceiverActiveBtn.set_connected()
#         elif self.app.fullyLoaded:
#             self.sendCmdBtn.set_disabled()
#             self.cmdEntry.set_disabled()
#             self.receivedMsg.set_disabled()
#             self._transceiverActiveBtn.set_glowing()
#         else:
#             self.sendCmdBtn.set_disabled()
#             self.cmdEntry.set_disabled()
#             self.receivedMsg.set_disabled()
#             self._transceiverActiveBtn.set_disabled()
#             self.app.logConfig._autoUpdate = False
#         self.app.logMenu.set_stylesheets()
#         condition = (
#             self.parent.configuration.py.portConnected and self.parent.configuration.py.delimiterLock and self.parent.configuration.py.nVarsLock and self.parent.configuration.py.dataTypeLock and self.parent.configuration.py.headLock and self.parent.configuration.py.unitsLock)
#         if condition and self.parent.configuration.py.transceiverActive:
#             receivedEntryStyleSheet = qss.entryConnected
#             sendCmdBtnStyleSheet = qss.buttonValid
#             sentEntryStyleSheet = qss.entryValid
#             transceiverActiveButtonStyleSheet = qss.buttonConnected
#             transceiverActiveButtonIcon = icons.connected
#         elif self.parent.configuration.py.portConnected:
#             receivedEntryStyleSheet = qss.entryDisconnected
#             sendCmdBtnStyleSheet = qss.buttonDisabled
#             sentEntryStyleSheet = qss.entryDisconnected
#             transceiverActiveButtonStyleSheet = qss.buttonGlow
#             transceiverActiveButtonIcon = icons.disconnected
#         else:
#             receivedEntryStyleSheet = qss.entryDisabled
#             sendCmdBtnStyleSheet = qss.buttonDisabled
#             sentEntryStyleSheet = qss.entryDisabled
#             transceiverActiveButtonStyleSheet = qss.buttonDisabled
#             transceiverActiveButtonIcon = icons.disconnected
#         self.receivedMsg.setStyleSheet(qss.format_qss(receivedEntryStyleSheet, self.parent.palette))
#         self.cmdEntry.setStyleSheet(qss.format_qss(sentEntryStyleSheet, self.parent.palette))
#         self.sendCmdBtn.setStyleSheet(qss.format_qss(sendCmdBtnStyleSheet, self.parent.palette))
#         self.transceiverActiveBtn.setStyleSheet(qss.format_qss(transceiverActiveButtonStyleSheet, self.parent.palette))
#         self.transceiverActiveBtn.setIcon(QtGui.QIcon(transceiverActiveButtonIcon))
#         self.cmdEntry.setReadOnly(not (self.parent.configuration.py.portConnected and self.parent.configuration.py.transceiverActive))
#         self.update_transceiver_active_button()
#         return receivedEntryStyleSheet, sentEntryStyleSheet, transceiverActiveButtonStyleSheet
#
#     def toggle_transceiver(self, *args, **kwargs):
#         try:
#             if self.app.fullyLoaded:
#                 self.app.transceiverConfig._transceiverActive = not self.app.transceiverConfig._transceiverActive
#                 if self.app.transceiverConfig._transceiverActive or self.app.generatorConfig.simulationEnabled:
#                     self.app.comPort.connect()
#                     if self.app.comPort.connected:
#                         self.update_status(*configuration.enabledTransceiver)
#                     else:
#                         self.app.comPort.disconnect()
#                         self.app.transceiverConfig.update('_transceiverActive', False)
#                 else:
#                     self.app.comPort.disconnect()
#                     self.update_status(*configuration.disabledTransceiver)
#                     self.app.transceiverConfig.update('_transceiverActive', False)
#                     self.app.logConfig._autoUpdate = False
#                 self.set_stylesheets()
#             else:
#                 self.app.comPort.disconnect()
#                 self.update_status(*configuration.notFullyLoaded)
#                 self.app.transceiverConfig.update('_transceiverActive', False)
#         except Exception as e:
#             print('Toggle transceiver error: {}'.format(str(e)))
#         condition = (
#             self.parent.configuration.py.portConnected and self.parent.configuration.py.delimiterLock and self.parent.configuration.py.nVarsLock and self.parent.configuration.py.dataTypeLock and self.parent.configuration.py.headLock and self.parent.configuration.py.unitsLock)
#         if condition:
#             self.parent.configuration.py.transceiverActive = not self.parent.configuration.py.transceiverActive
#             if self.parent.configuration.py.transceiverActive:
#                 verb = 'Enabled'
#             else:
#                 verb = 'Disabled'
#                 self.parent.logMenu.autoScroll = False
#             alertMsg = '{} transceiver.'.format(verb)
#             self.update_status(alertMsg, 'success')
#             self.set_stylesheets()
#             self.parent.logMenu.set_stylesheets()
#         else:
#             alertMsg = 'Must connect all parameters in settings menu before enabling transceiver.'
#             self.update_status(alertMsg, 'alert')
#             self.parent.configuration.py.transceiverActive = False
#         self.parent.configuration.py.save_to_file()
#         if self.parent.preferences.portConnected:
#             self.parent.preferences.transceiverActive = not self.parent.preferences.transceiverActive
#             if self.parent.preferences.transceiverActive:
#                 verb = 'Enabled'
#             else:
#                 verb = 'Disabled'
#                 self.parent.logMenu.autoScroll = False
#             alertMsg = '{} transceiver.'.format(verb)
#             self.update_status(alertMsg, qss.statusbarSuccess)
#             self.set_stylesheets()
#             self.parent.logMenu.set_stylesheets()
#         else:
#             alertMsg = 'Must connect to port before engaging transceiver. (File/Settings/Transceiver)'
#             self.update_status(alertMsg, qss.statusbarAlert)
#             self.parent.preferences.transceiverActive = False
#         self.parent.preferences.save_to_file()
#         if self.app.fullyLoaded:
#             self.app.transceiverConfig._transceiverActive = not self.app.transceiverConfig._transceiverActive
#             if self.app.transceiverConfig._transceiverActive:
#                 self.update_status(*configuration.enabledTransceiver)
#             else:
#                 self.update_status(*configuration.disabledTransceiver)
#                 self.app.transceiverConfig.update('transceiverActive', False)
#                 self.app.logConfig._autoUpdate = False
#             self.set_stylesheets()
#         else:
#             self.update_status(*configuration.notFullyLoaded)
#             self.app.transceiverConfig.update('transceiverActive', False)
#
#     def transmit(self):
#         cmd = self.cmdEntry.toPlainText()
#         if not cmd:
#             return
#         else:
#             print('sent: {}'.format(cmd))
#         if self.parent.preferences.transceiverActive:
#             self.parent.transceiver.transmit(self.cmdEntry.toPlainText())
#         else:
#             alertMsg = 'Must enable transceiver to send messages.'
#             self.update_status(alertMsg, qss.statusbarAlert)
#         if self.parent.configuration.py.transceiverActive:
#             self.parent.transceiver.transmit(self.cmdEntry.toPlainText())
#         else:
#             alertMsg = 'Must enable transceiver to send messages.'
#             self.update_status(alertMsg, 'alert')
#
#     def update_transceiver_active_button(self):
#         try:
#             condition = (
#                 self.parent.configuration.py.portConnected and self.parent.configuration.py.delimiterLock and self.parent.configuration.py.nVarsLock and self.parent.configuration.py.dataTypeLock and self.parent.configuration.py.headLock and self.parent.configuration.py.unitsLock)
#             # print(self.parent.configuration.py.portConnected)
#             # print(self.parent.configuration.py.delimiterLock)
#             # print(self.parent.configuration.py.nVarsLock)
#             # print(self.parent.configuration.py.dataTypeLock)
#             # print(self.parent.configuration.py.headLock)
#             # print(self.parent.configuration.py.unitsLock)
#             if condition and self.parent.configuration.py.transceiverActive:
#                 self.cmdEntry.setStyleSheet(qss.format_qss(qss.entryValid, self.parent.palette))
#                 self.receivedMsg.setStyleSheet(qss.format_qss(qss.entryConnected, self.parent.palette))
#                 self.transceiverActiveBtn.setStyleSheet(qss.format_qss(qss.buttonConnected, self.parent.palette))
#             elif condition:
#                 self.cmdEntry.setStyleSheet(qss.format_qss(qss.entryDisconnected, self.parent.palette))
#                 self.receivedMsg.setStyleSheet(qss.format_qss(qss.entryDisconnected, self.parent.palette))
#                 self.transceiverActiveBtn.setStyleSheet(qss.format_qss(qss.buttonGlow, self.parent.palette))
#             else:
#                 self.transceiverActiveBtn.setStyleSheet(qss.format_qss(qss.buttonDisabled, self.parent.palette))
#                 self.cmdEntry.setStyleSheet(qss.format_qss(qss.entryDisabled, self.parent.palette))
#                 self.receivedMsg.setStyleSheet(qss.format_qss(qss.entryDisabled, self.parent.palette))
#                 if self.parent.configuration.py.transceiverActive:
#                     alertMsg = 'Disabled transceiver.'
#                     self.update_status(alertMsg, 'alert')
#                     self.parent.settingsMenu.update_status(alertMsg, 'alert')
#                     self.parent.configuration.py.transceiverActive = False
#                 self.parent.logMenu.autoScroll = False
#                 self.parent.logMenu.set_stylesheets()
#         except Exception as e:
#             alertMsg = 'Update transceiver button error: '.format(str(e))
#             self.update_status(alertMsg)


# class SettingsData:
#     def __init__(self, app, parent, layout):
#         self.app = app
#         self.parent = parent
#         self.layout = layout
#         self.nVarsInputGroupBox, self.nVarsInput = lozoya.gui.pair(
#             self.parent, pair='numerical', inputSettings=(1, 999, self.app.dataConfig.nVars, 1),
#             command=lambda: self.update_n_vars(), layout=layout, readOnly=self.app.dataConfig.nVarsLock,
#             **app.cubesat.widgetkwargs.settings.nVarsInput, )
#         self.nVarsLkBtn = lockbutton.LockButton(
#             app=self.app, parent=self.parent, name='nVarsLock', configuration.py=self.app.dataConfig, pair=self.nVarsInput,
#             callback=self.toggle_transceiver_active, layout=self.nVarsInputGroupBox.layout(), pos=(0, 1),
#             tooltip=tooltips.nVarsLkBtn
#         )
#         self.delimiterEntryGroupBox, self.delimiterEntry = lozoya.gui.pair(
#             self.parent, pair='entry', pairText=self.app.dataConfig.delimiter, command=lambda: self.update_delimiter(),
#             layout=layout, readOnly=self.app.dataConfig.delimiterLock,
#             **app.cubesat.widgetkwargs.settings.delimiterEntry, )
#         self.delimiterLkBtn = lockbutton.LockButton(
#             app=self.app, parent=self.parent, name='delimiterLock', configuration.py=self.app.dataConfig,
#             pair=self.delimiterEntry, callback=self.toggle_transceiver_active,
#             layout=self.delimiterEntryGroupBox.layout(), pos=(0, 1), tooltip=tooltips.delimiterLkBtn
#         )
#         self.dataTypeEntryGroupBox, self.dataTypeEntry = lozoya.gui.pair(
#             self.parent, pair='entry', pairText=self.app.dataConfig.dataType, command=lambda: self.update_dtype(),
#             layout=layout, readOnly=self.app.dataConfig.dataTypeLock,
#             **app.cubesat.widgetkwargs.settings.dataTypeEntry, )
#         self.dataTypeLkBtn = lockbutton.LockButton(
#             app=self.app, parent=self.parent, name='dataTypeLock', configuration.py=self.app.dataConfig, pair=self.dataTypeEntry,
#             callback=self.toggle_transceiver_active, layout=self.dataTypeEntryGroupBox.layout(), pos=(0, 1),
#             tooltip=tooltips.dataTypeLkBtn
#         )
#         self.headEntryGroupBox, self.headEntry = lozoya.gui.pair(
#             self.parent, pair='entry', pairText=self.app.dataConfig.head, command=lambda: self.update_property('head'),
#             layout=layout, readOnly=self.app.dataConfig.headLock, **app.cubesat.widgetkwargs.settings.headEntry, )
#         self.headLkBtn = lockbutton.LockButton(
#             app=self.app, parent=self.parent, name='headLock', configuration.py=self.app.dataConfig, pair=self.headEntry,
#             callback=self.toggle_transceiver_active, layout=self.headEntryGroupBox.layout(), pos=(0, 1),
#             tooltip=tooltips.headLkBtn, )
#         self.unitsEntryGroupBox, self.unitsEntry = lozoya.gui.pair(
#             self.parent, pair='entry', pairText=self.app.dataConfig.units, command=lambda: self.update_property(
#                 'units'
#             ), layout=layout, readOnly=self.app.dataConfig.unitsLock, **app.cubesat.widgetkwargs.settings.unitsEntry, )
#         self.unitsLkBtn = lockbutton.LockButton(
#             app=self.app, parent=self.parent, name='unitsLock', configuration.py=self.app.dataConfig, pair=self.unitsEntry,
#             callback=self.toggle_transceiver_active, layout=self.unitsEntryGroupBox.layout(), pos=(0, 1),
#             tooltip=tooltips.unitsLkBtn
#         )
#         self.bufferGroupBox, self.bufferSlider = lozoya.gui.pair(
#             self.parent, pair='slider', layout=layout, inputSettings=(2, 100000, self.app.dataConfig.buffer, 10),
#             command=lambda: self.update_buffer(), pairSS=interface.styles.slider.slider,
#             **app.cubesat.widgetkwargs.settings.bufferSlider, )
#         self.set_stylesheets()
#         self.bufferGroupBox, self.bufferSlider = lozoya.gui.pair(
#             self.parent, pair='slider', layout=layout, settings=(2, 100000, self.app.dataConfig.buffer, 10),
#             command=lambda: self.update_buffer(), pairSS=lozoya.gui.styles.sliderstyles.slider,
#             **app.cubesat.widgetkwargs.settings.bufferSlider, )
#         self.set_stylesheets()
#         self.bufferGroupBox, self.bufferSlider = lozoya.gui.pair(
#             self.parent, pair='slider', layout=layout, settings=(2, 100000, self.app.dataConfig.buffer, 10),
#             command=lambda: self.update_buffer(), pairSS=gui.styles.sliderstyles.slider,
#             **app.cubesat.widgetkwargs.settings.bufferSlider, )
#         self.set_stylesheets()
#
#     def set_property(self, property, value):
#         self.app.dataConfig.update('{}Valid'.format(property), value)
#         self.app.dataConfig.update('{}Lock'.format(property), value)
#         vars(self)['{}Entry'.format(property)].setReadOnly(value)
#
#     def set_stylesheets(self, *args, **kwargs):
#         for property in ['delimiter', 'nVars']:
#             if getattr(self.app.dataConfig, '{}Lock'.format(property)):
#                 getattr(self, '{}LkBtn'.format(property)).set_connected()
#             else:
#                 getattr(self, '{}LkBtn'.format(property)).set_glowing()
#         for property in ['dataType', 'head', 'units']:
#             lock = getattr(self.app.dataConfig, '{}Lock'.format(property))
#             valid = getattr(self.app.dataConfig, '{}Valid'.format(property))
#             if valid and lock:
#                 getattr(self, '{}LkBtn'.format(property)).set_connected()
#             elif valid:
#                 getattr(self, '{}LkBtn'.format(property)).set_glowing()
#             else:
#                 getattr(self, '{}LkBtn'.format(property)).set_disabled()
#
#     def update_buffer(self, *args, **kwargs):
#         try:
#             self.toggle_transceiver_active()
#             self.app.dataConfig.update('buffer', self.bufferSlider.value())
#             self.app.plotMenu.plotSettings.plotDomainSlider.setRange(0, self.app.dataConfig.buffer - 10)
#             for plot in self.app.plotDude.plots:
#                 plot.ax.set_xlim(self.app.plotMenu.plotSettings.plotDomainSlider.value(), self.app.dataConfig.buffer)
#             self.parent.update_status('Buffer: {} samples.'.format(self.bufferSlider.value()), 'success')
#         except Exception as e:
#             statusMsg = 'Update buffer error: {}'.format(str(e))
#             self.parent.update_status(statusMsg, 'error')
#
#     def update_delimiter(self, *args, **kwargs):
#         self.app.transceiverConfig.update('_transceiverActive', False)
#         txt = self.delimiterEntry.text().strip('\n')
#         if txt == '':
#             self.parent.update_status(*configuration.emptyDelimiter)
#             self.app.dataConfig.update('delimiterValid', False)
#         else:
#             self.app.dataConfig.update('delimiterValid', True)
#         self.app.dataConfig.update('delimiter', txt)
#         self.update_dtype()
#         for property in ['head', 'units']:
#             self.update_property(property)
#         self.set_stylesheets()
#         self.app.transceiverConfig.update('transceiverActive', False)
#
#     def update_dtype(self, *args, **kwargs):
#         txt = self.dataTypeEntry.text().strip('\n')
#         self.update_property('dataType')
#         for _ in txt.split(self.app.dataConfig.delimiter):
#             if _ not in ['d', 'f', 'h', 's', 't']:  # int, float, hex, str, time
#                 self.app.dataConfig.update('dataTypeValid', False)
#         self.set_stylesheets()
#
#     def update_n_vars(self, *args, **kwargs):
#         self.toggle_transceiver_active(disable=True)
#         self.app.dataConfig.update('nVars', self.nVarsInput.value())
#         self.update_dtype()
#         for property in ['head', 'units']:
#             self.update_property(property)
#         self.set_stylesheets()
#
#     def update_property(self, property):
#         self.toggle_transceiver_active(disable=True)
#         txt = getattr(self, '{}Entry'.format(property)).text().strip('\n')
#         if len(txt.split(self.app.dataConfig.delimiter)) != self.app.dataConfig.nVars:
#             self.set_property(property, False)
#         else:
#             self.app.dataConfig.update('{}Valid'.format(property), True)
#         self.app.dataConfig.update(property, txt)
#         self.set_stylesheets()
#
#     def toggle_transceiver_active(self, disable=False):
#         if self.app.dataConfig.delimiterValid:
#             self.app.transceiverMenu.set_stylesheets()
#         if disable:
#             self.app.transceiverConfig.update('transceiverActive', False)
#         self.app.deviceMenu.reset_menu()
#         if self.app.dataConfig.delimiterValid:
#             self.app.transceiverMenu.set_stylesheets()
#         if disable:
#             self.app.transceiverConfig.update('_transceiverActive', False)
#         self.app.deviceMenu.reset_menu()
#
#
# class SettingsDevice:
#     def __init__(self, app, layout):
#         try:
#             self.app = app
#             self.layout = layout
#             self.coordinateDataGroupBox = lozoya.gui.make_group_box(
#                 parent=layout, **app.cubesat.widgetkwargs.device.coordinateDataGroupBox
#             )
#             _, self.xCombo = lozoya.gui.pair(
#                 parent=self.parent, pair='combo', comboItems=[], default=0, layout=self.coordinateDataGroupBox.layout(),
#                 command=lambda: self.update_pos_dropdown('x'), **app.cubesat.widgetkwargs.device.xCombo, )
#             _, self.yCombo = lozoya.gui.pair(
#                 parent=self.parent, pair='combo', comboItems=[], default=0, layout=self.coordinateDataGroupBox.layout(),
#                 command=lambda: self.update_pos_dropdown('y'), **app.cubesat.widgetkwargs.device.yCombo, )
#             _, self.zCombo = lozoya.gui.pair(
#                 parent=self.parent, pair='combo', comboItems=[], default=0, layout=self.coordinateDataGroupBox.layout(),
#                 command=lambda: self.update_pos_dropdown('z'), **app.cubesat.widgetkwargs.device.zCombo, )
#             self.units = ['Degrees', 'Radians']
#             _, self.unitsCombo = lozoya.gui.pair(
#                 parent=self.parent, pair='combo', comboItems=self.units, default=self.app.deviceConfig.unitsIndex,
#                 layout=self.coordinateDataGroupBox.layout(), command=lambda: self.update_units(),
#                 **app.cubesat.widgetkwargs.device.units, )
#         except Exception as e:
#             print('init device settings error: {}.'.format(str(e)))
#
#     def update_pos_dropdown(self, axis):
#         head = ['None'] + self.app.dataDude.get_plottable()
#         index = getattr(self, axis + 'Combo').currentIndex()
#         self.app.deviceConfig.update(axis, head[index])
#         self.app.deviceConfig.update(axis + 'Index', index)
#
#     def update_units(self, *args, **kwargs):
#         index = self.unitsCombo.currentIndex()
#         self.app.deviceConfig.update('units', self.units[index])
#         self.app.deviceConfig.update('unitsIndex', index)
#
#
# class SettingsDeviceRender:
#     def __init__(self, app, layout):
#         try:
#             self.app = app
#             self.layout = layout
#             _, self.renderAxesQuiverCheckBox = lozoya.gui.pair(
#                 self.parent, pair='check', layout=layout, gbHeight=40, pairKwargs={
#                     'command'  : lambda: self.update_render_axes_quiver(),
#                     'isChecked': self.app.deviceConfig.renderAxesQuiver, 'row': 0, 'column': 0,
#                     'tooltip'  : tooltips.renderAxesQuiver,
#                 }, **app.cubesat.widgetkwargs.settings.renderAxesQuiverCheckBox, )
#             _, self.renderCubeQuiverCheckBox = lozoya.gui.pair(
#                 self.parent, pair='check', text='Cube Quiver', layout=layout, gbHeight=40, pairKwargs={
#                     'command'  : lambda: self.update_render_cube_quiver(),
#                     'isChecked': self.app.deviceConfig.renderCubeQuiver, 'row': 0, 'column': 0,
#                     'tooltip'  : tooltips.renderCubeQuiver,
#                 }, **app.cubesat.widgetkwargs.settings.renderCubeQuiverCheckBox, )
#             _, self.renderFacesCheckBox = lozoya.gui.pair(
#                 self.parent, pair='check', layout=layout, gbHeight=40, pairKwargs={
#                     'command': lambda: self.update_render_faces(), 'isChecked': self.app.deviceConfig.renderFaces,
#                     'row'    : 0, 'column': 0, 'tooltip': tooltips.renderFaces,
#                 }, **app.cubesat.widgetkwargs.settings.renderFacesCheckBox, )
#             _, self.renderWireframeCheckBox = lozoya.gui.pair(
#                 self.parent, pair='check', layout=layout, gbHeight=40, pairKwargs={
#                     'command'  : lambda: self.update_render_wireframe(),
#                     'isChecked': self.app.deviceConfig.renderWireframe, 'row': 0, 'column': 0,
#                     'tooltip'  : tooltips.renderWireframe,
#                 }, **app.cubesat.widgetkwargs.settings.renderWireframeCheckBox
#             )
#             _, self.renderThrustersCheckBox = lozoya.gui.pair(
#                 self.parent, pair='check', layout=layout, gbHeight=40, pairKwargs={
#                     'command'  : lambda: self.update_render_thrusters(),
#                     'isChecked': self.app.deviceConfig.renderThrusters, 'row': 0, 'column': 0,
#                     'tooltip'  : tooltips.renderThrusters,
#                 }, **app.cubesat.widgetkwargs.settings.renderThrustersCheckBox, )
#             _, self.renderAVCheckBox = lozoya.gui.pair(
#                 self.parent, pair='check', layout=layout, gbHeight=40, pairKwargs={
#                     'command': lambda: self.update_render_av(), 'isChecked': self.app.deviceConfig.renderAV, 'row': 0,
#                     'column' : 0, 'tooltip': tooltips.renderAV,
#                 }, **app.cubesat.widgetkwargs.settings.renderAVCheckBox, )
#         except Exception as e:
#             print(e)
#
#     def update_render_av(self, *args, **kwargs):
#         try:
#             self.app.deviceConfig.update('renderAV', not self.app.deviceConfig.renderAV)
#             if self.app.deviceConfig.renderCubeQuiver:
#                 self.app.plotDude.devicePlot.draw_av()
#                 self.parent.update_status('Angular Velocity rendering enabled.', 'success')
#                 self.renderAVCheckBox.setStyleSheet(
#                     gui.qss._format(lozoya.gui.styles.btnstyles.checkboxChecked, self.app.palette)
#                 )
#             else:
#                 self.app.plotDude.devicePlot.remove_av()
#                 self.parent.update_status('Angular Velocity rendering disabled.', 'success')
#                 self.renderAVCheckBox.setStyleSheet(
#                     gui.qss._format(lozoya.gui.styles.btnstyles.checkboxUnchecked, self.app.palette)
#                 )
#         except Exception as e:
#             statusMsg = 'Update render angular velocity error: {}'.format(str(e))
#             self.parent.update_status(statusMsg, 'error')
#         try:
#             self.app.deviceConfig.update('renderAV', not self.app.deviceConfig.renderAV)
#             if self.app.deviceConfig.renderCubeQuiver:
#                 self.app.plotDude.devicePlot.draw_av()
#                 self.parent.update_status('Angular Velocity rendering enabled.', 'success')
#                 self.renderAVCheckBox.setStyleSheet(
#                     qss._format(interface.styles.btnstyles.checkboxChecked, self.app.palette)
#                 )
#             else:
#                 self.app.plotDude.devicePlot.remove_av()
#                 self.parent.update_status('Angular Velocity rendering disabled.', 'success')
#                 self.renderAVCheckBox.setStyleSheet(
#                     qss._format(interface.styles.btnstyles.checkboxUnchecked, self.app.palette)
#                 )
#         except Exception as e:
#             statusMsg = 'Update render angular velocity error: {}'.format(str(e))
#             self.parent.update_status(statusMsg, 'error')
#
#     def update_render_axes_quiver(self, *args, **kwargs):
#         try:
#             self.app.deviceConfig.update('renderAxesQuiver', not self.app.deviceConfig.renderAxesQuiver)
#             if self.app.deviceConfig.renderAxesQuiver:
#                 self.app.plotDude.devicePlot.draw_axes_quiver()
#                 self.parent.update_status('Axes quiver rendering enabled.', 'success')
#                 self.renderAxesQuiverCheckBox.setStyleSheet(
#                     gui.qss._format(lozoya.gui.styles.btnstyles.checkboxChecked, self.app.palette)
#                 )
#             else:
#                 self.app.plotDude.devicePlot.remove_axes_quiver()
#                 self.parent.update_status('Axes quiver rendering disabled.', 'success')
#                 self.renderAxesQuiverCheckBox.setStyleSheet(
#                     gui.qss._format(lozoya.gui.styles.btnstyles.checkboxUnchecked, self.app.palette)
#                 )
#         except Exception as e:
#             statusMsg = 'Update render axes quiver error: {}'.format(str(e))
#             self.parent.update_status(statusMsg, 'error')
#         try:
#             self.app.deviceConfig.update('renderAxesQuiver', not self.app.deviceConfig.renderAxesQuiver)
#             if self.app.deviceConfig.renderAxesQuiver:
#                 self.app.plotDude.devicePlot.draw_axes_quiver()
#                 self.parent.update_status('Axes quiver rendering enabled.', 'success')
#                 self.renderAxesQuiverCheckBox.setStyleSheet(
#                     qss._format(interface.styles.btnstyles.checkboxChecked, self.app.palette)
#                 )
#             else:
#                 self.app.plotDude.devicePlot.remove_axes_quiver()
#                 self.parent.update_status('Axes quiver rendering disabled.', 'success')
#                 self.renderAxesQuiverCheckBox.setStyleSheet(
#                     qss._format(interface.styles.btnstyles.checkboxUnchecked, self.app.palette)
#                 )
#         except Exception as e:
#             statusMsg = 'Update render axes quiver error: {}'.format(str(e))
#             self.parent.update_status(statusMsg, 'error')
#
#     def update_render_cube_quiver(self, *args, **kwargs):
#         try:
#             self.app.deviceConfig.update('renderCubeQuiver', not self.app.deviceConfig.renderCubeQuiver)
#             if self.app.deviceConfig.renderCubeQuiver:
#                 self.app.plotDude.devicePlot.draw_cube_quiver()
#                 self.parent.update_status('Cube quiver rendering enabled.', 'success')
#                 self.renderCubeQuiverCheckBox.setStyleSheet(
#                     qss._format(interface.styles.btnstyles.checkboxChecked, self.app.palette)
#                 )
#             else:
#                 self.app.plotDude.devicePlot.remove_cube_quiver()
#                 self.parent.update_status('Cube quiver rendering disabled.', 'success')
#                 self.renderCubeQuiverCheckBox.setStyleSheet(
#                     qss._format(interface.styles.btnstyles.checkboxUnchecked, self.app.palette)
#                 )
#         except Exception as e:
#             statusMsg = 'Update render cube quiver error: {}'.format(str(e))
#             self.parent.update_status(statusMsg, 'error')
#         try:
#             self.app.deviceConfig.update('renderCubeQuiver', not self.app.deviceConfig.renderCubeQuiver)
#             if self.app.deviceConfig.renderCubeQuiver:
#                 self.app.plotDude.devicePlot.draw_cube_quiver()
#                 self.parent.update_status('Cube quiver rendering enabled.', 'success')
#                 self.renderCubeQuiverCheckBox.setStyleSheet(
#                     gui.qss._format(lozoya.gui.styles.btnstyles.checkboxChecked, self.app.palette)
#                 )
#             else:
#                 self.app.plotDude.devicePlot.remove_cube_quiver()
#                 self.parent.update_status('Cube quiver rendering disabled.', 'success')
#                 self.renderCubeQuiverCheckBox.setStyleSheet(
#                     gui.qss._format(lozoya.gui.styles.btnstyles.checkboxUnchecked, self.app.palette)
#                 )
#         except Exception as e:
#             statusMsg = 'Update render cube quiver error: {}'.format(str(e))
#             self.parent.update_status(statusMsg, 'error')
#
#     def update_render_faces(self, *args, **kwargs):
#         try:
#             self.app.deviceConfig.update('renderFaces', not self.app.deviceConfig.renderFaces)
#             if self.app.deviceConfig.renderFaces:
#                 self.app.plotDude.devicePlot.draw_cube_faces()
#                 self.parent.update_status('Cube face rendering enabled.', 'success')
#                 self.renderFacesCheckBox.setStyleSheet(
#                     qss._format(interface.styles.btnstyles.checkboxChecked, self.app.palette)
#                 )
#             else:
#                 self.app.plotDude.devicePlot.remove_cube_faces()
#                 self.parent.update_status('Cube face rendering disabled.', 'success')
#                 self.renderFacesCheckBox.setStyleSheet(
#                     qss._format(interface.styles.btnstyles.checkboxUnchecked, self.app.palette)
#                 )
#         except Exception as e:
#             statusMsg = 'Update render cube faces error: {}'.format(str(e))
#             self.parent.update_status(statusMsg, 'error')
#         try:
#             self.app.deviceConfig.update('renderFaces', not self.app.deviceConfig.renderFaces)
#             if self.app.deviceConfig.renderFaces:
#                 self.app.plotDude.devicePlot.draw_cube_faces()
#                 self.parent.update_status('Cube face rendering enabled.', 'success')
#                 self.renderFacesCheckBox.setStyleSheet(
#                     gui.qss._format(lozoya.gui.styles.btnstyles.checkboxChecked, self.app.palette)
#                 )
#             else:
#                 self.app.plotDude.devicePlot.remove_cube_faces()
#                 self.parent.update_status('Cube face rendering disabled.', 'success')
#                 self.renderFacesCheckBox.setStyleSheet(
#                     gui.qss._format(lozoya.gui.styles.btnstyles.checkboxUnchecked, self.app.palette)
#                 )
#         except Exception as e:
#             statusMsg = 'Update render cube faces error: {}'.format(str(e))
#             self.parent.update_status(statusMsg, 'error')
#
#     def update_render_thrusters(self, *args, **kwargs):
#         try:
#             self.app.deviceConfig.update('renderThrusters', not self.app.deviceConfig.renderThrusters)
#             if self.app.deviceConfig.renderThrusters:
#                 self.app.plotDude.devicePlot.show_thrusters()
#                 self.parent.update_status('Thruster rendering enabled.', 'success')
#                 self.renderThrustersCheckBox.setStyleSheet(
#                     qss._format(interface.styles.btnstyles.checkboxChecked, self.app.palette)
#                 )
#             else:
#                 self.app.plotDude.devicePlot.hide_thrusters()
#                 self.parent.update_status('Thruster rendering disabled.', 'success')
#                 self.renderThrustersCheckBox.setStyleSheet(
#                     qss._format(interface.styles.btnstyles.checkboxUnchecked, self.app.palette)
#                 )
#         except Exception as e:
#             statusMsg = 'Update render thrusters error: {}'.format(str(e))
#             self.parent.update_status(statusMsg, 'error')
#         try:
#             self.app.deviceConfig.update('renderThrusters', not self.app.deviceConfig.renderThrusters)
#             if self.app.deviceConfig.renderThrusters:
#                 self.app.plotDude.devicePlot.show_thrusters()
#                 self.parent.update_status('Thruster rendering enabled.', 'success')
#                 self.renderThrustersCheckBox.setStyleSheet(
#                     gui.qss._format(lozoya.gui.styles.btnstyles.checkboxChecked, self.app.palette)
#                 )
#             else:
#                 self.app.plotDude.devicePlot.hide_thrusters()
#                 self.parent.update_status('Thruster rendering disabled.', 'success')
#                 self.renderThrustersCheckBox.setStyleSheet(
#                     gui.qss._format(lozoya.gui.styles.btnstyles.checkboxUnchecked, self.app.palette)
#                 )
#         except Exception as e:
#             statusMsg = 'Update render thrusters error: {}'.format(str(e))
#             self.parent.update_status(statusMsg, 'error')
#
#     def update_render_wireframe(self, *args, **kwargs):
#         try:
#             self.app.deviceConfig.update('renderWireframe', not self.app.deviceConfig.renderWireframe)
#             if self.app.deviceConfig.renderWireframe:
#                 self.app.plotDude.devicePlot.draw_wireframe()
#                 self.parent.update_status('Cube wireframe rendering enabled.', 'success')
#                 self.renderWireframeCheckBox.setStyleSheet(
#                     gui.qss._format(lozoya.gui.styles.btnstyles.checkboxChecked, self.app.palette)
#                 )
#             else:
#                 self.app.plotDude.devicePlot.remove_wireframe()
#                 self.parent.update_status('Cube wireframe rendering disabled.', 'success')
#                 self.renderWireframeCheckBox.setStyleSheet(
#                     gui.qss._format(lozoya.gui.styles.btnstyles.checkboxUnchecked, self.app.palette)
#                 )
#         except Exception as e:
#             statusMsg = 'Update render wireframe error: {}'.format(str(e))
#             self.parent.update_status(statusMsg, 'error')
#         try:
#             self.app.deviceConfig.update('renderWireframe', not self.app.deviceConfig.renderWireframe)
#             if self.app.deviceConfig.renderWireframe:
#                 self.app.plotDude.devicePlot.draw_wireframe()
#                 self.parent.update_status('Cube wireframe rendering enabled.', 'success')
#                 self.renderWireframeCheckBox.setStyleSheet(
#                     qss._format(interface.styles.btnstyles.checkboxChecked, self.app.palette)
#                 )
#             else:
#                 self.app.plotDude.devicePlot.remove_wireframe()
#                 self.parent.update_status('Cube wireframe rendering disabled.', 'success')
#                 self.renderWireframeCheckBox.setStyleSheet(
#                     qss._format(interface.styles.btnstyles.checkboxUnchecked, self.app.palette)
#                 )
#         except Exception as e:
#             statusMsg = 'Update render wireframe error: {}'.format(str(e))
#             self.parent.update_status(statusMsg, 'error')
#
#
# class SettingsGenerator:
#     def __init__(self, app, parent, layout):
#         try:
#             self.app = app
#             self.parent = parent
#             self.layout = layout
#             _, self.enableSimulatorCheckBox = lozoya.gui.pair(
#                 self.parent, pair='check', layout=layout, pairKwargs={
#                     'command': lambda: self.toggle_simulation(), 'checked': self.app.generatorConfig.simulationEnabled,
#                     'row'    : 0, 'column': 0, 'tooltip': 'Enable simulation.',
#                 }, **app.cubesat.widgetkwargs.settings.enableSimulationCheckBox, )
#             self.methodsEntryGroupBox, self.methodsEntry = lozoya.gui.pair(
#                 self.parent, pair='entry', pairText=self.app.generatorConfig.methods,
#                 command=lambda: self.update_methods(), layout=layout, readOnly=self.app.generatorConfig.methodsLock,
#                 **app.cubesat.widgetkwargs.settings.methodsEntry, )
#             self.methodsLkBtn = lockbutton.LockButton(
#                 app=self.app, parent=self.parent, name='methodsLock', configuration.py=self.app.generatorConfig,
#                 pair=self.methodsEntry, callback=None, layout=self.methodsEntryGroupBox.layout(), pos=(0, 1),
#                 tooltip='', )
#             self.reliabilityInputGroupBox, self.reliabilityInput = lozoya.gui.pair(
#                 self.parent, pair='numerical', settings=(1, 100, self.app.generatorConfig.reliability, 1),
#                 command=lambda: self.update_reliability(), layout=layout, readOnly=self.app.dataConfig.nVarsLock,
#                 **app.cubesat.widgetkwargs.settings.reliabilityInput, )
#             self.reliabilityLkBtn = lockbutton.LockButton(
#                 app=self.app, parent=self.parent, name='reliabilityLock', configuration.py=self.app.generatorConfig,
#                 pair=self.reliabilityInput, callback=None, layout=self.reliabilityInputGroupBox.layout(), pos=(0, 1),
#                 tooltip='', )
#             self.update_methods()
#         except Exception as e:
#             print(e)
#
#     def set_property(self, property, value):
#         self.app.dataConfig.update('{}Valid'.format(property), value)
#         self.app.dataConfig.update('{}Lock'.format(property), value)
#         vars(self)['{}Entry'.format(property)].setReadOnly(value)
#
#     def toggle_simulation(self, *args, **kwargs):
#         simulationStatus = not self.app.generatorConfig.simulationEnabled
#         if simulationStatus:
#             self.enableSimulatorCheckBox.set_checked()
#         else:
#             self.enableSimulatorCheckBox.set_unchecked()
#         self.app.generatorConfig.update('simulationEnabled', simulationStatus)
#
#     def update_methods(self, *args, **kwargs):
#         txt = self.methodsEntry.text().strip('\n')
#         if len(txt.split(self.app.dataConfig.delimiter)) != self.app.dataConfig.nVars:
#             self.set_property('methods', False)
#             self.methodsLkBtn.set_disabled()
#         else:
#             self.app.dataConfig.update('methodsValid', True)
#             self.methodsLkBtn.set_glowing()
#         self.app.generatorConfig.update('methods', txt)
#
#     def update_reliability(self, *args, **kwargs):
#         self.app.generatorConfig.update(
#             'reliability', self.reliabilityInput.value()
#         )  # print(inspect.stack()[1].function)


# class SettingsPlot:
#     def __init__(self, app, parent, layout):
#         try:
#             self.app = app
#             self.parent = parent
#             self.layout = layout
#             self.domainGroupBox = lozoya.gui.make_group_box(
#                 layout, height=54, **app.cubesat.widgetkwargs.plot.domainGroupBox, )
#             _, self.plotDomainSlider = lozoya.gui.pair(
#                 parent=self.parent, pair='slider', text='',
#                 settings=(0, self.app.dataConfig.buffer - 10, self.app.plotConfig.plots[-1]['xmin'], 1),
#                 command=lambda: self.update_plot_xlim(0), layout=self.domainGroupBox.layout(), row=0, column=0,
#                 tooltip=tooltips.domainSlider, gbHeight=40, )
#             self.rangeGroupBox = lozoya.gui.make_group_box(
#                 layout, height=54, **app.cubesat.widgetkwargs.plot.rangeGroupBox, )
#             _, self.plotMinYSlider = lozoya.gui.pair(
#                 parent=self.parent, pair='slider', text='Minimum',
#                 settings=(-10000, 10000, self.app.plotConfig.plots[-1]['ymin'], 1),
#                 command=lambda: self.update_plot_ylim(0), layout=self.rangeGroupBox.layout(), row=0, column=0,
#                 tooltip=tooltips.minYSlider, gbHeight=40, )
#             _, self.plotMaxYSlider = lozoya.gui.pair(
#                 parent=self.parent, pair='slider', text='Maximum',
#                 settings=(-10000, 10000, self.app.plotConfig.plots[-1]['ymax'], 1),
#                 command=lambda: self.update_plot_ylim(0), layout=self.rangeGroupBox.layout(), row=0, column=1,
#                 tooltip=tooltips.maxYSlider, gbHeight=40, )
#             self.variablesGroupBox = lozoya.gui.make_group_box(
#                 layout, height=54, **app.cubesat.widgetkwargs.plot.variablesGroupBox, )
#             options = ['None'] + self.app.dataDude.get_plottable()
#             trans = ['None'] + list(sorted(self.app.transformer.transforms.keys()))
#             self.transformGroupBox, self.transformCombo = lozoya.gui.pair(
#                 parent=self.parent, pair='combo', text='Transform', comboItems=trans, default=trans.index(
#                     self.app.plotConfig.plots[0]['transform']
#                 ), layout=self.variablesGroupBox.layout(), command=lambda: self.change_transform(
#                     0
#                 ), gbHeight=40, **app.cubesat.widgetkwargs.plot.xCombo, )
#             self.yGroupBox, self.yCombo = lozoya.gui.pair(
#                 parent=self.parent, pair='combo', text='y', comboItems=options, default=options.index(
#                     self.app.plotConfig.plots[0]['y']
#                 ), layout=self.variablesGroupBox.layout(), command=lambda: self.change_y(0), gbHeight=40,
#                 **app.cubesat.widgetkwargs.plot.yCombo, )
#         except Exception as e:
#             print('init plot settings error: ' + str(e))
#
#     def change_transform(self, var):
#         transforms = ['None'] + list(sorted(self.app.transformer.transforms.keys()))
#         self.app.plotDude.plots[var].transform = transforms[self.transformCombo.currentIndex()]
#         self.app.plotConfig.update('plots', [plot.dna for plot in self.app.plotDude.plots])
#         self.app.plotConfig.save_to_file()
#
#     def change_y(self, var):
#         try:
#             head = ['None'] + self.app.dataDude.get_plottable()
#             newY = head[self.yCombo.currentIndex()]
#             if newY != 'None':
#                 self.app.plotDude.plots[var].ax.set_ylabel('${}$'.format(self.get_units(newY)))
#             self.app.plotDude.plots[var].y = newY
#             self.app.plotConfig.update('plots', [plot.dna for plot in self.app.plotDude.plots])
#             self.app.plotConfig.save_to_file()
#         except Exception as e:
#             print('change y error: ' + str(e))
#
#     def get_units(self, newY):
#         head = self.app.dataConfig.head.split(self.app.dataConfig.delimiter)
#         units = self.app.dataConfig.units.split(self.app.dataConfig.delimiter)
#         return units[head.index(newY)]
#
#     def update_plot_xlim(self, var):
#         try:
#             self.app.plotDude.plots[var].ax.set_xlim(
#                 xmin=self.plotDomainSlider.value(), xmax=self.app.dataConfig.buffer, )
#             self.app.plotDude.plots[var].xmin = self.plotDomainSlider.value()
#             self.app.plotConfig.update('plots', [plot.dna for plot in self.app.plotDude.plots])
#             self.app.plotConfig.save_to_file()
#         except Exception as e:
#             print('update plot xlim error: ' + str(e))
#
#     def update_plot_ylim(self, var):
#         try:
#             self.app.plotDude.plots[var].ax.set_ylim(
#                 ymin=self.plotMinYSlider.value(), ymax=self.plotMaxYSlider.value(), )
#             self.app.plotDude.plots[var].ymin = self.plotMinYSlider.value()
#             self.plotMinYSlider.setRange(-10000, self.plotMaxYSlider.value() - 10)
#             self.plotMaxYSlider.setRange(self.plotMinYSlider.value() + 10, 10000)
#             self.app.plotDude.plots[var].ymax = self.plotMaxYSlider.value()
#             self.app.plotConfig.update('plots', [plot.dna for plot in self.app.plotDude.plots])
#             self.app.plotConfig.save_to_file()
#         except Exception as e:
#             print('update plot ylim error: ' + str(e))


# def apply_thrust(t, direction):
#     if direction == 'halt':
#         return 0
#     force = maxForce * (t / timeToMaxForce)
#     if direction == 'reverse':
#         return -force
#     return force
#
#
# def brute_rename_all_paths(path, ignoredDirectories, ignoredSubstrings, searchedExtensions):
#     for root, directories, files in os.walk(path):
#         try:
#             ignoreDirectory = match_substring(root, ignoredDirectories)
#             if (ignoreDirectory == False):
#                 update_directories(root, directories)
#                 for filename in files:
#                     ignoreFile = match_substring(filename, ignoredSubstrings) and not match_substring(
#                         filename, searchedExtensions
#                     )
#                     if (ignoreFile == False):
#                         update_filename(root, filename)
#         except Exception as e:
#             pass
#     print("Renamed all paths.")
#
#
# def cubeRCallback(*args):
#     msg, timeMS = args
#     if msg:
#         try:
#             cubeSat.write(msg)
#         except Exception as e:
#             return str(e)
#         try:
#             x = cubeSat.readline()
#             y = cubeSat.readline()
#             z = cubeSat.readline()
#             return '{},{},{}'.format(x, y, z)
#         except Exception as e:
#             return str(e)
#     return 'ERROR'
#
#
# def emit_pdf(finished):
#     loader.show()
#     loader.page().printToPdf("test.pdf")
#
#
# def get_angular_velocity(force, angularVelocity):
#     return angularVelocity + force / 10
#
#
# def get_displacement(angularVelocity, displacement):
#     return displacement + angularVelocity / 1000
#
#
# def get_position(displacement):
#     return displacement % 360
#
#
# def main_loop(degrees):
#     totalTime = 0
#     rotationTime = 0
#     forceTime = 0
#     direction = 'forward'
#     displacement = 0
#     angularVelocity = 0
#     criticalD = 0
#     while 1:
#         force = apply_thrust(forceTime, direction)
#         angularVelocity = get_angular_velocity(force, angularVelocity)
#         displacement = get_displacement(angularVelocity, displacement)
#         position = get_position(displacement)
#         timeVals.append(totalTime)
#         forceVals.append(force)
#         angularVelocityVals.append(angularVelocity)
#         displacementVals.append(displacement)
#         positionVals.append(position)
#         print(
#             'Time: {}ms | Force: {}N | '
#             'Angular Velocity: {}Hz | '
#             'Displacement: {}Degrees | Position: {}Degrees | '
#             'Direction: {} | '.format(totalTime, force, angularVelocity, displacement, position, direction)
#         )
#         totalTime += 1
#         rotationTime += 1
#         if abs(force) >= maxForce:
#             criticalD = displacement
#             direction = 'halt'
#             forceTime = 0
#         if displacement >= degrees - criticalD:
#             direction = 'reverse'
#         if direction != 'halt':
#             forceTime += 1
#         if angularVelocity <= 0 and totalTime > 1:
#             timeVals.append(totalTime)
#             forceVals.append(0)
#             angularVelocityVals.append(0)
#             positionVals.append(position)
#             return
#
#
# def plotter(*args):
#     msg, timeMS = args
#     try:
#         t = int(str(timeMS)[-4:-2])
#         msg = str(msg.split(r"b'")[1])
#         msg = str(msg.split(r'\r')[0])
#         x, y, z = [round(float(_)) for _ in msg.split(',')]
#         global graphics
#         global cube
#         global _x, _y, _z
#         x, y = 0, 0
#         x = -0.0174533 * x
#         y = -0.0174533 * y
#         z = -0.0174533 * z
#         if x != _x:
#             dx = x - _x
#             _x = x
#             x = dx
#             cube.set_avc(x, y, z)
#             print(cube.avc)
#             cube.rotate()
#             cube.set_avc(0, 0, 0)
#         else:
#             _x = x
#
#         if y != _y:
#             dy = y - _y
#             _y = y
#             y = dy
#             cube.set_avc(x, y, z)
#             print(cube.avc)
#             cube.rotate()
#             cube.set_avc(0, 0, 0)
#         else:
#             _y = y
#
#         if z != _z:
#             dz = z - _z
#             _z = z
#             z = dz
#             cube.set_avc(x, y, z)
#             print(cube.avc)
#             cube.rotate()
#             cube.set_avc(0, 0, 0)
#         else:
#             _z = z
#         x, y, z, u, v, w, nu, nv, nw = cube.quiver
#         if graphics['displayQuiver']:
#             graphics['bodyQuiverNegative'].remove()
#             graphics['bodyQuiverPositive'].remove()
#             graphics['bodyQuiverNegative'] = cubeplot.x.quiver(
#                 x, y, z, u, v, w, arrow_length_ratio=0.15, color='blue', lw=0.5, )
#             graphics['bodyQuiverPositive'] = cubeplot.ax.quiver(
#                 x, y, z, nu, nv, nw, arrow_length_ratio=0.15, color='red', lw=0.5, )
#         if graphics['displayAV']:
#             orientation = cube.avc
#             graphics['avText'].set_text("Angular Velocity: {}".format(orientation))
#         plt.pause(0.05)
#     except:
#         pass
#
#
# def update_animation(i, cube, graphics):
#     if cube.atc == cube.atcm:
#         reset_thruster_img(cube.activeThruster)
#     if (cube.activeThruster != None) and (cube.atc == 1):
#         for thruster in thrusterPoints:
#             reset_thruster_img(thruster)
#         fire_thruster_img(cube.activeThruster, cube.thrusters[cube.activeThruster].location.z)
#
#     cube.rotate()
#
#     if graphics['displayWireframe']:
#         for face in cube.faces:
#             x, y, z = cube.wf(face)
#             faces[face].set_data(x, y)
#             faces[face].set_3d_properties(z)
#
#     for thruster in cube.thrusters:
#         x, y, z = cube.thrusters[thruster].location.as_list()
#         thrusterPoints[thruster].set_data(x, y)
#         thrusterPoints[thruster].set_3d_properties(z)
#     x, y, z, u, v, w, nu, nv, nw = cube.quiver
#     if graphics['displayQuiver']:
#         graphics['bodyQuiverNegative'].remove()
#         graphics['bodyQuiverPositive'].remove()
#         graphics['bodyQuiverNegative'] = ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.15, color='blue', lw=0.5)
#         graphics['bodyQuiverPositive'] = ax.quiver(x, y, z, nu, nv, nw, arrow_length_ratio=0.15, color='red', lw=0.5)
#     controller.check_press(cube, graphics)
#
#     if graphics['displayAV']:
#         orientation = cube.avc
#         graphics['avText'].set_text("Angular Velocity: {}".format(orientation))
#     if graphics['displayBody']:
#         poly.set_verts(cube.verts)
#
#
# def run(
#     displayAV=False, displayAxes=True, randomAV=False, thrusterConfig=configurations['1u-1d-z-3'], displayBody=True,
#     displayQuiver=False, displayWireframe=False
# ):
#     global line_ani
#     Cube = CubeSat(randomAV=randomAV, thrusters=thrusterConfig)
#     avText = ax.text(-1, -1, -1, '')
#     graphics = {
#         'ax'                : ax, 'faces': faces, 'poly': poly, 'avText': avText, 'axesNegative': axesNegative,
#         'axesPositive'      : axesPositive, 'displayAxes': displayAxes, 'displayAV': displayAV,
#         'displayBody'       : displayBody, 'displayQuiver': displayQuiver, 'displayWireframe': displayWireframe,
#         'bodyQuiverNegative': bodyQuiverNegative, 'bodyQuiverPositive': bodyQuiverPositive,
#     }
#     fargs = [Cube, graphics]
#     line_ani = FuncAnimation(fig, update_animation, fargs=fargs, interval=1000 / fps)
#     plt.show()

# def _cleanaup():
#     # from cubeplot import ax, axesNegative, axesPositive, bodyQuiverNegative, bodyQuiverPositive, faces, fig, \
#     #     fire_thruster_img, fps, poly, reset_thruster_img, thrusterPoints
#     # from interface import colors, icons, tooltips
#     # import interface.styles.btnstyles
#     # from interface.settingmenus import datasettings, generatorsettings, transceiversettings
#     # from utility import constants, database, decorators, paths, variables
#     # from utility.interface import widgetkwargs
#
#     plt.locator_params(nbins=2)
#     _x, _y = [], []
#     maxForce = 25
#     timeToMaxForce = 50
#     hertz = 0
#     degrees = 10
#     timeVals = []
#     forceVals = []
#     angularVelocityVals = []
#     displacementVals = []
#     positionVals = []
#     directionVals = []
#     place = 55
#     avText = cubeplot.ax.text(-1, -1, -1, '')
#     plt.show()
#     callback = None
#     IP = "129.108.152.70"
#     logLabels = {
#         'IMU Status'   : '  Time            Vx            Vy            Ax            Ay',
#         'Sensor Status': '  Time            Pressure      Temperature',
#         'EPS Status'   : '  Time            Current       Voltage', 'Received': '  Time            Message',
#         'Transmitted'  : '  Time            Message', 'Error': '  Time            Error',
#         'Handshake'    : '  Time            Received',
#     }
#     types = {
#         'IMU Status'   : [str, float, float, float, float], 'EPS Status': [str, float, float],
#         'Sensor Status': [str, float, float], 'Received': [str, str], 'Transmitted': [str, str], 'Error': [str, str],
#         'Handshake'    : [str, int],
#     }
#     sizes2 = {
#         'IMU Status'   : '{},{:03.3f},{:03.3f},{:03.3f},{:03.3f}', 'EPS Status': '{},{:03.3f},{:03.3f}',
#         'Sensor Status': '{},{:03.3f},{:03.3f}', 'Received': '{},{}', 'Transmitted': '{},{}', 'Error': '{},{}',
#         'Handshake'    : '{},{}',
#     }
#     username, clientSocket = client.initialize(ip=configuration.ip, port=1234)
#     cubeSat = serial.Serial('COM5', 115200)
#     ## client.main_loop(username, clientSocket, rCallback=plotter)
#     colors = ['blue', 'orange', 'green']
#     configurations = {
#         '1z4' : [Thruster('rf', [0.5, 0.25, 0]), Thruster('rb', [0.5, -0.25, 0]), Thruster('lf', [-0.5, 0.25, 0]),
#                  Thruster('lb', [-0.5, -0.25, 0])],
#         '1xy4': [Thruster('rf', [0.25, 0.25, -0.5]), Thruster('rb', [0.25, -0.25, -0.5]),
#                  Thruster('lf', [-0.25, 0.25, -0.5]), Thruster('lb', [-0.25, -0.25, -0.5])]
#     }
#     x = threading.Thread(
#         target=client.main_loop, args=(username, clientSocket), kwargs={
#             'rCallback': cubeRCallback,
#         }
#     )
#     line_ani = None
#     controller = Controller()
#     app = QtWidgets.QApplication(sys.argv)
#     loader = QtWebEngineWidgets.QWebEngineView()
#     loader.setZoomFactor(1)
#     loader.page().pdfPrintingFinished.connect(lambda *args: print('finished:', args))
#     loader.loadFinished.connect(emit_pdf)
#     loader.load(
#         QtCore.QUrl.fromLocalFile(r'C:\\Users\\frano\PycharmProjects\AutomaticReportGenerator\HTML\Error Metrics.html')
#     )
#     app.exec()
#     print(str(p3)[0:0])  # bug fix
#     print(str(bodyQuiverPositive)[0:0])  # bug fix
#     print(str(bodyQuiverNegative)[0:0])  # bug fix
#     cubeSat.close()
#     # plt.figure(figsize=(10,10))
#     plt.plot(timeVals, forceVals, label='Thrust Force (N)')
#     plt.plot(timeVals, angularVelocityVals, label='Angular Velocity (degrees per second)')
#     # plt.plot(timeVals, displacementVals, label='Displacement (degrees)')
#     plt.plot(timeVals, positionVals, label='Position (degrees)')
#     plt.title('{} degree rotation'.format(degrees))
#     plt.legend()
#     plt.xlabel('Time (ms)')
#     plt.tick_params(labelright=True)
#     plt.show()
#     run(
#         thrusterConfig=configurations['1u-2d-xy-3']
#     )
#     x.start()
#     main_loop(degrees=degrees)
#     r = R.from_quat([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])

App(
    name=configuration.name,
    root=configuration.root,
).exec()

#############################################
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
