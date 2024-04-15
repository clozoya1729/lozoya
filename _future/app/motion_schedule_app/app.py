import numpy as np

import configuration
import lozoya.gui
import lozoya.math
import lozoya.physics
import forms


def displacement_model(x, initial, final, offset, slope):
    return initial + (final - initial) * (np.tanh(slope * x - offset) + 1) / 2


def velocity_model(x, a, b, c, d):
    return 1


def acceleration_model(x, a0, b0, c0, d0, a1, b1, c1, d1):
    return (a0 / (1 + (1 / (b0 ** 2)) * (x - c0) ** (5 * d0)) -
            (a1 / (1 + (1 / (b1 ** 2)) * (x - c1) ** (5 * d1))))


class MenuAxis(lozoya.gui.TSForm):
    def __init__(self, fields=None, name=None, callback0=None, callback1=None, callback2=None):
        self.formPosition = forms.TSFormPosition(
            name='Position',
            callback=callback0,
        )
        self.formVelocity = forms.TSFormVelocity(
            name='Velocity',
            callback=callback1,
        )
        self.formAcceleration = forms.TSFormAcceleration(
            name='Acceleration',
            callback=callback2,
        )
        widgets = [
            self.formPosition,
            self.formVelocity,
            self.formAcceleration
        ]
        self.controlForm = lozoya.gui.TSListWidgetSwitcher(
            name='Control Form',
            widgets=widgets
        )
        fields = [
            self.controlForm,
        ]
        lozoya.gui.TSForm.__init__(self, fields=fields, name=name)


class App(lozoya.gui.TSApp):
    def __init__(self, name, root, degrees=True, units='°'):
        lozoya.gui.TSApp.__init__(self, name, root)
        if degrees:
            self.v = 180
            self.units = '°'
        else:
            self.v = np.pi
            self.units = 'radians'
        self.positionUnits = f'{units}'
        self.velocityUnits = f'{units}/s'
        self.accelerationUnits = f'{units}/s^2'
        self.axis0 = MenuAxis(
            name='x',
            callback0=self.update_by_position,
            callback1=self.update_by_velocity,
            callback2=self.update_by_acceleration,
        )
        self.axis1 = MenuAxis(
            name='y',
            callback0=self.update_by_position,
            callback1=self.update_by_velocity,
            callback2=self.update_by_acceleration,
        )
        self.axis2 = MenuAxis(
            name='z',
            callback0=self.update_by_position,
            callback1=self.update_by_velocity,
            callback2=self.update_by_acceleration,
        )
        self.plot0 = lozoya.gui.TSPlot(name='')
        self.plot1 = lozoya.gui.TSPlot(name='')
        self.plot2 = lozoya.gui.TSPlot(name='')
        self.timeTotal = lozoya.gui.TSInputSlider(
            callback=self.update_by_position,
            name='time',
            range=(10, 1000),
            value=10,
        )
        self.resolution = lozoya.gui.TSInputSlider(
            callback=self.update_by_position,
            name='resolution',
            range=(10, 1000),
            value=1000,
        )
        fields = [
            self.axis0,
            self.axis1,
            self.axis2,
        ]
        self.axisForms = lozoya.gui.TSForm(
            name='',
            fields=fields,
            horizontal=True,
        )
        fields = [
            self.axisForms,
            self.timeTotal,
            self.resolution,
            self.plot0,
            self.plot1,
            self.plot2,
        ]
        self.form = lozoya.gui.TSForm(
            name='',
            fields=fields,
        )
        self.x = lozoya.physics.TSKinematics()
        self.y = lozoya.physics.TSKinematics()
        self.z = lozoya.physics.TSKinematics()
        self.axes = [self.axis0, self.axis1, self.axis2]
        self.frames = [self.x, self.y, self.z]
        self.window.setCentralWidget(self.form)
        self.update_by_position(0)

    def update_by_position(self, value):
        self.reset_plots()
        x = np.linspace(-self.timeTotal.get_value(), self.timeTotal.get_value(), 2 * self.resolution.get_value())
        for axis, frame in zip(self.axes, self.frames):
            initial = axis.formPosition.initial.get_value()
            final = axis.formPosition.final.get_value()
            offset = axis.formPosition.offset.get_value() / 10000
            slope = axis.formPosition.slope.get_value() / 10000
            center = offset / slope
            resolution = self.resolution.get_value()
            frame.update_by_position(
                resolution,
                displacement_model,
                (x, initial, final, offset, slope),
            )
            # self.plot2.axes.vlines(center, ymin=-self.v, ymax=self.v, linestyle='--', label=f'(t={center})', lw=0.5)
        self.draw_plots(x)

    def update_by_acceleration(self, value):
        self.reset_plots()
        x = np.linspace(-self.timeTotal.get_value(), self.timeTotal.get_value(), 2 * self.resolution.get_value())
        for axis, frame in zip(self.axes, self.frames):
            a0 = axis.formAcceleration.accelerationMax.get_value()
            b0 = axis.formAcceleration.accelerationDuration.get_value() / 1000
            c0 = axis.formAcceleration.accelerationOffset.get_value()
            d0 = lozoya.math.round_up_to_even(axis.formAcceleration.accelerationSlope.get_value())
            a1 = axis.formAcceleration.accelerationMax.get_value()
            b1 = axis.formAcceleration.accelerationDuration.get_value() / 1000
            c1 = axis.formAcceleration.accelerationOffset.get_value() + axis.formAcceleration.accelerationGap.get_value()
            d1 = lozoya.math.round_up_to_even(axis.formAcceleration.accelerationSlope.get_value())
            resolution = self.resolution.get_value()
            frame.update_by_acceleration(
                resolution,
                acceleration_model,
                (x, a0, b0, c0, d0, a1, b1, c1, d1),
            )
        self.draw_plots(x)

    def update_by_velocity(self, velocity, acceleration_model, x, y):
        pass

    def draw_plots(self, x):
        for frame in self.frames:
            positionModulated = np.mod(frame.position + self.v, 2 * self.v) - self.v
            self.plot2.plot(x, positionModulated)
            self.plot1.plot(x, frame.velocity)
            self.plot0.plot(x, frame.acceleration)
        self.plot0.draw()
        self.plot1.draw()
        self.plot2.draw()

    def reset_plots(self):
        self.plot0.axes.cla()
        self.plot1.axes.cla()
        self.plot2.axes.cla()
        self.plot0.axes.set_xticks([])
        self.plot1.axes.set_xticks([])
        self.plot2.axes.set_xlabel('Time (s)')
        self.plot2.axes.set_title('Position')
        self.plot1.axes.set_title('Velocity')
        self.plot0.axes.set_title('Acceleration')
        self.plot0.axes.set_ylabel(f'({self.accelerationUnits})')
        self.plot1.axes.set_ylabel(f'({self.velocityUnits})')
        self.plot2.axes.set_ylabel(f'({self.positionUnits})')
        self.plot2.axes.set_ylim(-self.v, self.v)


App(
    name=configuration.name,
    root=configuration.root
).exec()
