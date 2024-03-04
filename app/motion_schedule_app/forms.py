import random

import lozoya.gui


class TSFormAcceleration(lozoya.gui.TSForm):
    def __init__(self, name=None, fields=None, callback=None):
        self.accelerationMax = lozoya.gui.TSInputSlider(
            name='Acceleration Max',
            callback=callback,
            value=random.randrange(1, 2),
            range=(0, 10),
        )
        self.accelerationDuration = lozoya.gui.TSInputSlider(
            name='Acceleration Duration',
            callback=callback,
            value=random.randrange(1, 1000000),
            range=(2, 10000000),
        )
        self.accelerationGap = lozoya.gui.TSInputSlider(
            name='Acceleration Gap',
            callback=callback,
            value=random.randrange(1, 2),
            range=(0, 10),
        )
        self.accelerationSlope = lozoya.gui.TSInputSlider(
            name='Acceleration Slope',
            callback=callback,
            value=random.randrange(2, 20),
            range=(2, 20),
            step=2,
        )
        self.accelerationOffset = lozoya.gui.TSInputSlider(
            name='Acceleration Offset',
            callback=callback,
            value=random.randrange(1, 2),
            range=(0, 10),
        )
        fields = [
            self.accelerationDuration,
            self.accelerationGap,
            self.accelerationMax,
            self.accelerationOffset,
            self.accelerationSlope,
        ]
        lozoya.gui.TSForm.__init__(self, name=name, fields=fields)


class TSFormPosition(lozoya.gui.TSForm):
    def __init__(self, name=None, fields=None, callback=None):
        self.initial = lozoya.gui.TSInputSlider(
            name='Initial Position',
            callback=callback,
            range=(-180, 180),
            value=random.randrange(-180, 10),
        )
        self.final = lozoya.gui.TSInputSlider(
            name='Final Position',
            callback=callback,
            range=(-180, 180),
            value=random.randrange(10, 180),
        )
        self.slope = lozoya.gui.TSInputSlider(
            name='Slope',
            callback=callback,
            value=random.randrange(-100, 100),
            range=(-10000, 10000),
        )
        self.offset = lozoya.gui.TSInputSlider(
            name='Offset',
            callback=callback,
            value=random.randrange(0, 100),
            range=(0, 100000)
        )
        fields = [
            self.initial,
            self.final,
            self.offset,
            self.slope,
        ]
        lozoya.gui.TSForm.__init__(self, name=name, fields=fields)


class TSFormVelocity(lozoya.gui.TSForm):
    def __init__(self, name=None, fields=None, callback=None):
        lozoya.gui.TSForm.__init__(self, name=name, fields=fields)
        self.initial = lozoya.gui.TSInputSlider(
            name='Initial Position',
            callback=callback,
            range=(-180, 180),
            value=random.randrange(-180, 10),
        )
        self.final = lozoya.gui.TSInputSlider(
            name='Final Position',
            callback=callback,
            range=(-180, 180),
            value=random.randrange(10, 180),
        )
        self.slope = lozoya.gui.TSInputSlider(
            name='Slope',
            callback=callback,
            value=random.randrange(-100, 100),
            range=(-10000, 10000),
        )
        self.offset = lozoya.gui.TSInputSlider(
            name='Offset',
            callback=callback,
            value=random.randrange(0, 100),
            range=(0, 100000)
        )
