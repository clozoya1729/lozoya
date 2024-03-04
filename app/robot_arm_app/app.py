import numpy as np

import configuration
import lozoya.gui
import lozoya.math


def get_coordinates(lengths, offsets, axes, angles):
    # TODO FORWARD KINEMATICS EQUATIONS
    return np.array([np.array([0, 0, 0]), np.array([100, 100, 100])])


def get_all_coordinates(lengths, offsets, axes, angles):
    vector = np.array([0, 0, 0])
    coordinates = np.array([vector])
    for i in range(len(lengths)):
        length = lengths[i]
        offset = offsets[i]
        lengthVector = lozoya.math.rotate_vector_cumulative(np.array([0, 0, length]), axes, angles, i)
        vector = vector + lengthVector
        coordinates = np.append(coordinates, np.array([vector]), axis=0)
        if (offset):
            offsetVector = lozoya.math.rotate_vector_cumulative(np.array([0, offset, 0]), axes, angles, i)
            vector = vector + offsetVector
            coordinates = np.append(coordinates, np.array([vector]), axis=0)
    return coordinates


def plot_arm(allCoordinates, keyCoordinates, ax):
    """
    coordinates: vector (3 element numpy array)
    """
    ax.set_ylim3d(-200, 200)
    ax.set_xlim3d(-200, 200)
    ax.set_zlim3d(0, 400)
    x = allCoordinates[:, 0]
    y = allCoordinates[:, 1]
    z = allCoordinates[:, 2]
    ax.plot(x, y, z, marker='o')
    for i in range(len(allCoordinates)):
        coordinate = np.round(allCoordinates[i], 2)
        if configuration.labels[i]:
            x = coordinate[0] + 5
            y = coordinate[1]
            z = coordinate[2]
            ax.text(
                x, y, z,
                '%s:\n%s' % (configuration.motorLabels2[i], coordinate),
                size=10,
                zorder=1,
                color='k'
            )
    x = keyCoordinates[:, 0]
    y = keyCoordinates[:, 1]
    z = keyCoordinates[:, 2]
    ax.plot(x, y, z, marker='o')


def run(angles, ax):
    allCoordinates = get_all_coordinates(
        configuration.lengths,
        configuration.offsets,
        configuration.axes2,
        [0, angles[0], 0, angles[1], 0, angles[2]]
    )
    keyCoordinates = get_coordinates(
        configuration.armLinkLengths,
        configuration.armJointoffsetsLength,
        configuration.axes,
        angles
    )
    return plot_arm(allCoordinates, keyCoordinates, ax)


class App(lozoya.gui.TSApp):
    def __init__(self, name, root):
        lozoya.gui.TSApp.__init__(self, name=name, root=root)
        self.motor0 = lozoya.gui.TSInputSlider(
            name='motor0',
            value=45,
            callback=self.update_arm,
            range=(0, 360),
            tooltip='Motor at the base. Rotates about the z-axis.',
        )
        self.motor1 = lozoya.gui.TSInputSlider(
            name='motor1',
            value=0,
            callback=self.update_arm,
            range=(0, 360),
            tooltip='Rotates about y-axis.',
        )
        self.motor2 = lozoya.gui.TSInputSlider(
            name='motor2',
            value=0,
            callback=self.update_arm,
            range=(0, 360),
            tooltip='Rotates about y-axis.',
        )
        self.plotArea = lozoya.gui.TSPlot(
            projection='3d',
        )
        self.fields = [
            self.motor0,
            self.motor1,
            self.motor2,
            self.plotArea,
        ]
        self.form = lozoya.gui.TSForm(
            fields=self.fields,
            name='form'
        )
        self.window.setCentralWidget(self.form)

    def update_arm(self, value):
        angles = [
            self.motor0.get_value(),
            self.motor1.get_value(),
            self.motor2.get_value(),
        ]
        self.plotArea.axes.cla()
        run(angles, self.plotArea.axes)
        self.plotArea.draw()


App(
    name='Robot Arm',
    root='E:\github2\test\robot_arm'
).exec()
