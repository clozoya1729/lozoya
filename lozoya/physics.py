import numpy as np
import sympy
import sympy.abc
import lozoya.math


# fluid dynamics
def cp(density, pressure, pressureE, velocity):
    return (pressure - pressureE) / (0.5 * density * velocity ** 2)


def cylinder_stream_function(U, R, centroid):
    x = sympy.abc.x
    y = sympy.abc.y
    r = sympy.sqrt(((x - centroid[0]) ** 2) + ((y - centroid[1]) ** 2))
    theta = sympy.atan2((y - centroid[1]), (x - centroid[0]))
    return U * (r - (R ** 2) / r) * sympy.sin(theta)


def velocity_field(psi):
    x = sympy.abc.x
    y = sympy.abc.y
    u = sympy.lambdify((x, y), psi.diff(y), 'numpy')
    v = sympy.lambdify((x, y), -psi.diff(x), 'numpy')
    return u, v


def get_lims(radii, centroids, xmin, xmax, ymin, ymax):
    i = 0
    for radius, centroid in zip(radii, centroids):
        if i == 0:
            xmin = xmax = centroid[0]
            ymin = ymax = centroid[1]
        if centroid[0] >= xmax:
            xmax = centroid[0] + radius * 10
        if centroid[0] <= xmin:
            xmin = centroid[0] - radius * 10
        if centroid[1] >= ymax:
            ymax = centroid[1] + radius * 5
        if centroid[1] <= ymin:
            ymin = centroid[1] - radius * 5
        i += 1
    return (xmin, xmax), (ymin, ymax)


# kinetics and kinematics
class TSKinematics:
    def __init__(
        self, initialPosition: float = 0, initialVelocity: float = 0, initialAcceleration: float = 0
    ):
        self.initialPosition = initialPosition
        self.initialVelocity = initialVelocity
        self.initialAcceleration = initialAcceleration
        self.position = initialPosition
        self.velocity = initialVelocity
        self.acceleration = initialAcceleration

    def update_by_position(self, resolution, model, modelArgs):
        self.position = model(*modelArgs)
        self.velocity = np.diff(self.position) * resolution
        self.velocity = np.append(self.velocity, self.velocity[-1])
        self.acceleration = np.diff(self.velocity) * resolution
        self.acceleration = np.append(self.acceleration, self.acceleration[-1])

    def update_by_velocity(self, resolution, model, modelArgs):
        self.position = model(*modelArgs)
        self.velocity = np.diff(self.position) * resolution
        self.velocity = np.append(self.velocity, self.velocity[-1])
        self.acceleration = np.diff(self.velocity) * resolution
        self.acceleration = np.append(self.acceleration, self.acceleration[-1])

    def update_by_acceleration(self, resolution, model, modelArgs):
        self.acceleration = model(*modelArgs)
        x = modelArgs[0]
        self.velocity = self.initialVelocity + lozoya.math.integrate(x, self.acceleration)
        self.position = self.initialPosition + lozoya.math.integrate(x, self.velocity)


class Body:
    def __init__(self, app, name, mass=1, dimension=1, centroid=(0, 0, 0), randomAV=False):
        self.app = app
        self.name = name
        self.mass = mass
        self.centroid = centroid
        self.dimension = dimension
        self._angularVelocity = lozoya.math.Quaternion(0, 0, 0, 0)
        r = self.dimension / self.app.plotConfig.scale
        self.quiverP = [[r, 0, 0], [0, r, 0], [0, 0, r], ]
        self.quiverN = [[-r, 0, 0], [0, -r, 0], [0, 0, -r], ]
        if randomAV:
            self._angularVelocity = lozoya.math.Quaternion(0, 'r', 'r', 'r', )

    @property
    def inertia(self, *args, **kwargs):
        return (1 / 6) * self.mass * (self.dimension ** 2)

    @property
    def avc(self, *args, **kwargs):
        """
        Angular Velocity Components
        return: list of float - [x, y, z]
        """
        return self._angularVelocity.x, self._angularVelocity.y, self._angularVelocity.z

    def set_avc(self, x, y, z):
        self._angularVelocity.x = x
        self._angularVelocity.y = y
        self._angularVelocity.z = z

    @property
    def quiverVector(self, *args, **kwargs):
        return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], self.quiverP[0], self.quiverP[1], self.quiverP[2], ])

    def rotate(self):
        ox, oy, oz = self.centroid
        aX, aY, aZ = self.avc
        for name, prop in [('_vertices', self._vertices)]:
            results = {}
            for k in prop:
                c = prop[k]
                px, py, pz = c.as_list()
                qx, qy = lozoya.math.rotate_point(aZ, ox, oy, px, py)  # rotate about z-axis
                qy, qz = lozoya.math.rotate_point(aX, oy, oz, qy, pz)  # rotate about x-axis
                qx, qz = lozoya.math.rotate_point(aY, ox, oz, qx, qz)  # rotate about y-axis
                results[k] = lozoya.math.Coordinate(qx, qy, qz)
            setattr(self, name, results)
        for name, prop in [('thrusters', self.thrusters)]:
            for k in prop:
                c = prop[k].location
                px, py, pz = c.as_list()
                qx, qy = lozoya.math.rotate_point(aZ, ox, oy, px, py)  # rotate about z-axis
                qy, qz = lozoya.math.rotate_point(aX, oy, oz, qy, pz)  # rotate about x-axis
                qx, qz = lozoya.math.rotate_point(aY, ox, oz, qx, qz)  # rotate about y-axis
                prop[k].location = lozoya.math.Coordinate(qx, qy, qz)

        for i, k in enumerate(self.quiverP):
            px, py, pz = k
            qx, qy = lozoya.math.rotate_point(-aZ, ox, oy, px, py)  # rotate about z-axis
            qy, qz = lozoya.math.rotate_point(-aX, oy, oz, qy, pz)  # rotate about x-axis
            qx, qz = lozoya.math.rotate_point(-aY, ox, oz, qx, qz)  # rotate about y-axis
            self.quiverP[i] = [qx, qy, qz]
        if self.activeThruster != None:
            self.activeThrusterCounter += 1
            if self.activeThrusterCounter > self.activeThrusterCounterMaximum:
                self.activeThrusterCounter = 0
                self.activeThruster = None
