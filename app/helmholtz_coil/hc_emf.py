import math

import numpy as np

'''
Define magnetic field components to forth order.
F_ex, F_1x, F_2x, F_3x, F_1y, F_2y, F_3y
'''


def F_ex(wire):
    term1 = (18 * wire.thicknessHorizontalTotal ** 4 + 13 * wire.thicknessVerticalTotal ** 4) / (1250 * wire.Ravg ** 4)
    term2 = 31 * wire.thicknessHorizontalTotal ** 2 * wire.thicknessVerticalTotal ** 2 / (750 * wire.Ravg ** 4)
    term3 = (wire.tau + wire.rho) / wire.Ravg * (
            1 / 5 + 2 * (wire.thicknessHorizontalTotal ** 2 - wire.thicknessVerticalTotal ** 2) / (
            250 * wire.Ravg ** 2))
    term4 = (wire.tau ** 2 + wire.rho ** 2) / (250 * wire.Ravg ** 2) * (
            25 + (52 * wire.thicknessVerticalTotal ** 2 - 62 * wire.thicknessHorizontalTotal ** 2) / (wire.Ravg ** 2))
    term5 = 8 * (wire.tau ** 3 + wire.rho ** 3) / (25 * wire.Ravg ** 3)
    term6 = 52 * (wire.tau ** 4 + wire.rho ** 4) / (125 * wire.Ravg ** 4)
    return -term1 + term2 + term3 - term4 - term5 - term6


def F_1x(wire):
    term1 = (wire.tau - wire.rho) / wire.Ravg * (
            150 + (24 * wire.thicknessVerticalTotal ** 2 - 44 * wire.thicknessHorizontalTotal ** 2) / (wire.Ravg ** 2))
    term2 = 165 * (wire.tau ** 2 - wire.rho ** 2) / (wire.Ravg ** 2)
    term3 = 96 * (wire.tau ** 3 - wire.rho ** 3) / (wire.Ravg ** 3)
    return term1 + term2 + term3


def F_2x(wire):
    term1 = (31 * wire.thicknessVerticalTotal ** 2 - 36 * wire.thicknessHorizontalTotal ** 2) / (wire.Ravg ** 2)
    term2 = 60 * (wire.tau + wire.rho) / wire.Ravg
    term3 = 186 * (wire.tau ** 2 + wire.rho ** 2) / (wire.Ravg ** 2)
    return term1 + term2 + term3


def F_3x(wire):
    return 88 * (wire.tau - wire.rho) / wire.Ravg


def F_1y(wire):
    term1 = (wire.rho - wire.tau) / wire.Ravg * (
            75 + (12 * wire.thicknessVerticalTotal ** 2 - 22 * wire.thicknessHorizontalTotal ** 2) / (wire.Ravg ** 2))
    term2 = 165 * (wire.rho ** 2 - wire.tau ** 2) / (2 * wire.Ravg ** 2)
    term3 = 48 * (wire.rho ** 3 - wire.tau ** 3) / (wire.Ravg ** 3)
    return term1 + term2 + term3


def F_2y(wire):
    term1 = 2 * (36 * wire.thicknessHorizontalTotal ** 2 - 31 * wire.thicknessVerticalTotal ** 2) / (wire.Ravg ** 2)
    term2 = 120 * (wire.tau - wire.rho) / wire.Ravg
    term3 = 372 * (wire.tau ** 2 + wire.rho ** 2) / (wire.Ravg ** 2)
    return term1 - term2 - term3


def F_3y(wire):
    return 66 * (wire.tau - wire.rho) / wire.Ravg


class ElectroMagneticField:
    def __init__(self,
                 helmholtzCoils):
        self.helmholtzCoils = helmholtzCoils
        self.emfMapX, self.emfMapY = self.get_emf_maps()
        self.emfVectorMap = np.arctan2(self.emfMapY, self.emfMapX) * 180 / np.pi
        self.emfMagnitudeMap = np.sqrt(np.square(self.emfMapX) + np.square(self.emfMapY))
        self.center = (
            np.sqrt(
                np.square(self.Bx(0, 0)) + np.square(self.By(0, 0))
            )
        )
        self.deviation = self.get_deviation()

    def get_deviation(self):
        '''
        Calculate deviation from center field, B(0,0), in micro-Tesla
        '''
        return (self.emfMagnitudeMap - self.center) * 1E3

    def print_properties(self):
        # check center and average field specs, along with deviation over total area
        print('B(0,0)                   =', round(self.center, 3), 'mT')
        print('<Bx>                     =', round(np.average(self.emfMapX), 3), u'\xb1', round(np.std(self.emfMapX), 3), 'mT')
        print('<By>                     =', round(np.average(self.emfMapY), 3), u'\xb1', round(np.std(self.emfMapY), 3), 'mT')
        print('<B0>                     =', round(np.average(self.emfMagnitudeMap), 3), u'\xb1',
              round(np.std(self.emfMagnitudeMap), 3), 'mT')
        print('<theta>                  =', round(np.average(self.emfVectorMap), 3), u'\xb1',
              round(np.std(self.emfVectorMap), 3),
              'deg')

    def Bx(self, x, y):
        spacePermeability = 1.2566370614  # in units of mT*mm/A
        prefactor_0 = 8 * spacePermeability * self.helmholtzCoils.wire.windingsTotal * self.helmholtzCoils.wire.current / (
                5 * math.sqrt(5) * self.helmholtzCoils.wire.Ravg)
        term2 = self.helmholtzCoils.wire.thicknessVerticalTotal ** 2 / (60 * self.helmholtzCoils.wire.Ravg ** 2)
        prefactor_5 = (2 * x ** 2 - y ** 2) / (125 * self.helmholtzCoils.wire.Ravg ** 2)
        prefactor_6 = (3 * x * y ** 2 - 2 * x ** 3) / (125 * self.helmholtzCoils.wire.Ravg ** 3)
        term7 = 18 / (125 * self.helmholtzCoils.wire.Ravg ** 4) * (8 * x ** 4 - 24 * x ** 2 * y ** 2 + 3 * y ** 4)
        return prefactor_0 * (
                1 - term2 + F_ex(self.helmholtzCoils.wire)
                + x / (125 * self.helmholtzCoils.wire.Ravg) * F_1x(self.helmholtzCoils.wire)
                + prefactor_5 * F_2x(self.helmholtzCoils.wire)
                + prefactor_6 * F_3x(self.helmholtzCoils.wire) - term7)

    def By(self, x, y):
        spacePermeability = 1.2566370614  # in units of mT*mm/A
        prefactor_0 = 8 * spacePermeability * self.helmholtzCoils.wire.windingsTotal * self.helmholtzCoils.wire.current / (
                5 * math.sqrt(5) * self.helmholtzCoils.wire.Ravg)
        prefactor_3 = y * (4 * x ** 2 - y ** 2) / (125 * self.helmholtzCoils.wire.Ravg ** 3)
        term4 = x * y / (125 * self.helmholtzCoils.wire.Ravg ** 4) * (288 * x ** 2 - 216 * y ** 2)
        return prefactor_0 * (
                y / (125 * self.helmholtzCoils.wire.Ravg) * F_1y(self.helmholtzCoils.wire)
                + x * y / (125 * self.helmholtzCoils.wire.Ravg ** 2) * F_2y(self.helmholtzCoils.wire)
                + prefactor_3 * F_3y(self.helmholtzCoils.wire) + term4)

    def get_emf_maps(self):
        # Create empty arrays to store data in.
        self.emfMapX = np.empty(
            shape=[self.helmholtzCoils.x.size, self.helmholtzCoils.y.size],
            dtype=np.float64,
        )
        self.emfMapY = np.empty(
            shape=[self.helmholtzCoils.x.size, self.helmholtzCoils.y.size],
            dtype=np.float64,
        )
        # Iterate over all (x,y) positions to calculate Bx and By using O(4) equations.
        for ix, xval in enumerate(self.helmholtzCoils.x):
            for iy, yval in enumerate(self.helmholtzCoils.y):
                self.emfMapX[ix, iy] = self.Bx(xval, yval)
                self.emfMapY[ix, iy] = self.By(xval, yval)
        return self.emfMapX, self.emfMapY
