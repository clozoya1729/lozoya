import math

import numpy as np
from hc_util.hc_emf import ElectroMagneticField

class HelmholtzCoils:
    def __init__(self,
                 wireProperties,
                 resolution):
        self.wire = Wire(**wireProperties)
        self.resolution = resolution
        self.x, self.y = self.get_axes(resolution)
        self.emf = ElectroMagneticField(helmholtzCoils=self)
        self.baRatio = 6 / math.sqrt(31)
        self.Na0, self.Nb0 = self.get_winding_arrays()
        self.percentO2 = self.get_percent_O2()

    def get_winding_arrays(self):
        '''
        Define form dimensions for wire to be wound around:
        :param innerPoleRadius: (mm)
        :param formWallWidth: (mm)
        Number of windings in each direction:
        :param Nstart:
        :param Nstop:
        Define wire turn conditions for ideal case:
        :param windingsHorizontal: Number of horizontal windings.
        :param windingsVertical: Number of vertical windings.
        :param rho: (mm)
        :param tau: (mm)
        :param current: (Amps) (max allowed for AWG #20)
        Info on AWG #20 square magnet wire (converted to metric):
        :param wireCoreThickness: (mm)
        :param wireInsulationThickness: (mm)
        :param ohmPerMeter: (ohms/m)
        :param ptsPerAxis:
        :return:
        '''
        Na0 = np.arange(self.wire.Nstart, self.wire.Nstop, 1)
        Nb0 = np.arange(self.wire.Nstart, self.wire.Nstop, 1)
        # set ideal case and empty array for pct changes
        return Na0, Nb0

    def get_axes(self, resolution):
        '''
        Define the resolution in points per axis, area of interest, and axes
        '''
        imgSpan = self.wire.Ravg / 6  # (mm)
        x = np.linspace(
            -imgSpan,
            imgSpan,
            resolution,
            endpoint=True,
            dtype=np.float64,
        )  # (mm)
        y = np.linspace(
            -imgSpan,
            imgSpan,
            resolution,
            endpoint=True,
            dtype=np.float64,
        )  # (mm)
        return x, y

    def get_percent_O2(self):
        percentO2 = np.empty(
            shape=[int(self.Na0.shape[0]), int(self.Nb0.shape[0])],
            dtype=np.float64,
        )
        # Iterate through strength of each [a,b] case
        for ia, aVal in enumerate(self.Na0):
            for ib, bVal in enumerate(self.Nb0):
                percentO2[ib, ia] = 100 * abs((bVal / aVal - self.baRatio) / self.baRatio)
        return percentO2


class Wire:
    def __init__(self,
                 innerPoleRadius,
                 formWallWidth,
                 windingsHorizontal,
                 windingsVertical,
                 coreThickness,
                 insulationThickness,
                 current,
                 ohmPerMeter,
                 rho,
                 tau,
                 Nstart,
                 Nstop):
        self.Nstart = Nstart
        self.Nstop = Nstop
        self.rho = rho
        self.tau = tau
        self.windingsHorizontal = windingsHorizontal
        self.windingsVertical = windingsVertical
        self.coreThickness = coreThickness
        self.insulationThickness = insulationThickness
        # calculate coil cross-section dimensions
        self.thicknessHorizontalTotal = windingsHorizontal * (coreThickness + 2 * insulationThickness)
        self.coreThicknessVerticalTotal = windingsVertical * coreThickness
        self.thicknessVerticalTotal = self.vertical_thickness_effective()
        self.windingsTotal = windingsHorizontal * windingsVertical  # total number of winding turns/coil
        self.Rbase = innerPoleRadius + formWallWidth
        self.Ravg = self.Rbase + self.thicknessVerticalTotal / 2  # midpoint of coil radius
        self.approxWireLengthPerCoil = (2 * math.pi * self.Ravg) * self.windingsTotal / 1E3  # (meters)
        self.ohmsPerCoil = ohmPerMeter * self.approxWireLengthPerCoil
        self.current = current
        self.wattsPerCoil = current ** 2 * self.ohmsPerCoil

    def vertical_thickness_effective(self):
        '''
        Coil's effective vertical dimension due to wire insulation
        Note: ONLY valid for even number of vertical windings, N_b.
        If odd N_b are needed, change code below appropriately.
        '''
        n = self.windingsVertical / 2
        term2 = self.insulationThickness / self.coreThickness * (2 * n - 1) / (n ** 2)
        return self.coreThicknessVerticalTotal * math.sqrt(1 + term2)

    def print_properties(self):
        print('Coil Dimension [a]       =', round(self.thicknessHorizontalTotal, 3), 'mm')
        print('Coil Dimension [b_eff]   =', round(self.thicknessVerticalTotal, 3), 'mm')
        print('Coil Dimension [R_avg]   =', round(self.Ravg, 3), 'mm')
        print('Coil Wire Length         =', round(self.approxWireLengthPerCoil, 3), 'meters')
        print('Total Wire Length        =', round(2 * self.approxWireLengthPerCoil + self.Ravg, 3), 'meters')
        print('Coil Resistance [R]      =', round(self.ohmsPerCoil, 3), 'Ohms')
        print('Coil Heating [power]     =', round(self.wattsPerCoil, 3), 'Watts')
