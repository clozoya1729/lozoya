from collections import OrderedDict

import numpy as np
import wavio

import base

"""
-Project
--Session
---Task
"""


class Metadata(OrderedDict):
    """
    This is a dictionary that contains the
    input and output data to be used for
    training a machine learning model,
    as well as the model that the data
    should be given to.
    It also includes a Utilities14 function to
    provide more interpretable print behavior.
    E.g., if model='Equalizer', then the
    input and output data will be sent to
    the Equalizer Regression6 model for
    training.
    """

    def __init__(self, io, transformation, reference=None):
        _metadata = [
            ('model', transformation.type),
            ('input', io[0]),
            ('output', io[1]),
            ('reference', reference)
        ]
        OrderedDict.__init__(self, _metadata)

    def __repr__(self):
        r = '\t------\n'
        for key in self:
            r += '\t{}: {}\n'.format(key, self[key])
        r += '\t------\n'
        return r


class Reference(base.PathObject):
    def __init__(self, path):
        base.PathObject.__init__(self, path)

    @property
    def data(self):
        return self._data

    def update(self, old, new):
        self._data = np.add(np.subtract(self._data, old), new)

    def activate(self):
        self._ = wavio.read(self.path)
        self._data = self._.data


class Metadata2(OrderedDict):
    """
    This is a dictionary that contains the
    input and output data to be used for
    training a machine learning model,
    as well as the model that the data
    should be given to.
    It also includes a Utilities14 function to
    provide more interpretable print behavior.
    E.g., if model='Equalizer', then the
    input and output data will be sent to
    the Equalizer Regression6 model for
    training.
    """

    def __init__(self, input, output, transformation):
        _metadata = [
            ('model', transformation),
            ('input', input),
            ('output', output),
        ]
        OrderedDict.__init__(self, _metadata)

    def __repr__(self):
        r = '\t------\n'
        for key in self:
            r += '\t{}: {}\n'.format(key, self[key])
        r += '\t------\n'
        return r
