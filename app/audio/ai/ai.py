import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor

regressorSettings = {
    'hidden_layer_sizes':  (50, 50), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001,
    'batch_size':          'auto', 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'power_t': 0.5,
    'max_iter':            200, 'shuffle': True, 'random_state': None, 'tol': 0.0001, 'verbose': False,
    'warm_start':          False, 'momentum': 0.9, 'nesterovs_momentum': True, 'early_stopping': False,
    'validation_fraction': 0.1, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08,
    # 'n_iter_no_change': 10
}


def time_regressor():
    return MLPRegressor(**regressorSettings)


def frequency_regressor():
    return MLPRegressor()


def time_classifier():
    return MLPClassifier()


def frequency_classifier():
    return MLPClassifier()


TRANSFORMATIONS2 = {}
for key in TRANSFORMATIONS:
    for value in TRANSFORMATIONS[key]:
        TRANSFORMATIONS2[value] = key

MODELS = {model: time_regressor() for model in models}

_ = models[:-1]
_.append('Stop')
LABELS = {model: np.array([0 if m != model else 1 for m in _]) for model in _}


def get_label(vector):
    for key in LABELS:
        if np.array_equal(LABELS[key], vector):
            return key


import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from configuration import regressorSettings


def time_regressor():
    return MLPRegressor(**regressorSettings)


def frequency_regressor():
    return MLPRegressor()


def time_classifier():
    return MLPClassifier()


def frequency_classifier():
    return MLPClassifier()


TRANSFORMATIONS2 = {}
for key in TRANSFORMATIONS:
    for value in TRANSFORMATIONS[key]:
        TRANSFORMATIONS2[value] = key

MODELS = {model: time_regressor() for model in models}

_ = models[:-1]
_.append('Stop')
LABELS = {model: np.array([0 if m != model else 1 for m in _]) for model in _}


def get_label(vector):
    for key in LABELS:
        if np.array_equal(LABELS[key], vector):
            return key


import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor

regressorSettings = {
    'hidden_layer_sizes':  (4,), 'activation': 'relu', 'solver': 'adam', 'alpha': 1,
    'batch_size':          'auto', 'learning_rate': 'constant', 'learning_rate_init': 1, 'power_t': 0.5,
    'max_iter':            1, 'shuffle': True, 'random_state': None, 'tol': 1, 'verbose': False,
    'warm_start':          False, 'momentum': 0.9, 'nesterovs_momentum': True, 'early_stopping': False,
    'validation_fraction': 0.1, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08,
    # 'n_iter_no_change': 10
}


def time_regressor():
    return MLPRegressor(**regressorSettings)


def frequency_regressor():
    return MLPRegressor()


def time_classifier():
    return MLPClassifier()


def frequency_classifier():
    return MLPClassifier()


TRANSFORMATIONS2 = {}
for key in TRANSFORMATIONS:
    for value in TRANSFORMATIONS[key]:
        TRANSFORMATIONS2[value] = key

MODELS = {model: time_regressor() for model in models}

_ = models[:-1]
_.append('Stop')
LABELS = {model: np.array([0 if m != model else 1 for m in _]) for model in _}


def get_label(vector):
    for key in LABELS:
        if np.array_equal(LABELS[key], vector):
            return key


import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor

regressorSettings = {
    'hidden_layer_sizes':  (1000, 500), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001,
    'batch_size':          'auto', 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'power_t': 0.5,
    'max_iter':            200, 'shuffle': True, 'random_state': None, 'tol': 0.0001, 'verbose': False,
    'warm_start':          False, 'momentum': 0.9, 'nesterovs_momentum': True, 'early_stopping': False,
    'validation_fraction': 0.1, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08,
    # 'n_iter_no_change': 10
}


def time_regressor():
    return MLPRegressor(**regressorSettings)


def frequency_regressor():
    return MLPRegressor()


def time_classifier():
    return MLPClassifier()


def frequency_classifier():
    return MLPClassifier()


TRANSFORMATIONS2 = {}
for key in TRANSFORMATIONS:
    for value in TRANSFORMATIONS[key]:
        TRANSFORMATIONS2[value] = key

MODELS = {model: time_regressor() for model in models}

_ = models[:-1]
_.append('Stop')
LABELS = {model: np.array([0 if m != model else 1 for m in _]) for model in _}


def get_label(vector):
    for key in LABELS:
        if np.array_equal(LABELS[key], vector):
            return key


import os
import sys

import numpy as np
import plotly.graph_objs as go

# AI LIBRARY - sklearn or keras
aiLib = 'keras'

# ENVIRONMENT - development or production
environment = 'development'

# DIRECTORY SETTINGS
projectsDir = {'development': 'testing', 'production': 'projects', }
trainingDir = 'training'
modelsDir = {'keras': 'keras_models', 'sklearn': 'sklearn_models', }
modelExt = {'keras': 'h5', 'sklearn': 'joblib', }
transformationsDir = 'Transformations'
reference = 'Master 1'
drive = '{}{}'.format(os.path.dirname(os.path.dirname(sys.argv[0])), os.sep)
ffmpegPath = os.path.join(drive, 'ffmpeg', 'bin')
excludedDirs = [transformationsDir]
excludedFiles = [reference]

# AUDIO SETTINGS
audioformat = 'wav'
nchannels = 2
sWidth = 2
codec = 'pcm_s{}le'.format(8 * sWidth)
maxVal = (2 ** (8 * sWidth - 1)) - 1
sRate = 44100.0  # number of samples in a second
pSecs = 1  # number of seconds to pad
pRate = 10  # number of samples per second to pad
period = 1.0 / sRate
bSamples = 100  # samples in buffer
bSecs = bSamples / sRate
excludeformats = ['html']

# DQN SETTINGS
episodes = 1  # int(1e6)
steps = 1  # int(1e4)
memSize = 5000
memBatch = 64
nStates = 1000
nActions = 9
penalty = 10000
learningRate = 0.001
forgetThreshold = 0.95
uncertaintyMin = 0.01
uncertaintyDecay = 0.999
gamma = 0.95
minReward = 100

# AI SETTINGS
scale = False
multiModel = False
useFFT = False
useFreq = True
permutations = 3
epochs = 10
addNoise = True
batchSize = 32
beginInputNoise = 3
beginRefNoise = 5
inputSize = 2 * int(4 * pRate * pSecs + 2)
if useFFT:
    inputSize = int(inputSize + 4)
if useFreq:
    inputSize = int(inputSize + bSamples)
eP = 1
cP = 100

print(
    '======== GLOBAL SETTINGS ========\n'
    ' o aiLib: {}\n'
    ' o scale: {}\n'
    '  o maxVal: {}\n'
    ' o multiModel: {}\n'
    ' o permutations: {}\n'
    '  o epochs: {}\n'
    '   o addNoise: {}\n'
    '    o beginInputNoise: {}\n'
    '    o beginRefNoise: {}\n'
    ' o batchSize: {}\n'
    ' o useFFT: {}\n'
    ' o useFreq: {}\n'
    '  o buffer samples: {}\n'
    ' o inputSize: {}\n'
    '================================='.format(
        aiLib, scale, maxVal, multiModel, permutations, epochs, addNoise,
        beginInputNoise, beginRefNoise, batchSize, useFFT, useFFT, bSamples,
        inputSize, )
)
transfer2 = {}
for key in transfer:
    for value in transfer[key]:
        transfer2[value] = key

_ = models[:-1]
_.append('Stop')
labels = {model: np.array([0 if m != model else 1 for m in _]) for model in _}

plotConfig = {
    'auto_open':        False, 'show_link': False,
    'configuration.py': {'displaylogo': False, 'modeBarButtonsToRemove': ['sendDataToCloud'], }
}
lineWidth = 1.5
refPlotSettings = {'name': 'Reference', 'line': {'color': 'rgba(181, 244, 142, 1)', 'width': lineWidth, }}
inputPlotSettings = {'name': 'Input', 'line': {'color': 'rgba(243,82,141,0.5)', 'width': lineWidth, }}
outputPlotSettings = {'name': 'Output', 'line': {'color': 'rgba(141,82,243,0.5)', 'width': lineWidth, }}

import os
import sys

import numpy as np
import plotly.graph_objs as go

# AI LIBRARY - sklearn or keras
aiLib = 'keras'

# ENVIRONMENT - development or production
environment = 'development'

# DIRECTORY SETTINGS
projectsDir = {'development': 'testing', 'production': 'projects', }
trainingDir = 'training'
modelsDir = {'keras': 'keras_models', 'sklearn': 'sklearn_models', }
modelExt = {'keras': 'h5', 'sklearn': 'joblib', }
transformationsDir = 'Transformations'
reference = 'Master 1'
drive = '{}{}'.format(os.path.dirname(os.path.dirname(sys.argv[0])), os.sep)
ffmpegPath = os.path.join(drive, 'ffmpeg', 'bin')
excludedDirs = [transformationsDir]
excludedFiles = [reference]

# AUDIO SETTINGS
audioformat = 'wav'
nChannels = 2
sWidth = 2
codec = 'pcm_s{}le'.format(8 * sWidth)
maxVal = (2 ** (8 * sWidth - 1)) - 1
sRate = 44100.0  # number of samples in a second
pSecs = 1  # number of seconds to pad
pRate = 10  # number of samples per second to pad
period = 1.0 / sRate
bSamples = 100  # samples in buffer
bSecs = bSamples / sRate
excludeformats = ['html']

# DQN SETTINGS
episodes = 1  # int(1e6)
steps = 1  # int(1e4)
memSize = 5000
memBatch = 64
nStates = 1000
nActions = 9
penalty = 10000
learningRate = 0.001
forgetThreshold = 0.95
uncertaintyMin = 0.01
uncertaintyDecay = 0.999
gamma = 0.95
minReward = 100

# AI SETTINGS
scale = False
multiModel = False
useFFT = False
useFreq = True
permutations = 3
epochs = 10
addNoise = True
batchSize = 32
beginInputNoise = 3
beginRefNoise = 5
inputSize = 2 * int(4 * pRate * pSecs + 2)
if useFFT:
    inputSize = int(inputSize + 4)
if useFreq:
    inputSize = int(inputSize + bSamples)
eP = 1
cP = 100

print(
    '======== GLOBAL SETTINGS ========\n'
    ' o aiLib: {}\n'
    ' o scale: {}\n'
    '  o maxVal: {}\n'
    ' o multiModel: {}\n'
    ' o permutations: {}\n'
    '  o epochs: {}\n'
    '   o addNoise: {}\n'
    '    o beginInputNoise: {}\n'
    '    o beginRefNoise: {}\n'
    ' o batchSize: {}\n'
    ' o useFFT: {}\n'
    ' o useFreq: {}\n'
    '  o buffer samples: {}\n'
    ' o inputSize: {}\n'
    '================================='.format(
        aiLib, scale, maxVal, multiModel, permutations, epochs, addNoise,
        beginInputNoise, beginRefNoise, batchSize, useFFT, useFFT, bSamples,
        inputSize, )
)
transfer2 = {}
for key in transfer:
    for value in transfer[key]:
        transfer2[value] = key

_ = models[:-1]
_.append('Stop')
labels = {model: np.array([0 if m != model else 1 for m in _]) for model in _}

plotConfig = {
    'auto_open':        False, 'show_link': False,
    'configuration.py': {'displaylogo': False, 'modeBarButtonsToRemove': ['sendDataToCloud'], }
}
lineWidth = 1.5
refPlotSettings = {'name': 'Reference', 'line': {'color': 'rgba(181, 244, 142, 1)', 'width': lineWidth, }}
inputPlotSettings = {'name': 'Input', 'line': {'color': 'rgba(243,82,141,0.5)', 'width': lineWidth, }}
outputPlotSettings = {'name': 'Output', 'line': {'color': 'rgba(141,82,243,0.5)', 'width': lineWidth, }}

import os
import sys

import numpy as np
import plotly.graph_objs as go

# AI LIBRARY - sklearn or keras
aiLib = 'keras'

# ENVIRONMENT - development or production
environment = 'development'

# DIRECTORY SETTINGS
projectsDir = {'development': 'testing', 'production': 'projects', }
trainingDir = 'training'
modelsDir = {'keras': 'keras_models', 'sklearn': 'sklearn_models', }
modelExt = {'keras': 'h5', 'sklearn': 'joblib', }
transformationsDir = 'Transformations'
reference = 'Master 1'
drive = '{}{}'.format(os.path.dirname(os.path.dirname(sys.argv[0])), os.sep)
ffmpegPath = os.path.join(drive, 'ffmpeg', 'bin')
excludedDirs = [transformationsDir]
excludedFiles = [reference]

# AUDIO SETTINGS
audioformat = 'wav'
nchannels = 2
sWidth = 2
codec = 'pcm_s{}le'.format(8 * sWidth)
maxVal = (2 ** (8 * sWidth - 1)) - 1
sRate = 44100.0  # number of samples in a second
pSecs = 0  # number of seconds to pad
pRate = 1  # number of samples per second to pad
period = 1.0 / sRate
bSamples = 200  # samples in buffer
bSecs = bSamples / sRate
excludeformats = ['html']

# DQN SETTINGS
episodes = int(1e6)
steps = int(1e4)
memSize = 5000
memBatch = 64
nStates = 1000
nActions = 9
penalty = 10000
learningRate = 0.001
forgetThreshold = 0.95
uncertaintyMin = 0.01
uncertaintyDecay = 0.999
gamma = 0.95
minReward = 100

# AI SETTINGS
scale = False
multiModel = False
useFFT = False
useFreq = False
permutations = 1
epochs = 1
addNoise = False
batchSize = 32
beginInputNoise = 3
beginRefNoise = 5
inputSize = 2 * int(4 * pRate * pSecs + 2)
if useFFT:
    inputSize = int(inputSize + 4)
if useFreq:
    inputSize = int(inputSize + bSamples)
eP = 1
cP = 100

print(
    '======== GLOBAL SETTINGS ========\n'
    ' o aiLib: {}\n'
    ' o scale: {}\n'
    '  o maxVal: {}\n'
    ' o multiModel: {}\n'
    ' o permutations: {}\n'
    '  o epochs: {}\n'
    '   o addNoise: {}\n'
    '    o beginInputNoise: {}\n'
    '    o beginRefNoise: {}\n'
    ' o batchSize: {}\n'
    ' o useFFT: {}\n'
    ' o useFreq: {}\n'
    '  o buffer samples: {}\n'
    ' o inputSize: {}\n'
    '================================='.format(
        aiLib, scale, maxVal, multiModel, permutations, epochs, addNoise,
        beginInputNoise, beginRefNoise, batchSize, useFFT, useFFT, bSamples,
        inputSize, )
)
transfer2 = {}
for key in transfer:
    for value in transfer[key]:
        transfer2[value] = key

_ = models[:-1]
_.append('Stop')
labels = {model: np.array([0 if m != model else 1 for m in _]) for model in _}

plotConfig = {
    'auto_open':        False, 'show_link': False,
    'configuration.py': {'displaylogo': False, 'modeBarButtonsToRemove': ['sendDataToCloud'], }
}
lineWidth = 1.5
refPlotSettings = {'name': 'Reference', 'line': {'color': 'rgba(181, 244, 142, 1)', 'width': lineWidth, }}
inputPlotSettings = {'name': 'Input', 'line': {'color': 'rgba(243,82,141,0.5)', 'width': lineWidth, }}
outputPlotSettings = {'name': 'Output', 'line': {'color': 'rgba(141,82,243,0.5)', 'width': lineWidth, }}

import os
import sys

import numpy as np
import plotly.graph_objs as go

# AI LIBRARY - sklearn or keras
aiLib = 'keras'

# ENVIRONMENT - development or production
environment = 'development'

# DIRECTORY SETTINGS
projectsDir = {'development': 'testing', 'production': 'projects', }
trainingDir = 'training'
modelsDir = {'keras': 'keras_models', 'sklearn': 'sklearn_models', }
modelExt = {'keras': 'h5', 'sklearn': 'joblib', }
transformationsDir = 'Transformations'
reference = 'Master 1'
drive = '{}{}'.format(os.path.dirname(os.path.dirname(sys.argv[0])), os.sep)
ffmpegPath = os.path.join(drive, 'ffmpeg', 'bin')
excludedDirs = [transformationsDir]
excludedFiles = [reference]

# AUDIO SETTINGS
audioformat = 'wav'
nchannels = 2
sWidth = 2
codec = 'pcm_s{}le'.format(8 * sWidth)
maxVal = (2 ** (8 * sWidth - 1)) - 1
sRate = 44100.0  # number of samples in a second
pSecs = 1  # number of seconds to pad
pRate = 441  # number of samples per second to pad
period = 1.0 / sRate
bSamples = 200  # samples in buffer
bSecs = bSamples / sRate
excludeformats = ['html']

# DQN SETTINGS
episodes = int(1e6)
steps = int(1e4)
memSize = 5000
memBatch = 64
nStates = 10000
nActions = 9
penalty = 10000
learningRate = 0.001
forgetThreshold = 0.95
uncertaintyMin = 0.01
uncertaintyDecay = 0.999
gamma = 0.95

# AI SETTINGS
scale = True
multiModel = False
useFFT = True
useFreq = True
permutations = 1
epochs = 1
addNoise = True
batchSize = 32
beginInputNoise = 3
beginRefNoise = 5
inputSize = int(16 * pRate * pSecs + 4)
if useFFT:
    inputSize = int(inputSize + 4)
if useFreq:
    inputSize = int(inputSize + bSamples)
eP = 1
cP = 100

print(
    '======== GLOBAL SETTINGS ========\n'
    ' o aiLib: {}\n'
    ' o scale: {}\n'
    '  o maxVal: {}\n'
    ' o multiModel: {}\n'
    ' o permutations: {}\n'
    '  o epochs: {}\n'
    '   o addNoise: {}\n'
    '    o beginInputNoise: {}\n'
    '    o beginRefNoise: {}\n'
    ' o batchSize: {}\n'
    ' o useFFT: {}\n'
    ' o useFreq: {}\n'
    '  o buffer samples: {}\n'
    ' o inputSize: {}\n'
    '================================='.format(
        aiLib, scale, maxVal, multiModel, permutations, epochs, addNoise,
        beginInputNoise, beginRefNoise, batchSize, useFFT, useFFT, bSamples,
        inputSize, )
)
transfer2 = {}
for key in transfer:
    for value in transfer[key]:
        transfer2[value] = key

_ = models[:-1]
_.append('Stop')
labels = {model: np.array([0 if m != model else 1 for m in _]) for model in _}

plotConfig = {
    'auto_open':        False, 'show_link': False,
    'configuration.py': {'displaylogo': False, 'modeBarButtonsToRemove': ['sendDataToCloud'], }
}
lineWidth = 1.5
refPlotSettings = {'name': 'Reference', 'line': {'color': 'rgba(181, 244, 142, 1)', 'width': lineWidth, }}
inputPlotSettings = {'name': 'Input', 'line': {'color': 'rgba(243,82,141,0.5)', 'width': lineWidth, }}
outputPlotSettings = {'name': 'Output', 'line': {'color': 'rgba(141,82,243,0.5)', 'width': lineWidth, }}

import os
import sys

import numpy as np
import plotly.graph_objs as go

# AI LIBRARY - sklearn or keras
aiLib = 'keras'
# ENVIRONMENT - development or production
environment = 'development'
# DIRECTORY SETTINGS
projectsDir = {'development': 'testing', 'production': 'projects', }
trainingDir = 'training'
modelsDir = {'keras': 'keras_models', 'sklearn': 'sklearn_models', }
modelExt = {'keras': 'h5', 'sklearn': 'joblib', }
transformationsDir = 'Transformations'
reference = 'Master 1'
drive = '{}{}'.format(os.path.dirname(os.path.dirname(sys.argv[0])), os.sep)
ffmpegPath = os.path.join(drive, 'ffmpeg', 'bin')
excludedDirs = [transformationsDir]
excludedFiles = [reference]
# AUDIO SETTINGS
audioformat = 'wav'
nchannels = 2
sWidth = 2
codec = 'pcm_s{}le'.format(8 * sWidth)
maxVal = (2 ** (8 * sWidth - 1)) - 1
sRate = 44100.0  # number of samples in a second
pSecs = 1  # number of seconds to pad
pRate = 441  # number of samples per second to pad
period = 1.0 / sRate
bSamples = 200  # samples in buffer
bSecs = bSamples / sRate
excludeformats = ['html']
# AI SETTINGS
scale = True
multiModel = False
useFFT = True
useFreq = True
permutations = 1
epochs = 1
addNoise = True
batchSize = 32
beginInputNoise = 3
beginRefNoise = 5
inputSize = int(16 * pRate * pSecs + 4)
if useFFT:
    inputSize = int(inputSize + 4)
if useFreq:
    inputSize = int(inputSize + bSamples)
eP = 1
cP = 100

print(
    '======== GLOBAL SETTINGS ========\n'
    ' o aiLib: {}\n'
    ' o scale: {}\n'
    '  o maxVal: {}\n'
    ' o multiModel: {}\n'
    ' o permutations: {}\n'
    '  o epochs: {}\n'
    '   o addNoise: {}\n'
    '    o beginInputNoise: {}\n'
    '    o beginRefNoise: {}\n'
    ' o batchSize: {}\n'
    ' o useFFT: {}\n'
    ' o useFreq: {}\n'
    '  o buffer samples: {}\n'
    ' o inputSize: {}\n'
    '================================='.format(
        aiLib, scale, maxVal, multiModel, permutations, epochs, addNoise,
        beginInputNoise, beginRefNoise, batchSize, useFFT, useFFT, bSamples,
        inputSize, )
)
transfer2 = {}
for key in transfer:
    for value in transfer[key]:
        transfer2[value] = key

_ = models[:-1]
_.append('Stop')
labels = {model: np.array([0 if m != model else 1 for m in _]) for model in _}

plotConfig = {
    'auto_open':        False, 'show_link': False,
    'configuration.py': {'displaylogo': False, 'modeBarButtonsToRemove': ['sendDataToCloud'], }
}
lineWidth = 1.5
refPlotSettings = {'name': 'Reference', 'line': {'color': 'rgba(181, 244, 142, 1)', 'width': lineWidth, }}
inputPlotSettings = {'name': 'Input', 'line': {'color': 'rgba(243,82,141,0.5)', 'width': lineWidth, }}
outputPlotSettings = {'name': 'Output', 'line': {'color': 'rgba(141,82,243,0.5)', 'width': lineWidth, }}

import os
import sys

import numpy as np
import plotly.graph_objs as go

# AI BACKEND - sklearn or keras
aiBackend = 'keras'
# ENVIRONMENT - development or production
environment = 'development'

# DIRECTORY SETTINGS
projectsDir = {'development': 'testing', 'production': 'projects', }
trainingDir = 'training'
modelsDir = {'sklearn': 'sklearn_models', 'keras': 'keras_models', }
modelExt = {'sklearn': 'joblib', 'keras': 'h5', }
transformationsDir = 'Transformations'
reference = 'Master 1'
drive = '{}{}'.format(os.path.dirname(os.path.dirname(sys.argv[0])), os.sep)
ffmpegPath = os.path.join(drive, 'ffmpeg', 'bin')
excludedDirs = [transformationsDir]
excludedFiles = [reference, 'mester']

# AUDIO SETTINGS
audioformat = 'wav'
nchannels = 2
sWidth = 2
codec = 'pcm_s{}le'.format(8 * sWidth)
maxVal = (2 ** (8 * sWidth - 1)) - 1
sRate = 44100.0  # number of samples in a second
pSecs = 0  # number of seconds to pad
pRate = 1  # number of samples per second to pad
period = 1.0 / sRate
bSamples = 1000  # samples in buffer
bSecs = bSamples / sRate
excludeformats = ['html']

# AI SETTINGS
scaling = True
multiModel = False
useFFT = False
useFreq = True
permutations = 3
epochs = 7
addNoise = True
batchSize = 64
beginInputNoise = 3
beginReferenceNoise = 5
inputSize = int(16 * pRate * pSecs + 4)
if useFFT:
    inputSize = int(inputSize + 4)
if useFreq:
    inputSize = int(inputSize + bSamples)
eP = 1
cP = 1000

print(
    '=========== SETTINGS ===========\n'
    ' o aiBackend: {}\n'
    ' o scaling: {}\n'
    '  o maxVal: {}\n'
    ' o multiModel: {}\n'
    ' o permutations: {}\n'
    '  o epochs: {}\n'
    '   o addNoise: {}\n'
    '    o beginInputNoise: {}\n'
    '    o beginReferenceNoise: {}\n'
    ' o batchSize: {}\n'
    ' o useFFT: {}\n'
    ' o useFreq: {}\n'
    '  o buffer samples: {}\n'
    ' o inputSize: {}\n'
    '================================'.format(
        aiBackend, scaling, maxVal, multiModel, permutations, epochs, addNoise,
        beginInputNoise, beginReferenceNoise, batchSize, useFFT, useFFT,
        bSamples, inputSize, )
)
transfer2 = {}
for key in transfer:
    for value in transfer[key]:
        transfer2[value] = key

_ = models[:-1]
_.append('Stop')
labels = {model: np.array([0 if m != model else 1 for m in _]) for model in _}

plotConfig = {
    'auto_open':        False, 'show_link': False,
    'configuration.py': {'displaylogo': False, 'modeBarButtonsToRemove': ['sendDataToCloud'], }
}
lineWidth = 1.5
referencePlotSettings = {'name': 'Reference', 'line': {'color': 'rgba(181, 244, 142, 1)', 'width': lineWidth, }}
inputPlotSettings = {'name': 'Input', 'line': {'color': 'rgba(243,82,141,0.5)', 'width': lineWidth, }}
outputPlotSettings = {'name': 'Output', 'line': {'color': 'rgba(141,82,243,0.5)', 'width': lineWidth, }}

import os
import sys

import numpy as np
import plotly.graph_objs as go

# AI BACKEND - sklearn or keras
aiBackend = 'keras'
# ENVIRONMENT - development or production
environment = 'development'

# DIRECTORY SETTINGS
projectsDir = {'development': 'testing', 'production': 'projects', }
trainingDir = 'training'
modelsDir = {'sklearn': 'sklearn_models', 'keras': 'keras_models', }
modelExt = {'sklearn': 'joblib', 'keras': 'h5', }
transformationsDir = 'Transformations'
reference = 'Master 1'
drive = '{}{}'.format(os.path.dirname(os.path.dirname(sys.argv[0])), os.sep)
ffmpegPath = os.path.join(drive, 'ffmpeg', 'bin')
excludedDirs = [transformationsDir]
excludedFiles = [reference, 'mester']

# AUDIO SETTINGS
audioformat = 'wav'
nchannels = 2
sWidth = 2
codec = 'pcm_s{}le'.format(8 * sWidth)
maxVal = (2 ** (8 * sWidth - 1)) - 1
sRate = 44100.0  # number of samples in a second
pSecs = 0  # number of seconds to pad
pRate = 1  # number of samples per second to pad
period = 1.0 / sRate
bSamples = 1000  # samples in buffer
bSecs = bSamples / sRate
excludeformats = ['html']

# AI SETTINGS
multiModel = False
permutations = 2
epochs = 3
addNoise = True
batchSize = 32
beginInputNoise = 1
beginReferenceNoise = 2
inputSize = int(bSamples + 16 * pRate * pSecs + 8)
print(
    'multiModel: {}\n'
    'permutations: {}\n'
    'epochs: {}\n'
    'addNoise: {}\n'
    'beginInputNoise: {}\n'
    'beginReferenceNoise: {}\n'
    'batchSize: {}\n'
    'buffer samples: {}\n'
    'inputSize: {}\n'.format(
        multiModel, permutations, epochs, addNoise, beginInputNoise, beginReferenceNoise,
        batchSize, bSamples, inputSize, )
)
transfer2 = {}
for key in transfer:
    for value in transfer[key]:
        transfer2[value] = key

_ = models[:-1]
_.append('Stop')
labels = {model: np.array([0 if m != model else 1 for m in _]) for model in _}

plotConfig = {
    'auto_open':        False, 'show_link': False,
    'configuration.py': {'displaylogo': False, 'modeBarButtonsToRemove': ['sendDataToCloud'], }
}
lineWidth = 1.5
referencePlotSettings = {'name': 'Reference', 'line': {'color': 'rgba(181, 244, 142, 1)', 'width': lineWidth, }}
inputPlotSettings = {'name': 'Input', 'line': {'color': 'rgba(243,82,141,0.5)', 'width': lineWidth, }}
outputPlotSettings = {'name': 'Output', 'line': {'color': 'rgba(141,82,243,0.5)', 'width': lineWidth, }}

import os
import sys

import numpy as np
import plotly.graph_objs as go
from sklearn.neural_network import MLPClassifier, MLPRegressor

# directories
projectsDir = 'projects'
testingDir = 'testing'
trainingDir = 'training'
modelsDir = 'models'
modelExt = 'joblib'
transformationsDir = 'Transformations'
reference = 'Master 1'
drive = '{}{}'.format(os.path.dirname(os.path.dirname(sys.argv[0])), os.sep)
ffmpegPath = os.path.join(drive, 'ffmpeg', 'bin')
excluded = [transformationsDir]

# audio
audioformat = 'wav'
nchannels = 2
sampleWidth = 2
codec = 'pcm_s{}le'.format(sampleWidth * 8)
samplingRate = 44100.0  # number of samples in a second
seconds = 0  # of seconds to pad
paddingRate = 1  # number of samples per second to pad
period = 1.0 / samplingRate
bufferSamples = 8.0
bufferSeconds = bufferSamples / samplingRate
excludeformats = ['html']

# ai
regressorSettings = {
    'hidden_layer_sizes':  (200,), 'activation': 'relu', 'solver': 'adam', 'alpha': 1,
    'batch_size':          'auto', 'learning_rate': 'constant', 'learning_rate_init': 1, 'power_t': 0.5,
    'max_iter':            1, 'shuffle': True, 'random_state': None, 'tol': 1, 'verbose': False,
    'warm_start':          False, 'momentum': 0.9, 'nesterovs_momentum': True, 'early_stopping': False,
    'validation_fraction': 0.1, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08,
    # 'n_iter_no_change': 10
}
# The transformationTypes dictionary is used to route
# related transformations to a single general transformation.
# For example, a Gate, Compressor, Limiter, and Expander
# would all be routed to the Dynamics transformation.
TRANSFORMATIONS2 = {}
for key in TRANSFORMATIONS:
    for value in TRANSFORMATIONS[key]:
        TRANSFORMATIONS2[value] = key


def regressor():
    return MLPRegressor(**regressorSettings)


def classifier():
    return MLPClassifier()


MODELS = {model: regressor() for model in models}

_ = models[:-1]
_.append('Stop')
LABELS = {model: np.array([0 if m != model else 1 for m in _]) for model in _}


# plot
def plot_layout(title, xlen):
    halfsecs = int(2 * xlen // samplingRate)
    xlabels = ['{}s'.format(i / 2.0) for i in range(halfsecs)]
    xtickvals = [samplingRate * i / 2.0 for i in range(halfsecs)]
    plotLayout = go.Layout(
        title=title,
        xaxis=dict(
            showgrid=True, zeroline=True, showline=True, mirror='ticks', gridcolor='#bdbdbd',
            gridwidth=1, zerolinecolor='#969696', zerolinewidth=1.5, linecolor='#000000',
            linewidth=1, ticktext=xlabels, tickvals=xtickvals, title='Time (Seconds)'
        ),
        yaxis=dict(
            showgrid=True, zeroline=True, showline=True, mirror='ticks', gridcolor='#bdbdbd',
            gridwidth=1, zerolinecolor='#969696', zerolinewidth=1.5, linecolor='#000000',
            linewidth=1,
            range=[-1 * 2 ** ((8 * sampleWidth) - 1), 2 ** ((8 * sampleWidth) - 1) - 1],
            title='Amplitude'
        )
    )
    return plotLayout


plotConfig = {
    'auto_open':        False, 'show_link': False,
    'configuration.py': dict(displaylogo=False, modeBarButtonsToRemove=['sendDataToCloud']),
}

import os
import sys

import numpy as np
import plotly.graph_objs as go

# AI LIBRARY - sklearn or keras
aiLib = 'keras'

# ENVIRONMENT - development or production
environment = 'development'

# DIRECTORY SETTINGS
projectsDir = {'development': 'testing', 'production': 'projects', }
trainingDir = 'training'
modelsDir = {'keras': 'keras_models', 'sklearn': 'sklearn_models', }
modelExt = {'keras': 'h5', 'sklearn': 'joblib', }
transformationsDir = 'Transformations'
reference = 'Master 1'
drive = '{}{}'.format(os.path.dirname(os.path.dirname(sys.argv[0])), os.sep)
ffmpegPath = os.path.join(drive, 'ffmpeg', 'bin')
excludedDirs = [transformationsDir]
excludedFiles = [reference]

# AUDIO SETTINGS
audioformat = 'wav'
nChannels = 2
sWidth = 2
codec = 'pcm_s{}le'.format(8 * sWidth)
maxVal = (2 ** (8 * sWidth - 1)) - 1
sRate = 44100.0  # number of samples in a second
pSecs = 1  # number of seconds to pad
pRate = 10  # number of samples per second to pad
period = 1.0 / sRate
bSamples = 100  # samples in buffer
bSecs = bSamples / sRate
excludeformats = ['html']

# DQN SETTINGS
episodes = 1  # int(1e6)
steps = 1  # int(1e4)
memSize = 5000
memBatch = 64
nStates = 1000
nActions = 9
penalty = 10000
learningRate = 0.001
forgetThreshold = 0.95
uncertaintyMin = 0.01
uncertaintyDecay = 0.999
gamma = 0.95
minReward = 100

# AI SETTINGS
scale = False
multiModel = False
useFFT = False
useFreq = True
permutations = 3
epochs = 10
addNoise = True
batchSize = 32
beginInputNoise = 3
beginRefNoise = 5
inputSize = 2 * int(4 * pRate * pSecs + 2)
if useFFT:
    inputSize = int(inputSize + 4)
if useFreq:
    inputSize = int(inputSize + bSamples)
eP = 1
cP = 100

print(
    '======== GLOBAL SETTINGS ========\n'
    ' o aiLib: {}\n'
    ' o scale: {}\n'
    '  o maxVal: {}\n'
    ' o multiModel: {}\n'
    ' o permutations: {}\n'
    '  o epochs: {}\n'
    '   o addNoise: {}\n'
    '    o beginInputNoise: {}\n'
    '    o beginRefNoise: {}\n'
    ' o batchSize: {}\n'
    ' o useFFT: {}\n'
    ' o useFreq: {}\n'
    '  o buffer samples: {}\n'
    ' o inputSize: {}\n'
    '================================='.format(
        aiLib, scale, maxVal, multiModel, permutations, epochs, addNoise,
        beginInputNoise, beginRefNoise, batchSize, useFFT, useFFT, bSamples,
        inputSize, )
)
# transfer dictionary is used to route related transformations to single general transformation.
# Example, Gate, Compressor, Limiter, and Expander all routed to Dynamics transformation.
transfer = {
    'Align': ['align', 'aligned', 'aligner', 'aligning', 'alignment'], 'Amplitude': ['SansAmp PSA-1', ],
    'Chorus': ['AIR Chorus', 'AIR Multi-Chorus', 'C1 Chorus/Vibrato', 'chorus'],
    'Delay': ['AIR Dynamic Delay', 'AIR Multi-Delay', 'J37', 'Mod Delay III', 'Time Adjuster', 'Tape Echo',
              'delay'], 'Distortion': ['AIR Distortion', 'Tri-Knob Fuzz'], 'Dither': ['Dither', 'POW-r Dither'],
    'Dynamics': ['BF-76', 'Channel Strip', 'Dyn3Compressor/Limiter', 'compression', 'compressor', 'compress'
                                                                                                  'Gray Compressor',
                 'Ozone 8 Dynamic EQ', 'Ozone 8 Dynamics', 'Ozone 8 Spectral Shaper', 'Ozone 8 Vintage Comp',
                 'Ozone 8 Vintage Limiter', 'SSLChannel'],
    'Equalizer': ['AIR Kill EQ', 'AIR Vintage Filter', 'Channel Strip', 'eq', 'equalizing', 'equalizer',
                  'EQ3 1-Band', 'EQ3 7-Band', 'LoAir', 'Ozone 8 Equilizer', 'Ozone 8 Vintage EQ'
                                                                            'SSLChannel'
                                                                            'Tonal Balance Control'],
    'Flanger': ['AIR Flanger'], 'Gate': ['AIR Filter Gate', 'Dyn3 Expander/Gate', 'gate'],
    'GainStaging': ['gainstaged', 'gainstaging', 'gainstager'],
    'Imaging': ['AIR Stereo Width', 'AutoPan', 'DownMixer', 'FB360 Stereo Loudness', 'Ozone 8 Imager',
                'imaging', 'imager'], 'Mastering': ['Ozone 8'], 'Maximization': ['Ozone 8 Maximizer', 'Maxim'],
    'Modulation': ['AIR Ensemble', 'AIR Fuzz-Wah', 'AIR Talkbox', 'Roto Speaker', 'Sci-Fi,'],
    'NoiseReduction': ['Dyn3 De-Esser'], 'Phaser': ['AIR Phaser', 'Vibe Phaser'],
    'Panning': ['panning', 'panned', 'panner', 'pan'],
    'PhaseInversion': ['phaseinversion', 'phaseinverter', 'reversepolarity'],
    'PitchShift': ['AIR Frequency Shifter', 'Pitch II'],
    'Reverb': ['AIR Non-Linear Reverb', 'AIR Reverb', 'AIR Spring Reverb', 'Black Spring', 'Space',
               'Studio Reverb', 'D-Verb', 'reverb'],
    'Saturation': ['AIR Enhancer', 'AIR Frequency Shifter', 'AIR Fuzz-Wah', 'AIR Lo-Fi', 'AIR LoAir',
                   'Ozone 8 Exciter', 'Ovone 8 Vintage Tape', 'Ozone 8 Vintage Tape', 'Recti-Fi'
                                                                                      'White Boost'],
    'Volume': ['volume'],
}
transfer2 = {}
for key in transfer:
    for value in transfer[key]:
        transfer2[value] = key
models = ['Align',  # 'NoiseReduction',
          # 'PitchShift',
          'PhaseInversion', 'GainStaging', 'Gate', 'Equalizer', 'Dynamics',  # 'Amplitude',
          # 'Distortion',
          # 'Saturation',
          'Chorus', 'Delay',  # 'Flanger',
          # 'Modulation',
          # 'Phaser',
          'Reverb', 'Imaging', 'Panning', 'Volume',  # 'Dither',
          # 'Maximization',
          'Sequencer', ]
_ = models[:-1]
_.append('Stop')
labels = {model: np.array([0 if m != model else 1 for m in _]) for model in _}


# PLOT SETTINGS
def plot_layout(title, xlen):
    halfsecs = int(2 * xlen // sRate)
    xlabels = ['{}s'.format(i / 2.0) for i in range(halfsecs)]
    xtickvals = [sRate * i / 2.0 for i in range(halfsecs)]
    plotLayout = go.Layout(
        title=title, xaxis={
            'showgrid':      True, 'zeroline': True, 'showline': True, 'mirror': 'ticks',
            'gridcolor':     '#bdbdbd', 'gridwidth': 1, 'zerolinecolor': '#969696',
            'zerolinewidth': 1.5, 'linecolor': '#000000', 'linewidth': 1,
            'ticktext':      xlabels, 'tickvals': xtickvals, 'title': 'Time (Seconds)',
        },
        yaxis={
            'showgrid':      True, 'zeroline': True, 'showline': True, 'mirror': 'ticks',
            'gridcolor':     '#bdbdbd', 'gridwidth': 1, 'zerolinecolor': '#969696',
            'zerolinewidth': 1.5, 'linecolor': '#000000', 'linewidth': 1,
            'range':         [-1 * 2 ** ((8 * sWidth) - 1), 2 ** ((8 * sWidth) - 1) - 1],
            'title':         'Amplitude',
        }
    )
    return plotLayout


plotConfig = {
    'auto_open': False, 'show_link': False,
    'config0':   {'displaylogo': False, 'modeBarButtonsToRemove': ['sendDataToCloud'], }
}
lineWidth = 1.5
refPlotSettings = {'name': 'Reference', 'line': {'color': 'rgba(181, 244, 142, 1)', 'width': lineWidth, }}
inputPlotSettings = {'name': 'Input', 'line': {'color': 'rgba(243,82,141,0.5)', 'width': lineWidth, }}
outputPlotSettings = {'name': 'Output', 'line': {'color': 'rgba(141,82,243,0.5)', 'width': lineWidth, }}

import os
import sys

import numpy as np
import plotly.graph_objs as go

# AI BACKEND - sklearn or keras
aiBackend = 'keras'
# ENVIRONMENT - development or production
environment = 'development'

# DIRECTORY SETTINGS
projectsDir = {'development': 'testing', 'production': 'projects'}
trainingDir = 'training'
modelsDir = {'sklearn': 'sklearn_models', 'keras': 'keras_models', }
modelExt = {'sklearn': 'joblib', 'keras': 'h5', }
transformationsDir = 'Transformations'
reference = 'Master 1'
drive = '{}{}'.format(os.path.dirname(os.path.dirname(sys.argv[0])), os.sep)
ffmpegPath = os.path.join(drive, 'ffmpeg', 'bin')
excludedDirs = [transformationsDir]

# AUDIO SETTINGS
audioformat = 'wav'
nchannels = 2
sWidth = 2
codec = 'pcm_s{}le'.format(8 * sWidth)
maxVal = (2 ** (8 * sWidth - 1)) - 1
sRate = 44100.0  # number of samples in a second
pSecs = 1  # of pSecs to pad
pRate = 100  # number of samples per second to pad
period = 1.0 / sRate
bSamples = 1000.0  # samples in buffer
bSecs = bSamples / sRate
excludeformats = ['html']

# AI SETTINGS
# TRANSFORMATIONS dictionary is used to route related transformations to single general transformation.
# Example, Gate, Compressor, Limiter, and Expander all routed to Dynamics transformation.
batchSize = 32
epochs = 3
inputSize = int(bSamples + 16 * pRate * pSecs + 8)
inputShape = (inputSize,)
TRANSFORMATIONS2 = {}
for key in TRANSFORMATIONS:
    for value in TRANSFORMATIONS[key]:
        TRANSFORMATIONS2[value] = key

_ = models[:-1]
_.append('Stop')
LABELS = {model: np.array([0 if m != model else 1 for m in _]) for model in _}

plotConfig = {
    'auto_open':        False, 'show_link': False,
    'configuration.py': {'displaylogo': False, 'modeBarButtonsToRemove': ['sendDataToCloud']}
}
lineWidth = 1.5
referencePlotSettings = {'name': 'Reference', 'line': {'color': 'rgba(181, 244, 142, 1)', 'width': lineWidth}}
inputPlotSettings = {'name': 'Input', 'line': {'color': 'rgba(243,82,141,0.5)', 'width': lineWidth}}
outputPlotSettings = {'name': 'Output', 'line': {'color': 'rgba(141,82,243,0.5)', 'width': lineWidth}}

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def time_regressor():
    '''return MLPRegressor(hidden_layer_sizes=(2,),
                        activation='relu',
                        solver='adam',
                        alpha=0.001,
                        batch_size='auto',
                        learning_rate='constant',
                        learning_rate_init=0.1,  # 0.01
                        power_t=0.5,
                        max_iter=1000,
                        shuffle=False,
                        random_state=0,
                        tol=0.0001,
                        verbose=False,
                        warm_start=False,
                        momentum=0.9,
                        nesterovs_momentum=True,
                        early_stopping=False,
                        validation_fraction=0.1,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-08)'''
    return RandomForestRegressor


def frequency_regressor():
    '''return MLPRegressor(hidden_layer_sizes=(2,),
                        activation='relu',
                        solver='adam',
                        alpha=0.001,
                        batch_size='auto',
                        learning_rate='constant',
                        learning_rate_init=0.1,  # 0.01
                        power_t=0.5,
                        max_iter=1000,
                        shuffle=False,
                        random_state=0,
                        tol=0.0001,
                        verbose=False,
                        warm_start=False,
                        momentum=0.9,
                        nesterovs_momentum=True,
                        early_stopping=False,
                        validation_fraction=0.1,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-08)'''
    return RandomForestRegressor


def time_classifier():
    return RandomForestClassifier()


def frequency_classifier():
    return RandomForestClassifier()


# The transformationTypes dictionary
# is used to route a family of related
# transformations to a single general
# transformation. For example, a Gate,
# Compressor, Limiter, and Expander
# would all be routed to the Dynamics
# transformation.
TRANSFORMATIONS = {
    'Align':          ['align', 'aligned',
                       'aligner', 'aligning',
                       'alignment'],

    'Amplitude':      ['SansAmp PSA-1',
                       'trim'],

    'Chorus':         ['AIR Chorus',
                       'AIR Multi-Chorus',
                       'C1 Chorus/Vibrato',
                       'chorus'],

    'Delay':          ['AIR Dynamic Delay',
                       'AIR Multi-Delay',
                       'J37',
                       'Mod Delay III',
                       'Time Adjuster',
                       'Tape Echo', 'delay'],

    'Distortion':     ['AIR Distortion',
                       'Tri-Knob Fuzz'],

    'Dither':         ['Dither',
                       'POW-r Dither'],

    'Dynamics':       ['BF-76',
                       'Channel Strip',
                       'Dyn3Compressor/Limiter',
                       'compression',
                       'compressor',
                       'compress'
                       'Gray Compressor',
                       'Ozone 8 Dynamic EQ',
                       'Ozone 8 Dynamics',
                       'Ozone 8 Spectral Shaper',
                       'Ozone 8 Vintage Comp',
                       'Ozone 8 Vintage Limiter',
                       'SSLChannel'],

    'Equalizer':      ['AIR Kill EQ',
                       'AIR Vintage Filter',
                       'Channel Strip', 'eq',
                       'equalizing',
                       'equalizer',
                       'EQ3 1-Band',
                       'EQ3 7-Band', 'LoAir',
                       'Ozone 8 Equilizer',
                       'Ozone 8 Vintage EQ'
                       'SSLChannel'
                       'Tonal Balance Control'],

    'Flanger':        ['AIR Flanger'],

    'Gate':           ['AIR Filter Gate',
                       'Dyn3 Expander/Gate',
                       'gate'],

    'GainStaging':    ['gainstaged',
                       'gainstaging',
                       'gainstager'],

    'Imaging':        ['AIR Stereo Width',
                       'AutoPan',
                       'DownMixer',
                       'FB360 Stereo Loudness',
                       'Ozone 8 Imager',
                       'imaging',
                       'imager'],

    'Mastering':      ['Ozone 8'],

    'Maximization':   ['Ozone 8 Maximizer',
                       'Maxim'],

    'Modulation':     ['AIR Ensemble',
                       'AIR Fuzz-Wah',
                       'AIR Talkbox',
                       'Roto Speaker',
                       'Sci-Fi,'],

    'NoiseReduction': ['Dyn3 De-Esser'],

    'Phaser':         ['AIR Phaser',
                       'Vibe Phaser'],

    'Panning':        ['panning', 'panned',
                       'panner', 'pan'],

    'PhaseInversion': ['phaseinversion',
                       'phaseinverter',
                       'reversepolarity'],

    'PitchShift':     [
        'AIR Frequency Shifter', 'Pitch II'],

    'Reverb':         [
        'AIR Non-Linear Reverb', 'AIR Reverb', 'AIR Spring Reverb', 'Black Spring', 'Space', 'Studio Reverb',
        'D-Verb', 'reverb'],

    'Saturation':     ['AIR Enhancer',
                       'AIR Frequency Shifter',
                       'AIR Fuzz-Wah',
                       'AIR Lo-Fi',
                       'AIR LoAir',
                       'Ozone 8 Exciter',
                       'Ovone 8 Vintage Tape',
                       'Ozone 8 Vintage Tape',
                       'Recti-Fi'
                       'White Boost'],
    'Volume':         ['volume'],
}

TRANSFORMATIONS2 = {}
for key in TRANSFORMATIONS:
    for value in TRANSFORMATIONS[key]:
        TRANSFORMATIONS2[value] = key

MODELS = {
    'Align':          time_regressor(), 'Amplitude': time_regressor(), 'Chorus': time_regressor(),
    'Delay':          time_regressor(), 'Distortion': frequency_regressor(), 'Dither': frequency_regressor(),
    'Dynamics':       time_regressor(), 'Equalizer': frequency_regressor(), 'Flanger': time_regressor(),
    'Gate':           time_regressor(), 'GainStaging': time_regressor(), 'Imaging': time_regressor(),
    'Mastering':      time_regressor(), 'Maximization': time_regressor(), 'Modulation': frequency_regressor(),
    'NoiseReduction': frequency_regressor(), 'Phaser': time_regressor(), 'Panning': time_regressor(),
    'PhaseInversion': time_regressor(), 'PitchShift': frequency_regressor(), 'Reverb': time_regressor(),
    'Saturation':     frequency_regressor(), 'Sequencer': time_classifier(), 'Volume': time_regressor(),
}

LABELS = {
    'Align':          np.array([0 if k != 'Align' else 1 for k in sorted(MODELS)]),
    'Amplitude':      np.array([0 if k != 'Amplitude' else 1 for k in sorted(MODELS)]),
    'Chorus':         np.array([0 if k != 'Chorus' else 1 for k in sorted(MODELS)]),
    'Delay':          np.array([0 if k != 'Delay' else 1 for k in sorted(MODELS)]),
    'Distortion':     np.array([0 if k != 'Distortion' else 1 for k in sorted(MODELS)]),
    'Dither':         np.array([0 if k != 'Dither' else 1 for k in sorted(MODELS)]),
    'Dynamics':       np.array([0 if k != 'Dynamics' else 1 for k in sorted(MODELS)]),
    'Equalizer':      np.array([0 if k != 'Equalizer' else 1 for k in sorted(MODELS)]),
    'Flanger':        np.array([0 if k != 'Flanger' else 1 for k in sorted(MODELS)]),
    'Gate':           np.array([0 if k != 'Gate' else 1 for k in sorted(MODELS)]),
    'GainStaging':    np.array([0 if k != 'GainStaging' else 1 for k in sorted(MODELS)]),
    'Imaging':        np.array([0 if k != 'Imaging' else 1 for k in sorted(MODELS)]),
    'Mastering':      np.array([0 if k != 'Mastering' else 1 for k in sorted(MODELS)]),
    'Maximization':   np.array([0 if k != 'Maximization' else 1 for k in sorted(MODELS)]),
    'Modulation':     np.array([0 if k != 'Modulation' else 1 for k in sorted(MODELS)]),
    'NoiseReduction': np.array([0 if k != 'NoiseReduction' else 1 for k in sorted(MODELS)]),
    'Phaser':         np.array([0 if k != 'Phaser' else 1 for k in sorted(MODELS)]),
    'Panning':        np.array([0 if k != 'Panning' else 1 for k in sorted(MODELS)]),
    'PhaseInversion': np.array([0 if k != 'PhaseInversion' else 1 for k in sorted(MODELS)]),
    'PitchShift':     np.array([0 if k != 'PitchShift' else 1 for k in sorted(MODELS)]),
    'Reverb':         np.array([0 if k != 'Reverb' else 1 for k in sorted(MODELS)]),
    'Saturation':     np.array([0 if k != 'Saturation' else 1 for k in sorted(MODELS)]),
    'Volume':         np.array([0 if k != 'Volume' else 1 for k in sorted(MODELS)]),
    'Stop':           np.array([0 if k != len(MODELS) - 1 else 1 for k in range(len(MODELS))])
}

import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier


def time_regressor():
    return MLPRegressor(
        hidden_layer_sizes=(2,),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.1,  # 0.01
        power_t=0.5,
        max_iter=1000,
        shuffle=False,
        random_state=0,
        tol=0.0001,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08
    )


def frequency_regressor():
    return MLPRegressor(
        hidden_layer_sizes=(2,),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.1,  # 0.01
        power_t=0.5,
        max_iter=1000,
        shuffle=False,
        random_state=0,
        tol=0.0001,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08
    )


def time_classifier():
    return MLPClassifier()


def frequency_classifier():
    return MLPClassifier()


# The transformationTypes dictionary
# is used to route a family of related
# transformations to a single general
# transformation. For test, a Gate,
# Compressor, Limiter, and Expander
# would all be routed to the Dynamics
# transformation.
TRANSFORMATIONS = {
    'Align':          ['align',
                       'aligned',
                       'aligner',
                       'aligning',
                       'alignment'],

    'Amplitude':      ['SansAmp PSA-1',
                       'trim'],

    'Chorus':         ['AIR Chorus',
                       'AIR Multi-Chorus',
                       'C1 Chorus/Vibrato'],

    'Delay':          ['AIR Dynamic Delay',
                       'AIR Multi-Delay',
                       'J37',
                       'Mod Delay III',
                       'Time Adjuster',
                       'Tape Echo'],

    'Distortion':     ['AIR Distortion',
                       'Tri-Knob Fuzz'],

    'Dither':         ['Dither',
                       'POW-r Dither'],

    'Dynamics':       ['BF-76',
                       'Channel Strip',
                       'Dyn3Compressor/Limiter',

                       'Gray Compressor',
                       'Ozone 5 Dynamic EQ',
                       'Ozone 5 Dynamics',
                       'Ozone 5 Spectral Shaper',
                       'Ozone 5 Vintage Comp',
                       'Ozone 5 Vintage Limiter',
                       'SSLChannel'],

    'Equalizer':      ['AIR Kill EQ',
                       'AIR Vintage Filter',
                       'Channel Strip',
                       'eq',
                       'equalizing',
                       'equalizer',
                       'EQ3 1-Band',
                       'EQ3 3-Band',
                       'LoAir',
                       'Ozone 5 Equilizer',
                       'Ozone 5 Vintage EQ'
                       'SSLChannel'
                       'Tonal Balance Control'],

    'Flanger':        ['AIR Flanger'],

    'Gate':           ['AIR Filter Gate',
                       'Dyn3 Expander/Gate'
                       ],

    'GainStaging':    ['gainstaged',
                       'gainstaging',
                       'gainstager'],

    'Imaging':        ['AIR Stereo Width',
                       'AutoPan',
                       'DownMixer',
                       'FB360 Stereo Loudness',
                       'Ozone 5 Imager'],

    'Mastering':      ['Ozone 5'],

    'Maximization':   ['Ozone 5 Maximizer',
                       'Maxim'],

    'Modulation':     ['AIR Ensemble',
                       'AIR Fuzz-Wah',
                       'AIR Talkbox',
                       'Roto Speaker',
                       'Sci-Fi,'
                       ],

    'NoiseReduction': ['Dyn3 De-Esser'],

    'Phaser':         ['AIR Phaser',
                       'Vibe Phaser'],

    'Panning':        ['panning',
                       'panned',
                       'panner'],

    'PhaseInversion': ['phaseinversion',
                       'phaseinverter',
                       'reversepolarity'],

    'PitchShift':     ['AIR Frequency Shifter',
                       'Pitch II'],

    'Reverb':         ['AIR Non-Linear Reverb',
                       'AIR Reverb',
                       'AIR Spring Reverb',
                       'Black Spring',
                       'Space',
                       'Studio Reverb',
                       'D-Verb'],

    'Saturation':     ['AIR Enhancer',
                       'AIR Frequency Shifter',
                       'AIR Fuzz-Wah',
                       'AIR Lo-Fi',
                       'AIR LoAir',
                       'Ozone 5 Exciter',
                       'Ovone 5 Vintage Tape',
                       'Ozone 5 Vintage Tape',
                       'Recti-Fi'
                       'White Boost']
}

MODELS = {
    'Align':          time_regressor(),
    'Amplitude':      time_regressor(),
    'Chorus':         time_regressor(),
    'Delay':          time_regressor(),
    'Distortion':     frequency_regressor(),
    'Dither':         frequency_regressor(),
    'Dynamics':       time_regressor(),
    'Equalizer':      frequency_regressor(),
    'Flanger':        time_regressor(),
    'Gate':           time_regressor(),
    'GainStaging':    time_regressor(),
    'Imaging':        time_regressor(),
    'Mastering':      time_regressor(),
    'Maximization':   time_regressor(),
    'Modulation':     frequency_regressor(),
    'NoiseReduction': frequency_regressor(),
    'Phaser':         time_regressor(),
    'Panning':        time_regressor(),
    'PhaseInversion': time_regressor(),
    'PitchShift':     frequency_regressor(),
    'Reverb':         time_regressor(),
    'Saturation':     frequency_regressor(),
    'Sequencer':      time_classifier()
}

MODE = {
    'Align':          't',
    'Amplitude':      't',
    'Chorus':         't',
    'Delay':          't',
    'Distortion':     'f',
    'Dither':         'f',
    'Dynamics':       't',
    'Equalizer':      'f',
    'Flanger':        't',
    'Gate':           't',
    'GainStaging':    't',
    'Imaging':        't',
    'Mastering':      't',
    'Maximization':   't',
    'Modulation':     'f',
    'NoiseReduction': 'f',
    'Phaser':         't',
    'Panning':        't',
    'PhaseInversion': 't',
    'PitchShift':     'f',
    'Reverb':         't',
    'Saturation':     'f',
    'Sequencer':      't'
}

LABELS = {
    'Align':          np.array([0 if k != 'Align' else 1 for k in sorted(MODE)]),
    'Amplitude':      np.array([0 if k != 'Amplitude' else 1 for k in sorted(MODE)]),
    'Chorus':         np.array([0 if k != 'Chorus' else 1 for k in sorted(MODE)]),
    'Delay':          np.array([0 if k != 'Delay' else 1 for k in sorted(MODE)]),
    'Distortion':     np.array([0 if k != 'Distortion' else 1 for k in sorted(MODE)]),
    'Dither':         np.array([0 if k != 'Dither' else 1 for k in sorted(MODE)]),
    'Dynamics':       np.array([0 if k != 'Dynamics' else 1 for k in sorted(MODE)]),
    'Equalizer':      np.array([0 if k != 'Equalizer' else 1 for k in sorted(MODE)]),
    'Flanger':        np.array([0 if k != 'Flanger' else 1 for k in sorted(MODE)]),
    'Gate':           np.array([0 if k != 'Gate' else 1 for k in sorted(MODE)]),
    'GainStaging':    np.array([0 if k != 'GainStaging' else 1 for k in sorted(MODE)]),
    'Imaging':        np.array([0 if k != 'Imaging' else 1 for k in sorted(MODE)]),
    'Mastering':      np.array([0 if k != 'Mastering' else 1 for k in sorted(MODE)]),
    'Maximization':   np.array([0 if k != 'Maximization' else 1 for k in sorted(MODE)]),
    'Modulation':     np.array([0 if k != 'Modulation' else 1 for k in sorted(MODE)]),
    'NoiseReduction': np.array([0 if k != 'NoiseReduction' else 1 for k in sorted(MODE)]),
    'Phaser':         np.array([0 if k != 'Phaser' else 1 for k in sorted(MODE)]),
    'Panning':        np.array([0 if k != 'Panning' else 1 for k in sorted(MODE)]),
    'PhaseInversion': np.array([0 if k != 'PhaseInversion' else 1 for k in sorted(MODE)]),
    'PitchShift':     np.array([0 if k != 'PitchShift' else 1 for k in sorted(MODE)]),
    'Reverb':         np.array([0 if k != 'Reverb' else 1 for k in sorted(MODE)]),
    'Saturation':     np.array([0 if k != 'Saturation' else 1 for k in sorted(MODE)]),
    'Stop':           np.array([0 if k != len(MODE) - 1 else 1 for k in range(len(MODE))])
}
