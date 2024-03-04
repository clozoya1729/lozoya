samplingRate = 44100
rate = 1
seconds = 1
projectsDir = 'projects'
testingDir = 'testing'
trainingDir = 'training'
modelsDir = 'models'
modelExt = 'joblib'
transformationsDir = 'Transformations'
reference = 'Master 1'
nchannels = 2
samplewidth = 3
samplingRate = 44100.0  # number of samples in a second
seconds = 0  # of seconds to pad
paddingRate = 1  # number of samples per second to pad
period = 1.0/samplingRate
bufferSamples = 1000.0
bufferSeconds = bufferSamples/samplingRate
excluded = [transformationsDir]
projectsDir = 'projects'
trainingDir = 'training'
modelsDir = 'models'
transformationsDir = 'Transformations'
reference = 'Master 1'
samplingRate = 44100 #number of sampless in a second
seconds = 1 #of seconds to pad
rate = 1 #number of samples per second to pad
excluded = [transformationsDir]
modes = ['t']
mode = 't'

import os
import sys

projectsDir = 'projects'
testingDir = 'testing'
trainingDir = 'training'
modelsDir = 'models'
modelExt = 'joblib'
transformationsDir = 'Transformations'
reference = 'Master 1'
nchannels = 2
sampleWidth = 2
codec = 'pcm_s{}le'.format(sampleWidth * 8)
samplingRate = 44100.0  # number of samples in a second
seconds = 0  # of seconds to pad
paddingRate = 1  # number of samples per second to pad
period = 1.0 / samplingRate
bufferSamples = 8.0
bufferSeconds = bufferSamples / samplingRate
excluded = [transformationsDir]
drive = '{}{}'.format(sys.argv[0].split('/')[0], os.sep)
ffmpegPath = os.path.join(drive, 'ffmpeg', 'bin')
import os
projectsDir = 'projects'
testingDir = 'testing'
trainingDir = 'training'
modelsDir = 'models'
modelExt = 'joblib'
transformationsDir = 'Transformations'
reference = 'Master 1'
nchannels = 2
sampleWidth = 2
codec = 'pcm_s{}le'.format(sampleWidth*8)
samplingRate = 44100.0  # number of samples in a second
seconds = 0  # of seconds to pad
paddingRate = 1  # number of samples per second to pad
period = 1.0 / samplingRate
bufferSamples = 10.0
bufferSeconds = bufferSamples / samplingRate
excluded = [transformationsDir]
drive = 'C:{}'.format(os.sep)
ffmpegPath = os.path.join(drive, 'ffmpeg', 'bin')
import os
import sys

#directories
projectsDir = 'projects'
testingDir = 'testing'
trainingDir = 'training'
modelsDir = 'models'
modelExt = 'joblib'
transformationsDir = 'Transformations'
reference = 'Master 1'
drive = '{}{}'.format(sys.argv[0], os.sep)
ffmpegPath = os.path.join(drive, 'ffmpeg', 'bin')
excluded = [transformationsDir]

#audio
nchannels = 2
sampleWidth = 2
codec = 'pcm_s{}le'.format(sampleWidth * 8)
samplingRate = 44100.0  # number of samples in a second
seconds = 0  # of seconds to pad
paddingRate = 1  # number of samples per second to pad
period = 1.0 / samplingRate
bufferSamples = 8.0
bufferSeconds = bufferSamples / samplingRate

