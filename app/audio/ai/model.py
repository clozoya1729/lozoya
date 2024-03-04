# import matplotlib

# matplotlib.use('Qt5Agg')

# from matplotlib import pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor as Regressor
# import winsound
# from LoParDataProcessor0.DataProcessor import read_data, export_data

"""def train(inData, outData):
    leftIn, rightIn = split_channels(inData)
    leftOut, rightOut = split_channels(outData)
    leftRegressor = Regressor(n_estimators=1000)
    rightRegressor = Regressor(n_estimators=1000)
    return leftRegressor.fit(leftIn, leftOut), \
           rightRegressor.fit(rightIn, rightOut)


def mix_signals(signals):
    for i, signal in enumerate(signals):
        if i == 0:
            mixedSignal = signal
        else:
            mixedSignal = mixedSignal + signal
    return mixedSignal


def split_channels(dataFrame):
    left = dataFrame[0].values.reshape(-1, 1)
    right = dataFrame[1].values.reshape(-1, 1)
    return left, right


def merge_channels(left, right):
    return pd.DataFrame([left, right]).T.astype('int16')


def produce(dataFrame):
    left, right = split_channels(dataFrame)
    leftResult = leftRegressor.predict(left).astype('int16')
    rightResult = rightRegressor.predict(right).astype('int16')
    return merge_channels(leftResult, rightResult)


def random_data(size=25):
    return pd.DataFrame(
        np.random.randint(-10000, 10000, size=(size, 1)))"""


# soundDir = r'C:\Users\frano\Desktop\MusicProductionTest'
# savePath1 = os.path.join(soundDir, 'result1.wav')
# savePath2 = os.path.join(soundDir, 'result2.wav')
# inData = read_data(os.path.join(soundDir, 'in1_big.wav'))
# testData = read_data(os.path.join(soundDir, 'in2_big.wav'))
# outData = read_data(os.path.join(soundDir, 'out_big.wav'))
# leftRegressor, rightRegressor = train(inData, outData)
# result = produce(inData)
# export_data(data=result,
#            formats=['WAV'],
#            savePath=savePath1)
# result = produce(testData)
# export_data(data=result,
#            formats=['WAV'],
#            savePath=savePath2)
# winsound.PlaySound(savePath1, winsound.SND_FILENAME)
# plt.plot0(inData)
# plt.plot0(outData)
# plt.show()


class Model:
    def __init__(self, name):
        self.name = name
        self.paths = {}

    def __repr__(self):
        s = '\n<Model: {}\n'.format(self.name)
        s += 'Data: {}>'.format(self.paths)
        return s

    def insert_paths(self, inputPath, outputPath):
        self.paths[inputPath] = outputPath
