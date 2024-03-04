import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

csub1 = 8.176

frequencies = {
    8.176: 'C',
    8.662: 'C#',
    9.177: 'D',
    9.723: 'D#',
    10.301: 'E',
    10.913: 'F',
    11.562: 'F#',
    12.25: 'G',
    12.25: 'G#',
    13.75: 'A',
    14.568: 'A#',
    15.434: 'B'
}


def synthetic_data(
        sampleRate=44000,
        frequencies=(55,),
        duration=1,
):
    numberOfSamples = int(duration * sampleRate)
    data = np.zeros((numberOfSamples, 2), float)
    for frequency in frequencies:
        amplitude = 1  # np.random.random()
        frequencyArray = np.fromfunction(lambda x, y: amplitude * np.cos(2 * np.pi * frequency * x / sampleRate),
                                         data.shape)
        data += frequencyArray
    return data


def read_data(sampleRate=44000, filepath=''):
    if filepath:
        print('reading ' + filepath)
        return ''
    else:
        return synthetic_data(sampleRate)
    return data


def identify_pitch(data):
    pass


sampleRate = 250
duration = 1  ##np.random.random(minSeconds, maxSeconds)
data = synthetic_data(
    sampleRate=sampleRate,
    duration=duration
)
data = np.average(data, axis=1)
# plt.plot0(data)
# plt.xlabel('time (s)')
# plt.ylabel('amplitude (%/100)')
# seconds = np.round_(plt.xticks()[0] / sampleRate, 1)
# plt.xticks(plt.xticks()[0], seconds)
# plt.xlim(0, max(plt.xticks()[0]))
N = sampleRate
T = 1.0 / sampleRate
x = np.linspace(0.0, N, 1)
xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
yf = np.fft.fft(data)
print(scipy.stats.mode(yf))
plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
plt.show()
