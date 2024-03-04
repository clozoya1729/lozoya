import os

import numpy as np
from keras import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout
# from tensorflow import set_random_seed

import settings as gc


# set_random_seed(1)


def get_files(path: str, fullPath=True):
    # Returns all of the files in path.
    x = []
    for f in next(os.walk(path))[2]:
        if fullPath:
            x.append(os.path.join(path, f))
        else:
            x.append(f)
    return x


def path_end(path):
    # Returns last part of path. E.g. path='dir1/dir2/file.ext' returns 'file.ext'.
    return os.path.basename(os.path.normpath(path))


def strip_ext(filename):
    # Removes extension from filename.
    return filename.split('.')[0]


def encode(item, value):
    s = np.zeros(gc.el[item])
    s[int(value)] = 1
    return s.reshape(1, -1)


def munge(dataframe, year):
    result = dataframe.dropna(axis=0).rename(columns=gc.con)
    # result['Year Built'] = year - result['Year Built']
    # result['Year Reconstructed'] = year - result['Year Reconstructed']
    # result['ADT'] = result['ADT'] / (result['Deck Width (m)'] * result['Max Span Length (m)'])
    # result = result.drop(['Deck Width (m)', 'Max Span Length (m)'], axis=1)
    result = result.rename(
        columns={
            # 'Year Built': 'Age',
            # 'Year Reconstructed': 'Last Repair',
            # 'ADT': 'Capacity',
        }
    )
    return result


def kR(inputSize):
    model = Sequential()
    print(inputSize)
    model.add(
        Dense(
            units=64, activation='relu',
            input_shape=(inputSize,),
            kernel_initializer='lecun_normal',
            bias_initializer='lecun_normal',
            kernel_regularizer=regularizers.l2(0.01),
            bias_regularizer=regularizers.l2(0.01)
            )
        )
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=10, activation='softmax'))
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy'
        )
    return model


'''
Using State:
======= Epoch 0 =======
 Training Accuracy: 0.8526983178562834
 Testing Accuracy: 0.9186041086730012
========================

Using State and Age:
======= Epoch 0 =======
 Training Accuracy: 0.8502301374183936
 Testing Accuracy: 0.9182879982581378
========================


Using State and Last Repair:
======= Epoch 0 =======
 Training Accuracy: 0.4428871031375236
 Testing Accuracy: 0.5923656047784968
========================


Using State, Age, and Last Repair:
======= Epoch 0 =======
 Training Accuracy: 0.45730185427438275
 Testing Accuracy: 0.5970476605276257
========================

Using State and ADT / Area:
======= Epoch 0 =======
 Training Accuracy: 0.0014651011251810625
 Testing Accuracy: 0.0011819228017791049
========================

Using State and ADT:
======= Epoch 0 =======
 Training Accuracy: 0.5571867829089053
 Testing Accuracy: 0.6829230362502527
========================

Using State, Age, and ADT:
======= Epoch 0 =======
 Training Accuracy: 0.578690252851427
 Testing Accuracy: 0.663120732826327
========================

Using State, Lat, and Long:
======= Epoch 0 =======
 Training Accuracy: 0.37460270039252785
 Testing Accuracy: 0.19552464092868369
========================
'''
