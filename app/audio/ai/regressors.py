import json
import math
import os
import random
import shlex
import subprocess
import sys
import wave
from collections import OrderedDict
from functools import reduce
from subprocess import PIPE, Popen

import ai_models as ai
import config.settings as gc
import keras
import keras.backend as K
import numpy as np
import plotly
import plotly.graph_objs as go
import processor as dp
import scipy
import scipy.fftpack
import sklearn
import utils.ai
import wavio
from config.util import get_files, get_subdirs, path_end, strip_ext
from keras.layers import Dense, Dropout
from keras.models import load_model
from matplotlib import pyplot as plt
from processor import noise, scale, unscale
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier, MLPRegressor
from utils.ai import evaluate_classification, evaluate_regression
from utils.producer import get_next_dir, update_transformations
from utils.trainer import evaluate, get_data, get_sequences, get_transformations, permute
from utils.util import create_ref, homogenize_length

import ai
import configuration as gc
import util
from util import create_reference, get_files, get_subdirs, max_length, path_end, plot, print_info, printio, read, \
    standardize_string

if gc.multiModel:
    MODELS = {model: regressor() if model != 'Sequencer' else classifier() for model in gc.models}
else:
    MODELS = {model: regressor() if model != 'Sequencer' else classifier() for model in gc.models}
from utils.ai import activations, get_label
from utils.util import attempt

np.random.seed(3)
layersDict = {'dense': Dense, 'dropout': Dropout}
regularizersDict = {'L1': keras.regularizers.l1, 'L2': keras.regularizers.l2, }
activations = {
    'gauss':  gauss, 'laplace': laplace, 'lin': lin, 'cublin': cublin, 'fuzz': fuzz, 'sinlin': sinlin,
    'sinu':   sinu, 'rsin': rsin, 'sigmund': sigmund, 'sinerelu': sinerelu, 'swish': swish, 'xsin': xsin,
    'tanner': tanner, 'wavlin': wavlin, 'wavelet': wavelet,
}
custom_objects = {
    'lin':      lin, 'cublin': cublin, 'fuzz': fuzz, 'sinlin': sinlin, 'sinu': sinu, 'rsin': rsin,
    'sinerelu': sinerelu, 'swish': swish, 'xsin': xsin, 'tanner': tanner, 'wavlin': wavlin,
}
MODELS = {model: regressor() if model != 'Sequencer' else classifier() for model in gc.models}
regressor = ai.Model()
regressor = ai.Regressor()
regressor = regressors.Regressor()
run()


def add_noise(data):
    a = random.randint(-3, 4)
    b = random.randint(20, 41) / 10
    mean = random.randint(-9000, 9001)
    std = random.randint(1, 40001) / 10
    scale = (1 + math.tanh(a) / b)
    noise = np.random.normal(mean, std, data.shape)
    return scale * data + noise


class TimeRegressors():
    def __init__(self):
        self.model = None

    def set_model(self, model):
        """
        model: str, name of model
        """
        self.filename = os.path.join('Models', '{}.joblib'.format(model))
        if os.path.isfile(self.filename):
            self.model = self.load()
        else:
            self.model = ai.MODELS[model]

    def train(self, generator, length):
        for i in range(length):
            x, y = next(generator)
            util.printio(x, y, i)
            self.partial(x, y)

    def partial(self, x, y):
        if type(self.model) == MLPClassifier:
            self.model.partial_fit(x, y, classes=[i for i in range(len(y[0]))])
        else:
            self.model.partial_fit(x, y)

    def save(self):
        if os.path.isfile(self.filename):
            joblib.dump(self.model, self.filename + '_tmp')
            os.remove(self.filename)
            os.rename(self.filename + '_tmp', self.filename)

    def load(self):
        return joblib.load(self.filename)

    def close(self):
        self.model = None
        del self.model

    def run(self, model, generator, maxLength):
        self.set_model(model)
        self.train(generator, maxLength)
        self.save()
        self.close()


class Regressor(MLModel):
    def __init__(self):
        MLModel.__init__(self)

    def train(self, generator, length):
        for i in range(length):
            x, y = next(generator)
            printio(x, y, i)
            self.partial(x, y)

    def partial(self, x, y):
        if type(self.model) == RandomForestClassifier:
            self.model.partial_fit(x, y, classes=[i for i in range(len(y[0]))])
        else:
            self.model.partial_fit(x, y)

    def run(self, model, generator, maxLength):
        self.set_model(model)
        # self.train(generator, maxLength)
        self.save()
        self.close()

    def train_transformation(self, metadata, maxLength, sequencer=False):
        print(metadata)
        if sequencer:
            model = 'Sequencer'
        else:
            model = ai.TRANSFORMATIONS2[standardize_string(metadata['model'])]
        generator = dp.training_data_generator(metadata=metadata, sequencer=sequencer)
        self.run(model, generator, maxLength)


def evaluate_classification(input, output):
    return {  # 'report': sklearn.metrics.classification_report(input, output, output_dict=True),
        'reward': 100 * sklearn.metrics.accuracy_score(input, output),
    }


def circle(x):  # fail
    return K.sqrt(-K.pow(x, 2) + 1)


def cublin(x):
    return K.identity(x) + K.pow(x, 3) / (10 ** 9.7)


def gauss(x):
    return K.exp(-K.pow(x, 2) / 0.132)


def laplace(x):
    return K.exp(-K.abs(x) / 0.11)


def lin(x):
    return gc.maxVal * K.identity(x)


def xsin(x):
    return K.identity(x) * K.sin(x)


def sinlin(x):
    return K.identity(x) + K.sin(500 * x)


def sinu(x):
    return K.sin(x)


def rsin(x):
    return lin(K.sin(x))


def sigmund(x):
    return 1.6495 * K.exp(-K.pow(x, 2) / 2) * x


def sinerelu(x):
    m = 0.0025 * (K.sin(x) - K.cos(x))
    return K.maximum(m, x)


def swash(x):  # fail
    return x * circle(x)


def swish(x):
    return x * K.sigmoid(x)  # / 100000)


def tanner(x):
    return gc.maxVal * K.tanh(x / gc.maxVal)


def get_model_info(settings):
    cls = ''
    ccs = ''
    rls = ''
    rcs = ''
    if gc.aiLib == 'keras':
        # class
        for i, layer in enumerate(settings['c']['layers']):
            cls += '\t\tLayer {}\n'.format(i)
            for setting in sorted(layer):
                cls += '\t\t\t{}: {}\n'.format(setting, layer[setting])
        for setting in sorted(settings['r']['compileSettings']):
            ccs += '\t\t{}: {}\n'.format(setting, settings['c']['compileSettings'][setting])
        # regre
        for i, layer in enumerate(settings['r']['layers']):
            rls += '\t\tLayer {}\n'.format(i)
            for setting in sorted(layer):
                rls += '\t\t\t{}: {}\n'.format(setting, layer[setting])
        for setting in sorted(settings['r']['compileSettings']):
            rcs += '\t\t{}: {}\n'.format(setting, settings['r']['compileSettings'][setting])
        regressorSettings = 'o Regressor Settings:\n\to Layers: \n{}\n\to Compile Settings:\n{}\n'.format(rls, rcs)
        classifierSettings = 'o Classifier Settings:\n\to Layers: \n{}\n\to Compile Settings:\n{}\n'.format(cls, ccs)

    if gc.aiLib == 'sklearn':
        # class
        for setting in settings['c']:
            cls += '\t\t{}: {}\n'.format(setting, settings['c'][setting])
        # regre
        for setting in settings['r']:
            rls += '\t\t{}: {}\n'.format(setting, settings['r'][setting])
        regressorSettings = 'o Regressor Settings:\n{}\n'.format(rls)
        classifierSettings = 'o Classifier Settings:\n{}\n'.format(cls)
    sss = '======== MODELS SETTINGS ========\n' \
          '{}' \
          '{}' \
          '===================================\n'.format(classifierSettings, regressorSettings)
    print(sss)
    return sss


def get_label(vector):
    for key in gc.labels:
        if np.array_equal(gc.labels[key], vector):
            return key


def evaluate_regression(input, output, prediction):
    def cos_sim(x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    monoinput = input.sum(axis=1) / 2
    monooutput = output.sum(axis=1) / 2
    monoprediction = prediction.sum(axis=1) / 2
    op_euclidian_dist = np.linalg.norm(monooutput - monoprediction)
    op_cos_sim = cos_sim(monooutput, monoprediction)
    rloss = gc.eP * op_euclidian_dist + gc.cP * 1 - abs(op_cos_sim)
    try:
        return {
            'io_corr':           np.dot(monoinput, monooutput.T), 'op_corr': np.dot(monooutput, monoprediction.T),
            'ip_corr':           np.dot(monoinput, monoprediction.T),
            'io_euclidian_dist': np.linalg.norm(monoinput - monooutput), 'op_euclidian_dist': op_euclidian_dist,
            'ip_euclidian_dist': np.linalg.norm(monoinput - monoprediction),
            'io_cos_sim':        cos_sim(monoinput, monooutput), 'op_cos_sim': op_cos_sim,
            'ip_cos_sim':        cos_sim(monoinput, monoprediction),
            'evs':               sklearn.metrics.explained_variance_score(output, prediction),
            'mse':               sklearn.metrics.mean_squared_error(output, prediction),
            'monomse':           sklearn.metrics.mean_squared_error(monooutput, monoprediction),
            'msle':              sklearn.metrics.mean_squared_log_error(np.abs(output), np.abs(prediction)),
            'monomsle':          sklearn.metrics.mean_squared_log_error(np.abs(monooutput), np.abs(monoprediction)),
            'monomae':           sklearn.metrics.mean_absolute_error(monooutput, monoprediction),
            'monomedae':         sklearn.metrics.median_absolute_error(monooutput, monoprediction),
            'r2':                sklearn.metrics.r2_score(output, prediction), 'rloss': rloss,
        }
    except Exception as e:
        print(e)
        print(input)


def evaluate_classification(input, output):
    return {  # 'report': sklearn.metrics.classification_report(input, output, output_dict=True),
        'as': 100 * sklearn.metrics.accuracy_score(input, output),
    }


def circle(x):  # fail
    return K.sqrt(-K.pow(x, 2) + 1)


def cublin(x):
    return K.identity(x) + K.pow(x, 3) / (10 ** 9.7)


def gauss(x):
    return K.exp(-K.pow(x, 2) / 0.132)


def laplace(x):
    return K.exp(-K.abs(x) / 0.11)


def lin(x):
    return gc.maxVal * K.identity(x)


def xsin(x):
    return K.identity(x) * K.sin(x)


def fuzz(x):
    return 0.1 * K.pow(x, 3) + 0.9 * K.identity(x) + 0.01 * K.sin(311.02 * x)


def sinlin(x):
    return K.identity(x) + K.sin(500 * x)


def sinu(x):
    return K.sin(x)


def rsin(x):
    return lin(K.sin(x))


def sigmund(x):
    return 1.6495 * K.exp(-K.pow(x, 2) / 2) * x


def sinerelu(x):
    m = 0.0025 * (K.sin(x) - K.cos(x))
    return K.maximum(m, x)


def swash(x):  # fail
    return x * circle(x)


def swish(x):
    return x * K.sigmoid(x / 100000)


def tanner(x):
    return gc.maxVal * K.tanh(x / gc.maxVal)


def wavelet(x):
    return 1.03 * K.sin(x) * K.exp(-K.pow(x, 2) / 0.2)


def get_model_info(settings):
    cls = ''
    ccs = ''
    rls = ''
    rcs = ''
    if gc.aiLib == 'keras':
        # class
        for i, layer in enumerate(settings['c']['layers']):
            cls += '\t\tLayer {}\n'.format(i)
            for setting in sorted(layer):
                cls += '\t\t\t{}: {}\n'.format(setting, layer[setting])
        for setting in sorted(settings['r']['compileSettings']):
            ccs += '\t\t{}: {}\n'.format(setting, settings['c']['compileSettings'][setting])
        # regre
        for i, layer in enumerate(settings['r']['layers']):
            rls += '\t\tLayer {}\n'.format(i)
            for setting in sorted(layer):
                rls += '\t\t\t{}: {}\n'.format(setting, layer[setting])
        for setting in sorted(settings['r']['compileSettings']):
            rcs += '\t\t{}: {}\n'.format(setting, settings['r']['compileSettings'][setting])
        regressorSettings = 'o Regressor Settings:\n\to Layers: \n{}\n\to Compile Settings:\n{}\n'.format(rls, rcs)
        classifierSettings = 'o Classifier Settings:\n\to Layers: \n{}\n\to Compile Settings:\n{}\n'.format(cls, ccs)

    if gc.aiLib == 'sklearn':
        # class
        for setting in settings['c']:
            cls += '\t\t{}: {}\n'.format(setting, settings['c'][setting])
        # regre
        for setting in settings['r']:
            rls += '\t\t{}: {}\n'.format(setting, settings['r'][setting])
        regressorSettings = 'o Regressor Settings:\n{}\n'.format(rls)
        classifierSettings = 'o Classifier Settings:\n{}\n'.format(cls)
    sss = '======== MODELS SETTINGS ========\n' \
          '{}' \
          '{}' \
          '===================================\n'.format(classifierSettings, regressorSettings)
    print(sss)
    return sss


def get_label(vector):
    for key in gc.labels:
        if np.array_equal(gc.labels[key], vector):
            return key


def get_label(vector):
    for key in gc.labels:
        if np.array_equal(gc.labels[key], vector):
            return key


# CUSTOM ACTIVATION FUNCTIONS

def xsin(x):
    return keras.backend.identity(x) * keras.backend.sin(x)


def lin(x):
    return gc.maxVal * keras.backend.identity(x)


def cublin(x):
    return keras.backend.identity(x) + keras.backend.pow(x, 3) / (10 ** 9.7)


def fuzz(x):
    a = 0.1 * keras.backend.pow(x, 3)
    b = 0.9 * keras.backend.identity(x)
    c = 0.01 * keras.backend.sin(311.02 * x)
    return a + b + c


def sinlin(x):
    return keras.backend.identity(x) + keras.backend.sin(500 * x)


def sinu(x):
    return keras.backend.sin(x)


def ripjaw(x):
    return 0.83333333 * keras.backend.identity(x) + 0.25 * keras.backend.sin(62.8 * x) + 10000 * (
        keras.backend.sigmoid(x / 1000) - 0.5)


def rsin(x):
    return lin(keras.backend.sin(x))


def sinerelu(x):
    m = 0.0025 * (keras.backend.sin(x) - keras.backend.cos(x))
    return keras.backend.maximum(m, x)


def swish(x):
    return x * keras.backend.sigmoid(x / 100000)


def tanner(x):
    return gc.maxVal * keras.backend.tanh(x / gc.maxVal)


def wavlin(x):
    sin = keras.backend.sin(x)
    cos = keras.backend.cos(x)
    e = keras.backend.exp
    x2 = keras.backend.pow(x, 2)
    return keras.backend.identity(x) + 0.5 * sin * cos  # + 1000 * e(-x2 / 100000000) - e(-x2 / 500000000)


def attempt(func, args):
    # Prevent crash due to simultaneous access of file
    done = False
    while not done:
        try:
            r = func(*args)
            done = True
            return r
        except:
            pass


def regressor():
    if gc.multiModel:
        inputShape = (gc.inputSize + len(gc.models),)
    else:
        inputShape = (gc.inputSize,)
    if gc.aiBackend == 'sklearn':
        return MLPRegressor(
            **{
                'hidden_layer_sizes': (200,), 'activation': 'identity', 'solver': 'adam', 'alpha': 1,
                'batch_size':         'auto', 'learning_rate': 'constant', 'learning_rate_init': 1,
                'power_t':            0.5, 'max_iter': 1, 'shuffle': True, 'random_state': None, 'tol': 1,
                'verbose':            False, 'warm_start': False, 'momentum': 0.9, 'nesterovs_momentum': True,
                'early_stopping':     False, 'validation_fraction': 0.1, 'beta_1': 0.9, 'beta_2': 0.999,
                'epsilon':            1e-08,  # 'n_iter_no_change': 10
            }
        )

    if gc.aiBackend == 'keras':
        model = keras.Sequential()
        model.add(
            Dense(
                units=16, activation=fuzz, input_shape=inputShape,  # kernel_initializer='lecun_normal',
                # bias_initializer='lecun_normal',
                # kernel_regularizer=keras.regularizers.l2(0.01),
                # bias_regularizer=keras.regularizers.l2(0.01),
                # activity_regularizer=keras.regularizers.l2(0.01)
            )
        )
        model.add(
            Dense(
                units=16, activation='tanh',  # kernel_initializer='lecun_normal',
                # bias_initializer='lecun_normal',
                # kernel_regularizer=keras.regularizers.l2(0.01),
                # bias_regularizer=keras.regularizers.l2(0.01),
                # activity_regularizer=keras.regularizers.l2(0.01)
            )
        )
        model.add(
            Dense(
                units=2, activation='linear',  # kernel_initializer='lecun_normal',
                # bias_initializer='lecun_normal',
                # kernel_regularizer=keras.regularizers.l2(0.01),
                # bias_regularizer=keras.regularizers.l2(0.01),
                # activity_regularizer=keras.regularizers.l2(0.01)
            )
        )
        model.compile(loss='mse', optimizer='adam')
        return model


def classifier():
    inputShape = (gc.inputSize,)
    if gc.aiBackend == 'sklearn':
        return MLPClassifier(**{})

    if gc.aiBackend == 'keras':
        model = keras.Sequential()
        model.add(Dense(20, activation=swish, input_shape=inputShape))
        model.add(Dense(len(gc.models), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        return model


class Model:
    def __init__(self):
        self.model = None

    def set_model(self, name: str, model='', mode=''):
        if mode == 'train':
            print('>> Training {} <<'.format(name))
        elif mode == 'eval':
            print('>> Evaluating {} <<'.format(name))
        else:
            print('>> Running {} <<'.format(name))

        self.mode = mode
        if gc.multiModel or name == 'Sequencer':
            self.name = name
        else:
            self.name = 'Model'
        self.filename = os.path.join(gc.modelsDir[gc.aiLib], '{0}.{1}'.format(self.name, gc.modelExt[gc.aiLib]))
        if os.path.isfile(self.filename):
            self.model = self.load()
        else:
            self.model = model

    def save(self):
        if os.path.isfile(self.filename):
            if gc.aiLib == 'sklearn':
                attempt(joblib.dump, [self.model, '{}_tmp'.format(self.filename)])
            if gc.aiLib == 'keras':
                attempt(self.model.save, ['{}_tmp'.format(self.filename)])
            attempt(os.remove, [self.filename])
            attempt(os.rename, ['{}_tmp'.format(self.filename), self.filename])
        else:
            if gc.aiLib == 'sklearn':
                attempt(joblib.dump, [self.model, self.filename])
            if gc.aiLib == 'keras':
                attempt(self.model.save, [self.filename])

    def load(self):
        if gc.aiLib == 'sklearn':
            return attempt(joblib.load, [self.filename])
        if gc.aiLib == 'keras':
            return attempt(load_model, [self.filename, activations])

    def close(self):
        self.model = None
        del self.model

    def train(self, data):
        generator = dp.get_gen(data=data, sequencer=self.name == 'Sequencer', training=True)
        for i in range(0, data['length'], gc.batchSize):
            x, y = next(generator)
            for j in range(gc.batchSize - 1):
                x1, y1 = next(generator)
                x = np.vstack([x, x1])
                y = np.vstack([y, y1])
                i += 1
                if i >= data['length']:
                    break
            # printio(x, y, i - (gc.batchSize - 1))
            if gc.aiLib == 'sklearn':
                if self.name == 'Sequencer':
                    self.model.partial_fit(x, y, classes=[i for i in range(len(y[0]))])
                else:
                    self.model.partial_fit(x, y)
            if gc.aiLib == 'keras':
                self.model.train_on_batch(x, y)
        self.save()
        self.close()

    def predict(self, data):
        generator = dp.get_gen(data=data, sequencer=self.name == 'Sequencer')
        if self.name == 'Sequencer':
            bo = []
            result = np.zeros((1, len(gc.labels)))
        else:
            result = []
        for i in range(data['length']):
            x = next(generator)
            if gc.aiLib == 'sklearn':
                m = self.model.predict(x)
            if gc.aiLib == 'keras':
                m = self.model.predict_on_batch(x)
            if self.name == 'Sequencer':
                if self.mode == 'eval':
                    z = np.zeros_like(m)
                    z[np.arange(len(m)), m.argmax(1)] = 1
                    bo.append(z[0])
                result = np.add(result, m)
            else:
                result.append(m)
        if self.name == 'Sequencer':
            r = np.zeros_like(result)
            r[np.arange(len(result)), result.argmax(1)] = 1
            label = get_label(r[0])
            return label, r[0], np.array(bo).astype(np.int8)
        return np.vstack(result)


def get_label(vector):
    for key in gc.labels:
        if np.array_equal(gc.labels[key], vector):
            return key


def regressor():
    if gc.multiModel:
        inputShape = (gc.inputSize,)
    else:
        inputShape = (gc.inputSize + len(gc.models),)
    if gc.aiBackend == 'sklearn':
        return MLPRegressor(
            **{
                'hidden_layer_sizes': (200,), 'activation': 'tanh', 'solver': 'adam', 'alpha': 1, 'batch_size': 'auto',
                'learning_rate':      'constant', 'learning_rate_init': 1, 'power_t': 0.5, 'max_iter': 1,
                'shuffle':            True,
                'random_state':       None, 'tol': 1, 'verbose': False, 'warm_start': False, 'momentum': 0.9,
                'nesterovs_momentum': True, 'early_stopping': False, 'validation_fraction': 0.1, 'beta_1': 0.9,
                'beta_2':             0.999, 'epsilon': 1e-08,  # 'n_iter_no_change': 10
            }
        )

    def lin(x):
        return keras.backend.identity(x) * gc.maxVal

    if gc.aiBackend == 'keras':
        model = keras.Sequential()
        model.add(keras.layers.Dense(256, activation='linear', input_shape=inputShape))
        model.add(keras.layers.Dense(256, activation='tanh', input_shape=inputShape))
        model.add(keras.layers.Dense(2, activation=lin))
        model.compile(loss='mse', optimizer='sgd')
        return model


def classifier():
    if gc.aiBackend == 'sklearn':
        return MLPClassifier(**{})
    if gc.aiBackend == 'keras':
        model = keras.Sequential()
        model.add(keras.layers.Dense(256, activation='relu', input_shape=(gc.inputSize,)))
        model.add(keras.layers.Dense(len(gc.models), activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        return model


class Model:
    def __init__(self):
        self.model = None

    def set_model(self, name: str):
        print('>> Running {} <<'.format(name))
        if gc.multiModel or name == 'Sequencer':
            self.name = name
        else:
            self.name = 'Model'
        self.filename = os.path.join(gc.modelsDir[gc.aiBackend], '{0}.{1}'.format(self.name, gc.modelExt[gc.aiBackend]))
        if os.path.isfile(self.filename):
            self.model = self.load()
        else:
            self.model = MODELS[name]

    def save(self):
        if os.path.isfile(self.filename):
            if gc.aiBackend == 'sklearn':
                joblib.dump(self.model, '{}_tmp'.format(self.filename))
            if gc.aiBackend == 'keras':
                self.model.save('{}_tmp'.format(self.filename))
            os.remove(self.filename)
            os.rename('{}_tmp'.format(self.filename), self.filename)
        else:
            if gc.aiBackend == 'sklearn':
                joblib.dump(self.model, self.filename)
            if gc.aiBackend == 'keras':
                self.model.save(self.filename)

    def load(self):
        if gc.aiBackend == 'sklearn':
            return joblib.load(self.filename)
        if gc.aiBackend == 'keras':
            return load_model(self.filename)

    def close(self):
        self.model = None
        del self.model

    def train(self, generator, length):
        for i in range(0, length, gc.batchSize):
            x, y = next(generator)
            for j in range(gc.batchSize - 1):
                x1, y1 = next(generator)
                x = np.vstack([x, x1])
                y = np.vstack([y, y1])
                i += 1
                if i >= length:
                    break
            # printio(x, y, i - (gc.batchSize - 1))
            self.partial(x, y)
        self.save()
        self.close()


def regressor():
    if gc.multiModel:
        inputShape = (gc.inputSize,)
    else:
        inputShape = (gc.inputSize + len(gc.models),)
    if gc.aiBackend == 'sklearn':
        return MLPRegressor(
            **{
                'hidden_layer_sizes': (200,), 'activation': 'tanh', 'solver': 'adam', 'alpha': 1, 'batch_size': 'auto',
                'learning_rate':      'constant', 'learning_rate_init': 1, 'power_t': 0.5, 'max_iter': 1,
                'shuffle':            True,
                'random_state':       None, 'tol': 1, 'verbose': False, 'warm_start': False, 'momentum': 0.9,
                'nesterovs_momentum': True, 'early_stopping': False, 'validation_fraction': 0.1, 'beta_1': 0.9,
                'beta_2':             0.999, 'epsilon': 1e-08,  # 'n_iter_no_change': 10
            }
        )
    if gc.aiBackend == 'keras':
        model = keras.Sequential()
        model.add(keras.layers.Dense(8, activation='linear', input_shape=inputShape))
        model.add(keras.layers.Dropout(rate=0.1))
        model.add(keras.layers.Dense(2, activation='linear'))
        model.compile(loss='mae', optimizer='adam')
        return model


def classifier():
    if gc.aiBackend == 'sklearn':
        return MLPClassifier(**{})
    if gc.aiBackend == 'keras':
        model = keras.Sequential()
        model.add(keras.layers.Dense(8, activation='relu', input_shape=(gc.inputSize,)))
        model.add(keras.layers.Dropout(rate=0.1))
        model.add(keras.layers.Dense(len(gc.models), activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        return model


def get_label(vector):
    for key in gc.labels:
        if np.array_equal(gc.labels[key], vector):
            return key


def regressor():
    if gc.multiModel:
        inputShape = (gc.inputSize,)
    else:
        inputShape = (gc.inputSize + len(gc.models),)
    if gc.aiBackend == 'sklearn':
        return MLPRegressor(
            **{
                'hidden_layer_sizes': (200,), 'activation': 'tanh', 'solver': 'adam', 'alpha': 1, 'batch_size': 'auto',
                'learning_rate':      'constant', 'learning_rate_init': 1, 'power_t': 0.5, 'max_iter': 1,
                'shuffle':            True,
                'random_state':       None, 'tol': 1, 'verbose': False, 'warm_start': False, 'momentum': 0.9,
                'nesterovs_momentum': True, 'early_stopping': False, 'validation_fraction': 0.1, 'beta_1': 0.9,
                'beta_2':             0.999, 'epsilon': 1e-08,  # 'n_iter_no_change': 10
            }
        )
    if gc.aiBackend == 'keras':
        model = keras.Sequential()
        model.add(keras.layers.Dense(256, activation='linear', input_shape=inputShape))
        model.add(keras.layers.Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model


def get_label(vector):
    for key in gc.LABELS:
        if np.array_equal(gc.LABELS[key], vector):
            return key


class Regressor:
    def __init__(self):
        self.model = None

    def set_model(self, name: str):
        self.name = name
        self.filename = os.path.join(gc.modelsDir[gc.aiBackend], '{0}.{1}'.format(name, gc.modelExt[gc.aiBackend]))
        if os.path.isfile(self.filename):
            self.model = self.load()
        else:
            self.model = MODELS[name]

    def save(self):
        if os.path.isfile(self.filename):
            if gc.aiBackend == 'sklearn':
                joblib.dump(self.model, '{}_tmp'.format(self.filename))
            if gc.aiBackend == 'keras':
                self.model.save('{}_tmp'.format(self.filename))
            os.remove(self.filename)
            os.rename('{}_tmp'.format(self.filename), self.filename)
        else:
            if gc.aiBackend == 'sklearn':
                joblib.dump(self.model, self.filename)
            if gc.aiBackend == 'keras':
                self.model.save(self.filename)

    def load(self):
        if gc.aiBackend == 'sklearn':
            return joblib.load(self.filename)
        if gc.aiBackend == 'keras':
            return load_model(self.filename)

    def close(self):
        self.model = None
        del self.model

    def train(self, generator, length):
        for i in range(0, length, gc.batchSize):
            x, y = next(generator)
            for j in range(gc.batchSize - 1):
                x1, y1 = next(generator)
                x = np.vstack([x, x1])
                y = np.vstack([y, y1])
                i += 1
                if i >= length:
                    break
            printio(x, y, i - (gc.batchSize - 1))
            self.partial(x, y)
        self.save()
        self.close()

    def train_transformation(self, metadata):
        self.set_model(gc.TRANSFORMATIONS2[standardize_string(metadata['model'])])
        self.train(dp.training_data_generator(metadata=metadata), metadata['maxLength'])
        self.set_model('Sequencer')
        k = metadata['output']
        metadata['output'] = gc.LABELS[gc.TRANSFORMATIONS2[standardize_string(metadata['model'])]]
        metadata['model'] = 'sequencer'
        self.train(dp.training_data_generator(metadata=metadata, sequencer=True), metadata['maxLength'])
        if metadata['last']:
            self.set_model('Sequencer')
            metadata['output'] = k
            metadata['reference'] = dp.update_reference(metadata)
            metadata['output'] = gc.LABELS['Stop']
            self.train(dp.training_data_generator(metadata=metadata, sequencer=True), metadata['maxLength'])
        metadata['output'] = k

    def predict(self, data, d):
        generator = dp.get_generator(data=data, d=d)
        if self.name == 'Sequencer':
            result = np.zeros((1, len(gc.LABELS)))
        else:
            result = []
        for i in range(d['maxLength']):
            x = next(generator)
            if gc.aiBackend == 'sklearn':
                m = self.model.predict(x)
            if gc.aiBackend == 'keras':
                m = self.model.predict_on_batch(x)
            if self.name == 'Sequencer':
                result = np.add(result, m)
            else:
                result.append(m)
        if self.name == 'Sequencer':
            r = np.zeros_like(result)
            r[np.arange(len(result)), result.argmax(1)] = 1
            label = get_label(r[0])
            return label
        else:
            return np.vstack(result)


def regressor():
    if gc.aiBackend == 'sklearn':
        return MLPRegressor(
            **{
                'hidden_layer_sizes': (100,), 'activation': 'identity', 'solver': 'adam', 'alpha': 1,
                'batch_size':         'auto', 'learning_rate': 'constant', 'learning_rate_init': 1,
                'power_t':            0.5, 'max_iter': 1, 'shuffle': True, 'random_state': None, 'tol': 1,
                'verbose':            False, 'warm_start': False, 'momentum': 0.9, 'nesterovs_momentum': True,
                'early_stopping':     False, 'validation_fraction': 0.1, 'beta_1': 0.9, 'beta_2': 0.999,
                'epsilon':            1e-08,  # 'n_iter_no_change': 10
            }
        )
    if gc.aiBackend == 'keras':
        model = keras.Sequential()
        model.add(keras.layers.Dense(4, activation='linear', input_shape=gc.inputShape))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(2, activation='linear'))
        model.compile(loss='mae', optimizer='adam')
        return model


def classifier():
    if gc.aiBackend == 'sklearn':
        return MLPClassifier(**{})
    if gc.aiBackend == 'keras':
        model = keras.Sequential()
        model.add(keras.layers.Dense(4, activation='relu', input_shape=gc.inputShape))
        model.add(keras.layers.Dense(len(gc.models), activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        return model


def get_label(vector):
    for key in gc.LABELS:
        if np.array_equal(gc.LABELS[key], vector):
            return key


class Regressor(MLModel):
    def __init__(self):
        MLModel.__init__(self)

    def train(self, generator, length):
        for i in range(length):
            x, y = next(generator)
            printio(x, y, i)
            self.partial(x, y)
        self.save()
        self.close()
        pass

    def partial(self, x, y):
        if type(self.model) == MLPClassifier:
            self.model.partial_fit(x, y, classes=[i for i in range(len(y[0]))])
        else:
            self.model.partial_fit(x, y)

    def train_transformation(self, metadata):
        model = gc.TRANSFORMATIONS2[standardize_string(metadata['model'])]
        self.set_model(model)
        generator = dp.training_data_generator(metadata=metadata)
        self.train(generator, metadata['maxLength'])

        self.set_model('Sequencer')
        k = metadata['output']
        metadata['output'] = gc.LABELS[gc.TRANSFORMATIONS2[standardize_string(metadata['model'])]]
        metadata['model'] = 'sequencer'
        generator = dp.training_data_generator(metadata=metadata, sequencer=True)
        self.train(generator, metadata['maxLength'])
        if metadata['last']:
            self.set_model('Sequencer')
            metadata['output'] = k
            metadata['reference'] = dp.update_reference(metadata)
            metadata['output'] = gc.LABELS['Stop']
            generator = dp.training_data_generator(metadata=metadata, sequencer=True)
            self.train(generator, metadata['maxLength'])

        metadata['output'] = k


class Regressor(MLModel):
    def __init__(self):
        MLModel.__init__(self)

    def train(self, generator, length):
        for i in range(length):
            x, y = next(generator)
            printio(x, y, i)
            self.partial(x, y)
        self.save()
        self.close()
        pass

    def partial(self, x, y):
        if type(self.model) == MLPClassifier:
            self.model.partial_fit(x, y, classes=[i for i in range(len(y[0]))])
        else:
            self.model.partial_fit(x, y)

    def train_transformation(self, metadata):
        model = ai.TRANSFORMATIONS2[standardize_string(metadata['model'])]
        self.set_model(model)
        generator = dp.training_data_generator(metadata=metadata)
        self.train(generator, metadata['maxLength'])

        self.set_model('Sequencer')
        k = metadata['output']
        metadata['output'] = ai.LABELS[ai.TRANSFORMATIONS2[standardize_string(metadata['model'])]]
        metadata['model'] = 'sequencer'
        generator = dp.training_data_generator(metadata=metadata, sequencer=True)
        self.train(generator, metadata['maxLength'])
        if metadata['last']:
            self.set_model('Sequencer')
            metadata['output'] = k
            metadata['reference'] = dp.update_reference(metadata)
            metadata['output'] = ai.LABELS['Stop']
            generator = dp.training_data_generator(metadata=metadata, sequencer=True)
            self.train(generator, metadata['maxLength'])

        metadata['output'] = k


def evaluate_classification(input, output):
    return {  # 'report': sklearn.metrics.classification_report(input, output, output_dict=True),
        'reward': 100 * sklearn.metrics.accuracy_score(input, output),
    }


def circle(x):  # fail
    return K.sqrt(-K.pow(x, 2) + 1)


def cublin(x):
    return K.identity(x) + K.pow(x, 3) / (10 ** 9.7)


def gauss(x):
    return K.exp(-K.pow(x, 2) / 0.132)


def laplace(x):
    return K.exp(-K.abs(x) / 0.11)


def lin(x):
    return gc.maxVal * K.identity(x)


def xsin(x):
    return K.identity(x) * K.sin(x)


def sinlin(x):
    return K.identity(x) + 0.25 * K.sin(500 * x)


def sinu(x):
    return K.sin(x)


def rsin(x):
    return lin(K.sin(x))


def sigmund(x):
    return 1.6495 * K.exp(-K.pow(x, 2) / 2) * x


def sinerelu(x):
    m = 0.0025 * (K.sin(x) - K.cos(x))
    return K.maximum(m, x)


def swash(x):  # fail
    return x * circle(x)


def swish(x):
    return x * K.sigmoid(x)  # / 100000)


def tanner(x):
    return gc.maxVal * K.tanh(x / gc.maxVal)


def get_model_info(settings):
    cls = ''
    ccs = ''
    rls = ''
    rcs = ''
    if gc.aiLib == 'keras':
        # class
        for i, layer in enumerate(settings['c']['layers']):
            cls += '\t\tLayer {}\n'.format(i)
            for setting in sorted(layer):
                cls += '\t\t\t{}: {}\n'.format(setting, layer[setting])
        for setting in sorted(settings['r']['compileSettings']):
            ccs += '\t\t{}: {}\n'.format(setting, settings['c']['compileSettings'][setting])
        # regre
        for i, layer in enumerate(settings['r']['layers']):
            rls += '\t\tLayer {}\n'.format(i)
            for setting in sorted(layer):
                rls += '\t\t\t{}: {}\n'.format(setting, layer[setting])
        for setting in sorted(settings['r']['compileSettings']):
            rcs += '\t\t{}: {}\n'.format(setting, settings['r']['compileSettings'][setting])
        regressorSettings = 'o Regressor Settings:\n\to Layers: \n{}\n\to Compile Settings:\n{}\n'.format(rls, rcs)
        classifierSettings = 'o Classifier Settings:\n\to Layers: \n{}\n\to Compile Settings:\n{}\n'.format(cls, ccs)

    if gc.aiLib == 'sklearn':
        # class
        for setting in settings['c']:
            cls += '\t\t{}: {}\n'.format(setting, settings['c'][setting])
        # regre
        for setting in settings['r']:
            rls += '\t\t{}: {}\n'.format(setting, settings['r'][setting])
        regressorSettings = 'o Regressor Settings:\n{}\n'.format(rls)
        classifierSettings = 'o Classifier Settings:\n{}\n'.format(cls)
    sss = '======== MODELS SETTINGS ========\n' \
          '{}' \
          '{}' \
          '===================================\n'.format(classifierSettings, regressorSettings)
    print(sss)
    return sss


def get_label(vector):
    for key in gc.labels:
        if np.array_equal(gc.labels[key], vector):
            return key


def evaluate_regression(input, output, prediction):
    def cos_sim(x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    monoinput = input.sum(axis=1) / 2
    monooutput = output.sum(axis=1) / 2
    monoprediction = prediction.sum(axis=1) / 2
    op_euclid_dist = np.linalg.norm(monooutput - monoprediction)
    op_cos_sim = cos_sim(monooutput, monoprediction)
    rloss = gc.eP * op_euclid_dist + gc.cP * 1 - abs(op_cos_sim)
    try:
        a = {  # 'io_corr': np.dot(monoinput, monooutput.T),
            # 'op_corr': np.dot(monooutput, monoprediction.T),
            # 'ip_corr': np.dot(monoinput, monoprediction.T),
            # 'io_euclid_dist': np.linalg.norm(monoinput - monooutput),
            'op_euclid_dist': op_euclid_dist,  # 'ip_euclid_dist': np.linalg.norm(monoinput - monoprediction),
            # 'io_cos_sim': cos_sim(monoinput, monooutput),
            'op_cos_sim':     op_cos_sim,  # 'ip_cos_sim': cos_sim(monoinput, monoprediction),
            # 'evs': sklearn.metrics.explained_variance_score(output, prediction),
            'mse':            sklearn.metrics.mean_squared_error(output, prediction),
            # 'monomse': sklearn.metrics.mean_squared_error(monooutput, monoprediction),
            # 'msle': sklearn.metrics.mean_squared_log_error(np.abs(output), np.abs(prediction)),
            # 'monomsle': sklearn.metrics.mean_squared_log_error(np.abs(monooutput), np.abs(monoprediction)),
            # 'monomae': sklearn.metrics.mean_absolute_error(monooutput, monoprediction),
            # 'monomedae': sklearn.metrics.median_absolute_error(monooutput, monoprediction),
            'r2':             sklearn.metrics.r2_score(output, prediction),  # 'rloss': rloss,
        }
        return {
            'reward': -op_euclid_dist - 100 * op_cos_sim - a['mse'] - 100 * sklearn.metrics.r2_score(
                output,
                prediction
            )
        }
    except Exception as e:
        print(e)
        print(input)


def evaluate_classification(input, output):
    return {  # 'report': sklearn.metrics.classification_report(input, output, output_dict=True),
        'reward': 100 * sklearn.metrics.accuracy_score(input, output),
    }


def circle(x):  # fail
    return K.sqrt(-K.pow(x, 2) + 1)


def cublin(x):
    return K.identity(x) + K.pow(x, 3) / (10 ** 9.7)


def gauss(x):
    return K.exp(-K.pow(x, 2) / 0.132)


def laplace(x):
    return K.exp(-K.abs(x) / 0.11)


def lin(x):
    return gc.maxVal * K.identity(x)


def xsin(x):
    return K.identity(x) * K.sin(x)


def sinlin(x):
    return K.identity(x) + 0.25 * K.sin(500 * x)


def sinu(x):
    return K.sin(x)


def ripjaw(x):
    return 0.83333333 * K.identity(x) + 0.25 * K.sin(62.8 * x) + 10000 * (K.sigmoid(x / 1000) - 0.5)


def rsin(x):
    return lin(K.sin(x))


def sigmund(x):
    return 1.6495 * K.exp(-K.pow(x, 2) / 2) * x


def sinerelu(x):
    m = 0.0025 * (K.sin(x) - K.cos(x))
    return K.maximum(m, x)


def swash(x):  # fail
    return x * circle(x)


def swish(x):
    return x * K.sigmoid(x)  # / 100000)


def tanner(x):
    return gc.maxVal * K.tanh(x / gc.maxVal)


def wavlin(x):
    x2 = K.pow(x, 2)
    return K.identity(x) + 0.5 * K.sin(x) * K.cos(x) + 1000 * K.exp(-x2 / 100000000) - K.exp(-x2 / 500000000)


def get_model_info(settings):
    cls = ''
    ccs = ''
    rls = ''
    rcs = ''
    if gc.aiLib == 'keras':
        # class
        for i, layer in enumerate(settings['c']['layers']):
            cls += '\t\tLayer {}\n'.format(i)
            for setting in sorted(layer):
                cls += '\t\t\t{}: {}\n'.format(setting, layer[setting])
        for setting in sorted(settings['r']['compileSettings']):
            ccs += '\t\t{}: {}\n'.format(setting, settings['c']['compileSettings'][setting])
        # regre
        for i, layer in enumerate(settings['r']['layers']):
            rls += '\t\tLayer {}\n'.format(i)
            for setting in sorted(layer):
                rls += '\t\t\t{}: {}\n'.format(setting, layer[setting])
        for setting in sorted(settings['r']['compileSettings']):
            rcs += '\t\t{}: {}\n'.format(setting, settings['r']['compileSettings'][setting])
        regressorSettings = 'o Regressor Settings:\n\to Layers: \n{}\n\to Compile Settings:\n{}\n'.format(rls, rcs)
        classifierSettings = 'o Classifier Settings:\n\to Layers: \n{}\n\to Compile Settings:\n{}\n'.format(cls, ccs)

    if gc.aiLib == 'sklearn':
        # class
        for setting in settings['c']:
            cls += '\t\t{}: {}\n'.format(setting, settings['c'][setting])
        # regre
        for setting in settings['r']:
            rls += '\t\t{}: {}\n'.format(setting, settings['r'][setting])
        regressorSettings = 'o Regressor Settings:\n{}\n'.format(rls)
        classifierSettings = 'o Classifier Settings:\n{}\n'.format(cls)
    sss = '======== MODELS SETTINGS ========\n' \
          '{}' \
          '{}' \
          '===================================\n'.format(classifierSettings, regressorSettings)
    print(sss)
    return sss


def get_label(vector):
    for key in gc.labels:
        if np.array_equal(gc.labels[key], vector):
            return key


def permute(files, permutation):
    if permutation + 1 > len(files):
        return files
    f = list(files)
    result = []
    fact = reduce(lambda x, y: x * y, range(1, len(f) + 1), 1)
    permutation %= fact
    while f:
        fact = int(fact / len(f))
        choice, permutation = permutation // fact, permutation % fact
        result += [f.pop(choice)]
    return result


def permute(files, permutation):
    if permutation + 1 > len(files):
        return files
    f = list(files)
    result = []
    fact = reduce(lambda x, y: x * y, range(1, len(f) + 1), 1)
    permutation %= fact
    while f:
        fact = int(fact / len(f))
        choice, permutation = permutation // fact, permutation % fact
        result += [f.pop(choice)]
    return result


def evaluate(regressor, data, projectData, MODELS):
    if gc.scale:
        data['input'] = data['input'] / gc.maxVal
        data['ref'] = data['ref'] / gc.maxVal
        data['output'] = data['output'] / gc.maxVal
    # Regressor
    projectData['input'] = data['input']
    projectData['label'] = data['label']
    regressor.set_model(utils.ai.get_label(data['label']), MODELS[data['model']], 'eval')
    rPrediction = regressor.predict(projectData)
    rError = evaluate_regression(data['input'], data['output'], rPrediction)
    print(
        ' - Explained Variance Score: {}\n'
        ' - Stereo Error\n'
        '   - R2: {} [maximize to 1]\n'
        '   - Mean Squared: {}\n'
        '   - Mean Squared Log: {}\n'
        ' - Mono Error\n'
        '   - Mean Squared: {}\n'
        '   - Mean Squared Log: {}\n'
        '   - Mean Absolute: {}\n'
        '   - Median Absolute: {}\n'
        ' - Correlation\n'
        '   - Input vs Output: {}\n'
        '   - Input vs Prediction: {}\n'
        '     - Difference: {}\n'
        '   - Output vs Prediction: {} [maximize to 1]\n'
        ' - Euclidian Distance\n'
        '   - Input vs Output: {}\n'
        '   - Input vs Prediction: {}\n'
        '     - Difference: {}\n'
        '   - Output vs Prediction: {}[minimize to 0]\n'
        ' - Cosine Similarity\n'
        '   - Input vs Output: {}\n'
        '   - Input vs Prediction: {}\n'
        '     - Difference: {}\n'
        '   - Output vs Prediction: {}[maximize to 1]\n'
        ' - Reinforcement Loss: {}\n'
        ''.format(
            rError['evs'], rError['r2'], rError['mse'], rError['msle'], rError['monomse'], rError['monomsle'],
            rError['monomae'], rError['monomedae'], rError['io_corr'], rError['ip_corr'],
            abs(rError['io_corr'] - rError['ip_corr']), rError['op_corr'], rError['io_euclidian_dist'],
            rError['ip_euclidian_dist'], abs(rError['io_euclidian_dist'] - rError['ip_euclidian_dist']),
            rError['op_euclidian_dist'], rError['io_cos_sim'], rError['ip_cos_sim'],
            abs(rError['io_cos_sim'] - rError['ip_cos_sim']), rError['op_cos_sim'], rError['rloss'], )
    )
    # Classifier
    regressor.set_model('Sequencer', MODELS['Sequencer'], 'eval')
    c = np.vstack([data['label'] for _ in range(len(data['input']))])
    _, _, cPrediction = regressor.predict(projectData)
    cError = evaluate_classification(c, cPrediction)
    print(
        ' -- Accuracy Score: {}%\n'
        ''.format(cError['as'])
    )
    if gc.scale:
        data['input'] = data['input'] * gc.maxVal
        data['ref'] = data['ref'] * gc.maxVal
        data['output'] = data['output'] * gc.maxVal
    return cError, rError


def permute(files, permutation):
    if permutation + 1 > len(files):
        return files
    f = list(files)
    result = []
    fact = reduce(lambda x, y: x * y, range(1, len(f) + 1), 1)
    permutation %= fact
    while f:
        fact = int(fact / len(f))
        choice, permutation = permutation // fact, permutation % fact
        result += [f.pop(choice)]
    return result


def get_data(model, x, y, projectData, last):
    model = gc.transfer2[standardize_string(model)]
    return {
        'model':  model, 'label': gc.labels[model], 'input': read(x, length=projectData['length']),
        'output': read(y, projectData['length'], length=projectData['length']), 'length': projectData['length'],
        'ref':    projectData['ref'], 'last': last,
    }


def get_sequences(subdirs):
    fs = get_files(subdirs[0], exclude=gc.excludedFiles, fullPath=False)
    sequences = {strip_ext(f): [] for f in fs}
    for subdir in subdirs:
        for file in get_files(subdir, exclude=gc.excludedFiles, fullPath=False):
            sequences[strip_ext(file)].append(path_end(subdir))
    return sequences


def get_transformations(project):
    def rf(path):
        with open(path) as f:
            t = f.read().split('\n')
        return t

    transformationsDir = os.path.join(project, gc.transformationsDir)
    transformationFiles = get_files(transformationsDir)
    transformations = {strip_ext(path_end(f)): rf(f) for f in transformationFiles}
    return transformations


def evaluate(regressor, data, projectData, MODELS):
    if gc.scale:
        data['input'] = data['input'] / gc.maxVal
        data['ref'] = data['ref'] / gc.maxVal
        data['output'] = data['output'] / gc.maxVal
    # Regressor
    projectData['input'] = data['input']
    projectData['label'] = data['label']
    regressor.set_model(utils.ai.get_label(data['label']), MODELS[data['model']], 'eval')
    rPrediction = regressor.predict(projectData)
    rError = evaluate_regression(data['input'], data['output'], rPrediction)
    print(  # ' - Explained Variance Score: {:.2}\n'
        ' - Stereo Error\n'
        '   - R2: {:.2} [maximize to 1]\n'
        '   - Mean Squared: {:.2}\n'
        # '   - Mean Squared Log: {:.2}\n'
        # ' - Mono Error\n'
        # '   - Mean Squared: {:.2}\n'
        # '   - Mean Squared Log: {:.2}\n'
        # '   - Mean Absolute: {:.2}\n'
        # '   - Median Absolute: {:.2}\n'
        # ' - Correlation\n'
        # '   - Input vs Output: {:.2}\n'
        # '   - Input vs Prediction: {:.2}\n'
        # '     - Difference: {:.2}\n'
        # '   - Output vs Prediction: {:.2} [maximize to 1]\n'
        ' - Euclidian Distance\n'
        # '   - Input vs Output: {:.2}\n'
        # '   - Input vs Prediction: {:.2}\n'
        # '     - Difference: {:.2}\n'
        '   - Output vs Prediction: {:.2}[minimize to 0]\n'
        ' - Cosine Similarity\n'
        # '   - Input vs Output: {:.2}\n'
        # '   - Input vs Prediction: {:.2}\n'
        # '     - Difference: {:.2}\n'
        '   - Output vs Prediction: {:.2}[maximize to 1]\n'
        # ' - Reinforcement Loss: {:.2}\n'
        ''.format(  # rError['evs'],
            rError['r2'], rError['mse'],  # rError['msle'],
            # rError['monomse'],
            # rError['monomsle'],
            # rError['monomae'],
            # rError['monomedae'],
            # rError['io_corr'],
            # rError['ip_corr'],
            # abs(rError['io_corr'] - rError['ip_corr']),
            # rError['op_corr'],
            # rError['io_euclid_dist'],
            # rError['ip_euclid_dist'],
            # abs(rError['io_euclid_dist'] - rError['ip_euclidian_dist']),
            rError['op_euclid_dist'],  # rError['io_cos_sim'],
            # rError['ip_cos_sim'],
            # abs(rError['io_cos_sim'] - rError['ip_cos_sim']),
            rError['op_cos_sim'],  # rError['rloss'],
        )
    )
    # Classifier
    regressor.set_model('Sequencer', MODELS['Sequencer'], 'eval')
    c = np.vstack([data['label'] for _ in range(len(data['input']))])
    _, _, cPrediction = regressor.predict(projectData)
    cError = evaluate_classification(c, cPrediction)
    print(
        ' -- Accuracy Score: {:.2}%\n'
        ''.format(cError['as'])
    )
    if gc.scale:
        data['input'] = data['input'] * gc.maxVal
        data['ref'] = data['ref'] * gc.maxVal
        data['output'] = data['output'] * gc.maxVal
    return cError, rError


def interleave(signal: np.array):
    # If signal is mono, it will be duplicated so that the signal is stereo.
    if signal.shape[1] == 1:
        return np.hstack((signal, signal))
    return signal


def get_subdirs(directory: str, fullPath=True):
    # Returns sorted list of subdirectories in directory.
    subdirs = []
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if os.path.isdir(path) and (subdir not in gc.excludedDirs):
            if fullPath:
                subdirs.append(path)
            else:
                subdirs.append(subdir)
    return sorted(subdirs)


def get_files(directory: str, fullPath=True, exclude=[]):
    # Returns all of the files in directory.
    x = []
    for f in next(os.walk(directory))[2]:
        if f.split('.')[-1] not in gc.excludeformats and strip_ext(f) not in exclude:
            if fullPath:
                x.append(os.path.join(directory, f))
            else:
                x.append(f)
    return x


def path_end(path):
    # Returns last part of path. E.g. path='dir1/dir2/file.ext' returns 'file.ext'.
    return os.path.basename(os.path.normpath(path))


def plot(data, result, reference, path):
    title = '{}{}{}'.format(path_end(os.path.dirname(path)), os.sep, strip_ext(path_end(path)))
    x = [i for i in range(len(data))]
    referencePlot = go.Scatter(x=x, y=reference.sum(axis=1) / 2, **gc.referencePlotSettings)
    inputPlot = go.Scatter(x=x, y=data.sum(axis=1) / 2, **gc.inputPlotSettings)
    outputPlot = go.Scatter(x=x, y=result.sum(axis=1) / 2, **gc.outputPlotSettings)
    plots = [referencePlot, inputPlot, outputPlot]
    fig = go.Figure(data=plots, layout=gc.plot_layout(title=title, xlen=len(x)))
    filename = os.path.join(os.path.dirname(sys.argv[0]), path)
    plotly.offline.plot(fig, filename=filename, **gc.plotConfig)


def print_info(sf, m, x, y):
    print('\nInput: {}\nOutput: {}'.format(x.strip('training'), y.strip('training')))


def strip_ext(filename):
    # Removes extension from filename.
    return filename.split('.')[0]


def standardize_string(string):
    return string.replace(' ', '').replace('_', '').strip().lower()


def write(data, path):
    wavio.write(path + '.wav', data=data, rate=gc.sRate, sampwidth=gc.sWidth, scale='none')


def create_reference(directory, maxLength):
    fs = get_files(directory, exclude=gc.excludedFiles)
    master = np.zeros((maxLength, 2))
    for f in fs:
        master = np.add(master, read(f, length=maxLength))
    return master


def permute(files, permutation):
    if permutation + 1 > len(files):
        return files
    f = list(files)
    result = []
    fact = reduce(lambda x, y: x * y, range(1, len(f) + 1), 1)
    permutation %= fact
    while f:
        fact = int(fact / len(f))
        choice, permutation = permutation // fact, permutation % fact
        result += [f.pop(choice)]
    return result


def to_wav(p):
    m = os.environ.copy()['PATH'] + os.pathsep + gc.ffmpegPath + os.pathsep
    path = os.path.join(os.path.dirname(sys.argv[0]), p).replace('\\', os.sep).replace('/', os.sep)
    base, ext = os.path.splitext(path)
    cmd = 'ffprobe -v quiet -print_format json -show_format -show_streams "{0}"'.format(path)
    args = shlex.split(cmd)
    process = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={'PATH': m})
    ffprobe_dict = json.loads(str(process.communicate()[0].decode('utf-8')))
    if ext != '.wav' or int(ffprobe_dict['streams'][0]['sample_rate']) != gc.samplingRate or ffprobe_dict['streams'][0][
        'codec_name'] != gc.codec:
        newPath = '{}_.wav'.format(base)
        cmd = 'ffmpeg -i "{0}" -ar {1} -c:a {2} "{3}"'.format(path, int(gc.samplingRate), gc.codec, newPath)
        args = shlex.split(cmd)
        p = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={'PATH': m})
        p.communicate()
        if os.path.isfile(newPath):
            os.remove(path)
            os.rename(newPath, '{}.wav'.format(base))
            return '{}.wav'.format(base)
    return path


def plot(data, result, reference, path):
    title = path_end(os.path.dirname(path))
    title += '{}{}'.format(os.sep, strip_ext(path_end(path)))
    x = [i for i in range(len(data))]
    referencePlot = go.Scatter(
        x=x, y=reference.sum(axis=1) / 2, name='Reference',
        line={'color': 'rgba(181,244,142,1)', 'width': 1.5}
    )
    inputPlot = go.Scatter(
        x=x, y=data.sum(axis=1) / 2, name='Input',
        line={'color': 'rgba(243,82,141,0.5)', 'width': 1.5}
    )
    outputPlot = go.Scatter(
        x=x, y=result.sum(axis=1) / 2, name='Output',
        line={'color': 'rgba(141,82,243,0.5)', 'width': 1.5}
    )

    plots = [referencePlot, inputPlot, outputPlot]
    fig = go.Figure(data=plots, layout=gc.plot_layout(title=title, xlen=len(x)))
    filename = os.path.join(os.path.dirname(sys.argv[0]), path)
    plotly.offline.plot(fig, filename=filename, **gc.plotConfig)


def interleave(signal):
    """
    Turn mono into stereo, leave stereo as is.
    If signal contains one channel, it will
    be duplicated so that the signal has two
    channels.
    signal: numpy array
    """
    if signal.shape[1] == 1:
        return np.hstack((signal, signal))
    else:
        return signal


def get_files(directory, fullPath=True, exclude=[]):
    """
    Returns all of the files in directory.
    directory: str, path to directory to extract files from.
    """
    x = []
    for f in next(os.walk(directory))[2]:
        if f.split('.')[-1] not in gc.excludeformats and strip_ext(f) not in exclude:
            if fullPath:
                x.append(os.path.join(directory, f))
            else:
                x.append(f)
    return x


def path_end(path):
    """
    Returns last part of path. E.g. path='dir1/dir2/file.ext' returns 'file.ext'.
    """
    return os.path.basename(os.path.normpath(path))


def printio(x, y, i):
    if i % gc.samplingRate * 5 == 0:
        sec = 1 + (i + 1) / (gc.samplingRate)
        print('-----sec: {}-----'.format(sec))
        print('x: {}'.format(x))
        print('y: {}'.format(y))
        print('-----------------\n')


def read(path, padding=True, length=None):
    """
    Reads wav file specified by path.
    """
    peth = to_wav(path)
    signal = wavio.read(peth).data
    if padding:
        signal = interleave(homogenize_length(signal, length))
    return signal


def strip_ext(filename):
    """
    Removes extension from filename.
    """
    return filename.split('.')[0]


def standardize_string(string):
    return string.replace(' ', '').replace('_', '').strip().lower()


def max_length(project):
    max = 0
    for subdir in get_subdirs(project):
        for f in get_files(subdir):
            data = read(f, padding=False)
            if len(data) > max:
                max = len(data)
    return max


def to_wav(p):
    m = os.environ.copy()['PATH'] + os.pathsep + gc.ffmpegPath + os.pathsep
    path = os.path.join(os.path.dirname(sys.argv[0]), p).replace('\\', os.sep).replace('/', os.sep)
    base, ext = os.path.splitext(path)
    cmd = 'ffprobe -v quiet -print_format json -show_format -show_streams "{0}"'.format(path)
    args = shlex.split(cmd)
    process = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={'PATH': m})
    process.wait(timeout=10)
    ffprobe_dict = json.loads(str(process.communicate()[0].decode('utf-8')))
    if ext != '.wav' or int(ffprobe_dict['streams'][0]['sample_rate']) != gc.samplingRate or ffprobe_dict['streams'][0][
        'codec_name'] != gc.codec:
        newPath = '{}_.wav'.format(base)
        cmd = 'ffmpeg -i "{0}" -ar {1} -c:a {2} "{3}"'.format(path, int(gc.samplingRate), gc.codec, newPath)
        args = shlex.split(cmd)
        p = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={'PATH': m})
        p.wait(timeout=10)
        if os.path.isfile(newPath):
            os.remove(path)
            os.rename(newPath, '{}.wav'.format(base))
            return '{}.wav'.format(base)
    return path


def plot(data, result, path):
    name1 = path_end(os.path.dirname(path))
    name2 = str(int(name1) + 1)
    name1 += '{}{}'.format(os.sep, strip_ext(path_end(path)))
    name2 += '{}{}'.format(os.sep, strip_ext(path_end(path)))
    x = [i for i in range(len(data))]
    d = go.Scatter(x=x, y=data.sum(axis=1) / 2, name=name1)
    r = go.Scatter(x=x, y=result.sum(axis=1) / 2, name=name2)
    da = [d, r]
    filename = os.path.join(os.path.dirname(sys.argv[0]), path)
    plotly.offline.plot(
        da, filename=filename, auto_open=False, show_link=False,
        config=dict(displaylogo=False, modeBarButtonsToRemove=['sendDataToCloud'])
    )


def interleave(signal):
    """
    Turn mono into stereo, leave stereo as is.
    If signal contains one channel, it will
    be duplicated so that the signal has two
    channels.
    signal: numpy array
    """
    if signal.shape[1] == 1:
        return np.hstack((signal, signal))
    else:
        return signal


def path_end(path):
    """
    Returns the last part of the path.
    E.g. path='dir1/dir2/file.ext' returns 'file.ext'.
    """
    return os.path.basename(os.path.normpath(path))


def read(path, padding=True, length=None):
    """
    Reads wav file specified by path.
    """
    peth = to_wav(path)
    signal = wavio.read(peth).data
    if padding:
        signal = interleave(homogenize_length(signal, length))
    return signal


def strip_ext(string):
    """
    Removes extension from string.
    """
    return string.split('.')[0]


def wav_info(path):
    wav = wave.open(path, 'r')
    samplingRate = int(wav.getframerate())
    sampleWidth = int(wav.getsampwidth())
    return {'sampleWidth': sampleWidth, 'samplingRate': samplingRate, }


def to_wav(p):
    m = os.environ.copy()['PATH'] + os.pathsep + gc.ffmpegPath + os.pathsep
    path = os.path.join(os.path.dirname(sys.argv[0]), p).replace('\\', os.sep).replace('/', os.sep)
    base, ext = os.path.splitext(path)
    cmd = 'ffprobe -v quiet -print_format json -show_format -show_streams "{0}"'.format(path)
    args = shlex.split(cmd)
    process = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={'PATH': m})
    ffprobe_dict = json.loads(str(process.communicate()[0].decode('utf-8')))
    if ext != '.wav' or int(ffprobe_dict['streams'][0]['sample_rate']) != gc.samplingRate or ffprobe_dict['streams'][0][
        'codec_name'] != gc.codec:
        newPath = '{}_.wav'.format(base)
        cmd = 'ffmpeg -i "{0}" -ar {1} -c:a {2} "{3}"'.format(path, int(gc.samplingRate), gc.codec, newPath)
        args = shlex.split(cmd)
        p = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={'PATH': m})
        p.wait(timeout=10)
        if os.path.isfile(newPath):
            os.remove(path)
            os.rename(newPath, '{}.wav'.format(base))
            return '{}.wav'.format(base)
    return path


def plot(data, result):
    x = range(len(data))
    plt.plot(x, data.sum(axis=1) / 2)
    plt.plot(x, result.sum(axis=1) / 2)
    plt.show()
    plt.cla()


def homogenize_length(signal, length, side='finish'):
    """
    If # of samples in signal is less than length, append samples of 0s to end of signal.
    Ensures all signals  are the same length.
    signal: numpy array
    length: int
    """
    difference = int(length - len(signal))
    if difference > 0:
        z = np.zeros((difference, signal.shape[1]))
        if side == 'finish':
            return np.vstack((signal, z))
        elif side == 'start':
            return np.vstack((z, signal))
    return signal


def interleave(signal):
    """
    Turn mono into stereo, leave stereo as is.
    If signal contains one channel, it will
    be duplicated so that the signal has two
    channels.
    signal: numpy array
    """
    if signal.shape[1] == 1:
        return np.hstack((signal, signal))
    else:
        return signal


def path_end(path):
    """
    Returns the last part of the path.
    E.g. path='dir1/dir2/file.ext' returns 'file.ext'.
    """
    return os.path.basename(os.path.normpath(path))


def read(path, padding=True, length=None):
    """
    Reads wav file specified by path.
    Need to make work with any audio format.
    """
    peth = to_wav(path)
    signal = wavio.read(peth).data
    if padding:
        signal = interleave(homogenize_length(signal, length))
    return signal


def strip_ext(string):
    """
    Removes extension from string.
    """
    return string.split('.')[0]


def plot(data, result, path):
    name1 = path_end(os.path.dirname(path))
    name2 = str(int(name1) + 1)
    name1 += '{}{}'.format(os.sep, strip_ext(path_end(path)))
    name2 += '{}{}'.format(os.sep, strip_ext(path_end(path)))
    x = [i for i in range(len(data))]
    d = go.Scatter(x=x, y=data.sum(axis=1) / 2, name=name1)
    r = go.Scatter(x=x, y=result.sum(axis=1) / 2, name=name2)
    da = [d, r]
    filename = os.path.join(os.path.dirname(sys.argv[0]), path)
    plotly.offline.plot(da, filename=filename, **gc.plotConfig)


def interleave(signal):
    """
    Turn mono into stereo, leave stereo as is.
    If signal contains one channel, it will
    be duplicated so that the signal has two
    channels.
    signal: numpy array
    """
    if signal.shape[1] == 1:
        return np.hstack((signal, signal))
    else:
        return signal


def path_end(path):
    """
    Returns the last part of the path.
    E.g. path='dir1/dir2/file.ext' returns 'file.ext'.
    """
    return os.path.basename(os.path.normpath(path))


def read(path, padding=True, length=None):
    """
    Reads wav file specified by path.
    """
    peth = to_wav(path)
    signal = wavio.read(peth).data
    if padding:
        signal = interleave(homogenize_length(signal, length))
    return signal


def strip_ext(string):
    """
    Removes extension from string.
    """
    return string.split('.')[0]


def standardize_string(string):
    """
    Removes spaces & underscores from string, converts letters to lower case.
    """
    return string.replace(' ', '').replace('_', '').strip().lower()


def homogenize_length(signal: np.array, length: int, side='finish'):
    # If len(signal) < length, append samples of 0s to end of signal so all signals are same length.
    difference = int(length - len(signal))
    if difference > 0:
        z = np.zeros((difference, signal.shape[1]))
        if side == 'finish':
            return np.vstack((signal, z))
        elif side == 'start':
            return np.vstack((z, signal))
    return signal


def interleave(signal: np.array):
    # If signal is mono, it will be duplicated so that the signal is stereo.
    if signal.shape[1] == 1:
        return np.hstack((signal, signal))
    return signal


def get_subdirs(path: str, fullPath=True):
    # Returns sorted list of subdirectories in path.
    subdirs = []
    for subdir in os.listdir(path):
        _path = os.path.join(path, subdir)
        if os.path.isdir(_path) and (subdir not in gc.excludedDirs):
            if fullPath:
                subdirs.append(_path)
            else:
                subdirs.append(subdir)
    return sorted(subdirs)


def get_files(path: str, fullPath=True, exclude=[]):
    # Returns all of the files in path.
    x = []
    for f in next(os.walk(path))[2]:
        if f.split('.')[-1] not in gc.excludeformats and strip_ext(f) not in exclude:
            if fullPath:
                x.append(os.path.join(path, f))
            else:
                x.append(f)
    return x


def max_length(project):
    maxL = 0
    for subdir in get_subdirs(project):
        for f in get_files(subdir):
            data = read(f, padding=False)
            if len(data) > maxL:
                maxL = len(data)
    return maxL


def path_end(path):
    # Returns last part of path. E.g. path='dir1/dir2/file.ext' returns 'file.ext'.
    return os.path.basename(os.path.normpath(path))


def plot(input, output, ref, path):
    title = '{}{}{}'.format(path_end(os.path.dirname(path)), os.sep, strip_ext(path_end(path)))
    x = [i for i in range(len(input))]
    refPlot = go.Scatter(x=x, y=ref.sum(axis=1) / 2, **gc.refPlotSettings)
    inputPlot = go.Scatter(x=x, y=input.sum(axis=1) / 2, **gc.inputPlotSettings)
    outputPlot = go.Scatter(x=x, y=output.sum(axis=1) / 2, **gc.outputPlotSettings)
    plots = [refPlot, inputPlot, outputPlot]
    fig = go.Figure(data=plots, layout=gc.plot_layout(title=title, xlen=len(x)))
    filename = os.path.join(os.path.dirname(sys.argv[0]), path)
    plotly.offline.plot(fig, filename=filename, **gc.plotConfig)


def printio(x, y, i):
    if i % gc.sRate * 5 == 0:
        sec = 1 + (i + 1) / gc.sRate
        print('\n==sec: {}==\nx: {}\ny: {}\n=========\n'.format(sec, x, y))


def print_info(x, y):
    print('\nInput: {}\nOutput: {}'.format(x.strip('training'), y.strip('training')))


def read(path, padding=True, length=None):
    # Reads wav file specified by path.
    peth = to_wav(path)
    signal = wavio.read(peth).data
    if padding:
        signal = interleave(homogenize_length(signal, length))
    return signal


def strip_ext(filename):
    # Removes extension from filename.
    return filename.split('.')[0]


def standardize_string(string):
    return string.replace(' ', '').replace('_', '').strip().lower()


def to_wav(p):
    m = {'PATH': os.environ.copy()['PATH'] + os.pathsep + gc.ffmpegPath + os.pathsep}
    path = os.path.join(os.path.dirname(sys.argv[0]), p).replace('\\', os.sep).replace('/', os.sep)
    base, ext = os.path.splitext(path)
    args = shlex.split('ffprobe -v quiet -print_format json -show_format -show_streams "{0}"'.format(path))
    p = Popen(args, shell=True, stdout=PIPE, stderr=PIPE, env=m).communicate()
    r = json.loads(str(p[0].decode('utf-8')))
    if ext != '.wav' or int(r['streams'][0]['sample_rate']) != gc.sRate or r['streams'][0]['codec_name'] != gc.codec:
        newPath = '{}_.wav'.format(base)
        args = shlex.split('ffmpeg -i "{0}" -ar {1} -c:a {2} "{3}"'.format(path, int(gc.sRate), gc.codec, newPath))
        Popen(args, shell=True, stdout=PIPE, stderr=PIPE, env=m).communicate()
        if os.path.isfile(newPath):
            os.remove(path)
            os.rename(newPath, '{}.wav'.format(base))
            return '{}.wav'.format(base)
    return path


def write(data, path):
    wavio.write(path + '.wav', data=data, rate=gc.sRate, sampwidth=gc.sWidth, scale='none')


def create_ref(path, length):
    fs = get_files(path, exclude=gc.excludedFiles)
    master = np.zeros((length, 2))
    for f in fs:
        master = np.add(master, read(f, length=length))
    return master


def attempt(func, args):
    # Prevent crash due to simultaneous access of file
    done = False
    while not done:
        try:
            r = func(*args)
            done = True
            return r
        except:
            pass


def get_data(model, x, y, reference, length, last):
    return {
        'model':     model, 'label': gc.labels[gc.transfer2[standardize_string(model)]],
        'input':     read(x, length=length), 'output': read(y, length, length=length), 'maxLength': length,
        'reference': reference, 'last': last,
    }


def evaluate(data, projectData):
    if gc.scaling:
        data['input'] = data['input'] / gc.maxVal
        data['reference'] = data['reference'] / gc.maxVal
        data['output'] = data['output'] / gc.maxVal
    # Regressor
    regressor.set_model(ai.get_label(data['label']), 'eval')
    rPrediction = regressor.predict(data['input'], projectData)
    rError = regressor.evaluate_regression(data['input'], data['output'], rPrediction)
    print(
        ' - Explained Variance Score: {}\n'
        ' - Stereo Error\n'
        '   -- R2: {}\n'
        '   -- Mean Squared: {}\n'
        '   -- Mean Squared Log: {}\n'
        ' - Mono Error\n'
        '   -- Mean Squared: {}\n'
        '   -- Mean Squared Log: {}\n'
        '   -- Mean Absolute: {}\n'
        '   -- Median Absolute: {}\n'
        ' - Correlation\n'
        '   -- Input vs Output: {}\n'
        '   -- Input vs Prediction: {}\n'
        '     --- Difference: {}\n'
        '   -- Output vs Prediction: {} [maximize to 1]\n'
        ' - Euclidian Distance\n'
        '   -- Input vs Output: {}\n'
        '   -- Input vs Prediction: {}\n'
        '     --- Difference: {}\n'
        '   -- Output vs Prediction: {}[minimize to 0]\n'
        ' - Cosine Similarity\n'
        '   -- Input vs Output: {}\n'
        '   -- Input vs Prediction: {}\n'
        '     --- Difference: {}\n'
        '   -- Output vs Prediction: {}[maximize to 1]\n'
        ' - Reward: {}\n'
        ''.format(
            rError['evs'], rError['r2'], rError['mse'], rError['msle'], rError['monomse'], rError['monomsle'],
            rError['monomae'], rError['monomedae'], rError['io_corr'], rError['ip_corr'],
            abs(rError['io_corr'] - rError['ip_corr']), rError['op_corr'], rError['io_euclidian_dist'],
            rError['ip_euclidian_dist'], abs(rError['io_euclidian_dist'] - rError['ip_euclidian_dist']),
            rError['op_euclidian_dist'], rError['io_cos_sim'], rError['ip_cos_sim'],
            abs(rError['io_euclidian_dist'] - rError['ip_euclidian_dist']), rError['op_cos_sim'],
            rError['reward'], )
    )
    # Classifier
    regressor.set_model('Sequencer', 'eval')
    c = np.vstack([data['label'] for _ in range(len(data['input']))])
    _, _, cPrediction = regressor.predict(data['input'], projectData)
    cError = regressor.evaluate_classification(c, cPrediction)
    print(
        ' -- Accuracy Score: {}%\n'
        ''.format(cError['as'])
    )
    if gc.scaling:
        data['input'] = data['input'] * gc.maxVal
        data['reference'] = data['reference'] * gc.maxVal
        data['output'] = data['output'] * gc.maxVal
    return cError, rError


def apply_transformations(subdir, files, projectData, epoch):
    supersubdir = os.path.dirname(subdir)
    cErrors = []
    rErrors = []
    for f in files:
        sf = path_end(f)
        sequences = projectData['sequences'][strip_ext(sf)]
        for i in range(len(sequences)):
            if sequences[i] == path_end(subdir) and i < len(sequences) - 1:
                model = projectData['transformations'][strip_ext(sf)][i]
                x = os.path.join(supersubdir, sequences[i], sf)
                y = os.path.join(supersubdir, sequences[i + 1], sf)
                print_info(sf, model, x, y)
                last = sequences[-2] == sequences[i]
                data = get_data(model, x, y, projectData['reference'], projectData['maxLength'], last)
                if gc.addNoise:
                    if epoch >= gc.beginInputNoise:
                        data['input'] = dp.add_noise(data['input'])
                    if epoch >= gc.beginReferenceNoise:
                        data['reference'] = dp.add_noise(data['reference'])
                if gc.scaling:
                    data['input'] = data['input'] / gc.maxVal
                    data['reference'] = data['reference'] / gc.maxVal
                    data['output'] = data['output'] / gc.maxVal
                plotPath = os.path.join('{}.html'.format(strip_ext(f)))
                plot(data=data['input'], result=data['output'], reference=data['reference'], path=plotPath)
                regressor.train_transformation(data)
                if gc.scaling:
                    data['input'] = data['input'] * gc.maxVal
                    data['reference'] = data['reference'] * gc.maxVal
                    data['output'] = data['output'] * gc.maxVal
                projectData['reference'] = dp.update_reference(data)
                rError, cError = evaluate(data, projectData)
                cErrors.append(cError)
                rErrors.append(rError)
                break
    return projectData['reference'], cErrors, rErrors


def run():
    projects = get_subdirs(gc.trainingDir)
    for project in projects:
        subdirs = get_subdirs(project)
        projectData = {
            'maxLength':       max_length(project), 'sequences': get_sequences(subdirs),
            'transformations': get_transformations(project),
        }
        projectData['reference'] = create_reference(subdirs[0], projectData['maxLength'])
        for permutation in range(gc.permutations):
            print('- - - - Permutation: {} - - - -'.format(permutation))
            for epoch in range(gc.epochs):
                print('---------- Epoch: {} ----------'.format(epoch))
                for subdir in subdirs:
                    files = permute(get_files(subdir, exclude=gc.excludedFiles), permutation)
                    projectData['reference'], cErrors, rErrors = apply_transformations(
                        subdir, files, projectData,
                        epoch
                    )
                print('-------- End epoch: {} --------'.format(epoch))
            print('- - - End Permutation: {} - - -'.format(permutation))


def get_sequences(subdirs):
    fs = get_files(subdirs[0], exclude=gc.excludedFiles, fullPath=False)
    sequences = {strip_ext(f): [] for f in fs}
    for subdir in subdirs:
        files = get_files(subdir, exclude=gc.excludedFiles, fullPath=False)
        for file in files:
            sequences[strip_ext(file)].append(path_end(subdir))
    return sequences


def permute(files, permutation):
    if permutation + 1 > len(files):
        return files
    f = list(files)
    result = []
    fact = reduce(lambda x, y: x * y, range(1, len(f) + 1), 1)
    permutation %= fact
    while f:
        fact = int(fact / len(f))
        choice, permutation = permutation // fact, permutation % fact
        result += [f.pop(choice)]
    return result


def apply_transformations(subdir, files, projectData, epoch):
    supersubdir = os.path.dirname(subdir)
    for f in files:
        sf = path_end(f)
        sequences = projectData['sequences'][strip_ext(sf)]
        for i in range(len(sequences)):
            if sequences[i] == path_end(subdir) and i < len(sequences) - 1:
                model = projectData['transformations'][strip_ext(sf)][i]
                x = os.path.join(supersubdir, sequences[i], sf)
                y = os.path.join(supersubdir, sequences[i + 1], sf)
                print_info(sf, model, x, y)
                last = sequences[-2] == sequences[i]
                data = get_data(model, x, y, projectData['reference'], projectData['maxLength'], last)
                if gc.addNoise:
                    if epoch >= gc.beginInputNoise:
                        data['input'] = dp.add_noise(data['input'])
                    if epoch >= gc.beginReferenceNoise:
                        data['reference'] = dp.add_noise(data['reference'])
                plotPath = os.path.join('{}.html'.format(strip_ext(f)))
                plot(data=data['input'], result=data['output'], reference=data['reference'], path=plotPath)
                regressor.train_transformation(data)
                projectData['reference'] = dp.update_reference(data)
                break
    return projectData['reference']


def run():
    projects = get_subdirs(gc.trainingDir)
    for project in projects:
        subdirs = get_subdirs(project)
        projectData = {
            'maxLength':       max_length(project), 'sequences': get_sequences(subdirs),
            'transformations': get_transformations(project),
        }
        projectData['reference'] = create_reference(subdirs[0], projectData['maxLength'])
        for permutation in range(gc.permutations):
            print('- - - - Permutation: {} - - - -'.format(permutation))
            for epoch in range(gc.epochs):
                print('---------- Epoch: {} ----------'.format(epoch))
                for subdir in subdirs:
                    files = permute(get_files(subdir, exclude=gc.excludedFiles), permutation)
                    projectData['reference'] = apply_transformations(subdir, files, projectData, epoch)
                print('-------- End epoch: {} --------'.format(epoch))
            print('- - - End Permutation: {} - - -'.format(permutation))


def get_data(model, x, y, reference, length, last):
    return {
        'model':     model, 'input': read(x, length=length), 'output': read(y, length, length=length),
        'maxLength': length, 'reference': reference, 'last': last
    }


def apply_transformations(subdir, projectData):
    supersubdir = os.path.dirname(subdir)
    referencePath = '{}.{}'.format(os.path.join(subdir, gc.reference), gc.audioformat)
    reference = read(referencePath, length=projectData['maxLength'])
    for f in get_files(subdir, exclude=[gc.reference]):
        sf = path_end(f)
        sequences = projectData['sequences'][strip_ext(sf)]
        for i in range(len(sequences)):
            if sequences[i] == path_end(subdir) and i < len(sequences) - 1:
                model = projectData['transformations'][strip_ext(sf)][i]
                x = os.path.join(supersubdir, sequences[i], sf)
                y = os.path.join(supersubdir, sequences[i + 1], sf)
                print_info(sf, model, x, y)
                last = sequences[-2] == sequences[i]
                data = get_data(model, x, y, reference, projectData['maxLength'], last)
                plotPath = os.path.join('{}.html'.format(strip_ext(f)))
                plot(data=data['input'], result=data['output'], reference=data['reference'], path=plotPath)
                regressor.train_transformation(data)
                reference = dp.update_reference(data)
                break


def run():
    projects = get_subdirs(gc.trainingDir)
    for project in projects:
        subdirs = get_subdirs(project)
        projectData = {
            'maxLength':       max_length(project), 'sequences': get_sequences(subdirs),
            'transformations': get_transformations(project),
        }
        for subdir in subdirs:
            apply_transformations(subdir, projectData)


def interleave(signal: np.array):
    # If signal is mono, it will be duplicated so that the signal is stereo.
    if signal.shape[1] == 1:
        return np.hstack((signal, signal))
    return signal


def path_end(path):
    # Returns last part of path. E.g. path='dir1/dir2/file.ext' returns 'file.ext'.
    return os.path.basename(os.path.normpath(path))


def print_info(x, y):
    print('\nInput: {}\nOutput: {}'.format(x.strip('training'), y.strip('training')))


def strip_ext(filename):
    # Removes extension from filename.
    return filename.split('.')[0]


def standardize_string(string):
    return string.replace(' ', '').replace('_', '').strip().lower()


def write(data, path):
    wavio.write(path + '.wav', data=data, rate=gc.sRate, sampwidth=gc.sWidth, scale='none')


def create_ref(path, length):
    fs = get_files(path, exclude=gc.excludedFiles)
    master = np.zeros((length, 2))
    for f in fs:
        master = np.add(master, read(f, length=length))
    return master


def attempt(func, args):
    # Prevent crash due to simultaneous access of file
    done = False
    while not done:
        try:
            r = func(*args)
            done = True
            return r
        except:
            pass


def produce(d, currDir):
    keepGoing = False
    util.write(d['reference'], os.path.join(currDir, gc.reference))
    for f in get_files(currDir, exclude=gc.excludedFiles):
        data = read(f, length=d['maxLength'])
        if gc.scaling:
            data = data / gc.maxVal
            d['reference'] = d['reference'] / gc.maxVal
        # regressor.set_model('Sequencer')
        # label, labelVector, __ = regressor.predict(data, d)
        label = 'GainStaging'
        labelVector = gc.labels[label]
        update_transformations(d['transDir'], label, f)
        print(label)
        if label != 'Stop':
            keepGoing = True
            nextDir = get_next_dir(currDir)
            nextDirPath = os.path.join(d['projectDir'], str(nextDir))
            if not os.path.isdir(nextDirPath):
                os.mkdir(nextDirPath)
            regressor.set_model(label)
            result = regressor.predict(data, d, labelVector=labelVector)
            if gc.scaling:
                data = data * gc.maxVal
                d['reference'] = d['reference'] * gc.maxVal
                result = result * gc.maxVal
            print(result)
            resultFileName = os.path.join(nextDirPath, strip_ext(path_end(f)))
            util.write(result, resultFileName)
            plotPath = os.path.join('{}.html'.format(strip_ext(f)))
            plot(data=data, result=result, reference=d['reference'], path=plotPath)
            m = {'reference': d['reference'], 'input': data, 'output': result}
            d['reference'] = dp.update_reference(m)
    if keepGoing:
        produce(d, nextDirPath)


def produce(d, currDir):
    keepGoing = False
    util.write(d['reference'], os.path.join(currDir, gc.reference))
    for f in get_files(currDir, exclude=gc.excludedFiles):
        data = read(f, length=d['maxLength'])
        regressor.set_model('Sequencer')
        label, labelVector = regressor.predict(data, d)
        update_transformations(d['transDir'], label, f)
        print(label)
        if label != 'Stop':
            keepGoing = True
            nextDir = get_next_dir(currDir)
            nextDirPath = os.path.join(d['projectDir'], str(nextDir))
            resultFileName = os.path.join(nextDirPath, strip_ext(path_end(f)))
            if not os.path.isdir(nextDirPath):
                os.mkdir(nextDirPath)
            regressor.set_model(label)
            result = regressor.predict(data, d, labelVector=labelVector)
            util.write(result, resultFileName)
            plotPath = os.path.join('{}.html'.format(strip_ext(f)))
            plot(data=data, result=result, reference=d['reference'], path=plotPath)
            m = {'reference': d['reference'], 'input': data, 'output': result}
            d['reference'] = dp.update_reference(m)
    if keepGoing:
        produce(d, nextDirPath)


def create_reference(directory, maxLength):
    fs = get_files(directory)
    master = np.zeros((maxLength, 2))
    for f in fs:
        master = np.add(master, read(f, length=maxLength))
    return master


def produce(d, currDir):
    keepGoing = False
    util.write(d['reference'], os.path.join(currDir, gc.reference))
    for f in get_files(currDir, exclude=[gc.reference]):
        data = read(f, length=d['maxLength'])
        regressor.set_model('Sequencer')
        label = regressor.predict(data, d)
        update_transformations(d['transDir'], label, f)
        print(label)
        if label != 'Stop':
            keepGoing = True
            nextDir = get_next_dir(currDir)
            nextDirPath = os.path.join(d['projectDir'], str(nextDir))
            resultFileName = os.path.join(nextDirPath, strip_ext(path_end(f)))
            if not os.path.isdir(nextDirPath):
                os.mkdir(nextDirPath)
            regressor.set_model(label)
            result = regressor.predict(data, d)
            util.write(result, resultFileName)
            plotPath = os.path.join('{}.html'.format(strip_ext(f)))
            plot(data=data, result=result, reference=d['reference'], path=plotPath)
            m = {'reference': d['reference'], 'input': data, 'output': result}
            d['reference'] = dp.update_reference(m)
    if keepGoing:
        produce(d, nextDirPath)


def run(path):
    projects = get_subdirs(path)
    for project in projects:
        d = {}
        d['projectDir'] = project
        d['subdirs'] = get_subdirs(project)
        d['tracksDir'] = d['subdirs'][0]
        d['maxLength'] = max_length(project)
        d['reference'] = create_reference(d['tracksDir'], d['maxLength'])
        d['transDir'] = os.path.join(project, gc.transformationsDir)
        try:
            os.mkdir(d['transDir'])
        except:
            pass
        produce(d, d['tracksDir'])


def produce(d, currDir):
    regressor = regressors.Regressor()
    keepGoing = False
    dp.write(d['reference'], os.path.join(currDir, gc.reference))
    for f in get_files(currDir, exclude=[gc.reference]):
        data = read(f, length=d['maxLength'])
        regressor.set_model('Sequencer')
        label = regressor.predict(data, d)
        update_transformations(d['transDir'], label, f)
        print(label)
        if label != 'Stop':
            keepGoing = True
            nextDir = int(path_end(currDir)) + 1
            nextDirPath = os.path.join(d['projectDir'], str(nextDir))
            resultFileName = os.path.join(nextDirPath, strip_ext(path_end(f)))
            try:
                os.mkdir(nextDirPath)
            except:
                pass
            regressor.set_model(label)
            result = regressor.predict(data, d)
            dp.write(result, resultFileName)
            plotPath = os.path.join('{}.html'.format(strip_ext(f)))
            plot(data=data, result=result, reference=d['reference'], path=plotPath)
            m = {'reference': d['reference'], 'input': data, 'output': result}
            d['reference'] = dp.update_reference(m)
        else:
            print('Stopping')
    if keepGoing:
        produce(d, nextDirPath)


def produce(d, currDir):
    regressor = regressors.Regressor()
    keepGoing = False
    dp.write(d['reference'], os.path.join(currDir, gc.reference))
    for f in get_files(currDir, exclude=[gc.reference]):
        plotPath = os.path.join('{}.html'.format(strip_ext(f)))
        data = read(f, length=d['maxLength'])
        regressor.set_model('Sequencer')
        label = regressor.predict(data, d)
        update_transformations(d['transDir'], label, f)
        print(label)
        if label != 'Stop':
            keepGoing = True
            nextDir = int(path_end(currDir)) + 1
            nextDirPath = os.path.join(d['projectDir'], str(nextDir))
            resultFileName = os.path.join(nextDirPath, strip_ext(path_end(f)))
            try:
                os.mkdir(nextDirPath)
            except:
                pass
            regressor.set_model(label)
            result = regressor.predict(data, d)
            dp.write(result, resultFileName)
            plot(data, result, path=plotPath)
            m = {'reference': d['reference'], 'input': data, 'output': result}
            d['reference'] = dp.update_reference(m)
        else:
            print('Stopping')
    if keepGoing:
        produce(d, nextDirPath)


def produce(projectData, currDir):
    keepGoing = False
    util.write(projectData['ref'], os.path.join(currDir, gc.reference))
    for f in get_files(currDir, exclude=gc.excludedFiles):
        projectData['input'] = read(f, length=projectData['length'])
        if gc.scale:
            projectData['input'] = projectData['input'] / gc.maxVal
            projectData['ref'] = projectData['ref'] / gc.maxVal
        regressor.set_model('Sequencer')
        label, labelVector, _ = regressor.predict(projectData)
        update_transformations(projectData['transDir'], label, f)
        print(label)
        if label != 'Stop':
            keepGoing = True
            nextDir = get_next_dir(currDir)
            nextDirPath = os.path.join(projectData['projectDir'], str(nextDir))
            if not os.path.isdir(nextDirPath):
                os.mkdir(nextDirPath)
            regressor.set_model(label)
            projectData['label'] = labelVector
            result = regressor.predict(projectData)
            if gc.scale:
                projectData['input'] = projectData['input'] * gc.maxVal
                projectData['ref'] = projectData['ref'] * gc.maxVal
                result = result * gc.maxVal
            print(result)
            resultFileName = os.path.join(nextDirPath, strip_ext(path_end(f)))
            util.write(result, resultFileName)
            plotPath = os.path.join('{}.html'.format(strip_ext(f)))
            plot(input=projectData['input'], output=result, ref=projectData['ref'], path=plotPath)
            m = {'ref': projectData['ref'], 'input': projectData['input'], 'output': result}
            projectData['ref'] = dp.update_ref(m)
    if keepGoing:
        produce(projectData, nextDirPath)


def run():
    projects = get_subdirs(gc.projectsDir[gc.environment])
    for project in projects:
        projectData = {}
        projectData['projectDir'] = project
        projectData['subdirs'] = get_subdirs(project)
        projectData['tracksDir'] = projectData['subdirs'][0]
        projectData['length'] = max_length(project)
        projectData['ref'] = create_ref(projectData['tracksDir'], projectData['length'])
        projectData['transDir'] = os.path.join(project, gc.transformationsDir)
        try:
            os.mkdir(projectData['transDir'])
        except:
            pass
        produce(projectData, projectData['tracksDir'])


def produce(d, currDir):
    regressor = regressors.Regressor()
    keepGoing = False
    dp.write(d['reference'], os.path.join(currDir, gc.reference))
    for f in get_files(currDir, exclude=[gc.reference]):
        data = read(f, length=d['maxLength'])
        regressor.set_model('Sequencer')
        label = regressor.predict(data, d)
        update_transformations(d['transDir'], label, f)
        print(label)
        if label != 'Stop':
            keepGoing = True
            nextDir = int(path_end(currDir)) + 1
            nextDirPath = os.path.join(d['projectDir'], str(nextDir))
            resultFileName = os.path.join(nextDirPath, strip_ext(path_end(f)))
            try:
                os.mkdir(nextDirPath)
            except:
                pass
            regressor.set_model(label)
            result = regressor.predict(data, d)
            dp.write(result, resultFileName, d['info'])
            plot(data, result)
            m = {'reference': d['reference'], 'input': data, 'output': result}
            d['reference'] = dp.update_reference(m)
        else:
            print('Stopping')
    if keepGoing:
        produce(d, nextDirPath)


def create_reference(directory, maxLength):
    fs = get_files(directory)
    master = np.zeros((maxLength, 2))
    for f in fs:
        master = np.add(master, dp.read(f, maxLength))
    return master


def produce(d, aa):
    regressor = regressors.Regressor()
    dp.write(d['reference'], os.path.join(aa, gc.reference), d['info'])
    keepGoing = False
    for f in get_files(aa, exclude=[gc.reference]):
        data = dp.read(f, d['maxLength'])
        regressor.set_model('Sequencer')
        label = regressor.predict(data, d)
        update_transformations(d['transDir'], label, f)
        print(label)
        if label != 'Stop':
            keepGoing = True
            nextDir = int(path_end(aa)) + 1
            nextDirPath = os.path.join(d['projectDir'], str(nextDir))
            resultFileName = os.path.join(nextDirPath, strip_ext(path_end(f)))
            try:
                os.mkdir(nextDirPath)
            except:
                pass
            regressor.set_model(label)
            result = regressor.predict(data, d)
            dp.write(result, resultFileName, d['info'])
            m = {'reference': d['reference'], 'input': data, 'output': result}
            d['reference'] = dp.update_reference(m)
        else:
            print('Stopping')

    if keepGoing:
        produce(d, nextDirPath)


def run():
    projects = get_subdirs(gc.testingDir)
    for project in projects:
        d = {}
        d['projectDir'] = project
        d['subdirs'] = get_subdirs(project)
        d['tracksDir'] = d['subdirs'][0]
        d['maxLength'] = max_length(project)
        d['reference'] = create_reference(d['tracksDir'], d['maxLength'])
        d['transDir'] = os.path.join(project, gc.transformationsDir)
        d['info'] = dp.wav_info(get_files(d['tracksDir'])[0], d['maxLength'])
        try:
            os.mkdir(d['transDir'])
        except:
            pass
        produce(d, d['tracksDir'])


def interleave(signal: np.array):
    # If signal is mono, it will be duplicated so that the signal is stereo.
    if signal.shape[1] == 1:
        return np.hstack((signal, signal))
    return signal


def path_end(path):
    # Returns last part of path. E.g. path='dir1/dir2/file.ext' returns 'file.ext'.
    return os.path.basename(os.path.normpath(path))


def print_info(sf, m, x, y):
    print('\nInput: {}\nOutput: {}'.format(x.strip('training'), y.strip('training')))


def strip_ext(filename):
    # Removes extension from filename.
    return filename.split('.')[0]


def standardize_string(string):
    return string.replace(' ', '').replace('_', '').strip().lower()


def write(data, path):
    wavio.write(path + '.wav', data=data, rate=gc.sRate, sampwidth=gc.sWidth, scale='none')


def train_transformation(aiModel, data, MODELS):
    aiModel.set_model(utils.ai.get_label(data['label']), MODELS[data['model']], 'train')
    aiModel.train(data)
    aiModel.set_model('Sequencer', MODELS['Sequencer'], 'train')
    aiModel.train(data)
    if data['last']:
        aiModel.set_model('Sequencer', MODELS['Sequencer'], 'train')
        data['ref'] = dp.update_ref(data)
        label = np.copy(data['label'])
        data['label'] = gc.labels['Stop']
        aiModel.train(data)
        data['label'] = label


def apply_transformations(aiModel, MODELS, subdir, files, projectData, epoch):
    supersubdir = os.path.dirname(subdir)
    cErrs = []
    rErrs = []
    for f in files:
        sf = path_end(f)
        sequences = projectData['sequences'][strip_ext(sf)]
        for i in range(len(sequences)):
            if sequences[i] == path_end(subdir) and i < len(sequences) - 1:
                model = projectData['transformations'][strip_ext(sf)][i]
                x = os.path.join(supersubdir, sequences[i], sf)
                y = os.path.join(supersubdir, sequences[i + 1], sf)
                print_info(x, y)
                last = sequences[-2] == sequences[i]
                data = scale(noise(get_data(model, x, y, projectData, last), epoch))
                plotPath = os.path.join('{}.html'.format(strip_ext(f)))
                plot(input=data['input'], output=data['output'], ref=data['ref'], path=plotPath)
                train_transformation(aiModel, data, MODELS)
                data = unscale(data)
                projectData['ref'] = dp.update_ref(data)
                rErr, cErr = evaluate(aiModel, data, projectData, MODELS)
                cErrs.append(cErr)
                rErrs.append(rErr)
                break
    return projectData['ref'], cErrs, rErrs


def run(aiModel, MODELS):
    projects = get_subdirs(gc.trainingDir)
    for project in projects:
        subdirs = get_subdirs(project)
        projectData = {
            'length':          max_length(project), 'sequences': get_sequences(subdirs),
            'transformations': get_transformations(project),
        }
        projectData['ref'] = create_ref(subdirs[0], projectData['length'])
        for permutation in range(gc.permutations):
            print('- - - - Permutation: {} - - - -'.format(permutation))
            for epoch in range(gc.epochs):
                print('---------- Epoch: {} ----------'.format(epoch))
                for subdir in subdirs:
                    files = permute(get_files(subdir, exclude=gc.excludedFiles), permutation)
                    projectData['ref'], cErrs, rErrs = apply_transformations(
                        aiModel, MODELS, subdir, files,
                        projectData, epoch
                    )
                print('-------- End epoch: {} --------'.format(epoch))
            print('- - - End Permutation: {} - - -'.format(permutation))


def train_transformation(aiModel, data, MODELS):
    if not gc.multiModel:
        m = 'Model'
    else:
        m = data['model']
    aiModel.set_model(utils.ai.get_label(data['label']), MODELS[m], 'train')
    aiModel.train(data)
    aiModel.set_model('Sequencer', MODELS['Sequencer'], 'train')
    aiModel.train(data)
    if data['last']:
        aiModel.set_model('Sequencer', MODELS['Sequencer'], 'train')
        data['ref'] = dp.update_ref(data)
        label = np.copy(data['label'])
        data['label'] = gc.labels['Stop']
        aiModel.train(data)
        data['label'] = label


def apply_transforms(aiModel, MODELS, subdir, files, projectData, epoch):
    supersubdir = os.path.dirname(subdir)
    cErrs = []
    rErrs = []
    for f in files:
        sf = path_end(f)
        sequences = projectData['sequences'][strip_ext(sf)]
        for i in range(len(sequences)):
            if sequences[i] == path_end(subdir) and i < len(sequences) - 1:
                model = projectData['transformations'][strip_ext(sf)][i]
                x = os.path.join(supersubdir, sequences[i], sf)
                y = os.path.join(supersubdir, sequences[i + 1], sf)
                print_info(x, y)
                last = sequences[-2] == sequences[i]
                data = scale(noise(get_data(model, x, y, projectData, last), epoch))
                plotPath = os.path.join('{}.html'.format(strip_ext(f)))
                plot(input=data['input'], output=data['output'], ref=data['ref'], path=plotPath)
                train_transformation(aiModel, data, MODELS)
                data = unscale(data)
                projectData['ref'] = dp.update_ref(data)
                # rErr, cErr = utils.trainer.evaluate(aiModel, data, projectData, MODELS)
                rErr, cErr = {'reward': 0}, {'reward': 0}
                cErrs.append(cErr)
                rErrs.append(rErr)
                break
    return projectData['ref'], cErrs, rErrs


def run(aiModel, MODELS, permutation, epoch):
    projects = get_subdirs(gc.trainingDir)
    totalCErr = 0
    totalRErr = 0
    for project in projects:
        subdirs = get_subdirs(project)
        projectData = {
            'length':          max_length(project), 'sequences': get_sequences(subdirs),
            'transformations': get_transformations(project),
        }
        projectData['ref'] = create_ref(subdirs[0], projectData['length'])
        for subdir in subdirs:
            files = permute(get_files(subdir, exclude=gc.excludedFiles), permutation)
            projectData['ref'], cErrs, rErrs = apply_transforms(aiModel, MODELS, subdir, files, projectData, epoch)
            for err in cErrs:
                totalCErr += err['reward']
            for err in rErrs:
                totalRErr += err['reward']
    return totalCErr, totalRErr


def get_metadata(model, x, y, reference, length, last):
    return OrderedDict(
        [('model', model), ('input', dp.read(x, length)), ('output', dp.read(y, length)), ('maxLength', length),
         ('reference', reference), ('last', last)]
    )


def apply_transformations(subdir, projectData):
    supersubdir = os.path.dirname(subdir)
    reference = dp.read(os.path.join(subdir, gc.reference) + '.wav', projectData['maxLength'])
    for f in get_files(subdir, exclude=[gc.reference], fullPath=False):
        sequences = projectData['sequences'][strip_ext(f)]
        for i in range(len(sequences)):
            if sequences[i] == path_end(subdir) and i < len(sequences) - 1:
                model = projectData['transformations'][strip_ext(f)][i]
                x = os.path.join(supersubdir, sequences[i], f)
                y = os.path.join(supersubdir, sequences[i + 1], f)
                print('model: {}'.format(model))
                print('input: {}'.format(x.strip('training')))
                print('ouput: {}'.format(y.strip('training')))
                last = sequences[-2] == sequences[i]
                metadata = get_metadata(model, x, y, reference, projectData['maxLength'], last)
                regressor.train_transformation(metadata)
                reference = dp.update_reference(metadata)
                break


def apply_transformations(subdir, projectData):
    supersubdir = os.path.dirname(subdir)
    reference = read(os.path.join(subdir, gc.reference) + '.wav', length=projectData['maxLength'])
    for f in get_files(subdir, exclude=[gc.reference], fullPath=True):
        sf = path_end(f)
        print(sf)
        sequences = projectData['sequences'][strip_ext(sf)]
        for i in range(len(sequences)):
            if sequences[i] == path_end(subdir) and i < len(sequences) - 1:
                model = projectData['transformations'][strip_ext(sf)][i]
                x = os.path.join(supersubdir, sequences[i], sf)
                y = os.path.join(supersubdir, sequences[i + 1], sf)
                print('model: {}'.format(model))
                print('input: {}'.format(x.strip('training')))
                print('ouput: {}'.format(y.strip('training')))
                last = sequences[-2] == sequences[i]
                metadata = get_metadata(model, x, y, reference, projectData['maxLength'], last)
                regressor.train_transformation(metadata)
                reference = dp.update_reference(metadata)
                break


def get_metadata(model, x, y, reference, length, last):
    return OrderedDict(
        [('model', model), ('input', read(x, length=length)), ('output', read(y, length, length=length)),
         ('maxLength', length), ('reference', reference), ('last', last)]
    )


def print_info(sf, model, x, y):
    print(sf)
    print('model: {}'.format(model))
    print('input: {}'.format(x.strip('training')))
    print('ouput: {}'.format(y.strip('training')))


def max_length(project):
    max = 0
    for subdir in get_subdirs(project):
        for signal in get_files(subdir):
            data = wavio.read(signal).data
            if len(data) > max:
                max = len(data)
    return max


def read_transformation(path):
    with open(path) as f:
        t = f.read().split('\n')
    return t


def get_transformations(project):
    transformationsDir = os.path.join(project, gc.transformationsDir)
    transformationFiles = get_files(transformationsDir)
    transformations = {strip_ext(path_end(f)): read_transformation(f) for f in transformationFiles}
    return transformations


def get_sequences(subdirs):
    fs = get_files(subdirs[0], exclude=[gc.reference], fullPath=False)
    sequences = {strip_ext(f): [] for f in fs}
    for subdir in subdirs:
        files = get_files(subdir, exclude=[gc.reference], fullPath=False)
        for file in files:
            sequences[strip_ext(file)].append(path_end(subdir))
    return sequences


def update_reference(m):
    return np.add(np.subtract(m['reference'], m['input']), m['output'])


def do_IT(subdir, projectData):
    s = path_end(subdir)
    fs = get_files(subdir, exclude=[gc.reference], fullPath=False)
    supersubdir = os.path.dirname(subdir)
    reference = dp.read(os.path.join(subdir, gc.reference) + '.wav', projectData['maxLength'])
    for f in fs:
        sequences = projectData['sequences'][strip_ext(f)]
        for i in range(len(sequences)):
            if sequences[i] == s and i < len(sequences) - 1:
                x = os.path.join(supersubdir, sequences[i], f)
                y = os.path.join(supersubdir, sequences[i + 1], f)
                # print('input: {}'.format(os.path.join(x, f)))
                # print('ouput: {}'.format(os.path.join(y, f)))
                metadata = OrderedDict(
                    [('model', projectData['transformations'][strip_ext(f)][i]),
                     ('input', dp.read(x, projectData['maxLength'])),
                     ('output', dp.read(y, projectData['maxLength'])), ('reference', reference)]
                )
                regressor.train_transformation(metadata, projectData['maxLength'])
                reference = update_reference(metadata)
                break


def run():
    projects = get_subdirs(gc.trainingDir)
    for project in projects:
        maxLength = max_length(project)
        subdirs = get_subdirs(project)
        transformations = get_transformations(project)
        sequences = get_sequences(subdirs)
        projectData = {'maxLength': maxLength, 'sequences': sequences, 'transformations': transformations}
        for subdir in subdirs:
            do_IT(subdir, projectData)


def interleave(signal: np.array):
    # If signal is mono, it will be duplicated so that the signal is stereo.
    if signal.shape[1] == 1:
        return np.hstack((signal, signal))
    return signal


def path_end(path):
    # Returns last part of path. E.g. path='dir1/dir2/file.ext' returns 'file.ext'.
    return os.path.basename(os.path.normpath(path))


def printio(x, y, i):
    if i % gc.sRate * 5 == 0:
        sec = 1 + (i + 1) / gc.sRate
        print('-sec: {}-\nx: {}\ny: {}\n---------\n'.format(sec, x, y))


def print_info(sf, m, x, y):
    print('{}\nmodel: {}\ninput: {}\noutput: {}'.format(sf, m, x.strip('training'), y.strip('training')))


def strip_ext(filename):
    # Removes extension from filename.
    return filename.split('.')[0]


def standardize_string(string):
    return string.replace(' ', '').replace('_', '').strip().lower()


def write(data, path):
    wavio.write(path + '.wav', data=data, rate=gc.sRate, sampwidth=gc.sWidth, scale='none')


def get_subdirs(directory, exclude=[], fullPath=True):
    """
    Returns a sorted list of the subdirectories in a directory.

    path: str, path to the directory from which
    to collect subdirectories.

    exclude: list of str, subdirectories to exclude.

    fullPath: boolean, whether to return full
    subdirectory paths or only the subdirectory name.
    """
    subdirs = []
    for subdirectory in os.listdir(directory):
        path = os.path.join(directory, subdirectory)
        if os.path.isdir(path) and (subdirectory not in exclude):
            if fullPath:
                subdirs.append(path)
            else:
                subdirs.append(subdirectory)
    return sorted(subdirs)


def get_files(directory):
    """
    Returns all of the files in the
    specified directory.

    directory: str, the path to the
    directory to extract files from.
    """
    files = next(os.walk(directory))[2]
    return [os.path.join(directory, f) for f in files]


def path_end(path):
    """
    Returns the last part of the path.
    E.g. if the path is 'dir1/dir2/file.ext'
    this function will return 'file.ext'.
    """
    return os.path.basename(os.path.normpath(path))


def attempt_property(prop):
    """
    Decorator used to attempt to retrieve a
    property. If the attempt fails,
    a failure message is displayed.
    """

    def wrapper(*args, **kwargs):
        try:
            result = prop(*args, **kwargs)
            if result:
                return '{0}: {1}'.format(prop.__name__, result)
            else:
                return '{} unknown.'.format(prop.__name__)
        except:
            return 'Function {} failed.'.format(prop.__name__)

    return wrapper


def printio(x, y, i, sr=44100):
    if i % sr * 5 == 0:
        sec = 1 + (i + 1) / (sr)
        print('-----sec: {}-----'.format(sec))
        print('x: {}'.format(x))
        print('y: {}'.format(y))
        print('-----------------\n')


def remove_parenthesis(string):
    """
    Removes everything in a string
    starting from an open parenthesis
    until the end of the string.
    E.g. if string='Equalizer (mono)'
    this function will return 'Equalizer'.
    Note that the trailing whitespace
    is also removed.
    """
    s = ''
    for char in string:
        if char == '(':
            s = s.strip()
            break
        s += char
    return s


def strip_extension(string, extension):
    """
    Removes the extension from the string.
    E.g. if string='file.html' and
    extension='html' (or extension='.html'),
    this function will return 'file'.
    string: str, a file name
    extension: str, an extension
    """
    if extension in string:
        stripped = string[:-(len(extension))]
        if stripped[-1] == '.':
            stripped = stripped[:-1]
        return stripped
    else:
        return string


def standardize_string(string):
    """
    Removes all spaces and underscores from a string,
    and converts all letters to lower case.
    """
    return string.replace(' ', '').replace('_', '').strip().lower()


def wav_info(path):
    info = {}
    wav = wave.open(path, 'r')
    info['channels'] = wav.getnchannels()
    info['sampleWidth'] = wav.getsampwidth()
    info['samplingRate'] = wav.getframerate()
    info['samples'] = wav.getnframes()
    return info


def get_subdirs(directory, fullPath=True):
    """
    Returns sorted list of subdirectories in directory.
    path: str, path to directory from which to collect subdirectories.
    exclude: list of str, subdirectories to exclude.
    fullPath: boolean, whether to return full paths or only subdirectory name.
    """
    subdirs = []
    for subdirectory in os.listdir(directory):
        path = os.path.join(directory, subdirectory)
        if os.path.isdir(path) and (subdirectory not in gc.excluded):
            if fullPath:
                subdirs.append(path)
            else:
                subdirs.append(subdirectory)
    return sorted(subdirs)


def get_files(directory, fullPath=True, exclude=[]):
    """
    Returns all of the files in directory.
    directory: str, path to directory to extract files from.
    """
    x = []
    for f in next(os.walk(directory))[2]:
        if strip_ext(f) not in exclude:
            if fullPath:
                x.append(os.path.join(directory, f))
            else:
                x.append(f)
    return x


def path_end(path):
    """
    Returns the last part of the path.
    E.g. path='dir1/dir2/file.ext' returns 'file.ext'.
    """
    return os.path.basename(os.path.normpath(path))


def printio(x, y, i, sr=gc.samplingRate):
    if i % sr * 5 == 0:
        sec = 1 + (i + 1) / (sr)
        print('-----sec: {}-----'.format(sec))
        print('x: {}'.format(x))
        print('y: {}'.format(y))
        print('-----------------\n')


def strip_ext(string):
    """
    Removes extension from string.
    """
    return string.split('.')[0]


def standardize_string(string):
    """
    Removes spaces & underscores from string, converts letters to lower case.
    """
    return string.replace(' ', '').replace('_', '').strip().lower()


def read(path):
    """
    Reads the wav file specified by the path.
    Need to make work with any audio format.
    :param path:
    :return:
    """
    return wavio.read(path)


def homogenize_length(signal, length, side='finish'):
    """
    If the number of samples in the signal
    is shorter than length, samples of 0s
    will be appened to the end of the signal.
    This function is used to ensure all signals
    to be processed are the same length.
    signal: numpy array
    length: int
    """
    difference = length - len(signal)
    if difference >= 0:
        z = np.zeros((difference, signal.shape[1]))
        if side == 'finish':
            return np.vstack((signal, z))
        elif side == 'start':
            return np.vstack((z, signal))
    return signal


def interleave(signal):
    """
    Turn mono into stereo, leave stereo as is.
    If signal contains one channel, it will
    be duplicated so that the signal has two
    channels.
    signal: numpy array
    """
    if signal.shape[1] == 1:
        return np.hstack((signal, signal))
    else:
        return signal


def pad_data(data, samplingRate=44100, seconds=1, paddingRate=2):
    """
    This function is a generator1 that yields
    one sample at a time. Each sample is padded with
    values from before and after in order to embed
    temporal context.
    samplingRate: int, number of samples per second in the signal (Hz)
    paddingRate: int, number of samples per second to use for padding (Hz)
    seconds: int, number of seconds to look behind/ahead (sec)
    """
    padLength = int(seconds * paddingRate)
    skips = int(samplingRate / paddingRate)
    length, width = data.shape
    for i, sample in enumerate(data):
        behind = look_behind(data, padLength, skips, length, width, i)
        ahead = look_ahead(data, padLength, skips, length, width, i)
        yield np.hstack((behind, sample, ahead))


def training_data_generator(input, reference, output, rate=global_config.rate, seconds=global_config.seconds, mode='t'):
    """
    This function is a generator1 that outputs
    the training data for a model. It emits an (x, y)
    pair where the x vector has been padded with
    temporal context, and the y vector is untouched.
    input: dict
    output: dict
    reference: dict
    rate: int
    seconds: int
    """
    inputGenerator = pad_data(
        data=input['data'] if mode == 't' else fourier(input['data']),
        samplingRate=input['samplingRate'], seconds=seconds, paddingRate=rate, )

    referenceGenerator = pad_data(
        data=reference.data if mode == 't' else fourier(reference.data),
        samplingRate=input['samplingRate'], seconds=seconds, paddingRate=rate, )

    outputGenerator = pad_data(data=output['data'], samplingRate=output['samplingRate'], seconds=0, paddingRate=rate, )

    for i in range(len(input['data'])):
        x = np.hstack((next(inputGenerator), next(referenceGenerator)))
        y = next(outputGenerator)
        yield x.reshape(1, -1), y.reshape(1, -1)


def sequencer_training_data_generator(
    input, reference, output, rate=global_config.rate, seconds=global_config.seconds,
    mode='t'
):
    """
    This function is a generator1 that outputs
    the training data for a model. It emits an (x, y)
    pair where the x vector has been padded with
    temporal context, and the y vector is untouched.
    input: dict
    output: dict
    reference: dict
    rate: int
    seconds: int
    """
    inputGenerator = pad_data(
        data=input['data'] if mode == 't' else fourier(input['data']),
        samplingRate=input['samplingRate'], seconds=seconds, paddingRate=rate, )

    referenceGenerator = pad_data(
        data=reference.data if mode == 't' else fourier(reference.data),
        samplingRate=input['samplingRate'], seconds=seconds, paddingRate=rate, )

    outputGenerator = pad_data(
        data=np.vstack([output for _ in range(len(input['data']))]), samplingRate=0, seconds=0,
        paddingRate=rate
    )

    for i in range(len(input['data'])):
        x = np.hstack((next(inputGenerator), next(referenceGenerator)))
        y = next(outputGenerator)
        yield x.reshape(1, -1), y.reshape(1, -1)


def fourier(signal):
    print('fourier up your!')
    print(signal)
    x = np.fft.fft2(signal).real
    print(x)
    return x


def format_training_data(metadata, maxLength):
    results = []
    for m in ['input', 'output']:
        signal = read(metadata[m])
        data = {
            'data':  interleave(homogenize_length(signal.data, maxLength)), 'samplingRate': signal.rate,
            'width': signal.sampwidth,
        }
        results.append(data)
    return results


def format_sequencer_training_data(metadata, maxLength):
    results = []
    for m in ['input']:
        signal = read(metadata[m])
        data = {
            'data':  interleave(homogenize_length(signal.data, maxLength)), 'samplingRate': signal.rate,
            'width': signal.sampwidth,
        }
        results.append(data)
    return results[0]


def homogenize_length(signal, length, side='finish'):
    """
    If # of samples in signal is less than length, append samples of 0s to end of signal.
    Ensures all signals  are the same length.
    signal: numpy array
    length: int
    """
    difference = int(length) - len(signal)
    if difference > 0:
        z = np.zeros((difference, signal.shape[1]))
        if side == 'finish':
            return np.vstack((signal, z))
        elif side == 'start':
            return np.vstack((z, signal))
    return signal


def interleave(signal):
    """
    Turn mono into stereo, leave stereo as is. If signal contains one channel, it will
    be duplicated so that the signal has two channels.
    signal: numpy array
    """
    if signal.shape[1] == 1:
        return np.hstack((signal, signal))
    else:
        return signal


def pad(data, samplingRate=gc.sRate, seconds=gc.pSecs, paddingRate=gc.pRate):
    """
    This generator yields one temporally padded sample at a time.
    sRate: int, number of samples per second in the signal (Hz)
    pRate: int, number of samples per second to use for padding (Hz)
    pSecs: int, number of pSecs to look behind/ahead (sec)
    """
    padLength = int(seconds * paddingRate)
    skips = int(samplingRate / paddingRate)
    length, width = data.shape
    for i, sample in enumerate(data):
        behind = look_behind(data, padLength, skips, length, width, i)
        ahead = look_ahead(data, padLength, skips, length, width, i)
        yield np.hstack((behind, sample, ahead))


def training_data_generator(metadata, sequencer=False, label=None):
    """
    This generator outputs training data for a model.
    Emits (x, y) pair. x vector has been padded with temporal context.
    """
    inputGenerator = pad(data=metadata['input'])
    referenceGenerator = pad(data=metadata['reference'])
    if gc.useFFT:
        inputFFTGenerator = pad(data=fourier(metadata['input']))
        referenceFFTGenerator = pad(data=fourier([metadata['reference']]))
    if sequencer:
        o = np.vstack([metadata['label'] for _ in range(len(metadata['input']))])
    else:
        o = metadata['output']
    outputGenerator = pad(data=o, samplingRate=0, seconds=0)
    for i in range(len(metadata['input'])):
        x = np.hstack((next(inputGenerator), next(referenceGenerator)))
        if gc.multiModel:
            x = np.hstack((x, label))
        if gc.useFFT:
            x = np.hstack((x, next(inputFFTGenerator), next(referenceFFTGenerator)))
        if gc.useFreq:
            inputFrequencies = get_frequencies(metadata['input'], i)
            referenceFrequencies = get_frequencies(metadata['reference'], i)
            x = np.hstack((x, inputFrequencies, referenceFrequencies))
        y = next(outputGenerator)
        yield x.reshape(1, -1), y.reshape(1, -1)


def get_generator(data, d, label=None):
    inputGenerator = pad(data=data)
    referenceGenerator = pad(data=d['reference'])
    if gc.useFFT:
        inputFFTGenerator = pad(data=fourier(data))
        referenceFFTGenerator = pad(data=fourier([d['reference']]))
    for i in range(len(data)):
        x = np.hstack((next(inputGenerator), next(referenceGenerator)))
        if gc.multiModel:
            x = np.hstack((x, label))
        if gc.useFFT:
            x = np.hstack((x, next(inputFFTGenerator), next(referenceFFTGenerator)))
        if gc.useFreq:
            inputFrequencies = get_frequencies(data, i)
            referenceFrequencies = get_frequencies(d['reference'], i)
            x = np.hstack((x, inputFrequencies, referenceFrequencies))
        yield x.reshape(1, -1)


def get_frequencies(data, i):
    N = int(gc.bSamples / 2)
    if i == 0:
        return np.zeros(N)
    elif i < (gc.bSamples - 1):
        signal = homogenize_length(data[0:1], int(gc.bSamples), 'start')
    else:
        signal = data[i - int(gc.bSamples - 1): i]
    return abs(scipy.fft(signal.sum(axis=1) / 2))[range(N)]


def pad(data, samplingRate=gc.sRate, seconds=gc.pSecs, paddingRate=gc.pRate):
    """
    This generator yields one temporally padded sample at a time.
    sRate: int, number of samples per second in the signal (Hz)
    pRate: int, number of samples per second to use for padding (Hz)
    pSecs: int, number of pSecs to look behind/ahead (sec)
    """

    def look_behind(data, padLength, skips, length, width, i):
        behind = []
        for k in range(padLength, 0, -1):
            if (i - skips * k) < 0:
                behind.extend([0 for _ in range(width)])
            else:
                behind.extend(list(data[i - skips * k]))
        return behind

    def look_ahead(data, padLength, skips, length, width, i):
        ahead = []
        for k in range(1, padLength + 1):
            if (i + skips * k) >= length:
                ahead.extend([0 for _ in range(width)])
            else:
                ahead.extend(list(data[i + skips * k]))
        return ahead

    padLength = int(seconds * paddingRate)
    skips = int(samplingRate / paddingRate)
    length, width = data.shape
    for i, sample in enumerate(data):
        behind = look_behind(data, padLength, skips, length, width, i)
        ahead = look_ahead(data, padLength, skips, length, width, i)
        yield np.hstack((behind, sample, ahead))


def get_gen(data, sequencer=False, training=False):
    """
    This generator outputs training data for a model.
    Emits (x, y) pair. x vector has been padded with temporal context.
    """
    inputGen = pad(data=data['input'])
    refGen = pad(data=data['ref'])
    if training:
        if sequencer:
            o = np.vstack([data['label'] for _ in range(len(data['input']))])
        else:
            o = data['output']
        outputGenerator = pad(data=o, samplingRate=0, seconds=0)
    if gc.useFFT:
        inputFFTGen = pad(data=get_fft(data['input']))
        refFFTGen = pad(data=get_fft([data['ref']]))
    for i in range(len(data['input'])):
        x = np.hstack((next(inputGen), next(refGen)))
        if not gc.multiModel and not sequencer:
            x = np.hstack((x, data['label']))
        if gc.useFFT:
            x = np.hstack((x, next(inputFFTGen), next(refFFTGen)))
        if gc.useFreq:
            inputFreq = get_freq(data['input'], i)
            refFreq = get_freq(data['ref'], i)
            x = np.hstack((x, inputFreq, refFreq))
        if training:
            y = next(outputGenerator)
            yield x.reshape(1, -1), y.reshape(1, -1)
        else:
            yield x.reshape(1, -1)


def update_ref(m):
    return np.add(np.subtract(m['ref'], m['input']), m['output'])


def get_fft(signal):
    fft = np.fft.rfft(signal).real
    if len(fft.shape) == 3:
        fft = fft[0]
    return fft


def get_freq(data, i):
    N = int(gc.bSamples / 2)
    if i == 0:
        return np.zeros(N)
    elif i < (gc.bSamples - 1):
        signal = homogenize_length(data[0:1], int(gc.bSamples), 'start')
    else:
        signal = data[i - int(gc.bSamples - 1): i]
    return abs(scipy.fft(signal.sum(axis=1) / 2))[range(N)]


def noise(data, epoch):
    def add_noise(data):
        a = random.randint(-3, 4)
        b = random.randint(20, 41) / 10
        mean = random.randint(-9000, 9001)
        std = random.randint(1, 40001) / 10
        scale = (1 + math.tanh(a) / b)
        noise = np.random.normal(mean, std, data.shape)
        return scale * data + noise

    if gc.addNoise:
        if epoch >= gc.beginInputNoise:
            data['input'] = add_noise(data['input'])
        if epoch >= gc.beginRefNoise:
            data['ref'] = add_noise(data['ref'])
    return data


def fourier(signal):
    fft = np.fft.rfft(signal).real
    if len(fft.shape) == 3:
        fft = fft[0]
    return fft


def get_generator(data, d):
    inputGenerator = pad(data=data)
    inputFFTGenerator = pad(data=fourier(data))
    referenceGenerator = pad(data=d['reference'])
    referenceFFTGenerator = pad(data=fourier([d['reference']]))
    for i in range(len(data)):
        inputFrequencies = get_frequencies(data, i)
        referenceFrequencies = get_frequencies(d['reference'], i)
        x = np.hstack(
            (next(inputGenerator), next(inputFFTGenerator), inputFrequencies, next(referenceGenerator),
             next(referenceFFTGenerator), referenceFrequencies)
        )
        yield x.reshape(1, -1)


def update_reference(m):
    return np.add(np.subtract(m['reference'], 2 * m['input']), 2 * m['output'])


def get_frequencies(data, i):
    N = int(gc.bufferSamples / 2)
    if i == 0:
        return np.zeros(N)
    elif i < (gc.bufferSamples - 1):
        signal = homogenize_length(data[0:1], gc.bufferSamples, 'start')
    else:
        signal = data[i - int(gc.bufferSamples - 1): i]
    signal = signal.sum(axis=1) / 2
    return abs(scipy.fft(signal))[range(N)]


def write(data, path, info):
    wavio.write(path + '.wav', data, info['samplingRate'], scale='none', sampwidth=info['sampleWidth'])


def homogenize_length(signal, length, side='finish'):
    """
    If # of samples in signal is less than length, append samples of 0s to end of signal.
    Ensures all signals  are the same length.
    signal: numpy array
    length: int
    """
    difference = length - len(signal)
    if difference >= 0:
        z = np.zeros((difference, signal.shape[1]))
        if side == 'finish':
            return np.vstack((signal, z))
        elif side == 'start':
            return np.vstack((z, signal))
    return signal


def interleave(signal):
    """
    Turn mono into stereo, leave stereo as is.
    If signal contains one channel, it will
    be duplicated so that the signal has two
    channels.
    signal: numpy array
    """
    if signal.shape[1] == 1:
        return np.hstack((signal, signal))
    else:
        return signal


def look_behind(data, padLength, skips, length, width, i):
    behind = []
    for k in range(padLength, 0, -1):
        if (i - skips * k) < 0:
            behind.extend([0 for _ in range(width)])
        else:
            behind.extend(list(data[i - skips * k]))
    return behind


def look_ahead(data, padLength, skips, length, width, i):
    ahead = []
    for k in range(1, padLength + 1):
        if (i + skips * k) >= length:
            ahead.extend([0 for _ in range(width)])
        else:
            ahead.extend(list(data[i + skips * k]))
    return ahead


def pad_data(data, samplingRate=44100, seconds=1, paddingRate=2):
    """
    This function is a generator that yields one sample at a time.
    Each sample is padded with values from before and after in order
    to embed temporal context.
    samplingRate: int, number of samples per second in the signal (Hz)
    paddingRate: int, number of samples per second to use for padding (Hz)
    seconds: int, number of seconds to look behind/ahead (sec)
    """
    padLength = int(seconds * paddingRate)
    skips = int(samplingRate / paddingRate)
    length, width = data.shape
    for i, sample in enumerate(data):
        behind = look_behind(data, padLength, skips, length, width, i)
        ahead = look_ahead(data, padLength, skips, length, width, i)
        yield np.hstack((behind, sample, ahead))


def get_generator(data, d, label=None):
    inputGenerator = pad(data=data)
    inputFFTGenerator = pad(data=fourier(data))
    referenceGenerator = pad(data=d['reference'])
    referenceFFTGenerator = pad(data=fourier([d['reference']]))
    for i in range(len(data)):
        inputFrequencies = get_frequencies(data, i)
        referenceFrequencies = get_frequencies(d['reference'], i)
        if type(label) != type(None):
            x = np.hstack(
                (next(inputGenerator), next(inputFFTGenerator), inputFrequencies, next(referenceGenerator),
                 next(referenceFFTGenerator), referenceFrequencies, label)
            )
        else:
            x = np.hstack(
                (next(inputGenerator), next(inputFFTGenerator), inputFrequencies, next(referenceGenerator),
                 next(referenceFFTGenerator), referenceFrequencies)
            )
        yield x.reshape(1, -1)


def fourier(signal):
    fft = np.fft.rfft(signal).real
    if len(fft.shape) == 3:
        fft = fft[0]
    return fft


def fourier(signal):
    return np.fft.fft2(signal).real


def get_frequencies(data, i):
    N = int(gc.bufferSamples / 2)
    if i == 0:
        return np.zeros(N)
    elif i < 999:
        m = int(gc.bufferSamples - i)
        k = np.zeros((m, 2))
        signal = np.vstack((k, data[0:i]))
    else:
        signal = data[i - 999: i]
    signal = signal.sum(axis=1) / 2
    t = scipy.arange(0, gc.bufferSeconds, gc.period)
    fft = abs(scipy.fft(signal))[range(N)]
    freqs = scipy.fftpack.fftfreq(signal.size, t[1] - t[0])[range(N)]
    return fft


def get_frequencies(data, i):
    N = int(gc.bufferSamples / 2)
    if i == 0:
        return np.zeros(N)
    elif i < (gc.bufferSamples - 1):
        signal = homogenize_length(data[0:1], int(gc.bufferSamples), 'start')
    else:
        signal = data[i - int(gc.bufferSamples - 1): i]
    signal = signal.sum(axis=1) / 2
    return abs(scipy.fft(signal))[range(N)]


def homogenize_length(signal, length, side='finish'):
    """
    If # of samples in signal is less than length, append samples of 0s to end of signal.
    Ensures all signals  are the same length.
    signal: numpy array
    length: int
    """
    difference = length - len(signal)
    if difference > 0:
        z = np.zeros((difference, signal.shape[1]))
        if side == 'finish':
            return np.vstack((signal, z))
        elif side == 'start':
            return np.vstack((z, signal))
    return signal


def interleave(signal):
    """
    Turn mono into stereo, leave stereo as is.
    If signal contains one channel, it will
    be duplicated so that the signal has two
    channels.
    signal: numpy array
    """
    if signal.shape[1] == 1:
        return np.hstack((signal, signal))
    else:
        return signal


def strip_ext(string):
    """
    Removes extension from string.
    """
    return string.split('.')[0]


def max_length(project):
    max = 0
    for subdir in get_subdirs(project):
        for f in get_files(subdir):
            data = wavio.read(f).data
            if len(data) > max:
                max = len(data)
    return max


def pad(data, samplingRate=gc.samplingRate, seconds=gc.seconds, paddingRate=gc.paddingRate):
    """
    This is a generator that yields one temporally padded sample at a time.
    samplingRate: int, number of samples per second in the signal (Hz)
    paddingRate: int, number of samples per second to use for padding (Hz)
    seconds: int, number of seconds to look behind/ahead (sec)
    """
    padLength = int(seconds * paddingRate)
    skips = int(samplingRate / paddingRate)
    length, width = data.shape
    for i, sample in enumerate(data):
        behind = look_behind(data, padLength, skips, length, width, i)
        ahead = look_ahead(data, padLength, skips, length, width, i)
        yield np.hstack((behind, sample, ahead))


def path_end(path):
    """
    Returns the last part of the path.
    E.g. path='dir1/dir2/file.ext' returns 'file.ext'.
    """
    return os.path.basename(os.path.normpath(path))


def printio(x, y, i, sr=gc.samplingRate):
    if i % sr * 5 == 0:
        sec = 1 + (i + 1) / (sr)
        print('-----sec: {}-----'.format(sec))
        print('x: {}'.format(x))
        print('x size: {}'.format(x.shape))
        print('y: {}'.format(y))
        print('-----------------\n')


def read(path, maxLength):
    """
    Reads wav file specified by path.
    Need to make work with any audio format.
    """
    signal = wavio.read(path)
    signal = interleave(homogenize_length(signal.data, maxLength))
    return signal


def read(path, maxLength):
    """
    Reads wav file specified by path.
    Need to make work with any audio format.
    """
    signal = wavio.read(path)
    signal = interleave(homogenize_length(signal.data, maxLength))
    return signal


def read(path, maxLength):
    """
    Reads wav file specified by path.
    Need to make work with any audio format.
    """
    signal = wavio.read(path)
    signal = interleave(homogenize_length(signal.data, maxLength))
    return signal


def scale(data):
    if gc.scale:
        data['input'] = data['input'] / gc.maxVal
        data['ref'] = data['ref'] / gc.maxVal
        data['output'] = data['output'] / gc.maxVal
    return data


def unscale(data):
    if gc.scale:
        data['input'] = data['input'] * gc.maxVal
        data['ref'] = data['ref'] * gc.maxVal
        data['output'] = data['output'] * gc.maxVal
    return data


def training_data_generator(metadata, sequencer=False):
    """
    This generator outputs training data for a model.
    Emits (x, y) pair. x vector has been padded with temporal context, y vector is untouched.
    """
    inputGenerator = pad(data=metadata['input'])
    inputFFTGenerator = pad(data=fourier(metadata['input']))
    referenceGenerator = pad(data=metadata['reference'])
    referenceFFTGenerator = pad(data=fourier([metadata['reference']]))
    if sequencer:
        o = np.vstack([metadata['output'] for _ in range(len(metadata['input']))])
    else:
        o = metadata['output']
    outputGenerator = pad(data=o, samplingRate=0, seconds=0)
    for i in range(len(metadata['input'])):
        inputFrequencies = get_frequencies(metadata['input'], i)
        referenceFrequencies = get_frequencies(metadata['reference'], i)
        x = np.hstack(
            (next(inputGenerator), next(inputFFTGenerator), inputFrequencies, next(referenceGenerator),
             next(referenceFFTGenerator), referenceFrequencies)
        )
        y = next(outputGenerator)
        yield x.reshape(1, -1), y.reshape(1, -1)


def train_transformation(
    timeRegressor, metadata, reference, maxLength, model, label, rate=global_config.rate,
    seconds=global_config.seconds, mode='t'
):
    print("metadata: ", metadata)
    iD, oD = dp.format_training_data(metadata, maxLength)
    generator = dp.training_data_generator(
        input=iD, reference=reference, output=oD, rate=rate, seconds=seconds,
        mode=mode
    )
    timeRegressor.train_nn(model, generator, maxLength)
    return iD['data'], oD['data']


def train_sequencer(
    timeRegressor, metadata, reference, maxLength, model, label, rate=global_config.rate,
    seconds=global_config.seconds, mode='t'
):
    iD = dp.format_sequencer_training_data(metadata, maxLength)
    generator = dp.sequencer_training_data_generator(
        input=iD, reference=reference, output=label, rate=rate,
        seconds=seconds, mode=mode
    )
    timeRegressor.train_nn('Sequencer', generator, maxLength)


def training_data_generator(metadata, sequencer=False):
    """
    This function is a generator that outputs the training data for a
    model. It emits an (x, y) pair where the x vector has been padded with
    temporal context, and the y vector is untouched.
    input: dict
    output: dict
    reference: dict
    rate: int
    seconds: int
    """
    inputGenerator = pad_data(data=metadata['input'])

    referenceGenerator = pad_data(data=metadata['reference'])

    if sequencer:
        o = np.vstack([metadata['output'] for _ in range(len(metadata['input']))])
    else:
        o = metadata['output']
    outputGenerator = pad_data(data=o, samplingRate=0 if sequencer else gc.samplingRate, seconds=0)

    for i in range(len(metadata['input'])):
        x = np.hstack((next(inputGenerator), next(referenceGenerator)))
        y = next(outputGenerator)
        yield x.reshape(1, -1), y.reshape(1, -1)


def training_data_generator(metadata, sequencer=False, label=None):
    """
    This generator outputs training data for a model.
    Emits (x, y) pair. x vector has been padded with temporal context.
    """
    inputGenerator = pad(data=metadata['input'])
    inputFFTGenerator = pad(data=fourier(metadata['input']))
    referenceGenerator = pad(data=metadata['reference'])
    referenceFFTGenerator = pad(data=fourier([metadata['reference']]))
    if sequencer:
        o = np.vstack([metadata['label'] for _ in range(len(metadata['input']))])
    else:
        o = metadata['output']
    outputGenerator = pad(data=o, samplingRate=0, seconds=0)
    for i in range(len(metadata['input'])):
        inputFrequencies = get_frequencies(metadata['input'], i)
        referenceFrequencies = get_frequencies(metadata['reference'], i)
        if type(label) != type(None):
            x = np.hstack(
                (next(inputGenerator), next(inputFFTGenerator), inputFrequencies, next(referenceGenerator),
                 next(referenceFFTGenerator), referenceFrequencies, label)
            )
        else:
            x = np.hstack(
                (next(inputGenerator), next(inputFFTGenerator), inputFrequencies, next(referenceGenerator),
                 next(referenceFFTGenerator), referenceFrequencies)
            )
        y = next(outputGenerator)
        yield x.reshape(1, -1), y.reshape(1, -1)


def wav_info(path, length):
    wav = wave.open(path, 'r')
    samplingRate = wav.getframerate()
    sampleWidth = wav.getsampwidth()
    wav = read(path, length)
    nsamples, nchannels = wav.shape
    return {'nchannels': nchannels, 'sampleWidth': sampleWidth, 'samplingRate': samplingRate, 'nsamples': nsamples}


def wav_info(path):
    info = {}
    wav = wave.open(path, 'r')
    info['channels'] = wav.getnchannels()
    info['sampleWidth'] = wav.getsampwidth()
    info['samplingRate'] = wav.getframerate()
    info['samples'] = wav.getnframes()
    return info


def write(data, path):
    wavio.write(path + '.wav', data=data, rate=gc.samplingRate, sampwidth=gc.sampleWidth, scale='none')


def write(data, path, info):
    wavio.write(path + '.wav', data, info['samplingRate'], sampwidth=info['sampleWidth'])


def write_wav(path, data, info):
    wav = wave.open(path, 'w')
    wav.setnchannels(info['channels'])
    wav.setsampwidth(info['sampleWidth'])
    wav.setframerate(info['samplingRate'])
    wav.setnframes(info['samples'])
    wav.writeframes(data)


class Model:
    def __init__(self):
        self.model = None

    def set_model(self, name: str):
        print('>> Running {} <<'.format(name))
        if gc.multiModel or name == 'Sequencer':
            self.name = name
        else:
            self.name = 'Model'
        self.filename = os.path.join(gc.modelsDir[gc.aiBackend], '{0}.{1}'.format(self.name, gc.modelExt[gc.aiBackend]))
        if os.path.isfile(self.filename):
            self.model = self.load()
        else:
            self.model = MODELS[name]

    def save(self):
        if os.path.isfile(self.filename):
            if gc.aiBackend == 'sklearn':
                joblib.dump(self.model, '{}_tmp'.format(self.filename))
            if gc.aiBackend == 'keras':
                self.model.save('{}_tmp'.format(self.filename))
            os.remove(self.filename)
            os.rename('{}_tmp'.format(self.filename), self.filename)
        else:
            if gc.aiBackend == 'sklearn':
                joblib.dump(self.model, self.filename)
            if gc.aiBackend == 'keras':
                self.model.save(self.filename)

    def load(self):
        if gc.aiBackend == 'sklearn':
            return joblib.load(self.filename)
        if gc.aiBackend == 'keras':
            return load_model(self.filename)

    def close(self):
        self.model = None
        del self.model

    def train(self, generator, length):
        for i in range(0, length, gc.batchSize):
            x, y = next(generator)
            for j in range(gc.batchSize - 1):
                x1, y1 = next(generator)
                x = np.vstack([x, x1])
                y = np.vstack([y, y1])
                i += 1
                if i >= length:
                    break
            # printio(x, y, i - (gc.batchSize - 1))
            self.partial(x, y)
        self.save()
        self.close()


class Model:
    def __init__(self):
        self.model = None

    def set_model(self, name: str):
        print('>> Running {} <<'.format(name))
        if gc.multiModel or name == 'Sequencer':
            self.name = name
        else:
            self.name = 'Model'
        self.filename = os.path.join(gc.modelsDir[gc.aiBackend], '{0}.{1}'.format(self.name, gc.modelExt[gc.aiBackend]))
        if os.path.isfile(self.filename):
            self.model = self.load()
        else:
            self.model = MODELS[name]

    def save(self):
        if os.path.isfile(self.filename):
            if gc.aiBackend == 'sklearn':
                joblib.dump(self.model, '{}_tmp'.format(self.filename))
            if gc.aiBackend == 'keras':
                self.model.save('{}_tmp'.format(self.filename))
            os.remove(self.filename)
            os.rename('{}_tmp'.format(self.filename), self.filename)
        else:
            if gc.aiBackend == 'sklearn':
                joblib.dump(self.model, self.filename)
            if gc.aiBackend == 'keras':
                self.model.save(self.filename)

    def load(self):
        if gc.aiBackend == 'sklearn':
            return joblib.load(self.filename)
        if gc.aiBackend == 'keras':
            return load_model(self.filename)

    def close(self):
        self.model = None
        del self.model

    def train(self, generator, length):
        for i in range(0, length, gc.batchSize):
            x, y = next(generator)
            for j in range(gc.batchSize - 1):
                x1, y1 = next(generator)
                x = np.vstack([x, x1])
                y = np.vstack([y, y1])
                i += 1
                if i >= length:
                    break
            # printio(x, y, i - (gc.batchSize - 1))
            self.partial(x, y)
        self.save()
        self.close()

    def train_transformation(self, metadata):
        self.set_model(gc.transfer2[standardize_string(metadata['model'])])
        self.train(dp.training_data_generator(metadata=metadata, label=metadata['label']), metadata['maxLength'])
        self.set_model('Sequencer')
        metadata['model'] = 'sequencer'
        self.train(dp.training_data_generator(metadata=metadata, sequencer=True), metadata['maxLength'])
        if metadata['last']:
            self.set_model('Sequencer')
            metadata['reference'] = dp.update_reference(metadata)
            metadata['label'] = gc.labels['Stop']
            self.train(dp.training_data_generator(metadata=metadata, sequencer=True), metadata['maxLength'])

    def predict(self, data, d, labelVector=None):
        generator = dp.get_generator(data=data, d=d, label=labelVector)
        if self.name == 'Sequencer':
            result = np.zeros((1, len(gc.labels)))
        else:
            result = []
        for i in range(d['maxLength']):
            x = next(generator)
            if gc.aiBackend == 'sklearn':
                m = self.model.predict(x)
            if gc.aiBackend == 'keras':
                m = self.model.predict_on_batch(x)
            if self.name == 'Sequencer':
                result = np.add(result, m)
            else:
                result.append(m)
        if self.name == 'Sequencer':
            r = np.zeros_like(result)
            r[np.arange(len(result)), result.argmax(1)] = 1
            label = get_label(r[0])
            return label, r[0]
        else:
            return np.vstack(result)


class Model:
    def __init__(self):
        self.model = None

    def set_model(self, name: str, mode=''):
        if mode == 'train':
            print('>> Training {} <<'.format(name))
        elif mode == 'eval':
            print('>> Evaluating {} <<'.format(name))
        else:
            print('>> Running {} <<'.format(name))

        self.mode = mode
        if gc.multiModel or name == 'Sequencer':
            self.name = name
        else:
            self.name = 'Model'
        self.filename = os.path.join(gc.modelsDir[gc.aiBackend], '{0}.{1}'.format(self.name, gc.modelExt[gc.aiBackend]))
        if os.path.isfile(self.filename):
            self.model = self.load()
        else:
            self.model = MODELS[name]

    def save(self):
        if os.path.isfile(self.filename):
            if gc.aiBackend == 'sklearn':
                attempt(joblib.dump, [self.model, '{}_tmp'.format(self.filename)])
            if gc.aiBackend == 'keras':
                attempt(self.model.save, ['{}_tmp'.format(self.filename)])
            attempt(os.remove, [self.filename])
            attempt(os.rename, ['{}_tmp'.format(self.filename), self.filename])
        else:
            if gc.aiBackend == 'sklearn':
                attempt(joblib.dump, [self.model, self.filename])
            if gc.aiBackend == 'keras':
                attempt(self.model.save, [self.filename])

    def load(self):
        if gc.aiBackend == 'sklearn':
            return attempt(joblib.load, [self.filename])
        if gc.aiBackend == 'keras':
            return attempt(load_model, [self.filename, custom_objects])

    def close(self):
        self.model = None
        del self.model

    def train(self, generator, length):
        for i in range(0, length, gc.batchSize):
            x, y = next(generator)
            for j in range(gc.batchSize - 1):
                x1, y1 = next(generator)
                x = np.vstack([x, x1])
                y = np.vstack([y, y1])
                i += 1
                if i >= length:
                    break
            # printio(x, y, i - (gc.batchSize - 1))
            self.partial(x, y)
        self.save()
        self.close()

    def train_transformation(self, metadata):
        self.set_model(get_label(metadata['label']), 'train')
        self.train(dp.training_data_generator(metadata=metadata, label=metadata['label']), metadata['maxLength'])
        self.set_model('Sequencer', 'train')
        self.train(dp.training_data_generator(metadata=metadata, sequencer=True), metadata['maxLength'])
        if metadata['last']:
            self.set_model('Sequencer', 'train')
            metadata['reference'] = dp.update_reference(metadata)
            label = np.copy(metadata['label'])
            metadata['label'] = gc.labels['Stop']
            self.train(dp.training_data_generator(metadata=metadata, sequencer=True), metadata['maxLength'])
            metadata['label'] = label

    def partial(self, x, y):
        if gc.aiBackend == 'sklearn':
            if self.name == 'Sequencer':
                self.model.partial_fit(x, y, classes=[i for i in range(len(y[0]))])
            else:
                self.model.partial_fit(x, y)
        if gc.aiBackend == 'keras':
            self.model.train_on_batch(x, y)

    def predict(self, data, d, labelVector=None):
        generator = dp.get_generator(data=data, d=d, label=labelVector)
        if self.name == 'Sequencer':
            bo = []
            result = np.zeros((1, len(gc.labels)))
        else:
            result = []
        for i in range(d['maxLength']):
            x = next(generator)
            if gc.aiBackend == 'sklearn':
                m = self.model.predict(x)
            if gc.aiBackend == 'keras':
                m = self.model.predict_on_batch(x)
            if self.name == 'Sequencer':
                if self.mode == 'eval':
                    z = np.zeros_like(m)
                    z[np.arange(len(m)), m.argmax(1)] = 1
                    bo.append(z[0])
                result = np.add(result, m)
            else:
                result.append(m)
        if self.name == 'Sequencer':
            r = np.zeros_like(result)
            r[np.arange(len(result)), result.argmax(1)] = 1
            label = get_label(r[0])
            return label, r[0], np.array(bo).astype(np.int8)
        else:
            return np.vstack(result)

    def evaluate_regression(self, input, output, prediction):
        def cos_sim(x, y):
            return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

        monoinput = input.sum(axis=1) / 2
        monooutput = output.sum(axis=1) / 2
        monoprediction = prediction.sum(axis=1) / 2
        op_euclidian_dist = np.linalg.norm(monooutput - monoprediction)
        op_cos_sim = cos_sim(monooutput, monoprediction)
        reward = gc.eP * op_euclidian_dist + gc.cP * abs(1 - op_cos_sim)
        try:
            return {
                'io_corr':           np.dot(monoinput, monooutput.T), 'op_corr': np.dot(monooutput, monoprediction.T),
                'ip_corr':           np.dot(monoinput, monoprediction.T),
                'io_euclidian_dist': np.linalg.norm(monoinput - monooutput), 'op_euclidian_dist': op_euclidian_dist,
                'ip_euclidian_dist': np.linalg.norm(monoinput - monoprediction),
                'io_cos_sim':        cos_sim(monoinput, monooutput), 'op_cos_sim': op_cos_sim,
                'ip_cos_sim':        cos_sim(monoinput, monoprediction),
                'evs':               sklearn.metrics.explained_variance_score(output, prediction),
                'mse':               sklearn.metrics.mean_squared_error(output, prediction),
                'monomse':           sklearn.metrics.mean_squared_error(monooutput, monoprediction),
                'msle':              sklearn.metrics.mean_squared_log_error(np.abs(output), np.abs(prediction)),
                'monomsle':          sklearn.metrics.mean_squared_log_error(np.abs(monooutput), np.abs(monoprediction)),
                'monomae':           sklearn.metrics.mean_absolute_error(monooutput, monoprediction),
                'monomedae':         sklearn.metrics.median_absolute_error(monooutput, monoprediction),
                'r2':                sklearn.metrics.r2_score(output, prediction), 'reward': reward,
            }
        except Exception as e:
            print(e)
            print(input)

    def evaluate_classification(self, input, output):
        return {  # 'report': sklearn.metrics.classification_report(input, output, output_dict=True),
            'as': 100 * sklearn.metrics.accuracy_score(input, output),
        }


class MLModel:
    def __init__(self):
        self.model = None

    def set_model(self, model):
        """
        model: str, name of model
        """
        self.name = model
        self.filename = os.path.join(gc.modelsDir, '{0}.{1}'.format(model, gc.modelExt))
        if os.path.isfile(self.filename):
            self.model = self.load()
        else:
            self.model = gc.MODELS[model]

    def save(self):
        if os.path.isfile(self.filename):
            joblib.dump(self.model, '{}_tmp'.format(self.filename))
            os.remove(self.filename)
            os.rename('{}_tmp'.format(self.filename), self.filename)
        else:
            joblib.dump(self.model, self.filename)

    def load(self):
        return joblib.load(self.filename)

    def close(self):
        self.model = None
        del self.model

    def predict(self, data, d):
        generator = dp.get_generator(data=data, d=d)
        if self.name == 'Sequencer':
            result = np.zeros((1, len(gc.LABELS)))
        else:
            result = []
        for i in range(d['maxLength']):
            x = next(generator)
            m = self.model.predict(x)
            if self.name == 'Sequencer':
                result = np.add(result, m)
            else:
                result.append(m)
        if self.name == 'Sequencer':
            r = np.zeros_like(result)
            r[np.arange(len(result)), result.argmax(1)] = 1
            label = get_label(r[0])
            return label
        else:
            return np.vstack(result)


class MLModel:
    def __init__(self):
        self.model = None

    def set_model(self, model):
        """
        model: str, name of model
        """
        self.name = model
        self.filename = os.path.join(gc.modelsDir, '{0}.{1}'.format(model, gc.modelExt))
        if os.path.isfile(self.filename):
            self.model = self.load()
        else:
            self.model = ai.MODELS[model]

    def save(self):
        if os.path.isfile(self.filename):
            joblib.dump(self.model, '{}_tmp'.format(self.filename))
            os.remove(self.filename)
            os.rename('{}_tmp'.format(self.filename), self.filename)
        else:
            joblib.dump(self.model, self.filename)

    def load(self):
        return joblib.load(self.filename)

    def close(self):
        self.model = None
        del self.model

    def predict(self, data, d):
        generator = dp.get_generator(data=data, d=d)
        if self.name == 'Sequencer':
            result = np.zeros((1, len(ai.LABELS)))
        else:
            result = []
        for i in range(d['maxLength']):
            x = next(generator)
            m = self.model.predict(x)
            if self.name == 'Sequencer':
                result = np.add(result, m)
            else:
                result.append(m)
        if self.name == 'Sequencer':
            r = np.zeros_like(result)
            r[np.arange(len(result)), result.argmax(1)] = 1
            label = ai.get_label(r[0])
            return label
        else:
            return np.vstack(result)


class MLModel:
    def __init__(self):
        self.model = None

    def set_model(self, model):
        """
        model: str, name of model
        """
        self.filename = os.path.join(gc.modelsDir, '{}.joblib'.format(model))
        if os.path.isfile(self.filename):
            self.model = self.load()
        else:
            self.model = gc.MODELS[model]

    def save(self):
        if os.path.isfile(self.filename):
            joblib.dump(self.model, self.filename + '_tmp')
            os.remove(self.filename)
            os.rename(self.filename + '_tmp', self.filename)

    def load(self):
        return joblib.load(self.filename)

    def close(self):
        self.model = None
        del self.model
