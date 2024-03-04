import numpy as np
from tensorflow import set_random_seed

import configuration as gc

set_random_seed(1)
np.random.seed(3)


def regressor(**settings):
    return MLPRegressor(**settings)


def classifier(**settings):
    return MLPClassifier(**settings)


def run(cState, cModel, cTargetModel, cMemory, cUncertainty, rState, rModel, rTargetModel, rMemory, rUncertainty):
    for e in range(gc.episodes):
        cTargetModel.set_weights(cModel.get_weights())
        rTargetModel.set_weights(rModel.get_weights())
        print_episode_start(e, cState)
        cRewardSum = 0
        rRewardSum = 0
        for time in range(gc.steps):
            cAction = act(cModel, cState, cUncertainty)
            rAction = act(rModel, rState, rUncertainty)
            cNewState, rNewState, cReward, rReward, done = step(cState, rState, cAction, rAction)
            cRewardSum += cReward
            rRewardSum += rReward
            cMemory.append([cState, cAction, cReward, cNewState, done])
            rMemory.append([rState, rAction, rReward, rNewState, done])
            cState = cNewState
            rState = rNewState
            if len(cMemory) > gc.memBatch:
                cModel, cTargetModel = replay(cModel, cTargetModel, cMemory)
                cUncertainty = update_uncertainty(cUncertainty, time, cReward, cRewardSum)
                cMemory = forget(cMemory)
            if len(rMemory) > gc.memBatch:
                rModel, rTargetModel = replay(rModel, rTargetModel, rMemory)
                rUncertainty = update_uncertainty(rUncertainty, time, rReward, rRewardSum)
                rMemory = forget(rMemory)
            if time % 100 == 0:
                print_time(time, cState, rUncertainty, rReward, rRewardSum)
                print_time(time, rState, rUncertainty, rReward, rRewardSum)
            if time % 1000 == 0:
                cTargetModel.set_weights(cModel.get_weights())
                rTargetModel.set_weights(rModel.get_weights())
    return cModel, rModel, cTargetModel, rTargetModel, cMemory, rMemory


def enter_the_matrix(MODELS):
    totalCErr = 0
    totalRErr = 0
    for permutation in range(gc.permutations):
        print('- - - - Permutation: {} - - - -'.format(permutation))
        for epoch in range(gc.epochs):
            print('---------- Epoch: {} ----------'.format(epoch))
            cErr, rErr = trainer.run(ai.Model(), MODELS, permutation, epoch)
            totalCErr += cErr
            totalRErr += rErr
            print('-------- End epoch: {} --------'.format(epoch))
        print('- - - End Permutation: {} - - -'.format(permutation))
    return totalCErr, totalRErr


'''run(cState=random_initial_state(),
    cModel=get_model(),
    cTargetModel=get_model(),
    cMemory=deque(maxlen=gc.memSize),
    cUncertainty=1.0,
    rState=random_initial_state(),
    rModel=get_model(),
    rTargetModel=get_model(),
    rMemory=deque(maxlen=gc.memSize),
    rUncertainty=1.0)'''


def kR():
    model = Sequential()
    model.add(
        Dense(gc.inputSize * 2, activation='relu', input_shape=regressorInputShape, kernel_initializer='lecun_normal',
              bias_initializer='lecun_normal', kernel_regularizer=keras.regularizers.l2(0.01),
              bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(
        Dense(gc.inputSize * 3, activation='relu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
              kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(
        Dense(gc.inputSize * 4, activation='relu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
              kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(
        Dense(gc.inputSize * 4, activation='relu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
              kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(
        Dense(gc.inputSize * 3, activation='relu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
              kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(
        Dense(gc.inputSize * 2, activation='relu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
              kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(
        Dense(gc.inputSize * 1, activation='relu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
              kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(2, activation='linear', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.compile(optimizer='adam', loss='mse')
    return model


def kC():
    size = len(gc.models)
    model = Sequential()
    model.add(Dense(10 * size, activation='relu', input_shape=classifierInputShape, kernel_initializer='lecun_normal',
                    bias_initializer='lecun_normal', kernel_regularizer=keras.regularizers.l2(0.01),
                    bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(9 * size, activation='relu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(8 * size, activation='relu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(7 * size, activation='relu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(6 * size, activation='relu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(4 * size, activation='relu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(3 * size, activation='relu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(2 * size, activation='relu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(size, activation='softmax', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return model


classifierInputShape = (gc.inputSize,)
if gc.multiModel:
    regressorInputShape = (gc.inputSize,)
else:
    regressorInputShape = (gc.inputSize + len(gc.models),)
if gc.multiModel:
    MODELS = {model: kR() if model != 'Sequencer' else kC() for model in gc.models}
else:
    MODELS = {'Model': kR(), 'Sequencer': kC()}
enter_the_matrix(MODELS)

import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from tensorflow import set_random_seed

import configuration as gc
import trainer
import utils.ai as uai
from utils.researcher import print_time, print_episode_start, ac, update_uncertainty, forget, replay, act, int_to_state

set_random_seed(1)
np.random.seed(3)


def kerasNN(layers, compileSettings):
    def get_activations(KWARGS):
        if 'activation' in KWARGS:
            if KWARGS['activation'] in activations:
                KWARGS['activation'] = activations[KWARGS['activation']]

    def get_regularizers(KWARGS):
        for w in ['activation_regularizer', 'bias_regularizer', 'kernel_regularizer']:
            if w in KWARGS:
                KWARGS[w] = regularizersDict[KWARGS[w][0]](KWARGS[w][1])

    model = keras.Sequential()
    for layer in layers:
        LAYER = layer['type']
        if LAYER != None:
            KWARGS = {key: layer[key] for key in layer if key != 'type'}
            get_activations(KWARGS)
            get_regularizers(KWARGS)
            model.add(layersDict[LAYER](**KWARGS))
    model.compile(**compileSettings)
    return model


def regressor(**settings):
    return MLPRegressor(**settings)


def classifier(**settings):
    return MLPClassifier(**settings)


def run(cState, cModel, cTargetModel, cMemory, cUncertainty, rState, rModel, rTargetModel, rMemory, rUncertainty):
    for e in range(gc.episodes):
        cTargetModel.set_weights(cModel.get_weights())
        rTargetModel.set_weights(rModel.get_weights())
        print_episode_start(e, cState)
        cRewardSum = 0
        rRewardSum = 0
        for time in range(gc.steps):
            cAction = act(cModel, cState, cUncertainty)
            rAction = act(rModel, rState, rUncertainty)
            cNewState, rNewState, cReward, rReward, done = step(cState, rState, cAction, rAction)
            cRewardSum += cReward
            rRewardSum += rReward
            cMemory.append([cState, cAction, cReward, cNewState, done])
            rMemory.append([rState, rAction, rReward, rNewState, done])
            cState = cNewState
            rState = rNewState
            if len(cMemory) > gc.memBatch:
                cModel, cTargetModel = replay(cModel, cTargetModel, cMemory)
                cUncertainty = update_uncertainty(cUncertainty, time, cReward, cRewardSum)
                cMemory = forget(cMemory)
            if len(rMemory) > gc.memBatch:
                rModel, rTargetModel = replay(rModel, rTargetModel, rMemory)
                rUncertainty = update_uncertainty(rUncertainty, time, rReward, rRewardSum)
                rMemory = forget(rMemory)
            if time % 100 == 0:
                print_time(time, cState, rUncertainty, rReward, rRewardSum)
                print_time(time, rState, rUncertainty, rReward, rRewardSum)
            if time % 1000 == 0:
                cTargetModel.set_weights(cModel.get_weights())
                rTargetModel.set_weights(rModel.get_weights())
    return cModel, rModel, cTargetModel, rTargetModel, cMemory, rMemory


def step(cState, rState, cAction, rAction):
    try:
        cs = np.argmax(cState) + ac[cAction]
        rs = np.argmax(rState) + ac[rAction]
        cNewState = int_to_state(cs)
        rNewState = int_to_state(rs)
        cReward, rReward = get_reward(cs, rs)
        done = False
    except Exception as e:
        print(e)
        cNewState, rNewState = cState, rState  # TODO ??
        cReward = -gc.penalty
        rReward = -gc.penalty
        done = True
    return cNewState, rNewState, cReward, rReward, done


def get_reward(cState, rState):
    sklearnNNSettings = {'c': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001,
                               'batch_size': 'auto', 'learning_rate': 'constant', 'learning_rate_init': 0.001,
                               'power_t': 0.5, 'max_iter': 1, 'shuffle': True, 'random_state': None, 'tol': 1e-4,
                               'verbose': False, 'momentum': 0.9,  # 'nesterovs_momentum': True,
                               'early_stopping': False, 'validation_fraction': 0.1, 'beta_1': 0.999, 'beta_2': 0.999,
                               'uncertainty': 1e-8,  # 'n_iter_no_change': 10,
                               },
                         'r': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001,
                               'batch_size': 'auto', 'learning_rate': 'constant', 'learning_rate_init': 1,
                               'power_t': 0.5, 'max_iter': 1, 'shuffle': True, 'random_state': None, 'tol': 1e-4,
                               'verbose': False, 'warm_start': False, 'momentum': 0.9,  # 'nesterovs_momentum': True,
                               'early_stopping': False, 'validation_fraction': 0.1, 'beta_1': 0.9, 'beta_2': 0.999,
                               'uncertainty': 1e-08,  # 'n_iter_no_change': 10
                               }, }
    classifierInputShape = (gc.inputSize,)
    if gc.multiModel:
        regressorInputShape = (gc.inputSize,)
    else:
        regressorInputShape = (gc.inputSize + len(gc.models),)
    kerasNNSettings = {'c': {
        'layers': [{'type': 'dense', 'units': 1024, 'activation': 'swish', 'input_shape': classifierInputShape, },
                   {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': 512, 'activation': 'swish', 'input_shape': classifierInputShape, },
                   {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': 256, 'activation': 'swish', 'input_shape': classifierInputShape, },
                   {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': len(gc.models), 'activation': 'softmax',

                    }, ], 'compileSettings': {'loss': 'categorical_crossentropy', 'optimizer': 'rmsprop', }}, 'r': {
        'layers': [{'type': 'dense', 'units': 24, 'activation': 'swish', 'input_shape': regressorInputShape,
                    'kernel_initializer': 'lecun_normal', 'bias_initializer': 'lecun_normal',
                    'kernel_regularizer': ('L2', 0.01), 'bias_regularizer': ('L2', 0.01), },
                   {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': 24, 'activation': 'swish', 'kernel_initializer': 'lecun_normal',
                    'bias_initializer': 'lecun_normal', 'kernel_regularizer': ('L2', 0.01),
                    'bias_regularizer': ('L2', 0.01), }, {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': 24, 'activation': 'swish', 'kernel_initializer': 'lecun_normal',
                    'bias_initializer': 'lecun_normal', 'kernel_regularizer': ('L2', 0.01),
                    'bias_regularizer': ('L2', 0.01), }, {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': 24, 'activation': 'swish', 'input_shape': regressorInputShape,
                    'kernel_initializer': 'lecun_normal', 'bias_initializer': 'lecun_normal',
                    'kernel_regularizer': ('L2', 0.01), 'bias_regularizer': ('L2', 0.01), },
                   {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': 24, 'activation': 'swish', 'kernel_initializer': 'lecun_normal',
                    'bias_initializer': 'lecun_normal', 'kernel_regularizer': ('L2', 0.01),
                    'bias_regularizer': ('L2', 0.01), }, {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': 24, 'activation': 'swish', 'kernel_initializer': 'lecun_normal',
                    'bias_initializer': 'lecun_normal', 'kernel_regularizer': ('L2', 0.01),
                    'bias_regularizer': ('L2', 0.01), }, {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': 24, 'activation': 'swish', 'input_shape': regressorInputShape,
                    'kernel_initializer': 'lecun_normal', 'bias_initializer': 'lecun_normal',
                    'kernel_regularizer': ('L2', 0.01), 'bias_regularizer': ('L2', 0.01), },
                   {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': 24, 'activation': 'swish', 'kernel_initializer': 'lecun_normal',
                    'bias_initializer': 'lecun_normal', 'kernel_regularizer': ('L2', 0.01),
                    'bias_regularizer': ('L2', 0.01), }, {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': 24, 'activation': 'swish', 'kernel_initializer': 'lecun_normal',
                    'bias_initializer': 'lecun_normal', 'kernel_regularizer': ('L2', 0.01),
                    'bias_regularizer': ('L2', 0.01), }, {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': 2, 'activation': 'linear', 'kernel_initializer': 'lecun_normal',
                    'bias_initializer': 'lecun_normal', 'kernel_regularizer': ('L2', 0.01),
                    'bias_regularizer': ('L2', 0.01), }, ], 'compileSettings': {'loss': 'mse', 'optimizer': 'adam', }}}
    if gc.aiLib == 'keras':
        get_model_info(kerasNNSettings)
    if gc.aiLib == 'sklearn':
        get_model_info(sklearnNNSettings)

    if gc.aiLib == 'sklearn':
        if gc.multiModel:
            MODELS = {model: regressor(**sklearnNNSettings['r']) if model != 'Sequencer' else classifier(
                **sklearnNNSettings['c']) for model in gc.models}
        else:
            MODELS = {model: regressor(**sklearnNNSettings['r']) if model != 'Sequencer' else classifier(
                **sklearnNNSettings['c']) for model in gc.models}

    elif gc.aiLib == 'keras':
        if gc.multiModel:
            MODELS = {
                model: kerasNN(**kerasNNSettings['r']) if model != 'Sequencer' else kerasNN(**kerasNNSettings['c']) for
                model in gc.models}
        else:
            MODELS = {
                model: kerasNN(**kerasNNSettings['r']) if model != 'Sequencer' else kerasNN(**kerasNNSettings['c']) for
                model in gc.models}
    enter_the_matrix(MODELS)


def enter_the_matrix(MODELS):
    totalCErr = 0
    totalRErr = 0
    for permutation in range(gc.permutations):
        print('- - - - Permutation: {} - - - -'.format(permutation))
        for epoch in range(gc.epochs):
            print('---------- Epoch: {} ----------'.format(epoch))
            cErr, rErr = trainer.run(ai.Model(), MODELS, permutation, epoch)
            totalCErr += cErr
            totalRErr += rErr
            print('-------- End epoch: {} --------'.format(epoch))
        print('- - - End Permutation: {} - - -'.format(permutation))
    return totalCErr, totalRErr


'''run(cState=random_initial_state(),
    cModel=get_model(),
    cTargetModel=get_model(),
    cMemory=deque(maxlen=gc.memSize),
    cUncertainty=1.0,
    rState=random_initial_state(),
    rModel=get_model(),
    rTargetModel=get_model(),
    rMemory=deque(maxlen=gc.memSize),
    rUncertainty=1.0)'''


def kR():
    hiddenLayers = []
    model = Sequential()
    # input layer
    model.add(Dense(int(gc.inputSize * 4), activation='relu', input_shape=regressorInputShape,
                    kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(rate=0.2))

    # hidden layers
    for layer in hiddenLayers:
        model.add(Dense(units=layer[0], activation=layer[1], kernel_initializer='lecun_normal',
                        bias_initializer='lecun_normal', kernel_regularizer=keras.regularizers.l2(0.01),
                        bias_regularizer=keras.regularizers.l2(0.01)))
        model.add(Dropout(rate=0.2))

    # output layer
    model.add(
        Dense(gc.nChannels, activation='linear', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
              kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))
    model.compile(optimizer='adam', loss='mse')
    return model


def kC():
    size = len(gc.models)
    model = Sequential()
    model.add(Dense(size, activation=uai.swish, input_shape=classifierInputShape, kernel_initializer='lecun_normal',
                    bias_initializer='lecun_normal'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(size, activation='softmax', kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(0.01)))
    model.compile(loss='categorical_hinge', optimizer='adadelta', )
    return model


classifierInputShape = (gc.inputSize,)
if gc.multiModel:
    regressorInputShape = (gc.inputSize,)
else:
    regressorInputShape = (gc.inputSize + len(gc.models),)
if gc.multiModel:
    MODELS = {model: kR() if model != 'Sequencer' else kC() for model in gc.models}
else:
    MODELS = {'Model': kR(), 'Sequencer': kC()}
enter_the_matrix(MODELS)

from tensorflow import set_random_seed

import configuration as gc

set_random_seed(1)


def kerasNN(layers, compileSettings):
    def get_activations(KWARGS):
        if 'activation' in KWARGS:
            if KWARGS['activation'] in activations:
                KWARGS['activation'] = activations[KWARGS['activation']]

    def get_regularizers(KWARGS):
        for w in ['activation_regularizer', 'bias_regularizer', 'kernel_regularizer']:
            if w in KWARGS:
                KWARGS[w] = regularizersDict[KWARGS[w][0]](KWARGS[w][1])

    model = keras.Sequential()
    for layer in layers:
        LAYER = layer['type']
        KWARGS = {key: layer[key] for key in layer if key != 'type'}
        get_activations(KWARGS)
        get_regularizers(KWARGS)
        model.add(layersDict[LAYER](**KWARGS))
    model.compile(**compileSettings)
    return model


def regressor(**settings):
    return MLPRegressor(**settings)


def classifier(**settings):
    return MLPClassifier(**settings)


def run():
    sklearnNNSettings = {'c': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001,
                               'batch_size': 'auto', 'learning_rate': 'constant', 'learning_rate_init': 0.001,
                               'power_t': 0.5, 'max_iter': 1, 'shuffle': True, 'random_state': None, 'tol': 1e-4,
                               'verbose': False, 'momentum': 0.9,  # 'nesterovs_momentum': True,
                               'early_stopping': False, 'validation_fraction': 0.1, 'beta_1': 0.999, 'beta_2': 0.999,
                               'uncertainty': 1e-8,  # 'n_iter_no_change': 10,
                               },
                         'r': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001,
                               'batch_size': 'auto', 'learning_rate': 'constant', 'learning_rate_init': 1,
                               'power_t': 0.5, 'max_iter': 1, 'shuffle': True, 'random_state': None, 'tol': 1e-4,
                               'verbose': False, 'warm_start': False, 'momentum': 0.9,  # 'nesterovs_momentum': True,
                               'early_stopping': False, 'validation_fraction': 0.1, 'beta_1': 0.9, 'beta_2': 0.999,
                               'uncertainty': 1e-08,  # 'n_iter_no_change': 10
                               }, }
    classifierInputShape = (gc.inputSize,)
    if gc.multiModel:
        regressorInputShape = (gc.inputSize,)
    else:
        regressorInputShape = (gc.inputSize + len(gc.models),)
    kerasNNSettings = {'c': {
        'layers': [{'type': 'dense', 'units': 20, 'activation': 'swish', 'input_shape': classifierInputShape, },
                   {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': len(gc.models), 'activation': 'softmax',

                    }, ], 'compileSettings': {'loss': 'categorical_crossentropy', 'optimizer': 'rmsprop', }}, 'r': {
        'layers': [{'type': 'dense', 'units': 24, 'activation': 'swish', 'input_shape': regressorInputShape,
                    'kernel_initializer': 'lecun_normal', 'bias_initializer': 'lecun_normal',
                    'kernel_regularizer': ('L2', 0.01), 'bias_regularizer': ('L2', 0.01), },
                   {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': 24, 'activation': 'relu', 'kernel_initializer': 'lecun_normal',
                    'bias_initializer': 'lecun_normal', 'kernel_regularizer': ('L2', 0.01),
                    'bias_regularizer': ('L2', 0.01), }, {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': 24, 'activation': 'relu', 'kernel_initializer': 'lecun_normal',
                    'bias_initializer': 'lecun_normal', 'kernel_regularizer': ('L2', 0.01),
                    'bias_regularizer': ('L2', 0.01), }, {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': 2, 'activation': 'linear', 'kernel_initializer': 'lecun_normal',
                    'bias_initializer': 'lecun_normal', 'kernel_regularizer': ('L2', 0.01),
                    'bias_regularizer': ('L2', 0.01), }, ], 'compileSettings': {'loss': 'mse', 'optimizer': 'adam', }}}
    if gc.aiLib == 'keras':
        get_model_info(kerasNNSettings)
    if gc.aiLib == 'sklearn':
        get_model_info(sklearnNNSettings)

    if gc.aiLib == 'sklearn':
        if gc.multiModel:
            MODELS = {model: regressor(**sklearnNNSettings['r']) if model != 'Sequencer' else classifier(
                **sklearnNNSettings['c']) for model in gc.models}
        else:
            MODELS = {model: regressor(**sklearnNNSettings['r']) if model != 'Sequencer' else classifier(
                **sklearnNNSettings['c']) for model in gc.models}

    elif gc.aiLib == 'keras':
        if gc.multiModel:
            MODELS = {
                model: kerasNN(**kerasNNSettings['r']) if model != 'Sequencer' else kerasNN(**kerasNNSettings['c']) for
                model in gc.models}
        else:
            MODELS = {
                model: kerasNN(**kerasNNSettings['r']) if model != 'Sequencer' else kerasNN(**kerasNNSettings['c']) for
                model in gc.models}
    trainer.run(ai.Model(), MODELS)


run()

import keras
from sklearn.neural_network import MLPClassifier, MLPRegressor
from tensorflow import set_random_seed

import ai
from utils.ai import activations, layersDict, regularizersDict, get_model_info

set_random_seed(1)


def regressor(**settings):
    return MLPRegressor(**settings)


def classifier(**settings):
    return MLPClassifier(**settings)


def run():
    sklearnNNSettings = {'c': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001,
                               'batch_size': 'auto', 'learning_rate': 'constant', 'learning_rate_init': 0.001,
                               'power_t': 0.5, 'max_iter': 1, 'shuffle': True, 'random_state': None, 'tol': 1e-4,
                               'verbose': False, 'momentum': 0.9,  # 'nesterovs_momentum': True,
                               'early_stopping': False, 'validation_fraction': 0.1, 'beta_1': 0.999, 'beta_2': 0.999,
                               'uncertainty': 1e-8,  # uncertainty=epsilon? 'n_iter_no_change': 10,
                               },
                         'r': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001,
                               'batch_size': 'auto', 'learning_rate': 'constant', 'learning_rate_init': 1,
                               'power_t': 0.5, 'max_iter': 1, 'shuffle': True, 'random_state': None, 'tol': 1e-4,
                               'verbose': False, 'warm_start': False, 'momentum': 0.9,  # 'nesterovs_momentum': True,
                               'early_stopping': False, 'validation_fraction': 0.1, 'beta_1': 0.9, 'beta_2': 0.999,
                               'uncertainty': 1e-08,  # uncertainty=epsilon?'n_iter_no_change': 10
                               }, }
    classifierInputShape = (gc.inputSize,)
    if gc.multiModel:
        regressorInputShape = (gc.inputSize,)
    else:
        regressorInputShape = (gc.inputSize + len(gc.models),)
    kerasNNSettings = {'c': {
        'layers': [{'type': 'dense', 'units': 20, 'activation': 'swish', 'input_shape': classifierInputShape, },
                   {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': len(gc.models), 'activation': 'softmax',

                    }, ], 'compileSettings': {'loss': 'categorical_crossentropy', 'optimizer': 'rmsprop', }}, 'r': {
        'layers': [{'type': 'dense', 'units': 40, 'activation': 'linear', 'input_shape': regressorInputShape,
                    'kernel_initializer': 'lecun_normal', 'bias_initializer': 'lecun_normal',
                    'kernel_regularizer': ('L2', 0.01), 'bias_regularizer': ('L2', 0.01), },
                   {'type': 'dropout', 'rate': 0.2, },
                   {'type': 'dense', 'units': 2, 'activation': 'linear', 'kernel_initializer': 'lecun_normal',
                    'bias_initializer': 'lecun_normal', 'kernel_regularizer': ('L2', 0.01),
                    'bias_regularizer': ('L2', 0.01), }, ], 'compileSettings': {'loss': 'mse', 'optimizer': 'adam', }}}
    if gc.aiLib == 'keras':
        get_model_info(kerasNNSettings)
    if gc.aiLib == 'sklearn':
        get_model_info(sklearnNNSettings)

    if gc.aiLib == 'sklearn':
        if gc.multiModel:
            MODELS = {model: regressor(**sklearnNNSettings['r']) if model != 'Sequencer' else classifier(
                **sklearnNNSettings['c']) for model in gc.models}
        else:
            MODELS = {model: regressor(**sklearnNNSettings['r']) if model != 'Sequencer' else classifier(
                **sklearnNNSettings['c']) for model in gc.models}

    elif gc.aiLib == 'keras':
        if gc.multiModel:
            MODELS = {
                model: kerasNN(**kerasNNSettings['r']) if model != 'Sequencer' else kerasNN(**kerasNNSettings['c']) for
                model in gc.models}
        else:
            MODELS = {
                model: kerasNN(**kerasNNSettings['r']) if model != 'Sequencer' else kerasNN(**kerasNNSettings['c']) for
                model in gc.models}
    trainer.run(ai.Model(), MODELS)


run()
