import numpy as np
import pandas as pd
from tensorflow import set_random_seed

import settings as gc
import util

set_random_seed(1)
np.random.seed(3)


def enter_the_matrix():
    pCols = ['STRUCTURE_NUMBER_008', gc.deck] + gc.numericalCols + gc.categoricalCols
    cCols = ['STRUCTURE_NUMBER_008', gc.deck]
    inputSize = 0
    for col in pCols:
        if col == 'STRUCTURE_NUMBER_008':
            pass
        elif col in gc.numericalCols:
            inputSize += 1
        else:
            inputSize += gc.el[col]
    model = util.kR(inputSize=inputSize)
    trainFiles = util.get_files(gc.trainingDir)
    testFiles = util.get_files(gc.testingDir)
    pre = {}
    curr = {}

    for i in range(len(trainFiles)):
        if i > 0:
            f = trainFiles[i - 1]
            year = int(util.strip_ext(util.path_end(f))[-4:])
            df = pd.read_csv(trainFiles[i - 1], usecols=pCols, na_values=['N']).set_index('STRUCTURE_NUMBER_008')
            previous = util.munge(df, year)
            pre[trainFiles[i]] = previous
            current = pd.read_csv(trainFiles[i], usecols=cCols, na_values=['N']).set_index('STRUCTURE_NUMBER_008')
            current = current.dropna(axis=0).rename(columns=gc.con)['Deck']
            curr[trainFiles[i]] = current
    for i in range(len(testFiles)):
        if i > 0:
            f = trainFiles[i - 1]
            year = int(util.strip_ext(util.path_end(f))[-4:])
            df = pd.read_csv(testFiles[i - 1], usecols=pCols, na_values=['N']).set_index('STRUCTURE_NUMBER_008')
            previous = util.munge(df, year)
            pre[testFiles[i]] = previous
            current = pd.read_csv(testFiles[i], usecols=cCols, na_values=['N']).set_index('STRUCTURE_NUMBER_008')
            current = current.dropna(axis=0).rename(columns=gc.con)['Deck']
            curr[testFiles[i]] = current

    for epoch in range(gc.epochs):
        print('Epoch', epoch)
        for j in range(len(trainFiles)):
            if j > 0:
                previous = pre[trainFiles[j]]
                current = curr[trainFiles[j]]
                m = 0
                for i in range(len(previous.index)):
                    xBatch = None
                    yBatch = None
                    for b in range(gc.batchSize):
                        try:
                            x = previous.loc[previous.index[m], :]
                            # kind = util.encode('STRUCTURE_KIND_043A', x['STRUCTURE_KIND_043A'])
                            # type = util.encode('STRUCTURE_TYPE_043B', x['STRUCTURE_TYPE_043B'])
                            deck = util.encode(gc.deck, x['Deck'])
                            x = x.drop(
                                [
                                    'Deck',
                                    # 'STRUCTURE_KIND_043A',
                                    # 'STRUCTURE_TYPE_043B'
                                ]
                            )
                            x = np.hstack([deck,
                                           # kind,
                                           # type,
                                           x.to_numpy().reshape(1, -1)])
                            y = current.loc[previous.index[m]]
                            y = util.encode(gc.deck, y)

                            if b > 0:
                                xBatch = np.vstack([xBatch, x])
                                yBatch = np.vstack([yBatch, y])
                            else:
                                xBatch = x
                                yBatch = y

                        except Exception as e:
                            pass
                        finally:
                            m += 1
                            if m > len(previous.index):
                                break
                    try:
                        model.train_on_batch(xBatch, yBatch, class_weight=gc.classWeight)
                    except Exception as e:
                        pass

        if epoch % 5 == 0:
            print('======= Epoch {} ======='.format(epoch))
            trainAcc = validate(trainFiles, pre, curr, model)
            print(' Training Accuracy: {}'.format(trainAcc))
            testAcc = validate(testFiles, pre, curr, model)
            print(' Testing Accuracy: {}'.format(testAcc))
            print('========================')


def validate(set, pre, curr, model):
    gucci = []
    bacci = []
    for j in range(len(set)):
        if j > 0:
            previous = pre[set[j]]
            current = curr[set[j]]
            for i in previous.index:
                try:
                    x = previous.loc[i, :]
                    # kind = util.encode('STRUCTURE_KIND_043A', x['STRUCTURE_KIND_043A'])
                    # type = util.encode('STRUCTURE_TYPE_043B', x['STRUCTURE_TYPE_043B'])
                    deck = util.encode(gc.deck, x['Deck'])
                    x = x.drop(
                        [
                            'Deck',
                            # 'STRUCTURE_KIND_043A',
                            # 'STRUCTURE_TYPE_043B'
                        ]
                    )
                    x = np.hstack([deck,
                                   # kind,
                                   # type,
                                   x.to_numpy().reshape(1, -1)])
                    y = current.loc[i]
                    y = util.encode(gc.deck, y)
                    p = util.encode(gc.deck, np.argmax(model.predict(x)))
                    if np.array_equal(y, p):
                        gucci.append(0)
                    else:
                        bacci.append(0)
                except Exception as e:
                    pass
    acc = len(gucci) / (len(gucci) + len(bacci))
    return acc


enter_the_matrix()
