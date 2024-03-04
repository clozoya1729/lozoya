import numpy as np

np.random.seed(3)
import time

threshold = 0.5
nStates = 5
memorySize = 5

np.set_printoptions(suppress=True)

actions = {'0': (-1, -1), '1': (-1, 0), '2': (0, 0), '3': (1, 1), '4': (1, 1), }

actions2 = {'0': np.array([1, 0, 0, 0, 0]), '1': np.array([0, 1, 0, 0, 0]), '2': np.array([0, 0, 1, 0, 0]),
    '3': np.array([0, 0, 0, 1, 0]), '4': np.array([0, 0, 0, 0, 1]), }


def invert_dict(d, val):
    for k, v in d.items():
        if np.array_equal(v, val):
            return k


def act(x, y):
    _x = np.power(x, 2)
    _y = np.power(y, 2)
    a = np.sin(_x + _y) / 10
    b = np.exp(-(_x + _y) / 10)
    c = np.exp(-(_x + _y))
    return -(a + b + c)


def transition(transitionProbability):
    chance = np.random.randint(0, 100) / 100
    s = np.argmax(transitionProbability.cumsum() > chance)
    v = np.zeros(nStates, dtype=np.int8)
    v[s] = 1
    return v


def run(state, transitionMatrix, memory):
    transitionProbability = np.dot(state, transitionMatrix)
    newState = transition(transitionProbability)
    action = actions[invert_dict(actions2, newState)]
    reward = act(*action)
    if len(memory) >= memorySize:
        memory = memory[:-1]
    memory.append([state, action, newState, reward])

    print(newState, reward)
    # time.sleep(0.5)
    print(len(memory))
    return newState, transitionMatrix, memory


transitionMatrix = np.random.rand(nStates, nStates)
transitionMatrix = transitionMatrix / transitionMatrix.sum(axis=1)[:, None]
stateVector = np.array([0, 1, 0, 0, 0])
while True:
    stateVector, transitionMatrix, memory = run(stateVector, transitionMatrix, [])

import random

import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

random.seed(1)
import configuration as gc
from collections import deque
from utils.researcher import print_time, print_episode_start, print_episode_end, ac, huber_loss, update_uncertainty, \
    random_initial_state, forget, replay, step

np.set_printoptions(suppress=True)
np.random.seed(3)


def get_model():
    model = Sequential()
    model.add(Dense(24, input_dim=gc.nStates, activation='relu'))
    model.add(Dropout(rate=0.8))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(28, activation='relu'))
    model.add(Dense(gc.nActions, activation='linear'))
    model.compile(loss=huber_loss, optimizer=Adam(lr=gc.learningRate))
    return model


def run(state, model, targetModel, memory, uncertainty):
    for e in range(gc.episodes):
        targetModel.set_weights(model.get_weights())
        print_episode_start(e, state)
        done = False
        rewardSum = 0
        for time in range(gc.steps):
            action, newState, reward, done = step(state, model, uncertainty)
            rewardSum += reward
            memory.append([state, action, reward, newState, done])
            state = newState
            if len(memory) > gc.memBatch:
                model, targetModel = replay(model, targetModel, memory)
                uncertainty = update_uncertainty(uncertainty, time, reward, rewardSum)
                memory = forget(memory)
            if time % 100 == 0:
                print_time(time, state, uncertainty, reward, rewardSum)
            if done or time >= gc.steps - 1:
                segma = ac[action]
                s = np.argmax(state) + segma
                print_episode_end(e, time, s, uncertainty, rewardSum)
                break
            if time % 1000 == 0:
                targetModel.set_weights(model.get_weights())
    return model, targetModel, memory


run(state=random_initial_state(), model=get_model(), targetModel=get_model(), memory=deque(maxlen=gc.memSize),
    uncertainty=1.0)

import random

import numpy as np

random.seed(1)
import configuration as gc
from collections import deque
from utils.researcher import print_time, print_episode_start, print_episode_end, ac, update_uncertainty, \
    random_initial_state, forget, replay, step, get_model, act

np.set_printoptions(suppress=True)
np.random.seed(3)


def run(state, model, targetModel, memory, uncertainty):
    for e in range(gc.episodes):
        targetModel.set_weights(model.get_weights())
        print_episode_start(e, state)
        rewardSum = 0
        for time in range(gc.steps):
            action = act(model, state, uncertainty)
            newState, reward, done = step(state, action)
            rewardSum += reward
            memory.append([state, action, reward, newState, done])
            state = newState
            if len(memory) > gc.memBatch:
                model, targetModel = replay(model, targetModel, memory)
                uncertainty = update_uncertainty(uncertainty, time, reward, rewardSum)
                memory = forget(memory)
            if time % 100 == 0:
                print_time(time, state, uncertainty, reward, rewardSum)
            if done or time >= gc.steps - 1:
                segma = ac[action]
                s = np.argmax(state) + segma
                print_episode_end(e, time, s, uncertainty, rewardSum)
                break
            if time % 100 == 0:
                targetModel.set_weights(model.get_weights())
    return model, targetModel, memory


run(state=random_initial_state(), model=get_model(), targetModel=get_model(), memory=deque(maxlen=gc.memSize),
    uncertainty=1.0)

import random

import numpy as np
import tensorflow as tf
from keras import backend as K, Sequential
from keras.layers import Dense
from keras.optimizers import Adam

random.seed(1)
import configuration as gc
from collections import deque

np.set_printoptions(suppress=True)
np.random.seed(3)


def random_transition_matrix():
    transitionMatrix = np.random.rand(gc.nStates, gc.nStates)
    return transitionMatrix / transitionMatrix.sum(axis=1)[:, None]


def get_model():
    model = Sequential()
    model.add(Dense(500, input_dim=gc.nStates, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(gc.nActions, activation='linear'))
    model.compile(loss=huber_loss, optimizer=Adam(lr=gc.learningRate))
    return model


def invert_dict(d, val):
    for k, v in d.items():
        if np.array_equal(v, val):
            return k


def mcmc_transition(transitionProbability):
    t = transitionProbability[0]
    chance = np.random.randint(0, 101) / 100
    s = np.argmax(t.cumsum() > chance)
    v = np.zeros(gc.nStates, dtype=np.int8)
    v[s] = 1
    return v


ac = {0: -527, 1: -254, 2: -24, 3: -1, 4: 0, 5: 1, 6: 18, 7: 312, 8: 834, }


def step(state, model, uncertainty, memory, done):
    def r(s):
        newState = np.zeros(gc.nStates, dtype=np.int8)
        newState[s] = 1
        newState = np.reshape(newState, [1, gc.nStates])
        a = 20
        b = 100 * np.exp(-((s - 500) ** 2) / 10000)
        c = 80 * np.exp(-((s - 42) ** 2) / 8000)
        d = 100 * np.sin(0.1 * s)
        g = 200 * np.sin(0.03 * s)
        j = 1000 * np.exp(-((s - 620) ** 2 / 1000))
        reward = a + b + c + d + g + j
        return reward, newState

    action = act(model, state, uncertainty)
    segma = ac[action]
    s = np.argmax(state) + segma
    try:
        reward, newState = r(s)
    except:
        reward = -gc.penalty
        newState = state
        done = True
    return action, newState, reward, done, memory


def act(model, state, uncertainty):
    prediction = model.predict(state)[0]
    if uncertainty > 0.95:
        return random.randrange(gc.nActions)
    if np.random.rand() <= uncertainty:
        return transition(prediction)
    return np.argmax(prediction)


def update_uncertainty(uncertainty):
    if np.random.randint(-10001, 101) / 100 > 0.95:
        return 1.0
    if uncertainty > gc.uncertaintyMin and np.random.randint(0, 101) / 100 > 0.2:
        return uncertainty * gc.uncertaintyDecay
    return uncertainty


def transition(probabilities):
    t = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())
    t = t / t.sum()
    chance = np.random.randint(0, 1001) / 1000
    s = np.argmax(t.cumsum() > chance)
    return s


def replay(model, targetModel, memory):
    if np.random.randint(0, 101) / 100 > ((gc.memSize - len(memory)) / gc.memSize):
        minibatch = random.sample(memory, gc.memBatch)
        for state, action, reward, nextState, done in minibatch:
            if np.random.randint(0, 101) / 100 > ((gc.memSize - len(memory)) / gc.memSize):
                target = model.predict(state)
                if done:
                    target[0][action] = reward
                else:
                    target[0][action] = reward + gc.gamma * np.amax(targetModel.predict(nextState)[0])
                model.fit(state, target, epochs=1, verbose=0)
    return model, targetModel


def forget(memory):
    forgetChance = np.random.randint(-10001, 101) / 100
    if forgetChance > (gc.forgetThreshold + (1 - gc.forgetThreshold * ((gc.memSize - len(memory)) / gc.memSize))):
        s = int(np.random.randint(0, len(memory)))
        e = int(np.random.randint(s, len(memory)))
        remembered = deque(list(memory)[s:e])
        print('Forgot actions 0 to {} and {} to {}'.format(s, e, len(memory)))
        return remembered
    return memory


def random_initial_state():
    state = np.zeros(gc.nStates, dtype=np.int8)
    a = np.random.randint(0, gc.nStates)
    state[a] = 1
    return np.reshape(state, [1, gc.nStates])


def run(model, targetModel, memory):
    for e in range(gc.episodes):
        targetModel.set_weights(model.get_weights())
        uncertainty = 1.0
        state = random_initial_state()
        print('o Episode {} Start\n'
              '\tinitialState: {}\n'.format(e, np.argmax(state)))
        done = False
        rewardSum = 0
        for time in range(gc.steps):
            action, newState, reward, done, memory = step(state, model, uncertainty, memory, done)
            rewardSum += reward
            newState = np.reshape(newState, [1, gc.nStates])
            memory.append([state, action, reward, newState, done])
            state = newState
            if len(memory) > gc.memBatch:
                model, targetModel = replay(model, targetModel, memory)
                uncertainty = update_uncertainty(uncertainty)
                memory = forget(memory)

            if time % 100 == 0:
                print('\t - Time: {}\n\t'
                      '\t\t State: {}\n'
                      '\t\t Uncertainty: {:.2}\n'
                      '\t\t Reward: {}\n'
                      '\t\t Reward Sum: {}\n'.format(time, np.argmax(state), uncertainty, int(reward), int(rewardSum)))
            if done or time >= gc.steps - 1:
                segma = ac[action]
                s = np.argmax(state) + segma
                print("o Episode {} End\n"
                      "\t Time: {}\n"
                      '\t State: {}\n'
                      "\t Uncertainty: {:.2}\n"
                      "\t Reward Sum: {}\n".format(e, time, s, uncertainty, int(rewardSum)))
                break

            if time % 1000 == 0:
                targetModel.set_weights(model.get_weights())
    return model, targetModel, memory


model, targetModel, memory = run(model=get_model(), targetModel=get_model(), memory=deque(maxlen=gc.memSize))

import random
from collections import deque

import numpy as np
import configuration as gc
import tensorflow as tf
from keras import backend as K, Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

ac = {0: -527, 1: -254, 2: -24, 3: -1, 4: 0, 5: 1, 6: 18, 7: 312, 8: 834, }


def print_time(time, state, uncertainty, reward, rewardSum):
    print('\t - Time: {}\n\t'
          '\t\t State: {}\n'
          '\t\t Uncertainty: {:.2}\n'
          '\t\t Reward: {}\n'
          '\t\t Reward Sum: {}\n'.format(time, np.argmax(state), uncertainty, int(reward), int(rewardSum)))


def print_episode_start(e, state):
    print('o Episode {} Start\n'
          '\tinitialState: {}\n'.format(e, np.argmax(state)))


def huber_loss(yTrue, yPred, clipDelta=1.0):
    error = yTrue - yPred
    cond = K.abs(error) <= clipDelta
    squaredLoss = 0.5 * K.square(error)
    quadraticLoss = 0.5 * K.square(clipDelta) + clipDelta * (K.abs(error) - clipDelta)
    return K.mean(tf.where(cond, squaredLoss, quadraticLoss))


def random_initial_state():
    state = np.zeros(gc.nStates, dtype=np.int8)
    state[np.random.randint(0, gc.nStates)] = 1
    return np.reshape(state, [1, gc.nStates])


def forget(memory):
    forgetChance = np.random.randint(-10001, 101) / 100
    w = (1 - gc.forgetThreshold * ((gc.memSize - len(memory)) / gc.memSize))
    if forgetChance > (gc.forgetThreshold + w):
        s = int(np.random.randint(0, len(memory)))
        e = int(np.random.randint(s, len(memory)))
        remembered = deque(list(memory)[s:e])
        print('Forgot actions 0 to {} and {} to {}'.format(s, e, len(memory)))
        return remembered
    return memory


def act(model, state, uncertainty):
    prediction = model.predict(state)[0]
    if uncertainty > 0.75 or np.random.randint(0, 100) / 100 > 0.95:
        return random.randrange(gc.nActions)
    if uncertainty > 0.25 or np.random.randint(0, 100) / 100 > 0.9:
        return transition(prediction)
    return np.argmax(prediction)


def transition(dist):
    scaled = (dist - dist.min()) / (dist.max() - dist.min())
    scaled = scaled / scaled.sum()
    s = np.argmax(scaled.cumsum() > np.random.randint(0, 1001) / 1000)
    return s


import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras import backend as K

import configuration as gc
import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras import backend as K, Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import configuration as gc


def print_time(time, state, uncertainty, reward, rewardSum):
    print('\t - Time: {}\n\t'
          '\t\t State: {}\n'
          '\t\t Uncertainty: {:.2}\n'
          '\t\t Reward: {}\n'
          '\t\t Reward Sum: {}\n'.format(time, np.argmax(state), uncertainty, int(reward), int(rewardSum)))


def print_episode_start(e, state):
    print('o Episode {} Start\n'
          '\tinitialState: {}\n'.format(e, np.argmax(state)))


ac = {0: -527, 1: -254, 2: -24, 3: -1, 4: 0, 5: 1, 6: 18, 7: 312, 8: 834, }


def print_episode_start(e, state):
    print('o Episode {} Start\n'
          '\tinitialState: {}\n'.format(e, np.argmax(state)))


def print_episode_end(e, time, s, uncertainty, rewardSum):
    print("o Episode {} End\n"
          "\t Time: {}\n"
          '\t State: {}\n'
          "\t Uncertainty: {:.2}\n"
          "\t Reward Sum: {}\n".format(e, time, s, uncertainty, int(rewardSum)))


def step(state, model, uncertainty):
    action = act(model, state, uncertainty)
    try:
        s = np.argmax(state) + ac[action]
        newState = np.zeros(gc.nStates, dtype=np.int8)
        newState[s] = 1
        newState = np.reshape(newState, [1, gc.nStates])
        reward = get_reward(s)
        done = False
    except:
        newState = state  # TODO ??
        reward = -gc.penalty
        done = True
    return action, newState, reward, done


import random
from collections import deque

import numpy as np
import configuration as gc
import tensorflow as tf
from keras import backend as K, Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

ac = {0: -527, 1: -254, 2: -24, 3: -1, 4: 0, 5: 1, 6: 18, 7: 312, 8: 834, }


def print_episode_start(e, state):
    print('o Episode {} Start\n'
          '\tinitialState: {}\n'.format(e, np.argmax(state)))


def print_episode_end(e, time, s, uncertainty, rewardSum):
    print("o Episode {} End\n"
          "\t Time: {}\n"
          '\t State: {}\n'
          "\t Uncertainty: {:.2}\n"
          "\t Reward Sum: {}\n".format(e, time, s, uncertainty, int(rewardSum)))


def update_uncertainty(uncertainty, time, reward, rewardSum):
    h = np.tanh(-reward / 10000)
    g = np.tanh(-rewardSum / 10000)
    if rewardSum > gc.minReward + 100 * time:
        p = 0.999
    elif rewardSum > gc.minReward + 1000 * time:
        p = 0.9
    elif rewardSum > gc.minReward + 10000 * time:
        p = 0.8
    else:
        p = 1
    if np.random.randint(-100001, 101) / 100 > 0.95:
        return 1.0
    if uncertainty > gc.uncertaintyMin:
        if np.random.randint(0, 101) / 100 >= 0.95 * uncertainty:
            return uncertainty * gc.uncertaintyDecay * p
        elif uncertainty < 0.5 and np.random.randint(0, 101) / 100 > uncertainty:
            return uncertainty / gc.uncertaintyDecay
    else:
        return uncertainty / gc.uncertaintyDecay
    return uncertainty


def replay(model, targetModel, memory):
    si = 0.9 * ((gc.memSize - len(memory)) / gc.memSize)
    if np.random.randint(0, 101) / 100 > si:
        minibatch = random.sample(memory, gc.memBatch)
        for state, action, reward, newState, done in minibatch:
            if np.random.randint(0, 101) / 100 > si:
                target = model.predict(state)
                if done:
                    target[0][action] = reward
                else:
                    target[0][action] = reward + gc.gamma * np.amax(targetModel.predict(newState)[0])
                model.fit(state, target, epochs=1, verbose=0)
    return model, targetModel


def step(state, action):
    try:
        s = np.argmax(state) + ac[action]
        newState = int_to_state(s)
        reward = get_reward(s)
        done = False
    except:
        newState = state  # TODO ??
        reward = -gc.penalty
        done = True
    return newState, reward, done


def int_to_state(i):
    s = np.zeros(gc.nStates, dtype=np.int8)
    s[i] = 1
    s = np.reshape(s, [1, gc.nStates])
    return s


def get_reward(s):
    a = 20
    b = 100 * np.exp(-((s - 500) ** 2) / 10000)
    c = 80 * np.exp(-((s - 42) ** 2) / 8000)
    d = 0 * 100 * np.sin(0.1 * s)
    g = 0 * 200 * np.sin(0.03 * s)
    j = 1000 * np.exp(-((s - 620) ** 2 / 1000))
    h1 = 2000 * np.exp(-((10 * s - 6000) ** 2 / 400000))
    h2 = 1000 * np.cos(0.001 * 10 * s)
    return h1 + h2  # a + b + c + d + g + j


def get_model():
    model = Sequential()
    model.add(Dense(256, input_dim=gc.nStates, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(gc.nActions, activation='linear'))
    model.compile(loss=huber_loss, optimizer=Adam(lr=gc.learningRate))
    return model


# -*- coding: utf-8 -*-
import random

random.seed(1)

from collections import deque

import gym
import numpy as np
import configuration as gc
from vfdqn import get_model
from utils.researcher import update_uncertainty, forget, act, replay

env = gym.make('CartPole-v1')
don = False
mode = get_model()
targetMode = get_model()
targetMode.set_weights(mode.get_weights())
memor = deque(maxlen=gc.memSize)
uncertainty = 1.0  # exploration rate
for e in range(gc.episodes):
    state = np.reshape(env.reset(), [1, gc.nStates])
    for time in range(50):
        env.render()
        action = act(mode, state, None, uncertainty)
        nextState, reward, done, _ = env.step(action)
        reward = reward if not done else -gc.penalty
        nextState = np.reshape(nextState, [1, gc.nStates])
        print(state)
        print(action)
        print(nextState, reward, done, _)
        print(reward)
        memor = forget(memor)
        memor.append([state, action, reward, nextState, done])
        state = nextState
        if len(memor) > gc.memBatch:
            model, targetModel, memory = replay(mode, targetMode, memor)
            uncertainty = update_uncertainty(uncertainty)
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, gc.episodes, time, uncertainty))
            break
    if e % 100 == 0:
        targetMode.set_weights(mode.get_weights())


# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                # Logging training loss every 10 timesteps
                if time % 10 == 0:
                    print("episode: {}/{}, time: {}, loss: {:.4f}".format(e, EPISODES, time,
                                                                          loss))  # if e % 10 == 0:  #     agent.save("./save/cartpole-dqn.h5")

# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)  # if e % 10 == 0:  #     agent.save("./save/cartpole-dqn.h5")
