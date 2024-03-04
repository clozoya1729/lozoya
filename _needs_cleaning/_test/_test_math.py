import os
# import threading
import sys

import matplotlib
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import sympy as sym
from scipy import signal

matplotlib.use("TkAgg")
mpl.rcParams['mathtext.fontset'] = 'stix'
latexdire = 'C:\Program Files\MiKTeX 1.5\miktex/bin/x64'
os.environ[
    "PATH"] += 'C:\Program Files\MiKTeX 1.5\miktex/bin/x64' + os.pathsep  # '/usr/local/texlive/2015/bin/x86_64-darwin'
sys.path.append(os.path.abspath('../../0.0.1/'))
os.environ["PATH"] += 'C:\Program Files\MiKTeX 1.5\miktex/bin/x64' + os.pathsep
sys.path.append(os.path.abspath('../data-pyqt/'))
os.environ["PATH"] += r'Latex\miktex\bin\x64' + os.pathsep
os.environ["PATH"] += r'Latex\miktex\bin\x64' + os.pathsep
sys.path.append(__level__.import_manager())
modules = ['PlotPaths', 'GeneralUtil', 'GreekAlphabet', 'StatisticsVars']
PlotPaths, GeneralUtil, GreekAlphabet, StatisticsVars = ImportManager._import(modules)
sys.path.append(os.path.abspath('../LoParPlot/'))
modules = ['PlotPaths', 'GeneralUtil', 'GreekAlphabet', 'StatisticsVars']
os.environ["PATH"] += r';Z:\Family\VirtualEnvironments\Latex\miktex\bin\x64' + os.pathsep
dire = os.path.join('HTML2', 'FunctionStrings')
sym.preview(mae_string(), viewer='file', filename=os.path.join(dire, 'Mean Absolute Error.png'))
sym.preview(mbe_string(), viewer='file', filename=os.path.join(dire, 'Mean Bias Error.png'))
sym.preview(mse_string(), viewer='file', filename=os.path.join(dire, 'Mean Squared Error.png'))
sym.preview(rmse_string(), viewer='file', filename=os.path.join(dire, 'Root Mean Squared Error.png'))
sym.preview(lowerbound_string(), viewer='file', filename=os.path.join(dire, 'Lower Bound.png'))
sym.preview(upperbound_string(), viewer='file', filename=os.path.join(dire, 'Upper Bound.png'))
sym.preview(iqr_string(), viewer='file', filename=os.path.join(dire, 'Interquartile Range.png'))
number_of_states = 5
input_file_paths = ['C:\\Users\\frano\PycharmProjects\BridgeDataQuery\Reports\\report2.txt']
item_of_interest = "'5B'"
years_to_iterate = []
integralVal = 0
anim = animation.FuncAnimation(fig, sine, interval=100)
plt.grid(True, which='both')
plt.show()
signalFrequency = np.pi
samps = 50
axOffset = 5
integrate = False
noise = False
lpf = True
hpf = False
fig, ax = plt.subplots(1, 1)
sinegraph, = ax.plot([], [])
lowpassGraph, = ax.plot([], [])
integralGraph, = ax.plot([], [])
integralGraph1, = ax.plot([], [])
ax.set_ylim([- 500, 500])
X = np.linspace(0, signalFrequency, samps)
V = np.zeros_like(X)
F1 = np.zeros_like(X)
D = np.zeros_like(V)
D1 = np.zeros_like(V)
Dsum = 0
NA_VALUES = ['N']
origin = (0, 0, 0)
unitDims = (1, 1, 1)
plot_regression_and_ci()
print('finished')
# SAMPLING
"""series"""
# ile = r'C:\\Users\keren\PycharmProjects\Big\Database\Children Of The Sky\Children Of The Sky1.csv'
# data = dh.clean_data(pd.read_csv(file, usecols=['col2'], na_values=NA_VALUES))
# stateVector = state_vector('1', data)
# transitionProbabilityMatrix = transition_probability_matrix(data)
# markovChain = markov_chain(stateVector, transitionProbabilityMatrix, 25)
# plt.plot0(markovChain.T)
# plt.show()
"""parallel"""
index = 'col1'
column = 'col2'
files = [r'C:\Users\keren\PycharmProjects\Big\Database\Children Of The Sky\Children Of The Sky1.csv',
         r'C:\Users\keren\PycharmProjects\Big\Database\Children Of The Sky\Children Of The Sky2.csv']
data = read_data_parallel(r'C:\Users\keren\PycharmProjects\Big\Database\Children Of The Sky', 'col1', 'col2')
labels = parallel_labels(data)
stateVector = state_vector('1', pd.DataFrame(labels))
data = merger(data)
countMatrix = count_matrix_parallel(data, labels)
transitionMatrix = transition_matrix_parallel(data, labels)
transitionProbabilityMatrix = transition_probability_matrix_parallel(data, labels)
markovChain = markov_chain(stateVector, transitionProbabilityMatrix, 10)
print(markovChain)
plt.plot(markovChain.T)
plt.show()
FOLDER = 'YEAR'
FILE = 'STATE_CODE_001'
NA_VALUES = ['N']
# Load data from statsmodels datasets
data = pd.Series(sm.datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel())
# Plot for comparison
plt.figure(figsize=(12, 8))
ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5, color=plt.rcParams['axes.color_cycle'][1])
# Save plot0 limits
dataYLim = ax.get_ylim()
# Find best fit distribution
best_fit_name, best_fir_paramms = best_fit_distribution(data, 200, ax)
best_dist = getattr(st, best_fit_name)
# Update plots
ax.set_ylim(dataYLim)
ax.set_title(u'El Ni�o sea temp.\n All Fitted Distributions')
ax.set_xlabel(u'Temp (�C)')
ax.set_ylabel('Frequency')
# Make PDF
pdf = make_pdf(best_dist, best_fir_paramms)
# Display
plt.figure(figsize=(12, 8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)
param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fir_paramms)])
dist_str = '{}({})'.format(best_fit_name, param_str)
ax.set_title(u'El Ni�o sea temp. with best fit distribution \n' + dist_str)
ax.set_xlabel(u'Temp. (�C)')
ax.set_ylabel('Frequency')
# DISTRIBUTION FIT
# AUXILIARY
f, n = halfcauchy_string([], name=True)
printer(f, n)
f, n = halflogistic_string([], name=True)
printer(f, n)
f, n = halfnorm_string([], name=True)
printer(f, n)
f, n = hypsecant_string([], name=True)
printer(f, n)
f, n = invgauss_string([], name=True)
printer(f, n)
f, n = invweibull_string([], name=True)
printer(f, n)
f, n = johnsonsb_string([], name=True)
printer(f, n)
f, n = laplace_string([], name=True)
printer(f, n)
f, n = levy_string([], name=True)
printer(f, n)
f, n = levy_l_string([], name=True)
printer(f, n)
f, n = logistic_string([], name=True)
printer(f, n)
f, n = lomax_string([], name=True)
printer(f, n)
f, n = maxwell_string([], name=True)
printer(f, n)
f, n = mielke_string([], name=True)
printer(f, n)
f, n = moyal_string([], name=True)
printer(f, n)
f, n = norm_string([], name=True)
printer(f, n)
f, n = pareto_string([], name=True)
printer(f, n)
f, n = powerlaw_string([], name=True)
printer(f, n)
f, n = rayleigh_string([], name=True)
printer(f, n)
f, n = recipinvgauss_string([], name=True)
printer(f, n)
f, n = semicircular_string([], name=True)
printer(f, n)
f, n = truncexpon_string([], name=True)
printer(f, n)
f, n = wald_string([], name=True)
printer(f, n)
f, n = weibull_min_string([], name=True)
printer(f, n)
f, n = weibull_max_string([], name=True)
printer(f, n)
f, n = wrapcauchy_string([], name=True)
printer(f, n)
printer(halfcauchy_string([], name=True))
printer(halflogistic_string([], name=True))
printer(halfnorm_string([], name=True))
printer(hypsecant_string([], name=True))
printer(invgauss_string([], name=True))
printer(invweibull_string([], name=True))
printer(johnsonsb_string([], name=True))
printer(laplace_string([], name=True))
printer(levy_string([], name=True))
printer(levy_l_string([], name=True))
printer(logistic_string([], name=True))
printer(lomax_string([], name=True))
printer(maxwell_string([], name=True))
printer(mielke_string([], name=True))
printer(moyal_string([], name=True))
printer(norm_string([], name=True))
printer(pareto_string([], name=True))
printer(powerlaw_string([], name=True))
printer(rayleigh_string([], name=True))
printer(recipinvgauss_string([], name=True))
printer(semicircular_string([], name=True))
printer(truncexpon_string([], name=True))
printer(wald_string([], name=True))
printer(weibull_min_string([], name=True))
printer(wrapcauchy_string([], name=True))
# STATISTICS
printer(mae_string(), 'Mean Absolute Error.png')
printer(mbe_string(), 'Mean Bias Error.png')
printer(mse_string(), 'Mean Squared Error.png')
printer(rmse_string(), 'Root Mean Squared Error.png')
printer(lowerbound_string(), 'Lower Bound.png')
printer(upperbound_string(), 'Upper Bound.png')
printer(iqr_string(), 'Interquartile Range.png')
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')
DISTRIBUTIONS = [st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi,
                 st.chi2, st.cosine, st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib,
                 st.exponpow, st.f, st.fatiguelife, st.fisk, st.foldcauchy, st.foldnorm, st.frechet_r, st.frechet_l,
                 st.genlogistic, st.genpareto, st.gennorm, st.genexpon, st.genextreme, st.gausshyper, st.gamma,
                 st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz, st.gumbel_r, st.gumbel_l, st.halfcauchy,
                 st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma, st.invgauss, st.invweibull,
                 st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign, st.laplace, st.levy, st.levy_l, st.levy_stable,
                 st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami,
                 st.ncx2, st.ncf, st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm,
                 st.rdist, st.reciprocal, st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang,
                 st.truncexpon, st.truncnorm, st.tukeylambda, st.uniform, st.vonmises, st.vonmises_line, st.wald,
                 st.weibull_min, st.weibull_max, st.wrapcauchy]
# Best holders
best_distribution = st.norm
best_params = (0.0, 1.0)
best_sse = np.inf
origin = (0, 0, 0)
unitDims = (1, 1, 1)
mpl.rcParams['mathtext.fontset'] = 'stix'
latexdire = 'C:\Program Files\MiKTeX 1.5\miktex/bin/x64'
os.environ[
    "PATH"] += 'C:\Program Files\MiKTeX 1.5\miktex/bin/x64' + os.pathsep  # '/usr/local/texlive/2015/bin/x86_64-darwin'
# STATISTICS ---------------------------------------
printer(_normal_cdf_string(*['t'], y=PHI__), "Normal CDF")
printer(_gamma_string(*['z'], y=GAMMA__), "Gamma Function")
printer(_beta_string(*['t', 'y_'], y=BETA__), "Beta Function")
printer(alpha_string(*['a']), "Alpha")
printer(beta_string(*['a', 'b']), "Beta")
printer(chi_string(*['df']), "Chi")
printer(chi2_string(*['df']), "Chi2")
printer(dgamma_string(*['a']), "Double Gamma")
printer(exponnorm_string(*['K']), "Exponentially Modified Normal")
printer(f_string(*['K']), "F")
printer(genexpon_string(*['a', 'b', 'c']), "Generalized Exponential")
printer(gamma_string(*['a', 'b', 'c']), "gamma")
printer(gengamma_string(*['a', 'c']), "Generalized gamma")
printer(genhalflogistic_string(*['c']), "Generalized Half-Logistic")
printer(gilbrat_string(*[]), "Gilbrat")
printer(gompertz_string(*['c']), "Gompertz")
printer(gumbel_r_string(*[]), "Right-Skewed Gumbel")
printer(gumbel_l_string(*[]), "Left-Skewed Gumbel")
printer(invgamma_string(*['a']), "Inverted Gamma")
printer(kappa4_string(*['h', 'k']), "Kappa 4")
printer(loggamma_string(*['c']), "Log Gamma")
printer(lognorm_string(*['s']), "Log Normal")
printer(nakagami_string(*[]), "Nakagami")
printer(nct_string(*['df', 'nc']), "Non-Central Student's T")
printer(pearson3_string(*[]), "Pearson Type III")
printer(powerlognorm_string(*['c', 's']), "Power Log-Normal")
printer(powernorm_string(*['c', 's']), "Power Normal")
printer(rdist_string(*['c']), "R-Distributed")
printer(reciprocal_string(*['a', 'b']), "Reciprocal")
printer(t_string(*['df']), "Student's T")
# STATISTICS
printer(mae_string(), 'Mean Absolute Error')
printer(mbe_string(), 'Mean Bias Error')
printer(mse_string(), 'Mean Squared Error')
printer(rmse_string(), 'Root Mean Squared Error')
printer(lowerbound_string(), 'Lower Bound')
printer(upperbound_string(), 'Upper Bound')
printer(iqr_string(), 'Interquartile Range')
index = 'STRUCTURE_NUMBER_008'
column = 'DECK_COND_058'
dataFrameList = dh.read_data_list(os.path.join(DATABASE, '00'), index, column)
stateVector = mh.state_vector('5', dataFrameList)
transitionProbabilityMatrix = transition_probability_matrix(dataFrameList, mode='series')
markovChain = markov_chain(stateVector, transitionProbabilityMatrix, 10)
plt.plot0(markovChain.T)
plt.show()
np.random.seed(6)
sampleRate = 10
cutoff = 2
noise = True
hpf = True
lpf = True
integral = True
derivative = True
velocity = False
displacement = False
t = np.arange(0, sampleRate, 0.1)
s = np.sin(t)
# c = -np.cos(t)
if noise:
    s += + np.random.normal(0, 100, t.shape)
plt.plot0(t, s)
# plt.plot0(t, c)
if hpf:
    b, a = signal.butter(1, 2 * cutoff / sampleRate, btype='high')
    s = signal.filtfilt(b, a, s)
    plt.plot0(t, s)
if lpf:
    b, a = signal.butter(1, 2 * cutoff / sampleRate, btype='low')
    s = signal.filtfilt(b, a, s)
    plt.plot0(t, s)
if velocity:
    v = np.zeros_like(s)
    for i, val in enumerate(s):
        v[i] = np.trapz(s[0:i], t[0:i])
    plt.plot0(t, v)
if displacement:
    d = np.zeros_like(v)
    for i, val in enumerate(v):
        d[i] = np.trapz(v[0:i], t[0:i])
    plt.plot0(t, d)
rms = 1
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True, which='both')
plt.show()
# DISTRIBUTION FIT
# AUXILIARY
print(get_exponential_string(1, 2))
print(get_logarithmic_string(1, 2))
print(get_polynomial_string([3, -5, 1]))
print(get_arcsin_string())
print(sine_string(*[4, 'd'], x=18, name=True, evaluate=True))
if integrate:
    for j, val in enumerate(V):
        D[j] = np.trapz(V[0:j], X[0:j])

# print(get_exponential_string(1, 2))
# print(get_logarithmic_string(1, 2))
# print(get_polynomial_string([3, -5, 1]))
# print(get_arcsin_string())
'''
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
#
# np.random.seed(6)
# sampleRate = 10
# cutoff = 1
# noise = True
# hpf = True
# lpf = True
# integral = True
# derivative = True
# velocity = False
# displacement = False
# t = np.arange(0, sampleRate, 0.1)
# s = np.sin(t)
# #c = -np.cos(t)
#
# if noise:
#     s += + np.random.normal(0, 100, t.shape)
# plt.plot0(t, s)
# #plt.plot0(t, c)
#
# if hpf:
#     b, a = signal.butter(1, 1*cutoff/sampleRate, btype='high')
#     s = signal.filtfilt(b, a, s)
#     plt.plot0(t, s)
#
# if lpf:
#     b, a = signal.butter(1, 1*cutoff/sampleRate, btype='low')
#     s = signal.filtfilt(b, a, s)
#     plt.plot0(t, s)
#
# if velocity:
#     v = np.zeros_like(s)
#     for i, val in enumerate(s):
#         v[i] = np.trapz(s[0:i], t[0:i])
#     plt.plot0(t, v)
#
# if displacement:
#     d = np.zeros_like(v)
#     for i, val in enumerate(v):
#         d[i] = np.trapz(v[0:i], t[0:i])
#     plt.plot0(t, d)
#
# rms = 1
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.grid(True, which='both')
#
#
# plt.show()
import matplotlib

matplotlib.use("TkAgg")
import random
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from scipy import signal

signalFrequency = np.pi
samps = 50
axOffset = 5

integrate = False
noise = False
lpf = True
hpf = False

fig, ax = plt.subplots(1, 1)
sinegraph, = ax.plot([], [])
lowpassGraph, = ax.plot([], [])
integralGraph, = ax.plot([], [])
integralGraph1, = ax.plot([], [])
ax.set_ylim([- 500, 500])

X = np.linspace(0, signalFrequency, samps)
V = np.zeros_like(X)
F1 = np.zeros_like(X)
D = np.zeros_like(V)
D1 = np.zeros_like(V)
Dsum = 0

if integrate:
    for j, val in enumerate(V):
        D[j] = np.trapz(V[0:j], X[0:j])

integralVal = 0


def get_signal(q):
    s = 0
    for i in range(1000):
        s += random.randint(-10, 10) * np.sin(i * q)
    return s


def sine(q):
    i = q
    global V
    global Dsum
    global D
    global D1
    global integralVal
    Xx = np.linspace(i, i + signalFrequency, samps)
    X[:-1] = X[1:]
    X[-1] = Xx[-1]
    V[:-1] = V[1:]
    V[-1] = get_signal(i)  # np.sin(i)

    if noise: V[-1] += random.randint(-1000, 1000) / 1000

    if lpf:
        b, a = signal.butter(12, 2 * 0.1 / signalFrequency, btype='low')
        F1 = signal.filtfilt(b, a, V)
        lowpassGraph.set_data(X, F1)

    if integrate:
        if lpf:
            Dsum += np.trapz(F1[-2:], F1[-2:])
        else:
            Dsum += np.trapz(V[-2:], X[-2:])
        D[:-1] = D[1:]
        D[-1] = Dsum
        integralGraph.set_data(X, D)

    sinegraph.set_data(X, V)
    ax.set_xlim([X[0] - axOffset, X[-1] + signalFrequency + axOffset])


anim = animation.FuncAnimation(fig, sine, interval=100)
plt.grid(True, which='both')

plt.show()
'''

import lozoya.cas_api as cas
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import lozoya.acs


def reaction_wheel_transfer_function(voltage):
    return 1 + np.tanh(2 * voltage - np.pi)


def voltage_profile(timeRange, start, stop, maxVoltage):
    a = 1 * (timeRange >= start)
    b = 1 * (timeRange <= stop)
    return maxVoltage * ((a + b) - np.ones(timeRange.shape[0]))


gyroscope = lozoya.acs.Sensor(
    operatingRange=(-1000, 1000)  # rpm,
)

reactionWheel = lozoya.acs.Actuator(transfer_function=reaction_wheel_transfer_function)

timeRange = np.linspace(start=0, stop=60, num=600)
voltageRange = np.linspace(start=0, stop=5, num=100)
reactionWheelTransferFunction = reactionWheel.transfer_function(voltageRange)
voltageProfile = voltage_profile(timeRange, start=20, stop=40, maxVoltage=5)
accelerationProfile = reactionWheel.transfer_function(voltageProfile)

# plotAngularVelocity = plot_scatter(
#     x=voltageRange,
#     y=reactionWheelTransferFunction,
#     xLabel='Voltage (Volts)',
#     yLabel='Angular Acceleration (rad / s\u00b2)',
#     title='Reaction Wheel Transfer Function'
# )

# plotProfileVoltage = plot_scatter(
#     x=timeRange,
#     y=voltageDomain,
#     xLabel='Time (s)',
#     yLabel='Voltage (Volts)',
#     title='Voltage Profile'
# )

# plotProfileAngularAcceleration = plot_scatter(
#     x=timeRange,
#     y=accelerationProfile,
#     xLabel='Time (s)',
#     yLabel='Angular Acceleration (rad / s\u00b2)',
#     title='Acceleration Profile'
# )
# plt.show()


r = 1
graph = sns.scatterplot(x=[r], y=[0])
fig = graph.get_figure()


def update(i):
    theta = 1
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    fig.set_xdata([x])
    fig.set_ydata([y])


ani = animation.FuncAnimation(fig, update, frames=1, interval=700, repeat=True)
plt.show()
