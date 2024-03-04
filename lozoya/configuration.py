import datetime
import operator
import os
import sys
from pathlib import Path

import numpy as np
import scipy.stats as st

OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt, '==': operator.eq}
INVERTED_OPERATORS = {operator.ge: '>=', operator.le: '<=', operator.gt: '>', operator.lt: '<', operator.eq: '=='}
OPERATORS_WORDS = {
    'greaterthanorequalto': '>=', 'lessthanorequalto': '<=', 'greaterthan': '>', 'lessthan': '<',
    'equal':                '=='
}
UNITS = ('km', 'm', 'mi', 'ft')
DISTRIBUTIONS = [  # st.alpha,
    st.anglit, st.arcsine,  # st.beta,
    # st.betaprime,
    st.bradford, st.burr, st.cauchy,  # st.chi,
    # st.chi2,
    st.cosine,  # st.dgamma,
    st.dweibull,  # st.erlang,
    st.expon,  # st.exponnorm,
    st.exponweib, st.exponpow,  # st.f,
    st.fatiguelife, st.fisk, st.foldcauchy, st.foldnorm,  # st.frechet_r,
    # st.frechet_l,
    st.genlogistic, st.genpareto,  # st.gennorm,
    st.genexpon, '''st.genextreme, 
    st.gausshyper, 
    st.gamma, 
    st.gengamma,
    st.genhalflogistic, 
    st.gilbrat, 
    st.gompertz, 
    st.gumbel_r,
    st.gumbel_l, 
    st.halfcauchy, 
    st.halflogistic, 
    st.halfnorm,
    st.halfgennorm, 
    st.hypsecant, 
    st.invgamma, 
    st.invgauss,
    st.invweibull,
    st.johnsonsb,
    st.johnsonsu, 
    st.ksone,
    st.kstwobign, 
    st.laplace, 
    st.levy, 
    st.levy_l, 
    st.levy_stable,
    st.logistic, 
    st.loggamma, 
    st.loglaplace, 
    st.lognorm,
    st.lomax, 
    st.maxwell, 
    st.mielke, 
    st.nakagami, 
    st.ncx2, 
    st.ncf,
    st.nct, 
    st.norm, 
    st.pareto, 
    st.pearson3, 
    st.powerlaw,
    st.powerlognorm, 
    st.powernorm, 
    st.rdist, 
    st.reciprocal,
    st.rayleigh, 
    st.rice, 
    st.recipinvgauss, 
    st.semicircular,
    st.t, 
    st.triang, 
    st.truncexpon, 
    st.truncnorm,
    st.tukeylambda, 
    st.uniform, 
    st.vonmises, 
    st.vonmises_line,
    st.wald, 
    st.weibull_min, 
    st.weibull_max, 
    st.wrapcauchy''']
DISTRIBUTIONNAMES = ['alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'cauchy', 'chi', 'chi2',
                     'cosine', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f',
                     'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic',
                     'genpareto', 'gennorm', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma',
                     'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic',
                     'halfnorm', 'halfgennorm', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb',
                     'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'levy', 'levy_l', 'levy_stable', 'logistic',
                     'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf',
                     'nct', 'norm', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist',
                     'reciprocal', 'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon',
                     'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald', 'weibull_min',
                     'weibull_max', 'wrapcauchy']
'''
DISTRIBUTIONNAMES = {
    'alpha': 'Alpha',
    'anglit': ,
    'arcsine': ,
    'beta': ,
    'betaprime': ,
    'bradford': ,
    'burr': ,
    'cauchy': ,
    'chi': ,
    'chi2': ,
    'cosine': ,
    'dgamma': ,
    'dweibull': ,
    'erlang': ,
    'expon': ,
    'exponnorm': ,
    'exponweib': , 
    'exponpow': , 
    'f': , 
    'fatiguelife': , 
    'fisk': ,
    'foldcauchy': , 
    'foldnorm': , 
    'frechet_r': , 
    'frechet_l': ,
    'genlogistic': , 
    'genpareto': , 
    'gennorm': , 
    'genexpon': ,
    'genextreme': , 
    'gausshyper': , 
    'gamma': , 
    'gengamma': ,
    'genhalflogistic': , 
    'gilbrat': , 
    'gompertz': , 
    'gumbel_r': ,
    'gumbel_l': , 
    'halfcauchy': , 
    'halflogistic': , 
    'halfnorm': ,
    'halfgennorm': , 
    'hypsecant': , 
    'invgamma': , 
    'invgauss': ,
    'invweibull': , 
    'johnsonsb': , 
    'johnsonsu': , 
    'ksone': ,
    'kstwobign': , 
    'laplace': , 
    'levy': , 
    'levy_l': , 
    'levy_stable': ,
    'logistic': , 
    'loggamma': , 
    'loglaplace': , 
    'lognorm': ,
    'lomax': , 
    'maxwell': , 
    'mielke': , 
    'nakagami': , 
    'ncx2': , 
    'ncf': ,
    'nct': , 
    'norm': , 
    'pareto': , 
    'pearson3': , 
    'powerlaw': ,
    'powerlognorm': , 
    'powernorm': , 
    'rdist': , 
    'reciprocal': ,
    'rayleigh': , 
    'rice': , 
    'recipinvgauss': , 
    'semicircular': ,
    't': , 
    'triang': , 
    'truncexpon': , 
    'truncnorm': ,
    'tukeylambda': , 
    'uniform': , 
    'vonmises': , 
    'vonmises_line': ,
    'wald': , 
    'weibull_min': , 
    'weibull_max': , 
    'wrapcauchy': 
}'''
PlotsDir = 'plot'
BarPlotsDir = os.path.join(PlotsDir, 'BarPlots')
BoxPlotsDir = os.path.join(PlotsDir, 'BoxPlots')
DistributionPlotsDir = os.path.join(PlotsDir, 'DistributionPlots')
DistributionFitPlotsDir = os.path.join(PlotsDir, 'DistributionFitPlots')
HeatmapPlotsDir = os.path.join(PlotsDir, 'HeatmapPlots')
ScatterPlotsDir = os.path.join(PlotsDir, 'ScatterPlots')
TablePlotsDir = os.path.join(PlotsDir, 'TablePlots')
ViolinPlotsDir = os.path.join(PlotsDir, 'ViolinPlots')
StatComparisonDir = os.path.join(PlotsDir, 'StatisticComparison')
PlotDirs = [BarPlotsDir, BoxPlotsDir, DistributionPlotsDir, DistributionFitPlotsDir, HeatmapPlotsDir, ScatterPlotsDir,
            TablePlotsDir, ViolinPlotsDir, StatComparisonDir]
BarPlotSuffix = '_bar'
BoxPlotSuffix = '_box'
DistributionPlotSuffix = '_distribution'
DistributionFitPlotSuffix = '_distributionFit'
ScatterPlotSuffix = '_scatter'
TablePlotSuffix = '_table'
ViolinPlotSuffix = '_violin'
Quantiles = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1]
# FUNCTION STRINGS ------------------------------------------------------
DISTRIBUTIONS = [st.alpha, st.anglit,  # st.arcsine, # weird
                 # st.beta, #weird
                 # st.betaprime, # TIME CONSUMING
                 st.bradford, st.burr, st.burr12, st.cauchy, st.chi, st.chi2, st.cosine, st.dgamma, st.dweibull,
                 # st.erlang,
                 st.expon, st.exponnorm, st.exponweib,  # TIME CONSUMING
                 # st.exponpow, # TIME CONSUMING
                 st.f,  # st.fatiguelife,
                 st.fisk, st.foldcauchy, st.foldnorm,  # st.frechet_r,
                 # st.frechet_l,
                 # st.genlogistic, # TIME CONSUMING
                 # st.genpareto, # TIME CONSUMING
                 # st.gennorm,
                 # st.genexpon, # TIME CONSUMING
                 # st.genextreme,
                 # st.gausshyper,
                 st.gamma, st.gengamma, st.genhalflogistic, st.gibrat,  # st.gompertz, # TIME CONSUMING
                 st.gumbel_r, st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm,  # st.halfgennorm,
                 st.hypsecant, st.invgamma, st.invgauss, st.invweibull,  # st.johnsonsb, # TIME CONSUMING
                 # st.johnsonsu,
                 # st.kappa4, # TIME CONSUMING
                 # st.kappa3,
                 # st.ksone,
                 # st.kstwobign,
                 st.laplace, st.levy, st.levy_l,  # st.levy_stable,
                 st.logistic, st.loggamma,  # st.loglaplace,
                 st.lognorm,  # TIME CONSUMING
                 st.lomax, st.maxwell,  # st.mielke, # TIME CONSUMING
                 # st.moyal,
                 st.nakagami,  # st.ncx2,
                 # st.ncf,
                 # st.nct, # TIME CONSUMING
                 st.norm, st.pareto, st.pearson3,  # st.powerlaw, # TIME CONSUMING
                 # st.powerlognorm, # TIME CONSUMING
                 # st.powernorm, # TIME CONSUMING
                 # st.rdist,
                 # st.reciprocal, # TIME CONSUMING
                 # st.rayleigh, # DOES NOT WORK
                 # st.rice,
                 # st.recipinvgauss, # TIME CONSUMING
                 st.semicircular, st.t,  # st.triang,
                 st.truncexpon,  # st.truncnorm,
                 # st.tukeylambda, (NO FUNCTION PROVIDED)
                 # st.uniform,
                 # st.vonmises,
                 # st.vonmises_line,
                 st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy]
DISTRIBUTION_PARAMS = {
    'alpha':           ['a'], 'anglit': [], 'arcsine': [], 'beta': ['a', 'b'], '_beta': ['t', 'y_'],
    'betaprime':       ['a', 'b'], 'bradford': ['c'], 'burr': ['c', 'd'], 'burr12': ['c', 'd'],
    'cauchy':          [], 'chi': ['df'], 'chi2': ['df'], 'cosine': [], 'dgamma': ['a'], 'dweibull': ['c'],
    'erlang':          [], 'expon': [], 'exponnorm': ['K'], 'exponweib': ['a', 'c'], 'exponpow': ['b'],
    'f':               [],  # 'fatiguelife': ['c'],
    'fisk':            ['c'], 'foldcauchy': ['c'], 'foldnorm': ['c'], 'frechet_r': [], 'frechet_l': [],
    'genlogistic':     ['c'], 'genpareto': ['c'], 'gennorm': [], 'genexpon': ['a', 'b', 'c'],
    'genextreme':      [], 'gausshyper': [], 'gamma': ['a'], '_gamma': ['z'], 'gengamma': ['a', 'c'],
    'genhalflogistic': ['c'], 'gilbrat': [], 'gompertz': ['c'], 'gumbel_r': [], 'gumbel_l': [],
    'halfcauchy':      [], 'halflogistic': [], 'halfnorm': [], 'halfgennorm': [], 'hypsecant': [],
    'invgamma':        ['a'], 'invgauss': ['\mu'], 'invweibull': ['c'], 'johnsonsb': ['a', 'b'],
    'johnsonsu':       [], 'kappa4': ['h', 'k'], 'ksone': [], 'kstwobign': [], 'laplace': [], 'levy': [],
    'levy_l':          [], 'levy_stable': [], 'logistic': [], 'loggamma': ['c'], 'loglaplace': [],
    'lognorm':         ['s'], 'lomax': ['c'], 'maxwell': [], 'mielke': ['k', 's'], 'moyal': [],
    'nakagami':        ['nu'], 'ncx2': [], 'ncf': [], 'nct': ['df', 'nc'], 'norm': ['k', 's'],
    '_normal_cdf':     ['t'], 'pareto': ['b'], 'pearson3': [], 'powerlaw': ['a'],
    'powerlognorm':    ['c', 's'], 'powernorm': ['c'], 'rdist': [], 'reciprocal': ['a', 'b'],
    'rayleigh':        ['r'], 'rice': [], 'recipinvgauss': ['\mu'], 'semicircular': [], 't': ['df'],
    'triang':          [], 'truncexpon': ['b'], 'truncnorm': [], 'tukeylambda': [], 'uniform': [],
    'vonmises':        [], 'vonmises_line': [], 'wald': [], 'weibull_min': ['c'], 'weibull_max': ['c'],
    'wrapcauchy':      ['c']
}
DISTRIBUTION_NAMES = {
    'alpha':         'Alpha', 'anglit': 'Anglit', 'arcsine': 'Arcsine', 'beta': 'Beta',
    '_beta':         'Beta Function', 'betaprime': 'Beta Prime', 'bradford': 'Bradford',
    'burr':          'Burr Type III', 'burr12': 'Burr Type XII', 'cauchy': 'Cauchy', 'chi': 'Chi',
    'chi2':          'Chi-Squared', 'cosine': 'Cosine', 'dgamma': 'Double Gamma', 'dweibull': 'Double Weibull',
    'erlang':        'Erlang', 'expon': 'Exponential', 'exponnorm': 'Exponentially Modified Normal',
    'exponweib':     'Exponentiated Weibull', 'exponpow': 'Exponential Power', 'f': 'F',
    # 'fatiguelife': 'Fatigue-Life (Birnbaum-Saunders)',
    'fisk':          'Fisk', 'foldcauchy': 'Folded Cauchy', 'foldnorm': 'Folded Normal',
    'frechet_r':     'Frechet R', 'frechet_l': 'Frechet L', 'genlogistic': 'Generalized Logistic',
    'genpareto':     'Generalized Pareto', 'gennorm': 'Generalized Normal',
    'genexpon':      'Generalized Exponential', 'genextreme': 'Generalized Extreme',
    'gausshyper':    'Gauss Hypergeometric', 'gamma': 'Gamma', '_gamma': 'Gamma Function',
    'gengamma':      'Generalized Gamma', 'genhalflogistic': 'Generalized Half-Logistic',
    'gilbrat':       'Gilbrat', 'gompertz': 'Gompertz (Truncated Gumbel)',
    'gumbel_r':      'Right-Skewed Gumbel', 'gumbel_l': 'Left-Skewed Gumbel', 'halfcauchy': 'Half-Cauchy',
    'halflogistic':  'Half-Logistic', 'halfnorm': 'Half-Normal',
    'halfgennorm':   'The Upper Half of a Generalized Normal', 'hypsecant': 'Hyperbolic Secant',
    'invgamma':      'Inverted Gamma', 'invgauss': 'Inverse Gaussian', 'invweibull': 'Inverted Weibull',
    'johnsonsb':     'Johnson SB', 'johnsonsu': 'Johnson SU', 'kappa4': 'Kappa 3', 'ksone': 'ksone',
    'kstwobign':     'kstwobign', 'laplace': 'Laplace', 'levy': 'Levy', 'levy_l': 'Left-Skewed Levy',
    'levy_stable':   'Levy_Stable', 'logistic': 'Logistic', 'loggamma': 'Log Gamma',
    'loglaplace':    'Log-Laplace', 'lognorm': 'Lognormal', 'lomax': 'Lomax (Pareto of the Second Kind)',
    'maxwell':       'Maxwell', 'mielke': r"Mielke's Beta-Kappa", 'moyal': 'Moyal', 'nakagami': 'Nakagami',
    'ncx2':          'Non-Central Chi-Squared', 'ncf': 'Non-Central F Distribution',
    'nct':           r"Non-Central Student's T", 'norm': 'Normal', '_normal_cdf': 'Normal CDF',
    'pareto':        'Pareto', 'pearson3': 'Pearson Type III', 'powerlaw': 'Power-Function',
    'powerlognorm':  'Power Log-Normal', 'powernorm': 'Power Normal', 'rdist': 'R-Distributed',
    'reciprocal':    'Reciprocal', 'rayleigh': 'Rayleigh', 'rice': 'Rice',
    'recipinvgauss': 'Reciprocal Inverse Gaussian', 'semicircular': 'Semicircular',
    't':             r"Student's T", 'triang': 'Triangular', 'truncexpon': 'Truncated Exponential',
    'truncnorm':     'Truncated Normal', 'tukeylambda': 'Tukey-Lambda', 'uniform': 'Uniform',
    'vonmises':      'Von Mises', 'vonmises_line': 'Von Mises', 'wald': 'Wald',
    'weibull_min':   'Weibull Minimum', 'weibull_max': 'Weibull Maximum',
    'wrapcauchy':    'Wrapped Cauchy'
}
# FUNCTION STRINGS ------------------------------------------------------
DISTRIBUTIONS = [  # st.alpha,
    st.anglit, st.arcsine,  # st.beta,
    # st.betaprime,
    st.bradford, st.burr, st.burr12, st.cauchy,  # st.chi,
    # st.chi2,
    st.cosine,  # st.dgamma,
    st.dweibull,  # st.erlang,
    st.expon,  # st.exponnorm,
    st.exponweib, st.exponpow,  # st.f,
    st.fatiguelife, st.fisk, st.foldcauchy, st.foldnorm,  # st.frechet_r,
    # st.frechet_l,
    st.genlogistic, st.genpareto,  # st.gennorm,
    # st.genexpon,
    # st.genextreme,
    # st.gausshyper,
    # st.gamma,
    # st.gengamma,
    # st.genhalflogistic,
    # st.gilbrat,
    # st.gompertz,
    # st.gumbel_r,
    # st.gumbel_l,
    st.halfcauchy, st.halflogistic, st.halfnorm,  # st.halfgennorm,
    st.hypsecant,  # st.invgamma,
    st.invgauss, st.invweibull, st.johnsonsb,  # st.johnsonsu,
    # st.ksone,
    # st.kstwobign,
    st.laplace, st.levy, st.levy_l,  # st.levy_stable,
    st.logistic,  # st.loggamma,
    # st.loglaplace,
    # st.lognorm,
    st.lomax, st.maxwell, st.mielke,  # st.moyal,
    # st.nakagami,
    # st.ncx2,
    # st.ncf,
    # st.nct,
    st.norm, st.pareto,  # st.pearson3,
    st.powerlaw,  # st.powerlognorm,
    # st.powernorm,
    # st.rdist,
    # st.reciprocal,
    st.rayleigh,  # st.rice,
    st.recipinvgauss, st.semicircular,  # st.t,
    # st.triang,
    st.truncexpon,  # st.truncnorm,
    # st.tukeylambda,
    # st.uniform,
    # st.vonmises,
    # st.vonmises_line,
    st.wald, st.weibull_min, st.weibull_max,  # st.wrapcauchy
]
DISTRIBUTION_PARAMS = {
    'alpha':       [], 'anglit': [], 'arcsine': [], 'beta': [], 'betaprime': [], 'bradford': ['c'],
    'burr':        ['c', 'd'], 'burr12': ['c', 'd'], 'cauchy': [], 'chi': [], 'chi2': [], 'cosine': [],
    'dgamma':      [], 'dweibull': ['c'], 'erlang': [], 'expon': [], 'exponnorm': [],
    'exponweib':   ['a', 'c'], 'exponpow': ['b'], 'f': [], 'fatiguelife': ['c'], 'fisk': ['c'],
    'foldcauchy':  ['c'], 'foldnorm': ['c'], 'frechet_r': [], 'frechet_l': [], 'genlogistic': ['c'],
    'genpareto':   ['c'], 'gennorm': [], 'genexpon': [], 'genextreme': [], 'gausshyper': [],
    'gamma':       [], 'gengamma': [], 'genhalflogistic': [], 'gilbrat': [], 'gompertz': [],
    'gumbel_r':    [], 'gumbel_l': [], 'halfcauchy': [], 'halflogistic': [], 'halfnorm': [],
    'halfgennorm': [], 'hypsecant': [], 'invgamma': [], 'invgauss': ['\mu'], 'invweibull': ['c'],
    'johnsonsb':   ['a', 'b', 'c', 'd'], 'johnsonsu': [], 'ksone': [], 'kstwobign': [], 'laplace': [],
    'levy':        [], 'levy_l': [], 'levy_stable': [], 'logistic': [], 'loggamma': [], 'loglaplace': [],
    'lognorm':     [], 'lomax': ['c'], 'maxwell': [], 'mielke': ['k', 's'], 'moyal': [], 'nakagami': [],
    'ncx2':        [], 'ncf': [], 'nct': [], 'norm': ['k', 's'], 'pareto': ['b'], 'pearson3': [],
    'powerlaw':    ['a'], 'powerlognorm': [], 'powernorm': [], 'rdist': [], 'reciprocal': [],
    'rayleigh':    ['r'], 'rice': [], 'recipinvgauss': ['\mu'], 'semicircular': [], 't': [],
    'triang':      [], 'truncexpon': ['b'], 'truncnorm': [], 'tukeylambda': [], 'uniform': [],
    'vonmises':    [], 'vonmises_line': [], 'wald': [], 'weibull_min': ['c'], 'weibull_max': ['c'],
    'wrapcauchy':  ['c']
}
DISTRIBUTION_NAMES = {
    'alpha':         'Alpha', 'anglit': 'Anglit', 'arcsine': 'Arcsine', 'beta': 'Beta',
    'betaprime':     'Beta Prime', 'bradford': 'Bradford', 'burr': 'Burr Type III',
    'burr12':        'Burr Type XII', 'cauchy': 'Cauchy', 'chi': 'Chi', 'chi2': 'Chi-Squared',
    'cosine':        'Cosine', 'dgamma': 'Double Gamma', 'dweibull': 'Double Weibull', 'erlang': 'Erlang',
    'expon':         'Exponential', 'exponnorm': 'Exponentially Modified Normal',
    'exponweib':     'Exponentiated Weibull', 'exponpow': 'Exponential Power', 'f': 'F',
    'fatiguelife':   'Fatigue-Life (Birnbaum-Saunders)', 'fisk': 'Fisk', 'foldcauchy': 'Folded Cauchy',
    'foldnorm':      'Folded Normal', 'frechet_r': 'Frechet R', 'frechet_l': 'Frechet L',
    'genlogistic':   'Generalized Logistic', 'genpareto': 'Generalized Pareto',
    'gennorm':       'Generalized Normal', 'genexpon': 'Generalized Exponential',
    'genextreme':    'Generalized Extreme', 'gausshyper': 'Gauss Hypergeometric', 'gamma': 'Gamma',
    'gengamma':      'Generalized Gamma', 'genhalflogistic': 'Generalized Half-Logistic',
    'gilbrat':       'Gilbrat', 'gompertz': 'Gompertz (Truncated Gumbel)',
    'gumbel_r':      'Right-Skewed Gumbel', 'gumbel_l': 'Left-Skewed Gumbel', 'halfcauchy': 'Half-Cauchy',
    'halflogistic':  'Half-Logistic', 'halfnorm': 'Half-Normal',
    'halfgennorm':   'The Upper Half of a Generalized Normal', 'hypsecant': 'Hyperbolic Secant',
    'invgamma':      'Inverted Gamma', 'invgauss': 'Inverse Gaussian', 'invweibull': 'Inverted Weibull',
    'johnsonsb':     "Johnson's B", 'johnsonsu': "Johnson's U", 'ksone': 'ksone',
    'kstwobign':     'kstwobign', 'laplace': 'Laplace', 'levy': 'Levy', 'levy_l': 'Left-Skewed Levy',
    'levy_stable':   'Levy_Stable', 'logistic': 'Logistic', 'loggamma': 'Log Gamma',
    'loglaplace':    'Log-Laplace', 'lognorm': 'Lognormal', 'lomax': 'Lomax (Pareto of the Second Kind)',
    'maxwell':       'Maxwell', 'mielke': r"Mielke's Beta-Kappa", 'moyal': 'Moyal', 'nakagami': 'Nakagami',
    'ncx2':          'Non-Central Chi-Squared', 'ncf': 'Non-Central F Distribution',
    'nct':           r"Non-Central Student's T", 'norm': 'Normal', 'pareto': 'Pareto',
    'pearson3':      'Pearson Type III', 'powerlaw': 'Power-Function', 'powerlognorm': 'Power Log-Normal',
    'powernorm':     'Power Normal', 'rdist': 'R-Distributed', 'reciprocal': 'Reciprocal',
    'rayleigh':      'Rayleigh', 'rice': 'Rice', 'recipinvgauss': 'Reciprocal Inverse Gaussian',
    'semicircular':  'Semicircular', 't': r"Student's T", 'triang': 'Triangular',
    'truncexpon':    'Truncated Exponential', 'truncnorm': 'Truncated Normal',
    'tukeylambda':   'Tukey-Lambda', 'uniform': 'Uniform', 'vonmises': 'Von Mises',
    'vonmises_line': 'Von Mises', 'wald': 'Wald', 'weibull_min': 'Weibull Minimum',
    'weibull_max':   'Weibull Maximum', 'wrapcauchy': 'Wrapped Cauchy'
}
now = datetime.datetime.now()
# STATS CONFIG
sigfigs = 2
N = 1
centralTendenciesTableStats = ['Mean', 'Median']
dispersionTableStats = ['Standard Deviation', 'Variance', 'Skew', 'Kurtosis']
iqrTableStats = ['Lower IQR', 'First Quartile', 'Median', 'Third Quartile', 'Upper IQR']
# PATHS
server = os.path.dirname(os.path.dirname(sys.argv[0]))
jobRoot = os.path.join(server, 'jobs')
jobImgPath = 'plots'
templateDir = 'template'
analysisTemplatePath = os.path.join(templateDir, 'analysis.html')
navTemplatePath = os.path.join(templateDir, 'navigator.html')
# PLOT CONFIG
snsContext = {
    'lines.linewidth': 1, 'figure.figsize': (1, 1), 'figure.facecolor': 'white',
    'font.family':     ['sans-serif', ]
}
snsStyle = {'font.family': 'serif', 'font.serif': ['Times', 'Palatino', 'serif']}
lineStyles = ['--', ':', '-.']
ciFill = {'alpha': 0.2, 'color': '#00AFBB', 'linestyle': '-', 'zorder': -2}
ciBorder = {'color': '#ff9f00', 'linestyle': '--', 'zorder': -1}
piFill = {'alpha': 0.2, 'color': '#CC79A7', 'linestyle': '--', 'zorder': -3}
piBorder = {'color': '#CC79A7', 'linestyle': ':', 'zorder': -1}
rLine = {'alpha': 0.9, 'color': '0.1', 'label': 'Regression Fit', 'linewidth': 1.5, 'linestyle': '-', 'zorder': -1}
tufte = False
# STATISTICS ------------------------------------------------------------
Quantiles = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1]
now = datetime.datetime.now()
# STATS CONFIG
funcFams = {'Exponential': lambda x, a, b: a * np.exp(b * x), 'Logarithmic': lambda x, a, b: a * np.log(x) + b}
sigfigs = 2
N = 1
centralTendenciesTableStats = ['Mean', 'Median']
dispersionTableStats = ['Standard Deviation', 'Variance', 'Skew', 'Kurtosis']
iqrTableStats = ['Lower IQR', 'First Quartile', 'Median', 'Third Quartile', 'Upper IQR']
# PATHS
server = os.path.dirname(os.path.dirname(sys.argv[0]))
jobRoot = os.path.join(server, 'jobs')
jobImgPath = 'plots'
templateDir = 'template'
analysisTemplatePath = os.path.join(templateDir, 'analysis.html')
navTemplatePath = os.path.join(templateDir, 'navigator.html')
# LIST OF ALL EXTENSIONS ASSOCIATED WITH EACH FILE TYPE
EXTENSIONS = {
    'csv':   ['csv',
              'txt'],
    'excel': ['xls',
              'xlsx',
              'xlsm',
              'xltx',
              'xltm'],
    'hdf5':  ['hdf5'],
    'wav':   ['wav']
}
# PLOT CONFIG
snsContext = {
    'lines.linewidth': 1, 'figure.figsize': (1, 1), 'figure.facecolor': 'white',
    'font.family':     ['sans-serif', ]
}
snsStyle = {'font.family': 'serif', 'font.serif': ['Times', 'Palatino', 'serif']}
colors = ["#00AFBB", "#ff9f00", "#CC79A7", "#009E73", "#66ccff",  # Not as distinguishable with #00AFBB for colorblind
          "#F0E442"]
lineStyles = ['--', ':', '-.']
ciFill = {'alpha': 0.2, 'color': '#00AFBB', 'linestyle': '-', 'zorder': -2}
ciBorder = {'color': '#ff9f00', 'linestyle': '--', 'zorder': -1}
piFill = {'alpha': 0.2, 'color': '#CC79A7', 'linestyle': '--', 'zorder': -3}
piBorder = {'color': '#CC79A7', 'linestyle': ':', 'zorder': -1}
tufte = False
# STATISTICS ------------------------------------------------------------
Quantiles = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1]
LoParTechnologies = str(Path(__file__).parents[2])
ROOT = os.getcwd()
BASE_MAP = os.path.join(ROOT, 'map_api', 'baseMap.html')
GRAPHS = os.path.join(LoParTechnologies, os.path.join('LoParDynamicPlotGenerator', 'plot'))
SIMULATION_GRAPHS = os.path.join(GRAPHS, 'Simulation2')
REGRESSION_GRAPHS = os.path.join(GRAPHS, 'Regression6')
CLASSIFICATION_GRAPHS = os.path.join(GRAPHS, 'Classification')
CLUSTERING_GRAPHS = os.path.join(GRAPHS, 'Clustering')
DIMENSIONALITY_GRAPHS = os.path.join(GRAPHS, 'DimensionalityReduction')
BASE_GRAPH = os.path.join(GRAPHS, 'RegressionPlot.html')
CLASSIFICATION_GRAPH = os.path.join(CLASSIFICATION_GRAPHS, 'ClassificationPlot.html')
CLUSTERING_GRAPH = os.path.join(CLUSTERING_GRAPHS, 'ClusteringPlot.html')
DIMENSIONALITY_GRAPH = os.path.join(DIMENSIONALITY_GRAPHS, 'DimensionalityReductionPlot.html')
COUNT_GRAPH = os.path.join(GRAPHS, 'CountPlot.html')
MARKOV_CHAIN_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MarkovChainPlot2.html')
MARKOV_CHAIN_FREQUENCY_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MarkovChainFrequencyPlot.html')
MARKOV_CHAIN_FREQUENCY_CURVES_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MarkovChainFrequencyCurvesPlot.html')
SIMULATION_GRAPH = os.path.join(SIMULATION_GRAPHS, 'Simulation2.html')
SIMULATION_DATA_GRAPH = os.path.join(SIMULATION_GRAPHS, 'SimulationData.html')
SIMULATION_FREQUENCY_GRAPH = os.path.join(SIMULATION_GRAPHS, 'SimulationFrequency.html')
MONTE_CARLO_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MonteCarloPlot.html')
REGRESSION_GRAPH = os.path.join(REGRESSION_GRAPHS, 'RegressionPlot.html')
SEARCH_RESULTS_TABLE = os.path.join(GRAPHS, 'SearchResults2.html')
# DATABASE = os.path.join(ROOT, 'Database')
DATABASE = os.path.join(LoParTechnologies, 'LoParDatasets', 'CollectedData')
ICONS = os.path.join(ROOT, 'interface7', 'InterfaceUtilities0', 'icon2')
HEADERS = os.path.join(ROOT, 'interface7', 'InterfaceUtilities0', 'label', 'Headers.txt')
DATABASE_VARIABLES = os.path.join(ROOT, 'interface7', 'InterfaceUtilities0', 'label', 'Database Variables.txt')
temp_RSP = os.path.join(ROOT, 'Utilities', 'Preferences', 'Report_Preferences.txt')
RSP = os.path.join(ROOT, 'Utilities', 'Preferences', 'Report_Preferences0.txt')
TEMP_SAVE = os.path.join(DATABASE, 'temp.txt')
VARIABLES = os.path.join(ROOT, 'Utilities14', 'VarsVars.py')
# DOCUMENTATION = os.path.join(ROOT, '1', '1 sunburst.html')
DOCUMENTATION = os.path.join(LoParTechnologies, 'LoParDocumentation0', '1 sunburst.html')
UTILITIES = os.path.join(ROOT, 'Utilities14')
TRANSITION_MATRIX_PLOT = os.path.join(SIMULATION_GRAPHS, 'TransitionMatrix.html')
LoParTechnologies = str(Path(__file__).parents[2])
ROOT = os.getcwd()
BASE_MAP = os.path.join(ROOT, 'map_api', 'baseMap.html')
GRAPHS = os.path.join(LoParTechnologies, 'LoParPlotGenerator0', 'DynamicPlots0')
SIMULATION_GRAPHS = os.path.join(GRAPHS, 'Simulation2')
REGRESSION_GRAPHS = os.path.join(GRAPHS, 'Regression6')
CLASSIFICATION_GRAPHS = os.path.join(GRAPHS, 'Classification')
CLUSTERING_GRAPHS = os.path.join(GRAPHS, 'Clustering')
DIMENSIONALITY_GRAPHS = os.path.join(GRAPHS, 'DimensionalityReduction')
BASE_GRAPH = os.path.join(GRAPHS, 'RegressionPlot.html')
CLASSIFICATION_GRAPH = os.path.join(CLASSIFICATION_GRAPHS, 'ClassificationPlot.html')
CLUSTERING_GRAPH = os.path.join(CLUSTERING_GRAPHS, 'ClusteringPlot.html')
DIMENSIONALITY_GRAPH = os.path.join(DIMENSIONALITY_GRAPHS, 'DimensionalityReductionPlot.html')
COUNT_GRAPH = os.path.join(GRAPHS, 'CountPlot.html')
MARKOV_CHAIN_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MarkovChainPlot2.html')
MARKOV_CHAIN_FREQUENCY_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MarkovChainFrequencyPlot.html')
MARKOV_CHAIN_FREQUENCY_CURVES_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MarkovChainFrequencyCurvesPlot.html')
SIMULATION_GRAPH = os.path.join(SIMULATION_GRAPHS, 'Simulation2.html')
SIMULATION_DATA_GRAPH = os.path.join(SIMULATION_GRAPHS, 'SimulationData.html')
SIMULATION_FREQUENCY_GRAPH = os.path.join(SIMULATION_GRAPHS, 'SimulationFrequency.html')
MONTE_CARLO_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MonteCarloPlot.html')
REGRESSION_GRAPH = os.path.join(REGRESSION_GRAPHS, 'RegressionPlot.html')
SEARCH_RESULTS_TABLE = os.path.join(GRAPHS, 'SearchResults2.html')
# DATABASE = os.path.join(ROOT, 'Database')
DATABASE = os.path.join(LoParTechnologies, 'LoParDatasets', 'CollectedData')
ICONS = os.path.join(ROOT, 'interface7', 'InterfaceUtilities0', 'icon2')
DATABASE_VARIABLES = os.path.join(
    os.path.join(os.path.join(os.path.join(ROOT, 'interface7'), 'InterfaceUtilities0'), 'label'),
    'Database Variables.txt'
)
temp_RSP = os.path.join(os.path.join(os.path.join(ROOT, 'Utilities14'), 'Preferences'), 'Report_Preferences0.txt')
RSP = os.path.join(os.path.join(os.path.join(ROOT, 'Utilities14'), 'Preferences'), 'Report_Preferences0.txt')
TEMP_SAVE = os.path.join(DATABASE, 'temp.txt')
VARIABLES = os.path.join(os.path.join(ROOT, 'Utilities14'), 'VarsVars.py')
# DOCUMENTATION = os.path.join(os.path.join(ROOT, '1'), '1 sunburst.html')
DOCUMENTATION = os.path.join(LoParTechnologies, os.path.join(os.path.join('LoParDocumentation0', '1 sunburst.html')))
UTILITIES = os.path.join(ROOT, 'Utilities14')
TRANSITION_MATRIX_PLOT = os.path.join(SIMULATION_GRAPHS, 'TransitionMatrix.html')
dashboard = 'dashboard.html'
ROOT = os.getcwd()
BASE_MAP = os.path.join(os.path.join(ROOT, 'map_api'), 'baseMap.html')
GRAPHS = os.path.join(ROOT, 'graph2')
BASE_GRAPH = os.path.join(GRAPHS, 'RegressionPlot.html')
COUNT_GRAPH = os.path.join(GRAPHS, 'CountPlot.html')
MARKOV_CHAIN_GRAPH = os.path.join(GRAPHS, 'MarkovChainPlot2.html')
MONTE_CARLO_GRAPH = os.path.join(GRAPHS, 'MonteCarloPlot.html')
REGRESSOR_GRAPH = os.path.join(GRAPHS, 'RegressorPlot.html')
SEARCH_RESULTS_TABLE = os.path.join(GRAPHS, 'SearchResults2.html')
DATABASE = os.path.join(ROOT, 'Database')
ICONS = os.path.join(os.path.join(os.path.join(ROOT, 'interface7'), 'InterfaceUtilities0'), 'icon2')
STATES = os.path.join(os.path.join(DATABASE, 'Information'), 'allStates.txt')
STATE_CODES = os.path.join(os.path.join(DATABASE, 'Information'), 'allStateCodes.txt')
temp_RSP = os.path.join(
    os.path.join(os.path.join(ROOT, 'utilities/Utilities14'), 'Preferences'),
    'Report_Preferences0.txt'
)
RSP = os.path.join(os.path.join(os.path.join(ROOT, 'utilities/Utilities14'), 'Preferences'), 'Report_Preferences0.txt')
TEMP_SAVE = os.path.join(DATABASE, 'temp.txt')
VARIABLES = os.path.join(os.path.join(ROOT, 'utilities/Utilities14'), 'variable2.py')

ROOT = os.getcwd()
ICON = os.path.join(ROOT, 'icon2')
LOGO = os.path.join(ICON, 'signal_tool.png')
UTIL = os.path.join(ROOT, 'utilities/Utilities14')
BASE_MAP = os.path.join(os.path.join(ROOT, 'map_api'), 'baseMap.html')
GRAPHS = os.path.join(ROOT, 'graph2')
SIMULATION_GRAPHS = os.path.join(GRAPHS, 'Simulation2')
REGRESSION_GRAPHS = os.path.join(GRAPHS, 'Regression6')
CLASSIFICATION_GRAPHS = os.path.join(GRAPHS, 'Classification')
CLUSTERING_GRAPHS = os.path.join(GRAPHS, 'Clustering')
DIMENSIONALITY_GRAPHS = os.path.join(GRAPHS, 'DimensionalityReduction')
BASE_GRAPH = os.path.join(GRAPHS, 'RegressionPlot.html')
CLASSIFICATION_GRAPH = os.path.join(CLASSIFICATION_GRAPHS, 'ClassificationPlot.html')
CLUSTERING_GRAPH = os.path.join(CLUSTERING_GRAPHS, 'ClusteringPlot.html')
DIMENSIONALITY_GRAPH = os.path.join(DIMENSIONALITY_GRAPHS, 'DimensionalityReductionPlot.html')
COUNT_GRAPH = os.path.join(GRAPHS, 'CountPlot.html')
MARKOV_CHAIN_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MarkovChainPlot2.html')
MARKOV_CHAIN_FREQUENCY_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MarkovChainFrequencyPlot.html')
MARKOV_CHAIN_FREQUENCY_CURVES_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MarkovChainFrequencyCurvesPlot.html')
SIMULATION_GRAPH = os.path.join(SIMULATION_GRAPHS, 'Simulation2.html')
SIMULATION_DATA_GRAPH = os.path.join(SIMULATION_GRAPHS, 'SimulationData.html')
SIMULATION_FREQUENCY_GRAPH = os.path.join(SIMULATION_GRAPHS, 'SimulationFrequency.html')
MONTE_CARLO_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MonteCarloPlot.html')
REGRESSION_GRAPH = os.path.join(REGRESSION_GRAPHS, 'RegressionPlot.html')
SEARCH_RESULTS_TABLE = os.path.join(GRAPHS, 'SearchResults2.html')
DATABASE = os.path.join(ROOT, 'Database')
TEMP_SAVE = os.path.join(DATABASE, 'temp.txt')
VARIABLES = os.path.join(os.path.join(ROOT, 'utilities/Utilities14'), '../nbi/variable2.py')
DOCUMENTATION = os.path.join(os.path.join(ROOT, '1'), '1 sunburst.html')
UTILITIES = os.path.join(ROOT, 'utilities/Utilities14')
TRANSITION_MATRIX_PLOT = os.path.join(SIMULATION_GRAPHS, 'TransitionMatrix.html')
ROOT = os.getcwd()
BASE_MAP = os.path.join(os.path.join(ROOT, 'map_api'), 'baseMap.html')
GRAPHS = os.path.join(ROOT, 'graph2')
BASE_GRAPH = os.path.join(GRAPHS, 'RegressionPlot.html')
CLASSIFICATION_GRAPH = os.path.join(GRAPHS, 'ClassificationPlot.html')
COUNT_GRAPH = os.path.join(GRAPHS, 'CountPlot.html')
MARKOV_CHAIN_GRAPH = os.path.join(GRAPHS, 'MarkovChainPlot2.html')
MONTE_CARLO_GRAPH = os.path.join(GRAPHS, 'MonteCarloPlot.html')
REGRESSION_GRAPH = os.path.join(GRAPHS, 'RegressionPlot.html')
SEARCH_RESULTS_TABLE = os.path.join(GRAPHS, 'SearchResults2.html')
DATABASE = os.path.join(ROOT, 'Database')
TEMP_SAVE = os.path.join(DATABASE, 'temp.txt')
VARIABLES = os.path.join(os.path.join(ROOT, 'Utilities14'), 'Variables.py')
DOCUMENTATION = os.path.join(os.path.join(ROOT, '1'), '1 sunburst.html')
UTILITIES = os.path.join(ROOT, 'Utilities14')
TRANSITION_MATRIX = os.path.join(GRAPHS, 'TransitionMatrix.html')
ROOT = os.getcwd()
BASE_MAP = os.path.join(os.path.join(os.path.join(ROOT, 'graph2'), 'map_api'), 'baseMap.html')
BASE_GRAPH = os.path.join(os.path.join(ROOT, 'graph2'), 'Tooltip.html')
DATABASE = os.path.join(ROOT, 'Database')
DATABASE = os.path.join(DATABASE, 'Years')  # TODO SET THIS PATH FROM THE DATABASE MENU
VALUES = os.path.join(
    os.path.join(os.path.join(os.path.join(ROOT, 'interface7'), 'InterfaceUtilities0'), 'label'),
    'ParameterValues.txt'
)
ITEM_NAMES = os.path.join(
    os.path.join(os.path.join(os.path.join(ROOT, 'interface7'), 'InterfaceUtilities0'), 'label'), 'ItemNames.txt'
)
TEMP_SAVE = os.path.join(DATABASE, 'temp.txt')
ROOT = os.getcwd()
BASE_GRAPH = os.path.join(os.path.join(ROOT, 'graph2'), 'Tooltip.html')
DATABASE = os.path.join(ROOT, 'Database')
DATABASE = os.path.join(DATABASE, '00')  # TODO SET THIS PATH FROM THE DATABASE MENU
ROOT = os.getcwd()
BASE_GRAPH = os.path.join(os.path.join(ROOT, 'graph2'), 'Tooltip.html')
DATABASE = os.path.join(ROOT, 'Database')
DATABASE = os.path.join(DATABASE, 'Sample')  # TODO SET THIS PATH FROM THE DATABASE MENU
ROOT = os.getcwd()
BASE_MAP = os.path.join(os.path.join(ROOT, 'map_api'), 'baseMap.html')
GRAPHS = os.path.join(ROOT, 'graph2')
BASE_GRAPH = os.path.join(GRAPHS, 'RegressionPlot.html')
CLASSIFICATION_GRAPH = os.path.join(GRAPHS, 'ClassificationPlot.html')
COUNT_GRAPH = os.path.join(GRAPHS, 'CountPlot.html')
MARKOV_CHAIN_GRAPH = os.path.join(GRAPHS, 'MarkovChainPlot2.html')
MARKOV_CHAIN_FREQUENCY_GRAPH = os.path.join(GRAPHS, 'MarkovChainFrequencyPlot.html')
MARKOV_CHAIN_FREQUENCY_CURVES_GRAPH = os.path.join(GRAPHS, 'MarkovChainFrequencyCurvesPlot.html')
SIMULATION_GRAPH = os.path.join(GRAPHS, 'Simulation2.html')
SIMULATION_FREQUENCY_GRAPH = os.path.join(GRAPHS, 'SimulationFrequency.html')
MONTE_CARLO_GRAPH = os.path.join(GRAPHS, 'MonteCarloPlot.html')
REGRESSION_GRAPH = os.path.join(GRAPHS, 'RegressionPlot.html')
SEARCH_RESULTS_TABLE = os.path.join(GRAPHS, 'SearchResults2.html')
DATABASE = os.path.join(ROOT, 'Database')
TEMP_SAVE = os.path.join(DATABASE, 'temp.txt')
VARIABLES = os.path.join(os.path.join(ROOT, 'Utilities14'), 'Variables.py')
DOCUMENTATION = os.path.join(os.path.join(ROOT, '1'), '1 sunburst.html')
UTILITIES = os.path.join(ROOT, 'Utilities14')
TRANSITION_MATRIX_PLOT = os.path.join(GRAPHS, 'TransitionMatrix.html')
ROOT = os.getcwd()
BASE_MAP = os.path.join(os.path.join(ROOT, 'map_api'), 'baseMap.html')
GRAPHS = os.path.join(ROOT, 'graph2')
SIMULATION_GRAPHS = os.path.join(GRAPHS, 'Simulation2')
REGRESSION_GRAPHS = os.path.join(GRAPHS, 'Regression6')
CLASSIFICATION_GRAPHS = os.path.join(GRAPHS, 'Classification')
CLUSTERING_GRAPHS = os.path.join(GRAPHS, 'Clustering')
DIMENSIONALITY_GRAPHS = os.path.join(GRAPHS, 'DimensionalityReduction')
BASE_GRAPH = os.path.join(GRAPHS, 'RegressionPlot.html')
CLASSIFICATION_GRAPH = os.path.join(CLASSIFICATION_GRAPHS, 'ClassificationPlot.html')
CLUSTERING_GRAPH = os.path.join(CLUSTERING_GRAPHS, 'ClusteringPlot.html')
DIMENSIONALITY_GRAPH = os.path.join(DIMENSIONALITY_GRAPHS, 'DimensionalityReductionPlot.html')
COUNT_GRAPH = os.path.join(GRAPHS, 'CountPlot.html')
MARKOV_CHAIN_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MarkovChainPlot2.html')
MARKOV_CHAIN_FREQUENCY_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MarkovChainFrequencyPlot.html')
MARKOV_CHAIN_FREQUENCY_CURVES_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MarkovChainFrequencyCurvesPlot.html')
SIMULATION_GRAPH = os.path.join(SIMULATION_GRAPHS, 'Simulation2.html')
SIMULATION_DATA_GRAPH = os.path.join(SIMULATION_GRAPHS, 'SimulationData.html')
SIMULATION_FREQUENCY_GRAPH = os.path.join(SIMULATION_GRAPHS, 'SimulationFrequency.html')
MONTE_CARLO_GRAPH = os.path.join(SIMULATION_GRAPHS, 'MonteCarloPlot.html')
REGRESSION_GRAPH = os.path.join(REGRESSION_GRAPHS, 'RegressionPlot.html')
SEARCH_RESULTS_TABLE = os.path.join(GRAPHS, 'SearchResults2.html')
DATABASE = os.path.join(ROOT, 'Database')
TEMP_SAVE = os.path.join(DATABASE, 'temp.txt')
VARIABLES = os.path.join(os.path.join(ROOT, 'Utilities14'), 'Variables.py')
DOCUMENTATION = os.path.join(os.path.join(ROOT, '1'), '1 sunburst.html')
UTILITIES = os.path.join(ROOT, 'Utilities14')
TRANSITION_MATRIX_PLOT = os.path.join(SIMULATION_GRAPHS, 'TransitionMatrix.html')
ROOT = os.getcwd()
BASE_MAP = os.path.join(os.path.join(ROOT, 'map_api'), 'baseMap.html')
GRAPHS = os.path.join(ROOT, 'graph2')
BASE_GRAPH = os.path.join(GRAPHS, 'RegressionPlot.html')
COUNT_GRAPH = os.path.join(GRAPHS, 'CountPlot.html')
MARKOV_CHAIN_GRAPH = os.path.join(GRAPHS, 'MarkovChainPlot2.html')
MONTE_CARLO_GRAPH = os.path.join(GRAPHS, 'MonteCarloPlot.html')
REGRESSOR_GRAPH = os.path.join(GRAPHS, 'RegressorPlot.html')
SEARCH_RESULTS_TABLE = os.path.join(GRAPHS, 'SearchResults2.html')
DATABASE = os.path.join(ROOT, 'Database')
temp_RSP = os.path.join(
    os.path.join(os.path.join(ROOT, 'Utilities14'), '../Utilities0/Preferences'),
    'Report_Preferences0.txt'
)
TEMP_SAVE = os.path.join(DATABASE, 'temp.txt')
VARIABLES = os.path.join(os.path.join(ROOT, 'Utilities14'), '../Utilities4/Variables3.py')
DOCUMENTATION = os.path.join(os.path.join(ROOT, '1'), '1 sunburst.html')
ROOT = os.getcwd()
BASE_MAP = os.path.join(os.path.join(ROOT, 'map_api'), 'baseMap.html')
GRAPHS = os.path.join(ROOT, 'graph2')
BASE_GRAPH = os.path.join(GRAPHS, 'RegressionPlot.html')
CLASSIFICATION_GRAPH = os.path.join(GRAPHS, 'ClassificationPlot.html')
COUNT_GRAPH = os.path.join(GRAPHS, 'CountPlot.html')
MARKOV_CHAIN_GRAPH = os.path.join(GRAPHS, 'MarkovChainPlot2.html')
MARKOV_CHAIN_FREQUENCY_GRAPH = os.path.join(GRAPHS, 'MarkovChainFrequencyPlot.html')
MARKOV_CHAIN_FREQUENCY_CURVES_GRAPH = os.path.join(GRAPHS, 'MarkovChainFrequencyCurvesPlot.html')
SIMULATION_GRAPH = os.path.join(GRAPHS, 'Simulation2.html')
SIMULATION_FREQUENCY_GRAPH = os.path.join(GRAPHS, 'SimulationFrequency.html')
MONTE_CARLO_GRAPH = os.path.join(GRAPHS, 'MonteCarloPlot.html')
REGRESSION_GRAPH = os.path.join(GRAPHS, 'RegressionPlot.html')
SEARCH_RESULTS_TABLE = os.path.join(GRAPHS, 'SearchResults2.html')
DATABASE = os.path.join(ROOT, 'Database')
TEMP_SAVE = os.path.join(DATABASE, 'temp.txt')
VARIABLES = os.path.join(os.path.join(ROOT, 'Utilities14'), 'Variables.py')
DOCUMENTATION = os.path.join(os.path.join(ROOT, '1'), '1 sunburst.html')
UTILITIES = os.path.join(ROOT, 'Utilities14')
TRANSITION_MATRIX_PLOT = os.path.join(GRAPHS, 'TransitionMatrix.html')
# PLOT DIRECTORIES AND FILE NAMES ---------------------------------------
p = str(Path(__file__).parents[1])
PLOTS_DIR = os.path.join(p, 'StaticPlots0')
EXPRESSIONS_DIR = os.path.join(p, 'ExpressionStrings1')
BAR_PLOTS_DIR = os.path.join(PLOTS_DIR, 'Bar')
BOX_PLOTS_DIR = os.path.join(PLOTS_DIR, 'Box')
DISTRIBUTION_PLOTS_DIR = os.path.join(PLOTS_DIR, 'Distribution')
DISTRIBUTION_FIT_PLOTS_DIR = os.path.join(PLOTS_DIR, 'DistributionFit')
HEATMAP_PLOTS_DIR = os.path.join(PLOTS_DIR, 'Heatmap')
SCATTER_PLOTS_DIR = os.path.join(PLOTS_DIR, 'Scatter')
TABLE_PLOTS_DIR = os.path.join(PLOTS_DIR, 'Table')
CT_TABLES_DIR = os.path.join(PLOTS_DIR, 'CentralTendencies')
DISPERSION_TABLES_DIR = os.path.join(PLOTS_DIR, 'Dispersion')
IQR_TABLES_DIR = os.path.join(PLOTS_DIR, 'IQR')
VIOLIN_PLOTS_DIR = os.path.join(PLOTS_DIR, 'Violin')
STAT_COMPARISON_PLOTS_DIR = os.path.join(PLOTS_DIR, 'StatisticComparison')
PLOT_DIRS = [BAR_PLOTS_DIR, BOX_PLOTS_DIR, DISTRIBUTION_PLOTS_DIR, DISTRIBUTION_FIT_PLOTS_DIR, HEATMAP_PLOTS_DIR,
             SCATTER_PLOTS_DIR, TABLE_PLOTS_DIR, VIOLIN_PLOTS_DIR, STAT_COMPARISON_PLOTS_DIR]
BAR_PLOTS_SUFFIX = ''
BOX_PLOTS_SUFFIX = ''
DISTRIBUTION_PLOTS_SUFFIX = ''
DISTRIBUTION_FIT_PLOTS_SUFFIX = ''
SCATTER_PLOTS_SUFFIX = ''
TABLE_PLOTS_SUFFIX = ''
VIOLIN_PLOTS_SUFFIX = ''
DYNAMIC_PLOTS = os.path.join(os.path.join(os.path.abspath('../../Plot0/'), 'plot'), 'Dynamic')
# PLOT DIRECTORIES AND FILE NAMES ---------------------------------------
p = str(Path(__file__).parents[1])
PLOTS_DIR = os.path.join(p, os.path.join('plot'))
EXPRESSIONS_DIR = os.path.join(p, 'ExpressionStrings')
BAR_PLOTS_DIR = os.path.join(PLOTS_DIR, 'BarPlots')
BOX_PLOTS_DIR = os.path.join(PLOTS_DIR, 'BoxPlots')
DISTRIBUTION_PLOTS_DIR = os.path.join(PLOTS_DIR, 'DistributionPlots')
DISTRIBUTION_FIT_PLOTS_DIR = os.path.join(PLOTS_DIR, 'DistributionFitPlots')
HEATMAP_PLOTS_DIR = os.path.join(PLOTS_DIR, 'HeatmapPlots')
SCATTER_PLOTS_DIR = os.path.join(PLOTS_DIR, 'ScatterPlots')
TABLE_PLOTS_DIR = os.path.join(PLOTS_DIR, 'TablePlots')
CT_TABLES_DIR = os.path.join(PLOTS_DIR, 'CentralTendencies')
DISPERSION_TABLES_DIR = os.path.join(PLOTS_DIR, 'Dispersion')
IQR_TABLES_DIR = os.path.join(PLOTS_DIR, 'IQR')
VIOLIN_PLOTS_DIR = os.path.join(PLOTS_DIR, 'ViolinPlots')
STAT_COMPARISON_DIR = os.path.join(PLOTS_DIR, 'StatisticComparison')
PLOT_DIRS = [BAR_PLOTS_DIR, BOX_PLOTS_DIR, DISTRIBUTION_PLOTS_DIR, DISTRIBUTION_FIT_PLOTS_DIR, HEATMAP_PLOTS_DIR,
             SCATTER_PLOTS_DIR, TABLE_PLOTS_DIR, VIOLIN_PLOTS_DIR, STAT_COMPARISON_DIR]
BAR_PLOT_SUFFIX = '_bar'
BOX_PLOT_SUFFIX = '_box'
DISTRIBUTION_PLOT_SUFFIX = '_distribution'
DISTRIBUTION_FIT_PLOT_SUFFIX = '_distributionFit'
SCATTER_PLOT_SUFFIX = '_scatter'
TABLE_PLOT_SUFFIX = '_table'
VIOLIN_PLOT_SUFFIX = '_violin'
DIRE = os.path.join(p, os.path.join('FunctionStrings'))
# EXTENSIONS
CSV_EXT = '.csv'
HDF_EXT = '.h5'
KML_EXT = '.kml'
PDF_EXT = '.pdf'
QRY_EXT = '.qry'
TXT_EXT = '.txt'
XLS_EXT = '.xls'
XLSX_EXT = '.xlsx'
# EXTENSION TUPLES
ALL_FILES = 'All files (*.*)'
CSV_FILES = 'csv files (*.csv);;'
HDF_FILES = 'hdf files (*.h5);;'
KML_FILES = 'kml files (*.kml);;'
PDF_FILES = 'pdf files (*.pdf);;'
QRY_FILES = 'Query files (*.qry);;'
TXT_FILES = 'text files (*.txt);;'
XLS_FILES = 'xls files (*.xls);;'
XLSX_FILES = 'xlsx files (*.xlsx);;'
