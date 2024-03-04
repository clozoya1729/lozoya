# import queue
# import time
# import warnings
# from collections import OrderedDict
# from concurrent.futures.thread import ThreadPoolExecutor
#
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import scipy
# import scipy.stats as st
# import scipy.stats as stats
# import seaborn as sns
# from numba import jit
#
# import lozoya.math
# import lozoya.time
# import lozoya.configuration


# import sympy as sym


# import csv
# import math
# import os
# import threading
# import random
# import re
# from functools import wraps
# from scipy import signal
# from sympy.abc import e
# import random
# import sympy

'''
###########################################################################################################
def generate_new_transition_matrix(size):
    """
    This object is a randomly generated transition matrix.
    This can be used to add noise.
    """
    new_transition_matrix = {}
    for k in range(size):
        new_transition_matrix[k] = []
        p = random.random()
        new_transition_matrix[k].append(p)
        total = p
        for w in range(1, size):
            if total >= 1:
                new_transition_matrix[k].append(0)
            else:
                if w == size - 1:
                    p = 1 - total
                else:
                    p = random.uniform(0, 1 - total)
                new_transition_matrix[k].append(p)
                total += (p)
    for k in range(size):
        sum = 0
        for i in range(size):
            sum += new_transition_matrix[k][i]
        if sum > 1:
            correction = (sum - 1) / size
            for w in range(size):
                new_transition_matrix[k][w] += correction
    return new_transition_matrix


# PRINTER ---------------------------------------
def printer(f, n):
    sym.preview(f, viewer='file', filename=os.path.join(expressionsDir, n + '.png'))


def make_params(args, symbols):
    """
    args: list
    symbols: list
    """
    params = []
    for i in range(len(args)):
        params.append(lozoya.math.round_array(args[i]) if args[i] != None else sym.Symbol(symbols[i]))
    return params


def func_string(func):
    @wraps(func)
    def create_string(*args, **kwargs):
        params = func_string_rearg(args)
        kwparams = func_string_rekwarg(kwargs)
        with sym.evaluate(kwparams['evaluate']):
            f = func(*params, **kwparams)
        if kwparams['latex']:
            f = sym.latex(f, mode='equation*', mul_symbol='dot')
        else:
            f = sym.latex(f)
        return f

    return create_string


def func_string_rearg(args):
    params = [util.make_symbol(args, i) for i in range(len(args))]
    return params


def func_string_rekwarg(kwp):
    d = dict(kwp)
    if 'x' not in d:
        d['x'] = sym.Symbol('x')
    else:
        d['x'] = sym.Symbol(str(d['x']))
    if 'y' not in d:
        d['y'] = sym.Symbol('y')
    else:
        d['y'] = sym.Symbol(str(d['y']))
    if 'evaluate' not in d:
        d['evaluate'] = False
    if 'latex' not in d:
        d['latex'] = True
    return d


# REGRESSION FIT -----------------------------------
def exponential_string(*args, x='x', y='y'):
    """
    *args: list, contains: a: number, b: number
    x: str
    y: str
    """
    a = tryer(args, 0)
    b = tryer(args, 1)

    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    if a == None:
        a = sym.Symbol('a')
    if b == None:
        b = sym.Symbol('b')

    f = sym.Eq(dep, a * sym.exp(b * ind))
    return sym.latex(f)


def logarithmic_string(*args, x='x', y='y'):
    """
    *args: list, contains the following: a: number,  b: number
    x: str
    y: str
    """
    a = tryer(args, 0)
    b = tryer(args, 1)

    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if a == None:
        a = sym.Symbol('a')
    if b == None:
        b = sym.Symbol('b')

    f = sym.Eq(dep, a * sym.log(ind) + b)
    return sym.latex(f)


def polynomial_string(*args, x='x', y='y'):
    """
    coefficients: list
    x: str
    y: str
    """
    maxc = 7
    d = maxc - len(args)
    if len(args) < maxc:
        coeffs = [0 if i < d else args[i - d] for i in range(maxc)]
    else:
        coeffs = args

    ind = sym.Symbol(x)
    dep = sym.Symbol(y)

    f = sym.Eq(
        dep, coeffs[0] * ind ** 6 + coeffs[1] * ind ** 5 + coeffs[2] * ind ** 4 + coeffs[3] * ind ** 3 + coeffs[
            4] * ind ** 2 + coeffs[5] * ind + coeffs[6]
    )
    return sym.latex(f)


def sin_string(*args, x='x', y='y'):
    """
    *args: list, contains the following items: a: number, b: number, c: number, d: number
    x: str
    y: str
    """
    a = tryer(args, 0)
    b = tryer(args, 1)
    c = tryer(args, 2)
    d = tryer(args, 3)

    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if a == None:
        a = sym.Symbol('a')
    if b == None:
        b = sym.Symbol('b')
    if c == None:
        c = sym.Symbol('c')
    if d == None:
        d = sym.Symbol('d')

    f = sym.Eq(dep, a * sym.sin(lozoya.math.round_array(b * ind + c)) + d)
    return sym.latex(lozoya.math.round_array(f))



@func_string
def exponential_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: a: number, b: number
    """
    a = args[0]
    b = args[1]
    return sym.Eq(y, a * sym.exp(b * x))


@func_string
def logarithmic_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains the following: a: number,  b: number
    """
    a = args[0]
    b = args[1]
    return sym.Eq(y, a * sym.log(x) + b)


@func_string
def sin_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, a: number, b: number, c: number, d: number
    """
    a = args[0]
    b = args[1]
    c = args[2]
    d = args[3]
    return sym.Eq(y, a * sym.sin(util.round_array(b * x + c, 2)) + d)


def function_string(*args, fam=None, y='y', name=False, evaluate=False, latex=True):
    if fam == 'Exponential':
        return exponential_string(*args, y=y, evaluate=evaluate, latex=latex)
    if fam == 'Logarithmic':
        return logarithmic_string(*args, y=y, evaluate=evaluate, latex=latex)
    if fam == 'Polynomial' or fam == 'Linear':
        return polynomial_string(*args, y=y, evaluate=evaluate, latex=latex)
    if fam == 'Sinusoidal':
        return sin_string(*args, y=y, evaluate=evaluate, latex=latex)
    else:
        return None


def exponential_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains: a: number, b: number
    x: str
    y: str
    """
    n = 'Exponential'
    a = tryer(args, 0)
    b = tryer(args, 1)

    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    if a == None:
        a = sym.Symbol('a')
    if b == None:
        b = sym.Symbol('b')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, a * sym.exp(b * ind))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def logarithmic_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains the following: a: number,  b: number
    x: str
    y: str
    """
    n = 'Logarithmic'
    a = tryer(args, 0)
    b = tryer(args, 1)

    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if a == None:
        a = sym.Symbol('a')
    if b == None:
        b = sym.Symbol('b')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, a * sym.log(ind) + b)
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def polynomial_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    coefficients: list
    x: str
    y: str
    """
    n = 'Polynomial' if len(*args) > 2 else 'Linear'
    maxc = 7
    d = maxc - len(args)
    if len(args) < maxc:
        coeffs = [0 if i < d else args[i - d] for i in range(maxc)]
    else:
        coeffs = args

    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    with sym.evaluate(evaluate):
        f = sym.Eq(
            dep,
            coeffs[0] * ind ** 6 + coeffs[1] * ind ** 5 + coeffs[2] * ind ** 4 + coeffs[3] * ind ** 3 + coeffs[
                4] * ind ** 2 + coeffs[5] * ind + coeffs[6]
        )
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def sin_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains the following items: a: number, b: number, c: number, d: number
    x: str
    y: str
    """
    n = 'Sinusoidal'
    a = tryer(args, 0)
    b = tryer(args, 1)
    c = tryer(args, 2)
    d = tryer(args, 3)

    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if a == None:
        a = sym.Symbol('a')
    if b == None:
        b = sym.Symbol('b')
    if c == None:
        c = sym.Symbol('c')
    if d == None:
        d = sym.Symbol('d')

    with sym.evaluate(False):
        f = sym.Eq(dep, a * sym.sin(lozoya.math.round_array(b * ind + c, 2)) + d)
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


@func_string
def exponential_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: a: number, b: number
    x: str
    y: str
    """
    a = args[0]
    b = args[1]
    return sym.Eq(y, a * sym.exp(b * x))


@func_string
def logarithmic_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains the following: a: number,  b: number
    x: str
    y: str
    """
    a = args[0]
    b = args[1]
    return sym.Eq(y, a * sym.log(x) + b)


@func_string
def polynomial_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    coefficients: list
    x: str
    y: str
    """
    maxDegree = 7
    d = maxDegree - len(args)
    if len(args) < maxDegree:
        coeffs = [0 if i < d else args[i - d] for i in range(maxDegree)]
    else:
        coeffs = args

    return sym.Eq(
        y, coeffs[0] * x ** 6 + coeffs[1] * x ** 5 + coeffs[2] * x ** 4 + coeffs[3] * x ** 3 + coeffs[
            4] * x ** 2 + coeffs[5] * x + coeffs[6]
    )


@func_string
def sin_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains the following items: a: number, b: number, c: number, d: number
    x: str
    y: str
    """
    a = args[0]
    b = args[1]
    c = args[2]
    d = args[3]
    return sym.Eq(y, a * sym.sin(util.round_array(b * x + c, 2)) + d)


# DISTRIBUTION FIT ---------------------------------
# AUXILIARY FUNCITONS
@func_string
def _normal_cdf_string(*args, x='x', y='y', evaluate=False, latex=True):
    """phi function"""
    t = args[0]
    limits = (t, -sym.oo, x)
    integral = sym.Integral(e ** -((t ** 2) / 2), limits)
    return sym.Eq(y, (1 / sym.sqrt(2 * lozoya.symbol.pi__)) * integral)


@func_string
def _gamma_string(*args, x='x', y='y', evaluate=False, latex=True):
    """phi function"""
    z = args[0]
    limits = (x, 0, sym.oo)
    integral = sym.Integral((x ** (z - 1)) * (e ** -(x)), limits)
    return sym.Eq(y, integral)


@func_string
def _beta_string(*args, x='x', y='y', evaluate=False, latex=True):
    """phi function"""
    t = args[0]
    y_ = args[1]
    limits = (t, 0, 1)
    integral = sym.Integral((t ** (x - 1)) * ((1 - t) ** (y_ - 1)), limits)
    return sym.Eq(y, integral)


def distribution_string(names, params, errors, x='x', y='y', evaluate=False, latex=True):
    """
    name: str
    params: (parameters, location, scale)
    x: str
    y: str
    return: str or None
    """

    ns = {names[i]: StatisticsVars.DISTRIBUTION_NAMES[names[i]] for i in names}
    ps = {names[i]: make_params(params[i][0], StatisticsVars.DISTRIBUTION_PARAMS[names[i]]) for i in names}
    fs = {names[i]: globals()[names[i] + '_string'](*ps[names[i]], x=x, y=y, latex=latex) for i in names}
    es = {names[i]: errors[i] for i in names}
    return fs, ns, es


@func_string
def alpha_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Alpha
    args:x, a
    x:str
    y:str"""
    a = args[0]
    phi = sym.Symbol(str(lozoya.symbol.PHI__) + '(a)')
    return sym.Eq(y, (1 / ((x ** 2) * phi * (sym.sqrt(2 * sym.pi)))) * (e ** (-(1 / 2) * (a - 1 / x) ** 2)))


@func_string
def anglit_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Anglit
    args: list, unused
    x: str
    y: str
    """
    return sym.Eq(y, sym.cos(2 * x))


@func_string
def arcsine_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Arcsin
    args: list, unused
    x: str
    y: str
    """
    return sym.Eq(y, 1 / (lozoya.symbol.pi__ * sym.sqrt(x * (1 - x))))


@func_string
def beta_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Beta
    args:x, a, b
    x:str
    y:str"""
    a = args[0]
    b = args[1]
    gammaa = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(a))
    gammab = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(b))
    gammaab = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(a + b))
    return sym.Eq(y, (gammaab * ((x) ** (a - 1)) * ((1 - x) ** (b - 1))) / (gammaa * gammab))


@func_string
def betaprime_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Beta Prime
    args:x, a, b
    x:str
    y:str"""
    a = args[0]
    b = args[1]
    betaab = sym.Symbol(str(beta__) + '(a,b)')
    return sym.Eq(y, ((x ** (a - 1)) * ((1 + x) ** (-a - b))) / (betaab))


@func_string
def bradford_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Bradford
    args: list, contains c: number
    x: str
    y: str
    """
    c = args[0]
    return sym.Eq(y, c / ((sym.log(1 + c)) * (1 + c * x)))


@func_string
def burr_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Burr Type III
    args: list, contains: c: number, d: number
    x: str
    y: str
    """
    c = args[0]
    d = args[1]
    return sym.Eq(y, c * d * x ** (-c - 1) * (1 + x ** - c) ** (-d - 1))


@func_string
def burr12_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Burr Type XII
    args: list, contains: c: number, d: number
    x: str
    y: str
    """
    c = args[0]
    d = args[1]
    return sym.Eq(y, c * d * x ** (c - 1) * (1 + x ** c) ** (-d - 1))


@func_string
def cauchy_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Cauchy
    args: list, unused
    x: str
    y: str
    """
    return sym.Eq(y, 1 / (lozoya.symbol.pi__ * (1 + x ** 2)))


@func_string
def chi_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Chi
    args:x, df
    x:str
    y:str"""
    df = args[0]
    gammadf = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(df / 2))
    return sym.Eq(y, ((x ** (df - 1)) * (e ** -((x ** 2) / 2)) / (((2 ** df)) * gammadf)))


@func_string
def chi2_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Chi-Squared
    args:x, df
    x:str
    y:str
    """
    df = args[0]
    gammad = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(df / 2))
    return sym.Eq(y, 1 / (2 * gammad) * ((x / 2) ** (df)) * (e ** (-(x / 2))))


@func_string
def cosine_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Cosine
    args: list, unused
    x: str
    y: str
    """
    return sym.Eq(y, (1 / (2 * lozoya.symbol.pi__)) * (1 + sym.cos(x)))


"""@func_string
def crystalball_string(*args, x='x', y='y', evaluate=False, latex=True):
    """"""
    Double Gamma
    args: x, A, B
    x:str
    y:str
    """"""
    A = args[0]
    B = args[1]
    beta = sym.Symbol(beta__)
    return sym.Eq(Piecewise((1, x < 0), (2, True)))"""


@func_string
def dgamma_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Double Gamma
    args: x, a
    x:str
    y:str
    """
    a = args[0]
    gammaa = sym.Symbol(str(lozoya.symbol.lozoya.symbol.gamma__) + '({})'.format(a))
    return sym.Eq(y, (1 / (2 * gammaa)) * (abs(x) ** (a - 1)) * (e ** - abs(x)))


@func_string
def dweibull_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Double Weibull
    args: list, contains: c: number
    x: str
    y: str
    """
    c = args[0]
    return sym.Eq(y, (c / 2) * (sym.Abs(x) ** (c - 1) * sym.exp(-sym.Abs(x) ** c)))


def erlang_string(*args, x='x', y='y', evaluate=False, latex=True):
    pass


@func_string
def expon_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Exponential
    args: list, unused
    x: str
    y: str
    """
    return sym.Eq(y, sym.exp(-x))


@func_string
def exponnorm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Exponential Modified Normal
    args:x, K
    x: str
    y: str
    """
    K = args[0]
    return sym.Eq(
        y, util.round_array(1 / (2 * K), 4) * util.round_array(e ** (1 / (2 * (K ** 2))), 4) * (
            e ** -(x / K)) * sym.erfc(-util.round_array((x - 1 / K) / sym.sqrt(2), 4))
    )


@func_string
def exponweib_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Exponentiated Weibull
    args: list, contains: a: number, c: number
    x: str
    y: str
    """
    a = args[0]
    c = args[1]
    return sym.Eq(y, a * c * (1 - sym.exp(-x ** c)) ** (a - 1) * (sym.exp(-x ** c)) * (x ** (c - 1)))


@func_string
def exponpow_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Exponential Power
    args: list, contains: b: number
    x: str
    y: str
    """
    b = args[0]
    return sym.Eq(y, (b * x ** (b - 1)) * sym.exp(1 + (x ** b) - sym.exp(x ** b)))


# fix the (/1)
@func_string
def f_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    F Function
    args:x, df_1, df_2
    x: str
    y: str
    """
    df1 = sym.Symbol('df_1')
    df2 = sym.Symbol('df_2')
    Beta = sym.Symbol(str(lozoya.symbol.BETA__) + '(' + str(df1) + '/1,' + str(df2) + '/1)')
    return sym.Eq(
        y, ((df2 ** (df2 / 2)) * (df1 ** (df1 / 2)) * (x ** df1)) / (
            ((df2 + (df1 * x)) ** ((df1 + df2) / 2)) * Beta)
    )


@func_string
def fatiguelife_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Fatigue-Life (Birnbaum-Saunders)
    args: list, contains: c: number
    x: str
    y: str
    """
    c = args[0]
    return sym.Eq(
        y, (x + 1) / (util.round_array(
            (2 * c) * (sym.sqrt(2 * lozoya.symbol.pi__ * x ** 3))
        ) * util.round_array(
            sym.exp(util.round_array(-((x - 1) ** 2) / (2 * x * c ** 2)))
        ))
    )


@func_string
def fisk_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Fisk
    AKA Log-Logistic (same as Burr with d=1)
    args: list, contains c: number
    x: str
    y: str
    """
    c = args[0]
    return sym.Eq(y, c * x ** (-c - 1) * (1 + x ** -c) ** (-2))


@func_string
def foldcauchy_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Folded Cauchy
    args: list, contains: c: number
    x: str
    y: str
    """
    c = args[0]
    return sym.Eq(
        y,
        ((1 / (lozoya.symbol.lozoya.symbol.pi__ * (1 + (x + c) ** 2))) + (
            1 / (lozoya.symbol.lozoya.symbol.pi__ * (1 + (x + c) ** 2))))
    )


@func_string
def foldnorm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Folded Normal
    args: list, contains: c: number
    x: str
    y: str
    """
    c = args[0]
    return sym.Eq(y, (sym.sqrt(2 / lozoya.symbol.pi__)) * (sym.cosh(c * x)) * (sym.exp(-(x ** 2 + c ** 2) / 2)))


@func_string
def gamma_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
        gamma
        args: list, contains: a: number
        x: str
        y: str
        """
    a = args[0]
    Gamma = sym.Symbol(str(lozoya.symbol.GAMMA__) + '({})'.format(a))
    return sym.Eq(y, ((x ** (a - 1)) * (e ** (-x))) / Gamma)


@func_string
def genlogistic_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Generalized Logistic
    args: list, contains: c: number
    x: str
    y: str
    If c == 1, this function equals the Logistic function or Sech-Squared.
    """
    c = args[0]
    return sym.Eq(y, c * util.round_array(sym.exp(-x)) / util.round_array((1 + sym.exp(-x)) ** (c + 1)))


@func_string
def genpareto_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Generalized Pareto
    If c == 0, this functions equals the Exponential function
    If c == -1, this function equals Uniform [0, 1]
    args: list, contains: c: number
    x: str
    y: str
    """
    c = args[0]
    return sym.Eq(y, (1 + c * x) ** (-1 - (1 / c)))


@func_string
def genexpon_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Generalized Exponential
    args: list, contains: a: number, b:number, c:number
    x: str
    y: str
    """
    a = args[0]
    b = args[1]
    c = args[2]
    return sym.Eq(y, (a + b * (1 - e ** (-c * x))) * e ** ((-a * x - b * x) + ((b / c) * (1 - e ** (-c * x)))))


def genextreme_string():
    pass


@func_string
def gengamma_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Generalized Gamma
    args: list, contains: a: number, c:number
    x: str
    y: str
    """
    a = args[0]
    c = args[1]
    gamma = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(a))
    return sym.Eq(y, (abs(c) * (x ** (c * a - 1)) * (e ** (-x ** c))) / gamma)


@func_string
def genhalflogistic_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
        Generalized Half-Logistic
        args: list, contains: a: number, c:number
        x: str
        y: str
        """
    c = args[0]
    return sym.Eq(y, (2 * (1 - c * x) ** (1 / (c - 1))) / ((1 + (1 - c * x) ** (1 / c)) ** 2))


def gennormal_string():
    pass


@func_string
def gilbrat_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
        Gilbrat
        args: list, unused
        x: str
        y: str
        """
    return sym.Eq(y, (1 / (x * sym.sqrt(2 * lozoya.symbol.pi__))) * (e ** (- (1 / 2) * (sym.log(x) ** 2))))


@func_string
def gompertz_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
        Gompertz
        args: list, c:number
        x: str
        y: str
        """
    c = args[0]
    return sym.Eq(y, c * e ** (x) * e ** (- c * (e ** x - 1)))


@func_string
def gumbel_r_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
        Right-Skewed Gumbel
        args: list, c:number
        x: str
        y: str
        """
    return sym.Eq(y, e ** (- (x + e ** (- x))))


@func_string
def gumbel_l_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
        Leftt-Skewed Gumbel
        args: list, c:number
        x: str
        y: str
        """
    return sym.Eq(y, e ** (x - e ** (x)))


@func_string
def halfcauchy_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, unused
    x: str
    y: str
    """
    return sym.Eq(y, 2 / (sym.pi * 1 + x ** 2))


@func_string
def halflogistic_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, unused
    x: str
    y: str
    """
    return sym.Eq(y, (1 / 2) * (sym.sech(x / 2) ** 2))


@func_string
def halfnorm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, unused
    x: str
    y: str
    halfnorm is a special case of :math`chi` with df == 1.
    """
    return sym.Eq(y, sym.sqrt(2 / sym.pi * e ** (-x ** 2 / 2)))


@func_string
def hypsecant_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, unused
    x: str
    y: str
    """
    return sym.Eq(y, (1 / lozoya.symbol.pi__) * sym.sech(x))


def gausshypogeometric_string():
    pass


@func_string
def invgamma_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Inverted Gamma
    args: list, unused
    x: str
    y: str
    """
    a = args[0]
    gammaa = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(a))
    return sym.Eq(y, ((x ** (-a - 1)) / gammaa) * e ** (-1 / x))


@func_string
def invgauss_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: mu: number
    x: str
    y: str
    If mu == 1, this function equals the Wald function
    """
    mu = args[0]
    return sym.Eq(y, (1 / sym.sqrt(2 * sym.pi * x ** 3)) * e ** (-((x - mu) ** 2) / 2 * x * mu ** 2))


@func_string
def invweibull_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: c: number
    x: str
    y: str
    """
    c = args[0]
    return sym.Eq(y, c * x ** (-c - 1) * e ** (-x ** (-c)))


@func_string
def johnsonsb_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: a: number, b: number
    x: str
    y: str
    """
    a = args[0]
    b = args[1]
    phi = sym.Function(str(lozoya.symbol.phi_))
    return sym.Eq(y, b / (x * (1 - x)) * phi(a + b * sym.log(x / (1 - x))))


def johnsonsu_string():
    pass


@func_string
def kappa4_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Kappa 3
    args: list, contains: h: number, k: number
    x: str
    y: str
    """
    h = args[0]
    k = args[1]
    return sym.Eq(y, ((1 - k * x) ** (1 / (k - 1))) * ((1 - h * ((1 - k * x) ** (1 / k))) ** (1 / (h - 1))))


def ksone_string():
    pass


def kstwo_string():
    pass


@func_string
def laplace_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, unused
    x: str
    y: str
    """
    return sym.Eq(y, (1 / 2) * (e ** (-sym.Abs(x))))


@func_string
def levy_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, unused
    x: str
    y: str
    This is the same as levy_stable distribution with a=1/1 and b=1
    """
    return sym.Eq(y, (1 / x * sym.sqrt(2 * lozoya.symbol.pi__ * x)) * (e ** (-1 / 2 * x)))


@func_string
def levy_l_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, unused
    x: str
    y: str
    This is the same as levy_stable distribution with a=1/1 and b=-1
    """
    return sym.Eq(y, (1 / (sym.Abs(x) * sym.sqrt(2 * sym.pi * sym.Abs(x)))) * (e ** (-1 / 2 * sym.Abs(x))))


@func_string
def logistic_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, unused
    x: str
    y: str
    logistic is a special case of genlogistic with c == 1.
    """
    return sym.Eq(y, (e ** (-x)) / ((1 + e ** (-x)) ** 2))


def logdoubleexpon_string():
    pass


@func_string
def loggamma_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Log Gamma
    args: list, c:number
    x: str
    y: str
    logistic is a special case of genlogistic with c == 1.
    """
    c = args[0]
    gammac = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(c))
    return sym.Eq(y, (e ** (c * x - e ** (x))) / (gammac))


@func_string
def lognorm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Log Normal
    args: list, s:number
    x: str
    y: str
    logistic is a special case of genlogistic with c == 1.
    """
    s = args[0]
    return sym.Eq(y, (1 / (s * x * sym.sqrt(2 * sym.pi))) * (e ** (-(1 / 2) * ((sym.log(x) / s) ** 2))))


# pareto second kind
@func_string
def lomax_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: c: number
    x: str
    y: str    """
    c = args[0]
    return sym.Eq(y, (c / (1 + x) ** (c + 1)))


@func_string
def maxwell_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, unused
    x: str
    y: str
    This is a special case of a chi distribution, with df = 2, loc = 0.0, and given scale = a
    """
    return sym.Eq(y, sym.sqrt(2 / sym.pi) * (x ** 2) * e ** (-(x ** 2) / 2))


@func_string
def mielke_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: k: number, s:number
    x: str
    y: str
    """
    k = args[0]
    s = args[1]
    return sym.Eq(y, (k * (x ** k - 1)) / (1 + (x ** s) ** (1 + k / s)))


@func_string
def moyal_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, unused
    x: str
    y: str
    """
    return sym.Eq(y, (e ** (-(x + e ** (-x))) / 2) / sym.sqrt(2 * sym.pi))


@func_string
def nakagami_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Nakagami
    args: list, nu:number
    x: str
    y: str
    """
    nu = sym.Symbol(str(lozoya.symbol.nu__))
    Gammanu = sym.Symbol(str(lozoya.symbol.GAMMA__) + '(' + str(nu) + ')')
    return sym.Eq(y, ((2 * nu ** nu) / (Gammanu)) * (x ** (2 * nu - 1)) * (e ** (-nu * x ** 2)))


def ncx2_string():
    pass


def ncf_string():
    pass


@func_string
def nct_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Non-Central Student's T
    args: list, df:number, nc:number
    x: str
    y: str
    """
    df = args[0]
    nc = args[1]
    gammadf1 = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(df + 1))
    gammadf2 = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(df / 2))
    return sym.Eq(
        y, (df ** (df / 2) * gammadf1) / (
            (2 ** df) * (e ** ((nc ** 2) / 2)) * ((df + x ** 2) ** (df / 2)) * gammadf2)
    )


# TODO
@func_string
def norm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, unused
    x: str
    y: str
    """
    return sym.Eq(y, (e ** (-(x ** 2) / 2)) / sym.sqrt(2 * sym.pi))


@func_string
def pareto_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: b: number
    x: str
    y: str
    """
    b = args[0]
    return sym.Eq(y, b / (x ** (b + 1)))


@func_string
def pearson3_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Pearson Type III
    args: list, contains: skew: number
    x: str
    y: str
    powerlaw is a special case of beta with b == 1.
    """
    skew = sym.Symbol('skew')
    betax = sym.Symbol(str(lozoya.symbol.beta__) + '(x -' + str(lozoya.symbol.zeta__) + ')')
    gammaa = sym.Symbol(str(lozoya.symbol.lozoya.symbol.gamma__) + '(' + str(lozoya.symbol.alpha__) + ')')
    return sym.Eq(y, (abs(lozoya.symbol.beta__) / gammaa) * (betax ** (lozoya.symbol.alpha__ - 1)) * (e ** (-betax)))


@func_string
def powerlaw_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: a: number
    x: str
    y: str
    powerlaw is a special case of beta with b == 1.
    """
    a = args[0]
    return sym.Eq(y, a * x ** (a - 1))


@func_string
def powerlognorm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Power Log-Normal
    args: list, contains: c: number, s:number
    x: str
    y: str
    """
    c = args[0]
    s = args[1]
    phi = lozoya.symbol.phi_(sym.log(x) / s)
    Phi = lozoya.symbol.PHI__(-sym.log(x) / s) ** (c - 1)
    return sym.Eq(y, (c / x * s) * (phi) * Phi)


@func_string
def powernorm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Power Normal
    args: list, contains: c: number
    x: str
    y: str
    """
    c = args[0]
    phi = sym.Symbol(str(lozoya.symbol.phi_) + '(x)')
    Phi = sym.Symbol(str(lozoya.symbol.PHI__) + '(-x)')
    return sym.Eq(y, c * phi * Phi ** (c - 1))


@func_string
def rdist_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    R-Distributed
    args: list, contains: c: number
    x: str
    y: str
    """
    c = args[0]
    Beta = sym.Symbol(str(lozoya.symbol.BETA__) + '(1/1,c/1)')
    return sym.Eq(y, ((1 - x ** 2) ** (c)) / (Beta))


@func_string
def rayleigh_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: r: number
    x: str
    y: str
    rayleigh is a special case of chi with df == 1.
    """
    print(args)
    r = args[0]
    return sym.Eq(y, r * e ** (-(r ** 2) / 2))


@func_string
def reciprocal_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Reciprocal
    args: list, contains: a: number, b:number
    x: str
    y: str
    """
    a = args[0]
    b = args[1]
    return sym.Eq(y, 1 / (x * sym.log(b / a)))


@func_string
def recipinvgauss_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: mu: number
    x: str
    y: str
    """
    mu = args[0]
    return sym.Eq(y, (1 / sym.sqrt(2 * lozoya.symbol.pi__ * x)) * ((e ** (-(1 - mu * x) ** 2)) / (2 * x * (mu ** 2))))


@func_string
def semicircular_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, unused
    x: str
    y: str
    """
    return sym.Eq(y, (2 / sym.pi) * sym.sqrt(1 - (x ** 2)))


@func_string
def t_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Student's T
    args: list, df:number
    x: str
    y: str
    """
    df = args[0]
    gammadf1 = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format((df + 1) / 2))
    gammadf = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(df / 2))
    return sym.Eq(y, gammadf1 / (sym.sqrt(sym.pi * df) * gammadf * (1 + (x ** 2) / df) ** ((df + 1) / 2)))


def triangular_string():
    pass


@func_string
def truncexpon_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: b: number
    x: str
    y: str
    """
    b = args[0]
    return sym.Eq(y, (e ** (-x)) / (1 - e ** (-b)))


def truncnorm_string():
    pass


def tukeylambda_string():
    pass


def uniform_string():
    pass


def vonmises_string():
    pass


@func_string
def wald_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, unused
    x: str
    y: str
    This is a special case of Invgauss with mu == 1.
    """
    return sym.Eq(y, (1 / sym.sqrt(2 * sym.pi * x ** 3)) * (e ** (((x - 1) ** 2) / (2 * x))))


@func_string
def weibull_min_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: c: number
    x: str
    y: str
    """
    c = args[0]
    return sym.Eq(y, c * (x ** (c - 1)) * e ** (-x ** c))


@func_string
def weibull_max_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: c: number
    x: str
    y: str
    """
    c = args[0]
    return sym.Eq(y, c * ((-x) ** (c - 1)) * e ** (-(-x) ** c))


@func_string
def wrapcauchy_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: c: number
    x: str
    y: str
    """
    c = args[0]
    return sym.Eq(y, (1 - (c ** 2)) / (2 * sym.pi * (1 + (c ** 2) - 2 * c * sym.cos(x))))


# STATISTICS ---------------------------------------
def iqr_string():
    iqr = sym.Symbol('IQR')
    q2 = sym.Symbol('25^{th} Percentile')
    q3 = sym.Symbol('75^{th} Percentile')
    f = sym.Eq(iqr, q3 + q2)
    f = sym.latex(f, mode='equation*', mul_symbol='dot')
    return f


def lowerbound_string():
    iqr = sym.Symbol('IQR')
    q1 = sym.Symbol('25^{th} Percentile')
    lb = sym.Symbol('Minimum')
    f = sym.Eq(lb, q1 - iqr)
    f = sym.latex(f, mode='equation*', mul_symbol='dot')
    return f


def mae_string():
    mae = sym.Symbol('MAE')
    d = sym.Symbol('y_i-\hat y_i')
    i = sym.Symbol('i')
    n = sym.Symbol('n')
    with sym.evaluate(False):
        f = sym.Eq(mae, (1 / n) * sym.Sum(sym.Abs(d), (i, 0, n)))
    return sym.latex(f, mode='equation*', mul_symbol='dot')


def mse_string():
    mse = sym.Symbol('MSE')
    d = sym.Symbol('(y_i-\hat y_i)')
    i = sym.Symbol('i')
    n = sym.Symbol('n')
    with sym.evaluate(False):
        f = sym.Eq(mse, (1 / n) * sym.Sum((d) ** 2, (i, 0, n)))
    return sym.latex(f, mode='equation*', mul_symbol='dot')


def rmse_string():
    rmse = sym.Symbol('RMSE')
    d = sym.Symbol('(y_i-\hat y_i)')
    i = sym.Symbol('i')
    n = sym.Symbol('n')
    with sym.evaluate(False):
        f = sym.Eq(rmse, sym.sqrt((1 / n) * sym.Sum((d) ** 2, (i, 0, n))))
    return sym.latex(f, mode='equation*', mul_symbol='dot')


def upperbound_string():
    iqr = sym.Symbol('IQR')
    q3 = sym.Symbol('75^{th} Percentile')
    ub = sym.Symbol('Maximum')
    f = sym.Eq(ub, q3 + iqr)
    f = sym.latex(f, mode='equation*', mul_symbol='dot')


@func_string
def iqr_string(*args, x='x', y='IQR', evaluate=False, latex=True):
    q2 = sym.Symbol('25^{th} Percentile')
    q3 = sym.Symbol('75^{th} Percentile')
    return sym.Eq(y, q3 + q2)


@func_string
def kurtosis_string(*args, x='x', y='Kurtosis', evaluate=False, latex=True):
    n = sym.Symbol('n')
    g = sym.Symbol('sigma')
    d = sym.Symbol(r'(x-\mu)')
    i = sym.Symbol('i')
    return sym.Eq(y, ((1 / g ** 4) * sym.Sum((d) ** 4 / n - 3, (i, 1, n))))


@func_string
def lowerbound_string(*args, x='x', y='Minimum', evaluate=False, latex=True):
    iqr = sym.Symbol('IQR')
    q1 = sym.Symbol('25^{th} Percentile')
    return sym.Eq(y, q1 - iqr)


@func_string
def mae_string(*args, x='x', y='MAE', evaluate=False, latex=True):
    d = sym.Symbol('y_i-\hat y_i')
    i = sym.Symbol('i')
    n = sym.Symbol('n')
    return sym.Eq(y, (1 / n) * sym.Sum(sym.Abs(d), (i, 0, n)))


@func_string
def mbe_string(*args, x='x', y='MBE', evaluate=False, latex=True):
    d = sym.Symbol('(y_i-\hat y_i)')
    i = sym.Symbol('i')
    n = sym.Symbol('n')
    return sym.Eq(y, (1 / n) * sym.Sum((d), (i, 0, n)))


@func_string
def mean_string(*args, x='x', y='Mean', evaluate=False, latex=True):
    n = sym.Symbol('n')
    i = sym.Symbol('i')
    xi = sym.Symbol('x_i')
    return sym.Eq(y, (1 / n) * sym.Sum(xi, (i, 0, n)))


@func_string
def median_string(*args, x='x', y='Median', evaluate=False, latex=True):
    c = sym.Symbol('n = odd')
    n = sym.Symbol('n')
    expr = sym.Piecewise(
        (sym.exp(n + 1 / (n)), c), (sym.exp(n / 2 + (n / 2 + 1)) / (2), sym.Not(c)),
        (sym.exp(n / 2 + (n / 2 + 1)) / (2), True)
    )
    return sym.Eq(y, expr)


@func_string
def mse_string(*args, x='x', y='MSE', evaluate=False, latex=True):
    d = sym.Symbol('(y_i-\hat y_i)')
    i = sym.Symbol('i')
    n = sym.Symbol('n')
    return sym.Eq(y, (1 / n) * sym.Sum((d) ** 2, (i, 0, n)))


@func_string
def rmse_string(*args, x='x', y='RMSE', evaluate=False, latex=True):
    d = sym.Symbol('(y_i-\hat y_i)')
    i = sym.Symbol('i')
    n = sym.Symbol('n')
    return sym.Eq(y, sym.sqrt((1 / n) * sym.Sum((d) ** 2, (i, 0, n))))


@func_string
def skew_string(*args, x='x', y='Skew', evaluate=True, latex=True):
    n = sym.Symbol('n')
    g = sym.Symbol('sigma')
    d = sym.Symbol(r'(x-\mu)')
    i = sym.Symbol('i')
    return sym.Eq(y, ((1 / g ** 3) * sym.Sum((d) ** 3 / n, (i, 1, n))))


@func_string
def upperbound_string(*args, x='x', y='Maximum', evaluate=False, latex=True):
    iqr = sym.Symbol('IQR')
    q3 = sym.Symbol('75^{th} Percentile')
    return sym.Eq(y, q3 + iqr)


def function_string(func):
    @wraps(func)
    def create_string(*args, **kwargs):
        params = function_string_rearg(*args)
        d = function_string_rekwarg(**kwargs)
        with sym.evaluate(d['evaluate']):
            f, n = func(*params, **d)
        f = sym.latex(f, mode='equation', mul_symbol='dot')
        if d['name']:
            return f, n
        else:
            return f

    return create_string


def function_string_rearg(*args):
    params = [sym.Symbol(str(args[i])) for i in range(len(args))]
    return params


def function_string_rekwarg(**kwargs):
    d = dict(kwargs)
    if 'x' not in d:
        d['x'] = sym.Symbol('x')
    else:
        d['x'] = sym.Symbol(str(d['x']))
    if 'y' not in d:
        d['y'] = sym.Symbol('y')
    else:
        d['y'] = sym.Symbol(str(d['y']))
    if 'evaluate' not in d:
        d['evaluate'] = False
    if 'name' not in d:
        d['name'] = False
    return d


@function_string
def sine_string(*args, x='x', y='y', name=False, evaluate=False):
    n = 'Sine'
    f = sym.Eq(y, sym.sin(x) ** args[0] + sym.cos(x) ** args[1])
    return f, n


# DISTRIBUTION FIT ---------------------------------
def distribution_string(names, params, errors, x='x', y='y', evaluate=False, latex=True):
    """
    name: str
    params: (parameters, location, scale)
    x: str
    y: str
    return: str or None
    """
    ns = {names[i]: DISTRIBUTION_NAMES[names[i]] for i in names}
    ps = {names[i]: make_params(params[i][0], DISTRIBUTION_PARAMS[names[i]]) for i in names}
    fs = {names[i]: globals()[names[i] + '_string'](*ps[names[i]], x=x, y=y, latex=latex) for i in names}
    es = {names[i]: errors[i] for i in names}
    return fs, ns, es


def distribution_string(names, params, x='x', y='y', evaluate=False, latex=True):
    """
    name: str
    params: list
    x: str
    y: str
    return: str or None
    """
    ns = {}
    ps = {}
    fs = {}
    for i in names:
        try:
            ns[names[i]] = DISTRIBUTION_NAMES[names[i]]
            ps[names[i]] = make_params(params[i], DISTRIBUTION_PARAMS[names[i]])
            fs[names[i]] = globals()[names[i] + '_string'](*ps[names[i]], x=x, y=y, latex=latex)
        except Exception as e:
            print(e)
    return fs, ns


@func_string
def dgamma_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Double Gamma
    args: x, a
    x:str
    y:str
    """
    a = args[0]
    gammaa = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(a))
    return sym.Eq(y, (1 / (2 * gammaa)) * (abs(x) ** (a - 1)) * (e ** - abs(x)))


@func_string
def exponnorm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Exponential Modified Normal
    args:x, K
    x: str
    y: str
    """
    K = args[0]
    return sym.Eq(
        y, lozoya.math.round_array(1 / (2 * K), 4) * lozoya.math.round_array(e ** (1 / (2 * (K ** 2))), 4) * (
            e ** -(x / K)) * sym.erfc(
            -lozoya.math.round_array((x - 1 / K) / sym.sqrt(2), 4)
        )
    )


@func_string
def pearson3_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Pearson Type III
    args: list, contains: skew: number
    x: str
    y: str
    powerlaw is a special case of beta with b == 1.
    """
    skew = sym.Symbol('skew')
    betax = sym.Symbol(str(beta__) + '(x -' + str(zeta__) + ')')
    gammaa = sym.Symbol(str(lozoya.symbol.gamma__) + '(' + str(alpha__) + ')')
    return sym.Eq(y, (abs(beta__) / gammaa) * (betax ** (alpha__ - 1)) * (e ** (-betax)))


def normal_cdf_string():
    """phi function"""


@func_string
def fatiguelife_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Fatigue-Life (Birnbaum-Saunders)
    args: list, contains: c: number
    x: str
    y: str
    """
    c = args[0]
    return sym.Eq(
        y, (x + 1) / (lozoya.math.round_array(
            (2 * c) * (sym.sqrt(2 * lozoya.symbol.pi__ * x ** 3))
        ) * lozoya.math.round_array(
            sym.exp(lozoya.math.round_array(-((x - 1) ** 2) / (2 * x * c ** 2)))
        ))
    )


@func_string
def foldcauchy_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    Folded Cauchy
    args: list, contains: c: number
    x: str
    y: str
    """
    c = args[0]
    return sym.Eq(
        y, ((1 / (lozoya.symbol.pi__ * (1 + (x + c) ** 2))) + (1 / (lozoya.symbol.pi__ * (1 + (x + c) ** 2))))
    )


@func_string
def johnsonsb_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: a: number, b: number, c: number, d: number
    x: str
    y: str
    """
    a = args[0]
    b = args[1]
    c = args[2]
    d = args[3]
    return sym.Eq(y, a * sym.sin(b * x + c) + d)


# TODO
@func_string
def norm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: k: number, s:number
    x: str
    y: str
    """
    k = args[0]
    s = args[1]
    return sym.Eq(y, (e ** (-(x ** 2) / 2)) / sym.sqrt(2 * sym.pi))


@func_string
def rayleigh_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains: r: number
    x: str
    y: str
    rayleigh is a special case of chi with df == 1.
    """
    r = args[0]
    return sym.Eq(y, r * e ** (-(r ** 2) / 2))


def normal_cdf_string():
    """phi function"""


def distribution_string(names, params, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    name: str
    params: list
    x: str
    y: str
    return: str or None
    """
    strs = {}
    ns = {}
    for i in names:
        try:
            strs[names[i]], ns[names[i]] = globals()[names[i] + '_string'](*params[i], x=x, y=y, name=name, latex=latex)
        except Exception as e:
            print(e)
    return strs, ns


def anglit_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Anglit
    *args: list, unused
    x: str
    y: str
    """
    n = 'Anglit'
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)

    with sym.evaluate(evaluate):
        f = sym.Eq(dep, sym.cos(2 * ind))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def arcsin_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Arcsin
    *args: list, unused
    x: str
    y: str
    """
    n = 'Arcsine'
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)

    with sym.evaluate(evaluate):
        f = sym.Eq(dep, 1 / (lozoya.symbol.pi__ * sym.sqrt(ind * (1 - ind))))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def bradford_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Bradford
    *args: list, unused
    x: str
    y: str
    """
    n = ''
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')

    with sym.evaluate(evaluate):
        f = sym.Eq(dep, c / ((sym.log(1 + c)) * (1 + c * ind)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def burr_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Burr Type III
    *args: list, contains: c: number, d: number
    x: str
    y: str
    """
    n = 'Burr Type III'
    c = tryer(args, 0)
    d = tryer(args, 1)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    if d == None:
        d = sym.Symbol('d')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, c * d * ind ** (-c - 1) * (1 + ind ** -c) ** (-d - 1))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


# TODO redundant
def burr12_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Burr Type XII
    *args: list, contains: c: number, d: number
    x: str
    y: str
    """
    n = 'Burr Type XII'
    c = tryer(args, 0)
    d = tryer(args, 1)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    if d == None:
        d = sym.Symbol('d')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, c * d * ind ** (c - 1) * (1 + ind ** c) ** (-d - 1))
    if name:
        f = sym.latex(f, mode='equation*', mul_symbol='dot'), "Burr XII"
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def cauchy_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Cauchy
    *args: list, unused
    x: str
    y: str
    """
    n = 'Cauchy'
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, 1 / (lozoya.symbol.pi__ * (1 + ind ** 2)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def cosine_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Cosine
    *args: list, unused
    x: str
    y: str
    """
    n = 'Cosine'
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (1 / (2 * lozoya.symbol.pi__)) * (1 + sym.cos(ind)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def dweibull_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Double Weibull
    *args: list, contains: c: number
    x: str
    y: str
    """
    n = 'Double Weibull'
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (c / 2) * (sym.Abs(ind) ** (c - 1) * sym.exp(-sym.Abs(ind) ** c)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def expon_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Exponential
    *args: list, unused
    x: str
    y: str
    """
    n = 'Exponential'
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, sym.exp(-ind))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def exponweib_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Exponentiated Weibull
    *args: list, contains: a: number, c: number
    x: str
    y: str
    """
    n = 'Exponentiated Weibull'
    a = tryer(args, 0)
    c = tryer(args, 1)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if a == None:
        a = sym.Symbol('a')
    if c == None:
        c = sym.Symbol('c')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, a * c * (1 - sym.exp(-ind ** c)) ** (a - 1) * (sym.exp(-ind ** c)) * (ind ** (c - 1)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def exponpow_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Exponential Power
    *args: list, contains: b: number
    x: str
    y: str
    """
    n = 'Exponential Power'
    b = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if b == None:
        b = sym.Symbol('b')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (b * ind ** (b - 1)) * sym.exp(1 + (ind ** b) - sym.exp(ind ** b)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def fatiguelife_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Fatigue-Life (Birnbaum-Saunders)
    *args: list, contains: c: number
    x: str
    y: str
    """
    n = 'Fatigue-Life (Birnbaum-Saunders)'
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    with sym.evaluate(evaluate):
        f = sym.Eq(
            dep,
            (ind + 1) / (lozoya.math.round_array(
                (2 * c) * (sym.sqrt(2 * lozoya.symbol.pi__ * ind ** 3))
            ) * lozoya.math.round_array(
                sym.exp(lozoya.math.round_array(-((ind - 1) ** 2) / (2 * ind * c ** 2)))
            ))
        )
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


# TODO redundant

def fisk_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Fisk
    AKA Log-Logistic (same as Burr with d=1), redundant
    *args: list, contains c: number
    x: str
    y: str
    """
    n = 'Fisk'
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, c * ind ** (-c - 1) * (1 + ind ** -c) ** (-2))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def foldcauchy_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Folded Cauchy
    *args: list, contains: c: number
    x: str
    y: str
    """
    n = 'Folded Cauchy'
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    with sym.evaluate(evaluate):
        f = sym.Eq(
            dep, ((1 / (lozoya.symbol.pi__ * (1 + (ind + c) ** 2))) + (1 / (lozoya.symbol.pi__ * (1 + (ind + c) ** 2))))
        )
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def foldnorm_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Folded Normal
    *args: list, contains: c: number
    x: str
    y: str
    """
    n = 'Folded Normal'
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (sym.sqrt(2 / lozoya.symbol.pi__)) * (sym.cosh(c * ind)) * (sym.exp(-(ind ** 2 + c ** 2) / 2)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def genlogistic_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Generalized Logistic
    *args: list, contains: c: number
    x: str
    y: str
    If c == 1, this function equals the Logistic function.
    """
    n = 'Generalized Logistic'
    c = tryer(args, 0)
    if c == 1:
        n = 'Logistic (or Sech-squared)'
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    with sym.evaluate(evaluate):
        f = sym.Eq(
            dep, c * lozoya.math.round_array(sym.exp(-ind)) / lozoya.math.round_array((1 + sym.exp(-ind)) ** (c + 1))
        )
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


# MAKES EXPONENTIAL REDUNDANT with c=0
def genpareto_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    Generalized Pareto
    If c == 0, this functions equals the Exponential function
    If c == -1, this function equals Uniform [0, 1]
    *args: list, contains: c: number
    x: str
    y: str
    """
    n = 'Generalized Pareto'
    c = tryer(args, 0)
    if c == 0:
        n = 'Exponential'
    elif c == -1:
        n = 'Uniform [0, 1]'
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (1 + c * ind) ** (-1 - (1 / c)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def halfcauchy_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, unused
    x: str
    y: str
    """
    n = 'Half-Cauchy'
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, 2 / (sym.pi * 1 + ind ** 2))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def halflogistic_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, unused
    x: str
    y: str
    """
    n = 'Half-Logistic'
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (1 / 2) * (sym.sech(ind / 2) ** 2))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def halfnorm_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, unused
    x: str
    y: str
    halfnorm is a special case of :math`chi` with df == 1.
    """
    n = 'Half-Normal'
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, sym.sqrt(2 / sym.pi * e ** (-ind ** 2 / 2)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)

    if name:
        return f, n
    return f


def hypsecant_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, unused
    x: str
    y: str
    """
    n = 'Hyperbolic Secant'
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (1 / sym.pi) * sym.sech(ind))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def invgauss_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains: mu: number
    x: str
    y: str
    If mu == 1, this function equals the Wald function
    """
    n = 'Inverse Gaussian'
    mu = tryer(args, 0)
    if mu == 1:
        n = 'Wald'
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    mu = mu if mu != None else sym.Symbol('mu')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (1 / sym.sqrt(2 * sym.pi * ind ** 3)) * e ** (-((ind - mu) ** 2) / 2 * ind * mu ** 2))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def invweibull_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains: c: number
    x: str
    y: str
    """
    n = 'Inverted Weibull'
    c = tryer(args, 2)
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    c = c if c != None else sym.Symbol('c')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, c * ind ** (-c - 1) * e ** (-ind ** (-c)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def johnsonsb_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains: a: number, b: number, c: number, d: number
    x: str
    y: str
    """
    n = 'Johnson SB'
    a = tryer(args, 0)
    b = tryer(args, 1)
    c = tryer(args, 2)
    d = tryer(args, 3)
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    a = a if a != None else sym.Symbol('a')
    b = b if b != None else sym.Symbol('b')
    c = c if c != None else sym.Symbol('c')
    d = d if d != None else sym.Symbol('d')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, a * sym.sin(b * ind + c) + d)
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def laplace_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, unused
    x: str
    y: str
    """
    n = 'Laplace'
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (1 / 2) * (e ** (-sym.Abs(ind))))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def levy_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, unused
    x: str
    y: str
    This is the same as levy_stable distribution with a=1/1 and b=1
    """
    n = 'Levy'
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (1 / ind * sym.sqrt(2 * sym.pi * ind)) * (e ** (-1 / 2 * ind)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def levy_l_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, unused
    x: str
    y: str
    This is the same as levy_stable distribution with a=1/1 and b=-1
    """
    n = 'Left-Skewed Levy'
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (1 / (sym.Abs(ind) * sym.sqrt(2 * sym.pi * sym.Abs(ind)))) * (e ** (-1 / 2 * sym.Abs(ind))))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def logistic_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, unused
    x: str
    y: str
    logistic is a special case of genlogistic with c == 1.
    """
    n = 'Logistic (or Sech-squared)'
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (e ** (-ind)) / ((1 + e ** (-ind)) ** 2))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def lomax_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains: c: number
    x: str
    y: str    """
    n = 'Lomax (Pareto of the Second Kind)'
    c = tryer(args, 0)
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    c = c if c != None else sym.Symbol('c')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (c / (1 + ind) ** (c + 1)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def maxwell_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, unused
    x: str
    y: str
    This is a special case of a chi distribution, with df = 2, loc = 0.0, and given scale = a
    """
    n = 'Maxwell'
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, sym.sqrt(2 / sym.pi) * (ind ** 2) * e ** (-(ind ** 2) / 2))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def mielke_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains: k: number, s:number
    x: str
    y: str
    """
    n = "Mielke's Beta-Kappa"
    k = tryer(args, 0)
    s = tryer(args, 0)
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    k = k if k != None else sym.Symbol('k')
    s = s if s != None else sym.Symbol('s')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (k * (ind ** k - 1)) / (1 + (ind ** s) ** (1 + k / s)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def moyal_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, unused
    x: str
    y: str
    """
    n = 'Moyal'
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (e ** (-(ind + e ** (-ind))) / 2) / sym.sqrt(2 * sym.pi))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def norm_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains: k: number, s:number
    x: str
    y: str
    """
    n = 'Normal'
    k = tryer(args, 0)
    s = tryer(args, 0)
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    k = k if k != None else sym.Symbol('k')
    s = s if s != None else sym.Symbol('s')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (e ** (-(ind ** 2) / 2)) / sym.sqrt(2 * sym.pi))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def pareto_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains: b: number
    x: str
    y: str
    """
    n = 'Pareto'
    b = tryer(args, 0)
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    b = b if b != None else sym.Symbol('b')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, b / (ind ** (b + 1)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def powerlaw_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains: a: number
    x: str
    y: str
    powerlaw is a special case of beta with b == 1.
    """
    n = 'Power-Function'
    a = tryer(args, 0)
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    a = a if a != None else sym.Symbol('a')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, a * ind ** (a - 1))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def rayleigh_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains: r: number
    x: str
    y: str
    rayleigh is a special case of chi with df == 1.
    """
    n = 'Rayleigh'
    r = tryer(args, 0)
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    r = r if r != None else sym.Symbol('r')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, r * e ** (-(r ** 2) / 2))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def recipinvgauss_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains: mu: number
    x: str
    y: str
    """
    n = 'Reciprocal Inverse Gaussian'
    mu = tryer(args, 0)
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    mu = mu if mu != None else sym.Symbol('mu')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (1 / sym.sqrt(2 * sym.pi * ind)) * ((e ** (-(1 - mu * ind) ** 2)) / (2 * ind * (mu ** 2))))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def semicircular_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, unused
    x: str
    y: str
    """
    n = 'Semicircular'
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (2 / sym.pi) * sym.sqrt(1 - (ind ** 2)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def truncexpon_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains: b: number
    x: str
    y: str
    """
    n = 'Truncated Exponential'
    b = tryer(args, 0)
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    b = b if b != None else sym.Symbol('b')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (e ** (-ind)) / (1 - e ** (-b)))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def wald_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, unused
    x: str
    y: str
    This is a special case of Invgauss with mu == 1.
    """
    n = 'Wald'
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (1 / sym.sqrt(2 * sym.pi * ind ** 3)) * (e ** (((ind - 1) ** 2) / (2 * ind))))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def weibull_min_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains: c: number
    x: str
    y: str
    """
    n = 'Weibull Minimum'
    c = tryer(args, 0)
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    c = c if c != None else sym.Symbol('c')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, c * (ind ** (c - 1)) * e ** (-ind ** c))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def weibull_max_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains: c: number
    x: str
    y: str
    """
    n = 'Weibull Maximum'
    c = tryer(args, 0)
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    c = c if c != None else sym.Symbol('c')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, c * ((-ind) ** (c - 1)) * e ** (-(-ind) ** c))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def wrapcauchy_string(*args, x='x', y='y', name=False, evaluate=False, latex=True):
    """
    *args: list, contains: c: number
    x: str
    y: str
    """
    n = 'Wrapped Cauchy'
    c = tryer(args, 0)
    dep = sym.Symbol(y)
    ind = sym.Symbol(x)
    c = c if c != None else sym.Symbol('c')
    with sym.evaluate(evaluate):
        f = sym.Eq(dep, (1 - (c ** 2)) / (2 * sym.pi * (1 + (c ** 2) - 2 * c * sym.cos(ind))))
    if latex:
        f = sym.latex(f, mode='equation*', mul_symbol='dot')
    else:
        f = sym.latex(f)
    if name:
        return f, n
    return f


def make_params(args, symbols):
    """
    args: list
    symbols: list
    """
    params = []
    for i in range(len(args)):
        params.append(args[i] if args[i] != None else sym.Symbol(symbols[i]))
    return params


"""
def make_params(**kwargs):
    def make_p(func):
        @wraps(func)
        def make(*args):
            params = []
            for i in args:
                params[i] = args[i] if args[i] != None else sym.Symbol(kwargs[i])
            return params
        return make
    return make_p"""


def func_string(func):
    @wraps(func)
    def create_string(*args, **kwp):
        params = func_string_rearg(args)
        d = func_string_rekwarg(kwp)
        with sym.evaluate(d['evaluate']):
            f = func(*params, **d)
        if d['latex']:
            f = sym.latex(f, mode='equation*', mul_symbol='dot')
        else:
            f = sym.latex(f)
        return f

    return create_string


def func_string_rearg(args):
    params = [make_symbol(args, i) for i in range(len(args))]
    return params


@func_string
def sin_string(*args, x='x', y='y', evaluate=False, latex=True):
    """
    args: list, contains the following items: a: number, b: number, c: number, d: number
    x: str
    y: str
    """
    a = args[0]
    b = args[1]
    c = args[2]
    d = args[3]
    return sym.Eq(y, a * sym.sin(lozoya.math.round_array(b * x + c, 2)) + d)


# AUXILIARY
def make_params(args, symbols):
    """
    args: list
    symbols: list
    """
    params = []
    for i in range(len(args)):
        params.append(lozoya.math.round_array(args[i]) if args[i] != None else sym.Symbol(symbols[i]))
    return params


# STATISTICS ---------------------------------------
@func_string
def median_string2(*args, x='x', y='Median', evaluate=False, latex=True):
    n = sym.Symbol('n')
    t = sym.Symbol('th')
    return sym.Eq(y, (n + 1) ** t / (n))


@func_string
def median_even_string2(*args, x='x', y='Median', evaluate=True, latex=True):
    n = sym.Symbol('n')
    t = sym.Symbol('th')
    return sym.Eq(y, (n / 2) ** t + (n / 2 + 1) ** t) / (2)


# DISTRIBUTION FIT ---------------------------------
# TODO missing johnsonsu
"""@func_string
def crystalball_string(*args, x='x', y='y', evaluate=False, latex=True):
    """"""
    Double Gamma
    args: x, A, B
    x:str
    y:str
    """"""
    A = args[0]
    B = args[1]
    beta = sym.Symbol(beta__)
    return sym.Eq(Piecewise((1, x < 0), (2, True)))"""


def normal_cdf_string():
    """phi function"""
    pass


# TODO wrap entire module in a function
def get_function_string(*args, fam=None, y='y'):
    if fam == 'Exponential':
        return exponential_string(*args, y=y)
    if fam == 'Logarithmic':
        return logarithmic_string(*args, y=y)
    if fam == 'Polynomial' or fam == 'Linear':
        return polynomial_string(*args, y=y)
    if fam == 'Sinusoidal':
        return sin_string(*args, y=y)
    else:
        return None


# DISTRIBUTION FIT
def anglit_string(*args, x='x', y='y'):
    """
    Anglit
    *args: list, unused
    x: str
    y: str
    """

    ind = sym.Symbol(x)
    dep = sym.Symbol(y)

    f = sym.Eq(dep, sym.cos(2 * ind))
    return sym.latex(f)


def arcsin_string(*args, x='x', y='y'):
    """
    Arcsin
    *args: list, unused
    x: str
    y: str
    """

    ind = sym.Symbol(x)
    dep = sym.Symbol(y)

    f = sym.Eq(dep, 1 / (lozoya.symbol.pi__ * sym.sqrt(ind * (1 - ind))))
    return sym.latex(f)


def bradford_string(*args, x='x', y='y'):
    """
    Bradford
    *args: list, contains: c: number
    x: str
    y: str
    """
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')

    f = sym.Eq(dep, c / ((sym.log(1 + c)) * (1 + c * ind)))
    return sym.latex(f)


def burr_string(*args, x='x', y='y'):
    """
    Burr Type III
    *args: list, contains: c: number, d: number
    x: str
    y: str
    """
    c = tryer(args, 0)
    d = tryer(args, 1)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    if d == None:
        d = sym.Symbol('d')

    f = sym.Eq(dep, c * d * ind ** (-c - 1) * (1 + ind ** -c) ** (-d - 1))
    return sym.latex(f)


# TODO redundant
def burr12_string(*args, x='x', y='y'):
    """
    Burr Type XII
    *args: list, contains: c: number, d: number
    x: str
    y: str
    """
    c = tryer(args, 0)
    d = tryer(args, 1)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    if d == None:
        d = sym.Symbol('d')

    f = sym.Eq(dep, c * d * ind ** (c - 1) * (1 + ind ** c) ** (-d - 1))
    return sym.latex(f)


def cauchy_string(*args, x='x', y='y'):
    """
    Cauchy
    *args: list, unused
    x: str
    y: str
    """

    ind = sym.Symbol(x)
    dep = sym.Symbol(y)

    f = sym.Eq(dep, 1 / (lozoya.symbol.pi__ * (1 + ind ** 2)))
    return sym.latex(f)


def cosine_string(*args, x='x', y='y'):
    """
    Cosine
    *args: list, unused
    x: str
    y: str
    """

    ind = sym.Symbol(x)
    dep = sym.Symbol(y)

    f = sym.Eq(dep, (1 / (2 * lozoya.symbol.pi__)) * (1 + sym.cos(ind)))
    return sym.latex(f)


def dweibull_string(*args, x='x', y='y'):
    """
    Double Weibull
    *args: list, contains: c: number
    x: str
    y: str
    """
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')

    f = sym.Eq(dep, (c / 2) * (sym.Abs(ind) ** (c - 1) * sym.exp(-sym.Abs(ind) ** c)))
    return sym.latex(f)


def expon_string(*args, x='x', y='y'):
    """
    Exponential
    *args: list, unused
    x: str
    y: str
    """

    ind = sym.Symbol(x)
    dep = sym.Symbol(y)

    f = sym.Eq(dep, sym.exp(-ind))
    return sym.latex(f)


def exponweib_string(*args, x='x', y='y'):
    """
    Exponentiated Weibull
    *args: list, contains: a: number, c: number
    x: str
    y: str
    """
    a = tryer(args, 0)
    c = tryer(args, 1)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if a == None:
        a = sym.Symbol('a')
    if c == None:
        c = sym.Symbol('c')

    f = sym.Eq(dep, a * c * (1 - sym.exp(-ind ** c)) ** (a - 1) * (sym.exp(-ind ** c)) * (ind ** (c - 1)))
    return sym.latex(f)


def exponpow_string(*args, x='x', y='y'):
    """
    Exponential Power
    *args: list, contains: b: number
    x: str
    y: str
    """
    b = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if b == None:
        b = sym.Symbol('b')

    f = sym.Eq(dep, (b * ind ** (b - 1)) * sym.exp(1 + (ind ** b) - sym.exp(ind ** b)))
    return sym.latex(f)


def fatiguelife_string(*args, x='x', y='y'):
    """
    Fatigue-Life (Birnbaum-Saunders)
    *args: list, contains: c: number
    x: str
    y: str
    """
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')

    f = sym.Eq(
        dep, (ind + 1) / (
            (2 * c) * (sym.sqrt(2 * lozoya.symbol.pi__ * ind ** 3)) * (sym.exp(-((ind - 1) ** 2) / (2 * ind * c ** 2))))
    )
    return sym.latex(f)


# TODO redundant
def fisk_string(*args, x='x', y='y'):
    """
    Fisk
    AKA Log-Logistic (same as Burr with d=1), redundant
    *args: list, contains c: number
    x: str
    y: str
    """
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')

    f = sym.Eq(dep, c * ind ** (-c - 1) * (1 + ind ** -c) ** (-2))
    return sym.latex(f)


def foldcauchy_string(*args, x='x', y='y'):
    """
    Folded Cauchy
    *args: list, contains: c: number
    x: str
    y: str
    """
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    f = sym.Eq(
        dep, ((1 / (lozoya.symbol.pi__ * (1 + (ind + c) ** 2))) + (1 / (lozoya.symbol.pi__ * (1 + (ind + c) ** 2))))
    )
    return sym.latex(f)


def foldnorm_string(*args, x='x', y='y'):
    """
    Folded Normal
    *args: list, contains: c: number
    x: str
    y: str
    """
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    with sym.evaluate(False):
        f = sym.Eq(dep, (sym.sqrt(2 / lozoya.symbol.pi__)) * (sym.cosh(c * ind)) * (sym.exp(-(ind ** 2 + c ** 2) / 2)))
    return sym.latex(f)


def genlogistic_string(*args, x='x', y='y'):
    """
    Generalized Logistic
    *args: list, contains: c: number
    x: str
    y: str
    """
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    with sym.evaluate(False):
        f = sym.Eq(dep, c * (sym.exp(-ind)) / ((1 + sym.exp(-ind)) ** (c + 1)))
    return sym.latex(f)


# MAKES EXPONENTIAL REDUNDANT with c=0
def genpareto_string(*args, x='x', y='y'):
    """
    Generalized Pareto
    *args: list, contains: c: number
    x: str
    y: str
    """
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    with sym.evaluate(False):
        f = sym.Eq(dep, (1 + c * ind) ** (-1 - (1 / c)))
    return sym.latex(f)


def get_distribution_string(names, params, x='x', y='y'):
    """
    name: str
    params: list
    x: str
    y: str
    return: str or None
    """
    strs = {}
    for i in names:
        try:
            strs[names[i]] = globals()[names[i] + '_string'](*params[i], x=x, y=y)
        except Exception as e:
            print(e)
    return strs


def make_params(args, symbols):
    """
    args: list
    symbols: list
    """
    params = []
    for i in range(len(args)):
        params.append(util.round_array(args[i]) if args[i] != None else sym.Symbol(symbols[i]))
    return params


def func_string_rearg(args):
    params = [util.make_symbol(args[i]) for i in range(len(args))]
    return params


# DISTRIBUTION FIT ---------------------------------
def distribution_string(names, params, errors, x='x', y='y', evaluate=False, latex=True):
    """ name: str, params: (parameters, location, scale), return: str or None """
    ns = {names[i]: settings.DISTRIBUTION_NAMES[names[i]] for i in names}
    ps = {names[i]: make_params(params[i][0], settings.DISTRIBUTION_PARAMS[names[i]]) for i in names}
    fs = {names[i]: globals()[names[i] + '_string'](*ps[names[i]], x=x, y=y, latex=latex) for i in names}
    es = {names[i]: errors[i] for i in names}
    return fs, ns, es


@func_string
def arcsine_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Arcsin """
    return sym.Eq(y, 1 / (lozoya.symbol.pi__ * sym.sqrt(x * (1 - x))))


@func_string
def betaprime_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Beta Prime args:x, a, b """
    a = args[0]
    b = args[1]
    betaab = sym.Symbol(str(beta__) + '(a,b)')
    return sym.Eq(y, ((x ** (a - 1)) * ((1 + x) ** (-a - b))) / (betaab))


@func_string
def bradford_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Bradford args: list, contains c: number """
    c = args[0]
    return sym.Eq(y, c / ((sym.log(1 + c)) * (1 + c * x)))


@func_string
def burr_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Burr Type III args: list, contains: c: number, d: number """
    c = args[0]
    d = args[1]
    return sym.Eq(y, c * d * x ** (-c - 1) * (1 + x ** - c) ** (-d - 1))


@func_string
def burr12_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Burr Type XII args: list, contains: c: number, d: number """
    c = args[0]
    d = args[1]
    return sym.Eq(y, c * d * x ** (c - 1) * (1 + x ** c) ** (-d - 1))


@func_string
def cauchy_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Cauchy args: list, unused """
    return sym.Eq(y, 1 / (lozoya.symbol.pi__ * (1 + x ** 2)))


@func_string
def chi_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Chi args:x, df """
    df = args[0]
    gammadf = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(df / 2))
    return sym.Eq(y, ((x ** (df - 1)) * (e ** -((x ** 2) / 2)) / (((2 ** df)) * gammadf)))


@func_string
def chi2_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Chi-Squared args:x, df """
    df = args[0]
    gammad = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(df / 2))
    return sym.Eq(y, 1 / (2 * gammad) * ((x / 2) ** (df)) * (e ** (-(x / 2))))


@func_string
def cosine_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Cosine args: list, unused """
    return sym.Eq(y, (1 / (2 * lozoya.symbol.pi__)) * (1 + sym.cos(x)))


"""@func_string
def crystalball_string(*args, x='x', y='y', evaluate=False, latex=True):
    """"""
    Double Gamma
    args: x, A, B
    x:str
    y:str
    """"""
    A = args[0]
    B = args[1]
    beta = sym.Symbol(beta__)
    return sym.Eq(Piecewise((2, x < 0), (3, True)))"""


@func_string
def dgamma_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Double Gamma args: x, a """
    a = args[0]
    gammaa = sym.Symbol(str(util.lozoya.symbol.gamma__) + '({})'.format(a))
    return sym.Eq(y, (1 / (2 * gammaa)) * (abs(x) ** (a - 1)) * (e ** - abs(x)))


@func_string
def dweibull_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Double Weibull args: list, contains: c: number """
    c = args[0]
    return sym.Eq(y, (c / 2) * (sym.Abs(x) ** (c - 1) * sym.exp(-sym.Abs(x) ** c)))


@func_string
def exponnorm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Exponential Modified Normal args:x, K """
    K = args[0]
    return sym.Eq(
        y, util.round_array(1 / (2 * K), 4) * util.round_array(e ** (1 / (2 * (K ** 2))), 4) * (
            e ** -(x / K)) * sym.erfc(-util.round_array((x - 1 / K) / sym.sqrt(2), 4))
    )


@func_string
def exponweib_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Exponentiated Weibull args: list, contains: a: number, c: number """
    a = args[0]
    c = args[1]
    return sym.Eq(y, a * c * (1 - sym.exp(-x ** c)) ** (a - 1) * (sym.exp(-x ** c)) * (x ** (c - 1)))


# fix the (/2)
@func_string
def f_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ F Function args:x, df_1, df_2 """
    df1 = sym.Symbol('df_1')
    df2 = sym.Symbol('df_2')
    Beta = sym.Symbol(str(lozoya.symbol.BETA__) + '(' + str(df1) + '/2,' + str(df2) + '/2)')
    return sym.Eq(
        y, ((df2 ** (df2 / 2)) * (df1 ** (df1 / 2)) * (x ** df1)) / (
            ((df2 + (df1 * x)) ** ((df1 + df2) / 2)) * Beta)
    )


@func_string
def fisk_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Fisk AKA Log-Logistic (same as Burr with d=1) args: list, contains c: number """
    c = args[0]
    return sym.Eq(y, c * x ** (-c - 1) * (1 + x ** -c) ** (-2))


@func_string
def foldcauchy_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Folded Cauchy args: list, contains: c: number """
    c = args[0]
    return sym.Eq(
        y, ((1 / (util.lozoya.symbol.pi__ * (1 + (x + c) ** 2))) + (1 / (util.lozoya.symbol.pi__ * (1 + (x + c) ** 2))))
    )


@func_string
def foldnorm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Folded Normal args: list, contains: c: number """
    c = args[0]
    return sym.Eq(y, (sym.sqrt(2 / lozoya.symbol.pi__)) * (sym.cosh(c * x)) * (sym.exp(-(x ** 2 + c ** 2) / 2)))


@func_string
def gamma_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ gamma args: list, contains: a: number """
    a = args[0]
    Gamma = sym.Symbol(str(lozoya.symbol.GAMMA__) + '({})'.format(a))
    return sym.Eq(y, ((x ** (a - 1)) * (e ** (-x))) / Gamma)


@func_string
def genlogistic_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Generalized Logistic args: list, contains: c: number If c == 1, this function equals the Logistic function or Sech-Squared. """
    c = args[0]
    return sym.Eq(y, c * util.round_array(sym.exp(-x)) / util.round_array((1 + sym.exp(-x)) ** (c + 1)))


def genextreme_string():
    pass


@func_string
def genhalflogistic_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Generalized Half-Logistic args: list, contains: a: number, c:number """
    c = args[0]
    return sym.Eq(y, (2 * (1 - c * x) ** (1 / (c - 1))) / ((1 + (1 - c * x) ** (1 / c)) ** 2))


def gennormal_string():
    pass


@func_string
def gilbrat_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Gilbrat args: list, unused """
    return sym.Eq(y, (1 / (x * sym.sqrt(2 * lozoya.symbol.pi__))) * (e ** (- (1 / 2) * (sym.log(x) ** 2))))


@func_string
def gompertz_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Gompertz args: list, c:number """
    c = args[0]
    return sym.Eq(y, c * e ** (x) * e ** (- c * (e ** x - 1)))


@func_string
def gumbel_r_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Right-Skewed Gumbel args: list, c:number """
    return sym.Eq(y, e ** (- (x + e ** (- x))))


@func_string
def gumbel_l_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Leftt-Skewed Gumbel args: list, c:number """
    return sym.Eq(y, e ** (x - e ** (x)))


@func_string
def halfcauchy_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, unused """
    return sym.Eq(y, 2 / (sym.pi * 1 + x ** 2))


@func_string
def halflogistic_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, unused """
    return sym.Eq(y, (1 / 2) * (sym.sech(x / 2) ** 2))


@func_string
def halfnorm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, unused halfnorm is a special case of `chi` with df==1. """
    return sym.Eq(y, sym.sqrt(2 / sym.pi * e ** (-x ** 2 / 2)))


@func_string
def hypsecant_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, unused """
    return sym.Eq(y, (1 / lozoya.symbol.pi__) * sym.sech(x))


def gausshypogeometric_string():
    pass


@func_string
def invgamma_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Inverted Gamma args: list, unused """
    a = args[0]
    gammaa = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(a))
    return sym.Eq(y, ((x ** (-a - 1)) / gammaa) * e ** (-1 / x))


@func_string
def invgauss_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, contains: mu: number If mu == 1, this function equals the Wald function """
    mu = args[0]
    return sym.Eq(y, (1 / sym.sqrt(2 * sym.pi * x ** 3)) * e ** (-((x - mu) ** 2) / 2 * x * mu ** 2))


@func_string
def invweibull_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, contains: c: number """
    c = args[0]
    return sym.Eq(y, c * x ** (-c - 1) * e ** (-x ** (-c)))


@func_string
def johnsonsb_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, contains: a: number, b: number """
    a = args[0]
    b = args[1]
    phi = sym.Function(str(lozoya.symbol.phi_))
    return sym.Eq(y, b / (x * (1 - x)) * phi(a + b * sym.log(x / (1 - x))))


def johnsonsu_string():
    pass


@func_string
def kappa4_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Kappa 4 args: list, contains: h: number, k: number """
    h = args[0]
    k = args[1]
    return sym.Eq(y, ((1 - k * x) ** (1 / (k - 1))) * ((1 - h * ((1 - k * x) ** (1 / k))) ** (1 / (h - 1))))


def ksone_string():
    pass


def kstwo_string():
    pass


@func_string
def laplace_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, unused """
    return sym.Eq(y, (1 / 2) * (e ** (-sym.Abs(x))))


@func_string
def levy_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, unused This is the same as levy_stable distribution with a=1/2 and b=1 """
    return sym.Eq(y, (1 / x * sym.sqrt(2 * lozoya.symbol.pi__ * x)) * (e ** (-1 / 2 * x)))


@func_string
def levy_l_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, unused This is the same as levy_stable distribution with a=1/2 and b=-1 """
    return sym.Eq(y, (1 / (sym.Abs(x) * sym.sqrt(2 * sym.pi * sym.Abs(x)))) * (e ** (-1 / 2 * sym.Abs(x))))


@func_string
def logistic_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, unused logistic is a special case of genlogistic with c == 1. """
    return sym.Eq(y, (e ** (-x)) / ((1 + e ** (-x)) ** 2))


def logdoubleexpon_string():
    pass


@func_string
def loggamma_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Log Gamma args: list, c:number logistic is a special case of genlogistic with c == 1. """
    c = args[0]
    gammac = sym.Symbol(str(lozoya.symbol.gamma__) + '({})'.format(c))
    return sym.Eq(y, (e ** (c * x - e ** (x))) / (gammac))


@func_string
def lognorm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Log Normal args: list, s:number logistic is a special case of genlogistic with c == 1. """
    s = args[0]
    return sym.Eq(y, (1 / (s * x * sym.sqrt(2 * sym.pi))) * (e ** (-(1 / 2) * ((sym.log(x) / s) ** 2))))


# pareto second kind
@func_string
def lomax_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, contains: c: number """
    c = args[0]
    return sym.Eq(y, (c / (1 + x) ** (c + 1)))


@func_string
def maxwell_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, unused This is a special case of a chi distribution, with df = 3, loc = 0.0, and given scale = a """
    return sym.Eq(y, sym.sqrt(2 / sym.pi) * (x ** 2) * e ** (-(x ** 2) / 2))


@func_string
def mielke_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, contains: k: number, s:number """
    k = args[0]
    s = args[1]
    return sym.Eq(y, (k * (x ** k - 1)) / (1 + (x ** s) ** (1 + k / s)))


@func_string
def moyal_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, unused """
    return sym.Eq(y, (e ** (-(x + e ** (-x))) / 2) / sym.sqrt(2 * sym.pi))


def ncx2_string():
    pass


def ncf_string():
    pass


# TODO
@func_string
def norm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, unused """
    return sym.Eq(y, (e ** (-(x ** 2) / 2)) / sym.sqrt(2 * sym.pi))


@func_string
def pareto_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, contains: b: number """
    b = args[0]
    return sym.Eq(y, b / (x ** (b + 1)))


@func_string
def pearson3_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Pearson Type III args: list, contains: skew: number powerlaw is a special case of beta with b == 1. """
    skew = sym.Symbol('skew')
    betax = sym.Symbol(str(util.beta__) + '(x -' + str(util.zeta__) + ')')
    gammaa = sym.Symbol(str(util.lozoya.symbol.gamma__) + '(' + str(util.alpha__) + ')')
    return sym.Eq(y, (abs(util.beta__) / gammaa) * (betax ** (util.alpha__ - 1)) * (e ** (-betax)))


@func_string
def powerlaw_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, contains: a: number powerlaw is a special case of beta with b == 1. """
    a = args[0]
    return sym.Eq(y, a * x ** (a - 1))


@func_string
def powerlognorm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Power Log-Normal args: list, contains: c: number, s:number """
    c = args[0]
    s = args[1]
    phi = lozoya.symbol.phi_(sym.log(x) / s)
    Phi = lozoya.symbol.PHI__(-sym.log(x) / s) ** (c - 1)
    return sym.Eq(y, (c / x * s) * (phi) * Phi)


@func_string
def powernorm_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Power Normal args: list, contains: c: number """
    c = args[0]
    phi = sym.Symbol(str(lozoya.symbol.phi_) + '(x)')
    Phi = sym.Symbol(str(lozoya.symbol.PHI__) + '(-x)')
    return sym.Eq(y, c * phi * Phi ** (c - 1))


@func_string
def rdist_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ R-Distributed args: list, contains: c: number """
    c = args[0]
    Beta = sym.Symbol(str(lozoya.symbol.BETA__) + '(1/2,c/2)')
    return sym.Eq(y, ((1 - x ** 2) ** (c)) / (Beta))


@func_string
def rayleigh_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, contains: r: number rayleigh is a special case of chi with df == 2. """
    r = args[0]
    return sym.Eq(y, r * e ** (-(r ** 2) / 2))


@func_string
def reciprocal_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Reciprocal args: list, contains: a: number, b:number """
    a = args[0]
    b = args[1]
    return sym.Eq(y, 1 / (x * sym.log(b / a)))


@func_string
def recipinvgauss_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, contains: mu: number """
    mu = args[0]
    return sym.Eq(y, (1 / sym.sqrt(2 * lozoya.symbol.pi__ * x)) * ((e ** (-(1 - mu * x) ** 2)) / (2 * x * (mu ** 2))))


@func_string
def semicircular_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, unused """
    return sym.Eq(y, (2 / sym.pi) * sym.sqrt(1 - (x ** 2)))


def triangular_string():
    pass


@func_string
def truncexpon_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, contains: b: number """
    b = args[0]
    return sym.Eq(y, (e ** (-x)) / (1 - e ** (-b)))


def truncnorm_string():
    pass


def tukeylambda_string():
    pass


def uniform_string():
    pass


def vonmises_string():
    pass


@func_string
def wald_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, unused This is a special case of Invgauss with mu == 1. """
    return sym.Eq(y, (1 / sym.sqrt(2 * sym.pi * x ** 3)) * (e ** (((x - 1) ** 2) / (2 * x))))


@func_string
def weibull_min_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, contains: c: number """
    c = args[0]
    return sym.Eq(y, c * (x ** (c - 1)) * e ** (-x ** c))


@func_string
def weibull_max_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, contains: c: number """
    c = args[0]
    return sym.Eq(y, c * ((-x) ** (c - 1)) * e ** (-(-x) ** c))


@func_string
def wrapcauchy_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, contains: c: number """
    c = args[0]
    return sym.Eq(y, (1 - (c ** 2)) / (2 * sym.pi * (1 + (c ** 2) - 2 * c * sym.cos(x))))


# In all following functions x: str, y: str
# DISTRIBUTION FIT ---------------------------------
@func_string
@func_string
def _beta_string(*args, x='x', y='y', evaluate=False, latex=True):
    """phi function"""
    t = args[0]
    y_ = args[1]
    limits = (t, 0, 1)
    integral = sym.Integral((t ** (x - 1)) * ((1 - t) ** (y_ - 1)), limits)
    return sym.Eq(y, integral)


def distribution_string(names, params, errors, x='x', y='y', evaluate=False, latex=True):
    """ name: str, params: (parameters, location, scale), return: str or None """
    ns = {names[i]: distfit.util.util.dicts.DISTRIBUTION_NAMES[names[i]] for i in names}
    ps = {names[i]: make_params(params[i][0], distfit.util.util.dicts.DISTRIBUTION_PARAMS[names[i]]) for i in names}
    fs = {names[i]: globals()[names[i] + '_string'](*ps[names[i]], x=x, y=y, latex=latex) for i in names}
    es = {names[i]: errors[i] for i in names}
    return fs, ns, es


def distribution_string(names, params, errors, x='x', y='y', evaluate=False, latex=True):
    """ name: str, params: (parameters, location, scale), return: str or None """
    ns = {names[i]: _future.stats.distfit.util.util.dicts.DISTRIBUTION_NAMES[names[i]] for i in names}
    ps = {names[i]: make_params(params[i][0], _future.stats.distfit.util.util.dicts.DISTRIBUTION_PARAMS[names[i]]) for i
          in names}
    fs = {names[i]: globals()[names[i] + '_string'](*ps[names[i]], x=x, y=y, latex=latex) for i in names}
    es = {names[i]: errors[i] for i in names}
    return fs, ns, es


@func_string
def anglit_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Anglit args: list, unused """
    return sym.Eq(y, sym.cos(2 * x))


@func_string
def expon_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Exponential args: list, unused """
    return sym.Eq(y, sym.exp(-x))


@func_string
def fatiguelife_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ Fatigue-Life (Birnbaum-Saunders) args: list, contains: c: number """
    c = args[0]
    return sym.Eq(
        y, (x + 1) / (util.round_array((2 * c) * (sym.sqrt(2 * lozoya.symbol.pi__ * x ** 3))) * util.round_array(
            sym.exp(util.round_array(-((x - 1) ** 2) / (2 * x * c ** 2)))
        ))
    )


@func_string
def semicircular_string(*args, x='x', y='y', evaluate=False, latex=True):
    """ args: list, unused """
    return sym.Eq(y, (2 / sym.pi) * sym.sqrt(1 - (x ** 2)))


# TODO wrap entire module in a function
def sin_string(*args, x='x', y='y'):
    """
    *args: list, contains the following items: a: number, b: number, c: number, d: number
    x: str
    y: str
    """
    a = tryer(args, 0)
    b = tryer(args, 1)
    c = tryer(args, 2)
    d = tryer(args, 3)

    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if a == None:
        a = sym.Symbol('a')
    if b == None:
        b = sym.Symbol('b')
    if c == None:
        c = sym.Symbol('c')
    if d == None:
        d = sym.Symbol('d')

    with sym.evaluate(False):
        f = sym.Eq(dep, a * sym.sin(lozoya.math.round_array(b * ind + c, 2)) + d)
    return sym.latex(f)


# DISTRIBUTION FIT
def anglit_string(*args, x='x', y='y'):
    """
    Anglit
    *args: list, unused
    x: str
    y: str
    """

    ind = sym.Symbol(x)
    dep = sym.Symbol(y)

    f = sym.Eq(dep, sym.cos(2 * ind))
    return sym.latex(f)


def expon_string(*args, x='x', y='y'):
    """
    Exponential
    *args: list, unused
    x: str
    y: str
    """

    ind = sym.Symbol(x)
    dep = sym.Symbol(y)

    f = sym.Eq(dep, sym.exp(-ind))
    return sym.latex(f)


def fatiguelife_string(*args, x='x', y='y'):
    """
    Fatigue-Life (Birnbaum-Saunders)
    *args: list, contains: c: number
    x: str
    y: str
    """
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')

    f = sym.Eq(
        dep, (ind + 1) / (lozoya.math.round_array(
            (2 * c) * (sym.sqrt(2 * lozoya.symbol.pi__ * ind ** 3))
        ) * lozoya.math.round_array(
            sym.exp(lozoya.math.round_array(-((ind - 1) ** 2) / (2 * ind * c ** 2)))
        ))
    )
    return sym.latex(f)


def genlogistic_string(*args, x='x', y='y'):
    """
    Generalized Logistic
    *args: list, contains: c: number
    x: str
    y: str
    """
    c = tryer(args, 0)
    ind = sym.Symbol(x)
    dep = sym.Symbol(y)
    if c == None:
        c = sym.Symbol('c')
    with sym.evaluate(False):
        f = sym.Eq(
            dep, c * lozoya.math.round_array(sym.exp(-ind)) / lozoya.math.round_array((1 + sym.exp(-ind)) ** (c + 1))
        )
    return sym.latex(f)


def mbe_string():
    mae = sym.Symbol('MBE')
    d = sym.Symbol('(y_i-\hat y_i)')
    i = sym.Symbol('i')
    n = sym.Symbol('n')
    with sym.evaluate(False):
        f = sym.Eq(mae, (1 / n) * sym.Sum((d), (i, 0, n)))
    return sym.latex(f, mode='equation*', mul_symbol='dot')


def lowerbound_string():
    iqr = sym.Symbol('IQR')
    q1 = sym.Symbol('25^{th} Percentile')
    lb = sym.Symbol('Minimum')
    f = sym.Eq(lb, q1 - iqr)
    return sym.latex(f, mode='equation*', mul_symbol='dot')


def upperbound_string():
    iqr = sym.Symbol('IQR')
    q3 = sym.Symbol('75^{th} Percentile')
    ub = sym.Symbol('Maximum')
    f = sym.Eq(ub, q3 + iqr)
    return sym.latex(f, mode='equation*', mul_symbol='dot')


def iqr_string():
    iqr = sym.Symbol('IQR')
    q2 = sym.Symbol('25^{th} Percentile')
    q3 = sym.Symbol('75^{th} Percentile')
    f = sym.Eq(iqr, q3 + q2)
    return sym.latex(f, mode='equation*', mul_symbol='dot')


def count_vector(dataFrame):
    countVector = pd.Series(index=dh.get_uniques(dataFrame)).fillna(0)
    for j, column in enumerate(dataFrame.columns):
        for i, row in enumerate(dataFrame.index):
            a = dataFrame.iloc[i, j]
            if not pd.isnull(a):
                countVector[dataFrame.iloc[i, j]] += 1
    return countVector


def frequency(dataFrame):
    count = count_vector(dataFrame)
    frq = count.divide(count.sum())
    return frq


def min_max_mean(dataFrame, mode):
    if mode == 'series':
        axis = 0
        dF = dataFrame.T
    if mode == 'parallel':
        axis = 1
        dF = dataFrame
    dF = dF.astype(float)
    minMaxMean = pd.DataFrame(columns=('Upper Bound', 'Average', 'Lower Bound'))
    minMaxMean['Upper Bound'] = dF.max(axis=axis)
    minMaxMean['Average'] = dF.mean(axis=axis)
    minMaxMean['Lower Bound'] = dF.min(axis=axis)
    return minMaxMean


def denumerify_transition_matrix(transitionMatrix, keys):
    if not keys:
        return transitionMatrix
    return pd.DataFrame(
        transitionMatrix.values, columns=[keys[c] for c in transitionMatrix.columns],
        index=[keys[i] for i in transitionMatrix.index]
    )


def symmetrical_matrix(dataFrame):
    """
    dataFrame: pandas DataFrame
    return: pandas DataFrame
    """
    labels = dh.get_uniques(dataFrame)
    symmetricalLabelMatrix = pd.DataFrame(columns=labels, index=labels)
    symmetricalLabelMatrix.fillna(0, inplace=True)
    return symmetricalLabelMatrix


def time_steps2(dataFrame, mode):
    """
    Time steps represent the number of samples required per iteration in a simulation
    dataFrame: pandas DataFrame
    mode: str ('series' or 'parallel')
    return:
    """
    steps = dh.get_uniques(pd.DataFrame(dataFrame.index))
    if mode == 'series':
        return len(steps), steps
    elif mode == 'parallel':
        count = list(dataFrame.index).count(0)
        return count, list(range(count))


def time_steps(dataFrame, mode):
    print(mode)
    print(dataFrame)
    if mode == 'series':
        return len(dataFrame.index), dh.get_uniques(pd.DataFrame(dataFrame.index))
    elif mode == 'parallel':
        return len(dataFrame.columns), dh.get_uniques(pd.DataFrame(dataFrame.columns))


def frequency_curves(dataFrame, mode='series', keys=None):
    labels = dh.get_uniques(dataFrame)
    if mode == 'series':
        frqCurves = pd.DataFrame(columns=labels, index=dataFrame.index).fillna(0)
        for i in dataFrame.index:
            for j in dataFrame.columns:
                a = dataFrame.loc[i, j]
                if not pd.isnull(a):
                    frqCurves.loc[i, a] += 1
        for i in frqCurves.index:
            frqCurves.loc[i, :] = frqCurves.loc[i, :].divide(frqCurves.loc[i, :].sum())
        if keys:
            labels = [keys[label] for label in labels]
            frqCurves = pd.DataFrame(frqCurves.values, columns=labels, index=frqCurves.index)
    elif mode == 'parallel':
        frqCurves = pd.DataFrame(columns=labels, index=dataFrame.columns).fillna(0)
        for i in dataFrame.index:
            for j in dataFrame.columns:
                a = dataFrame.loc[i, j]
                if not pd.isnull(a):
                    frqCurves.loc[j, a] += 1
        for i in frqCurves.index:
            frqCurves.loc[i, :] = frqCurves.loc[i, :].divide(frqCurves.loc[i, :].sum())
        if keys:
            labels = [keys[label] for label in labels]
            frqCurves = pd.DataFrame(frqCurves.values, columns=labels, index=frqCurves.index)
    return frqCurves


def count_vector(dataFrame):
    countVector = pd.Series(index=dh.get_uniques(dataFrame)).fillna(0)
    for j, column in enumerate(dataFrame.columns):
        for i, row in enumerate(dataFrame.index):
            a = dataFrame.iloc[i, j]
            if not pd.isnull(a):
                countVector[dataFrame.iloc[i, j]] += 1
    return countVector


def symmetrical_matrix(dataFrame):
    """
    dataFrame: pandas DataFrame
    return: pandas DataFrame
    """
    labels = dh.get_uniques(dataFrame)
    symmetricalLabelMatrix = pd.DataFrame(columns=labels, index=labels)
    symmetricalLabelMatrix.fillna(0, inplace=True)
    return symmetricalLabelMatrix


def time_steps(dataFrame, mode):
    print(mode)
    print(dataFrame)
    if mode == 'series':
        return len(dataFrame.index), dh.get_uniques(pd.DataFrame(dataFrame.index))
    elif mode == 'parallel':
        return len(dataFrame.columns), dh.get_uniques(pd.DataFrame(dataFrame.columns))


def frequency_curves(dataFrame, mode='series', keys=None):
    labels = dh.get_uniques(dataFrame)
    if mode == 'series':
        frqCurves = pd.DataFrame(columns=labels, index=dataFrame.index).fillna(0)
        for i in dataFrame.index:
            for j in dataFrame.columns:
                a = dataFrame.loc[i, j]
                if not pd.isnull(a):
                    frqCurves.loc[i, a] += 1
        for i in frqCurves.index:
            frqCurves.loc[i, :] = frqCurves.loc[i, :].divide(frqCurves.loc[i, :].sum())
        if keys:
            labels = [keys[label] for label in labels]
            frqCurves = pd.DataFrame(frqCurves.values, columns=labels, index=frqCurves.index)
    elif mode == 'parallel':
        frqCurves = pd.DataFrame(columns=labels, index=dataFrame.columns).fillna(0)
        for i in dataFrame.index:
            for j in dataFrame.columns:
                a = dataFrame.loc[i, j]
                if not pd.isnull(a):
                    frqCurves.loc[j, a] += 1
        for i in frqCurves.index:
            frqCurves.loc[i, :] = frqCurves.loc[i, :].divide(frqCurves.loc[i, :].sum())
        if keys:
            labels = [keys[label] for label in labels]
            frqCurves = pd.DataFrame(frqCurves.values, columns=labels, index=frqCurves.index)
    return frqCurves


def count_matrix(dataFrame):
    countMatrix = symmetrical_matrix(dataFrame)
    for i, row in enumerate(dataFrame.iterrows()):
        for j, column in enumerate(dataFrame.columns):
            a = dataFrame.iloc[i, j]
            countMatrix.loc[a, a] += 1
    return countMatrix


def count_matrix(dataFrameList):
    countMatrix = symmetrical_matrix(dataFrameList)
    for dataFrame in dataFrameList:
        for i, row in enumerate(dataFrame.iterrows()):
            for j, column in enumerate(dataFrame.columns):
                a = dataFrame.iloc[i, j]
                countMatrix.loc[a, a] += 1
    return countMatrix


def count_matrix(dataFrame):
    countMatrix = symmetrical_label_matrix(dataFrame)
    for i, value in enumerate(dataFrame.iloc[:, 0]):
        a = dataFrame.iloc[i, 0]
        countMatrix.loc[a, a] += 1
    return countMatrix


def count_matrix(dataFrame):
    countMatrix = symmetrical_matrix(dataFrame)
    for i, row in enumerate(dataFrame.iterrows()):
        for j, column in enumerate(dataFrame.columns):
            a = dataFrame.iloc[i, j]
            countMatrix.loc[a, a] += 1
    return countMatrix


def count_vector(dataFrame):
    countVector = pd.Series(index=dh.get_uniques(dataFrame)).fillna(0)
    for j, column in enumerate(dataFrame.columns):
        for i, row in enumerate(dataFrame.index):
            a = dataFrame.iloc[i, j]
            if not pd.isnull(a):
                countVector[dataFrame.iloc[i, j]] += 1
    return countVector


def frequency_curves(dataFrame, mode='series', keys=None):
    labels = dh.get_uniques(dataFrame)
    if mode == 'series':
        frqCurves = pd.DataFrame(columns=labels, index=dataFrame.index).fillna(0)
        for i in dataFrame.index:
            for j in dataFrame.columns:
                a = dataFrame.loc[i, j]
                if not pd.isnull(a):
                    frqCurves.loc[i, a] += 1
        for i in frqCurves.index:
            frqCurves.loc[i, :] = frqCurves.loc[i, :].divide(frqCurves.loc[i, :].sum())
        if keys:
            labels = [keys[label] for label in labels]
            frqCurves = pd.DataFrame(frqCurves.values, columns=labels, index=frqCurves.index)
    elif mode == 'parallel':
        frqCurves = pd.DataFrame(columns=labels, index=dataFrame.columns).fillna(0)
        for i in dataFrame.index:
            for j in dataFrame.columns:
                a = dataFrame.loc[i, j]
                if not pd.isnull(a):
                    frqCurves.loc[j, a] += 1
        for i in frqCurves.index:
            frqCurves.loc[i, :] = frqCurves.loc[i, :].divide(frqCurves.loc[i, :].sum())
        if keys:
            labels = [keys[label] for label in labels]
            frqCurves = pd.DataFrame(frqCurves.values, columns=labels, index=frqCurves.index)
    return frqCurves


def count_matrix(dataFrame):
    countMatrix = symmetrical_matrix(dataFrame)
    for i, row in enumerate(dataFrame.iterrows()):
        for j, column in enumerate(dataFrame.columns):
            a = dataFrame.iloc[i, j]
            countMatrix.loc[a, a] += 1
    return countMatrix


def count_vector(dataFrame):
    labels = dh.get_uniques(dataFrame)
    countVector = {label: 0 for label in labels}
    for i, row in enumerate(dataFrame.iterrows()):
        for j, column in enumerate(dataFrame.columns):
            a = dataFrame.iloc[i, j]
            countVector[a] += 1
    return countVector


# TODO this function does not work
def frequency(dataFrame):
    frq = dh.normalize_axis(dataFrame, axis=1)
    return frq


def state_vector(state, dataFrame):
    """
    state: str
    data: pandas DataFrame
    return: pandas Series
    """
    labels = dh.get_uniques(dataFrame)
    stateVector = [1 if float(s) == float(state) else 0 for s in labels]
    return pd.Series(stateVector, index=labels)


def state_vector(state, dataFrame):
    labels = dh.get_uniques(dataFrame)
    stateVector = pd.Series([1 if float(s) == float(state) else 0 for s in labels], index=labels)
    return stateVector


def state_vector(state, dataFrameList):
    data = dh.merge_data_list(dataFrameList)
    labels = dh.get_uniques(data)
    stateVector = [1 if float(s) == float(state) else 0 for s in labels]
    return pd.Series(stateVector, index=labels)


def state_vector(state, data):
    d = data.iloc[:, 0].unique()
    stateVector = [1 if s == float(state) else 0 for s in d]
    return pd.Series(stateVector, index=data.iloc[:, 0].unique())


def time_steps(dataFrame, mode):
    if mode == 'series':
        return len(dataFrame.index)
    elif mode == 'parallel':
        return len(dataFrame.columns)


def time_steps(dataFrame, mode):
    if mode == 'series':
        return len(dataFrame.index), dh.get_uniques(pd.DataFrame(dataFrame.index))
    elif mode == 'parallel':
        return len(dataFrame.columns), dh.get_uniques(pd.DataFrame(dataFrame.columns))


def count_vector(dataFrame):
    countVector = pd.Series(index=dh.get_uniques(dataFrame)).fillna(0)
    for j, column in enumerate(dataFrame.columns):
        for i, row in enumerate(dataFrame.index):
            a = dataFrame.iloc[i, j]
            if not pd.isnull(a):
                countVector[dataFrame.iloc[i, j]] += 1
    return countVector


def frequency_curves(dataFrame, mode='series', keys=None):
    labels = dh.get_uniques(dataFrame)
    if mode == 'series':
        frqCurves = pd.DataFrame(columns=labels, index=dataFrame.index).fillna(0)
        for i in dataFrame.index:
            for j in dataFrame.columns:
                a = dataFrame.loc[i, j]
                if not pd.isnull(a):
                    frqCurves.loc[i, a] += 1
        for i in frqCurves.index:
            frqCurves.loc[i, :] = frqCurves.loc[i, :].divide(frqCurves.loc[i, :].sum())
        if keys:
            labels = [keys[label] for label in labels]
            frqCurves = pd.DataFrame(frqCurves.values, columns=labels, index=frqCurves.index)
    elif mode == 'parallel':
        frqCurves = pd.DataFrame(columns=labels, index=dataFrame.columns).fillna(0)
        for i in dataFrame.index:
            for j in dataFrame.columns:
                a = dataFrame.loc[i, j]
                if not pd.isnull(a):
                    frqCurves.loc[j, a] += 1
        for i in frqCurves.index:
            frqCurves.loc[i, :] = frqCurves.loc[i, :].divide(frqCurves.loc[i, :].sum())
        if keys:
            labels = [keys[label] for label in labels]
            frqCurves = pd.DataFrame(frqCurves.values, columns=labels, index=frqCurves.index)
    return frqCurves


def frequency_curves(dataFrame):
    labels = dh.get_uniques(dataFrame)
    frqCurves = pd.DataFrame(columns=dataFrame.columns, index=labels).fillna(0)
    for i in dataFrame.index:
        for j in dataFrame.columns:
            frqCurves.loc[dataFrame.loc[i, j], j] += 1
    for j in frqCurves.columns:
        frqCurves.loc[:, j] = frqCurves.loc[:, j].divide(frqCurves.loc[:, j].sum())
    return frqCurves


def count_vector(dataFrame):
    labels = dh.get_uniques(dataFrame)
    countVector = {label: 0 for label in labels}
    for i, row in enumerate(dataFrame.iterrows()):
        for j, column in enumerate(dataFrame.columns):
            a = dataFrame.iloc[i, j]
            countVector[a] += 1
    return countVector


def count_vector(dataFrame):
    countVector = pd.Series(index=dh.get_uniques(dataFrame)).fillna(0)
    for j, column in enumerate(dataFrame.columns):
        for i, row in enumerate(dataFrame.index):
            a = dataFrame.iloc[i, j]
            if not pd.isnull(a):
                countVector[dataFrame.iloc[i, j]] += 1
    return countVector


def count_vector(dataFrame):
    countVector = pd.Series(index=dh.get_uniques(dataFrame)).fillna(0)
    for j, column in enumerate(dataFrame.columns):
        for i, row in enumerate(dataFrame.index):
            countVector[dataFrame.iloc[i, j]] += 1
    return countVector


def spin(q, w):
    return 0.5 * w * q

'''

'''
# MARKOV CHAIN MONTE CARLO
def determine_state(states, currentState):
    for state, prob in zip(states, currentState):
        if prob == 1:
            return state


# TODO BAD
def old_count_matrix(data, countMatrix, columns=None, type=None):
    """
    Count data and store the count in matrix (pre-labeled DataFrame objects)
    data: pandas DataFrame
    columns: list of strings
    hist: boolean
    """
    cols = columns if columns else data.columns
    matrix = countMatrix
    if not hst:
        matrix.fillna(0, inplace=True)
    for i, row in enumerate(data.iterrows()):
        for j in range(len(data.loc[row[0], :])):
            if True:  # If columns are specified: row=the value in (i,j), column=column j
                matrix.loc[data.iloc[i, j], cols[j]] += 1
            if type == 'histogram':
                matrix[data.iloc[i, j]].append(int(cols[j]) - 1991)

    return matrix


def process_data(column, dir):
    """
    column: str
    dir: str
    returns: list of pandas dataFrames and a list of numerics to name the files
    """
    data = []
    for subdir, dirs, files in os.walk(dir):
        for i, file in enumerate(files):
            data.append(
                pd.read_csv(
                    os.path.join(dir, file), usecols=[dh.vars.ID] + [column], na_values=NA_VALUES, dtype=str,
                    encoding=dh.vars.ENCODING
                )
            )
            data[i].set_index(dh.vars.ID, inplace=True)
    data = dh.clean_data(dh.concat_data(dataFrame=data, index=column))
    return data


def frq(focus, matrix):
    """
    Count state transitions in data and store the count in matrix (pre-labeled DataFrame objects)
    """
    matrix = matrix
    if focus == "year":
        frq = matrix.T
        frq = frq[frq.columns[::-1]]
        return dh.normalize_rows(frq).as_matrix()
    elif focus == "state":
        return dh.normalize_rows(matrix.T).T.as_matrix()


def hst(data, matrix, columns):  # TODO bad method + needs generalizing (only works for freq vs state)
    print(data)
    print(matrix)
    hst = count_matrix(data=data, matrix=matrix, columns=columns, hst=True)
    hst = pd.DataFrame([hst[m] for m in hst], index=[m for m in hst])
    print(hst)
    hst.fillna(0, inplace=True)
    hst.sort_index(ascending=False, inplace=True)
    return hst.as_matrix().astype(np.float64)


def sim_frq(simulation, model, focus):
    """
    Count state transitions in data and store the count in matrix (pre-labeled DataFrame objects)
    """
    matrix = count_matrix(pd.DataFrame(simulation), pd.DataFrame(index=model.states, columns=model.yrs), model.yrs)
    return frq(focus, matrix)


def sim_hst(simulation, model, time, focus):
    if focus == "year":
        return pd.DataFrame(
            simulation, index=[x for x in range(len(simulation))],
            columns=[y for y in range(time)]
        ).T.as_matrix().astype(np.float64)
    elif focus == "state":
        return hst(data=pd.DataFrame(simulation), matrix={s: [] for s in model.states}, columns=model.yrs)


def sample(states, probabilityVector):
    """
    states: list
    probabilityVector is a probability vector: list
    iteration: int
    currentState is the current state: list
    returns a state vector: list
    """
    sum = 0
    randomNumber = random.uniform(0, 1)
    # Assign state if randomNumber is within its range
    for state, prob in zip(states, probabilityVector):
        sum += prob
        if (sum >= randomNumber):
            return state, np.transpose((pd.Series([1 if s == state else 0 for s in states], index=states)))


def raw_frq(data, states, yrs, focus):
    matrix = count_matrix(data, pd.DataFrame(index=states, columns=yrs), yrs)
    return frq(focus, matrix)


def raw_hst(data, states, yrs, focus):
    if focus == "year":
        return data.T.as_matrix().astype(np.float64)
    elif focus == "state":
        return hst(data=data, matrix={s: [] for s in states}, columns=yrs)


def symmetrical_label_matrix(dataFrame):
    labels = dataFrame.iloc[:, 0].unique()
    symmetricalLabelMatrix = pd.DataFrame(columns=labels, index=labels)
    symmetricalLabelMatrix.fillna(0, inplace=True)
    return symmetricalLabelMatrix


def transition_matrix(dataFrame):
    """
    dataFrame: pandas dataFrame if mode is 'series' or list of pandas dataFrames if mode is 'parallel
    mode: str 'series' or 'parallel'
    return: pandas dataFrame
    """
    transitionMatrix = symmetrical_label_matrix(dataFrame)
    for i, value in enumerate(dataFrame.iloc[:, 0]):
        if i != len(dataFrame) - 1:
            a = dataFrame.iloc[i, 0]
            b = dataFrame.iloc[i + 1, 0]
            transitionMatrix.loc[a, b] += 1
    return transitionMatrix


def transition_probability_matrix(dataFrame):
    transitionProbabilityMatrix = dh.normalize_rows(transition_matrix(dataFrame), rows=True)
    return transitionProbabilityMatrix


def parallel_labels(dataFrameList):
    labels = []
    for dataFrame in dataFrameList:
        for i, value in enumerate(pd.DataFrame(dataFrame).iloc[:, 0]):
            labels.append(value)
    labels = pd.Series(labels).unique()
    return labels


def read_data_parallel(dir, index, column):
    data = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            data.append(
                dh.clean_data(pd.read_csv(os.path.join(dir, file), usecols=[index, column], na_values=NA_VALUES))
            )
    for dF in data:
        dF.set_index(index, inplace=True)
    return data


def merger(dataFrameList):
    d = pd.DataFrame()
    for i, dataFrame in enumerate(dataFrameList):
        dataFrame.columns = [str(dataFrame.columns.values[0]) + " " + str(i)]
        d = d.join(dataFrame, how='outer')
    return d


def count_matrix_parallel(data, labels):
    countMatrix = pd.DataFrame(columns=labels, index=labels)
    countMatrix.fillna(0, inplace=True)
    for i, row in enumerate(data.iterrows()):
        for j, column in enumerate(data.columns):
            a = data.iloc[i, j]
            try:
                countMatrix.loc[a, a] += 1
            except:
                pass
    return countMatrix


def transition_matrix_parallel(data, labels):
    transitionMatrix = pd.DataFrame(columns=labels, index=labels)
    transitionMatrix.fillna(0, inplace=True)
    for i, row in enumerate(data.iterrows()):
        for j, column in enumerate(data.columns):
            if j != len(data.columns) - 1:
                a = data.iloc[i, j]
                b = data.iloc[i, j + 1]
                try:
                    transitionMatrix.loc[a, b] += 1
                except:
                    pass
    return transitionMatrix


def transition_probability_matrix_parallel(data, labels):
    transitionProbabilityMatrix = dh.normalize_rows(transition_matrix_parallel(data, labels), rows=True)
    return transitionProbabilityMatrix


def old_monte_carlo(model):
    """currentState is the current state"""
    simulation = []
    time = len(model.data.columns)
    for i in range(model.iterations):
        simulation.append([])
        currentState = model.initialState
        for t in range(time):
            if t != 0:
                # Multiply current state and transition matrix, resulting in a probability vector
                probabilityVector = currentState.dot(model.matrix)
                state, currentState = sample(model.states, probabilityVector)
                simulation[i].append(state)
            else:
                state = determine_state(model.states, currentState)
                simulation[i].append(state)
        simulation[i] = pd.Series(simulation[i])


# TODO BAD
def create_histogram(data, countMatrix):
    """
    Count data and store the count in matrix (pre-labeled DataFrame objects)
    data: pandas DataFrame
    columns: list of strings
    hist: boolean
    """
    matrix = countMatrix
    if not hst:
        matrix.fillna(0, inplace=True)
    for i, row in enumerate(data.iterrows()):
        for j in range(len(data.loc[row[0], :])):
            if True:  # If columns are specified: row=the value in (i,j), column=column j
                matrix.loc[data.iloc[i, j], cols[j]] += 1
            if type == 'histogram':
                matrix[data.iloc[i, j]].append(int(cols[j]) - 1991)

    return matrix


def get_probability_vectors(states, transitionProbabilityMatrix):
    probabilityVectors = []
    for state in states:
        probabilityVectors.append(pd.Series([1 if s == state else 0 for s in states], index=states))
    probabilityVectorsDict = {state: pd.Series(probabilityVector.dot(transitionProbabilityMatrix)) for
                              state, probabilityVector in zip(states, probabilityVectors)}
    return probabilityVectorsDict


def get_state_vectors(states):
    stateVectorsDict = {state: np.transpose((pd.Series([1 if s == state else 0 for s in states], index=states))) for
                        state in states}
    return stateVectorsDict


def monte_carlo_curve_fit(
    initialStateVector, transitionProbabilityMatrix, iterations, timeSteps,
    metropolisHastings=False
):
    monteCarlo = []
    index = initialStateVector.index
    probabilityVectors = get_probability_vectors(index, transitionProbabilityMatrix)
    stateVectorsDict = get_state_vectors(index)
    for i in range(iterations):
        currentStateVector = initialStateVector
        state = [state for state, prob in zip(index, currentStateVector) if prob == 1][0]
        simulation = [state]
        for t in range(1, timeSteps):
            probabilityVector = probabilityVectors[state]
            if probabilityVector.sum() != 0:
                randomNumber, proposedState, proposedStateVector = sample(index, probabilityVector, stateVectorsDict)
            if metropolisHastings:
                state, currentStateVector = metropolis_hastings(
                    proposedState=proposedState,
                    proposedStateVector=proposedStateVector,
                    priorState=state, priorStateVector=currentStateVector,
                    randomNumber=randomNumber, p=0
                )
            else:
                state, currentStateVector = proposedState, proposedStateVector
            simulation.append(state)
        monteCarlo.append(pd.Series(simulation))
    monteCarloMatrix = dh.concat_data(monteCarlo, columns=list(range(iterations)))
    return monteCarloMatrix, pd.DataFrame()


def metropolis_hastings(proposedState, proposedStateVector, priorState, priorStateVector, randomNumber, p):
    if randomNumber <= p:
        print('proposed')
        return proposedState, proposedStateVector
    else:
        print('prior')
        return priorState, priorStateVector


def experimental_monte_carlo(distribution, transitionProbabilityMatrix, iterations):
    try:
        simulation = []
        likelihood = []
        for i in range(iterations):
            sum = 0
            randomNumber = np.random.uniform(0, 1)
            for state in distribution.index:
                sum += distribution.loc[state]
                if sum >= randomNumber:
                    simulation.append(state)
                    if len(simulation) > 1:
                        likelihood.append(get_likelihood(simulation[-1], simulation[-2], transitionProbabilityMatrix))
                    break
        monteCarlo = pd.DataFrame(simulation)
        likelihood = pd.DataFrame(likelihood)
        return monteCarlo, likelihood
    except Exception as e:
        print(e)


def monte_carlo(stateVector, transitionProbabilityMatrix, iterations):
    simulation = []
    for i in range(iterations):
        simulation.append([])
        currentState = stateVector


def sample(states, probabilityVector, stateVectorsDict):
    """
    states: list
    probabilityVector is a probability vector: list
    returns state value, state vector: float, list
    """
    sum = 0
    randomNumber = np.random.uniform(0, 1)
    for state, prob in zip(states, probabilityVector):
        sum += prob
        if (sum >= randomNumber):
            return randomNumber, state, stateVectorsDict[state]


def get_likelihood(proposedState, priorState, transitionProbabilityMatrix):
    pX0 = transitionProbabilityMatrix.loc[priorState, proposedState]
    pX1 = transitionProbabilityMatrix.loc[proposedState, priorState]
    if pX1 != 0:
        return pX0 / pX1
    else:
        return 0


def monte_carlo_frequency(monteCarloMatrix, keys):
    temp = mh.frequency(monteCarloMatrix)
    frq = pd.Series(index=keys).fillna(0)
    for i in temp.index:
        frq[i] = temp[i]
    return frq


def monte_carlo_curve_fit(
    initialStateVector, transitionProbabilityMatrix, iterations, timeSteps,
    metropolisHastings=False
):
    monteCarlo = []
    index = initialStateVector.index
    probabilityVectors = get_probability_vectors(index, transitionProbabilityMatrix)
    stateVectorsDict = get_state_vectors(index)
    for i in range(iterations):
        currentStateVector = initialStateVector
        state = [state for state, prob in zip(index, currentStateVector) if prob == 1][0]
        simulation = [state]
        for t in range(1, timeSteps):
            probabilityVector = probabilityVectors[state]
            if probabilityVector.sum() != 0:
                randomNumber, proposedState, proposedStateVector = sample(index, probabilityVector, stateVectorsDict)
            if metropolisHastings:
                state, currentStateVector = metropolis_hastings(
                    proposedState=proposedState,
                    proposedStateVector=proposedStateVector,
                    priorState=state, priorStateVector=currentStateVector,
                    randomNumber=randomNumber, p=0
                )
            else:
                state, currentStateVector = proposedState, proposedStateVector
            simulation.append(state)
        monteCarlo.append(pd.Series(simulation))
    monteCarloMatrix = dh.concat_data(monteCarlo, columns=list(range(iterations)))
    return monteCarloMatrix, pd.DataFrame()


def monte_carlo(initialStateVector, transitionProbabilityMatrix, iterations, timeSteps):
    simulations = []
    index = initialStateVector.index
    for i in range(iterations):
        simulation = []
        currentState = initialStateVector
        for t in range(timeSteps):
            if t != 0:
                probabilityVector = pd.Series(currentState.dot(transitionProbabilityMatrix))
                if probabilityVector.sum() != 0:
                    state, currentState = sample(index, probabilityVector)
                else:
                    pass  # le = len(probabilityVector)  # probabilityVector = pd.Series(np.random.rand(le), index=probabilityVector.index)
            else:
                state = determine_state(index, currentState)
            simulation.append(state)
        simulations.append(pd.Series(simulation))
    return dh.concat_data(simulations, columns=list(range(iterations)))


def monte_carlo(initialStateVector, transitionProbabilityMatrix, iterations, metropolisHastings=False):
    """
    initialStateVector: pandas Series
    transitionProbabilityMatrix: pandas DataFrame
    iterations: int
    metropolisHastings: boolean
    return: tuple of (pandas Dataframe, pandas DataFrame)
    """
    index = initialStateVector.index
    probabilityVectors = get_probability_vectors(index, transitionProbabilityMatrix)
    stateVectorsDict = get_state_vectors(index)
    currentStateVector = initialStateVector
    state = [state for state, prob in zip(index, currentStateVector) if prob == 1][0]
    simulation = [state]
    likelihoodList = []
    for i in range(iterations):
        probabilityVector = probabilityVectors[state]
        if probabilityVector.sum() != 0:
            randomNumber, proposedState, proposedStateVector = sample(index, probabilityVector, stateVectorsDict)
        p = get_likelihood(
            proposedState=proposedState, priorState=state,
            transitionProbabilityMatrix=transitionProbabilityMatrix
        )
        if metropolisHastings:
            state, currentStateVector = metropolis_hastings(
                proposedState=proposedState,
                proposedStateVector=proposedStateVector, priorState=state,
                priorStateVector=currentStateVector,
                randomNumber=randomNumber, p=p
            )
        else:
            state, currentStateVector = proposedState, proposedStateVector
        simulation.append(state)
        likelihoodList.append(p)
    monteCarloMatrix = pd.DataFrame(simulation)
    likelihood = pd.DataFrame(likelihoodList)
    return monteCarloMatrix, likelihood


def monte_carlo(initialStateVector, transitionProbabilityMatrix, iterations, timeSteps, metropolisHastings=False):
    monteCarlo = []
    index = initialStateVector.index
    probabilityVectors = get_probability_vectors(index, transitionProbabilityMatrix)
    stateVectorsDict = get_state_vectors(index)
    for i in range(iterations):
        currentStateVector = initialStateVector
        state = [state for state, prob in zip(index, currentStateVector) if prob == 1][0]
        simulation = [state]
        for t in range(1, timeSteps):
            probabilityVector = probabilityVectors[state]
            if probabilityVector.sum() != 0:
                randomNumber, proposedState, proposedStateVector = sample(index, probabilityVector, stateVectorsDict)
            if metropolisHastings:
                state, currentStateVector = metropolis_hastings(
                    proposedState=proposedState,
                    proposedStateVector=proposedStateVector,
                    priorState=state, priorStateVector=currentStateVector,
                    randomNumber=randomNumber,
                    transitionProbabilityMatrix=transitionProbabilityMatrix
                )
            else:
                state, currentStateVector = proposedState, proposedStateVector
            simulation.append(state)
        monteCarlo.append(pd.Series(simulation))
    return dh.concat_data(monteCarlo, columns=list(range(iterations)))


def frq(focus, matrix):
    """
    Count state transitions in data and store the count in matrix (pre-labeled DataFrame objects)
    """
    matrix = matrix
    distribution = matrix.T
    if focus == "year":
        distribution = distribution[distribution.columns[::-1]]
        return dh.normalize_rows(distribution).as_matrix()
    elif focus == "state":
        return dh.normalize_rows(distribution).T.as_matrix()


# TODO BAD
def create_histogram(data, countMatrix):
    """
    Count data and store the count in matrix (pre-labeled DataFrame objects)
    data: pandas DataFrame
    columns: list of strings
    hist: boolean
    """
    matrix = countMatrix
    if not hst:
        matrix.fillna(0, inplace=True)
    for i, row in enumerate(data.iterrows()):
        for j in range(len(data.loc[row[0], :])):
            if True:  # If columns are specified: row=the value in (i,j), column=column j
                matrix.loc[data.iloc[i, j], cols[j]] += 1
            if type == 'histogram':
                matrix[data.iloc[i, j]].append(int(cols[j]) - 1991)
    return matrix


def metropolis_hastings(
    proposedState, proposedStateVector, priorState, priorStateVector, randomNumber,
    transitionProbabilityMatrix
):
    pX0 = transitionProbabilityMatrix.loc[priorState, proposedState]
    pX1 = transitionProbabilityMatrix.loc[proposedState, priorState]
    if pX1 == 0:
        print('goofed!')
        return priorState, priorStateVector
    if randomNumber <= pX0 / pX1:
        print('proposed')
        return proposedState, proposedStateVector
    else:
        print('prior')
        return priorState, priorStateVector


def metropolis_hastings(proposedState, proposedStateVector, priorState, priorStateVector, randomNumber, p):
    if randomNumber <= p:
        print('proposed')
        return proposedState, proposedStateVector
    else:
        print('prior')
        return priorState, priorStateVector


def experimental_monte_carlo(distribution, transitionProbabilityMatrix, iterations):
    try:
        simulation = []
        likelihood = []
        for i in range(iterations):
            sum = 0
            randomNumber = np.random.uniform(0, 1)
            for state in distribution.index:
                sum += distribution.loc[state]
                if sum >= randomNumber:
                    simulation.append(state)
                    if len(simulation) > 1:
                        likelihood.append(get_likelihood(simulation[-1], simulation[-2], transitionProbabilityMatrix))
                    break
        monteCarlo = pd.DataFrame(simulation)
        likelihood = pd.DataFrame(likelihood)
        return monteCarlo, likelihood
    except Exception as e:
        print(e)


def sample(states, probabilityVector, stateVectorsDict):
    """
    states: list
    probabilityVector is a probability vector: list
    returns state value, state vector: float, list
    """
    # random.seed(1)
    sum = 0
    randomNumber = np.random.uniform(0, 1)
    for state, prob in zip(states, probabilityVector):
        sum += prob
        if (sum >= randomNumber):
            return randomNumber, state, stateVectorsDict[state]


def get_likelihood(proposedState, priorState, transitionProbabilityMatrix):
    pX0 = transitionProbabilityMatrix.loc[priorState, proposedState]
    pX1 = transitionProbabilityMatrix.loc[proposedState, priorState]
    if pX1 != 0:
        return pX0 / pX1
    return 0


def get_strata(iterations):
    randomNumbers = []
    strata = [1 / iterations]
    # strata = np.empty(iterations)
    # strata[0] = 1/iterations
    for i in range(1, iterations):
        strata.append(strata[-1] + 1 / iterations)  # strata[i] = strata[i-1] + 1/iterations
    for i in range(iterations):
        if i == 0:
            randomNumbers.append(np.random.uniform(0, strata[i]))
        else:
            randomNumbers.append(np.random.uniform(strata[i - 1], strata[i]))
    return randomNumbers


def latin_hypercube(distribution, iterations, transitionProbabilityMatrix):
    try:
        strata = get_strata(iterations)
        simulation = []
        likelihood = []
        for i in range(iterations):
            sum = 0
            for state in distribution.index:
                sum += distribution.loc[state]
                if sum >= strata[i]:
                    simulation.append(state)
                    if len(simulation) > 1:
                        likelihood.append(get_likelihood(simulation[-1], simulation[-2], transitionProbabilityMatrix))
                    break
        latinHypercube = pd.DataFrame(simulation)
        likelihood = pd.DataFrame(likelihood)
        return latinHypercube, likelihood
    except Exception as e:
        print(e)


def markov_chain(stateVector, transitionMatrix, keys=None):
    """
    initialState is multiplied by transitionMatrix as many times as iterations.
    Each result is stored in a list.
    stateVector: pandas Series
    transitionMatrix: pandas DataFrame
    iterations: int
    """
    tol = 0.001
    lookBehind = 20
    markovChain = [stateVector]
    lookAhead = 0
    for i in range(10000 - 1):
        markovChain.append(pd.Series(markovChain[i].dot(transitionMatrix)))
        comparison = np.allclose(markovChain[i], markovChain[i + 1], tol, tol)
        if comparison:
            lookAhead += 1
        else:
            lookAhead = 0

        if lookAhead >= lookBehind:
            comparison2 = np.allclose(markovChain[i + 1 - lookAhead], markovChain[i + 1], tol, tol)
            if comparison2:
                break
            else:
                lookAhead = 0
    markovChain = dp.concat_data(seriesList=markovChain)
    """ALTERNATE METHOD
    identityMatrix = np.eye(len(transitionMatrix))
    r = transitionMatrix - identityMatrix
    s = []
    for i in range(len(r)):
        if i == len(r) - 1:
            s.append([1 for _ in range(len(transitionMatrix))])
        else:
            s.append(r.iloc[:, i].as_matrix())
    x = np.array(s)
    y = np.array([0 for _ in range(len(transitionMatrix) - 1)] + [1])
    z = pd.Series(np.linalg.solve(x, y), index=markovChain.index)
    """
    if keys:
        markovChain.index = [keys[i] for i in stateVector.index]
    return markovChain, markovChain.iloc[:, -1]


def create_transition_matrix_html(transitionMatrix, path):
    with open(path, 'w') as f:
        f.truncate()
        f.write(
            r"""<!DOCTYPE html>
                            <html lang="en">
                            <head>
                                <meta charset="UTF-5">
                                <link rel="stylesheet" href="TransitionMatrix.css" type="text/css" />
                                <title>Tables</title>
                            </head>
                            <body>

                                <table border="1" object_oriented="table">
                                    <caption object_oriented="caption">Transition Matrix</caption>
                                    <!--columns-->

                                    <thead>
                                        <tr>
                                            <th></th>"""
        )
        for column in transitionMatrix.columns:
            f.write('<th>' + str(column) + '</th>\n')
        f.write(
            """
                                    </tr>
                                    </thead>
                                    <!--rows-->
                                    <tbody>"""
        )
        for index in transitionMatrix.index:
            f.write('<tr>\n')
            f.write('<td><span object_oriented="labels">' + str(index) + '</span></td>\n')
            for column in transitionMatrix.columns:
                f.write('<td>' + str(transitionMatrix.loc[index, column]) + '</td>\n')

            f.write('</tr>\n')

        f.write(
            """
                                    </tbody>
                                </table>
                            </body>
                            </html>"""
        )


def markov_chain(stateVector, transitionMatrix, iterations):
    """
    initialState is multiplied by transitionMatrix as many times as iterations.
    Each result is stored in a list.
    stateVector: pandas Series
    transitionMatrix: pandas DataFrame
    iterations: int
    """
    markovChain = [stateVector]
    for i in range(iterations - 1):
        markovChain.append(pd.Series(markovChain[i].dot(transitionMatrix)))
    return dh.concat_data(seriesList=markovChain, columns=list(range(iterations)))


def markov_chain(stateVector, transitionMatrix, iterations, keys=None):
    """
    initialState is multiplied by transitionMatrix as many times as iterations.
    Each result is stored in a list.
    stateVector: pandas Series
    transitionMatrix: pandas DataFrame
    iterations: int
    """
    tol = 0.001
    lookBehind = 20
    markovChain = [stateVector]
    lookAhead = 0
    for i in range(10000 - 1):
        markovChain.append(pd.Series(markovChain[i].dot(transitionMatrix)))
        comparison = np.allclose(markovChain[i], markovChain[i + 1], tol, tol)
        if comparison:
            lookAhead += 1
        else:
            lookAhead = 0

        if lookAhead >= lookBehind:
            comparison2 = np.allclose(markovChain[i + 1 - lookAhead], markovChain[i + 1], tol, tol)
            if comparison2:
                break
            else:
                lookAhead = 0
    markovChain = dh.concat_data(seriesList=markovChain)
    """ALTERNATE METHOD
    identityMatrix = np.eye(len(transitionMatrix))
    r = transitionMatrix - identityMatrix
    s = []
    for i in range(len(r)):
        if i == len(r) - 1:
            s.append([1 for _ in range(len(transitionMatrix))])
        else:
            s.append(r.iloc[:, i].as_matrix())
    x = np.array(s)
    y = np.array([0 for _ in range(len(transitionMatrix) - 1)] + [1])
    z = pd.Series(np.linalg.solve(x, y), index=markovChain.index)
    """
    if keys:
        markovChain.index = [keys[i] for i in stateVector.index]
    return markovChain, markovChain.iloc[:, -1]


def transition_matrix(dataFrameList, mode='parallel'):
    """
    series considers transitions from row to row for each column, regardless of index.
    parallel considers transitions from column to column for each index.
    param dataFrameList: list of pandas DataFrames
    param mode: 'series' or 'parallel'
    return: pandas DataFrame
    """
    transitionMatrix = vs.symmetrical_matrix(dataFrameList)
    data = dh.merge_data_list(dataFrameList)
    if mode == 'series':
        for j, column in enumerate(data.columns):
            for i, row in enumerate(data.iterrows()):
                if i != len(data.index) - 1:
                    a = data.iloc[i, j]
                    b = data.iloc[i + 1, j]
                    try:
                        transitionMatrix.loc[a, b] += 1
                    except:
                        pass
    elif mode == 'parallel':
        for i, row in enumerate(data.iterrows()):
            for j, column in enumerate(data.columns):
                if j != len(data.columns) - 1:
                    a = data.iloc[i, j]
                    b = data.iloc[i, j + 1]
                    try:
                        transitionMatrix.loc[a, b] += 1
                    except:
                        pass
    return transitionMatrix


def transition_matrix(data, mode='series', condition=None):
    transitionMatrix = mh.symmetrical_matrix(data)
    if mode == 'series':
        for j, column in enumerate(data.columns):
            for i, row in enumerate(data.iterrows()):
                if i != len(data.index) - 1:
                    a = data.iloc[i, j]
                    b = data.iloc[i + 1, j]
                    try:
                        if not condition:
                            transitionMatrix.loc[a, b] += 1
                        else:
                            if condition(a, b):
                                transitionMatrix.loc[a, b] += 1
                    except:
                        pass
    elif mode == 'parallel':
        for i, row in enumerate(data.iterrows()):
            for j, column in enumerate(data.columns):
                if j != len(data.columns) - 1:
                    a = data.iloc[i, j]
                    b = data.iloc[i, j + 1]
                    try:
                        if not condition:
                            transitionMatrix.loc[a, b] += 1
                        else:
                            if condition(a, b):
                                transitionMatrix.loc[a, b] += 1
                    except:
                        pass
    return transitionMatrix


def transition_matrix(data, mode='series', condition=None):
    """
    series considers transitions from row to row for each column, regardless of index.
    parallel considers transitions from column to column for each index.
    param dataFrameList: list of pandas DataFrames
    param mode: 'series' or 'parallel'
    return: pandas DataFrame
    """
    transitionMatrix = mh.symmetrical_matrix(data)
    if mode == 'series':
        for i, row in enumerate(data.iterrows()):
            if i != len(data.index) - 1:
                a = data.iloc[i, 0]
                b = data.iloc[i + 1, 0]
                if not condition:
                    transitionMatrix.loc[a, b] += 1
                else:
                    if condition(a, b):
                        transitionMatrix.loc[a, b] += 1
    elif mode == 'parallel':
        for i, row in enumerate(data.iterrows()):
            for j, column in enumerate(data.columns):
                if j != len(data.columns) - 1:
                    a = data.iloc[i, j]
                    b = data.iloc[i, j + 1]
                    try:
                        if not condition:
                            transitionMatrix.loc[a, b] += 1
                        else:
                            if condition(a, b):
                                transitionMatrix.loc[a, b] += 1
                    except:
                        pass
    return transitionMatrix


def transition_matrix(data, mode='series', condition=None):
    transitionMatrix = mh.symmetrical_matrix(data)
    if mode == 'series':
        for j, column in enumerate(data.columns):
            for i, row in enumerate(data.iterrows()):
                if i != len(data.index) - 1:
                    a = data.iloc[i, j]
                    b = data.iloc[i + 1, j]
                    try:
                        if not condition:
                            transitionMatrix.loc[a, b] += 1
                        else:
                            if condition(a, b):
                                transitionMatrix.loc[a, b] += 1
                    except:
                        pass
    elif mode == 'parallel':
        for i, row in enumerate(data.iterrows()):
            for j, column in enumerate(data.columns):
                if j != len(data.columns) - 1:
                    a = data.iloc[i, j]
                    b = data.iloc[i, j + 1]
                    try:
                        if not condition:
                            transitionMatrix.loc[a, b] += 1
                        else:
                            if condition(a, b):
                                transitionMatrix.loc[a, b] += 1
                    except:
                        pass
    return transitionMatrix


def transition_matrix(data, mode='series', condition=None):
    """
    series considers transitions from row to row for each column, regardless of index.
    parallel considers transitions from column to column for each index.
    param dataFrameList: list of pandas DataFrames
    param mode: 'series' or 'parallel'
    return: pandas DataFrame
    """
    transitionMatrix = mh.symmetrical_matrix(data)
    if mode == 'series':
        for j, column in enumerate(data.columns):
            for i, row in enumerate(data.iterrows()):
                if i != len(data.index) - 1:
                    a = data.iloc[i, j]
                    b = data.iloc[i + 1, j]
                    if type(a) != type(np.nan) and type(b) != type(np.nan):
                        if not condition:
                            transitionMatrix.loc[a, b] += 1
                        else:
                            if condition(a, b):
                                transitionMatrix.loc[a, b] += 1
    elif mode == 'parallel':
        for i, row in enumerate(data.iterrows()):
            for j, column in enumerate(data.columns):
                if j != len(data.columns) - 1:
                    a = data.iloc[i, j]
                    b = data.iloc[i, j + 1]
                    try:
                        if not condition:
                            transitionMatrix.loc[a, b] += 1
                        else:
                            if condition(a, b):
                                transitionMatrix.loc[a, b] += 1
                    except:
                        pass
    return transitionMatrix


def transition_matrix(data, mode='series', condition=None):
    transitionMatrix = mh.symmetrical_matrix(data)
    if mode == 'series':
        for j, column in enumerate(data.columns):
            for i, row in enumerate(data.iterrows()):
                if i != len(data.index) - 1:
                    a = data.iloc[i, j]
                    b = data.iloc[i + 1, j]
                    try:
                        if not condition:
                            transitionMatrix.loc[a, b] += 1
                        else:
                            if condition(a, b):
                                transitionMatrix.loc[a, b] += 1
                    except:
                        pass
    elif mode == 'parallel':
        for i, row in enumerate(data.iterrows()):
            for j, column in enumerate(data.columns):
                if j != len(data.columns) - 1:
                    a = data.iloc[i, j]
                    b = data.iloc[i, j + 1]
                    try:
                        if not condition:
                            transitionMatrix.loc[a, b] += 1
                        else:
                            if condition(a, b):
                                transitionMatrix.loc[a, b] += 1
                    except:
                        pass
    return transitionMatrix


def transition_matrix(data, mode='series', condition=None):
    """
    series considers transitions from row to row for each column, regardless of index.
    parallel considers transitions from column to column for each index.
    param dataFrameList: list of pandas DataFrames
    param mode: 'series' or 'parallel'
    return: pandas DataFrame
    """
    transitionMatrix = mh.symmetrical_matrix(data)
    if mode == 'series':
        for j, column in enumerate(data.columns):
            for i, row in enumerate(data.iterrows()):
                if i != len(data.index) - 1:
                    a = data.iloc[i, j]
                    b = data.iloc[i + 1, j]
                    try:
                        if not condition:
                            transitionMatrix.loc[a, b] += 1
                        else:
                            if condition(a, b):
                                transitionMatrix.loc[a, b] += 1
                    except:
                        pass
    elif mode == 'parallel':
        for i, row in enumerate(data.iterrows()):
            for j, column in enumerate(data.columns):
                if j != len(data.columns) - 1:
                    a = data.iloc[i, j]
                    b = data.iloc[i, j + 1]
                    try:
                        if not condition:
                            transitionMatrix.loc[a, b] += 1
                        else:
                            if condition(a, b):
                                transitionMatrix.loc[a, b] += 1
                    except:
                        pass
    return transitionMatrix


def transition_probability_matrix(dataFrameList, mode='parallel', condition=None):
    """
    series considers transitions from row to row for each column, regardless of index.
    parallel considers transitions from column to column for each index.
    dataFrame: list of pandas DataFrames
    mode: 'series' or 'parallel'
    return: pandas DataFrame
    """
    transitionProbabilityMatrix = dh.normalize_axis(transition_matrix(dataFrameList, mode, condition))
    return transitionProbabilityMatrix


def transition_probability_matrix(dataFrameList, mode='parallel', condition=None):
    """
    mode=series considers transitions from row to row for each column, regardless of index.
    mode=parallel considers transitions from column to column for each index.
    dataFrameList: list of pandas DataFrames
    mode: 'series' or 'parallel'
    condition: '
    return: pandas DataFrame
    """
    return dp.normalize_axis(transition_matrix(dataFrameList, mode, condition))


def create_transition_matrix_html(transitionMatrix):
    with open(TRANSITION_MATRIX, 'w') as f:
        f.truncate()
        f.write(
            r"""
                        <!DOCTYPE html>
                        <html lang="en">
                        <head>
                        <meta charset="UTF-5">
                        <link rel="stylesheet" href="TransitionMatrix.css" type="text/css" />
                        <title>Tables</title>
                        </head>
                        <body>
                        <table border="1" object_oriented="table">
                        <caption object_oriented="caption">Transition Matrix</caption>
                        <!--columns-->
                        <thead>
                        <tr>
                        <th>
                        </th>
                        """
        )
        for column in transitionMatrix.columns:
            f.write('<th>' + str(column) + '</th>\n')
        f.write(
            """
                        </tr>
                        </thead>
                        <!--rows-->
                        <tbody>
                        """
        )
        for index in transitionMatrix.index:
            f.write('<tr>\n')
            f.write('<td><span object_oriented="labels">' + str(index) + '</span></td>\n')
            for column in transitionMatrix.columns:
                f.write('<td>' + str(transitionMatrix.loc[index, column]) + '</td>\n')
            f.write('</tr>\n')
        f.write(
            """
                        </tbody>
                        </table>
                        </body>
                        </html>
                        """
        )


def transition_probability_matrix(dataFrameList, mode='parallel'):
    """
    series considers transitions from row to row for each column, regardless of index.
    parallel considers transitions from column to column for each index.
    dataFrame: list of pandas DataFrames
    mode: 'series' or 'parallel'
    return: pandas DataFrame
    """
    transitionProbabilityMatrix = dh.normalize_rows(transition_matrix(dataFrameList, mode=mode), rows=True)
    return transitionProbabilityMatrix



def get_corr_cov(df):
    return df.corr(), df.cov()


def count_vector(dataFrameList):
    data = dh.merge_data_list(dataFrameList)
    labels = dh.get_uniques(data)
    countVector = {label: 0 for label in labels}
    for dataFrame in dataFrameList:
        for i, row in enumerate(dataFrame.iterrows()):
            for j, column in enumerate(dataFrame.columns):
                a = dataFrame.iloc[i, j]
                countVector[a] += 1
    return countVector


def symmetrical_matrix(dataFrameList):
    """
    dataFrame: pandas DataFrame
    return: pandas DataFrame
    """
    data = dh.merge_data_list(dataFrameList)
    labels = dh.get_uniques(data)
    symmetricalLabelMatrix = pd.DataFrame(columns=labels, index=labels)
    symmetricalLabelMatrix.fillna(0, inplace=True)
    return symmetricalLabelMatrix


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
    if noise:
        V[-1] += random.randint(-1000, 1000) / 1000
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


###########################################################################################################
class Cube:
    def __init__(self, app, dimension=1, centroid=maths.origin, mass=1, randomAV=False):
        """
        r=right, l=left, f=front, b=back, u=up, d=down
        """
        super(Cube, self).__init__(app, '', mass, dimension)
        self.app = app
        self.dimension = dimension
        self.centroid = centroid
        self.mass = mass
        self._angularVelocity = Quaternion(0, 0, 0, 0)
        self.app.plotConfig.scale = 1
        r = self.dimension / self.app.plotConfig.scale
        self.faces = {
            'up':   ['lfu', 'rfu', 'rbu', 'lbu', 'lfu', ], 'down': ['lfd', 'rfd', 'rbd', 'lbd', 'lfd', ],
            'left': ['lfu', 'lfd', 'lbd', 'lbu', 'lfu', ], 'right': ['rfu', 'rfd', 'rbd', 'rbu', 'rfu', ],
            'back': ['lbu', 'rbu', 'rbd', 'lbd', 'lbu', ], 'front': ['lfu', 'rfu', 'rfd', 'lfd', 'lfu', ],
        }
        x = dimension / 2
        y = dimension / 2
        z = dimension / 2
        if randomAV:
            self._angularVelocity = maths.Quaternion(0, 'r', 'r', 'r', )
        self._vertices = {
            'rfu': maths.Coordinate(x, y, z), 'rbu': maths.Coordinate(x, -y, z),
            'rfd': maths.Coordinate(x, y, -z), 'rbd': maths.Coordinate(x, -y, -z),
            'lfu': maths.Coordinate(-x, y, z), 'lbu': maths.Coordinate(-x, -y, z),
            'lfd': maths.Coordinate(-x, y, -z), 'lbd': maths.Coordinate(-x, -y, -z),
        }
        self.quiverP = [[r, 0, 0], [0, r, 0], [0, 0, r], ]
        self.quiverN = [[-r, 0, 0], [0, -r, 0], [0, 0, -r], ]

    def __init__(self, app, dimension=1, centroid=maths.origin, mass=1, randomAV=False):
        """
        r=right, l=left, f=front, b=back, u=up, d=down
        """
        super(Cube, self).__init__(
            app=app, name='', mass=mass, dimension=dimension, centroid=centroid,
            randomAV=randomAV, )
        self.app = app
        self.app.plotConfig.scale = 1
        self.faces = {
            'up':   ['lfu', 'rfu', 'rbu', 'lbu', 'lfu', ], 'down': ['lfd', 'rfd', 'rbd', 'lbd', 'lfd', ],
            'left': ['lfu', 'lfd', 'lbd', 'lbu', 'lfu', ], 'right': ['rfu', 'rfd', 'rbd', 'rbu', 'rfu', ],
            'back': ['lbu', 'rbu', 'rbd', 'lbd', 'lbu', ], 'front': ['lfu', 'rfu', 'rfd', 'lfd', 'lfu', ],
        }
        x = dimension / 2
        y = dimension / 2
        z = dimension / 2
        self._vertices = {
            'rfu': maths.Coordinate(x, y, z), 'rbu': maths.Coordinate(x, -y, z),
            'rfd': maths.Coordinate(x, y, -z), 'rbd': maths.Coordinate(x, -y, -z),
            'lfu': maths.Coordinate(-x, y, z), 'lbu': maths.Coordinate(-x, -y, z),
            'lfd': maths.Coordinate(-x, y, -z), 'lbd': maths.Coordinate(-x, -y, -z),
        }

    def set_avc(self, x, y, z):
        self._angularVelocity.x = x
        self._angularVelocity.y = y
        self._angularVelocity.z = z

    @property
    def avc(self):
        """
        Angular Velocity Components
        return: list of float - [x, y, z]
        """
        return self._angularVelocity.x, self._angularVelocity.y, self._angularVelocity.z

    @property
    def inertia(self):
        return (1 / 6) * self.mass * (self.dimension ** 2)

    @property
    def vertices(self):
        return [self._vertices[v] for v in self._vertices]

    def wf(self, face):
        """
        Wire Frame
        face: str
        return:
        """
        v = self._vertices
        w = [v[point] for point in self.faces[face]]
        return np.array([c.x for c in w]), np.array([c.y for c in w]), np.array([c.z for c in w])

    @property
    def cubeQuiverVector(self):
        return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], self.quiverP[0], self.quiverP[1], self.quiverP[2], ])

    def rotate(self):
        ox, oy, oz = self.centroid
        aX, aY, aZ = self.avc
        for name, prop in [('_vertices', self._vertices)]:
            results = {}
            for k in prop:
                c = prop[k]
                px, py, pz = c.as_list()
                qx, qy = rotate_point(aZ, ox, oy, px, py)  # rotate about z-axis
                qy, qz = rotate_point(aX, oy, oz, qy, pz)  # rotate about x-axis
                qx, qz = rotate_point(aY, ox, oz, qx, qz)  # rotate about y-axis
                results[k] = Coordinate(qx, qy, qz)
            setattr(self, name, results)
        for name, prop in [('thrusters', self.thrusters)]:
            for k in prop:
                c = prop[k].location
                px, py, pz = c.as_list()
                qx, qy = rotate_point(aZ, ox, oy, px, py)  # rotate about z-axis
                qy, qz = rotate_point(aX, oy, oz, qy, pz)  # rotate about x-axis
                qx, qz = rotate_point(aY, ox, oz, qx, qz)  # rotate about y-axis
                prop[k].location = Coordinate(qx, qy, qz)
        for i, k in enumerate(self.quiverP):
            px, py, pz = k
            qx, qy = rotate_point(-aZ, ox, oy, px, py)  # rotate about z-axis
            qy, qz = rotate_point(-aX, oy, oz, qy, pz)  # rotate about x-axis
            qx, qz = rotate_point(-aY, ox, oz, qx, qz)  # rotate about y-axis
            self.quiverP[i] = [qx, qy, qz]
        if self.activeThruster != None:
            self.activeThrusterCounter += 1
            if self.activeThrusterCounter > self.activeThrusterCounterMaximum:
                self.activeThrusterCounter = 0
                self.activeThruster = None

    @property
    def vertices(self, *args, **kwargs):
        return [self._vertices[v] for v in self._vertices]

    def wf(self, face):
        """
        Wire Frame
        face: str
        return:
        """
        v = self._vertices
        w = [v[point] for point in self.faces[face]]
        return np.array([c.x for c in w]), np.array([c.y for c in w]), np.array([c.z for c in w])

    @property
    def verts(self, *args, **kwargs):
        rfu = self._vertices['rfu'].as_list()
        lfu = self._vertices['lfu'].as_list()
        rbu = self._vertices['rbu'].as_list()
        lbu = self._vertices['lbu'].as_list()
        rfd = self._vertices['rfd'].as_list()
        lfd = self._vertices['lfd'].as_list()
        rbd = self._vertices['rbd'].as_list()
        lbd = self._vertices['lbd'].as_list()
        v = [[lfu, rfu, rbu, lbu], [lfd, rfd, rbd, lbd], [rfu, rfd, rbd, rbu], [lfu, lfd, lbd, lbu],
             [lfu, rfu, rfd, lfd], [lbu, rbu, rbd, lbd]]
        return v

    def __init__(self, parent, dimension=1, centroid=origin, mass=1, randomAV=False):
        """
        r=right, l=left, f=front, b=back, u=up, d=down
        """
        self.parent = parent
        self.dimension = dimension
        self.centroid = centroid
        self.mass = mass
        self._angularVelocity = Quaternion(0, 0, 0, 0)
        r = self.dimension / self.parent.configuration.scale
        self.faces = {
            'up':   ['lfu', 'rfu', 'rbu', 'lbu', 'lfu', ], 'down': ['lfd', 'rfd', 'rbd', 'lbd', 'lfd', ],
            'left': ['lfu', 'lfd', 'lbd', 'lbu', 'lfu', ], 'right': ['rfu', 'rfd', 'rbd', 'rbu', 'rfu', ],
            'back': ['lbu', 'rbu', 'rbd', 'lbd', 'lbu', ], 'front': ['lfu', 'rfu', 'rfd', 'lfd', 'lfu', ],
        }
        x = dimension / 2
        y = dimension / 2
        z = dimension / 2

        if randomAV:
            self._angularVelocity = Quaternion(0, 'r', 'r', 'r', )

        self._vertices = {
            'rfu': Coordinate(x, y, z), 'rbu': Coordinate(x, -y, z), 'rfd': Coordinate(x, y, -z),
            'rbd': Coordinate(x, -y, -z), 'lfu': Coordinate(-x, y, z), 'lbu': Coordinate(-x, -y, z),
            'lfd': Coordinate(-x, y, -z), 'lbd': Coordinate(-x, -y, -z),
        }
        self.quiverP = [[r, 0, 0], [0, r, 0], [0, 0, r], ]
        self.quiverN = [[-r, 0, 0], [0, -r, 0], [0, 0, -r], ]

    @property
    def inertia(self):
        return (1 / 6) * self.mass * (self.dimension ** 2)

    @property
    def vertices(self):
        return [self._vertices[v] for v in self._vertices]

    def wf(self, face):
        """
        Wire Frame
        face: str
        return:
        """
        v = self._vertices
        w = ([v[point] for point in self.faces[face]])
        return [c.x for c in w], [c.y for c in w], [c.z for c in w]

    @property
    def avc(self):
        """
        Angular Velocity Components
        return: list of float - [x, y, z]
        """
        return self._angularVelocity.x, self._angularVelocity.y, self._angularVelocity.z

    def set_avc(self, x, y, z):
        self._angularVelocity.x = x
        self._angularVelocity.y = y
        self._angularVelocity.z = z

    def rotate(self):
        ox, oy, oz = self.centroid
        aX, aY, aZ = self.avc
        for name, prop in [('_vertices', self._vertices)]:
            results = {}
            for k in prop:
                c = prop[k]
                px, py, pz = c.as_list()
                qx, qy = rotate_point(aZ, ox, oy, px, py)  # rotate about z-axis
                qy, qz = rotate_point(aX, oy, oz, qy, pz)  # rotate about x-axis
                qx, qz = rotate_point(aY, ox, oz, qx, qz)  # rotate about y-axis
                results[k] = Coordinate(qx, qy, qz)
            setattr(self, name, results)
        for name, prop in [('thrusters', self.thrusters)]:
            for k in prop:
                c = prop[k].location
                px, py, pz = c.as_list()
                qx, qy = rotate_point(aZ, ox, oy, px, py)  # rotate about z-axis
                qy, qz = rotate_point(aX, oy, oz, qy, pz)  # rotate about x-axis
                qx, qz = rotate_point(aY, ox, oz, qx, qz)  # rotate about y-axis
                prop[k].location = Coordinate(qx, qy, qz)

        for i, k in enumerate(self.quiverP):
            px, py, pz = k
            qx, qy = rotate_point(-aZ, ox, oy, px, py)  # rotate about z-axis
            qy, qz = rotate_point(-aX, oy, oz, qy, pz)  # rotate about x-axis
            qx, qz = rotate_point(-aY, ox, oz, qx, qz)  # rotate about y-axis
            self.quiverP[i] = [qx, qy, qz]
        if self.activeThruster != None:
            self.activeThrusterCounter += 1
            if self.activeThrusterCounter > self.activeThrusterCounterMaximum:
                self.activeThrusterCounter = 0
                self.activeThruster = None

    @property
    def cubeQuiverVector(self):
        return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], self.quiverP[0], self.quiverP[1], self.quiverP[2], ])


# This object is a randomly generated initial state.
class Generate_New_Random_Initial_State:
    def __init__(self, size, significant_figures=3):
        self.size = size
        self.significant_figures = significant_figures
        self.initial_state = []
        self.total, self.error = 0, 0
        for i in range(self.size):
            if self.total < 1:
                self.p = self.get_Random_Probability(self.size, self.total)
                self.initial_state.append(round((self.p), significant_figures))

                self.total += self.initial_state[i]
            else:
                self.initial_state.append(0)
        if self.total != 1:
            self.error = 1 - self.total
            # ---start---#       1
            for i in range(self.size):
                self.initial_state[i] = round(
                    self.initial_state[i] + self.error *
                    self.initial_state[i] / self.total,
                    significant_figures
                )
            print(self.initial_state)

    def get_Random_Probability(self, size, total):
        self.size = size
        self.total = total
        self.p = 0
        for m in range(self.size):
            x = random.uniform(0, 0.5) + random.uniform(0, 0.5)
            while (random.uniform(0, 1) < x):
                pass
            if (m < self.size) and (random.uniform(0, 1) < 0.6):
                self.p += random.uniform(0, 1 - random.uniform(0.75, 1))
            elif (random.uniform(0, 1) > 0.5):
                self.p += 0
            else:
                self.p += random.uniform(0, 1 - self.total)
        self.p /= self.size
        return self.p

    def get_New_Random_Initial_State(self):
        return self.initial_state

    def __init__(self, size, significant_figures=3):
        self.size = size
        self.significant_figures = significant_figures
        self.initial_state = []
        self.total, self.error = 0, 0
        # ---start----#      1
        for i in range(self.size):
            if self.total < 1:
                self.p = self.get_Random_Probability(self.size, self.total)
                self.initial_state.append(round((self.p), significant_figures))
                self.total += self.initial_state[i]
            else:
                self.initial_state.append(0)
        # ---end---#         1
        if self.total != 1:
            self.error = 1 - self.total
            # ---start---#       1
            for i in range(self.size):
                self.initial_state[i] = round(
                    self.initial_state[i] + self.error * self.initial_state[i] / self.total,
                    significant_figures
                )
            # ---end---#         1
            print(self.initial_state)

    def get_Random_Probability(self, size, total):
        self.size = size
        self.total = total
        self.p = 0
        # ---start---#       2
        for m in range(self.size):
            x = random.uniform(0, 0.5) + random.uniform(0, 0.5)
            while (random.uniform(0, 1) < x):
                pass
            if (m < self.size) and (random.uniform(0, 1) < 0.6):
                self.p += random.uniform(0, 1 - random.uniform(0.75, 1))
            elif (random.uniform(0, 1) > 0.5):
                self.p += 0
            else:
                self.p += random.uniform(0, 1 - self.total)
        # ---end---#         2
        self.p /= self.size
        return self.p

    def get_New_Random_Initial_State(self):
        return self.initial_state


class Model:
    def __init__(self, item, files, clean=True, iterations=1, initialState=1, states=(0, 1)):
        self.initialState = np.transpose((pd.Series(initialState, index=states)))
        self.data, self.yrs = process_data(column=item, dir=files)

    def __init__(self, parent):
        self.parent = parent  # parent is Markov Chain Menu
        self.i_sv = parent.i_sv  # Initial state vector dictionary
        self.iterations = 24  # The number of iterations that the Markov Chain will run
        self.sig_figs = 30  # The number of significant figures the program will use in calculations
        self.size = len(self.i_sv)  # The size of the initial state vector
        self.item = self.parent.item.get()
        path = self.parent.input_fp_list[0]
        print(path)
        file = open(path, newline='')
        reader = csv.reader(file)
        header = next(reader)  # first line is the header
        data = []
        for row in reader:
            print(row)

        #    def read_the_csv(self):
        #        # print([item for item in self.entryList if item!='YEAR' and self.entryList[item].get()!=''])
        #       # Reading the csv
        #        for y_4, y_2 in zip(self.y_4, self.y_2):
        #            self.data.append(
        #                pd.read_csv(path + y_4 + '\AK' + y_2 + '.txt',
        #                            usecols=['STRUCTURE_NUMBER_008'] + [item for item in self.entryList if
        ###                                                                item != 'YEAR' and self.entryList[
        #                                                                  item].get() != ''],
        ###                          na_values=self.na_values,
        #                        dtype={**{'STRUCTURE_NUMBER_008': str},
        #                               **{item: str for item in self.entryList if item != 'YEAR' and
        #                                  self.entryList[item].get() != ''}},
        #                        encoding='latin1'))

        # TODO FIX
        # def join_all_results(self):
        # Joining all results
        #    self.data2 = pd.concat([d['CULVERT_COND_062'] for d in self.data], axis=1,
        #                          keys=[str(i) for i in self.years])
        #  self.data2 = pd.DataFrame(self.data2)

        #        def clear_incomplete_data(self):
        # Clearing incomplete data
        #           for i in self.years:
        #              self.data2 = (self.data2[np.isfinite(self.data2[str(i)].astype(float))])

        #     def plot_the_results(self):
        # Plot the results
        #        for i in range(len(self.data2)):
        #           if float(self.data2.iloc[i][0]) == 5:
        #              self.data2.iloc[i].astype(float).plot0()
        # for k,column in enumerate(self.data2):
        # if k ==0:
        # self.data2[column].astype(float).hist(bins=10,histtype='step')
        #     plt.show()
        # illegal_drugs=[self.join_all_results,self.clear_incomplete_data,self.plot_the_results]

        """self.tm_object = TransitionMatrix.TransitionMatrix(self, np.arange(1992,2017))
        # The matrix of probability of transitioning between states (dictionary of dictionaries)
        self.tm = self.tm_object.get_tm()  # TRANSITION MATRIX

        # List of dictionaries
        # Dictionary = year(independent variable), Value = dependent variable
        # TODO convert self.pv_list to numpy array of numpy arrays
        self.pv_list = [self.i_sv]
        self.insert_pv()
        self.transpose()
        self.monte_carlo()

    def insert_pv(self):
        # This for loop inserts probability vector dictionaries into the vector list.
        for m in range(self.iterations):
            self.new_pv = {}
            self.populate_pv(m)
        self.results = []

    def populate_pv(self, m):
        for state_1 in (self.i_sv):
            self.new_value = 0  # Placeholder initial value for the items to be inserted into the empty dictionary
            self.markov_chain(m, state_1)
            # Once the probability vector is filled, it will be inserted to the vector list
        self.pv_list.append(self.new_pv)  # Inserting the probability vector to the vector list

    def markov_chain(self, m, state_1):
        # Initially, this for loop will multiply the initial state vector and the transition matrix.
        # Then, it will multiply the result by the transition matrix.
        # This process is repeated until the outputs fill the vector list to an equivalent size.
        for state_2 in (self.i_sv):
            self.new_value += self.pv_list[m][state_2] * self.tm[state_2][state_1]  # MARKOV CHAIN
        self.new_pv[state_1] = (round(self.new_value, self.sig_figs))  # Inserting the result to the probability vector

    def transpose(self):
        # TRANSPOSING MATRIX
        self.legend_labels = []
        for j, k in enumerate(self.pv_list[0]):
            self.results.append([])
            self.legend_labels.append(str(k))
            self.results[j] = [self.pv_list[i][k] for i in range(len(self.pv_list))]

    def monte_carlo(self):
        self.sub_model = MonteCarlo2.MonteCarlo2(self, self.legend_labels)
        # print(self.probability_vector_list)"""


class MarkovChain(Model):
    """
    Performs a Markov Chain Process. Requires an initial state and a transition matrix to function.
    Number of iterations and significant figures to use may be specified but are not required
    """
    name = "Markov Chain "

    def run(self):
        self.countMatrix = Model.count_matrix(self.data, pd.DataFrame(index=self.vS, columns=self.vS))
        self.matrix = normalize_rows(self.countMatrix)  # Transition Matrix
        self.pdf = MarkovChain.markov_chain(self.yrs, self.iS, self.matrix)
        self.sampler = MonteCarlo.MonteCarlo(self)
        self.sampler.run()

    @staticmethod
    def markov_chain(columns, initial, matrix):
        """
        initial (initial state) is multiplied by matrix (transition matrix)
        as many times as the number of columns
        """
        markovChain = [initial]
        for i in range(len(columns) - 1):
            markovChain.append(pd.Series(markovChain[i].dot(matrix)))
        return concat_data(dataFrame=markovChain, columns=columns)

    def __init__(self, MarkovChainMenu, iterations=24, significant_figures=30):
        self.iterations = iterations  # The number of iterations that the Markov Chain will run
        self.significant_figures = significant_figures  # The number of significant figures the program will use in calculations
        self.Menu = MarkovChainMenu
        self.size = len(self.Menu.initial_state_vector)  # The size of the initial state vector
        # A list which will hold dictionaries.
        # Each dictionary in this list represents a year (independent variable)
        # Each value in the dictionary represents an dependent variable
        self.probability_vector_list = [self.Menu.initial_state_vector]
        # This for loop generates new, unique, empty probability vector dictionaries to be inserted into the vector list.
        # ---start---#       1
        for m in range(self.iterations):
            new_probability_vector_object = VectorGenerator.VectorGenerator()
            new_probability_vector = new_probability_vector_object.get_New_Dict_Vector()  # A new, unique, empty dictionary
            # ---start---#       1
            for state_1 in (self.Menu.initial_state_vector):
                self.new_value = 0  # Placeholder initial value for the items to be inserted into the empty dictionary
                # Initially, this for loop will multiply the initial state vector and the transition matrix.
                # Then, it will multiply the result by the transition matrix.
                # This process is repeated until the outputs fill the vector list to an equivalent size.
                # ---start---#       2
                for state_2 in (self.Menu.initial_state_vector):
                    self.new_value += self.probability_vector_list[m][state_2] * self.Menu.transition_matrix[state_2][
                        state_1]  # MARKOV CHAIN
                # ---end---#         2
                new_probability_vector[state_1] = (round(
                    self.new_value,
                    self.significant_figures
                ))  # Inserting the result to the probability vector  # Once the probability vector is filled, it will be inserted to the vector list
            # ---end---#         1
            self.probability_vector_list.append(
                new_probability_vector
            )  # Inserting the probability vector to the vector list
        # ---end---#         1
        self.results = []
        # TRANSPOSING MATRIX
        legend_labels = []
        for j, value in enumerate(self.probability_vector_list[0]):
            self.results.append([])
            legend_labels.append(str(value))
            self.results[j] = [self.probability_vector_list[i][value] for i in range(len(self.probability_vector_list))]
        self.sub_model = MonteCarlo.MonteCarlo(self, legend_labels)  # print(self.probability_vector_list)

    def __init__(self, MarkovChainMenu, initial_state, transition_matrix, iterations=24, significant_figures=30):
        self.size = initial_state.__len__()  # The size of the initial state vector
        self.iterations = iterations  # The number of iterations that the Markov Chain will run
        self.initial_state = initial_state  # The initial position of the probabilities (dictionary)
        self.transition_matrix = transition_matrix  # The matrix of probability of transitioning between states (dictionary of dictionaries)
        self.significant_figures = significant_figures  # The number of significant figures the program will use in calculations
        self.MarkovChainMenu = MarkovChainMenu
        # A list which will hold dictionaries.
        # Each dictionary in this list represents a year (independent variable)
        # Each value in the dictionary represents an dependent variable
        self.probability_vector_list = []
        self.probability_vector_list.append(
            initial_state
        )  # The initial state vector dictionary is inserted at the first index
        # This for loop generates new, unique, empty probability vector dictionaries to be inserted into the vector list.
        # ---start---#       1
        for m in range(self.iterations):
            new_probability_vector_object = VectorGenerator.VectorGenerator()
            new_probability_vector = new_probability_vector_object.get_New_Dict_Vector()  # A new, unique, empty dictionary

            # ---start---#       1
            for state_1 in (self.initial_state):
                self.new_value = 0  # Placeholder initial value for the items to be inserted into the empty dictionary

                # Initially, this for loop will multiply the initial state vector and the transition matrix.
                # Then, it will multiply the result by the transition matrix.
                # This process is repeated until the outputs fill the vector list to an equivalent size.
                # ---start---#       2
                for state_2 in (self.initial_state):
                    self.new_value += self.probability_vector_list[m][state_2] * self.transition_matrix[state_2][
                        state_1]  # MARKOV CHAIN

                # ---end---#         2

                new_probability_vector[state_1] = (round(
                    self.new_value,
                    self.significant_figures
                ))  # Inserting the result to the probability vector  # Once the probability vector is filled, it will be inserted to the vector list  # TEST TEST TEST TEST TEST TEST TEST

            # ---end---#         1
            self.probability_vector_list.append(
                new_probability_vector
            )  # Inserting the probability vector to the vector list  # total = 0  # for state in new_probability_vector:  # total += new_probability_vector[state]  # print(total)
        # ---end---#         1
        # monteCarlo = MonteCarlo2.MonteCarlo2(self.probability_vector_list)
        self.markov_chain_results = []
        # TRANSPOSING MATRIX
        j = 0
        legend_labels = []
        for value in self.probability_vector_list[0]:
            self.markov_chain_results.append([])
            legend_labels.append(str(value))
            for i in range(self.probability_vector_list.__len__()):
                self.markov_chain_results[j].append(self.probability_vector_list[i][value])
            j += 1
        monteCarlo2 = MonteCarlo2.MonteCarlo2(
            self, self.transition_matrix, self.markov_chain_results,
            legend_labels
        )  # print(self.probability_vector_list)  # (self.transition_matrix)  # self.Plot(self.probability_vector_list)  # Calling the Plot function

    # This function transposes the vector list, which is a matrix.
    # Once the data output by the Markov Chain is prepared by this function, it is plotted.
    # The plotter requires a matrix of input data.
    # This matrix is built as a list of lists.
    # Each list corresponds to a set of y values over time.
    # The number of y values is equivalent to the number of x values required.
    # Optional entries are step size, a list of labels for the legend of the plot0, a starting year, axis labels, and a title.
    """def Plot(self, input_data, step_size=1, legend_labels_list=[], starting_year=0, x_axis_label='Time (Years)',
                 y_axis_label='Probability (%)', title='Probability of Item Value'):
        self.input_data, self.step_size, self.legend_labels_list, self.starting_year, self.x_axis_label, self.y_axis_label, self.title = input_data, step_size, legend_labels_list, starting_year, x_axis_label, y_axis_label, title
        if legend_labels_list.__len__() == 0:  # If no list of labels is provided, each list will simply be numbered
            for i in (self.input_data[0]):
                legend_labels_list.append(str(i))
        self.x_axis = []
        self.size = 0
        self.line_styles = ['.--', '.--', '.--', '.--']  # The plotter wraps around these line style
        self.x_axis = np.arange(self.starting_year, self.starting_year + self.input_data.__len__(), self.step_size)
        plt.figure(1)
        plt.subplot(211)
        self.y_axis = []
        # TRANSPOSING MATRIX
        j=0
        for value in self.input_data[0]:
            self.y_axis.append([])
            for i in range(self.input_data.__len__()):
                self.y_axis[j].append(self.input_data[i][value])
            j+=1
        # This for loop is plotting all the points in each list in the input_data list.
        for i in range(self.y_axis.__len__()):
            plt.plot0(self.x_axis, self.y_axis[i], self.line_styles[i % self.line_styles.__len__()],
                     label=self.legend_labels_list[i % legend_labels_list.__len__()], linewidth=0.5)
        # Placing the axis labels and the title on the graph2
        plt.xlabel(self.x_axis_label)
        plt.ylabel(self.y_axis_label)
        plt.title(self.title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.5)  # Placing a legend beside the plot0
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # This and the preceding line ensure a full screen display upon rendering
        plt.show()"""


class MonteCarlo(Model):
    name = "Monte Carlo"

    def run(self):
        """cS is the current state"""
        self.model = self.parent
        self.simulation = []
        self.time = len(self.model.data.columns)
        for i in range(self.model.iterations):
            self.simulation.append([])
            cS = self.model.iS
            for t in range(self.time):
                if t != 0:
                    # Multiply current state and transition matrix, resulting in a probability vector
                    pV = cS.dot(self.model.matrix)
                    cS = self.sample(self.model.vS, pV, i, cS)
                else:
                    for state, prob in zip(self.model.vS, cS):
                        if prob == 1:
                            self.simulation[i].append(state)
            self.simulation[i] = pd.Series(self.simulation[i])

    def sample(self, vS, pV, iteration, cS):
        """
        vS is a list of valid state names
        pV is a probability vector
        iteration is the current iteration
        cS is the current state
        returns a state vector
        """
        sum = 0
        # Generate random number
        randomNumber = random.uniform(0, 1)
        # Assign state if randomNumber is within its range
        for state, prob in zip(vS, pV):
            sum += prob
            if (sum >= randomNumber):
                self.simulation[iteration].append(state)
                return np.transpose((pd.Series([1 if s == state else 0 for s in vS], index=vS)))

    def sim_frq(self, focus):
        """
        Count state transitions in data and store the count in matrix (pre-labeled DataFrame objects)
        """
        matrix = Model.count_matrix(
            pd.DataFrame(self.simulation),
            pd.DataFrame(index=self.model.vS, columns=self.model.yrs), self.model.yrs
        )
        return Model.frq(focus, matrix)

    def sim_hst(self, focus):
        if focus == "year":
            return pd.DataFrame(
                self.simulation, index=[x for x in range(len(self.simulation))],
                columns=[y for y in range(self.time)]
            ).T.as_matrix().astype(np.float64)
        elif focus == "state":
            return Model.hst(
                data=pd.DataFrame(self.simulation), matrix={s: [] for s in self.model.vS},
                columns=self.model.yrs
            )

    def sample(self, vS, pV, iteration, cS):
        """
        vS is a list of valid state names
        pV is a probability vector
        iteration is the current iteration
        cS is the current state
        returns a state vector
        """
        sum = 0
        # Generate random number
        randomNumber = random.uniform(0, 1)
        # Assign state if randomNumber is within its range
        for state, prob in zip(vS, pV):
            sum += prob
            if (sum >= randomNumber):
                self.simulation[iteration].append(state)
                return np.transpose((pd.Series([1 if s == state else 0 for s in vS], index=vS)))

    def __init__(self, probability_vector_list):
        self.iterations = 5
        self.monte_carlo_simulation = []
        self.probability_vector_list = probability_vector_list
        for iteration in range(self.iterations):
            maximum = 100
            self.monte_carlo_simulation.append([])
            retry = True
            while retry:
                for dict in self.probability_vector_list:
                    sum = 0
                    random_number = random.uniform(0, 1)
                    for state in dict:
                        sum += dict[state]
                        if (sum - random_number) >= 0:
                            try:
                                s = float(state)
                                if s > maximum:
                                    retry = True
                                else:
                                    retry = False
                                    self.monte_carlo_simulation[iteration].append(s)
                                    maximum = s
                                    # print(self.monte_carlo_simulation)
                                    break
                            except Exception as e:
                                print(
                                    e
                                )  # distribution_list.append(state)  # print(distribution_list.__len__())  # Generate a random number mapping to the distribution list, selecting a state to transition to  # random_number = int((random.randrange(0,100)))  # self.monte_carlo_simulation[iteration].append(distribution_list[random_number])
        self.Plot(self.monte_carlo_simulation)
        plt.show()

    # The plotter requires a matrix of input data.
    # This matrix is built as a list of lists.
    # Each list corresponds to a set of y values over time.
    # The number of y values is equivalent to the number of x values required.
    # Optional entries are step size, a list of labels for the legend of the plot, a starting year, axis labels, and a title.
    def Plot(
        self, input_data, step_size=1, starting_year=0, x_axis_label='Time (Years)', y_axis_label='State',
        title='Monte Carlo Simulation of Item Value'
    ):
        self.y_axis, self.step_size, self.starting_year, self.x_axis_label, self.y_axis_label, self.title = input_data, step_size, starting_year, x_axis_label, y_axis_label, title

        self.size = 0
        self.plot_styles = ['.--', '*--', 'x--', '^--']  # The plotter wraps around these line styles

        plt.figure(1)
        plt.subplot(211)

        # This for loop is plotting all the points in each list in the input_data list.
        for i in range(self.y_axis.__len__()):
            self.x_axis = []
            self.x_axis = np.arange(self.starting_year, self.starting_year + self.y_axis[0].__len__(), self.step_size)
            plt.plot(self.x_axis, self.y_axis[i], self.plot_styles[i % self.plot_styles.__len__()], linewidth=0.5)

        # Placing the axis labels and the title on the graph
        plt.xlabel(self.x_axis_label)
        plt.ylabel(self.y_axis_label)
        plt.title(self.title)

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # This and the preceding line ensure a full screen display upon rendering

        # PLOT MENU
        # MarkovChainResultsMenu.MarkovChainResultsMenu(self)

        """p = Plotter.Plotter()

        # Plot the Markov Chain results (Probability vs Year)
        p.Plot(self.MarkovChain.Menu.figure, self.MarkovChain.results, 211,
               x_axis_label='Time (Years)', y_axis_label='Probability of State (%)', title='Markov Chain',
               legend_labels_list=legend_labels)

        # Plot the Monte Carlo simulation results in the same figure (State vs Year)
        p.Plot(self.MarkovChain.Menu.figure, self.monte_carlo_simulation, 212)

        self.MarkovChain.Menu.figure += 1"""

        # TODO

    """
            # Plot Histogram 1 (Frequency vs State per Year)
            for year in range(24):
                histogram_data = []
                for iteration in range(self.iterations):
                    histogram_data.append(int(self.monte_carlo_simulation[iteration][year]))
                p.Histogram(self.MarkovChain.MarkovChainMenu.figure, histogram_data)
            # Plot Histogram 2 (Frequency vs Year per State)
            # Prepare the data for the histograms
            self.bar_chart_3d = p.Generate_Subplot_3D(self.MarkovChain.MarkovChainMenu.figure)
            for year in range(self.iterations):
                pass
                self.histogram_data = []
                self.monte_carlo_simulation[year] = list(map(int, self.monte_carlo_simulation[year]))
                p.Bar_Chart_3D(self.bar_chart_3d, self.monte_carlo_simulation[year], self.monte_carlo_simulation[year], self.iterations)
    p.Show()"""

    def __init__(self, probability_vector_list):
        self.iterations = 5
        self.monte_carlo_simulation = []
        self.probability_vector_list = probability_vector_list
        for iteration in range(self.iterations):
            maximum = 100
            self.monte_carlo_simulation.append([])
            retry = True
            while retry:
                for dict in self.probability_vector_list:
                    sum = 0
                    random_number = random.uniform(0, 1)
                    for state in dict:
                        sum += dict[state]
                        if (sum - random_number) >= 0:
                            try:
                                s = float(state)
                                if s > maximum:
                                    retry = True
                                else:
                                    retry = False
                                    self.monte_carlo_simulation[iteration].append(s)
                                    maximum = s
                                    # print(self.monte_carlo_simulation)
                                    break
                            except Exception as e:
                                print(
                                    e
                                )  # distribution_list.append(state)  # print(distribution_list.__len__())  # Generate a random number mapping to the distribution list, selecting a state to transition to  # random_number = int((random.randrange(0,100)))  # self.monte_carlo_simulation[iteration].append(distribution_list[random_number])
        self.Plot(self.monte_carlo_simulation)
        plt.show()

    def Plot(
        self, input_data, step_size=1, starting_year=0, x_axis_label='Time (Years)', y_axis_label='State',
        title='Monte Carlo Simulation2 of Item Value'
    ):
        self.y_axis, self.step_size, self.starting_year, self.x_axis_label, self.y_axis_label, self.title = input_data, step_size, starting_year, x_axis_label, y_axis_label, title
        self.size = 0
        self.plot_styles = ['.--', '*--', 'x--', '^--']  # The plotter wraps around these line style
        plt.figure(1)
        plt.subplot(211)
        # This for loop is plotting all the points in each list in the input_data list.
        for i in range(self.y_axis.__len__()):
            self.x_axis = []
            self.x_axis = np.arange(self.starting_year, self.starting_year + self.y_axis[0].__len__(), self.step_size)
            plt.plot(self.x_axis, self.y_axis[i], self.plot_styles[i % self.plot_styles.__len__()], linewidth=0.5)
        # Placing the axis labels and the title on the graph2
        plt.xlabel(self.x_axis_label)
        plt.ylabel(self.y_axis_label)
        plt.title(self.title)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # This and the preceding line ensure a full screen display upon rendering
        # PLOT MENU
        # MarkovChainResultsMenu.MarkovChainResultsMenu(self)
        """p = Plotter.Plotter()
        # Plot the Markov Chain results (Probability vs Year)
        p.Plot(self.MarkovChain2.Menu.figure, self.MarkovChain2.results, 211,
               x_axis_label='Time (Years)', y_axis_label='Probability of State (%)', title='Markov Chain',
               legend_labels_list=legend_labels)
        # Plot the Monte Carlo simulation results in the same figure (State vs Year)
        p.Plot(self.MarkovChain2.Menu.figure, self.monte_carlo_simulation, 212)
        self.MarkovChain2.Menu.figure += 1"""  # TODO

    def __init__(self, MarkovChain, legend_labels):
        self.simulation = []
        self.MarkovChain, self.legend_labels = MarkovChain, legend_labels
        self.iterations = int(self.MarkovChain.CubesatAppMenu.iterations.get())
        self.legend_labels = legend_labels
        for iteration in range(self.iterations):
            self.simulation.append([])
            self.current_state = self.MarkovChain.CubesatAppMenu.get_initial_state()
            for i in range(25):
                if i != 0:
                    sum = 0
                    # Multiply current state and transition matrix, resulting in a probability vector
                    self.pv_dict = {}
                    for state1 in self.MarkovChain.CubesatAppMenu.tm:
                        vector_matrix_product = 0
                        for state2 in self.MarkovChain.CubesatAppMenu.tm:
                            vector_matrix_product += self.current_state[state2] * \
                                                     self.MarkovChain.CubesatAppMenu.tm[state2][
                                                         state1]
                        self.pv_dict[state1] = vector_matrix_product
                    # Generate random number and assign a state based on probability vector
                    random_number = random.uniform(0, 1)
                    # print(self.probability_vector_dict)
                    for state in self.pv_dict:
                        sum += self.pv_dict[state]
                        if (sum - random_number) >= 0:
                            self.simulation[iteration].append(state)
                            self.current_state = {s: 1 if s == state else 0 for s in self.current_state}
                            break
                else:
                    self.simulation[iteration] = [state for state in self.current_state if
                                                  self.current_state[state] == 1]

    def get_f_s_1(self, normalize=False):  # Returns Frequency vs State histogram of raw data (per Year)
        self.f_s_1 = []
        for year in range(25):
            self.f_s_1.append([])
            for state in self.MarkovChain.CubesatAppMenu.vs:
                self.f_s_1[year].append(0)
            for iteration in range(self.iterations):
                self.f_s_1[year][int(self.simulation[iteration][year])] += 1
        if normalize:
            for i in range(len(self.f_s_1)):
                row_sum = 0
                for j in range(len(self.f_s_1[i])):
                    row_sum += self.f_s_1[i][j]
                for j in range(len(self.f_s_1[i])):
                    self.f_s_1[i][j] = self.f_s_1[i][j] / row_sum
        return self.f_s_1

    def get_f_s_2(self, normalize=False):  # Returns Frequency vs Year histogram of raw data (per State)
        self.f_s_2 = []
        for state in self.MarkovChain.CubesatAppMenu.vs:
            self.f_s_2.append([])
        for state in self.MarkovChain.CubesatAppMenu.vs:
            for year in range(25):
                self.f_s_2[int(state)].append(0)
        for iteration in range(self.iterations):
            for s, year in enumerate(range(len(self.simulation[iteration]))):
                # print(self.simulation[iteration][year])
                self.f_s_2[abs(int(self.simulation[iteration][year]) - 9)][
                    s] += 1  # Subtracting the arbitrary 5 causes the list to be populated in reverse
        if normalize:
            for i in range(len(self.f_s_2)):
                for j in range(len(self.f_s_2[i])):
                    self.f_s_2[i][j] /= self.iterations
        # print(self.frequency_summary_2)
        return self.f_s_2

    def __init__(self, probability_vector_list):
        self.iterations = 5
        self.monte_carlo_simulation = []
        self.probability_vector_list = probability_vector_list
        for iteration in range(self.iterations):
            maximum = 100
            self.monte_carlo_simulation.append([])
            retry = True
            while retry:
                for dict in self.probability_vector_list:
                    sum = 0
                    random_number = random.uniform(0, 1)
                    for state in dict:
                        sum += dict[state]
                        if (sum - random_number) >= 0:
                            try:
                                s = float(state)
                                if s > maximum:
                                    retry = True
                                else:
                                    retry = False
                                    self.monte_carlo_simulation[iteration].append(s)
                                    maximum = s
                                    # print(self.monte_carlo_simulation)
                                    break
                            except Exception as e:
                                print(e)
                        # distribution_list.append(state)  # print(distribution_list.__len__())
                # Generate a random number mapping to the distribution list, selecting a state to transition to  # random_number = int((random.randrange(0,100)))  # self.monte_carlo_simulation[iteration].append(distribution_list[random_number])
        self.Plot(self.monte_carlo_simulation)
        plt.show()

    # The plotter requires a matrix of input data.
    # This matrix is built as a list of lists.
    # Each list corresponds to a set of y values over time.
    # The number of y values is equivalent to the number of x values required.
    # Optional entries are step size, a list of labels for the legend of the plot0, a starting year, axis labels, and a title.
    def Plot(
        self, input_data, step_size=1, starting_year=0, x_axis_label='Time (Years)', y_axis_label='State',
        title='Monte Carlo Simulation2 of Item Value'
    ):
        self.y_axis, self.step_size, self.starting_year, self.x_axis_label, self.y_axis_label, self.title = input_data, step_size, starting_year, x_axis_label, y_axis_label, title
        self.size = 0
        self.plot_styles = ['.--', '*--', 'x--', '^--']  # The plotter wraps around these line style
        plt.figure(1)
        plt.subplot(211)
        # This for loop is plotting all the points in each list in the input_data list.
        for i in range(self.y_axis.__len__()):
            self.x_axis = []
            self.x_axis = np.arange(self.starting_year, self.starting_year + self.y_axis[0].__len__(), self.step_size)
            plt.plot(self.x_axis, self.y_axis[i], self.plot_styles[i % self.plot_styles.__len__()], linewidth=0.5)
        # Placing the axis labels and the title on the graph2
        plt.xlabel(self.x_axis_label)
        plt.ylabel(self.y_axis_label)
        plt.title(self.title)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # This and the preceding line ensure a full screen display upon rendering

        # PLOT MENU
        # MarkovChainResultsMenu.MarkovChainResultsMenu(self)

        """p = Plotter.Plotter()

        # Plot the Markov Chain results (Probability vs Year)
        p.Plot(self.MarkovChain2.Menu.figure, self.MarkovChain2.results, 211,
               x_axis_label='Time (Years)', y_axis_label='Probability of State (%)', title='Markov Chain',
               legend_labels_list=legend_labels)

        # Plot the Monte Carlo simulation results in the same figure (State vs Year)
        p.Plot(self.MarkovChain2.Menu.figure, self.monte_carlo_simulation, 212)

        self.MarkovChain2.Menu.figure += 1"""

        # TODO

    def __init__(self, probability_vector_list):

        self.iterations = 5
        self.monte_carlo_simulation = []
        self.probability_vector_list = probability_vector_list
        for iteration in range(self.iterations):
            maximum = 100
            self.monte_carlo_simulation.append([])
            retry = True

            while retry:
                for dict in self.probability_vector_list:
                    sum = 0
                    random_number = random.uniform(0, 1)
                    for state in dict:
                        sum += dict[state]
                        if (sum - random_number) >= 0:
                            try:
                                s = float(state)
                                if s > maximum:
                                    retry = True

                                else:
                                    retry = False
                                    self.monte_carlo_simulation[iteration].append(s)
                                    maximum = s
                                    # print(self.monte_carlo_simulation)
                                    break

                            except Exception as e:
                                print(e)

                        # distribution_list.append(state)  # print(distribution_list.__len__())

                # Generate a random number mapping to the distribution list, selecting a state to transition to  # random_number = int((random.randrange(0,100)))  # self.monte_carlo_simulation[iteration].append(distribution_list[random_number])

        self.Plot(self.monte_carlo_simulation)
        plt.show()


class TransitionMatrix:
    def __init__(self, parent, years_to_iterate):
        self.parent = parent
        self.Menu, self.years_to_iterate = parent.parent, years_to_iterate
        self.summary = {}  # Summary of states with respect to time
        self.cm_row_sum = {}  # Each value in this list represents the sum of the row corresponding to this list's index, which will be used to normalize the row
        self.cm = {}
        self.tm = {}  # The normalized count matrix
        self.year = -1
        self.relevant = {
            0: False,
            1: False
        }  # When this variable is true, it searches for values. When it is false, it ignores them.
        self.up_counter = 0
        # This is used to find the index of the appropriate item
        with open(
            self.Menu.parent.util.MASTERPATH + "\BridgeDataQuery\Interface\InterfaceUtilities\Labels\parameterNumbers.txt",
            "r"
        ) as numbers:
            for item_index, line in enumerate(numbers, start=1):
                if line.strip() == self.Menu.item.get():
                    break
        self.start_cm()  # The matrix which holds the count of how many times a transition occurs, which must be counted
        self.populate_tm()  # The transition matrix which will be built

    def __init__(self, input_file_paths, item_of_interest, years_to_iterate, ):
        self.input_file_paths = input_file_paths
        self.item_of_interest = item_of_interest
        self.years_to_iterate = years_to_iterate
        M = MasterPath.MasterPath()
        self.MASTERPATH = M.getMasterPath()
        self.summary_of_states_with_respect_to_time = {}
        self.sum_of_row_of_count_matrix = {}  # Each value in this list represents the sum of the row corresponding to this list's index, which will be used to normalize the row
        self.transition_matrix = {}  # The normalized count matrix
        # This is used to find the index of the appropriate item
        with open(self.MASTERPATH + "\BridgeDataQuery\Interface\Labels\parameterNumbers.txt", "r") as numbers:
            item_of_interest_index = 1
            for line in numbers:
                if line.strip() == self.item_of_interest:
                    break
                item_of_interest_index += 1
        # Now a list of the possible states for the item of interest is formed
        with open(self.MASTERPATH + "\BridgeDataQuery\Interface\Labels\parameterValues.txt", "r") as values:
            counter = 1
            for line in values:
                if counter == item_of_interest_index:
                    self.list_of_possible_states = re.findall(r"\w+?", line)
                    break
                counter += 1
        self.Populate_Count_Matrix()  # The matrix which holds the count of how many times a transition occurs, which must be counted
        self.Populate_Transition_Matrix()  # The transition matrix which will be built

    def __init__(self, MarkovChainMenu, years_to_iterate):
        self.Menu, self.years_to_iterate = MarkovChainMenu, years_to_iterate
        self.summary = {}  # Summary of states with respect to time
        self.cm_row_sum = {}  # Each value in this list represents the sum of the row corresponding to this list's index, which will be used to normalize the row
        self.tm = {}  # The normalized count matrix
        # This is used to find the index of the appropriate item
        with open(
            self.Menu.parent.util.MASTERPATH + "\BridgeDataQuery\Interface\Labels\parameterNumbers.txt",
            "r"
        ) as numbers:
            for item_of_interest_index, line in enumerate(numbers, start=1):
                if line.strip() == self.Menu.item.get():
                    break
        self.populate_count_matrix()  # The matrix which holds the count of how many times a transition occurs, which must be counted
        self.populate_transition_matrix()  # The transition matrix which will be built

    def __init__(self, number_of_states, input_file_paths, item_of_interest, years_to_iterate):
        self.number_of_states = number_of_states  # The number of possible values for the item value
        self.input_file_paths = input_file_paths
        self.item_of_interest = item_of_interest
        self.years_to_iterate = years_to_iterate
        self.summary_of_states_with_respect_to_time = {}
        self.sum_of_row_of_count_matrix = []  # Each value in this list represents the sum of the row corresponding to this list's index
        self.transition_matrix = []
        self.Populate_Count_Matrix()  # The matrix which holds the count of how many times a transition occurs, which must be counted
        self.Populate_Transition_Matrix()  # The transition matrix which will be built

    def create_summary(self):  # Returns the deterioration curve of the raw data
        self.summary_as_list = []
        for list in self.summary:
            if len(self.summary[list]) == 25 and 'N' not in self.summary[list] and self.summary[list][
                0] == self.Menu.i_s.get():
                self.summary_as_list.append(self.summary[list])
            if len(self.summary_as_list) == int(self.Menu.iterations.get()):
                pass

    def negative_first(self, temporary_file):
        for line in temporary_file:  # Iterate through the lines in the file
            self.new_year(line)
            if self.relevant[1]:
                reader = re.findall(
                    r'\'\'|\' \'|\'\w+?\'|\'.+?\'',
                    line
                )  # Separating the line into individual item labels and values
                self.zeroth(line, reader)
                if self.year > len(self.years_to_iterate):
                    break
            else:
                pass

    def zeroth(self, line, reader):
        for omega in range(len(reader) - 1):  # Finding the item of interest for which the matrix will be dedicated to
            self.t1 = reader[omega].strip("'").strip()  # Key
            self.t2 = reader[omega + 1].strip("'").strip()  # Value

            self.check_existence()

            if self.t1 == str(self.Menu.item.get()):  # The item of interest has been found
                self.summary[self.key].append(self.t2)
                break

    def check_existence(self):
        if self.t1 == '5':
            self.key = str(self.t2)  # Bridge name
            if self.t2 not in self.summary:  # Check if the bridge is already represented in the dictionary
                self.summary[self.t2] = []  # Inserting a list into a newly created bridge

    def new_year(self, line):
        if ("##########################################" in line):  # The New Year flag is found
            for r in range(len(self.relevant)):
                self.relevant[r] = False  # Boolean logic reset
            self.year += 1  # New Year
        if ("##########################################" in line) and (
            line.strip("#")[0:4] in self.years_to_iterate):
            self.relevant[0] = True
            self.relevant[1] = False
        elif self.relevant[0]:
            self.relevant[1] = True

    def new_year(self, line):
        if ("##########################################" in line):  # The New Year flag is found
            for r in range(len(self.relevant)):
                self.relevant[r] = False  # Boolean logic reset
            self.year += 1  # New Year
        if ("##########################################" in line) and (
            int(line.strip("#")[0:4]) in self.years_to_iterate):
            self.relevant[0] = True
            self.relevant[1] = False
        elif self.relevant[0]:
            self.relevant[1] = True

    def populate_cm(self):
        # POPULATING THE COUNT MATRIX
        for omega in (self.Menu.vs):  # Creating a square matrix of zeros of length equivalent to number of states
            self.cm[str(omega)] = {}  # A list is inserted into "count_matrix" which is a list of lists
            for gamma in (self.Menu.vs):  # Every position in the inner list is filled with zeros
                self.cm[str(omega)][str(gamma)] = 0

    def count_transitions(self):
        for list in self.summary:
            for counter in range(len(self.summary[list]) - 1):
                try:
                    x1 = str(self.summary[list][counter])
                    x2 = str(self.summary[list][counter + 1])

                    try:  # The matrix only considers values that transition strictly from one number to another number less than or equal to the first
                        float(x1)
                        float(x2)
                        is_number = True
                    except ValueError:
                        is_number = False

                    if is_number:
                        if x2 <= x1:
                            self.cm[x1][x2] += 1
                        elif x2 > x1:
                            self.up_counter += 1
                    else:
                        pass

                except:
                    pass

    def populate_tm(self):

        # POPULATING THE TRANSITION MATRIX
        for row in (self.Menu.vs):  # Iterating through the rows
            self.tm[row] = {}  # Adding a list inside the current index so as to create a matrix
            self.sum_of_row = 0  # Resetting the placeholder sum variable
            self.sum_the_row(row)
            self.insert_pv_into_tm(row)

        self.create_summary()
        return self.tm

    def start_cm(self):
        # EXTRACTING THE SUMMARY OF CHANGES IN STATE
        for file in self.Menu.input_fp_list:  # Iterate through the files in the list of files
            with open(str(file)) as temporary_file:  # Open file
                self.negative_first(temporary_file)
        self.populate_cm()
        self.count_transitions()

    def sum_the_row(self, row):
        for column in (self.Menu.vs):  # Iterating through the columns of the count matrix in the current row (i)
            self.sum_of_row += self.cm[row][column]  # Summing all the values in each column of the current row (i)

        self.cm_row_sum[row] = (self.sum_of_row)  # Store the sum in a list in the (i) index

    def insert_pv_into_tm(self, row):
        for column in (self.Menu.vs):  # Iterating through the columns of the transition matrix in the current row (i)
            # Inserting the probability into the transition matrix at position (i), (j)
            self.tm[row][column] = (self.cm[row][column] / self.cm_row_sum[row]) if self.cm_row_sum[row] != 0 else 0

    def get_tm(self):
        return self.tm

    def get_summary(self):
        return self.summary_as_list

    def get_f_s_1(self, normalize=False):  # Returns Frequency vs State histogram of raw data (per Year)
        self.f_s_1 = []
        for year in range(25):
            self.f_s_1.append([])
            for i in range(len(self.Menu.vs)):
                self.f_s_1[year].append(0)
            for i in range(len(self.summary_as_list)):
                self.f_s_1[year][int(self.summary_as_list[i][year])] += 1
        if normalize:
            for i in range(len(self.f_s_1)):
                row_sum = 0
                for j in range(len(self.f_s_1[i])):
                    row_sum += self.f_s_1[i][j]
                for j in range(len(self.f_s_1[i])):
                    self.f_s_1[i][j] = self.f_s_1[i][j] / row_sum
        return self.f_s_1

    def get_f_s_2(self, normalize=False):  # Returns Frequency vs Year histogram of raw data (per State)
        self.f_s_2 = []
        for i in range(len(self.Menu.vs)):
            self.f_s_2.append([])
        for state in self.Menu.vs:
            for year in range(25):
                self.f_s_2[int(state)].append(0)
        # TODO
        for iteration in range(int(len(self.summary_as_list))):
            for s, year in enumerate(range(len(self.summary_as_list[iteration]))):
                # print(self.simulation[iteration][year])
                self.f_s_2[abs(int(self.summary_as_list[iteration][year]) - 9)][s] += 1
        if normalize:
            for i in range(len(self.f_s_2)):
                for j in range(len(self.f_s_2[i])):
                    self.f_s_2[i][j] /= int(len(self.summary_as_list))
        # print(self.frequency_summary_2)
        return self.f_s_2

    def populate_count_matrix(self):
        self.cm = {}
        year = -1
        relevant = {
            0: False,
            1: False
        }  # When this variable is true, it searches for values. When it is false, it ignores them.
        up_counter = 0
        # EXTRACTING THE SUMMARY OF CHANGES IN STATE
        # ---start---#
        for file in self.Menu.input_fp_list:  # Iterate through the files in the list of files
            with open(str(file)) as temporary_file:  # Open file
                # ---start---#
                for line in temporary_file:  # Iterate through the lines in the file
                    if ("##########################################" in line):  # The New Year flag is found
                        for r in range(len(relevant)):
                            relevant[r] = False  # Boolean logic reset
                        year += 1  # New Year
                    if ("##########################################" in line) and (
                        line.strip("#")[0:4] in self.years_to_iterate):
                        relevant[0] = True
                        relevant[1] = False
                    elif relevant[0]:
                        relevant[1] = True
                    if relevant[1]:
                        reader = re.findall(
                            r'\'\'|\' \'|\'\w+?\'|\'.+?\'',
                            line
                        )  # Separating the line into individual item labels and values
                        # ---start---#
                        for omega in range(
                            len(reader) - 1
                        ):  # Finding the item of interest for which the matrix will be dedicated to
                            t1 = reader[omega].strip("'").strip()  # Key
                            t2 = reader[omega + 1].strip("'").strip()  # Value
                            if t1 == '5':
                                self.key = str(t2)  # Bridge name
                                if t2 not in self.summary:  # Check if the bridge is already represented in the dictionary
                                    self.summary[t2] = []  # Inserting a list into a newly created bridge
                            if t1 == str(self.Menu.item.get()):  # The item of interest has been found
                                self.summary[self.key].append(t2)
                                """try:
                                    self.summary[self.key].append(int(t2))  # The value of the item is added to the summary
                                except:
                                    pass"""
                                break
                        # ---end---#
                        if year > len(self.years_to_iterate):
                            break
                    else:
                        pass  # ---end---#
        # ---end---#
        # POPULATING THE COUNT MATRIX
        for omega in (self.Menu.vs):  # Creating a square matrix of zeros of length equivalent to number of states
            self.cm[str(omega)] = {}  # A list is inserted into "count_matrix" which is a list of lists
            for gamma in (self.Menu.vs):  # Every position in the inner list is filled with zeros
                self.cm[str(omega)][str(gamma)] = 0
        for list in self.summary:
            for counter in range(len(self.summary[list]) - 1):
                try:
                    x1 = str(self.summary[list][counter])
                    x2 = str(self.summary[list][counter + 1])
                    try:  # The matrix only considers values that transition strictly from one number to another number less than or equal to the first
                        float(x1)
                        float(x2)
                        is_number = True
                    except ValueError:
                        is_number = False
                    if is_number:
                        if x2 <= x1:
                            self.cm[x1][x2] += 1
                        elif x2 > x1:
                            up_counter += 1
                    else:
                        pass
                except:
                    pass  # print(up_counter)

    def populate_transition_matrix(self):
        # POPULATING THE TRANSITION MATRIX
        # ---start---#      1
        for row in (self.Menu.vs):  # Iterating through the rows
            self.tm[row] = {}  # Adding a list inside the current index so as to create a matrix
            sum_of_row = 0  # Resetting the placeholder sum variable
            # ---start---#      1
            for column in (self.Menu.vs):  # Iterating through the columns of the count matrix in the current row (i)
                sum_of_row += self.cm[row][column]  # Summing all the values in each column of the current row (i)
            # ---end---#        1
            self.cm_row_sum[row] = (sum_of_row)  # Store the sum in a list in the (i) index
            # ---start---#   2
            for column in (
                self.Menu.vs):  # Iterating through the columns of the transition matrix in the current row (i)
                # Inserting the probability into the transition matrix at position (i), (j)
                self.tm[row][column] = (self.cm[row][column] / self.cm_row_sum[row]) if self.cm_row_sum[
                                                                                            row] != 0 else 0  # ---end---#     2
        # ---end---#        1
        # print(self.count_matrix)
        self.create_summary()
        return self.tm

    def get_transition_matrix(self):
        # print(self.transition_matrix)
        return self.tm

    def get_h_1(self):
        self.h = [[]]
        for a in self.summary_as_list:
            for b in self.summary_as_list[0][a]:
                self.h.append(b)
        return self.h

    def get_f_s_1(self, normalize=False):  # Returns Frequency vs State histogram of raw data (per Year)
        self.f_s_1 = []
        for year in range(25):
            self.f_s_1.append([])
            for state in self.Menu.vs:
                self.f_s_1[year].append(0)
            for i in range(len(self.summary_as_list)):
                self.f_s_1[year][int(self.summary_as_list[i][year])] += 1
        if normalize:
            for i in range(len(self.f_s_1)):
                row_sum = 0
                for j in range(len(self.f_s_1[i])):
                    row_sum += self.f_s_1[i][j]
                for j in range(len(self.f_s_1[i])):
                    self.f_s_1[i][j] = self.f_s_1[i][j] / row_sum
        return self.f_s_1

    def get_f_s_2(self, normalize=False):  # Returns Frequency vs Year histogram of raw data (per State)
        self.f_s_2 = []
        for state in self.Menu.vs:
            self.f_s_2.append([])
        for state in self.Menu.vs:
            for year in range(25):
                self.f_s_2[int(state)].append(0)
        # TODO
        for iteration in range(int(len(self.summary_as_list))):
            for s, year in enumerate(range(len(self.summary_as_list[iteration]))):
                # print(self.simulation[iteration][year])
                self.f_s_2[abs(int(self.summary_as_list[iteration][year]) - 9)][s] += 1
        if normalize:
            for i in range(len(self.f_s_2)):
                for j in range(len(self.f_s_2[i])):
                    self.f_s_2[i][j] /= int(len(self.summary_as_list))
        # print(self.frequency_summary_2)
        return self.f_s_2

    def count_transitions(self):
        for list in self.summary:
            for counter in range(len(self.summary[list]) - 1):
                try:
                    x1 = str(self.summary[list][counter])
                    x2 = str(self.summary[list][counter + 1])

                    try:  # The matrix only considers values that transition strictly from one number to another number less than or equal to the first
                        float(x1)
                        float(x2)
                        is_number = True
                    except ValueError:
                        is_number = False

                    if is_number:
                        if x2 <= x1:
                            self.cm[x1][x2] += 1
                        elif x2 > x1:
                            self.up_counter += 1
                    else:
                        pass

                except:
                    pass

    def sum_the_row(self, row):
        for column in (self.Menu.vs):  # Iterating through the columns of the count matrix in the current row (i)
            self.sum_of_row += self.cm[row][column]  # Summing all the values in each column of the current row (i)
        self.cm_row_sum[row] = (self.sum_of_row)  # Store the sum in a list in the (i) index

    def get_f_s_1(self, normalize=False):  # Returns Frequency vs State histogram of raw data (per Year)
        self.f_s_1 = []
        for year in range(25):
            self.f_s_1.append([])
            for i in range(len(self.Menu.vs)):
                self.f_s_1[year].append(0)
            for i in range(len(self.summary_as_list)):
                self.f_s_1[year][int(self.summary_as_list[i][year])] += 1
        if normalize:
            for i in range(len(self.f_s_1)):
                row_sum = 0
                for j in range(len(self.f_s_1[i])):
                    row_sum += self.f_s_1[i][j]
                for j in range(len(self.f_s_1[i])):
                    self.f_s_1[i][j] = self.f_s_1[i][j] / row_sum
        return self.f_s_1

    def get_f_s_2(self, normalize=False):  # Returns Frequency vs Year histogram of raw data (per State)
        self.f_s_2 = []
        for i in range(len(self.Menu.vs)):
            self.f_s_2.append([])
        for state in self.Menu.vs:
            for year in range(25):
                self.f_s_2[int(state)].append(0)
        # TODO
        for iteration in range(int(len(self.summary_as_list))):
            for s, year in enumerate(range(len(self.summary_as_list[iteration]))):
                # print(self.simulation[iteration][year])
                self.f_s_2[abs(int(self.summary_as_list[iteration][year]) - 9)][s] += 1
        if normalize:
            for i in range(len(self.f_s_2)):
                for j in range(len(self.f_s_2[i])):
                    self.f_s_2[i][j] /= int(len(self.summary_as_list))
        # print(self.frequency_summary_2)
        return self.f_s_2

    def get_Transition_Matrix(self):
        return self.transition_matrix

    ############## T R O U B L E
    def Populate_Count_Matrix(self):
        self.count_matrix = []
        year = 0
        relevant = {
            0: False,
            1: False
        }  # When this variable is true, it searches for values. When it is false, it ignores them.
        for omega in range(self.number_of_states):
            self.count_matrix.append([])
            for gamma in range(self.number_of_states):
                self.count_matrix[omega].append(0)
        # EXTRACTING THE SUMMARY OF CHANGES IN STATE
        # ---start---#
        for file in self.input_file_paths:  # Iterate through the files in the list of files
            with open(str(file)) as temporary_file:  # Open file
                print(temporary_file.read())
                # ---start---#
                for line in temporary_file:  # Iterate through the lines in the file
                    if ("#" in line) and (line.strip("#")[0:4] in self.years_to_iterate):
                        relevant[0] = True
                        relevant[1] = False
                    elif relevant[0]:
                        relevant[1] = True
                    if relevant[1]:
                        reader = re.findall(
                            r'\'\'|\' \'|\'\w+?\'',
                            line
                        )  # Separating the line into individual item labels and values
                        # ---start---#
                        for omega in range(
                            reader.__len__()
                        ):  # Finding the item of interest for which the matrix will be dedicated to
                            if reader[omega] == self.item_of_interest:  # The item of interest has been found
                                self.summary_of_states_with_respect_to_time[self.years_to_iterate[year]] = reader[
                                    omega + 1]  # The value of the item is added to the summary
                                break
                        # ---end---#
                        year += 1
                        for r in range(relevant.__len__()):
                            relevant[r] = False
                        if year > self.years_to_iterate.__len__():
                            return
                    else:
                        pass  # ---end---#
        # ---end---#
        # POPULATING THE COUNT MATRIX
        print(self.summary_of_states_with_respect_to_time)
        temporary_dict = {}
        counter = 0
        for value in self.summary_of_states_with_respect_to_time:
            temporary_dict[counter] = int((self.summary_of_states_with_respect_to_time[value].strip("'")))
            counter += 1
        for i in range(temporary_dict.__len__()):
            try:
                self.count_matrix[temporary_dict[i]][temporary_dict[i + 1]] += 1
            except:
                pass
        print(self.count_matrix)

    def Populate_Count_Matrix(self):
        self.count_matrix = []
        year = 0
        relevant = {
            0: False,
            1: False
        }  # When this variable is true, it searches for values. When it is false, it ignores them.
        for omega in range(self.number_of_states):
            self.count_matrix.append([])
            for gamma in range(self.number_of_states):
                self.count_matrix[omega].append(0)
        # EXTRACTING THE SUMMARY OF CHANGES IN STATE
        for file in self.input_file_paths:  # Iterate through the files in the list of files
            with open(str(file)) as temporary_file:  # Open file
                print(temporary_file.read())
                for line in temporary_file:  # Iterate through the lines in the file
                    if ("#" in line) and (line.strip("#")[0:4] in self.years_to_iterate):
                        relevant[0] = True
                        relevant[1] = False
                    elif relevant[0]:
                        relevant[1] = True
                    if relevant[1]:
                        reader = re.findall(
                            r'\'\'|\' \'|\'\w+?\'',
                            line
                        )  # Separating the line into individual item labels and values
                        for omega in range(
                            reader.__len__()
                        ):  # Finding the item of interest for which the matrix will be dedicated to
                            if reader[omega] == self.item_of_interest:  # The item of interest has been found
                                self.summary_of_states_with_respect_to_time[self.years_to_iterate[year]] = reader[
                                    omega + 1]  # The value of the item is added to the summary
                                break
                        year += 1
                        for r in range(relevant.__len__()):
                            relevant[r] = False
                        if year > self.years_to_iterate.__len__():
                            return
                    else:
                        pass
        # POPULATING THE COUNT MATRIX
        print(self.summary_of_states_with_respect_to_time)
        temporary_dict = {}
        counter = 0
        for value in self.summary_of_states_with_respect_to_time:
            temporary_dict[counter] = int((self.summary_of_states_with_respect_to_time[value].strip("'")))
            counter += 1

        for i in range(temporary_dict.__len__()):
            try:
                self.count_matrix[temporary_dict[i]][temporary_dict[i + 1]] += 1
            except:
                pass

        print(self.count_matrix)

    def Populate_Transition_Matrix(self):
        # POPULATING THE TRANSITION MATRIX
        # ---start---#      1
        for i in range(self.number_of_states):  # Iterating through the rows
            self.transition_matrix.append([])  # Adding a list inside the current index so as to create a matrix
            sum_of_row = 0  # Resetting the placeholder sum variable
            # ---start---#      1
            for k in range(
                self.number_of_states
            ):  # Iterating through the columns of the count matrix in the current row (i)
                sum_of_row += self.count_matrix[i][k]  # Summing all the values in each column of the current row (i)
            # ---end---#        1
            self.sum_of_row_of_count_matrix.append(sum_of_row)  # Store the sum in a list in the (i) index
            # ---start---#   2
            for j in range(
                self.number_of_states
            ):  # Iterating through the columns of the transition matrix in the current row (i)
                # Inserting the probability into the transition matrix at position (i), (j)
                if self.sum_of_row_of_count_matrix[i] != 0:
                    self.transition_matrix[i].append(self.count_matrix[i][j] / self.sum_of_row_of_count_matrix[i])
                else:
                    self.transition_matrix[i].append(0)  # ---end---#     2  # ---end---#        1

    def Populate_Transition_Matrix(self):
        for i in range(self.number_of_states):  # Iterating through the rows
            self.transition_matrix.append([])  # Adding a list inside the current index so as to create a matrix
            sum_of_row = 0  # Resetting the placeholder sum variable
            for k in range(
                self.number_of_states
            ):  # Iterating through the columns of the count matrix in the current row (i)
                sum_of_row += self.count_matrix[i][k]  # Summing all the values in each column of the current row (i)
            self.sum_of_row_of_count_matrix.append(sum_of_row)  # Store the sum in a list in the (i) index
            for j in range(
                self.number_of_states
            ):  # Iterating through the columns of the transition matrix in the current row (i)
                # Inserting the probability into the transition matrix at position (i), (j)
                if self.sum_of_row_of_count_matrix[i] != 0:
                    self.transition_matrix[i].append(self.count_matrix[i][j] / self.sum_of_row_of_count_matrix[i])
                else:
                    self.transition_matrix[i].append(0)

    def Populate_Count_Matrix(self):
        self.count_matrix = {}
        year = -1
        relevant = {
            0: False,
            1: False
        }  # When this variable is true, it searches for values. When it is false, it ignores them.
        up_counter = 0
        # EXTRACTING THE SUMMARY OF CHANGES IN STATE
        # ---start---#
        for file in self.input_file_paths:  # Iterate through the files in the list of files
            with open(str(file)) as temporary_file:  # Open file
                # ---start---#
                for line in temporary_file:  # Iterate through the lines in the file
                    if ("##########################################" in line):  # The New Year flag is found
                        for r in range(relevant.__len__()):
                            relevant[r] = False  # Boolean logic reset
                        year += 1  # New Year
                    if ("##########################################" in line) and (
                        line.strip("#")[0:4] in self.years_to_iterate):
                        relevant[0] = True
                        relevant[1] = False
                    elif relevant[0]:
                        relevant[1] = True
                    if relevant[1]:
                        reader = re.findall(
                            r'\'\'|\' \'|\'\w+?\'|\'.+?\'',
                            line
                        )  # Separating the line into individual item labels and values
                        # ---start---#
                        for omega in range(
                            reader.__len__() - 1
                        ):  # Finding the item of interest for which the matrix will be dedicated to
                            t1 = reader[omega].strip("'").strip()
                            t2 = reader[omega + 1].strip("'").strip()
                            if t1 == '5':
                                self.key = str(t2)
                                if t2 not in self.summary_of_states_with_respect_to_time:
                                    self.summary_of_states_with_respect_to_time[t2] = []
                            if t1 == str(self.item_of_interest):  # The item of interest has been found
                                self.summary_of_states_with_respect_to_time[self.key].append(
                                    t2
                                )  # The value of the item is added to the summary
                                break
                        # ---end---#
                        if year > self.years_to_iterate.__len__():
                            break
                    else:
                        pass  # ---end---#
        # ---end---#
        # POPULATING THE COUNT MATRIX
        for omega in (
            self.list_of_possible_states):  # Creating a square matrix of zeros of length equivalent to number of states
            self.count_matrix[str(omega)] = {}  # A list is inserted into "count_matrix" which is a list of lists
            for gamma in (self.list_of_possible_states):  # Every position in the inner list is filled with zeros
                self.count_matrix[str(omega)][str(gamma)] = 0
        for list in self.summary_of_states_with_respect_to_time:
            for counter in range(self.summary_of_states_with_respect_to_time[list].__len__() - 1):
                try:
                    x1 = str(self.summary_of_states_with_respect_to_time[list][counter])
                    x2 = str(self.summary_of_states_with_respect_to_time[list][counter + 1])
                    try:  # The matrix only considers values that transition strictly from one number to another number less than or equal to the first
                        float(x1)
                        float(x2)
                        is_number = True
                    except ValueError:
                        is_number = False
                    if is_number:
                        if x2 <= x1:
                            self.count_matrix[x1][x2] += 1
                        elif x2 > x1:
                            up_counter += 1
                    else:
                        pass
                except:
                    pass  # print(up_counter)

    def Populate_Transition_Matrix(self):
        # POPULATING THE TRANSITION MATRIX
        # ---start---#      1
        for row in (self.list_of_possible_states):  # Iterating through the rows
            self.transition_matrix[row] = {}  # Adding a list inside the current index so as to create a matrix
            sum_of_row = 0  # Resetting the placeholder sum variable
            # ---start---#      1
            for column in (
                self.list_of_possible_states):  # Iterating through the columns of the count matrix in the current row (i)
                sum_of_row += self.count_matrix[row][
                    column]  # Summing all the values in each column of the current row (i)
            # ---end---#        1
            self.sum_of_row_of_count_matrix[row] = (sum_of_row)  # Store the sum in a list in the (i) index
            # ---start---#   2
            for column in (
                self.list_of_possible_states):  # Iterating through the columns of the transition matrix in the current row (i)
                # Inserting the probability into the transition matrix at position (i), (j)
                if self.sum_of_row_of_count_matrix[row] != 0:
                    self.transition_matrix[row][column] = (
                        self.count_matrix[row][column] / self.sum_of_row_of_count_matrix[row])
                else:
                    self.transition_matrix[row][
                        column] = 0  # ---end---#     2  # ---end---#        1  # print(self.count_matrix)

    def get_List_of_Possible_States(self):
        return self.list_of_possible_states


'''

import random

import numpy as np
import scipy
import scipy.stats
import scipy.signal
from lozoya.time import get_timestamp

np.random.seed(0)


def kalman_filter(x0, z, p0=1, a=1, q=0.5, r=5):
    """
    :param x: state
    :param p: error (covariance)
    :param a: state transition
    :param q: transition error
    :param z: measurement
    :param h: state-to-measurement transformation
    :param y: difference
    :param k: kalman gain
    :param r: measurement error
    :return:
    """
    h = 1
    # prediction
    x1 = a * x0
    p1 = a * p0 * a + q
    # update
    y = np.array([z[i] - h * x1[i] for i in range(len(x1))])
    k = p1 * h / (h * p1 * h + r)
    x1 = np.array([x1[i] + k * y[i] for i in range(len(x1))])
    p1 = (1 - k * h) * p1
    x = x1[-1] + k * y[-1]
    kalmanFiltered = np.append(x1, [x])
    raw = np.append(x0, [x])
    return raw, kalmanFiltered, 1


def notch_filter(y, samplingRate, frequency):
    ya = np.array(y)
    # yFFT = scipy.fftpack.fft(ya)
    # yFFT2 = 1*np.abs(yFFT[0:int(len(y)/1)])/samplingRate
    # yF = scipy.fftpack.ifft(yF)[:-1]
    Q = 5
    w0 = frequency / (samplingRate / 2)
    b, a = scipy.signal.iirnotch(w0, Q)
    yF = scipy.signal.lfilter(b, a, ya)
    return yF


def lowpass_filter(y, samplingRate, frequency):
    ya = np.array(y)
    w0 = frequency / (samplingRate / 2)
    b, a = scipy.signal.butter(1, w0, btype='low')
    yF = scipy.signal.lfilter(b, a, ya)
    return yF


def highpass_filter(y, samplingRate, frequency):
    ya = np.array(y)
    w0 = frequency / (samplingRate / 2)
    b, a = scipy.signal.butter(1, w0, btype='high')
    yF = scipy.signal.lfilter(b, a, ya)
    return yF


def synthetic_data_uniform(size=5):
    return np.random.randint(-1000, 1000, size=(size, 2)).astype('int32')


def synthetic_data_linear(size=5, start=0):
    return np.array([i for i in range(start, size)])


class SignalGenerator:
    def __init__(self, dataType, delimiter, reliability, methods):
        self.dataType = dataType
        self.delimiter = delimiter
        self.reliability = reliability
        self.methods = methods
        self.reliability = 100
        self._min = [0, 0, -1, -9, -9, -99, -99, -99, 1000, 0, 0]
        self._max = [0, 0, 1, 9, 9, 99, 99, 99, 9999, 5000, 360]
        self._max = [0, 0, 1, 9, 9, 99, 99, 99, 9999, 5000, 5000]
        self.rates = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1]
        self.rates = [0, 0, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 1]
        self.rates = [0, 0, 0.1, 0.001, 1, 1, 1, 1, 1, 1, 1]
        self.scales = [1, 1, 360, 1, 1, 1, 20, 10, 1, 1, 1]
        self.scales = [0, 0, 360, 180, 5, 1, 20, 10, 1, 100, 100]
        self.scales = [0, 0, 359.9, 179.9, 5, 1, 20, 10, 1, 100, 100]
        self.g = scipy.stats.exponweib(1, 2)
        self.a = 'abcdefghijklmnopqrstuvwxyz'
        self.i = 0
        self.methods = ['random', 'time', 'linear', 'synthesis', 'sample', 'random', 'synthesis', 'sample', 'random',
                        'synthesis', 'sample']
        self._g = {'l': self.linear, 's': self.synthesis, 'd': self.sample, 'r': self._random, 'q': self.power, }
        self._g = {'l': self.linear, 's': self.synthesis, 'd': self.sample, 'r': self._random, 'q': self.quadratic, }

    def generate_number(self, index, decimals=False):
        try:
            scale = self.scales[index]
            method = self.methods[index]
            rate = self.rates[index]
            if method == 'linear':
                num = (rate * self.i) % scale
            elif method == 'sample':
                num = scale * self.g.rvs((1,))[0]
            elif method == 'synthesis':
                num = scale * np.sin(rate * self.i)
            else:
                num = random.randint(self._min[index], self._max[index]) / scale
            if not decimals:
                return int(num)
            return num
        except Exception as e:
            alertMsg = 'Generate number error: {}'.format(str(e))
            self.parent.transceiverMenu.update_status(alertMsg, 'error')

    def generate_number(self, index, decimals=False):
        scale = self.scales[index]
        method = self.methods.split(self.delimiter)[index]
        rate = self.rates[index]
        num = self._g[method](index, rate, scale)
        if not decimals:
            return int(num)
        return num

    def generate_number(self, index, decimals=False):
        try:
            scale = self.scales[index]
            method = self.app.generatorConfig.methods.split(self.app.dataConfig.delimiter)[index]
            rate = self.rates[index]
            num = self._g[method](index, rate, scale)
            if not decimals:
                return int(num)
            return num
        except Exception as e:
            self.app.transceiverMenu.update_status(*status.generateError, e)

    def linear(self, index, rate, scale, offsetHorizontal, offsetVertical):
        return offsetVertical + (rate * self.i + offsetHorizontal) % scale

    def quadratic(self, index, rate, scale):
        return (1 / scale) * (((rate * self.i) % scale) ** 2)

    def power(self, index, rate, scale, offsetHorizontal, offsetVertical, power):
        return offsetVertical + (1 / scale) * (((rate * self.i + offsetHorizontal) % scale) ** power)

    def _random(self, index, rate, scale, offsetHorizontal, offsetVertical):
        return offsetVertical + random.randint(self._min[index], self._max[index]) / scale

    def sample(self, index, rate, scale, offsetHorizontal, offsetVertical):
        return offsetVertical + scale * self.g.rvs((1,))[0]

    def synthesis(self, index, rate, scale, offsetHorizontal, offsetVertical):
        return offsetVertical + scale * np.sin(rate * self.i + offsetHorizontal)

    def simulate_signal(self, i):
        self.i = i
        dataTypes = self.parent.preferences.dataType.split(',')
        data = ''
        errMsg = 'error!'
        if (random.randint(0, 100) > self.reliability):
            return errMsg
        for _i, dataType in enumerate(dataTypes):
            if dataType == 'd' or dataType == 'f':
                data += str(self.generate_number(_i, True if dataType == 'f' else False))
            elif dataType == 'h':
                data += '0x' + str(random.randint(1000, 9999))
            elif dataType == 's':
                data += random.choice(self.a + self.a.upper())
            elif dataType == 't':
                data += self.parent.dataDude.timestampHMSms
            if _i != len(dataTypes) - 1:
                data += ','
        return data

    def simulate_signal(self, i):
        try:
            self.i = i
            dataTypes = self.app.dataConfig.dataType.split(self.app.dataConfig.delimiter)
            data = ''
            errMsg = 'error!'
            if (random.randint(0, 100) > self.app.generatorConfig.reliability):
                return errMsg
            for _i, dataType in enumerate(dataTypes):
                if dataType == 'd' or dataType == 'f':
                    data += str(self.generate_number(_i, True if dataType == 'f' else False))
                elif dataType == 'h':
                    data += '0x' + str(random.randint(1000, 9999))
                elif dataType == 's':
                    data += random.choice(self.a + self.a.upper())
                elif dataType == 't':
                    data += self.app.dataDude.timestampHMSms
                if _i != len(dataTypes) - 1:
                    data += ','
            return data
        except Exception as e:
            self.app.transceiverMenu.update_status(*status.simulateError, e)

    def simulate_signal(self, i):
        """
        This is a signal generator that simulates the data sent by the cubesat.
        The reliability variable sets the probability of a message being sent
        from the cubesat and received by the controller. The probability that
        the controller will not receive a message is (1 - reliability).
        The data generated will be between the _min and _max variables, inclusive
        of _min and _max. The data will be divided by the scale variable.
        e.g. setting _min=0, _max=100, scale=100 will generate numbers between
             0 and 1 with a precision of 2 decimal places (0.00, 0.01, ... 0.99, 1.00).
        i:
        return:
        """
        self.i = i
        dataTypes = self.dataType.split(self.delimiter)
        data = ''
        errMsg = 'error!'
        if (random.randint(0, 100) > self.reliability):
            return errMsg
        for _i, dataType in enumerate(dataTypes):
            if dataType == 'd' or dataType == 'f':
                data += str(self.generate_number(_i, True if dataType == 'f' else False))
            elif dataType == 'h':
                data += '0x' + str(random.randint(1000, 9999))
            elif dataType == 's':
                data += random.choice(self.a + self.a.upper())
            elif dataType == 't':
                data += get_timestamp()
            if _i != len(dataTypes) - 1:
                data += ','
        return data

    def simulate_signal(self, i):
        try:
            self.i = i
            dataTypes = self.parent.configuration.py.dataType.split(self.parent.configuration.py.delimiter)
            data = ''
            errMsg = 'error!'
            if (random.randint(0, 100) > self.reliability):
                return errMsg
            for _i, dataType in enumerate(dataTypes):
                if dataType == 'd' or dataType == 'f':
                    data += str(self.generate_number(_i, True if dataType == 'f' else False))
                elif dataType == 'h':
                    data += '0x' + str(random.randint(1000, 9999))
                elif dataType == 's':
                    data += random.choice(self.a + self.a.upper())
                elif dataType == 't':
                    data += self.parent.dataDude.timestampHMSms
                if _i != len(dataTypes) - 1:
                    data += ','
            return data
        except Exception as e:
            alertMsg = 'Simulate signal error: {}'.format(str(e))
            self.parent.transceiverMenu.update_status(alertMsg, 'error')


class Transformer:
    def __init__(self):
        self.transforms = {'integrate': self.integrate, 'differentiate': self.differentiate, 'fft': self.fft, }

    def integrate(self, x, y):
        return x, scipy.integrate.simps(y, x, axis=0)

    def differentiate(self, x, y):
        return x, np.gradient(y)

    def fft(self, x, y):
        return x[:len(x) // 2], np.fft.fft(y)
