import collections
import concurrent.futures
import math
import queue
import random
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import scipy.stats
import seaborn as sns
import sympy

import lozoya.configuration
import lozoya.time


def integrate(x, y):
    integral = np.zeros(x.shape)
    for i, value in enumerate(y):
        integral[i] = np.trapz(y[0:i], x[0:i])
    return integral


def line(x, a, b):
    return a * x + b


def quadratic(x, a, b, c, z):
    return a * np.power(x - z, 2) + b * x + c


def cubic(x, a, b, c, d, z):
    return a * np.power(x - z, 3) + quadratic(x, b, c, d, z)


def quadric(x, a, b, c, d, e, z):
    return a * np.power(x - z, 4) + cubic(x, b, c, d, e, z)


def quintic(x, a, b, c, d, e, f, z):
    return a * np.power(x - z, 5) + quadric(x, b, c, d, e, f, z)


def logistic(x, L, a, k):
    return L / (1 + np.exp(-k * (x - a)))


def rank_parametric_fits(models, x, y):
    for model in models:
        params, _ = scipy.optimize.curve_fit(model, x, y, maxfev=10000)
        bestModel = model
        bestParams = params
    return bestModel, bestParams


def round_up_to_even(f):
    return math.ceil(f / 2.) * 2


def rotate_point(t, o1, o2, p1, p2):
    q1 = o1 + np.cos(t) * (p1 - o1) - np.sin(t) * (p2 - o2)
    q2 = o2 + np.sin(t) * (p1 - o1) + np.cos(t) * (p2 - o2)
    return q1, q2


def round_sigfigs(num, sig_figs):
    """
    num: number
    sig_figs: int
    """
    if num != 0:
        m = round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs - 1)))
        if float(m) - int(m) == 0:
            return int(m)
        return m
    return 0  # Can't take the log of 0


def round_array(arr, figures=2):
    """
    arr: numpy array
    return: numpy array
    """
    m = []
    try:
        for i in range(len(arr)):
            m.append(round_sigfigs(arr[i], figures))
        return np.array(m)
    except Exception as e:
        pass
    try:
        return round_sigfigs(arr, figures)
    except Exception as e:
        pass
    try:
        return sympy.N(arr, figures)
    except Exception as e:
        pass
    return arr


def rotation_matrix(axis, degrees):
    radians = np.deg2rad(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    rotationMatrices = \
        {
            'x': np.array(
                ((1, 0, 0),
                 (0, cosine, -sine),
                 (0, sine, cosine))
            ),
            'y': np.array(
                ((cosine, 0, sine),
                 (0, 1, 0),
                 (-sine, 0, cosine))
            ),
            'z': np.array(
                ((cosine, -sine, 0),
                 (sine, cosine, 0),
                 (0, 0, 1))
            ),
        }
    return rotationMatrices[axis]


def rotate_vector(vector, axis, degrees):
    return np.dot(rotation_matrix(axis, degrees), vector)


def rotate_vector_cumulative(vector, axes, angles, i):
    for j in range(i + 1):
        if axes[j] != 'z':
            vector = rotate_vector(vector, axes[j], angles[j])
    for j in range(i + 1):
        if axes[j] == 'z':
            vector = rotate_vector(vector, axes[j], angles[j])
    return vector


## REGRESSION

# REGRESSION FIT -----------------------------------
def best_fit_distribution(data, ax=None, N=2):
    topNSSE = enum_dict(N, np.inf)
    topNDist = enum_dict(N, scipy.stats.norm)
    topNParams = enum_dict(N, (0.0, 1.0))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        def get_histogram(data):
            y, x = np.histogram(data, density=True)
            x = (x + np.roll(x, -1))[:-1] / 2.0
            return x, y

        def get_pdf(distribution):
            # fit dist to data
            params = round_array(distribution.fit(data))
            arg, loc, scale = separate_params(params)
            # Calculate fitted PDF and error with fit in distribution
            pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
            return pdf, params

        # Get histogram of original data
        x, y = get_histogram(data)
        for distribution in lozoya.configuration.DISTRIBUTIONS:
            try:
                pdf, params = get_pdf(distribution)  # get new function to try
                sse = np.sum(np.power(y - pdf, 2.0))  # sum of squares error
                for best in topNSSE:
                    # Compare sse with top N sse
                    if topNSSE[best] > sse > 0:  # top 'N' functions are stored in topNSSE
                        topNDist[best] = distribution
                        topNParams[best] = params
                        topNSSE[best] = sse
                        break
            except Exception as e:
                print(e)
    return {i: topNDist[i].name for i in range(N)}, topNParams


# Create models from data
def best_fit_distribution0(data, bins=200, ax=None):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        """Model data by finding best fit distribution to data"""

        def tryplot2(pdf, x, ax):
            try:
                if ax:
                    pd.Series(pdf, x).plot(ax=ax)
            except Exception:
                pass

        def get_xy(data, bins):
            y, x = np.histogram(data, bins=bins, density=True)
            x = (x + np.roll(x, -1))[:-1] / 2.0

        def get_pdf():
            # fit dist to data
            params = distribution.fit(data)
            arg, loc, scale = separate_params(params)
            # Calculate fitted PDF and error with fit in distribution
            pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
            return pdf

        # Get histogram of original data
        x, y = get_xy(data, bins)
        # Estimate distribution parameters from data
        for distribution in lozoya.configuration.DISTRIBUTIONS:
            # Try to fit the distribution
            try:
                pdf = get_pdf()
                sse = np.sum(np.power(y - pdf, 2.0))
                # if axis pass in add to plot0
                tryplot2(pdf, x, ax)
                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse
            except Exception:
                pass
    return best_distribution.name, best_params


@lozoya.time.timer
def best_fit_distribution(data, N=2):
    """
    data: pandas Series
    N: int
    return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topNPDF = enum_dict(N, None)
        topNSSE = enum_dict(N, np.inf)
        topNDist = enum_dict(N, scipy.stats.norm)
        topNParams = enum_dict(N, (0.0, 1.0))
        histogram = make_histogram(data)
        results = fit_all(data, histogram)
        for result in results:
            n = rank(result['sse'], topNSSE)  # Compare sse with top N sse
            if type(n) != type(False):
                topNPDF[n] = make_pdf(result['distribution'], result['params'])
                topNDist[n] = result['name']
                topNParams[n] = result['params']
                topNSSE[n] = result['sse']
        return topNPDF, topNDist, topNParams, topNSSE


@lozoya.time.timer
def best_fit_distribution(data, N=2):
    """
    data: pandas Series
    N: int
    return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topNPDF = enum_dict(N, None)
        topNSSE = enum_dict(N, np.inf)
        topNDist = enum_dict(N, scipy.stats.norm)
        topNParams = enum_dict(N, (0.0, 1.0))
        # Get histogram of original data
        histogram = make_histogram(data)
        # results = fit_all(data, histogram)
        """loop = asyncio.new_event_loop()
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            fit_all(data, histogram)
        )"""
        results = fit_all(data, histogram)
        for result in results:
            n = rank(result['sse'], topNSSE)  # Compare sse with top N sse
            if type(n) != type(False):
                topNPDF[n] = make_pdf(result['distribution'], result['params'])
                topNDist[n] = result['name']
                topNParams[n] = result['params']
                topNSSE[n] = result['sse']
        # loop.close()
        return topNPDF, topNDist, topNParams, topNSSE


@lozoya.time.timer
def best_fit_distribution456(data, ax=None, N=2):
    topNSSE = enum_dict(N, np.inf)
    topNDist = enum_dict(N, scipy.stats.norm)
    topNParams = enum_dict(N, (0.0, 1.0))

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        def get_histogram(data):
            y, x = np.histogram(data, density=True)
            x = (x + np.roll(x, -1))[:-1] / 2.0
            return x, y

        def get_pdf(distribution):
            # fit dist to data
            params = round_array(distribution.fit(data))
            arg, loc, scale = separate_params(params)
            # Calculate fitted PDF and error with fit in distribution
            pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
            return pdf, params

        def test(distribution):
            pdf, params = get_pdf(distribution)  # get new function to try
            sse = np.sum(np.power(y - pdf, 2.0))  # sum of squares error
            # dists[distribution.name] = (pdf, params, sse)
            return (pdf, params, sse)

        # Get histogram of original data
        x, y = get_histogram(data)
        dists = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for distribution in lozoya.configuration.DISTRIBUTIONS:
                dists[distribution.name] = executor.submit(test, distribution)

        print('makes no sense')
        print(dists)
        for dist in dists:
            pass

    return {i: topNDist[i].name for i in range(N)}, topNParams


@lozoya.time.timer
def best_fit_distribution123(data, ax=None, N=2):
    def get_histogram(data):
        y, x = np.histogram(data, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        return x, y

    def get_pdf(distribution):
        # fit dist to data
        params = distribution.fit(data)
        params = round_array(params)
        arg, loc, scale = separate_params(params)
        # Calculate fitted PDF and error with fit in distribution
        pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
        return pdf, params

    bestN = queue.PriorityQueue(N)
    x, y = get_histogram(data)

    for distribution in lozoya.configuration.DISTRIBUTIONS:
        pdf, params = get_pdf(distribution)
        sse = np.sum(np.power(y - pdf, 2.0))
        if bestN.full():
            worst = bestN.get()
            if sse < -float(worst[0]):
                bestN.put((-sse, distribution, params))
            else:
                bestN.put((worst[0], worst[1], worst[2]))
        else:
            bestN.put((-sse, distribution, params))
    a = {}
    for i in range(N):
        a[N - (i + 1)] = bestN.get()
    names = {N - (i + 1): a[N - (i + 1)][1].name for i in range(N)}
    params = {N - (i + 1): a[N - (i + 1)][2] for i in range(N)}
    return names, params


def better(bestErrors, errors):
    if (2 * errors['MSE'] < bestErrors['MSE'] or errors['R2'] / 1.1 > bestErrors['R2']) and errors['R2'] <= 1:
        return True
    return False


# TODO must have way to limit stats0 that are calculated
def collect_stats(v, dtype):
    """
    v: pandas Series
    Calculate various statistics per variable
    and the correlation and covariance matrices
    of the dataset
    """

    def stats_dict():
        sd = collections.OrderedDict(
            [('Distinct', None), ('Outliers', None), ('Mean', None), ('Median', None), ('Mode', None), ('Min', None),
             ('Max', None), ('Standard Deviation', None), ('Variance', None), ('Skew', None), ('Kurtosis', None),
             ('Lower IQR', None), ('First Quartile', None), ('Third Quartile', None), ('Upper IQR', None)]
        )
        return sd

    # Descriptive Stats for each column (variable)
    sd = stats_dict()

    # Number of uniques
    sd['Distinct'] = v.nunique()

    # Central Tendencies
    if dtype == 'Numerical':
        sd['Mean'] = round_array(v.mean(), 4)
        sd['Median'] = round_array(v.median(), 4)
    if len(v.mode()) == 1:
        sd['Mode'] = v.mode()[0]
    """elif len(v.mode()) == 1:
        sd['Mode'] = 'Bimodal'
    elif len(v.mode()) == 2:
        sd['Mode'] = 'Trimodal'
    else:
        sd['Mode'] = 'Multimodal'
        """

    # Dispersion
    if dtype == 'Numerical':
        sd['Min'] = round_array(v.min(), 4)
        sd['Max'] = round_array(v.max(), 4)
        sd['Standard Deviation'] = round_array(v.std(), 4)
        sd['Variance'] = round_array(v.var(), 4)
        sd['Skew'] = round_array(v.skew(), 4)
        sd['Kurtosis'] = round_array(v.kurt(), 4)
        sd['First Quartile'] = round_array(v.quantile(0.25), 4)
        sd['Third Quartile'] = round_array(v.quantile(0.75), 4)
        sd['IQR'] = round_array(sd['Third Quartile'] - sd['First Quartile'], 4)
        sd['Lower IQR'] = round_array(sd['First Quartile'] - 1.5 * sd['IQR'], 4)
        sd['Upper IQR'] = round_array(sd['Third Quartile'] + 1.5 * sd['IQR'], 4)
        sd['Lower Outliers'] = v[v < sd['Lower IQR']]
        sd['Upper Outliers'] = v[v > sd['Upper IQR']]

    # sd['Quantiles'] = df[col].quantiles(Quantiles)
    # stats0[v.name] = sd
    return sd


def collect_stats(v, dtype):
    """
    v: pandas Series
    Calculate various stats per variable & correlation&covariance matrices of dataset
    """

    def stats_dict():
        sd = collections.OrderedDict(
            [('Distinct', None), ('Outliers', None), ('Mean', None), ('Median', None), ('Mode', None), ('Min', None),
             ('Max', None), ('Standard Deviation', None), ('Variance', None), ('Skew', None), ('Kurtosis', None),
             ('Lower IQR', None), ('First Quartile', None), ('Third Quartile', None), ('Upper IQR', None)]
        )
        return sd

    # Descriptive Stats for each column (variable)
    sd = stats_dict()
    # Number of uniques
    sd['Distinct'] = v.nunique()
    # Central Tendencies
    if dtype == 'Numerical':
        try:
            sd['Mean'] = round_array(v.mean(), 4)
            sd['Median'] = round_array(v.median(), 4)
        except:
            pass
    if len(v.mode()) == 1:
        sd['Mode'] = v.mode()[0]
    """elif len(v.mode()) == 2:
        sd['Mode'] = 'Bimodal'
    elif len(v.mode()) == 3:
        sd['Mode'] = 'Trimodal'
    else:
        sd['Mode'] = 'Multimodal'
        """
    # Dispersion
    if dtype == 'Numerical':
        try:
            sd['Min'] = round_array(v.min(), 4)
            sd['Max'] = round_array(v.max(), 4)
            sd['Standard Deviation'] = round_array(v.std(), 4)
            sd['Variance'] = round_array(v.var(), 4)
            sd['Skew'] = round_array(v.skew(), 4)
            sd['Kurtosis'] = round_array(v.kurt(), 4)
            sd['First Quartile'] = round_array(v.quantile(0.25), 4)
            sd['Third Quartile'] = round_array(v.quantile(0.75), 4)
            sd['IQR'] = round_array(sd['Third Quartile'] - sd['First Quartile'], 4)
            sd['Lower IQR'] = round_array(sd['First Quartile'] - 1.5 * sd['IQR'], 4)
            sd['Upper IQR'] = round_array(sd['Third Quartile'] + 1.5 * sd['IQR'], 4)
            sd['Lower Outliers'] = v[v < sd['Lower IQR']]
            sd['Upper Outliers'] = v[v > sd['Upper IQR']]
        except:
            pass
    # sd['Quantiles'] = df[col].quantiles(Quantiles)
    # stats[v.name] = sd
    return sd


def enum_dict(r, obj):
    return {i: obj for i in range(r)}


def density_kde(v):
    """
    This KDE estimate uses "Scott's Rule" to determine bandwidth:
    bandwidth = (# of observations) ^ (1 / (# of variables + 1)
    v:
    """
    density = scipy.stats.gaussian_kde(v)
    x = np.linspace(v.min(), v.max(), len(v))
    bandwidth = round_array((density.covariance_factor(),), 4)[0]
    density.covariance_factor = lambda: bandwidth
    density._compute_covariance()
    density = pd.Series(density(x), index=x)
    return density, bandwidth


def density_kde(v):
    # This KDE estimate uses "Scott's Rule" to determine bandwidth:
    # bandwidth = (# of observations) ^ (1 / (# of variables + 1)
    density = scipy.stats.gaussian_kde(v)
    xs = np.linspace(v.min(), v.max(), len(v))
    # density.set_bandwidth(bw_method=len(df[col])**(-1/(1+3)))
    # density.covariance_factor = lambda: len(df[col])**(-1/(1+3))#.25
    # density._compute_covariance()
    # TODO MOVE THIS TO DISTRIBUTION FITTER FILE
    bandwidth = round_array((density.covariance_factor(),), 4)[0]
    density.covariance_factor = lambda: bandwidth
    density._compute_covariance()
    density = pd.Series(density(xs), index=xs)
    return density, bandwidth


def estimate_error(y, yModel, dF, errors=None):
    """
    Calculate, residuals, chi squared, reduced chi squared, and standard error
    y: pandas Series
    yModel: pandas Series
    dF: int (degrees of freedom)
    errors: dictionary
    :return:
    """
    residuals = y - yModel
    chi2 = np.sum((residuals / yModel) ** 2)  # chi2 estimates error in data
    reducedChi2 = chi2 / (dF)  # reduced chi2 measures goodness of fit
    standardError = np.sqrt(np.sum(residuals ** 2) / (dF))  # standard deviation of the error
    # errors['residuals'] = residuals
    errors['Sum of Residuals'] = np.sum(residuals)
    errors['Chi Squared'] = chi2
    errors['Reduced Chi Squared'] = reducedChi2
    errors['Standard Error'] = standardError
    return errors


def estimate_error(y, yModel, dF):
    resid = y - yModel
    chi2 = np.sum((resid / yModel) ** 2)  # chi-squared; estimates error in data
    chi2_red = chi2 / (dF)  # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid ** 2) / (dF))  # standard deviation of the error
    return resid, chi2, chi2_red, s_err


def _fit(distribution, data):
    start = lozoya.time.time()
    fit = distribution.fit(data)
    if lozoya.time.time() - start > 0.05:
        pass  # print(distribution.name)
    return fit


def fit_all(data, histogram):
    """with ThreadPoolExecutor(1) as executor:
        futures = [executor.submit(fit_distribution, *(data, distribution, histogram)) for distribution in lozoya.configuration.DISTRIBUTIONS]
        results = [future.result() for future in as_completed(futures)]"""
    return [fit_distribution(data, distribution, histogram) for distribution in lozoya.configuration.DISTRIBUTIONS]


def fit_distribution(data, distribution, histogram=None):
    """
    data: pandas Series
    distribution: scipy continuous distribution
    histogram: tuple (x, y)
    """
    if type(histogram) == type(None):
        X, Y = make_histogram(data)
    else:
        X, Y = histogram
    params = _fit(distribution, data)  # distribution.fit(data)
    args, loc, scale = separate_params(params)
    pdf = distribution.pdf(X, loc=loc, scale=scale, *args)  # calculate pdf
    residuals = Y - pdf
    sse = np.sum(np.power(residuals, 2.0))
    return {
        'distribution': distribution, 'name': distribution.name, 'params': (args, loc, scale),
        'residuals':    residuals, 'sse': sse
    }


def fit_explog(x, y, fam, po):
    p, cov = scipy.optimize.curve_fit(lozoya.configuration.funcFams[fam], x, y, p0=po)
    p = round_array(p)
    a, b = p[0:2]
    yModel = lozoya.configuration.funcFams[fam](x, a, b)
    errors = get_model_data(y, yModel, p)
    return yModel, p, cov, errors


def fit_sin(x, y):
    ff = np.fft.fftfreq(len(x), (x[1] - x[0]))  # assume uniform spacing
    fftY = abs(np.fft.fft(y))
    freq0 = abs(ff[np.argmax(fftY[1:]) + 1])  # exclude 0 Hz "peak" related to offset
    amp0 = np.std(y) * 2. ** 0.5
    offset0 = np.mean(y)
    guess = np.array([amp0, 2. * np.pi * freq0, 0., offset0])
    p, cov = scipy.optimize.curve_fit(lambda t, a, b, c, d: a * np.sin(b * t + c) + d, x, y, p0=(guess))
    p = round_array(p)
    a, b, c, d = p[0:4]
    yModel = a * np.sin(b * x + c) + d
    errors = get_model_data(y, yModel, p)
    return yModel, p, cov, errors


def fit_poly(x, y, degree):
    p, cov = np.polyfit(x, y, degree, cov=True)  # parameters and covariance from of the fit
    p = round_array(p, 2)
    yModel = np.polyval(p, x)  # model using the fit parameters; NOTE: parameters here are coefficients
    errors = get_model_data(y, yModel, p)
    return yModel, p, cov, errors


# SERIES FIT
def get_errors(y, yModel):
    mse = ((y - yModel) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs((y - yModel)).mean()
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = np.sum((yModel - ybar) ** 2)  # or sum([ (yihat - ybar)**1 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**1 for yi in y])
    R2 = ssreg / sstot
    errors = {'MSE': mse, 'R Squared': R2, 'RMSE': rmse, 'MAE': mae}
    # return mse, rmse, mae, R2
    return errors


def get_ci_stats(y, p):
    n = y.size  # number of observations
    m = p.size  # number of parameters
    dF = n - m  # degrees of freedom
    t = scipy.stats.t.ppf(0.95, n - m)  # used for CI and PI bands
    return n, m, dF, t


def get_model(x, y):
    # Using Grid search Cross Validation to Select Best Degree of Polynomial
    minError = np.inf
    bestR2 = 0
    # POLYNOMIAL
    try:
        for i in range(1, 3):
            # Modeling with Numpy, choose best of 1st, 2nd, or 3rd degree polynomial
            p, cov = np.polyfit(x, y, i, cov=True)  # parameters and covariance from of the fit
            p = round_array(p, 2)
            yModel = np.polyval(p, x)  # model using the fit parameters; NOTE: parameters here are coefficients
            mse, rmse, mae, R2 = get_errors(y, yModel)
            if 2 * i * mse < minError or R2 / 1.1 > bestR2:
                minError = mse
                bestP, bestCov, bestModel, bestR2 = p, cov, yModel, R2
                fam = 'Polynomial' if i != 1 else 'Linear'
    except Exception as e:
        print(e)
    # EXPONENTIAL
    pos = ((4, 0.1), (-10, 3), (1, 1))
    for po in pos:
        try:
            p, cov = scipy.optimize.curve_fit(lambda t, a, b: a * np.exp(b * t), x, y, p0=po)
            p = round_array(p, 2)
            a, b = p[0], p[1]

            # c = p[1]
            yModel = a * np.exp(b * x)  # + c
            mse, rmse, mae, R2 = get_errors(y, yModel)
            if 2 * mse < minError or R2 / 1.1 > bestR2 and R2 <= 1:
                minError = mse
                bestP, bestCov, bestModel, bestR2 = p, cov, yModel, R2
                fam = 'Exponential'
                break
        except Exception as e:
            print(e)
    # NATURAL LOG
    try:
        p, cov = scipy.optimize.curve_fit(lambda t, a, b: a * np.log(t) + b, x, y, p0=(0, 1))
        p = round_array(p, 2)
        a = p[0]
        b = p[1]
        yModel = pd.Series(a * np.log(x) + b)
        mse, rmse, mae, R2 = get_errors(y, yModel)
        if 2 * mse < minError or R2 / 1.1 > bestR2:
            minError = mse
            bestP, bestCov, bestModel, bestR2 = p, cov, yModel, R2
            fam = 'Logarithmic'
    except Exception as e:
        print(e)
    # SINUSOIDAL
    try:
        ff = np.fft.fftfreq(len(x), (x[1] - x[0]))  # assume uniform spacing
        fftY = abs(np.fft.fft(y))
        freq0 = abs(ff[np.argmax(fftY[1:]) + 1])  # exclude 0 Hz "peak" related to offset
        amp0 = np.std(y) * 2. ** 0.5
        offset0 = np.mean(y)
        guess = np.array([amp0, 2. * np.pi * freq0, 0., offset0])
        p, cov = scipy.optimize.curve_fit(lambda t, a, b, c, d: a * np.sin(b * t + c) + d, x, y, p0=(guess))

        p = round_array(p, 2)
        a, b, c, d = p[0], p[1], p[2], p[3]

        yModel = a * np.sin(b * x + c) + d
        mse, rmse, mae, R2 = get_errors(y, yModel)
        if 2 * mse < minError or R2 / 1.1 > bestR2:
            minError = mse
            bestP, bestCov, bestModel, bestR2 = p, cov, yModel, R2
            fam = 'Sinusoidal'
    except Exception as e:
        print(e)
    # POLYNOMIAL
    try:
        for i in range(3, 6):
            # Modeling with Numpy, choose best of 1st, 2nd, or 3rd degree polynomial
            p, cov = np.polyfit(x, y, i, cov=True)  # parameters and covariance from of the fit
            p = round_array(p, 2)
            yModel = np.polyval(p, x)  # model using the fit parameters; NOTE: parameters here are coefficients
            mse, rmse, mae, R2 = get_errors(y, yModel)
            if 2 * i * mse < minError or R2 / (1.1 + i / 2) > bestR2 and R2 <= 1:
                minError = mse
                bestP, bestCov, bestModel, bestR2 = p, cov, yModel, R2
                fam = 'Polynomial'
    except Exception as e:
        print(e)
    return bestP, bestCov, bestModel, minError, bestR2, fam, rmse, mae


def get_model(x, y):
    # Using Grid search Cross Validation to Select Best Degree of Polynomial
    bestErrors = {'R Squared': 0, 'MSE': np.inf}
    # POLYNOMIAL
    try:
        for i in range(1, 3):
            # Modeling with Numpy, choose best of 1st, 2nd, or 3rd degree polynomial
            p, cov = np.polyfit(x, y, i, cov=True)  # parameters and covariance from of the fit
            p = round_array(p, 2)
            yModel = np.polyval(p, x)  # model using the fit parameters; NOTE: parameters here are coefficients
            errors = get_errors(y, yModel)
            R2 = errors['R Squared']
            if 2 * i * errors['MSE'] < bestErrors['MSE'] or R2 / 1.1 > bestErrors['R Squared']:
                minError = errors['MSE']
                bestP, bestCov, bestModel, bestErrors = p, cov, yModel, errors
                fam = 'Polynomial' if i != 1 else 'Linear'
    except Exception as e:
        print(e)
    # EXPONENTIAL
    pos = ((4, 0.1), (-10, 3), (1, 1))
    for po in pos:
        try:
            p, cov = scipy.optimize.curve_fit(lambda t, a, b: a * np.exp(b * t), x, y, p0=po)
            p = round_array(p, 2)
            a, b = p[0], p[1]
            # c = p[1]
            yModel = a * np.exp(b * x)  # + c
            errors = get_errors(y, yModel)
            R2 = errors['R Squared']
            if 2 * errors['MSE'] < bestErrors['MSE'] or R2 / 1.1 > bestErrors['R Squared'] and R2 <= 1:
                minError = errors['MSE']
                bestP, bestCov, bestModel, bestErrors = p, cov, yModel, errors
                fam = 'Exponential'
                break
        except Exception as e:
            print(e)
    # NATURAL LOG
    try:
        p, cov = scipy.optimize.curve_fit(lambda t, a, b: a * np.log(t) + b, x, y, p0=(0, 1))
        p = round_array(p, 2)
        a = p[0]
        b = p[1]
        yModel = pd.Series(a * np.log(x) + b)
        errors = get_errors(y, yModel)
        R2 = errors['R Squared']

        if 2 * errors['MSE'] < bestErrors['MSE'] or R2 / 1.1 > bestErrors['R Squared']:
            minError = errors['MSE']
            bestP, bestCov, bestModel, bestErrors = p, cov, yModel, errors
            fam = 'Logarithmic'
    except Exception as e:
        print(e)
    # SINUSOIDAL
    try:
        ff = np.fft.fftfreq(len(x), (x[1] - x[0]))  # assume uniform spacing
        fftY = abs(np.fft.fft(y))
        freq0 = abs(ff[np.argmax(fftY[1:]) + 1])  # exclude 0 Hz "peak" related to offset
        amp0 = np.std(y) * 2. ** 0.5
        offset0 = np.mean(y)
        guess = np.array([amp0, 2. * np.pi * freq0, 0., offset0])
        p, cov = scipy.optimize.curve_fit(lambda t, a, b, c, d: a * np.sin(b * t + c) + d, x, y, p0=(guess))

        p = round_array(p, 2)
        a, b, c, d = p[0], p[1], p[2], p[3]

        yModel = a * np.sin(b * x + c) + d
        errors = get_errors(y, yModel)
        R2 = errors['R Squared']

        if 2 * errors['MSE'] < bestErrors['MSE'] or R2 / 1.1 > bestErrors['R Squared']:
            minError = errors['MSE']
            bestP, bestCov, bestModel, bestErrors = p, cov, yModel, errors
            fam = 'Sinusoidal'
    except Exception as e:
        print(e)
    # POLYNOMIAL
    try:
        for i in range(3, 6):
            # Modeling with Numpy, choose best of 1st, 2nd, or 3rd degree polynomial
            p, cov = np.polyfit(x, y, i, cov=True)  # parameters and covariance from of the fit
            p = round_array(p, 2)
            yModel = np.polyval(p, x)  # model using the fit parameters; NOTE: parameters here are coefficients
            errors = get_errors(y, yModel)
            R2 = errors['R Squared']
            if 2 * i * errors['MSE'] < bestErrors['MSE'] or R2 / (1.1 + i / 2) > bestErrors['R Squared'] and R2 <= 1:
                minError = errors['MSE']
                bestP, bestCov, bestModel, bestErrors = p, cov, yModel, errors
                fam = 'Polynomial'
    except Exception as e:
        print(e)
    return bestP, bestCov, bestModel, fam, bestErrors


def get_model(x, y):
    # Grid Search Cross Validation to select best mdoel
    bestErrors = {'R2': 0, 'MSE': np.inf}
    # POLYNOMIAL
    for i in range(1, 3):
        try:
            yModel, p, cov, errors = fit_poly(x, y, degree=i)
            if better(bestErrors, errors):
                bestP, bestCov, bestModel, bestErrors = p, cov, yModel, errors
                fam = 'Linear' if i == 1 else 'Polynomial'
        except Exception as e:
            print(e)

    # EXPONENTIAL & LOG
    for fc in lozoya.configuration.funcFams:
        pos = ((4, 0.1), (-10, 3), (1, 1), (0, 1))
        for po in pos:
            try:
                yModel, p, cov, errors = fit_explog(x, y, fc, po)
                if better(bestErrors, errors):
                    bestP, bestCov, bestModel, bestErrors = p, cov, yModel, errors
                    fam = fc
            except Exception as e:
                print(e)

    # SINUSOIDAL
    try:
        yModel, p, cov, errors = fit_sin(x, y)
        if better(bestErrors, errors):
            bestP, bestCov, bestModel, bestErrors = p, cov, yModel, errors
            fam = 'Sinusoidal'
    except Exception as e:
        print(e)

    # POLYNOMIAL
    for i in range(3, 6):
        try:
            errors, p, cov, yModel = fit_poly(x, y, degree=i)
            if better(bestErrors, errors):
                bestP, bestCov, bestModel, bestErrors = p, cov, yModel, errors
                fam = 'Polynomial'
        except Exception as e:
            print(e)

    return bestP, bestCov, bestModel, fam, bestErrors


def get_pdfs(topDists, topParams, N):
    return {i: make_pdf(topDists[i], topParams[i]) for i in range(N)}


def get_model_data(y, yModel, p):
    observations = y.size
    numParams = p.size
    dof = observations - numParams
    t = scipy.stats.t.ppf(0.95, observations - numParams)
    residuals = y - yModel
    mse = (residuals ** 2).mean()
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yModel - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    chi2 = np.sum((residuals / yModel) ** 2)  # chi2 estimates error in data
    modelData = {
        'Observations':     observations, 'Parameters': numParams, 'Degrees of Freedom': dof, 't': t,
        # used for CI and PI bands
        # 'Residuals': residuals,
        'Sum of Residuals': np.sum(residuals), 'Chi Squared': chi2, 'Reduced Chi Squared': chi2 / dof,
        'Standard Error':   np.sqrt(np.sum(residuals ** 2) / dof), 'MSE': mse, 'R2': ssreg / sstot,
        'RMSE':             np.sqrt(mse), 'MAE': np.abs(y - yModel).mean()
    }
    return modelData


@numba.jit
def make_histogram(data):
    y, x = np.histogram(data, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    return x, y


def make_pdf(dist, params, size=10000):
    """Generate Propbability Distribution Function """

    def get_startend():
        # Get sane start and end points of distribution
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if len(arg) != 0 else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if len(arg) != 0 else dist.ppf(0.99, loc=loc, scale=scale)
        return start, end

    def get_starten0d():
        # Get sane start and end points of distribution
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)
        return start, end

    def get_pdf():
        # Build PDF and turn into pandas Series
        def get_xy():
            x = np.linspace(start, end, size)
            y = dist.pdf(x, loc=loc, scale=scale, *arg)
            return x, y

        x, y = get_xy()
        pdf = pd.Series(y, x)
        return pdf

    arg, loc, scale = separate_params(params)
    start, end = get_startend()
    pdf = get_pdf()
    return pdf


def make_pdf(dist, params, lims=None, size=10000):
    """
    Generate Probability Distribution Function
    dist: scipy continuous distribution
    params: tuple (parameters, location, scale)
    lims: tuple (lower x, upper x)
    size: int
    """
    args, loc, scale = params
    if lims == None:
        start = dist.ppf(0.01, *args, loc=loc, scale=scale)
        end = dist.ppf(0.99, *args, loc=loc, scale=scale)
    else:
        start, end = lims
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *args)
    pdf = pd.Series(y, x)
    return pdf


# https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
def plot_regression_and_ci(col, ax):
    """
    This function is to be called within plot_scatter()
    col: Pandas Series
    min: minimum value of col
    max: maximum value of col
    """

    def get_poly_model(x, y):
        """Use Grid search Cross Validation"""
        minError = np.inf
        for i in range(1, 4):
            # Modeling with Numpy, choose best of 1st, 2nd, or 3rd degree polynomial
            p, cov = np.polyfit(x, y, 1, cov=True)  # parameters and covariance from of the fit
            yModel = np.polyval(p, x)  # model using the fit parameters; NOTE: parameters here are coefficients
            # TODO check error
            mse = ((y - yModel) ** 2).mean()

            if mse < minError:
                minError = mse
                bestP, bestCov, bestModel = p, cov, yModel

        return bestP, bestCov, bestModel

    def get_ci_stats(y, p):
        n = y.size  # number of observations
        m = p.size  # number of parameters
        DF = n - m  # degrees of freedom
        t = scipy.stats.t.ppf(0.95, n - m)  # used for CI and PI bands
        return n, m, DF, t

    def estimate_error(y, y_model, DF):
        # Estimates of Error in Data/Model
        resid = y - y_model
        chi2 = np.sum((resid / y_model) ** 2)  # chi-squared; estimates error in data
        chi2_red = chi2 / (DF)  # reduced chi-squared; measures goodness of fit
        s_err = np.sqrt(np.sum(resid ** 2) / (DF))  # standard deviation of the error
        return resid, chi2, chi2_red, s_err

    def plot_ci(t, s_err, n, x, x2, y2, ax=None):
        """Return an axes of confidence bands using a simple approach.
        """
        if ax is None:
            ax = plt.gca()

            ci = t * s_err * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
            ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor="")

        return ax

    def plot_pi(t, s_err, x, x2, y2):
        pi = t * s_err * np.sqrt(1 + 1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
        ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
        ax.plot(x2, y2 + pi, "--", color="0.5")

    def modify_borders(ax):
        for s in ["top", "bottom", "left", "right"]:
            ax.spines[s].set_color("0.5")
        ax.get_xaxis().set_tick_params(direction="out")
        ax.get_yaxis().set_tick_params(direction="out")
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

    def modify_labels():
        plt.title("Fit Plot for Weight", fontsize="14", fontweight="bold")
        plt.xlabel("Height")
        plt.ylabel("Weight")
        plt.xlim(np.min(x) - 1, np.max(x) + 1)

    def modify_legend(ax, plt):
        handles, labels = ax.get_legend_handles_labels()
        display = (0, 1)
        anyArtist = plt.Line2D((0, 1), (0, 0), color="#b9cfe7")  # Create custom artists
        a = [handle for i, handle in enumerate(handles) if i in display]
        b = [label for i, label in enumerate(labels) if i in display]
        legend = plt.legend(
            a + [anyArtist], b + ["95% Confidence Limits"], loc=9, bbox_to_anchor=(0, -0.21, 1., .102),
            ncol=3, mode="expand"
        )
        frame = legend.get_frame().set_edgecolor("0.5")

    def export_(plt):
        plt.tight_layout()
        plt.savefig("filename.png", bbox_extra_artists=(legend,), bbox_inches="tight")

    def plot_dataTRASH(ax):
        ax.plot(
            x, y, "o", color="#b9cfe7", markersize=8, markeredgewidth=1, markeredgecolor="b",
            markerfacecolor="None"
        )

    def plot_r_ci(x, x2, y2, y_model, t, s_err, n, ax):
        # Plot Regression6
        ax.plot(x, y_model, "-", color="0.1", linewidth=1.5, alpha=0.5, label="Fit")
        # Plot Confidence Interval (select one)
        plot_ci(t, s_err, n, x, x2, y2, ax=ax)
        # Plot Prediction Interval
        plot_pi(t, s_err, x, x2, y2)

    def plot_best_fit(ax):
        props = dict(boxstyle='round', alpha=0.5, color=sns.color_palette()[0])
        textstr = r'\$y=-499.2 + 1.0x\$'
        ax.text(0.0, 0.0, textstr, transform=g.ax.transAxes, fontsize=14, bbox=props)
        ax.text(0.7, 0.9, textstr, transform=g.ax.transAxes, fontsize=14, bbox=props)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Start --------------------------------------------------------------------
    x = np.arange(len(col))
    y = col
    p, cov, y_model = get_poly_model(x, y)
    n, m, DF, t = collect_stats(y, p)
    resid, chi2, chi2_red, s_err = estimate_error(y, y_model, DF)
    x2 = np.linspace(np.min(x), np.max(x), 100)
    y2 = np.linspace(np.min(y_model), np.max(y_model), 100)
    # Plot Data
    plot_dataTRASH(ax)
    plot_r_ci(x, x2, y2, y_model, t, s_err, n, ax)
    # Figure Modifications
    modify_borders(ax)
    modify_labels(plt)
    modify_legend(ax, plt)
    # Save Figure
    export_(plt)


def rank(a, b):
    """
    Iterate through list b, comparing each value in b
    against the value of a. If the value of a is less
    than the value of b[i], the value of a will
    replace the value of b[i].
    a: number
    b: list of numbers
    return: index of b that was replaced if replacement took place, otherwise False
    """
    for i in range(len(b)):
        if a < b[i]:
            return i
    return False


@numba.jit
def separate_params(parameters):
    # Separate parts of parameters
    args = parameters[:-2]
    loc = parameters[-2]
    scale = parameters[-1]
    return args, loc, scale


def separate_params(pa):
    # Separate parts of parameters
    args = pa[:-2]
    loc, scale = pa[-2:]
    return args, loc, scale


#
class Coordinate:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.high = 2
        self.current = -1

    def as_list(self, *args, **kwargs):
        return self.x, self.y, self.z

    def __iter__(self, *args, **kwargs):
        return self

    def __next__(self, *args, **kwargs):
        self.current += 1
        if self.current <= self.high:
            return self.as_list()[self.current]
        self.current = -1
        return StopIteration


class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w if w != 'r' else random.randint(-2, 2) / 25
        self.x = x if x != 'r' else random.randint(-2, 2) / 25
        self.y = y if y != 'r' else random.randint(-2, 2) / 25
        self.z = z if z != 'r' else random.randint(-2, 2) / 25

    def __add__(self, other):
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z, )

    def __mul__(self, other):
        return ''

    @property
    def norm(self, *args, **kwargs):
        return np.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)


polynomials = [line, quadratic, cubic, quadric, quintic]
regressionModels = polynomials
