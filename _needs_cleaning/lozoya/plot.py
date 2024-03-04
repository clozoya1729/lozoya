# import math
# import queue
# import random
# import re
# import threading
# import time
# from decimal import Decimal
# from math import pi
#
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
# import unicodedata
# from bokeh.layouts import layout, row
# from bokeh.models import FuncTickFormatter, Range1d
# from bokeh.models import Legend, HoverTool, ColumnDataSource
'''
from matplotlib import style
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
import matplotlib.animation as animation
from bokeh.models.widgets import DataTable, Panel, Tabs, TableColumn
from bokeh.plotting import figure
from bokeh.plotting import save, output_file
from matplotlib import animation
from matplotlib import ticker as ticker
from numba import jit
from utility import decorators, status, clear_directories, s_round, text_formatter, timer

import math_api
from __variable import *
from file import *

try:
    from matplotlib.backends.backend_qt5agg import (FigureCanvas)
except:
    from matplotlib.backends.backend_qt5agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qt5agg import (FigureCanvas)
except:
    from matplotlib.backends.backend_qt5agg import FigureCanvasAgg as FigureCanvas
np.random.seed(345145)
os.environ["PATH"] += 'C:\Program Files\MiKTeX 1.5\miktex/bin/x64' + os.pathsep
os.environ["PATH"] += 'Z:\Family\Latex\miktex/bin/x64' + os.pathsep
sns.set_context('paper', rc={'lines.linewidth': 1, 'figure.figsize': (1, 1), 'figure.facecolor': 'white',
                             'font.family': ['sans-serif', ]})
sns.set()
sns.set_style("whitegrid")
sns.set(font='serif', font_scale=.7)
sns.set_style('ticks', {'font.family': 'serif', 'font.serif': ['Times', 'Palatino', 'serif']})
plt.legend(fontsize=7, loc='best', frameon=False, borderaxespad=5)
matplotlib.use('agg')
matplotlib.use('Qt5Agg')
plt.switch_backend('agg')
PLOTS_DIR = PlotPaths.DYNAMIC_PLOTS
PLOTS_DIR = os.path.join(os.path.join(os.path.abspath('../../1/'), 'LoParPlotGenerator0'), 'DynamicPlots')
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
BarPlotSuffix = '_bar'
BoxPlotSuffix = '_box'
DistributionPlotSuffix = '_distribution'
DistributionFitPlotSuffix = '_distributionFit'
ScatterPlotSuffix = '_scatter'
TablePlotSuffix = '_table'
ViolinPlotSuffix = '_violin'
lozoya.data_api.file_api.file_functions.clear_directories()
STATS = ['Distinct', 'Kurtosis', 'Max', 'Mean', 'Median', 'Min', 'Mode', 'Skew', 'Standard\nDeviation', 'Variance']
fig = plt.figure()
x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
im = plt.imshow(f(x, y), animated=True)
ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()
lozoya.data_api.file_api.file_functions.clear_directories(PlotsDir)
figs = []
dash = ((7, 2.5, 3, 2.5, 3, 2.5, 7, 2.5, 3, 2.5, 3, 2.5, 7, 2.5, 7, 2.5),
        (11, 1.7, 5, 1.7, 1.3, 1.7, 5, 1.7, 1.3, 1.7, 5, 1.7, 11, 1.7, 1.3, 1.7), (2, 2, 5, 2, 5, 2, 5, 2),
        (7, 2.1, 4.9, 2.1, 4.9, 2.1, 4.9, 2.1, 2.1, 2.1, 7, 2.1, 2.1, 2.8, 2.1, 2.1, 7, 2.1, 2.8, 2.1),
        (4.8, 1.8, 4.8, 1.8, 1.8, 1.8, 4.8, 1.8, 6.4, 1.8, 4.8, 1.8, 1.8, 1.8),
        (6, 3, 12, 3, 2, 3, 2, 3, 6, 3, 6, 3, 2, 3), (6, 2, 1.5, 2, 1.5, 2, 1.5, 2), (8, 4, 2, 4, 2, 4, 8, 4),
        (5, 1.6, 2, 1.6, 7, 1.6, 2, 1.3), (9, 3, 3, 3, 3, 3))
mark = ('o',)
fill_style = ('none',)  # top, bottom, right, left, full: markerfacecoloralt='gray'
colors = (
    ('#3CE1E0', '#56A71E', '#7CD530', '#B6ED15', '#E6F70E', '#F5AB23', '#F06D12', '#B43519', '#891A1A', '#000000'), (
        'maroon', 'darkred', 'brown', 'firebrick', 'crimson', 'indianred', 'lightcoral', 'salmon', 'rosybrown',
        'darksalmon', 'lightsalmon', 'tomato', 'orangered', 'darkorange', 'orange'), (
        'darkcyan', 'teal', 'steelblue', 'cadetblue', 'lightslategrey', 'skyblue', 'turquoise', 'lightseagreen',
        'darkturquoise'))

# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and search serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
# matplotlib.rcParams['mathtext.fontset'] = ''
# os.environ["PATH"] += 'C:\Program Files\MiKTeX 1.5\miktex/bin/x64' + os.pathsep

import inspect
import warnings
from collections import OrderedDict
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from . import expressions
    from . import settings as gc
    from . import statscalc
    from . import util
except:
    import expressions
    import settings as gc
    import statscalc
    import tools

try:
    from data.tensorstone_statistics import expressions, settings as gc, statscalc0, util0
except:
    import data.tensorstone_statistics.expressions
    import data.tensorstone_statistics.settings as gc
    import statscalc0
    import tools

plt.switch_backend('agg')
sns.set_context(None, rc=gc.snsContext)
sns.set(font='serif', font_scale=.7)
sns.set_style('ticks', gc.snsStyle)
plt.legend(fontsize=7, loc='best', frameon=False, borderaxespad=5)


def format_plot(data=None, plotStyle='', plot=None, ax=None, title=''):
    """
    name: string
    vector: pandas Series
    plotStyle: string
    plot: Matplotlib plt
    ax: Matplotlib figure
    title: boolean
    """
    plot.title(title, fontsize='9', fontweight='bold', fontname='Georgia')
    sns.set_context(None, rc=gc.snsContext)
    sns.set(font='serif', font_scale=.7)
    sns.set_style('ticks', gc.snsStyle)
    plt.legend(fontsize=7, loc='best', frameon=False, borderaxespad=5)
    if gc.tufte:
        sns.despine(top=True, right=True, trim=True)
    sns.set_palette(gc.colors)


def plot(func):
    # Plot, format, and export.

    @wraps(func)
    def make_plot(*args, **kwargs):
        plt.gcf().clear()
        plot, payload = func(*args, **kwargs)
        defaults = get_default_args(func)
        for d in defaults:
            if d not in kwargs:
                kwargs[d] = defaults[d]
        format_plot(data=args[0], plotStyle=func.__name__[5:], plot=plt, ax=plot, title=kwargs['title'])
        export_plot(plot, path=kwargs['path'])
        return plot, payload

    return make_plot


def export_plot(plot, path):
    fig = plot.get_figure()
    fig.savefig(path, bbox_inches='tight')


@plot
def plot_bar(data, path, stats=None, fileName='', title=''):
    m = data.value_counts()
    barPlot = sns.barplot(x=m.index, y=m.values)
    plt.ylabel('Quantity')
    plt.ylim(np.min(data.index.values) - 1, np.max(data.index.values) + 1)
    return barPlot, None


@plot
def plot_box(data, path, stats=None, fileName='', title=''):
    boxPlot = sns.boxplot(y=data, width=.25)
    plt.xlim(np.min(data) - 1, np.max(data) + 1)
    sns.boxplot(capsize=.5)
    return boxPlot, None


def plot_corrcov(df, path, fileName=('Correlation', 'Covariance')):
    corr, cov = df.corr(), df.cov()
    plot_heatmap(corr, title='', path=path, fileName=fileName[0])
    plot_heatmap(cov, title='', path=path, fileName=fileName[1])
    return corr, cov


@plot
def plot_heatmap(matrix, path, fileName='', title=''):
    heatmapPlot = sns.heatmap(matrix, annot=False, vmin=0, vmax=None)
    sns.set_palette(gc.colors)
    return heatmapPlot, None


@plot
def plot_histogram(data, path, fileName='', title=''):
    distPlot = sns.distplot(data, norm_hist=False, kde=False, bins=None, rug=True)
    plt.ylabel('Frequency')
    plt.xlim(np.min(data) - 1, np.max(data) + 1)
    return distPlot, None


@plot
def plot_scatter(data, path, fit=True, fileName='', title=''):
    scatterPlot = sns.regplot(np.arange(len(data)), data, fit_reg=False)  # , label=v.name)
    xRange = np.max(data.index.values) - np.min(data.index.values)
    # strata = 5
    # xTicks = np.arange(np.min(data.index.values) - 1, np.max(data.index.values) + 1, step=int(xRange / strata))
    # plot.xticks(xTicks)
    plt.xlim(np.min(data.index.values) - 1, np.max(data.index.values) + 1)
    # plt.gca().set_aspect('equal', adjustable='box')
    if fit:
        regressionFit, regressionError = plot_regression_and_ci(data, scatterPlot)
        return scatterPlot, (regressionFit, regressionError)
    return scatterPlot, None


def plot_stats(df, path, statsDict=None):
    plt.gcf().clear()
    cellText = []
    colLabels = []
    s = 0
    for col in df:
        f = sorted(statsDict[col])
        cellText.append([str(statsDict[col][s]) for s in f])
        colLabels.append(col)
        if s == 0:
            rowLabels = [s for s in f]
            s += 1
    cellText = pd.DataFrame(cellText).T.as_matrix()
    table = plt.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, loc='center', rowLoc='center',
                      colLoc='center')
    table.auto_set_column_width(0)
    table.scale(2, 3)
    table.set_fontsize(12)
    plt.axis('off')
    # format_plot(name='', data=df, plotStyle='Table', plot=plt, ax=table)
    export_plot(table, path, '' + gc.TABLE_PLOTS_SUFFIX)


@plot
def plot_violin(data, path, fileName='', title=''):
    violinPlot = sns.violinplot(y=data)
    plt.ylabel('Frequency')
    plt.xlim(np.min(data) - 1, np.max(data) + 1)
    sns.violinplot(scale='count', linewidth=3, )
    return violinPlot, None


def plot_stats_comparison(stats, stat):
    """
    stats: pandas DataFrame
    stat: str
    """
    plt.gcf().clear()
    h = {}
    for col in stats:
        h[col] = stats[col][stat]

    barplot = sns.barplot(x=[col for col in h], y=[h[col] for col in h])
    plt.title(stat + ' Comparison')
    format_plot(name=stat, data=stats, plotStyle='Bar Plot', plot=plt, ax=barplot)
    fig = barplot.get_figure()
    fig.savefig(os.path.join(gc.statComparePlotDir, stat[:3] + '_comparison'))


@plot
def plot_table(data, path, fileName='', title=''):
    """
    data: dictionary
    """
    cellText = [[str(data[k])] for k in data]
    rowLabels = [[k] for k in data]
    rowLabels = [util.text_formatter(r, 10) for r in rowLabels]
    table = plt.table(cellText=cellText, rowLabels=rowLabels, rowLoc='center', loc='center', colLoc='center')
    plt.axis('off')
    for key, cell in table.get_celld().items():
        cell.set_linewidth(1)
    table.auto_set_column_width(0)
    return table, None


def compare_all_stats(stats):
    """
    stats: pandas DataFrame
    """
    for stat in stats:
        plot_stats_comparison(stats, stat)


def plot_stats(df, path, statsDict=None):
    plt.gcf().clear()
    cellText = []
    colLabels = []
    s = 0
    for col in df:
        f = sorted(statsDict[col])
        cellText.append([str(statsDict[col][s]) for s in f])
        colLabels.append(col)
        if s == 0:
            rowLabels = [s for s in f]
            s += 1
    cellText = pd.DataFrame(cellText).T.as_matrix()
    table = plt.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, loc='center', rowLoc='center',
                      colLoc='center')
    table.auto_set_column_width(0)
    table.scale(2, 3)
    table.set_fontsize(12)
    plt.axis('off')
    # format_plot(name='', data=df, plotStyle='Table', plot=plt, ax=table)
    export_plot(table, path, '' + gc.TABLE_PLOTS_SUFFIX)


def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


# This function is to be called within plot_scatter()
def plot_regression_and_ci(y, ax):
    x = np.array(y.index)
    p, cov, yModel, fam, er = statscalc.get_model(x, y)
    x2 = np.linspace(np.min(x), np.max(x), len(x))
    expression = expressions.function_string(*p, fam=fam, y=y.label, evaluate=True, latex=False)
    if er['R2'] > 0.05:
        # Regression
        ax.plot(x, yModel, **gc.rLine)
        # Confidence Interval
        ci = er['t'] * er['Standard Error'] * np.sqrt(
            1 / er['Observations'] + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        ax.fill_between(x2, yModel + ci, yModel - ci, **gc.ciFill)
        ax.plot(x2, yModel - ci, label='95% Confidence Interval', **gc.ciBorder)
        ax.plot(x2, yModel + ci, **gc.ciBorder)
        # Prediction Interval
        pi = er['t'] * er['Standard Error'] * np.sqrt(
            1 + 1 / er['Observations'] + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        ax.fill_between(x2, yModel + pi, yModel - pi, **gc.piFill)
        ax.plot(x2, yModel - pi, label='95% Prediction Limits', **gc.piBorder)
        ax.plot(x2, yModel + pi, **gc.piBorder)
    return expression, er


def plot_stats(df, path, statsDict=None):
    plt.gcf().clear()
    cellText = []
    colLabels = []
    s = 0
    for col in df:
        f = sorted(statsDict[col])
        cellText.append([str(statsDict[col][s]) for s in f])
        colLabels.append(col)
        if s == 0:
            rowLabels = [s for s in f]
            s += 1
    cellText = pd.DataFrame(cellText).T.as_matrix()
    table = plt.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, loc='center', rowLoc='center',
                      colLoc='center')
    table.auto_set_column_width(0)
    table.scale(2, 3)
    table.set_fontsize(12)
    plt.axis('off')
    # format_plot(name='', data=df, plotStyle='Table', plot=plt, ax=table)
    export_plot(table, path, '' + gc.TABLE_PLOTS_SUFFIX)


@plot
def plot_distribution(data, path, N=1, fileName='', title=''):
    """
    Plots the top N distributions as well as a Gaussian Kernel Density Estimate
    """
    distPlot = sns.distplot(data, kde=False, norm_hist=True)
    pdfs, topNames, topParams, topNSSE = statscalc.best_fit_distribution(data, N)
    density, bandwidth = statscalc.density_kde(data)
    density.plot(lw=2, label="Gaussian Kernel Density Estimate\n", legend=True)
    fit, names, errors = expressions.distribution_string(topNames, topParams, topNSSE, y='PDF', latex=False)
    for i in pdfs:
        # paramNames = (topDists[i].shapes + ', loc, scale').split(', ') if topDists[i].shapes else ['loc', 'scale']
        # param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(paramNames, topParams)])
        distPlot = pdfs[i].plot(lw=2, linestyle=gc.lineStyles[i], label=names[topNames[i]], legend=True)
    f = {names[k]: (fit[k], errors[k]) for k in names}
    plt.ylabel('Probability')
    plt.xlim(np.min(data) - 1, np.max(data) + 1)
    return distPlot, f


# MAIN
@tools.timer
def generate_all_plots(df, dtypes, path):
    """
    Returns all information used in creation of plots
    dtypes: list of str, 'Numerical' or 'Categorical'
    return: list of str, dict, dict, dict, dict
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        variables = []
        categoricals = {}
        numericals = {}
        distFits = {}
        regression = {}
        dfNum = []
        dfCat = []
        for col, label in zip(df, dtypes):
            if label == 'Categorical':
                categoricals[col] = statscalc.collect_stats(df[col], label)
                dfCat.append(col)
                plot_bar(df[col], stats=categoricals[col], path=os.path.join(path, 'bar',
                                                                             col))  # plot_scatter(df[col], fit=False)  # plot_stats(df[col], stats=['Mode'])
            elif label == 'Numerical':
                try:
                    numericals[col] = statscalc.collect_stats(df[col], label)
                    dfNum.append(col)  # TODO not sure what this is
                    plot_table(OrderedDict([(k, numericals[col][k]) for k in gc.iqrTableStats]),
                               path=os.path.join(path, 'iqr', col))
                    plot_table(OrderedDict([(k, numericals[col][k]) for k in gc.dispersionTableStats]),
                               path=os.path.join(path, 'dispersion', col))
                    plot_table(OrderedDict([(k, numericals[col][k]) for k in gc.centralTendenciesTableStats]),
                               path=os.path.join(path, 'centraltendencies', col))
                    plot_histogram(df[col], path=os.path.join(path, 'distribution', col))
                    _, distFits[col] = plot_distribution(df[col], N=gc.N,
                                                         path=os.path.join(path, 'distributionfit', col))
                    _, regression[col] = plot_scatter(df[col], fit=True, path=os.path.join(path, 'scatter', col))
                    plot_table(OrderedDict(
                        [(k, util.s_round(regression[col][1][k])) for k in sorted(regression[col][1].keys())]),
                        path=os.path.join(path, 'regressionerror', col))
                    plot_table(
                        OrderedDict([(k, util.s_round(distFits[col][k][1])) for k in sorted(distFits[col].keys())]),
                        path=os.path.join(path, 'distributionerror', col))
                    plot_box(df[col], path=os.path.join(path, 'box', col))
                    plot_violin(df[col], path=os.path.join(path, 'violin', col))
                    variables.append(col)
                except Exception as e:
                    print(e)
            if 'Numerical' in dtypes:
                try:
                    plot_corrcov(df[dfNum], path=os.path.join(path, 'heatmap', 'correlation'))
                except:
                    pass
                """plot_stats(df[dfNum], statsDict=numericals, path=os.path.join(path, 'StatisticComparison'))"""
        return variables, numericals, categoricals, distFits, regression


def histogram(data, yCol):
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Histogram(x=data.loc[:, col].values, histnorm='percent', name=col, opacity=0.75, nbinsx=20, ))
    fig.update_layout(barmode='overlay', bargap=0.2, bargroupgap=0.1, xaxis_title_text=yCol,
                      yaxis_title_text='Percent (%)')
    div = po.plot(fig, auto_open=False, output_type='div', configuration.py=configuration.py)
    return div


def plot_2d(xCols, yCol, X, y, yFits):
    plot = ''
    for xCol in xCols:
        x = X.sort_values(xCol).loc[:, xCol]
        d = go.Scatter(x=x, y=y, mode='markers', name=yCol)
        traces = [d]
        for model in sorted(yFits.keys()):
            traces.append(go.Scatter(x=x, y=yFits[model], mode='lines', name=model))
        figure = format(go.Figure(data=traces), xCol, yCol)
        plot += str(po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py))
    return plot


def plot_3d(xCols, yCol, X, y, yFits):
    plot = ''
    for xCol in itertools.combinations(xCols, 2):
        x1 = X.sort_values(xCol[0]).iloc[:, 0]
        x2 = X.sort_values(xCol[1]).iloc[:, 1]
        d = go.Scatter3d(x=x1, y=x2, z=y, mode='markers', name=yCol, marker=dict(size=3))
        traces = [d]
        for i, model in enumerate(sorted(yFits.keys())):
            # traces.append(
            #     go.Mesh3d(
            #         x=x1,
            #         y=x2,
            #         z=yFits[model],
            #         color=colors[i],
            #         opacity=0.25,
            #         name='{} Surface'.format(model),
            #     )
            #
            # )
            traces.append(go.Scatter3d(x=x1, y=x2, z=yFits[model], line=dict(color=colors[i]), mode='lines',
                                       name='{}'.format(model), )

                          )
        plot += str(
            po.plot(format(go.Figure(data=traces), xCol[0], xCol[1], yCol, nvars=3), auto_open=False, output_type='div',
                    configuration.py=configuration.py))
    return plot


def get_histogram(x, y, yData, project):
    histogramData = get_histogram_data(path.fitPath.format(project), x, y, yData)  # TODO
    return histogram(histogramData, y)


def histogram(data, yCol):
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Histogram(x=data.loc[:, col].values, histnorm='percent', name=col, opacity=0.75, nbinsx=20, ))
    fig.update_layout(barmode='overlay', bargap=0.2, bargroupgap=0.1, xaxis_title_text=yCol,
                      yaxis_title_text='Percent (%)')
    fig = format(fig)
    div = po.plot(fig, auto_open=False, output_type='div', configuration.py=configuration.py)
    return div


def plot2d(X, y, yFits, xCols, yCol):
    plot = ''
    for xCol in xCols:
        x = X.sort_values(xCol).loc[:, xCol]
        d = go.Scatter(x=x, y=y, mode='markers', name=yCol)
        traces = [d]
        for model in sorted(yFits.keys()):
            traces.append(go.Scatter(x=x, y=yFits[model], mode='lines', name=model))
        figure = format(go.Figure(data=traces), xCol, yCol)
        plot += str(po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py))

    for xCol in itertools.combinations(xCols, 2):
        x1 = X.sort_values(xCol[0]).iloc[:, 0]
        x2 = X.sort_values(xCol[1]).iloc[:, 1]
        d = go.Scatter3d(x=x1, y=x2, z=y, mode='markers', name=yCol, marker=dict(size=3))
        traces = [d]
        for i, model in enumerate(sorted(yFits.keys())):
            # traces.append(
            #     go.Mesh3d(
            #         x=x1,
            #         y=x2,
            #         z=yFits[model],
            #         color=colors[i],
            #         opacity=0.25,
            #         name='{} Surface'.format(model),
            #     )
            #
            # )
            traces.append(go.Scatter3d(x=x1, y=x2, z=yFits[model], line=dict(color=colors[i]), mode='lines',
                                       name='{}'.format(model), )

                          )
        plot += str(
            po.plot(format(go.Figure(data=traces), xCol[0], xCol[1], yCol, nvars=3), auto_open=False, output_type='div',
                    configuration.py=configuration.py))
    return plot


configuration.py = {'displaylogo': False, }


def histogram(data, yCol):
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Histogram(x=data.loc[:, col].values, histnorm='percent', name=col, opacity=0.75, nbinsx=20, ))
    fig.update_layout(barmode='overlay', bargap=0.2, bargroupgap=0.1, xaxis_title_text=yCol,
                      yaxis_title_text='Percent (%)')
    div = po.plot(fig, auto_open=False, output_type='div', configuration.py=configuration.py)
    return div


def plot_2d(xCols, yCol, X, y, yFits):
    plot = ''
    for xCol in xCols:
        x = X.sort_values(xCol).loc[:, xCol]
        d = go.Scatter(x=x, y=y, mode='markers', name=yCol)
        traces = [d]
        for model in sorted(yFits.keys()):
            traces.append(go.Scatter(x=x, y=yFits[model], mode='lines', name=model))
        figure = format(go.Figure(data=traces), xCol, yCol)
        plot += str(po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py))
    return plot


def plot_3d(xCols, yCol, X, y, yFits):
    plot = ''
    for xCol in itertools.combinations(xCols, 2):
        x1 = X.sort_values(xCol[0]).iloc[:, 0]
        x2 = X.sort_values(xCol[1]).iloc[:, 1]
        d = go.Scatter3d(x=x1, y=x2, z=y, mode='markers', name=yCol, marker=dict(size=3))
        traces = [d]
        for i, model in enumerate(sorted(yFits.keys())):
            # traces.append(
            #     go.Mesh3d(
            #         x=x1,
            #         y=x2,
            #         z=yFits[model],
            #         color=colors[i],
            #         opacity=0.25,
            #         name='{} Surface'.format(model),
            #     )
            #
            # )
            traces.append(go.Scatter3d(x=x1, y=x2, z=yFits[model], line=dict(color=colors[i]), mode='lines',
                                       name='{}'.format(model), )

                          )
        plot += str(
            po.plot(format(go.Figure(data=traces), xCol[0], xCol[1], yCol, nvars=3), auto_open=False, output_type='div',
                    configuration.py=configuration.py))
    return plot


def get_histogram(x, y, yData, project):
    histogramData = get_histogram_data(path.fitPath.format(project), x, y, yData)  # TODO
    return histogram(histogramData, y)


from tools.processor import get_histogram_data

from . import path

configuration.py = {'displaylogo': False, }


def new_plot2(project, filename, cols):
    filePath = os.path.join(path.uploadsPath, project, filename)
    data = pd.read_csv(filePath, usecols=cols)

    if len(cols) == 3:
        xData = data.loc[:, cols[0]]
        yData = data.loc[:, cols[1]]
        zData = data.loc[:, cols[2]]
        trace = go.Scatter3d(x=xData, y=yData, z=zData, mode='markers', marker=dict(size=3))
    else:
        xData = data.loc[:, cols[0]]
        yData = data.loc[:, cols[1]]
        trace = go.Scatter(x=xData, y=yData, mode='markers')
    figure = go.Figure(data=trace)
    if len(cols) == 2:
        figure = format(figure, xLabel=cols[0], yLabel=cols[1])
    elif len(cols) == 3:
        figure = format(figure, xLabel=cols[0], yLabel=cols[1], zLabel=cols[2], nvars=3)
    div = po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py)
    return div


def new_plot(project, filename, xCol=None, yCol=None):
    filePath = os.path.join(path.uploadsPath, project, filename)
    columns = pd.read_csv(filePath, nrows=1).columns
    if xCol and yCol:
        data = pd.read_csv(filePath, usecols=[xCol, yCol])
    else:
        data = pd.read_csv(filePath, usecols=[columns[0], columns[1]])
    if not xCol:
        xCol = columns[0]
    if not yCol:
        yCol = columns[1]
    xData = data.loc[:, xCol]
    yData = data.loc[:, yCol]
    trace = go.Scatter(x=xData, y=yData, mode='markers', name=yCol)
    figure = go.Figure(data=trace)
    figure = format(figure, xCol, yCol)
    div = po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=yData.values, histnorm='percent', name=yCol, opacity=0.75, ))
    fig.update_layout(barmode='overlay', bargap=0.2, bargroupgap=0.1, xaxis_title_text=yCol,
                      yaxis_title_text='Percent (%)')
    div2 = po.plot(fig, auto_open=False, output_type='div', configuration.py=configuration.py)
    return div, div2


def histogram(data, yCol):
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Histogram(x=data.loc[:, col].values, histnorm='percent', name=col, opacity=0.75, nbinsx=20, ))
    fig.update_layout(barmode='overlay', bargap=0.2, bargroupgap=0.1, xaxis_title_text=yCol,
                      yaxis_title_text='Percent (%)')
    div = po.plot(fig, auto_open=False, output_type='div', configuration.py=configuration.py)
    return div


def histogram2(project, filename, xCol=None, yCol=None):
    filePath = os.path.join(path.uploadsPath, project, filename)
    columns = pd.read_csv(filePath, nrows=1).columns
    if True:
        data = pd.read_csv(filePath)
    elif xCol and yCol:
        data = pd.read_csv(filePath, usecols=[xCol, yCol])
    else:
        data = pd.read_csv(filePath, usecols=[columns[0], columns[1]])
    if not yCol:
        yCol = columns[1]
    yData = data.loc[:, yCol]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=yData.values, histnorm='percent', name=yCol, opacity=0.75, ))
    fig.update_layout(barmode='overlay', bargap=0.2, bargroupgap=0.1, xaxis_title_text=yCol,
                      yaxis_title_text='Percent (%)')
    return po.plot(fig, auto_open=False, output_type='div', configuration.py=configuration.py)


def format(figure, xLabel='', yLabel='', zLabel='', title='', nvars=2):
    if nvars == 3:
        scene = dict(xaxis_title=xLabel, yaxis_title=yLabel, zaxis_title=zLabel,
                     xaxis=dict(backgroundcolor="rgb(200, 200, 230)", gridcolor="white", showbackground=True,
                                zerolinecolor="white", ),
                     yaxis=dict(backgroundcolor="rgb(230, 200,230)", gridcolor="white", showbackground=True,
                                zerolinecolor="white"),
                     zaxis=dict(backgroundcolor="rgb(230, 230,200)", gridcolor="white", showbackground=True,
                                zerolinecolor="white", ))
        figure.update_layout(title=go.layout.Title(text=title, ), scene=scene)
    else:
        figure.update_layout(title=go.layout.Title(text=title, ),
                             xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=xLabel, )),
                             yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text=yLabel, )))
    return figure


def plot_2d(xCols, yCol, X, y, yFits):
    plot = ''
    for xCol in xCols:
        x = X.sort_values(xCol).loc[:, xCol]
        d = go.Scatter(x=x, y=y, mode='markers', name=yCol)
        traces = [d]
        for model in sorted(yFits.keys()):
            traces.append(go.Scatter(x=x, y=yFits[model], mode='lines', name=model))
        figure = format(go.Figure(data=traces), xCol, yCol)
        plot += str(po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py))
    return plot


def plot_3d(xCols, yCol, X, y, yFits):
    plot = ''
    for xCol in itertools.combinations(xCols, 2):
        x1 = X.sort_values(xCol[0]).iloc[:, 0]
        x2 = X.sort_values(xCol[1]).iloc[:, 1]
        d = go.Scatter3d(x=x1, y=x2, z=y, mode='markers', name=yCol, marker=dict(size=3))
        traces = [d]
        for i, model in enumerate(sorted(yFits.keys())):
            # traces.append(
            #     go.Mesh3d(
            #         x=x1,
            #         y=x2,
            #         z=yFits[model],
            #         color=colors[i],
            #         opacity=0.25,
            #         name='{} Surface'.format(model),
            #     )
            #
            # )
            traces.append(go.Scatter3d(x=x1, y=x2, z=yFits[model], line=dict(color=colors[i]), mode='lines',
                                       name='{}'.format(model), )

                          )
        plot += str(
            po.plot(format(go.Figure(data=traces), xCol[0], xCol[1], yCol, nvars=3), auto_open=False, output_type='div',
                    configuration.py=configuration.py))
    return plot


def get_scatter(x, y, xData, yData, yFits):
    plot2d = plot_2d(x, y, xData, yData, yFits)  # TODO
    plot3d = plot_3d(x, y, xData, yData, yFits)
    plot = plot2d + plot3d
    return plot


def get_histogram(x, y, yData, project):
    histogramData = get_histogram_data(path.fitPath.format(project), x, y, yData)  # TODO
    return histogram(histogramData, y)


from util.colors import colors

configuration.py = {'displaylogo': False, }


def histogram(data, yCol):
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Histogram(x=data.loc[:, col].values, histnorm='percent', name=col, opacity=0.75, nbinsx=20, ))
    fig.update_layout(barmode='overlay', bargap=0.2, bargroupgap=0.1, xaxis_title_text=yCol,
                      yaxis_title_text='Percent (%)')
    div = po.plot(fig, auto_open=False, output_type='div', configuration.py=configuration.py)
    return div


def plot2d(X, y, yFits, xCols, yCol):
    plot = ''
    for xCol in xCols:
        x = X.sort_values(xCol).loc[:, xCol]
        d = go.Scatter(x=x, y=y, mode='markers', name=yCol)
        traces = [d]
        for model in sorted(yFits.keys()):
            traces.append(go.Scatter(x=x, y=yFits[model], mode='lines', name=model))
        figure = format(go.Figure(data=traces), xCol, yCol)
        plot += str(po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py))

    for xCol in itertools.combinations(xCols, 2):
        x1 = X.sort_values(xCol[0]).iloc[:, 0]
        x2 = X.sort_values(xCol[1]).iloc[:, 1]
        d = go.Scatter3d(x=x1, y=x2, z=y, mode='markers', name=yCol, marker=dict(size=3))
        traces = [d]
        for i, model in enumerate(sorted(yFits.keys())):
            # traces.append(
            #     go.Mesh3d(
            #         x=x1,
            #         y=x2,
            #         z=yFits[model],
            #         color=colors[i],
            #         opacity=0.25,
            #         name='{} Surface'.format(model),
            #     )
            #
            # )
            traces.append(go.Scatter3d(x=x1, y=x2, z=yFits[model], line=dict(color=colors[i]), mode='lines',
                                       name='{}'.format(model), )

                          )
        plot += str(
            po.plot(format(go.Figure(data=traces), xCol[0], xCol[1], yCol, nvars=3), auto_open=False, output_type='div',
                    configuration.py=configuration.py))
    return plot


import tool.general.path
import tool.general.util

configuration.py = {'displaylogo': False, 'responsive': False, 'scrollZoom': False, }


def plot_2d(xCols, yCol, X, y, yFits):
    plot = ''
    for xCol in xCols:
        x = X.sort_values(xCol).loc[:, xCol]
        d = go.Scatter(x=x, y=y, mode='markers', name=yCol, )
        traces = [d]
        for model in sorted(yFits.keys()):
            traces.append(go.Scatter(x=x, y=yFits[model], mode='lines', name=model, ))
        figure = format(go.Figure(data=traces), xCol, yCol)
        plot += str(po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py, ))
    return plot


def plot_3d(xCols, yCol, X, y, yFits):
    plot = ''
    for xCol in itertools.combinations(xCols, 2):
        x1 = X.sort_values(xCol[0]).iloc[:, 0]
        x2 = X.sort_values(xCol[1]).iloc[:, 1]
        d = go.Scatter3d(x=x1, y=x2, z=y, mode='markers', name=yCol, marker=dict(size=3), )
        traces = [d]
        for i, model in enumerate(sorted(yFits.keys())):
            # traces.append(
            #     go.Mesh3d(
            #         x=x1,
            #         y=x2,
            #         z=yFits[model],
            #         color=colors[i],
            #         opacity=0.25,
            #         name='{} Surface'.format(model),
            #     )
            #
            # )
            traces.append(
                go.Scatter3d(x=x1, y=x2, z=yFits[model], line=dict(color=tool.general.util.colors[i]), mode='lines',
                             name='{}'.format(model), )

            )
        plot += str(
            po.plot(format(figure=go.Figure(data=traces), xLabel=xCol[0], yLabel=xCol[1], zLabel=yCol, nvars=3, ),
                    auto_open=False, output_type='div', configuration.py=configuration.py, ))
    return plot


def format(figure, xLabel='', yLabel='', zLabel='', title='', nvars=2):
    bgX = "rgb(200, 200, 230)"
    bgY = "rgb(230, 200, 230)"
    bgZ = "rgb(230, 230, 200)"
    fontColor = "black"
    paperBgColor = "white"
    plotBgColor = "#f1f7fc"
    gridColor = 'rgb(50, 50, 50)'
    zeroLineColor = 'rgb(0, 0, 0)'
    font = dict(  # family='Courier New, monospace',
        # size=18,
        color=fontColor, )
    if nvars == 3:
        scene = dict(xaxis_title=xLabel, yaxis_title=yLabel, zaxis_title=zLabel,
                     xaxis=dict(backgroundcolor=bgX, gridcolor=gridColor, showbackground=True, zerolinecolor="white", ),
                     yaxis=dict(backgroundcolor=bgY, gridcolor=gridColor, showbackground=True, zerolinecolor="white"),
                     zaxis=dict(backgroundcolor=bgZ, gridcolor=gridColor, showbackground=True, zerolinecolor="white", ))
        figure.update_layout(scene=scene)
    else:
        figure.update_layout(
            xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text='<b>{}<b>'.format(xLabel)), gridcolor=gridColor,
                                  linewidth=2, linecolor='black', mirror=True, showline=True, ticks='outside',
                                  zerolinecolor=zeroLineColor, ),
            yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text='<b>{}<b>'.format(yLabel)), gridcolor=gridColor,
                                  linewidth=2, linecolor='black', mirror=True, showline=True, ticks='outside',
                                  zerolinecolor=zeroLineColor, ), )
    figure.update_layout(font=font, paper_bgcolor=paperBgColor, plot_bgcolor=plotBgColor,
                         title=go.layout.Title(text='<b>{}<b>'.format(title)), )
    return figure


def plot2d(data, parameters, yFits):
    plot = ''
    y = data['y']
    for xCol in parameters['x']:
        x = data['x'].sort_values(xCol).loc[:, xCol]
        d = go.Scatter(x=x, y=y, mode='markers', name=parameters['y'], )
        traces = [d]
        for model in sorted(yFits.keys()):
            traces.append(go.Scatter(x=x, y=yFits[model], mode='lines', name=model, ))

        figure = format(figure=go.Figure(data=traces), xLabel=xCol, yLabel=parameters['y'], )
        plot += str(po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py), )

    for xCol in itertools.combinations(parameters['x'], 2):
        x1 = data['x'].sort_values(xCol[0]).iloc[:, 0]
        x2 = data['x'].sort_values(xCol[1]).iloc[:, 1]
        d = go.Scatter3d(x=x1, y=x2, z=y, mode='markers', name=parameters['y'], marker=dict(size=3), )
        traces = [d]
        for i, model in enumerate(sorted(yFits.keys())):
            # traces.append(
            #     go.Mesh3d(
            #         x=x1,
            #         y=x2,
            #         z=yFits[model],
            #         # color=colors[i],
            #         opacity=0.25,
            #         name='{} Surface'.format(model),
            #     )
            #
            # )
            traces.append(
                go.Scatter3d(x=x1, y=x2, z=yFits[model], line=dict(color=tool.general.util.colors[i]), mode='lines',
                             name='{}'.format(model), )

            )
        plot += str(po.plot(
            figure_or_data=format(figure=go.Figure(data=traces), xLabel=xCol[0], yLabel=xCol[1], zLabel=parameters['y'],
                                  nvars=3, ), auto_open=False, output_type='div', configuration.py=configuration.py, ))
    return plot


import tool.general.path
import tool.general.util

configuration.py = {'displaylogo': False, 'responsive': False, 'scrollZoom': False, }


def new_plot(dashboardID, filename, xCol=None, yCol=None):
    filePath = os.path.join(tool.general.path.uploadsPath, dashboardID, filename)
    columns = pd.read_csv(filePath, nrows=1).columns
    if xCol and yCol:
        data = pd.read_csv(filePath, usecols=[xCol, yCol], )
    else:
        data = pd.read_csv(filePath, usecols=[columns[0], columns[1]], )
    if not xCol:
        xCol = columns[0]
    if not yCol:
        yCol = columns[1]
    xData = data.loc[:, xCol]
    yData = data.loc[:, yCol]
    trace = go.Scatter(x=xData, y=yData, mode='markers', name=yCol, )
    figure = format(go.Figure(data=trace), xCol, yCol, )
    div = po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py, )

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=yData.values, histnorm='percent', name=yCol, opacity=0.75, ))
    fig.update_layout(barmode='overlay', bargap=0.2, bargroupgap=0.1, width='100%', height='100%',
                      xaxis_title_text=yCol, yaxis_title_text='Percent (%)', )
    div2 = po.plot(fig, auto_open=False, output_type='div', configuration.py=configuration.py, )
    return div, div2


def plot_2d(xCols, yCol, X, y, yFits):
    plot = ''
    for xCol in xCols:
        x = X.sort_values(xCol).loc[:, xCol]
        d = go.Scatter(x=x, y=y, mode='markers', name=yCol, )
        traces = [d]
        for model in sorted(yFits.keys()):
            traces.append(go.Scatter(x=x, y=yFits[model], mode='lines', name=model, ))
        figure = format(go.Figure(data=traces), xCol, yCol)
        plot += str(po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py, ))
    return plot


def plot_3d(xCols, yCol, X, y, yFits):
    plot = ''
    for xCol in itertools.combinations(xCols, 2):
        x1 = X.sort_values(xCol[0]).iloc[:, 0]
        x2 = X.sort_values(xCol[1]).iloc[:, 1]
        d = go.Scatter3d(x=x1, y=x2, z=y, mode='markers', name=yCol, marker=dict(size=3), )
        traces = [d]
        for i, model in enumerate(sorted(yFits.keys())):
            # traces.append(
            #     go.Mesh3d(
            #         x=x1,
            #         y=x2,
            #         z=yFits[model],
            #         color=colors[i],
            #         opacity=0.25,
            #         name='{} Surface'.format(model),
            #     )
            #
            # )
            traces.append(
                go.Scatter3d(x=x1, y=x2, z=yFits[model], line=dict(color=tool.general.util.colors[i]), mode='lines',
                             name='{}'.format(model), )

            )
        plot += str(
            po.plot(format(figure=go.Figure(data=traces), xLabel=xCol[0], yLabel=xCol[1], zLabel=yCol, nvars=3, ),
                    auto_open=False, output_type='div', configuration.py=configuration.py, ))
    return plot


def plot2d(data, parameters, yFits):
    plot = ''
    y = data['y']
    for xCol in parameters['x']:
        x = data['x'].sort_values(xCol).loc[:, xCol]
        d = go.Scatter(x=x, y=y, mode='markers', name=parameters['y'], )
        traces = [d]
        for model in sorted(yFits.keys()):
            traces.append(go.Scatter(x=x, y=yFits[model], mode='lines', name=model, ))

        figure = format(figure=go.Figure(data=traces), xLabel=xCol, yLabel=parameters['y'], )
        plot += str(po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py), )

    for xCol in itertools.combinations(parameters['x'], 2):
        x1 = data['x'].sort_values(xCol[0]).iloc[:, 0]
        x2 = data['x'].sort_values(xCol[1]).iloc[:, 1]
        d = go.Scatter3d(x=x1, y=x2, z=y, mode='markers', name=parameters['y'], marker=dict(size=3), )
        traces = [d]
        for i, model in enumerate(sorted(yFits.keys())):
            # traces.append(
            #     go.Mesh3d(
            #         x=x1,
            #         y=x2,
            #         z=yFits[model],
            #         # color=colors[i],
            #         opacity=0.25,
            #         name='{} Surface'.format(model),
            #     )
            #
            # )
            traces.append(
                go.Scatter3d(x=x1, y=x2, z=yFits[model], line=dict(color=tool.general.util.colors[i]), mode='lines',
                             name='{}'.format(model), )

            )
        plot += str(po.plot(
            figure_or_data=format(figure=go.Figure(data=traces), xLabel=xCol[0], yLabel=xCol[1], zLabel=parameters['y'],
                                  nvars=3, ), auto_open=False, output_type='div', configuration.py=configuration.py, ))
    return plot


def make_the_scatter_plot_2d(x, y):
    d = go.Scatter(x=x, y=y, mode='markers',  # name='',
                   )
    figure = format(go.Figure(data=[d]), 'x', 'y')
    plt = str(po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py, ))
    return plt


import tool.general
import tool.path

configuration.py = {'displaylogo': False, 'responsive': False, 'scrollZoom': False, }


def new_plot(dashboardID, filename, xCol=None, yCol=None):
    filePath = os.path.join(tool.path.uploadsPath, dashboardID, filename)
    columns = pd.read_csv(filePath, nrows=1).columns
    if xCol and yCol:
        data = pd.read_csv(filePath, usecols=[xCol, yCol], )
    else:
        data = pd.read_csv(filePath, usecols=[columns[0], columns[1]], )
    if not xCol:
        xCol = columns[0]
    if not yCol:
        yCol = columns[1]
    xData = data.loc[:, xCol]
    yData = data.loc[:, yCol]
    trace = go.Scatter(x=xData, y=yData, mode='markers', name=yCol, )
    figure = format(go.Figure(data=trace), xCol, yCol, )
    div = po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py, )

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=yData.values, histnorm='percent', name=yCol, opacity=0.75, ))
    fig.update_layout(barmode='overlay', bargap=0.2, bargroupgap=0.1, width='100%', height='100%',
                      xaxis_title_text=yCol, yaxis_title_text='Percent (%)', )
    div2 = po.plot(fig, auto_open=False, output_type='div', configuration.py=configuration.py, )
    return div, div2


def plot_2d(xCols, yCol, X, y, yFits):
    plot = ''
    for xCol in xCols:
        x = X.sort_values(xCol).loc[:, xCol]
        d = go.Scatter(x=x, y=y, mode='markers', name=yCol, )
        traces = [d]
        for model in sorted(yFits.keys()):
            traces.append(go.Scatter(x=x, y=yFits[model], mode='lines', name=model, ))
        figure = format(go.Figure(data=traces), xCol, yCol)
        plot += str(po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py, ))
    return plot


def plot_3d(xCols, yCol, X, y, yFits):
    plot = ''
    for xCol in itertools.combinations(xCols, 2):
        x1 = X.sort_values(xCol[0]).iloc[:, 0]
        x2 = X.sort_values(xCol[1]).iloc[:, 1]
        d = go.Scatter3d(x=x1, y=x2, z=y, mode='markers', name=yCol, marker=dict(size=3), )
        traces = [d]
        for i, model in enumerate(sorted(yFits.keys())):
            # traces.append(
            #     go.Mesh3d(
            #         x=x1,
            #         y=x2,
            #         z=yFits[model],
            #         color=colors[i],
            #         opacity=0.25,
            #         name='{} Surface'.format(model),
            #     )
            #
            # )
            traces.append(
                go.Scatter3d(x=x1, y=x2, z=yFits[model], line=dict(color=tool.general.colors[i]), mode='lines',
                             name='{}'.format(model), )

            )
        plot += str(
            po.plot(format(figure=go.Figure(data=traces), xLabel=xCol[0], yLabel=xCol[1], zLabel=yCol, nvars=3, ),
                    auto_open=False, output_type='div', configuration.py=configuration.py, ))
    return plot


def plot2d(data, parameters, yFits):
    plot = ''
    y = data['y']
    for xCol in parameters['x']:
        x = data['x'].sort_values(xCol).loc[:, xCol]
        d = go.Scatter(x=x, y=y, mode='markers', name=parameters['y'], )
        traces = [d]
        for model in sorted(yFits.keys()):
            traces.append(go.Scatter(x=x, y=yFits[model], mode='lines', name=model, ))

        figure = format(figure=go.Figure(data=traces), xLabel=xCol, yLabel=parameters['y'], )
        plot += str(po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py), )

    for xCol in itertools.combinations(parameters['x'], 2):
        x1 = data['x'].sort_values(xCol[0]).iloc[:, 0]
        x2 = data['x'].sort_values(xCol[1]).iloc[:, 1]
        d = go.Scatter3d(x=x1, y=x2, z=y, mode='markers', name=parameters['y'], marker=dict(size=3), )
        traces = [d]
        for i, model in enumerate(sorted(yFits.keys())):
            # traces.append(
            #     go.Mesh3d(
            #         x=x1,
            #         y=x2,
            #         z=yFits[model],
            #         # color=colors[i],
            #         opacity=0.25,
            #         name='{} Surface'.format(model),
            #     )
            #
            # )
            traces.append(
                go.Scatter3d(x=x1, y=x2, z=yFits[model], line=dict(color=tool.general.colors[i]), mode='lines',
                             name='{}'.format(model), )

            )
        plot += str(po.plot(
            figure_or_data=format(figure=go.Figure(data=traces), xLabel=xCol[0], yLabel=xCol[1], zLabel=parameters['y'],
                                  nvars=3, ), auto_open=False, output_type='div', configuration.py=configuration.py, ))
    return plot


import itertools
import os

import pandas as pd
import plotly.graph_objs as go
import plotly.offline as po
import ts_util.general
import ts_util.path

configuration.py = {'displaylogo': False, 'responsive': False, 'scrollZoom': False, }


def new_plot(dashboardID, filename, xCol=None, yCol=None):
    filePath = os.path.join(ts_util.path.uploadsPath, dashboardID, filename)
    columns = pd.read_csv(filePath, nrows=1).columns
    if xCol and yCol:
        data = pd.read_csv(filePath, usecols=[xCol, yCol], )
    else:
        data = pd.read_csv(filePath, usecols=[columns[0], columns[1]], )
    if not xCol:
        xCol = columns[0]
    if not yCol:
        yCol = columns[1]
    xData = data.loc[:, xCol]
    yData = data.loc[:, yCol]
    trace = go.Scatter(x=xData, y=yData, mode='markers', name=yCol, )
    figure = format(go.Figure(data=trace), xCol, yCol, )
    div = po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py, )

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=yData.values, histnorm='percent', name=yCol, opacity=0.75, ))
    fig.update_layout(barmode='overlay', bargap=0.2, bargroupgap=0.1, width='100%', height='100%',
                      xaxis_title_text=yCol, yaxis_title_text='Percent (%)', )
    div2 = po.plot(fig, auto_open=False, output_type='div', configuration.py=configuration.py, )
    return div, div2


def plot_2d(xCols, yCol, X, y, yFits):
    plot = ''
    for xCol in xCols:
        x = X.sort_values(xCol).loc[:, xCol]
        d = go.Scatter(x=x, y=y, mode='markers', name=yCol, )
        traces = [d]
        for model in sorted(yFits.keys()):
            traces.append(go.Scatter(x=x, y=yFits[model], mode='lines', name=model, ))
        figure = format(go.Figure(data=traces), xCol, yCol)
        plot += str(po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py, ))
    return plot


def plot_3d(xCols, yCol, X, y, yFits):
    plot = ''
    for xCol in itertools.combinations(xCols, 2):
        x1 = X.sort_values(xCol[0]).iloc[:, 0]
        x2 = X.sort_values(xCol[1]).iloc[:, 1]
        d = go.Scatter3d(x=x1, y=x2, z=y, mode='markers', name=yCol, marker=dict(size=3), )
        traces = [d]
        for i, model in enumerate(sorted(yFits.keys())):
            # traces.append(
            #     go.Mesh3d(
            #         x=x1,
            #         y=x2,
            #         z=yFits[model],
            #         color=colors[i],
            #         opacity=0.25,
            #         name='{} Surface'.format(model),
            #     )
            #
            # )
            traces.append(
                go.Scatter3d(x=x1, y=x2, z=yFits[model], line=dict(color=ts_util.general.colors[i]), mode='lines',
                             name='{}'.format(model), )

            )
        plot += str(
            po.plot(format(figure=go.Figure(data=traces), xLabel=xCol[0], yLabel=xCol[1], zLabel=yCol, nvars=3, ),
                    auto_open=False, output_type='div', configuration.py=configuration.py, ))
    return plot


def plot2d(data, parameters, yFits):
    plot = ''
    y = data['y']
    for xCol in parameters['x']:
        x = data['x'].sort_values(xCol).loc[:, xCol]
        d = go.Scatter(x=x, y=y, mode='markers', name=parameters['y'], )
        traces = [d]
        for model in sorted(yFits.keys()):
            traces.append(go.Scatter(x=x, y=yFits[model], mode='lines', name=model, ))

        figure = format(figure=go.Figure(data=traces), xLabel=xCol, yLabel=parameters['y'], )
        plot += str(po.plot(figure, auto_open=False, output_type='div', configuration.py=configuration.py), )

    for xCol in itertools.combinations(parameters['x'], 2):
        x1 = data['x'].sort_values(xCol[0]).iloc[:, 0]
        x2 = data['x'].sort_values(xCol[1]).iloc[:, 1]
        d = go.Scatter3d(x=x1, y=x2, z=y, mode='markers', name=parameters['y'], marker=dict(size=3), )
        traces = [d]
        for i, model in enumerate(sorted(yFits.keys())):
            # traces.append(
            #     go.Mesh3d(
            #         x=x1,
            #         y=x2,
            #         z=yFits[model],
            #         # color=colors[i],
            #         opacity=0.25,
            #         name='{} Surface'.format(model),
            #     )
            #
            # )
            traces.append(
                go.Scatter3d(x=x1, y=x2, z=yFits[model], line=dict(color=ts_util.general.colors[i]), mode='lines',
                             name='{}'.format(model), )

            )
        plot += str(po.plot(
            figure_or_data=format(figure=go.Figure(data=traces), xLabel=xCol[0], yLabel=xCol[1], zLabel=parameters['y'],
                                  nvars=3, ), auto_open=False, output_type='div', configuration.py=configuration.py, ))
    return plot


def export_plot(plot, path: str):
    """
    plot: matplotlib object.
    Saves the matplotlib plot object's figure as a png in path.
    Path should be the absolute path including the .png extension.
    e.g. export_plot(plot, 'C:\image.png')
    """
    fig = plot.get_figure()
    fig.savefig(path, bbox_inches='tight')


def plot_scatter(x: list, y: list, xLabel: str, yLabel: str, title: str):
    """
    x, y: list, numpy array, pandas series.
    Plots a 2D scatterplot, formats the plot and returns it.
    """
    scatterPlot = sns.lineplot(x=x, y=y)
    scatterPlot.set(xlabel=xLabel, ylabel=yLabel, title=title, )
    scatterPlot.yaxis.set_major_locator(ticker.MultipleLocator(1))
    return scatterPlot


#######################################################################################################################
def plot_corrcov(corr, cov):
    plot_heatmap(corr, 'Correlation')
    plot_heatmap(cov, 'Covariance')


def plot_corrcov(df, saveDir, fileName=('Correlation', 'Covariance')):
    """
    plot a heatmap of the correlation matrix, and a heatmap of the covariance matrix.
    df: pandas DataFrame
    return: pandas DataFrame, pandas DataFrame
    """
    corr, cov = math_api.get_corr_cov(df)
    plot_heatmap(corr, title='', saveDir=saveDir, fileName=fileName[0])
    plot_heatmap(cov, title='', saveDir=saveDir, fileName=fileName[1])
    return corr, cov


def plot_corrcov(df, savePath):
    """
    plot a heatmap of the correlation matrix, and a heatmap of the covariance matrix.
    df: pandas DataFrame
    return: pandas DataFrame, pandas DataFrame
    """
    corr, cov = StatsCollector.get_corr_cov(df)
    plot_heatmap(corr, name='Correlation', savePath=savePath)
    plot_heatmap(cov, name='Covariance', savePath=savePath)
    return corr, cov


def plot(func):
    """
    Plot, format, and export.
    func:
    return:
    """

    @wraps(func)
    def make_plot(*args, **kwargs):
        plt.gcf().clear()
        name, savePath, plot, payload = func(*args, **kwargs)
        format_plot(args[0].label, args[0], plotStyle=name, plot=plt, ax=plot)
        export_plot(plot, savePath=savePath, fileName=args[0].label)
        figs.append((plot.get_figure(), savePath, args[0].label))
        return plot, payload

    return make_plot


def plot_ci(t, s_err, n, x, x2, y2, ax=None):
    # Return an axes of confidence bands using a simple approach.
    if ax is not None:
        ax = plt.gca()
        ci = t * s_err * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        ax.fill_between(x2, y2 + ci, y2 - ci, color='#5CB0C2', edgecolor='#5CB0C2', linestyle='-', alpha=0.2, zorder=-1)
        ax.plot(x2, y2 - ci, '--', color='#D2691E', alpha=0.5, zorder=-1, label='95% Confidence Interval')
        ax.plot(x2, y2 + ci, '--', color='#D2691E', alpha=0.5, zorder=-1)
    return ax


def plot_pi(t, s_err, x, x2, y2, ax=None):
    pi = t * s_err * np.sqrt(1 + 1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    ax.fill_between(x2, y2 + pi, y2 - pi, color='None', linestyle='--', alpha=0.5)
    ax.plot(x2, y2 - pi, '-.', color='0.5', alpha=0.5, label='95% Prediction Limits', )
    ax.plot(x2, y2 + pi, '-.', color='0.5', alpha=0.5)
    return ax


def plot_regression_and_ci(v, ax):
    """
    This function is to be called within plot_scatter()
    col: str
    df: Pandas Series
    min: minimum value of col
    max: maximum value of col
    """

    def plot_r_ci(x, x2, y2, y_model, t, s_err, n, ax, f):
        # Regression6, Confidence Interval, and Prediction Interval
        ax.plot(x, y_model, '-', color='0.1', linewidth=1.5, alpha=0.9, label='Regression6 Fit',  # '$' + f + '$',
                zorder=-2)
        plot_ci(t, s_err, n, x, x2, y2, ax=ax)
        plot_pi(t, s_err, x, x2, y2, ax=ax)

    x = np.array(v.index)
    y = v
    p, cov, yModel, fam, errors = Regression.get_model(x, y)
    n, m, dF, t = Regression.get_ci_stats(y, p)
    errors = Regression.estimate_error(y, yModel, dF, errors)
    x2 = np.linspace(np.min(x), np.max(x), len(x))
    y2 = yModel
    fit = ExpressionGenerator.function_string(*p, fam=fam, y=v.label, evaluate=True, latex=False)
    R2 = errors['R2']
    if R2 > 0.05:  # mse < 76057057
        plot_r_ci(x, x2, y2, yModel, t, errors['standard error'], n, ax, fit)
    return (fit, errors)


def set_style():
    sns.set_context('paper', rc={'lines.linewidth': 1, 'figure.figsize': (1, 1), 'figure.facecolor': 'white',
                                 'font.family': ['sans-serif', ]})
    sns.set(font='serif', font_scale=.7)
    sns.set_style('ticks', {'font.family': 'serif', 'font.serif': ['Times', 'Palatino', 'serif']})
    plt.legend(fontsize=7, loc='best', frameon=False, borderaxespad=5)
    sns.despine(top=True, right=True, trim=True)


def format_plot(col, v, plotStyle, plot, ax):
    """
    col: string
    v: pandas Series
    plotStyle: string
    plot0: Matplotlib plt
    ax: Matplotlib figure
    """
    plot.title(col + ' ' + plotStyle, fontsize='3', fontweight='bold', fontname='Georgia')

    if plotStyle != 'Statistics Summary':
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1.)
        ax.spines['left'].set_linewidth(1.)

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_linewidth(1.)
        ax.spines['right'].set_linewidth(1.)

        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')
    if ax and plotStyle != 'Statistics Summary':
        for s in ['bottom', 'left', ]:
            ax.spines[s].set_color('0.5')
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

    if plotStyle == 'Bar Plot':
        plot.ylabel('Quantity')

    elif plotStyle == 'Box Plot':
        plot.xlim(np.min(v) - 1, np.max(v) + 1)
        sns.boxplot(capsize=.5)

    elif plotStyle == 'Distribution Fit Plot':
        plot.ylabel('Probability')
        plot.xlim(np.min(v) - 1, np.max(v) + 1)

    elif plotStyle == 'Distribution Plot' or plotStyle == 'Histogram':
        plot.ylabel('Frequency')
        plot.xlim(np.min(v) - 1, np.max(v) + 1)

    elif plotStyle == 'Heat Map':
        pass

    elif plotStyle == 'Scatter Plot':
        xRange = np.max(v.index.values) - np.min(v.index.values)
        strata = 5
        xTicks = np.arange(np.min(v.index.values) - 1, np.max(v.index.values) + 1, step=int(xRange / strata))
        plot.xticks(xTicks)
        plot.xlim(np.min(v.index.values) - 1, np.max(v.index.values) + 1)
        # plt.gca().set_aspect('equal', adjustable='box')
        pass

    elif plotStyle == 'Statistics Summary':
        pass

    elif plotStyle == 'Violin Plot':
        plt.ylabel('Frequency')
        plot.xlim(np.min(v) - 1, np.max(v) + 1)
        sns.violinplot(scale='count', linewidth=3, )
    set_style()


@plot
def plot_bar(v, name='Bar Plot', stats=None, savePath=BAR_PLOTS_DIR):
    m = v.value_counts()
    barPlot = sns.barplot(x=m.index, y=m.values)
    if stats:
        plot_bar_stats(v, stats, barPlot)
    return name, savePath, barPlot, None


def plot_box_stats(v, stats, ax):
    # props = dict(boxstyle='round', alpha=0.5, color=sns.color_palette()[0])
    minimum = s_round(stats['Lower IQR'], 2)
    q1 = s_round(stats['First Quartile'], 2)
    median = s_round(stats['Median'], 2)
    q2 = s_round(stats['Third Quartile'], 2)
    maximum = s_round(stats['Upper IQR'], 2)
    iqr = s_round(stats['IQR'], 2)

    cellText = [[str(minimum)], [str(q1)], [str(median)], [str(q2)], [str(maximum)], [str(iqr)]]
    rowLabels = ['$Minimum$', '$25^{th}$ $Percentile$', '$Median$', '$75^{th}$ $Percentile$', '$Maximum$', '$IQR$']
    table = plt.table(cellText=cellText, rowLabels=rowLabels,
                      bbox=[1.25, 0.65, 0.05, 0.35])  # [1.025, 0.6475, 0.75, 0.75])

    for key, cell in table.get_celld().items():
        cell.set_linewidth(1)

    table.auto_set_column_width(0)
    table.scale(2, 3)


@plot
def plot_box(v, name='Box Plot', stats=None, savePath=BOX_PLOTS_DIR):
    """
    v: pandas Series, (vector)
    return: void
    """
    boxPlot = sns.boxplot(y=v, width=.25)
    # if stats0 != None:
    #    plot_box_stats(v, stats0, boxPlot)
    return name, savePath, boxPlot, None


def plot_heatmap(matrix, name=None, savePath=HEATMAP_PLOTS_DIR):
    plt.gcf().clear()
    heatmapPlot = sns.heatmap(matrix, annot=False)
    export_plot(heatmapPlot, savePath=savePath, fileName=name)


@plot
def plot_histogram(v, name='Distribution Plot', savePath=DISTRIBUTION_PLOTS_DIR):
    """
    v: pandas Series, (vector)
    fit: bool
    return:
    """
    distPlot = sns.distplot(v, norm_hist=False, kde=False, bins=None, rug=True)
    return name, savePath, distPlot, None


@plot
def plot_scatter(v, name='Scatter Plot', fit=True, savePath=SCATTER_PLOTS_DIR):
    """
    df: pandas Series (vector)
    fit: bool
    return: void
    """
    scatterPlot = sns.regplot(np.arange(len(v)), v, order=2, fit_reg=False)  # , label=v.name)
    if fit:
        regressionFit, regressionError = plot_regression_and_ci(v, scatterPlot)
        return name, savePath, scatterPlot, (regressionFit, regressionError)
    return name, savePath, scatterPlot, None


def plot_stats(df, statsDict=None, savePath=TABLE_PLOTS_DIR):
    plt.gcf().clear()
    cellText = []
    colLabels = []
    s = 0
    for col in df:
        f = sorted(statsDict[col])
        cellText.append([str(statsDict[col][s]) for s in f])
        colLabels.append(col)
        if s == 0:
            rowLabels = [s for s in f]
            s += 1
    cellText = pd.DataFrame(cellText).T.as_matrix()
    table = plt.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, loc='upper center')
    table.auto_set_column_width(0)
    table.scale(2, 3)
    plt.axis('off')
    format_plot('', df, 'Statistics Summary', plt, table)
    export_plot(table, savePath, '' + TABLE_PLOTS_SUFFIX)


@plot
def plot_topN_distributions(v, N, name='Distribution Fit Plot', savePath=DISTRIBUTION_FIT_PLOTS_DIR):
    """
    plot the top N distributions as well as a Gaussian Kernel Density Estimate
    :param v:
    :param N:
    :param name:
    :param savePath:
    :return:
    """
    distPlot = sns.distplot(v, kde=False, norm_hist=True)
    pdfs, topNames, topParams, topNSSE = DistributionFitter.best_fit_distribution(v)
    density, bandwidth = DistributionFitter.density_kde(v)
    density.plot(lw=2, label="Gaussian Kernel Density Estimate\n", legend=True)
    fit, names, errors = ExpressionGenerator.distribution_string(topNames, topParams, topNSSE, y='PDF', latex=False)
    for i in pdfs:
        # paramNames = (topDists[i].shapes + ', loc, scale').split(', ') if topDists[i].shapes else ['loc', 'scale']
        # param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(paramNames, topParams)])
        distPlot = pdfs[i].plot(lw=2, label=names[topNames[i]], legend=True)
    f = {names[k]: (fit[k], errors[k]) for k in names}
    return name, savePath, distPlot, f


def plot_violin_stats(v, stats, ax):
    # props = dict(boxstyle='round', alpha=0.5, color=sns.color_palette()[0])
    mean = s_round(stats['Mean'], 2)
    mode = s_round(stats['Mode'], 2)
    std = s_round(stats['Standard\nDeviation'], 2)
    variance = s_round(stats['Variance'], 2)
    skew = s_round(stats['Skew'], 2)
    kurtosis = s_round(stats['Kurtosis'], 2)
    cellText = [[str(mean)], [str(mode)], [str(std)], [str(variance)], [str(skew)], [str(kurtosis)]]
    rowLabels = ['$Mean$', '$Mode$', '$Standard$\n$Deviation$', '$Variance$', '$Skew$', '$Kurtosis$']
    table = plt.table(cellText=cellText, rowLabels=rowLabels, bbox=[1.25, 0, 0.09, 1])  # [1.025, 0.6475, 0.75, 0.75])
    for key, cell in table.get_celld().items():
        cell.set_linewidth(1)
    table.auto_set_column_width(0)
    table.scale(2, 3)


@plot
def plot_violin(v, name='Violin Plot', stats=None, savePath=VIOLIN_PLOTS_DIR):
    violinPlot = sns.violinplot(y=v)
    # if stats0 != None:
    #    plot_violin_stats(v, stats0, violinPlot)
    return name, savePath, violinPlot, None


def plot_stats_comparison(stats, stat):
    """
    stats0: pandas DataFrame
    stat: str
    """
    plt.gcf().clear()
    h = {}
    for col in stats:
        h[col] = stats[col][stat]

    barplot = sns.barplot(x=[col for col in h], y=[h[col] for col in h])
    plt.title(stat + ' Comparison')
    format_plot(stat, stats, 'Bar Plot', plt, barplot)
    fig = barplot.get_figure()
    fig.savefig(os.path.join(STAT_COMPARISON_PLOTS_DIR, stat[:3] + '_comparison'))


# MAIN
@timer
def generate_all_plots(df, dtypes, savePath=PLOTS_DIR):
    """
    This function returns all the information used in the creation of the plots
    df: pandas DataFrame
    dtypes: list of str, 'Numerical' or 'Categorical'
    savePath: str, path where to save the plots as png
    return: list of str, dict, dict, dict, dict
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clear_directories(savePath)
        variables = []
        categoricals = {}
        numericals = {}
        distFits = {}
        regression = {}
        dfNum = []
        dfCat = []
        for col, label in zip(df, dtypes):
            if label == 'Categorical':
                categoricals[col] = StatsCollector.collect_stats(df[col], label)
                dfCat.append(col)
                plot_bar(df[col], stats=categoricals[
                    col])  # plot_scatter(df[col], fit=False)  # plot_stats(df[col], stats0=['Mode'])

            elif label == 'Numerical':
                numericals[col] = StatsCollector.collect_stats(df[col], label)
                dfNum.append(col)  # TODO not sure what this is
                plot_histogram(df[col], savePath=os.path.join(savePath, 'Distribution'))
                _, distFits[col] = plot_topN_distributions(df[col], 2,
                                                           savePath=os.path.join(savePath, 'DistributionFit'))
                _, regression[col] = plot_scatter(df[col], fit=True, savePath=os.path.join(savePath, 'Scatter'))
                plot_box(df[col], stats=numericals[col], savePath=os.path.join(savePath, 'Box'))
                plot_violin(df[col], stats=numericals[col], savePath=os.path.join(savePath, 'Violin'))
            variables.append(col)
        """if 'Numerical' in dtypes:
            plot_corrcov(df[dfNum],
                         savePath=os.path.join(savePath, 'Heatmap'))
            plot_stats(df[dfNum], statsDict=numericals,
                       savePath=os.path.join(savePath, 'StatisticComparison'))"""
        # map(export_plot, figs)
        return variables, numericals, categoricals, distFits, regression


def plot(func):
    """
    Plot, format, and export.
    func:
    return:
    """

    @wraps(func)
    def make_plot(*args, **kwargs):
        plt.gcf().clear()
        name, savePath, plot, payload = func(*args, **kwargs)
        format_plot(args[0].label, args[0], plotStyle=name, plot=plt, ax=plot)
        export_plot(plot, savePath=savePath, fileName=args[0].label)
        return plot, payload

    return make_plot


"""def modify_legend2(ax, plt):
    handles, labels = ax.get_legend_handles_labels()
    display = (0, 1)
    artist = plt.Line2D((0, 1), (0, 0), color='#b9cfe7')  # Create custom artists
    a = [handle for i, handle in enumerate(handles) if i in display]
    b = [label for i, label in enumerate(labels) if i in display]
    legend = plt.legend(a + [artist], b + ['95% Confidence Limits'],
                        loc=5, bbox_to_anchor=(0, -0.21, 1., .102), ncol=2, mode='expand')
    frame = legend.get_frame().set_edgecolor('0.5')
    return legend"""


def plot_r_ci(x, x2, y2, y_model, t, s_err, n, ax, f):
    # Regression6, Confidence Interval, and Prediction Interval
    ax.plot(x, y_model, '-', color='0.1', linewidth=1.5, alpha=0.9, label='Regression6 Fit',  # '$' + f + '$',
            zorder=-2)
    plot_ci(t, s_err, n, x, x2, y2, ax=ax)
    plot_pi(t, s_err, x, x2, y2, ax=ax)


def plot_best_fit_function_string(var, ax, regParams, mse, R2, fam, rmse, mae):
    props = dict(boxstyle='round', alpha=0.5, color=sns.color_palette()[0])
    f = ExpressionGenerator.function_string(*regParams, fam=fam, y=var.label, evaluate=True, latex=False)
    msE, rmsE, maE = s_round(mse, 4), s_round(rmse, 4), s_round(mae, 4)
    """textstr = '{0} Best Fit:'.format(fam) + '\n' \
                                            '${0}$\n' \
                                            '$R^1 = {1}$\n' \
                                            '$MSE = {1}$\n' \
                                            '$RMSE = {2}$\n' \
                                            '$MAE = {3}$'.format(f, round(R2, 3), msE,
                                                                 rmsE, maE)
    ax.text(1.025, 0.6575, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))"""
    return f


def plot_regression_and_ci(v, ax):
    """
    This function is to be called within plot_scatter()
    col: str
    df: Pandas Series
    min: minimum value of col
    max: maximum value of col
    """
    x = np.array(v.index)
    y = v
    p, cov, yModel, mse, R2, fam, rmse, mae = Regression.get_model(x, y)
    n, m, DF, t = Regression.get_ci_stats(y, p)
    resid, chi2, chi2_red, s_err = Regression.estimate_error(y, yModel, DF)
    x2 = np.linspace(np.min(x), np.max(x), len(x))
    y2 = yModel
    f = plot_best_fit_function_string(v, ax, p, mse, R2, fam, rmse, mae)
    if R2 > 0.2:  # mse < 76057057
        plot_r_ci(x, x2, y2, yModel, t, s_err, n, ax, f)  # legend = modify_legend2(ax, plt)
    return f, (mse, R2, rmse, mae)


def set_style():
    sns.set_context('paper', rc={'lines.linewidth': 1, 'figure.figsize': (1, 1), 'figure.facecolor': 'white',
                                 'font.family': ['sans-serif', ]})
    sns.set(font='serif', font_scale=.7)
    sns.set_style('ticks', {'font.family': 'serif', 'font.serif': ['Times', 'Palatino', 'serif']})
    plt.legend(fontsize=7, loc='best', frameon=False, borderaxespad=5)


def format_plot(col, v, plotStyle, plot, ax):
    """
    col: string
    v: pandas Series
    plotStyle: string
    plot0: Matplotlib plt
    ax: Matplotlib figure
    """
    plot.title(col + ' ' + plotStyle, fontsize='3', fontweight='bold', fontname='Georgia')
    set_style()

    if plotStyle != 'Statistics Summary':
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1.)
        ax.spines['left'].set_linewidth(1.)

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_linewidth(1.)
        ax.spines['right'].set_linewidth(1.)

        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')
    # plt.xlim(v.min() - 1, v.max() + 1)
    if ax and plotStyle != 'Statistics Summary':
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_color('0.5')
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

    if plotStyle == 'Bar Plot':
        plot.ylabel('Quantity')

    elif plotStyle == 'Box Plot':
        plot.xlim(np.min(v) - 1, np.max(v) + 1)
        sns.despine(trim=True)
        sns.boxplot(capsize=.5)

    elif plotStyle == 'Distribution Fit Plot':
        plot.ylabel('Frequency')
        plot.xlim(np.min(v) - 1, np.max(v) + 1)
        sns.despine(top=True, right=True, trim=True)

    elif plotStyle == 'Distribution Plot':
        plot.xlim(np.min(v) - 1, np.max(v) + 1)
        sns.despine(top=True, right=True, trim=True)

    elif plotStyle == 'Heat Map':
        pass

    elif plotStyle == 'Histogram':
        plot.ylabel('Frequency')


    elif plotStyle == 'Scatter Plot':
        xRange = np.max(v.index.values) - np.min(v.index.values)
        strata = 5
        xTicks = np.arange(np.min(v.index.values) - 1, np.max(v.index.values) + 1, step=int(xRange / strata))
        plot.xticks(xTicks)
        plot.xlim(np.min(v.index.values) - 1, np.max(v.index.values) + 1)
        # plt.gca().set_aspect('equal', adjustable='box')
        pass

    elif plotStyle == 'Statistics Summary':
        pass

    elif plotStyle == 'Violin Plot':
        plt.ylabel('Frequency')
        plot.xlim(np.min(v) - 1, np.max(v) + 1)
        sns.despine(trim=True)
        sns.violinplot(scale='count', linewidth=3, )


def plot_stats(df, statsDict=None, savePath=TABLE_PLOTS_DIR):
    plt.gcf().clear()
    cellText = []
    colLabels = []
    s = 0
    for col in df:
        f = sorted(statsDict[col])
        cellText.append([str(statsDict[col][s]) for s in f])
        colLabels.append(col)
        if s == 0:
            rowLabels = [s for s in f]
            s += 1
    cellText = pd.DataFrame(cellText).T.as_matrix()
    table = plt.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, loc='upper center')
    table.auto_set_column_width(0)
    table.scale(2, 3)
    plt.axis('off')
    format_plot('', df, 'Statistics Summary', plt, table)
    export_plot(table, savePath, '' + TABLE_PLOTS_SUFFIX)


@plot
def plot_topN_distributions(v, N, name='Distribution Fit Plot', savePath=DISTRIBUTION_FIT_PLOTS_DIR):
    def plot_best_fit_function_string(ax, regParams, names, bw):
        props = dict(boxstyle='round', alpha=0.5, color=sns.color_palette()[0])
        f, n = ExpressionGenerator.distribution_string(names, regParams, y='PDF', latex=False)
        """textstr = '{0}:'.format(n[names[0]]) + '\n' \
                                               '${0}$'.format(f[names[0]]) + '\n' \
                                                                             '{0}:'.format(n[names[1]]) + '\n' \
                                                                                                          '${0}$'.format(
            f[names[1]]) + '\n' \
                           'KDE Bandwidth:' + '\n' \
                                              '${0}$'.format(bw)
        ax.text(1.025, 0.6475, textstr, transform=ax.transAxes, fontsize=14, bbox=props)"""
        return f, n

    distPlot = sns.distplot(v, kde=False, norm_hist=True)
    topNames, topParams = DistributionFitter.best_fit_distribution(v, distPlot)
    topDists = {i: getattr(st, topNames[i]) for i in range(N)}
    pdfs = DistributionFitter.get_pdfs(topDists, topParams, N)
    density, bandwidth = DistributionFitter.density_kde(v)
    density.plot(lw=2, label="Gaussian Kernel Density Estimate\n", legend=True)
    f, n = plot_best_fit_function_string(distPlot, topParams, topNames, bandwidth)
    for i in pdfs:
        # paramNames = (topDists[i].shapes + ', loc, scale').split(', ') if topDists[i].shapes else ['loc', 'scale']
        # param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(paramNames, topParams)])
        distPlot = pdfs[i].plot(lw=2, label=n[topNames[i]], legend=True)
    f = {n[k]: f[k] for k in n}
    return name, savePath, distPlot, f


# MAIN
def generate_all_plots(df, dtypes, savePath=PLOTS_DIR):
    """
    This function returns all the information used in the creation of the plots
    df: pandas DataFrame
    dtypes: list of str, 'Numerical' or 'Categorical'
    savePath: str, path where to save the plots as png
    return: list of str, dict, dict, dict, dict
    """
    clear_directories(savePath)
    variables = []
    categoricals = {}
    numericals = {}
    distFits = {}
    regressionFits = {}
    dfNum = []
    dfCat = []
    for col, label in zip(df, dtypes):
        if label == 'Categorical':
            categoricals[col] = StatsCollector.collect_stats(df[col], label)
            dfCat.append(col)
            plot_bar(df[col], stats=categoricals[
                col])  # plot_scatter(df[col], fit=False)  # plot_stats(df[col], stats0=['Mode'])

        elif label == 'Numerical':
            numericals[col] = StatsCollector.collect_stats(df[col], label)
            dfNum.append(col)  # TODO not sure what this is
            plot_histogram(df[col], savePath=os.path.join(savePath, 'Distribution'))
            _, distFits[col] = plot_topN_distributions(df[col], 2, savePath=os.path.join(savePath, 'DistributionFit'))
            _, regressionFits[col] = plot_scatter(df[col], fit=True, savePath=os.path.join(savePath, 'Scatter'))
            plot_box(df[col], stats=numericals[col], savePath=os.path.join(savePath, 'Box'))
            plot_violin(df[col], stats=numericals[col], savePath=os.path.join(savePath, 'Violin'))
        variables.append(col)
    if 'Numerical' in dtypes:
        plot_corrcov(df[dfNum], savePath=os.path.join(savePath, 'Heatmap'))
        plot_stats(df[dfNum], statsDict=numericals, savePath=os.path.join(savePath, 'StatisticComparison'))
    return variables, numericals, categoricals, distFits, regressionFits


def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def plot(func):
    """
    Plot, format, and export.
    func:
    return:
    """

    @wraps(func)
    def make_plot(*args, **kwargs):
        plt.gcf().clear()
        saveDir, plot, payload = func(*args, **kwargs)
        defaults = get_default_args(func)
        for d in defaults:
            if d not in kwargs:
                kwargs[d] = defaults[d]

        StaticPlotFormatter.format_plot(data=args[0], plotStyle=func.__name__[5:], plot=plt, ax=plot,
                                        title=kwargs['title'], tufte=kwargs['tufte'])

        export_plot(plot, directory=saveDir, fileName=kwargs['fileName'])
        return plot, payload

    return make_plot


def plot_regression_and_ci(v, ax):
    """
    This function is to be called within plot_scatter()
    col: str
    df: Pandas Series
    min: minimum value of col
    max: maximum value of col
    """

    def plot_ci(t, s_err, n, x, x2, y2, ax):
        # Return an axes of confidence bands using a simple approach.
        ax = plt.gca()
        ci = t * s_err * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        ax.fill_between(x2, y2 + ci, y2 - ci, alpha=0.2, color='#00AFBB', linestyle='-', zorder=-2)
        ax.plot(x2, y2 - ci, color='#ff9f00', label='95% Confidence Interval', linestyle='--', zorder=-1)
        ax.plot(x2, y2 + ci, color='#ff9f00', linestyle='--', zorder=-1)
        return ax

    def plot_pi(t, s_err, x, x2, y2, ax):
        pi = t * s_err * np.sqrt(1 + 1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        ax.fill_between(x2, y2 + pi, y2 - pi, alpha=0.5, color='None', linestyle='--')
        ax.plot(x2, y2 - pi, color='#CC79A7', label='95% Prediction Limits', linestyle=':')
        ax.plot(x2, y2 + pi, color='#CC79A7', linestyle=':')
        return ax

    def plot_r_ci(x, x2, y2, y_model, t, s_err, n, ax, f):
        # Regression6, Confidence Interval, and Prediction Interval
        ax.plot(x, y_model, '-', alpha=0.9, color='0.1', label='Regression6 Fit', linewidth=1.5, linestyle='-',
                zorder=-1)
        plot_ci(t, s_err, n, x, x2, y2, ax=ax)
        plot_pi(t, s_err, x, x2, y2, ax=ax)

    x = np.array(v.index)
    y = v
    p, cov, yModel, fam, errors = Regression.get_model(x, y)
    n, m, dF, t = Regression.get_ci_stats(y, p)
    errors = Regression.estimate_error(y, yModel, dF, errors)
    x2 = np.linspace(np.min(x), np.max(x), len(x))
    y2 = yModel
    fit = ExpressionGenerator.function_string(*p, fam=fam, y=v.label, evaluate=True, latex=False)
    R2 = errors['R Squared']
    if R2 > 0.05:  # mse < 76057057
        plot_r_ci(x, x2, y2, yModel, t, errors['Standard Error'], n, ax, fit)
    return (fit, errors)


@plot
def plot_bar(vector, stats=None, saveDir=BAR_PLOTS_DIR, fileName='', title='', tufte=True):
    m = vector.value_counts()
    barPlot = sns.barplot(x=m.index, y=m.values)
    return saveDir, barPlot, None


@plot
def plot_box(vector, stats=None, saveDir=BOX_PLOTS_DIR, fileName='', title='', tufte=True):
    """
    v: pandas Series, (vector)
    return: void
    """
    boxPlot = sns.boxplot(y=vector, width=.25)
    return saveDir, boxPlot, None


@plot
def plot_heatmap(matrix, saveDir=HEATMAP_PLOTS_DIR, fileName='', title='', tufte=True):
    heatmapPlot = sns.heatmap(matrix, annot=False, vmin=0, vmax=None)
    return saveDir, heatmapPlot, None


@plot
def plot_histogram(vector, saveDir=DISTRIBUTION_PLOTS_DIR, fileName='', title='', tufte=True):
    """
    v: pandas Series, (vector)
    fit: bool
    return:
    """
    distPlot = sns.distplot(vector, norm_hist=False, kde=False, bins=None, rug=True)
    return saveDir, distPlot, None


@plot
def plot_scatter(vector, fit=True, saveDir=SCATTER_PLOTS_DIR, fileName='', title='', tufte=True):
    """
    df: pandas Series (vector)
    fit: bool
    return: void
    """
    scatterPlot = sns.regplot(np.arange(len(vector)), vector, fit_reg=False)  # , label=v.name)
    if fit:
        regressionFit, regressionError = plot_regression_and_ci(vector, scatterPlot)
        return saveDir, scatterPlot, (regressionFit, regressionError)
    return saveDir, scatterPlot, None


def plot_stats(df, statsDict=None, saveDir=TABLE_PLOTS_DIR, tufte=True):
    plt.gcf().clear()
    cellText = []
    colLabels = []
    s = 0
    for col in df:
        f = sorted(statsDict[col])
        cellText.append([str(statsDict[col][s]) for s in f])
        colLabels.append(col)
        if s == 0:
            rowLabels = [s for s in f]
            s += 1
    cellText = pd.DataFrame(cellText).T.as_matrix()
    table = plt.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, loc='center', rowLoc='center',
                      colLoc='center')
    table.auto_set_column_width(0)
    table.scale(2, 3)
    table.set_fontsize(12)
    plt.axis('off')
    # format_plot(name='', data=df, plotStyle='Table', plot0=plt, ax=table)
    export_plot(table, saveDir, '' + TABLE_PLOTS_SUFFIX)


@plot
def plot_distribution(vector, N=1, saveDir=DISTRIBUTION_FIT_PLOTS_DIR, fileName='', title='', tufte=True):
    """
    plot the top N distributions as well as a Gaussian Kernel Density Estimate
    :param series:
    :param N:
    :param name:
    :param saveDir:
    :return:
    """
    distPlot = sns.distplot(vector, kde=False, norm_hist=True)
    pdfs, topNames, topParams, topNSSE = DistributionFitter.best_fit_distribution(vector, N)
    density, bandwidth = DistributionFitter.density_kde(vector)
    density.plot(lw=2, label="Gaussian Kernel Density Estimate\n", legend=True)
    fit, names, errors = ExpressionGenerator.distribution_string(topNames, topParams, topNSSE, y='PDF', latex=False)
    lineStyles = ['--', ':', '-.']
    for i in pdfs:
        # paramNames = (topDists[i].shapes + ', loc, scale').split(', ') if topDists[i].shapes else ['loc', 'scale']
        # param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(paramNames, topParams)])
        distPlot = pdfs[i].plot(lw=2, linestyle=lineStyles[i], label=names[topNames[i]], legend=True)
    f = {names[k]: (fit[k], errors[k]) for k in names}
    return saveDir, distPlot, f


@plot
def plot_violin(vector, saveDir=VIOLIN_PLOTS_DIR, fileName='', title='', tufte=True):
    violinPlot = sns.violinplot(y=vector)
    return saveDir, violinPlot, None


def plot_stats_comparison(stats, stat):
    """
    stats0: pandas DataFrame
    stat: str
    """
    plt.gcf().clear()
    h = {}
    for col in stats:
        h[col] = stats[col][stat]

    barplot = sns.barplot(x=[col for col in h], y=[h[col] for col in h])
    plt.title(stat + ' Comparison')
    format_plot(name=stat, data=stats, plotStyle='Bar Plot', plot=plt, ax=barplot)
    fig = barplot.get_figure()
    fig.savefig(os.path.join(STAT_COMPARISON_PLOTS_DIR, stat[:3] + '_comparison'))


def compare_all_stats(stats):
    """
    stats0: pandas DataFrame
    """
    for stat in stats:
        plot_stats_comparison(stats, stat)


# MAIN
@timer
def generate_all_plots(df, dtypes, saveDir=PLOTS_DIR, N=1):
    """
    This function returns all the information used in the creation of the plots
    df: pandas DataFrame
    dtypes: list of str, 'Numerical' or 'Categorical'
    saveDir: str, path where to save the plots as png
    return: list of str, dict, dict, dict, dict
    """
    iqrTableStats = ['Lower IQR', 'First Quartile', 'Median', 'Third Quartile', 'Upper IQR']
    dispersionTableStats = ['Standard Deviation', 'Variance', 'Skew', 'Kurtosis']
    centralTendenciesTableStats = ['Mean', 'Median']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clear_directories(saveDir)
        variables = []
        categoricals = {}
        numericals = {}
        distFits = {}
        regression = {}
        dfNum = []
        dfCat = []
        for col, label in zip(df, dtypes):
            if label == 'Categorical':
                categoricals[col] = StatsCollector.collect_stats(df[col], label)
                dfCat.append(col)
                plot_bar(df[col], stats=categoricals[col], saveDir=os.path.join(saveDir, 'Bar'),
                         fileName=col)  # plot_scatter(df[col], fit=False)  # plot_stats(df[col], stats0=['Mode'])

            elif label == 'Numerical':
                numericals[col] = StatsCollector.collect_stats(df[col], label)
                dfNum.append(col)  # TODO not sure what this is
                plot_table(OrderedDict([(k, numericals[col][k]) for k in iqrTableStats]),
                           saveDir=os.path.join(saveDir, 'IQR'), fileName=col, title='', tufte=False)
                plot_table(OrderedDict([(k, numericals[col][k]) for k in dispersionTableStats]),
                           saveDir=os.path.join(saveDir, 'Dispersion'), fileName=col, title='', tufte=False)
                plot_table(OrderedDict([(k, numericals[col][k]) for k in centralTendenciesTableStats]),
                           saveDir=os.path.join(saveDir, 'CentralTendencies'), fileName=col, title='', tufte=False)

                plot_histogram(df[col], saveDir=os.path.join(saveDir, 'Distribution'), fileName=col, tufte=False)
                _, distFits[col] = plot_distribution(df[col], N=N, saveDir=os.path.join(saveDir, 'DistributionFit'),
                                                     fileName=col, tufte=False)
                _, regression[col] = plot_scatter(df[col], fit=True, saveDir=os.path.join(saveDir, 'Scatter'),
                                                  fileName=col, tufte=False)
                plot_table(
                    OrderedDict([(k, s_round(regression[col][1][k])) for k in sorted(regression[col][1].keys())]),
                    saveDir=os.path.join(saveDir, 'RegressionError'), fileName=col, title='', tufte=False)
                plot_table(OrderedDict([(k, s_round(distFits[col][k][1])) for k in sorted(distFits[col].keys())]),
                           saveDir=os.path.join(saveDir, 'DistributionError'), fileName=col, title='', tufte=False)

                plot_box(df[col], saveDir=os.path.join(saveDir, 'Box'), fileName=col, tufte=False)
                plot_violin(df[col], saveDir=os.path.join(saveDir, 'Violin'), fileName=col, tufte=False)

                variables.append(col)
            if 'Numerical' in dtypes:
                plot_corrcov(df[dfNum], saveDir=os.path.join(saveDir, 'Heatmap'))
                """plot_stats(df[dfNum], statsDict=numericals,
                           saveDir=os.path.join(saveDir, 'StatisticComparison'))"""
        return variables, numericals, categoricals, distFits, regression


@plot
def plot_table(vector, saveDir=TABLE_PLOTS_DIR, fileName='', title='', tufte=True):
    """
    vector: dictionary
    saveDir: str, path
    title: str
    tufte: boolean
    return:
    """
    cellText = [[str(vector[k])] for k in vector]
    rowLabels = [[k] for k in vector]
    rowLabels = [text_formatter(r, 10) for r in rowLabels]
    table = plt.table(cellText=cellText, rowLabels=rowLabels, rowLoc='center', loc='center', colLoc='center')
    return saveDir, table, None


def plot(func):
    """
    Plot, format, and export.
    func:
    return:
    """

    @wraps(func)
    def make_plot(*args, **kwargs):
        plt.gcf().clear()
        plot, payload = func(*args, **kwargs)
        defaults = get_default_args(func)
        for d in defaults:
            if d not in kwargs:
                kwargs[d] = defaults[d]

        StaticPlotFormatter.format_plot(data=args[0], plotStyle=func.__name__[5:], plot=plt, ax=plot,
                                        title=kwargs['title'], tufte=kwargs['tufte'])

        export_plot(plot, directory=kwargs['saveDir'], fileName=kwargs['fileName'])
        return plot, payload

    return make_plot


@plot
def plot_bar(data, stats=None, saveDir=BAR_PLOTS_DIR, fileName='', title='', tufte=True):
    m = data.value_counts()
    barPlot = sns.barplot(x=m.index, y=m.values)
    return barPlot, None


@plot
def plot_box(data, stats=None, saveDir=BOX_PLOTS_DIR, fileName='', title='', tufte=True):
    """
    v: pandas Series, (data)
    return: void
    """
    boxPlot = sns.boxplot(y=data, width=.25)
    return boxPlot, None


@plot
def plot_heatmap(matrix, saveDir=HEATMAP_PLOTS_DIR, fileName='', title='', tufte=True):
    heatmapPlot = sns.heatmap(matrix, annot=False, vmin=0, vmax=None)
    return heatmapPlot, None


@plot
def plot_histogram(data, saveDir=DISTRIBUTION_PLOTS_DIR, fileName='', title='', tufte=True):
    """
    v: pandas Series, (data)
    fit: bool
    return:
    """
    distPlot = sns.distplot(data, norm_hist=False, kde=False, bins=None, rug=True)
    return distPlot, None


@plot
def plot_scatter(data, fit=True, saveDir=SCATTER_PLOTS_DIR, fileName='', title='', tufte=True):
    """
    df: pandas Series (data)
    fit: bool
    return: void
    """
    scatterPlot = sns.regplot(np.arange(len(data)), data, fit_reg=False)  # , label=v.name)
    if fit:
        regressionFit, regressionError = plot_regression_and_ci(data, scatterPlot)
        return scatterPlot, (regressionFit, regressionError)
    return scatterPlot, None


def plot_stats(df, statsDict=None, saveDir=TABLE_PLOTS_DIR, tufte=True):
    plt.gcf().clear()
    cellText = []
    colLabels = []
    s = 0
    for col in df:
        f = sorted(statsDict[col])
        cellText.append([str(statsDict[col][s]) for s in f])
        colLabels.append(col)
        if s == 0:
            rowLabels = [s for s in f]
            s += 1
    cellText = pd.DataFrame(cellText).T.as_matrix()
    table = plt.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, loc='center', rowLoc='center',
                      colLoc='center')
    table.auto_set_column_width(0)
    table.scale(2, 3)
    table.set_fontsize(12)
    plt.axis('off')
    # format_plot(name='', data=df, plotStyle='Table', plot0=plt, ax=table)
    export_plot(table, saveDir, '' + TABLE_PLOTS_SUFFIX)


@plot
def plot_distribution(data, N=1, saveDir=DISTRIBUTION_FIT_PLOTS_DIR, fileName='', title='', tufte=True):
    """
    plot the top N distributions as well as a Gaussian Kernel Density Estimate
    :param series:
    :param N:
    :param name:
    :param saveDir:
    :return:
    """
    distPlot = sns.distplot(data, kde=False, norm_hist=True)
    pdfs, topNames, topParams, topNSSE = DistributionFitter.best_fit_distribution(data, N)
    density, bandwidth = DistributionFitter.density_kde(data)
    density.plot(lw=2, label="Gaussian Kernel Density Estimate\n", legend=True)
    fit, names, errors = distribution_string(topNames, topParams, topNSSE, y='PDF', latex=False)
    lineStyles = ['--', ':', '-.']
    for i in pdfs:
        # paramNames = (topDists[i].shapes + ', loc, scale').split(', ') if topDists[i].shapes else ['loc', 'scale']
        # param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(paramNames, topParams)])
        distPlot = pdfs[i].plot(lw=2, linestyle=lineStyles[i], label=names[topNames[i]], legend=True)
    f = {names[k]: (fit[k], errors[k]) for k in names}
    return distPlot, f


@plot
def plot_violin(data, saveDir=VIOLIN_PLOTS_DIR, fileName='', title='', tufte=True):
    violinPlot = sns.violinplot(y=data)
    return violinPlot, None


@plot
def plot_table(data, saveDir=TABLE_PLOTS_DIR, fileName='', title='', tufte=True):
    """
    data: dictionary
    saveDir: str, path
    title: str
    tufte: boolean
    return:
    """
    cellText = [[str(data[k])] for k in data]
    rowLabels = [[k] for k in data]
    rowLabels = [text_formatter(r, 10) for r in rowLabels]
    table = plt.table(cellText=cellText, rowLabels=rowLabels, rowLoc='center', loc='center', colLoc='center')
    return table, None


def plot_regression_and_ci(v, ax):
    """
    This function is to be called within plot_scatter()
    col: str
    df: Pandas Series
    min: minimum value of col
    max: maximum value of col
    """

    def plot_ci(t, s_err, n, x, x2, y2, ax=None):
        # Return an axes of confidence bands using a simple approach.
        if ax is not None:
            ax = plt.gca()
            ci = t * s_err * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
            ax.fill_between(x2, y2 + ci, y2 - ci, color='#5CB0C2', edgecolor='#5CB0C2', linestyle='-', alpha=0.2,
                            zorder=-1)
            ax.plot(x2, y2 - ci, '--', color='#D2691E', alpha=0.5, zorder=-1)
            ax.plot(x2, y2 + ci, '--', color='#D2691E', alpha=0.5, zorder=-1)
        return ax

    def plot_pi(t, s_err, x, x2, y2, ax=None):
        pi = t * s_err * np.sqrt(1 + 1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        ax.fill_between(x2, y2 + pi, y2 - pi, color='None', linestyle='--', alpha=0.5)
        ax.plot(x2, y2 - pi, '-.', color='0.5', label='95% Prediction Limits', alpha=0.5)
        ax.plot(x2, y2 + pi, '-.', color='0.5', alpha=0.5)
        return ax

    def modify_legend2(ax, plt):
        handles, labels = ax.get_legend_handles_labels()
        display = (0, 1)
        artist = plt.Line2D((0, 1), (0, 0), color='#b9cfe7')  # Create custom artists
        a = [handle for i, handle in enumerate(handles) if i in display]
        b = [label for i, label in enumerate(labels) if i in display]
        legend = plt.legend(a + [artist], b + ['95% Confidence Limits'], loc=9, bbox_to_anchor=(0, -0.21, 1., .102),
                            ncol=3, mode='expand')
        frame = legend.get_frame().set_edgecolor('0.5')
        return legend

    def plot_r_ci(x, x2, y2, y_model, t, s_err, n, ax):
        # Regression6, Confidence Interval, and Prediction Interval
        ax.plot(x, y_model, '-', color='0.1', linewidth=1.5, alpha=0.9, label='Fit', zorder=-2)
        plot_ci(t, s_err, n, x, x2, y2, ax=ax)
        plot_pi(t, s_err, x, x2, y2, ax=ax)

    # def plot_best_fit_function_string(v, ax, regParams, mse, R2, fam, rmse, mae):
    #     props = dict(boxstyle='round', alpha=0.5, color=sns.color_palette()[0])
    #     f = ExpressionGenerator.function_string(*regParams, fam=fam, y=v.name,
    #                                             latex=False)  # get_function_string(fam, regParams, col)
    #     msE, rmsE, maE = s_round(mse, 4), s_round(rmse, 4), s_round(mae, 4)
    #     textstr = '{0} Best Fit:'.format(fam) + '\n' \
    #                                             '${0}$\n' \
    #                                             '$R^1 = {1}$\n' \
    #                                             '$MSE = {2}$\n' \
    #                                             '$RMSE = {3}$\n' \
    #                                             '$MAE = {4}$'.format(f, round(R2, 4), msE, rmsE, maE)
    #     ax.text(1.025, 0.6575, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
    #     ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    #     ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    #
    # x = np.array(v.index)
    # y = v
    # p, cov, yModel, mse, R2, fam, rmse, mae = get_model(x, y)
    # n, m, DF, t = get_ci_stats(y, p)
    # resid, chi2, chi2_red, s_err = estimate_error(y, yModel, DF)
    # x2 = np.linspace(np.min(x), np.max(x), 100)
    # y2 = yModel
    # if R2 > 0.2:  # mse < 76057057
    #     plot_r_ci(x, x2, y2, yModel, t, s_err, n, ax)
    #     plot_best_fit_function_string(v, ax, p, mse, R2, fam, rmse, mae)
    # legend = modify_legend2(ax, plt)

    def plot_best_fit_function_string(var, ax, regParams, mse, R2, fam, rmse, mae):
        props = dict(boxstyle='round', alpha=0.5, color=sns.color_palette()[0])
        print(regParams)
        print(fam)
        print(var.label)
        f = ExpressionGenerator.function_string(*regParams, fam=fam, y=var.label, evaluate=True, latex=False)
        msE, rmsE, maE = s_round(mse, 4), s_round(rmse, 4), s_round(mae, 4)
        textstr = '{0} Best Fit:'.format(fam) + '\n' \
                                                '${0}$\n' \
                                                '$R^1 = {1}$\n' \
                                                '$MSE = {2}$\n' \
                                                '$RMSE = {3}$\n' \
                                                '$MAE = {4}$'.format(f, round(R2, 4), msE, rmsE, maE)
        ax.text(1.025, 0.6575, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    x = np.array(v.index)
    y = v
    p, cov, yModel, mse, R2, fam, rmse, mae = Regression.get_model(x, y)
    n, m, DF, t = Regression.get_ci_stats(y, p)
    resid, chi2, chi2_red, s_err = Regression.estimate_error(y, yModel, DF)
    x2 = np.linspace(np.min(x), np.max(x), len(x))
    y2 = yModel
    if R2 > 0.2:  # mse < 76057057
        plot_r_ci(x, x2, y2, yModel, t, s_err, n, ax)
        plot_best_fit_function_string(v, ax, p, mse, R2, fam, rmse, mae)
    legend = modify_legend2(ax, plt)


def format_plot(col, v, plotStyle, plot, ax):
    """
    col: string
    colMin: number
    colMax: number
    plotStyle: string
    plot0: Matplotlib plt
    """
    plot.title(col + ' ' + plotStyle, fontsize='18', fontweight='bold', fontname='Georgia')
    if plotStyle != 'Statistics Summary':
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1.)
        ax.spines['left'].set_linewidth(1.)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_linewidth(1.)
        ax.spines['right'].set_linewidth(1.)
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')
    # plt.xlim(v.min() - 1, v.max() + 1)
    if ax and plotStyle != 'Statistics Summary':
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_color('0.5')
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
    if plotStyle == 'Bar Plot':
        plot.ylabel('Frequency')
    elif plotStyle == 'Box Plot':
        pass
    elif plotStyle == 'Distribution Fit':
        plot.ylabel('Frequency')
        plot.xlim(np.min(v) - 1, np.max(v) + 1)
    elif plotStyle == 'Histogram':
        plot.ylabel('Frequency')
    elif plotStyle == 'Scatter Plot':
        plot.xticks(np.arange(np.min(v.index.values) - 1, np.max(v.index.values) + 1))
        plot.xlim(np.min(v.index.values) - 1, np.max(v.index.values) + 1)
        # plt.gca().set_aspect('equal', adjustable='box')
        pass
    elif plotStyle == 'Statistics Summary':
        pass
    elif plotStyle == 'Violin Plot':
        plt.ylabel('Frequency')


def plot_box(v, stats=None):
    """
    v: pandas Series, (vector)
    return: void
    """
    # bins = np.arange(stats0[ col ][ 'min' ]-0.5,stats0[ col ][ 'max' ] + 1.5, step=dist_min(df[col]))
    plt.gcf().clear()
    boxPlot = sns.boxplot(x=v)
    if stats != None:
        plot_box_stats(v, stats, boxPlot)
    format_plot(v.label, v, 'Box Plot', plt, boxPlot)
    export_plot(boxPlot, BoxPlotsDir, v.label + BoxPlotSuffix)


def plot_histogram(v, fit=True):
    """
    v: pandas Series, (vector)
    fit: bool
    return:
    """
    # bins = np.arange(stats0[ col ][ 'min' ]-0.5,stats0[ col ][ 'max' ] + 1.5, step=dist_min(df[col]))
    plt.gcf().clear()
    distPlot = sns.distplot(v, norm_hist=False, kde=False, bins=None, rug=True)
    format_plot(v.label, v, 'Histogram', plt, distPlot)
    export_plot(distPlot, DistributionPlotsDir, v.label + DistributionPlotSuffix)
    if fit:
        plot_topN_distributions(v, 2)


def plot_heatmap(matrix, matrixType):
    plt.gcf().clear()
    heatmapPlot = sns.heatmap(matrix, annot=True)
    export_plot(heatmapPlot, HeatmapPlotsDir, matrixType)


def plot_stats(df, statsDict):
    plt.gcf().clear()
    cellText = []
    colLabels = []
    s = 0
    for col in df:
        f = sorted(statsDict[col])
        cellText.append([str(statsDict[col][s]) for s in f])
        colLabels.append(col)
        if s == 0:
            rowLabels = [s for s in f]
            s += 1
    cellText = pd.DataFrame(cellText).T.as_matrix()
    table = plt.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, loc='upper center')
    table.auto_set_column_width(0)
    table.scale(2, 3)
    plt.axis('off')
    format_plot('', df, 'Statistics Summary', plt, table)
    export_plot(table, TablePlotsDir, '' + TablePlotSuffix)


def plot_topN_distributions(v, N):
    def plot_best_fit_function_string(ax, regParams, names, bw):
        props = dict(boxstyle='round', alpha=0.5, color=sns.color_palette()[0])
        f, n = ExpressionGenerator.distribution_string(names, regParams, y='PDF', latex=False)
        textstr = '{0}:'.format(n[names[0]]) + '\n' \
                                               '${0}$'.format(f[names[0]]) + '\n' \
                                                                             '{0}:'.format(n[names[1]]) + '\n' \
                                                                                                          '${0}$'.format(
            f[names[1]]) + '\n' \
                           'KDE Bandwidth:' + '\n' \
                                              '${0}$'.format(bw)
        ax.text(1.025, 0.6475, textstr, transform=ax.transAxes, fontsize=14, bbox=props)

    plt.gcf().clear()
    ax = sns.distplot(v, kde=False, norm_hist=True)
    topNames, topParams = DistributionFitter.best_fit_distribution(v, ax)
    topDists = {i: getattr(st, topNames[i]) for i in range(N)}
    pdfs = DistributionFitter.get_pdfs(topDists, topParams, N)

    for i in pdfs:
        paramNames = (topDists[i].shapes + ', loc, scale').split(', ') if topDists[i].shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(paramNames, topParams)])
        ax = pdfs[i].plot(lw=2, label=topNames[i] + '\n' + param_str, legend=True)

    density, bandwidth = DistributionFitter.density_kde(v)
    density.plot(lw=2, label="Gaussian Kernel Density Estimate", legend=True)
    plot_best_fit_function_string(ax, topParams, topNames, bandwidth)
    format_plot(v.label, v, 'Distribution Fit', plt, ax)
    export_plot(ax, DistributionFitPlotsDir, v.label + DistributionFitPlotSuffix)


def add_plots_to_report(col):
    plots = {}
    for directory in PlotDirs:
        plots[directory] = os.path.join(directory, col)


def generate_report_plots(df, dtypes):
    categoricals = {}
    numericals = {}
    variables = []
    dfCat = []  # this will be the version of df with only the categorical
    dfNum = []  # this will be the version of df with only the numerical
    for col, label in zip(df, dtypes):
        if label == 'Categorical':
            categoricals[col] = StatsCollector.collect_stats(df[col], label)
            dfCat.append(col)
            plot_bars(df[col],
                      categoricals[col])  # plot_scatter(df[col], fit=False)  # plot_stats(df[col], stats0=['Mode'])

        elif label == 'Numerical':
            numericals[col] = StatsCollector.collect_stats(df[col], label)
            dfNum.append(col)
            plot_histogram(df[col])
            plot_scatter(df[col])
            plot_box(df[col], numericals[col])
            plot_violin(df[col], numericals[col])
        variables.append(col)
    if 'Numerical' in dtypes:
        stats_corr_cov(df[dfNum], numericals)
        plot_stats(df[dfNum], numericals)
    return variables


def stats_corr_cov(df, stats):
    # stats0, corr, cov = collect_stats(df)
    # make_all_plots(df, stats0, corr, cov)
    corr, cov = StatsCollector.get_corr_cov(df)
    plot_corrcov(corr, cov)
    compare_all_stats(stats)


def plot(func):
    """
    Plot, format, and export.
    func:
    return:
    """

    @wraps(func)
    def make_plot(*args, **kwargs):
        plt.gcf().clear()
        name, savePath, plot, payload = func(*args, **kwargs)
        try:
            n = args[0].label
        except:
            n = kwargs['name']
        try:
            tufte = kwargs['tufte']
        except:
            tufte = True
        format_plot(data=args[0], plotStyle=name, plot=plt, ax=plot, name=n, tufte=tufte)
        export_plot(plot, savePath=savePath, fileName=n)
        figs.append((plot.get_figure(), savePath, n))
        return plot, payload

    return make_plot


def plot_regression_and_ci(v, ax):
    """
    This function is to be called within plot_scatter()
    col: str
    df: Pandas Series
    min: minimum value of col
    max: maximum value of col
    """

    def plot_ci(t, s_err, n, x, x2, y2, ax):
        # Return an axes of confidence bands using a simple approach.
        ax = plt.gca()
        ci = t * s_err * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        ax.fill_between(x2, y2 + ci, y2 - ci, color='#5CB0C2', edgecolor='#5CB0C2', linestyle='-', alpha=0.2, zorder=-1)
        ax.plot(x2, y2 - ci, '--', color='#D2691E', alpha=0.5, zorder=-1, label='95% Confidence Interval')
        ax.plot(x2, y2 + ci, '--', color='#D2691E', alpha=0.5, zorder=-1)
        return ax

    def plot_r_ci(x, x2, y2, y_model, t, s_err, n, ax, f):
        # Regression6, Confidence Interval, and Prediction Interval
        ax.plot(x, y_model, '-', color='0.1', linewidth=1.5, alpha=0.9, label='Regression6 Fit',  # '$' + f + '$',
                zorder=-2)
        plot_ci(t, s_err, n, x, x2, y2, ax=ax)
        plot_pi(t, s_err, x, x2, y2, ax=ax)

    x = np.array(v.index)
    y = v
    p, cov, yModel, fam, errors = Regression.get_model(x, y)
    n, m, dF, t = Regression.get_ci_stats(y, p)
    errors = Regression.estimate_error(y, yModel, dF, errors)
    x2 = np.linspace(np.min(x), np.max(x), len(x))
    y2 = yModel
    fit = ExpressionGenerator.function_string(*p, fam=fam, y=v.label, evaluate=True, latex=False)
    R2 = errors['R2']
    if R2 > 0.05:  # mse < 76057057
        plot_r_ci(x, x2, y2, yModel, t, errors['standard error'], n, ax, fit)
    return (fit, errors)


@plot
def plot_bar(vector, name='Bar Plot', stats=None, savePath=BAR_PLOTS_DIR, tufte=True):
    m = vector.value_counts()
    barPlot = sns.barplot(x=m.index, y=m.values)
    return name, savePath, barPlot, None


@plot
def plot_ct_stats(vector, name='Table', stats=None, savePath=CT_TABLES_DIR, tufte=True):
    mode = stats['Mode']
    cellText = [[str(mode)]]
    rowLabels = ['$Mode$']
    table = plt.table(cellText=cellText, rowLabels=rowLabels, loc='center')
    table.set_fontsize(12)
    for key, cell in table.get_celld().items():
        cell.set_linewidth(1)
    table.auto_set_column_width(0)
    # table.scale(1, 2)
    return name, savePath, table, None


@plot
def plot_box(vector, name='Box Plot', stats=None, savePath=BOX_PLOTS_DIR, tufte=True):
    """
    v: pandas Series, (vector)
    return: void
    """
    boxPlot = sns.boxplot(y=vector, width=.25)
    return name, savePath, boxPlot, None


@plot
def plot_iqr_stats(vector, name='Table', stats=None, savePath=IQR_TABLES_DIR, tufte=True):
    minimum = s_round(stats['Lower IQR'], 2)
    q1 = s_round(stats['First Quartile'], 2)
    median = s_round(stats['Median'], 2)
    q2 = s_round(stats['Third Quartile'], 2)
    maximum = s_round(stats['Upper IQR'], 2)
    iqr = s_round(stats['IQR'], 2)
    cellText = [[str(minimum)], [str(q1)], [str(median)], [str(q2)], [str(maximum)], [str(iqr)]]
    rowLabels = ['$Minimum$', '$25^{th}$ $Percentile$', '$Median$', '$75^{th}$ $Percentile$', '$Maximum$', '$IQR$']
    table = plt.table(cellText=cellText, rowLabels=rowLabels, loc='center')
    table.set_fontsize(12)
    for key, cell in table.get_celld().items():
        cell.set_linewidth(1)
    table.auto_set_column_width(0)
    return name, savePath, table, None


@plot
def plot_heatmap(matrix, name='Heatmap Plot', savePath=HEATMAP_PLOTS_DIR, tufte=True):
    heatmapPlot = sns.heatmap(matrix, annot=False)
    return name, savePath, heatmapPlot, None


@plot
def plot_histogram(vector, name='Distribution Plot', savePath=DISTRIBUTION_PLOTS_DIR, tufte=True):
    """
    v: pandas Series, (vector)
    fit: bool
    return:
    """
    distPlot = sns.distplot(vector, norm_hist=False, kde=False, bins=None, rug=True)
    return name, savePath, distPlot, None


@plot
def plot_scatter(vector, name='Scatter Plot', fit=True, savePath=SCATTER_PLOTS_DIR, tufte=True):
    """
    df: pandas Series (vector)
    fit: bool
    return: void
    """
    scatterPlot = sns.regplot(np.arange(len(vector)), vector, fit_reg=False)  # , label=v.name)
    if fit:
        regressionFit, regressionError = plot_regression_and_ci(vector, scatterPlot)
        return name, savePath, scatterPlot, (regressionFit, regressionError)
    return name, savePath, scatterPlot, None


def plot_stats(df, statsDict=None, savePath=TABLE_PLOTS_DIR, tufte=True):
    plt.gcf().clear()
    cellText = []
    colLabels = []
    s = 0
    for col in df:
        f = sorted(statsDict[col])
        cellText.append([str(statsDict[col][s]) for s in f])
        colLabels.append(col)
        if s == 0:
            rowLabels = [s for s in f]
            s += 1
    cellText = pd.DataFrame(cellText).T.as_matrix()
    table = plt.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, loc='center')
    table.auto_set_column_width(0)
    table.scale(2, 3)
    table.set_fontsize(12)
    plt.axis('off')
    format_plot(name='', data=df, plotStyle='Table', plot=plt, ax=table)
    export_plot(table, savePath, '' + TABLE_PLOTS_SUFFIX)


@plot
def plot_topN_distributions(vector, N=1, name='Distribution Fit Plot', savePath=DISTRIBUTION_FIT_PLOTS_DIR, tufte=True):
    """
    plot the top N distributions as well as a Gaussian Kernel Density Estimate
    :param series:
    :param N:
    :param name:
    :param savePath:
    :return:
    """
    distPlot = sns.distplot(vector, kde=False, norm_hist=True)
    pdfs, topNames, topParams, topNSSE = DistributionFitter.best_fit_distribution(vector, N)
    density, bandwidth = DistributionFitter.density_kde(vector)
    density.plot(lw=2, label="Gaussian Kernel Density Estimate\n", legend=True)
    fit, names, errors = ExpressionGenerator.distribution_string(topNames, topParams, topNSSE, y='PDF', latex=False)
    lineStyles = [':', '--', '-.']
    for i in pdfs:
        # paramNames = (topDists[i].shapes + ', loc, scale').split(', ') if topDists[i].shapes else ['loc', 'scale']
        # param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(paramNames, topParams)])
        distPlot = pdfs[i].plot(lw=2, linestyle=lineStyles[i], label=names[topNames[i]], legend=True)
    f = {names[k]: (fit[k], errors[k]) for k in names}
    return name, savePath, distPlot, f


@plot
def plot_violin(vector, name='Violin Plot', stats=None, savePath=VIOLIN_PLOTS_DIR, tufte=True):
    violinPlot = sns.violinplot(y=vector)
    return name, savePath, violinPlot, None


@plot
def plot_dispersion_stats(vector, name='Table', stats=None, savePath=DISPERSION_TABLES_DIR, tufte=True):
    mean = s_round(stats['Mean'], 2)
    mode = s_round(stats['Mode'], 2)
    std = s_round(stats['Standard Deviation'], 2)
    variance = s_round(stats['Variance'], 2)
    skew = s_round(stats['Skew'], 2)
    kurtosis = s_round(stats['Kurtosis'], 2)
    cellText = [[str(mean)], [str(mode)], [str(std)], [str(variance)], [str(skew)], [str(kurtosis)]]
    rowLabels = ['Mean', 'Mode', 'Standard Deviation', 'Variance', 'Skew', 'Kurtosis']
    rowLabels = [text_formatter(r, 100) for r in rowLabels]
    table = plt.table(cellText=cellText, rowLabels=rowLabels, rowLoc='center', loc='best', colLoc='center',
                      bbox=[0, 0, .1, .7])
    for key, cell in table.get_celld().items():
        cell.set_linewidth(1)
    table.auto_set_column_width(0)
    return name, savePath, table, None


@timer
def generate_all_plots(df, dtypes, savePath=PLOTS_DIR, N=1):
    """
    This function returns all the information used in the creation of the plots
    df: pandas DataFrame
    dtypes: list of str, 'Numerical' or 'Categorical'
    savePath: str, path where to save the plots as png
    return: list of str, dict, dict, dict, dict
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clear_directories(savePath)
        variables = []
        categoricals = {}
        numericals = {}
        distFits = {}
        regression = {}
        dfNum = []
        dfCat = []
        for col, label in zip(df, dtypes):
            if label == 'Categorical':
                categoricals[col] = StatsCollector.collect_stats(df[col], label)
                dfCat.append(col)
                plot_bar(df[col], stats=categoricals[col], savePath=os.path.join(savePath,
                                                                                 'Bar'))  # plot_scatter(df[col], fit=False)  # plot_stats(df[col], stats0=['Mode'])
            elif label == 'Numerical':
                numericals[col] = StatsCollector.collect_stats(df[col], label)
                dfNum.append(col)  # TODO not sure what this is
                plot_iqr_stats(df[col], stats=numericals[col], tufte=False)
                plot_ct_stats(df[col], stats=numericals[col], tufte=False)
                plot_dispersion_stats(df[col], stats=numericals[col], tufte=False)

                plot_histogram(df[col], savePath=os.path.join(savePath, 'Distribution'), tufte=False)
                _, distFits[col] = plot_topN_distributions(df[col], N=N,
                                                           savePath=os.path.join(savePath, 'DistributionFit'),
                                                           tufte=False)
                _, regression[col] = plot_scatter(df[col], fit=True, savePath=os.path.join(savePath, 'Scatter'),
                                                  tufte=False)
                plot_box(df[col], stats=numericals[col], savePath=os.path.join(savePath, 'Box'), tufte=False)
                plot_violin(df[col], stats=numericals[col], savePath=os.path.join(savePath, 'Violin'), tufte=False)
            variables.append(col)
        if 'Numerical' in dtypes:
            plot_corrcov(df[dfNum], savePath=os.path.join(savePath, 'Heatmap'))
            plot_stats(df[dfNum], statsDict=numericals, savePath=os.path.join(savePath, 'StatisticComparison'))
        return variables, numericals, categoricals, distFits, regression


def format_plot(data=None, plotStyle='', plot=None, ax=None, title='', tufte=True):
    """
    name: string
    vector: pandas Series
    plotStyle: string
    plot0: Matplotlib plt
    ax: Matplotlib figure
    title: boolean
    """
    plot.title(title, fontsize='5', fontweight='bold', fontname='Georgia')

    formatter = globals()['format_{}'.format(plotStyle)]
    formatter(data, plot, ax)

    sns.set_context('paper', rc={'lines.linewidth': 1, 'figure.figsize': (1, 1), 'figure.facecolor': 'white',
                                 'font.family': ['sans-serif', ]})
    sns.set(font='serif', font_scale=.7)
    sns.set_style('ticks', {'font.family': 'serif', 'font.serif': ['Times', 'Palatino', 'serif']})
    plt.legend(fontsize=7, loc='best', frameon=False, borderaxespad=5)
    if tufte:
        sns.despine(top=True, right=True, trim=True)
    flatui = ["#00AFBB", "#ff9f00", "#CC79A7", "#009E73", "#66ccff",
              # Not as distinguishable with #00AFBB for colorblind
              "#F0E442"]
    sns.set_palette(flatui)  # sns.set_palette(sns.color_palette('Set2'))  # print(sns.color_palette('Set2'))


def format_box(data, plot, ax):
    plot.xlim(np.min(data) - 1, np.max(data) + 1)
    sns.boxplot(capsize=.5)


def format_heatmap(data, plot, ax):
    flatui = ["#00AFBB", "#ff9f00", "#CC79A7", "#009E73", "#66ccff",
              # Not as distinguishable with #00AFBB for colorblind
              "#F0E442"]
    sns.set_palette(flatui)


def format_histogram(data, plot, ax):
    plot.ylabel('Frequency')
    plot.xlim(np.min(data) - 1, np.max(data) + 1)


def format_scatter(data, plot, ax):
    xRange = np.max(data.index.values) - np.min(data.index.values)
    # strata = 5
    # xTicks = np.arange(np.min(data.index.values) - 1, np.max(data.index.values) + 1, step=int(xRange / strata))
    # plot0.xticks(xTicks)
    plot.xlim(np.min(data.index.values) - 1,
              np.max(data.index.values) + 1)  # plt.gca().set_aspect('equal', adjustable='box')


def format_table(data, plot, ax):
    """
    data: data
    plot0: plt
    ax: table
    return:
    """
    plot.axis('off')
    for key, cell in ax.get_celld().items():
        cell.set_linewidth(1)
    ax.auto_set_column_width(0)


@plot
def plot_box(data, stats=None, saveDir=PlotPaths.BOX_PLOTS_DIR, fileName='', title='', tufte=True):
    """
    v: pandas Series, (data)
    return: void
    """
    boxPlot = sns.boxplot(y=data, width=.25)
    return boxPlot, None


@plot
def plot_heatmap(matrix, saveDir=PlotPaths.HEATMAP_PLOTS_DIR, fileName='', title='', tufte=True):
    heatmapPlot = sns.heatmap(matrix, annot=False, vmin=0, vmax=None)
    return heatmapPlot, None


@plot
def plot_histogram(data, saveDir=PlotPaths.DISTRIBUTION_PLOTS_DIR, fileName='', title='', tufte=True):
    """
    v: pandas Series, (data)
    fit: bool
    return:
    """
    distPlot = sns.distplot(data, norm_hist=False, kde=False, bins=None, rug=True)
    return distPlot, None


def plot_stats(df, statsDict=None, saveDir=PlotPaths.TABLE_PLOTS_DIR, tufte=True):
    plt.gcf().clear()
    cellText = []
    colLabels = []
    s = 0
    for col in df:
        f = sorted(statsDict[col])
        cellText.append([str(statsDict[col][s]) for s in f])
        colLabels.append(col)
        if s == 0:
            rowLabels = [s for s in f]
            s += 1
    cellText = pd.DataFrame(cellText).T.as_matrix()
    table = plt.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, loc='center', rowLoc='center',
                      colLoc='center')
    table.auto_set_column_width(0)
    table.scale(2, 3)
    table.set_fontsize(12)
    plt.axis('off')
    # format_plot(name='', data=df, plotStyle='Table', plot0=plt, ax=table)
    export_plot(table, saveDir, '' + TABLE_PLOTS_SUFFIX)


@plot
def plot_violin(data, saveDir=PlotPaths.VIOLIN_PLOTS_DIR, fileName='', title='', tufte=True):
    violinPlot = sns.violinplot(y=data)
    return violinPlot, None


@GeneralUtil.timer
def generate_all_plots(df, dtypes, saveDir=PlotPaths.PLOTS_DIR, N=1):
    """
    This function returns all the information used in the creation of the plots
    df: pandas DataFrame
    dtypes: list of str, 'Numerical' or 'Categorical'
    saveDir: str, path where to save the plots as png
    return: list of str, dict, dict, dict, dict
    """
    iqrTableStats = ['Lower IQR', 'First Quartile', 'Median', 'Third Quartile', 'Upper IQR']
    dispersionTableStats = ['Standard Deviation', 'Variance', 'Skew', 'Kurtosis']
    centralTendenciesTableStats = ['Mean', 'Median']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lozoya.data_api.file_api.file_functions.clear_directories(saveDir)
        variables = []
        categoricals = {}
        numericals = {}
        distFits = {}
        regression = {}
        dfNum = []
        dfCat = []
        for col, label in zip(df, dtypes):
            if label == 'Categorical':
                categoricals[col] = StatisticsCollector.collect_stats(df[col], label)
                dfCat.append(col)
                plot_bar(df[col], stats=categoricals[col], saveDir=os.path.join(saveDir, 'Bar'),
                         fileName=col)  # plot_scatter(df[col], fit=False)  # plot_stats(df[col], stats0=['Mode'])

            elif label == 'Numerical':
                numericals[col] = StatisticsCollector.collect_stats(df[col], label)
                dfNum.append(col)  # TODO not sure what this is
                plot_table(OrderedDict([(k, numericals[col][k]) for k in iqrTableStats]),
                           saveDir=os.path.join(saveDir, 'IQR'), fileName=col, title='', tufte=False)
                plot_table(OrderedDict([(k, numericals[col][k]) for k in dispersionTableStats]),
                           saveDir=os.path.join(saveDir, 'Dispersion'), fileName=col, title='', tufte=False)
                plot_table(OrderedDict([(k, numericals[col][k]) for k in centralTendenciesTableStats]),
                           saveDir=os.path.join(saveDir, 'CentralTendencies'), fileName=col, title='', tufte=False)

                plot_histogram(df[col], saveDir=os.path.join(saveDir, 'Distribution'), fileName=col, tufte=False)
                _, distFits[col] = plot_distribution(df[col], N=N, saveDir=os.path.join(saveDir, 'DistributionFit'),
                                                     fileName=col, tufte=False)
                _, regression[col] = plot_scatter(df[col], fit=True, saveDir=os.path.join(saveDir, 'Scatter'),
                                                  fileName=col, tufte=False)
                plot_table(OrderedDict(
                    [(k, GeneralUtil.s_round(regression[col][1][k])) for k in sorted(regression[col][1].keys())]),
                    saveDir=os.path.join(saveDir, 'RegressionError'), fileName=col, title='', tufte=False)
                plot_table(
                    OrderedDict([(k, GeneralUtil.s_round(distFits[col][k][1])) for k in sorted(distFits[col].keys())]),
                    saveDir=os.path.join(saveDir, 'DistributionError'), fileName=col, title='', tufte=False)

                plot_box(df[col], saveDir=os.path.join(saveDir, 'Box'), fileName=col, tufte=False)
                plot_violin(df[col], saveDir=os.path.join(saveDir, 'Violin'), fileName=col, tufte=False)

                variables.append(col)
            if 'Numerical' in dtypes:
                plot_corrcov(df[dfNum], saveDir=os.path.join(saveDir, 'Heatmap'))
                """plot_stats(df[dfNum], statsDict=numericals,
                           saveDir=os.path.join(saveDir, 'StatisticComparison'))"""
        return variables, numericals, categoricals, distFits, regression


@plot
def plot_table(data, saveDir=PlotPaths.TABLE_PLOTS_DIR, fileName='', title='', tufte=True):
    """
    data: dictionary
    saveDir: str, path
    title: str
    tufte: boolean
    return:
    """
    cellText = [[str(data[k])] for k in data]
    rowLabels = [[k] for k in data]
    rowLabels = [GeneralUtil.text_formatter(r, 10) for r in rowLabels]
    table = plt.table(cellText=cellText, rowLabels=rowLabels, rowLoc='center', loc='center', colLoc='center')
    return table, None


os.environ["PATH"] += 'Z:\Family\VirtualEnvironments\Latex\miktex/bin/x64' + os.pathsep

sns.set_context('paper', rc={'lines.linewidth': 1, 'figure.figsize': (1, 1), 'figure.facecolor': 'white',
                             'font.family': ['sans-serif', ]})
sns.set(font='serif', font_scale=.7)
sns.set_style('ticks', {'font.family': 'serif', 'font.serif': ['Times', 'Palatino', 'serif']})

plt.legend(fontsize=7, loc='best', frameon=False, borderaxespad=5)


def set_style(tufte):
    sns.set_context('paper', rc={'lines.linewidth': 1, 'figure.figsize': (1, 1), 'figure.facecolor': 'white',
                                 'font.family': ['sans-serif', ]})
    sns.set(font='serif', font_scale=.7)
    sns.set_style('ticks', {'font.family': 'serif', 'font.serif': ['Times', 'Palatino', 'serif']})
    plt.legend(fontsize=7, loc='best', frameon=False, borderaxespad=5)
    if tufte:
        sns.despine(top=True, right=True, trim=True)
    flatui = ["#00AFBB", "#ff9f00", "#CC79A7", "#009E73", "#66ccff",
              # Not as distinguishable with #00AFBB for colorblind
              "#F0E442"]
    sns.set_palette(flatui)  # sns.set_palette(sns.color_palette('Set2'))  # print(sns.color_palette('Set2'))


def format_plot(data=None, plotStyle='', plot=None, ax=None, name=None, title=False, tufte=True):
    """
    name: string
    vector: pandas Series
    plotStyle: string
    plot0: Matplotlib plt
    ax: Matplotlib figure
    title: boolean
    """
    if title == True:
        plot.title(name + ' ' + plotStyle, fontsize='5', fontweight='bold', fontname='Georgia')

    if plotStyle == 'Bar Plot':
        format_bar(data, plot)
    elif plotStyle == 'Box Plot':
        format_box(data, plot)
    elif plotStyle == 'Distribution Fit Plot':
        format_distribution_fit(data, plot)
    elif plotStyle == 'Distribution Plot' or plotStyle == 'Histogram':
        format_histogram(data, plot)
    elif plotStyle == 'Heatmap Plot':
        format_heatmap(data, plot)
    elif plotStyle == 'Scatter Plot':
        format_scatter(data, plot)
    elif plotStyle == 'Table':
        format_table(data, plot)
    elif plotStyle == 'Violin Plot':
        format_violin(data, plot)

    set_style(tufte=tufte)


def format_bar(data, plot):
    plot.ylabel('Quantity')
    plot.ylim(np.min(data.index.values) - 1, np.max(data.index.values) + 1)


def format_distribution_fit(data, plot):
    plot.ylabel('Probability')
    plot.xlim(np.min(data) - 1, np.max(data) + 1)


def format_heatmap(data, plot):
    sns.heatmap(vmin=0, vmax=None)


def format_histogram(data, plot):
    plot.ylabel('Frequency')
    plot.xlim(np.min(data) - 1, np.max(data) + 1)


def format_scatter(data, plot):
    xRange = np.max(data.index.values) - np.min(data.index.values)
    # strata = 5
    # xTicks = np.arange(np.min(data.index.values) - 1, np.max(data.index.values) + 1, step=int(xRange / strata))
    # plot0.xticks(xTicks)
    plot.xlim(np.min(data.index.values) - 1,
              np.max(data.index.values) + 1)  # plt.gca().set_aspect('equal', adjustable='box')


def format_table(data, plot):
    plot.axis('off')  # table.set_fontsize(12)


def format_violin(data, plot):
    plot.ylabel('Frequency')
    plot.xlim(np.min(data) - 1, np.max(data) + 1)
    sns.violinplot(scale='count', linewidth=3, )


def plot_stats(df, statsDict=None, saveDir=PlotPaths.TABLE_PLOTS_DIR, tufte=True):
    plt.gcf().clear()
    cellText = []
    colLabels = []
    s = 0
    for col in df:
        f = sorted(statsDict[col])
        cellText.append([str(statsDict[col][s]) for s in f])
        colLabels.append(col)
        if s == 0:
            rowLabels = [s for s in f]
            s += 1
    cellText = pd.DataFrame(cellText).T.as_matrix()
    table = plt.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, loc='center', rowLoc='center',
                      colLoc='center')
    table.auto_set_column_width(0)
    table.scale(2, 3)
    table.set_fontsize(12)
    plt.axis('off')
    # format_plot(name='', data=df, plotStyle='Table', plot0=plt, ax=table)
    export_plot(table, saveDir, '' + TABLE_PLOTS_SUFFIX)


def plot_stats(df, statsDict=None, saveDir=PlotPaths.TABLE_PLOTS_DIR, tufte=True):
    plt.gcf().clear()
    cellText = []
    colLabels = []
    s = 0
    for col in df:
        f = sorted(statsDict[col])
        cellText.append([str(statsDict[col][s]) for s in f])
        colLabels.append(col)
        if s == 0:
            rowLabels = [s for s in f]
            s += 1
    cellText = pd.DataFrame(cellText).T.as_matrix()
    table = plt.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, loc='center', rowLoc='center',
                      colLoc='center')
    table.auto_set_column_width(0)
    table.scale(2, 3)
    table.set_fontsize(12)
    plt.axis('off')
    # format_plot(name='', data=df, plotStyle='Table', plot0=plt, ax=table)
    export_plot(table, saveDir, '' + TABLE_PLOTS_SUFFIX)


class Plotter:
    def __init__(self, style='bmh'):
        # Setting the font and style
        matplotlib.rcParams['font.serif'] = "Times New Roman"
        matplotlib.rcParams['font.family'] = "serif"
        self.font = {'fontname': 'Times New Roman'}
        plt.style.use(style)

    def Plot(self, figure, input_data, subplot, step_size=1, legend_labels_list=[], starting_year=1,
             x_axis_label='Time (Years)', y_axis_label='State', title='Monte Carlo Simulation2 of Item Value'):
        # Setting labels and style
        self.y_axis, self.step_size, self.starting_year, self.x_axis_label, self.y_axis_label, self.title = input_data, step_size, starting_year, x_axis_label, y_axis_label, title
        self.figure = figure
        self.size = 0
        self.line_styles = ['.--', '*--', 'x--', '^--']  # The plotter wraps around these line style
        # Creating the plot0
        plt.figure(self.figure)
        plt.subplot(subplot)
        # This for loop is plotting all the points in each list in the input_data list.
        for i in range(self.y_axis.__len__()):
            self.x_axis = []
            self.x_axis = np.arange(self.starting_year, self.starting_year + self.y_axis[0].__len__(), self.step_size)
            if legend_labels_list.__len__() != 0:
                plt.plot(self.x_axis, self.y_axis[i], self.line_styles[i % self.line_styles.__len__()],
                         label=legend_labels_list[i % legend_labels_list.__len__()], linewidth=0.5)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)  # Placing a legend beside the plot0
            else:
                plt.plot(self.x_axis, self.y_axis[i], self.line_styles[i % self.line_styles.__len__()], linewidth=0.5)
        self.Generate_Graph()

    def Histogram(self, figure, input_data, x_axis_label='State', y_axis_label='Frequency',
                  title='Frequency Distribution'):
        # Setting labels
        self.y_axis, self.x_axis_label, self.y_axis_label, self.title = input_data, x_axis_label, y_axis_label, title
        self.figure = figure
        # Creating the figure
        plt.figure(self.figure)
        self.title = str('Frequency Distribution at Year ' + str(int(self.figure) - 1))
        # Creating the plot0
        plt.hist(self.y_axis, bins=20, facecolor='pink')
        self.Generate_Graph()

    def Bar_Chart_3D(self, graph, y_input_data, z_input_data, iterations, x_axis_label='Time (Years)',
                     y_axis_label='State', z_axis_label='Frequency', title='Frequency Distribution'):
        # Setting labels
        self.y_axis, self.z_axis, self.x_axis_label, self.y_axis_label, self.z_axis_label, self.title = y_input_data, z_input_data, x_axis_label, y_axis_label, z_axis_label, title
        self.iterations = iterations
        self.title = 'Frequency Distribution'
        self.graph = graph
        # TODO
        xpos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        ypos = y_input_data
        zpos = np.zeros_like(xpos)
        # Construct arrays with the dimensions for the bars.
        dx = 0.05 * np.ones_like(zpos)
        dy = 0.05 * np.ones_like(zpos)
        dz = z_input_data
        # Creating the plot0
        self.graph.bar3d(xpos, ypos, zpos, dx, dy, dz, color='teal')
        self.Generate_Graph_3D()

    def Generate_Graph(self):
        # Placing the axis labels and the title on the graph2
        plt.xlabel(self.x_axis_label, **self.font)
        plt.ylabel(self.y_axis_label, **self.font)
        plt.title(self.title, **self.font)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # This and the preceding line ensure a full screen display upon rendering

    def Generate_Graph_3D(self):
        # Placing the axis labels and the title on the graph2
        self.bar_chart_3d.set_xlabel(self.x_axis_label)
        self.bar_chart_3d.set_ylabel(self.y_axis_label)
        self.bar_chart_3d.set_zlabel(self.z_axis_label)
        plt.title(self.title)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # This and the preceding line ensure a full screen display upon rendering

    def Generate_Subplot_3D(self, figure):
        # Creating the figure
        self.bar_chart_3d = plt.figure(figure).add_subplot(111, projection='3d')
        return self.bar_chart_3d

    def Show(self):
        plt.grid(True)
        plt.show()


class PlotGenerator:
    @staticmethod
    def plot_curve(plt, index, curve, color, lims, fill):
        plt.plot([x for x in range(lims[0][0], lims[0][0] + lims[0][1])], curve,
                 GRAPH_COLORS[color][index % len(GRAPH_COLORS[color])],
                 marker=GRAPH_MARKERS[index % len(GRAPH_MARKERS)], markersize=2,
                 fillstyle=GRAPH_FILL_STYLES[index % len(GRAPH_FILL_STYLES)],
                 dashes=GRAPH_DASHES[index % len(GRAPH_DASHES)])
        if fill:
            plt.fill_between([x for x in range(lims[0][0], lims[0][0] + lims[0][1])], 0, curve,
                             facecolors=GRAPH_COLORS[color][index % len(GRAPH_COLORS[color])], alpha=0.5)

    @staticmethod
    def plot_histogram(plt, index, curve, color, lims, norm=False):
        plt.hist(curve, alpha=0.5, bins=np.arange(lims[0][1] + 1), normed=False, align='left',
                 facecolor=GRAPH_COLORS[color][index % len(GRAPH_COLORS[color])])

    @staticmethod
    def plot_configure(plt, legend, title, labels, lims):
        if legend is not None:
            plt.legend(legend, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)

        plt.set_title(title)
        plt.set_xlabel(labels[0])
        plt.set_ylabel(labels[1])
        if lims:
            plt.set_xlim(lims[0])
            plt.set_ylim(lims[1])
            plt.axes.set_xticks([v for v in range(lims[0][0], lims[0][1] + 1)])


class PlotGenerator:
    @staticmethod
    def plot_curve(plt, index, curve, color, lims, fill):
        plt.plot([x for x in range(lims[0][0], lims[0][0] + lims[0][1])], curve,
                 colors[color][index % len(colors[color])], marker=mark[index % len(mark)], markersize=2,
                 fillstyle=fill_style[index % len(fill_style)], dashes=dash[index % len(dash)])
        if fill:
            plt.fill_between([x for x in range(lims[0][0], lims[0][0] + lims[0][1])], 0, curve,
                             facecolors=colors[color][index % len(colors[color])], alpha=0.5)

    @staticmethod
    def plot_histogram(plt, index, curve, color, lims, norm=False):
        plt.hist(curve, alpha=0.5, bins=np.arange(lims[0][1] + 1), normed=norm, align='left',
                 facecolor=colors[color][index % len(colors[color])])


def add_plots(figure, x, y, type='line', lineWidth=None):
    """
    figure: Bokeh figure
    x: list
    y: list of list
    return: list of Bokeh lines
    """
    if lineWidth == None:
        lineWidth = 0.1
    if type == 'line':
        f = figure.line
    elif type == 'scatter':
        f = figure.scatter
    elif type == 'bar':
        f = figure.vbars

    lines = []
    for j, column in enumerate(y):
        if type == 'bar':
            try:
                lines.append(f(x, top=y.loc[:, column], width=0.1,
                               fill_color='#' + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])))
            except Exception as e:
                print(e)
        else:
            lines.append(f(x, y.loc[:, column], line_width=lineWidth,
                           line_color='#' + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])))
    return lines


def determine_names(Y, legends):
    names = []
    for i, y in enumerate(Y):
        for j, dF in enumerate(y):
            if legends[i][j]:
                if j > 0:
                    names[i].append([str(name) for name in dF.columns])
                else:
                    names.append([[str(name) for name in dF.columns]])
    return names


def make_image(figure, xRange, yRange, data):
    dx = xRange[1] - xRange[0]
    dy = yRange[1] - yRange[0]
    figure.image(image=[data], x=xRange[0], y=yRange[0], dw=dx, dh=dy, palette='Spectral11')


def make_legend_items(lines, names):
    """
    lines: list of lists of Bokeh lines
    names: list of lists of str
    return: list of tuples
    """
    items = []
    for name, line in zip(names, lines):
        for n, l in zip(name, line):
            items.append((n, [l]))
    return items


def make_contour_plot(file, X, Y, boundaries, titles, xLabels, yLabels, names=None, lineWidths=None, legends=None,
                      types=None, xKeys=(None,), yKeys=(None,), colors=(None,)):
    temporaryFile = os.path.join(GRAPHS, 'temp.html')
    output_file(temporaryFile)
    if legends == None:
        legends = [(True,) for i in range(len(Y))]
    if types == None:
        types = [('line',) for i in range(len(Y))]
    if names == None:
        names = determine_names(Y, legends)
    plotList = []
    xRange, yRange, data = boundaries[0], boundaries[1], boundaries[2]

    for i in range(len(Y)):
        linesList = []
        for j in range(len(Y[i])):
            if j == 0:
                p = make_figure(title=titles[i], tools=make_tools(xLabels[i], yLabels[i]), xLabel=xLabels[i],
                                yLabel=yLabels[i], xKeys=xKeys[i], yKeys=yKeys[i], xRange=xRange, yRange=yRange,
                                bound=True)
                make_image(p, xRange, yRange, data)
            lines = add_plots(p, x=X[i][j].astype(float), y=Y[i][j].astype(float), type=types[i][j],
                              lineWidth=lineWidths[i][j])
            if legends[i][j]:
                linesList.append(lines)
        set_legend(p, linesList, names[i])
        plotList.append(p)
    lay = row(plotList, sizing_mode='stretch_both')
    save(lay)
    _fix_html(temporaryFile, file)


def make_plot(file, X, Y, titles, xLabels, yLabels, names=None, lineWidths=None, legends=None, types=None,
              xKeys=(None,), yKeys=(None,), colors=(None,)):
    """
    X: list of lists
    Y: list of lists
    names: list lists of str to specify the label to use for each line in the legend
    titles: list of str labels to use as title for each figure
    xLabels: list of str labels to use as x axis for each figure
    yLabels: list of str labels to use as y axis for each figure
    type: list lists of str 'line', 'histogram', 'scatter' to specify the type of line
    lineWidths: list of lists of int to specify the width of each line
    legends: list of lists of booleans to determine whether to represent lines in the legend
    return:
    """
    temporaryFile = os.path.join(GRAPHS, 'temp.html')
    output_file(temporaryFile)
    if legends == None:
        legends = [(True,) for i in range(len(Y))]
    if types == None:
        types = [('line',) for i in range(len(Y))]
    if names == None:
        names = determine_names(Y, legends)
    plotlist_function(X, Y, file, legends, lineWidths, names, temporaryFile, titles, types, xKeys, xLabels, yKeys,
                      yLabels)


def set_style(figure):
    figure.axis.axis_label_standoff = 20
    figure.axis.axis_label_text_color = CONTRASTED_COLOR
    figure.axis.axis_line_width = 1
    figure.axis.axis_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_in = 5
    figure.axis.major_tick_out = 10
    figure.axis.minor_tick_line_color = CONTRASTED_COLOR
    figure.axis.minor_tick_in = -2
    figure.axis.minor_tick_out = 7
    figure.axis.major_label_text_color = CONTRASTED_COLOR

    figure.background_fill_color = GRAPH_BACKGROUND_COLOR
    # figure.background_fill_alpha = 0.5
    figure.border_fill_color = BACKGROUND_COLOR_LIGHT

    figure.grid.grid_line_alpha = 0.5
    figure.grid.grid_line_dash = [2, 4]
    figure.grid.grid_line_color = CONTRASTED_COLOR

    figure.min_border_bottom = 100
    figure.min_border_left = 50
    figure.min_border_right = 50
    figure.min_border_top = 100

    # figure.outline_line_width = 5
    # figure.outline_line_alpha = 0.5
    figure.outline_line_color = CONTRASTED_COLOR

    figure.title.text_color = CONTRASTED_COLOR  # figure.title.text_font = ''  # figure.title.text_font_style = ''


def add_plots(figure, x, y, type='line', lineWidth=None):
    """
    figure: Bokeh figure
    x: list
    y: list of list
    return: list of Bokeh lines
    """
    if lineWidth == None:
        lineWidth = 0.1
    if type == 'line':
        f = figure.line
    elif type == 'scatter':
        f = figure.scatter

    lines = []
    for j, column in enumerate(y):
        lines.append(f(x, y.loc[:, column], line_width=lineWidth,
                       line_color='#' + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])))
    return lines


def make_figure(title, tools, xLabel, yLabel, xKeys, yKeys):
    p = figure(title=title, tools=tools, toolbar_location='above', x_axis_label=xLabel, y_axis_label=yLabel, logo=None)
    if xKeys != None:
        p.xaxis.bounds = (min(xKeys, key=xKeys.get), max(xKeys, key=xKeys.get))
        p.xaxis.ticker = [key for key in xKeys]
        lozoya.data_api.html_api.html_apicrawl.formatter = FuncTickFormatter(code="""
            var labels = %s;
            return labels[tick];""" % xKeys)
        p.xaxis.major_label_orientation = -pi / 3

    if yKeys != None:
        p.yaxis.bounds = (min(yKeys, key=yKeys.get), max(yKeys, key=yKeys.get))
        p.yaxis.ticker = [key for key in yKeys]
        lozoya.data_api.html_api.html_apicrawl.formatter = FuncTickFormatter(code="""
            var labels = %s;
            return labels[tick];""" % yKeys)

    return p


def make_tabbed_plot(file, X, Y, titles, xLabels, yLabels, names=None, lineWidths=None, legends=None, types=None,
                     xKeys=(None,), yKeys=(None,)):
    """
    X: list of lists
    Y: list of lists
    names: list lists of str to specify the label to use for each line in the legend
    titles: list of str labels to use as title for each figure
    xLabels: list of str labels to use as x axis for each figure
    yLabels: list of str labels to use as y axis for each figure
    type: list lists of str 'line', 'histogram', 'scatter' to specify the type of line
    lineWidths: list of lists of int to specify the width of each line
    legends: list of lists of booleans to determine whether to represent lines in the legend
    return:
    """
    output_file(file)
    if legends == None:
        legends = [(True,) for i in range(len(Y))]
    if types == None:
        types = [('line',) for i in range(len(Y))]
    if names == None:
        names = []
        for i, y in enumerate(Y):
            for j, dF in enumerate(y):
                if legends[i][j]:
                    if j > 0:
                        names[i].append([str(name) for name in dF.columns])
                    else:
                        names.append([[str(name) for name in dF.columns]])
    tabsList = []
    for i in range(len(Y)):
        linesList = []
        for j in range(len(Y[i])):
            if j == 0:
                p = make_figure(title=titles[i], tools=make_tools(xLabels[i], yLabels[i]), xLabel=xLabels[i],
                                yLabel=yLabels[i], xKeys=xKeys[i], yKeys=yKeys[i])
            try:
                x = X[i][j].astype(float)
            except:
                x = X[i][j]
            lines = add_plots(p, x, Y[i][j].astype(float), type=types[i][j], lineWidth=lineWidths[i][j])
            if legends[i][j]:
                linesList.append(lines)
        set_legend(p, linesList, names[i])
        tabsList.append(Panel(child=p, title=titles[i]))

    tabs = Tabs(tabs=tabsList)
    save(tabs)


def make_tabbed_table(path, dataFrameList, titles):
    """
    Creates a bokeh html file containing a table containing the data in dataFrame
    path: str
    dataFrameList: list of pandas DataFrame
    titles: list of str
    return: None
    """
    output_file(path)
    tabsList = []
    for dataFrame, title in zip(dataFrameList, titles):
        source = ColumnDataSource(dataFrame)
        names = dataFrame.columns
        columns = [TableColumn(field=names[i], title=names[i]) for i in range(len(names))]
        data_table = DataTable(source=source, columns=columns, width=400, height=280)
        tabsList.append(Panel(child=data_table, title=title))
    tabs = Tabs(tabs=tabsList)
    save(tabs)


def set_legend(figure, lines, names):
    """
    figure: Bokeh figure
    lines: list of lists of Bokeh lines
    names: list of lists of str
    return: None
    """
    items = make_legend_items(lines, names)
    legend = Legend(items=items)
    figure.add_layout(legend, 'right')
    figure.legend.click_policy = 'hide'


def make_image(figure, xRange, yRange, data, colors):
    dx = xRange[1] - xRange[0]
    dy = yRange[1] - yRange[0]
    colorSet = colors
    figure.image(image=[data], x=xRange[0], y=yRange[0], dw=dx, dh=dy, palette=colorSet)


def make_contour_plot(file, X, Y, boundaries, titles, xLabels, yLabels, names=None, lineWidths=None, legends=None,
                      types=None, xKeys=(None,), yKeys=(None,), colors=None):
    temporaryFile = os.path.join(GRAPHS, 'temp.html')
    output_file(temporaryFile)
    if legends == None:
        legends = [(True,) for i in range(len(Y))]
    if types == None:
        types = [('line',) for i in range(len(Y))]
    if names == None:
        names = determine_names(Y, legends)
    if colors == None:
        colors = [[[None] for j in range(len(Y[i]))] for i in range(len(Y))]

    plotList = []
    xRange, yRange, data = boundaries[0], boundaries[1], boundaries[2]

    for i in range(len(Y)):
        linesList = []
        for j in range(len(Y[i])):
            if j == 0:
                p = make_figure(title=titles[i], tools=make_tools(xLabels[i], yLabels[i]), xLabel=xLabels[i],
                                yLabel=yLabels[i], xKeys=xKeys[i], yKeys=yKeys[i], xRange=xRange, yRange=yRange,
                                bound=True)
                make_image(p, xRange, yRange, data, COLOR_SET[:len(colors[i])])
            lines = add_plots(p, x=X[i][j].astype(float), y=Y[i][j].astype(float), type=types[i][j],
                              lineWidth=lineWidths[i][j], colors=colors[i][j])
            if legends[i][j]:
                linesList.append(lines)
        set_legend(p, linesList, names[i])
        plotList.append(p)
    lay = row(plotList, sizing_mode='stretch_both')
    save(lay)
    _fix_html(temporaryFile, file)


def set_style(figure):
    figure.axis.axis_label_standoff = 20
    figure.axis.axis_label_text_color = CONTRASTED_COLOR
    figure.axis.axis_line_width = 1
    figure.axis.axis_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_in = 5
    figure.axis.major_tick_out = 10
    figure.axis.minor_tick_line_color = CONTRASTED_COLOR
    figure.axis.minor_tick_in = -2
    figure.axis.minor_tick_out = 7
    figure.axis.major_label_text_color = CONTRASTED_COLOR
    figure.background_fill_color = GRAPH_BACKGROUND_COLOR
    # figure.background_fill_alpha = 0.5
    figure.border_fill_color = BACKGROUND_COLOR_LIGHT
    figure.grid.grid_line_alpha = 0.5
    figure.grid.grid_line_dash = [2, 4]
    figure.grid.grid_line_color = CONTRASTED_COLOR
    figure.min_border_bottom = 100
    figure.min_border_left = 50
    figure.min_border_right = 50
    figure.min_border_top = 100
    # figure.outline_line_width = 5
    # figure.outline_line_alpha = 0.5
    figure.outline_line_color = CONTRASTED_COLOR
    figure.title.text_color = CONTRASTED_COLOR  # figure.title.text_font = ''  # figure.title.text_font_style = ''


def set_style(figure):
    figure.axis.axis_label_standoff = 20
    figure.axis.axis_label_text_color = CONTRASTED_COLOR
    figure.axis.axis_line_width = 1
    figure.axis.axis_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_in = 5
    figure.axis.major_tick_out = 10
    figure.axis.minor_tick_line_color = CONTRASTED_COLOR
    figure.axis.minor_tick_in = -2
    figure.axis.minor_tick_out = 7
    figure.axis.major_label_text_color = CONTRASTED_COLOR
    figure.background_fill_color = GRAPH_BACKGROUND_COLOR
    # figure.background_fill_alpha = 0.5
    figure.border_fill_color = BACKGROUND_COLOR_LIGHT
    figure.grid.grid_line_alpha = 0.5
    figure.grid.grid_line_dash = [2, 4]
    figure.grid.grid_line_color = CONTRASTED_COLOR
    figure.min_border_bottom = 100
    figure.min_border_left = 50
    figure.min_border_right = 50
    figure.min_border_top = 100
    # figure.outline_line_width = 5
    # figure.outline_line_alpha = 0.5
    figure.outline_line_color = CONTRASTED_COLOR
    figure.title.text_color = CONTRASTED_COLOR  # figure.title.text_font = ''  # figure.title.text_font_style = ''


def set_style(figure):
    figure.axis.axis_label_standoff = 20
    figure.axis.axis_label_text_color = CONTRASTED_COLOR
    figure.axis.axis_line_width = 1
    figure.axis.axis_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_in = 5
    figure.axis.major_tick_out = 10
    figure.axis.minor_tick_line_color = CONTRASTED_COLOR
    figure.axis.minor_tick_in = -2
    figure.axis.minor_tick_out = 7
    figure.axis.major_label_text_color = CONTRASTED_COLOR
    figure.background_fill_color = GRAPH_BACKGROUND_COLOR
    # figure.background_fill_alpha = 0.5
    figure.border_fill_color = BACKGROUND_COLOR_LIGHT
    figure.grid.grid_line_alpha = 0.5
    figure.grid.grid_line_dash = [2, 4]
    figure.grid.grid_line_color = CONTRASTED_COLOR
    figure.min_border_bottom = 100
    figure.min_border_left = 50
    figure.min_border_right = 50
    figure.min_border_top = 100
    # figure.outline_line_width = 5
    # figure.outline_line_alpha = 0.5
    figure.outline_line_color = CONTRASTED_COLOR
    figure.title.text_color = CONTRASTED_COLOR  # figure.title.text_font = ''  # figure.title.text_font_style = ''


def add_plots(figure, x, y, type='line', lineWidth=None, colors=None):
    """
    figure: Bokeh figure
    x: list
    y: list of list
    return: list of Bokeh lines
    """
    if lineWidth == None:
        lineWidth = 0.1
    if type == 'line':
        f = figure.line
    elif type == 'scatter':
        f = figure.scatter
    elif type == 'bar':
        f = figure.vbars

    lines = []
    for j, column in enumerate(y):
        if colors:
            color = colors[j]
        else:
            color = '#' + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])
        if type == 'bar':
            try:
                lines.append(f(x, top=y.loc[:, column], width=0.1, fill_color=color))
            except Exception as e:
                print(e)
        else:
            lines.append(f(x, y.loc[:, column], line_width=lineWidth, line_color=color))
    return lines


def set_style(figure):
    figure.axis.axis_label_standoff = 20
    figure.axis.axis_label_text_color = CONTRASTED_COLOR
    figure.axis.axis_line_width = 1
    figure.axis.axis_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_in = 5
    figure.axis.major_tick_out = 10
    figure.axis.minor_tick_line_color = CONTRASTED_COLOR
    figure.axis.minor_tick_in = -2
    figure.axis.minor_tick_out = 7
    figure.axis.major_label_text_color = CONTRASTED_COLOR

    figure.background_fill_color = GRAPH_BACKGROUND_COLOR
    # figure.background_fill_alpha = 0.5
    figure.border_fill_color = BACKGROUND_COLOR_LIGHT

    figure.grid.grid_line_alpha = 0.5
    figure.grid.grid_line_dash = [2, 4]
    figure.grid.grid_line_color = CONTRASTED_COLOR

    figure.min_border_bottom = 100
    figure.min_border_left = 50
    figure.min_border_right = 50
    figure.min_border_top = 100

    # figure.outline_line_width = 5
    # figure.outline_line_alpha = 0.5
    figure.outline_line_color = CONTRASTED_COLOR

    figure.title.text_color = CONTRASTED_COLOR  # figure.title.text_font = ''  # figure.title.text_font_style = ''


def make_figure(title, tools, xLabel, yLabel, xKeys, yKeys):
    p = figure(title=title, tools=tools, toolbar_location='above', x_axis_label=xLabel, y_axis_label=yLabel, logo=None,
               sizing_mode='stretch_both')
    set_style(p)

    if xKeys != None:
        p.xaxis.bounds = (min(xKeys, key=xKeys.get), max(xKeys, key=xKeys.get))
        p.xaxis.ticker = [key for key in xKeys]
        lozoya.data_api.html_api.html_apicrawl.formatter = FuncTickFormatter(code="""
            var labels = %s;
            return labels[tick];""" % xKeys)
        p.xaxis.major_label_orientation = -pi / 3

    if yKeys != None:
        p.yaxis.bounds = (min(yKeys, key=yKeys.get), max(yKeys, key=yKeys.get))
        p.yaxis.ticker = [key for key in yKeys]
        lozoya.data_api.html_api.html_apicrawl.formatter = FuncTickFormatter(code="""
            var labels = %s;
            return labels[tick];""" % yKeys)

    return p


def set_style(figure):
    figure.axis.axis_label_standoff = 20
    figure.axis.axis_label_text_color = 'white'
    figure.axis.axis_line_width = 1
    figure.axis.axis_line_color = 'white'
    figure.axis.major_tick_line_color = 'white'
    figure.axis.major_tick_in = 5
    figure.axis.major_tick_out = 10
    figure.axis.minor_tick_line_color = 'white'
    figure.axis.minor_tick_in = -2
    figure.axis.minor_tick_out = 7
    figure.axis.major_label_text_color = 'white'

    figure.background_fill_color = BACKGROUND_COLOR_LIGHTERER
    # figure.background_fill_alpha = 0.5
    figure.border_fill_color = BACKGROUND_COLOR_LIGHT

    figure.grid.grid_line_alpha = 0.5
    figure.grid.grid_line_dash = [2, 1]
    figure.grid.grid_line_color = 'white'

    figure.min_border_bottom = 100
    figure.min_border_left = 50
    figure.min_border_right = 50
    figure.min_border_top = 100

    # figure.outline_line_width = 5
    # figure.outline_line_alpha = 0.5
    figure.outline_line_color = BACKGROUND_COLOR_LIGHTER

    figure.title.text_color = 'white'  # figure.title.text_font = ''  # figure.title.text_font_style = ''


def make_tabbed_plot(file, X, Y, titles, xLabels, yLabels, names=None, lineWidths=None, legends=None, types=None,
                     xKeys=(None,), yKeys=(None,)):
    """
    X: list of lists
    Y: list of lists
    names: list lists of str to specify the label to use for each line in the legend
    titles: list of str labels to use as title for each figure
    xLabels: list of str labels to use as x axis for each figure
    yLabels: list of str labels to use as y axis for each figure
    type: list lists of str 'line', 'histogram', 'scatter' to specify the type of line
    lineWidths: list of lists of int to specify the width of each line
    legends: list of lists of booleans to determine whether to represent lines in the legend
    return:
    """
    output_file(file)
    if legends == None:
        legends = [(True,) for i in range(len(Y))]
    if types == None:
        types = [('line',) for i in range(len(Y))]
    if names == None:
        names = []
        for i, y in enumerate(Y):
            for j, dF in enumerate(y):
                if legends[i][j]:
                    if j > 0:
                        names[i].append([str(name) for name in dF.columns])
                    else:
                        names.append([[str(name) for name in dF.columns]])
    tabsList = []
    for i in range(len(Y)):
        linesList = []
        for j in range(len(Y[i])):
            if j == 0:
                p = make_figure(title=titles[i], tools=make_tools(xLabels[i], yLabels[i]), xLabel=xLabels[i],
                                yLabel=yLabels[i], xKeys=xKeys[i], yKeys=yKeys[i])

            lines = add_plots(p, X[i][j].astype(float), Y[i][j].astype(float), type=types[i][j],
                              lineWidth=lineWidths[i][j])
            if legends[i][j]:
                linesList.append(lines)
        set_legend(p, linesList, names[i])
        tabsList.append(Panel(child=p, title=titles[i]))

    tabs = Tabs(tabs=tabsList)
    save(tabs)


def make_plot(file, X, Y, titles, xLabels, yLabels, names=None, lineWidths=None, legends=None, types=None,
              xKeys=(None,), yKeys=(None,), classLabels=(None,)):
    """
    X: list of lists
    Y: list of lists
    names: list lists of str to specify the label to use for each line in the legend
    titles: list of str labels to use as title for each figure
    xLabels: list of str labels to use as x axis for each figure
    yLabels: list of str labels to use as y axis for each figure
    type: list lists of str 'line', 'histogram', 'scatter' to specify the type of line
    lineWidths: list of lists of int to specify the width of each line
    legends: list of lists of booleans to determine whether to represent lines in the legend
    return:
    """
    temporaryFile = os.path.join(GRAPHS, 'temp.html')
    output_file(temporaryFile)
    if legends == None:
        legends = [(True,) for i in range(len(Y))]
    if types == None:
        types = [('line',) for i in range(len(Y))]
    if names == None:
        names = []
        for i, y in enumerate(Y):
            for j, dF in enumerate(y):
                if legends[i][j]:
                    if j > 0:
                        names[i].append([str(name) for name in dF.columns])
                    else:
                        names.append([[str(name) for name in dF.columns]])
    plotlist_function(X, Y, file, legends, lineWidths, names, temporaryFile, titles, types, xKeys, xLabels, yKeys,
                      yLabels)


def plotlist_function(X, Y, file, legends, lineWidths, names, temporaryFile, titles, types, xKeys, xLabels, yKeys,
                      yLabels):
    plotList = []
    for i in range(len(Y)):
        linesList = []
        for j in range(len(Y[i])):
            if j == 0:
                p = make_figure(title=titles[i], tools=make_tools(xLabels[i], yLabels[i]), xLabel=xLabels[i],
                                yLabel=yLabels[i], xKeys=xKeys[i], yKeys=yKeys[i])

            lines = add_plots(p, x=X[i][j].astype(float), y=Y[i][j].astype(float), type=types[i][j],
                              lineWidth=lineWidths[i][j])
            if legends[i][j]:
                linesList.append(lines)
        set_legend(p, linesList, names[i])
        plotList.append(p)
    lay = row(plotList, sizing_mode='stretch_both')
    save(lay)
    _fix_html(temporaryFile, file)


def set_style(figure):
    figure.axis.axis_label_standoff = 20
    figure.axis.axis_label_text_color = 'white'
    figure.axis.axis_line_width = 1
    figure.axis.axis_line_color = 'white'
    figure.axis.major_tick_line_color = 'white'
    figure.axis.major_tick_in = 5
    figure.axis.major_tick_out = 10
    figure.axis.minor_tick_line_color = 'white'
    figure.axis.minor_tick_in = -2
    figure.axis.minor_tick_out = 7
    figure.axis.major_label_text_color = 'white'

    figure.background_fill_color = BACKGROUND_COLOR_LIGHTERER
    # figure.background_fill_alpha = 0.5
    figure.border_fill_color = BACKGROUND_COLOR_LIGHT

    figure.grid.grid_line_alpha = 0.5
    figure.grid.grid_line_dash = [2, 1]
    figure.grid.grid_line_color = 'white'

    figure.min_border_bottom = 100
    figure.min_border_left = 50
    figure.min_border_right = 50
    figure.min_border_top = 100

    # figure.outline_line_width = 5
    # figure.outline_line_alpha = 0.5
    figure.outline_line_color = BACKGROUND_COLOR_LIGHTER

    figure.title.text_color = 'white'  # figure.title.text_font = ''  # figure.title.text_font_style = ''


def delegate(df, col, label):
    if label == 'Nominal':
        plot_bars(df[col])
        plot_scatter(df[col], fit=False)
        plot_stats(df[cols], stats=['Mode'])

    elif label == 'Ordinal':
        plot_bars(df[col])
        plot_scatter(df[col], fit=False)
        plot_stats(df[col], stats=['Mean', 'Median', 'Mode'])  # Std, Variance?

    elif label == 'Continuous':
        plot_histogram(df[col])
        plot_scatter(df[col])
        plot_box(df[col])
        plot_violin(df[col])
        plot_stats(df[col])


# TODO ANNOTATE HEATMAP
# just add annot=True to heatmap call


def plot_bars(df):
    for col in df:
        plt.gcf().clear()
        barPlot = sns.barplot(data=df[col])
        format_plot(col, "Bar Plot", plt)
        export_plot(barPlot, "BarPlots", col + "_barplot.png")


def plot_violin(df):
    for col in df:
        plt.gcf().clear()
        violinPlot = sns.violinplot(data=df[col], cut=0, scale="width")
        format_plot(col, "Violin Plot", plt)
        export_plot(violinPlot, "ViolinPlots", col + "_violin.png"))

        def plot_table(df, stats):
            for col in df:
                plt.gcf().clear()
                plt.table(cellText=[v for v in stats[col][s] for s in stats[col]], rowLabels=[s for s in stats[col]],
                          colLabels=['Statistic', 'Value'])

        def format_plot(col, colMin, colMax, plotStyle, plot):
            """
            col: string
            colMin: number
            colMax: number
            plotStyle: string
            plot0: Matplotlib plt
            """
            plot.title(col + " " + plotStyle, fontsize="14", fontweight="bold")
            plt.xlim(colMin - 1, colMax + 1)
            for s in ["top", "bottom", "left", "right"]:
                ax.spines[s].set_color("0.5")
            ax.get_xaxis().set_tick_params(direction="out")
            ax.get_yaxis().set_tick_params(direction="out")
            ax.xaxis.tick_bottom()
            ax.yaxis.tick_left()

            if plotStyle == "Bar Plot":
                plt.ylabel("Frequency")

            elif plotStyle == "Histogram":
                plt.ylabel("Frequency")

            elif plotStyle == "Scatter Plot":
                pass

            elif plotStyle == "Statistics Table":
                pass

            elif plotStyle == "Violin Plot":
                plt.ylabel("Frequency")


class AnimatedPlot2D:
    @decorators.update_list_widget_decorator('mainMenu', *status.initAnimated2DPlotError)
    def __init__(self, app, name, xmin=None, y='None', ymin=-100, ymax=100, transform='None'):
        self.app = app
        self._name = name
        self.transform = transform
        self.y = y
        self.ymax = ymax
        self.ymin = ymin
        self.xmax = self.app.dataConfig.buffer
        if xmin:
            self.xmin = xmin
        else:
            self.xmin = self.xmax - 10
        self.color = (125 / 255, 175 / 255, 225 / 255)
        self.lw = 1
        self.zorder = 3
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.plot, = self.ax.plot([], [], color=self.color, lw=self.lw, zorder=self.zorder, )
        self.anim = animation.FuncAnimation(self.fig, self.update_plot, frames=self.app.plotConfig.frames,
                                            interval=self.app.plotConfig.fps, fargs=[], )
        self.format_plot()

    @decorators.update_list_widget_decorator('plotMenu', *status.plotUpdateError)
    def update_plot(self, i):
        if self.app.transceiverConfig._transceiverActive or self.app.generatorConfig.simulationEnabled:
            if not (self.app.plotMenu.isHidden() or self.app.plotMenu.isMinimized()):
                xdata = range(len(self.app.dataDude.data))
                ydata = self.app.dataDude.get(self.y).astype(float)
                if self.transform != 'None':
                    xdata, ydata = self.app.transformer.transforms[self.transform](xdata, ydata)
                self.plot.set_xdata(x=xdata)
                self.plot.set_ydata(y=ydata)

    def format_plot(self, *args, **kwargs):
        for spine in self.ax.spines:
            self.ax.spines[spine].set_color('w')
        self.ax.tick_params(colors='w')
        self.fig.patch.set_facecolor(self.app.palette.plotBackground)
        self.ax.set_facecolor('black')
        self.ax.grid(color=self.app.palette.plotBackground)
        self.ax.set_xlabel('Samples')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')

    @property
    def dna(self, *args, **kwargs):
        return {'name': self._name, 'transform': self.transform, 'xmin': self.xmin, 'ymin': self.ymin,
                'ymax': self.ymax, 'y': self.y, }


class AnimatedPlot2D:
    def __init__(self, app, name, xmin=None, y='None', ymin=-100, ymax=100, transform=None):
        try:
            self.app = app
            self._name = name
            self.transform = transform
            self.y = y
            self.ymax = ymax
            self.ymin = ymin
            self.xmax = self.app.dataConfig.buffer
            if xmin:
                self.xmin = xmin
            else:
                self.xmin = self.xmax - 10
            self.color = (125 / 255, 175 / 255, 225 / 255)
            self.lw = 1
            self.zorder = 3
            self.fig = Figure()
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111)
            self.plot, = self.ax.plot([], [], color=self.color, lw=self.lw, zorder=self.zorder, )
            self.anim = animation.FuncAnimation(self.fig, self.update_plot, frames=self.app.plotConfig.frames,
                                                interval=self.app.plotConfig.fps, fargs=[], )
            self.format_plot()
        except Exception as e:
            print('animated plot0 2d init error: {}'.format(e))

    def update_plot(self, i):
        if self.app.transceiverConfig._transceiverActive:
            if not (self.app.plotMenu.isHidden() or self.app.plotMenu.isMinimized()):
                try:
                    xdata = range(len(self.app.dataDude.data))
                    ydata = self.app.dataDude.get(self.y).astype(float)
                    if self.transform != 'None':
                        xdata, ydata = self.app.transformer.transforms[self.transform](xdata, ydata)
                    self.plot.set_xdata(x=xdata)
                    self.plot.set_ydata(y=ydata)
                except Exception as e:
                    statusMsg = 'Plot update error: {}'.format(str(e))
                    self.app.plotMenu.update_status(statusMsg, 'error')


def f(x, y):
    return np.sin(x) + np.cos(y)


def updatefig(*args):
    global x, y
    x += np.pi / 15.
    y += np.pi / 20.
    im.set_array(f(x, y))
    return im,


def plot_bar_stats(v, stats, ax):
    # props = dict(boxstyle='round', alpha=0.5, color=sns.color_palette()[0])
    mode = stats['Mode']
    cellText = [[str(mode)]]
    rowLabels = ['$Mode$']
    table = plt.table(cellText=cellText, rowLabels=rowLabels,
                      bbox=[1.25, 0.65, 0.05, 0.35])  # [1.025, 0.6475, 0.75, 0.75])

    for key, cell in table.get_celld().items():
        cell.set_linewidth(1)

    table.auto_set_column_width(0)
    table.scale(2, 3)


def plot_bars(v, stats=None):
    plt.gcf().clear()
    m = v.value_counts()
    barPlot = sns.barplot(x=m.index, y=m.values)
    if stats:
        plot_bar_stats(v, stats, barPlot)
    format_plot(v.label, v, 'Bar Plot', plt, barPlot)
    export_plot(barPlot, BarPlotsDir, v.label + BarPlotSuffix)


def plot_scatter(v, fit=True):
    """
    df: pandas Series (vector)
    fit: bool
    return: void
    """
    # sns.set_palette("pastel")
    # bins = np.arange(stats0[ col ][ 'min' ]-0.5,stats0[ col ][ 'max' ] + 1.5, step=dist_min(df[col]))
    plt.gcf().clear()
    scatterPlot = sns.regplot(np.arange(len(v)), v, order=2, fit_reg=False, label=v.label)
    # scatterPlot(color='#b9cfe7', markersize=5, markeredgewidth=1, markeredgecolor='#A54218', markerfacecolor='#4AA0C0')
    if fit: plot_regression_and_ci(v, scatterPlot)
    format_plot(v.label, v, 'Scatter Plot', plt, scatterPlot)
    export_plot(scatterPlot, ScatterPlotsDir, v.label + ScatterPlotSuffix)


def plot_topN_distributions(v, N):
    def plot_best_fit_function_string(ax, regParams, names, bw):
        props = dict(boxstyle='round', alpha=0.5, color=sns.color_palette()[0])
        f, n = ExpressionGenerator.distribution_string(names, regParams, y='PDF', name=True,
                                                       latex=False)  # get_function_string(fam, regParams, col)
        textstr = '{0}:'.format(n[names[0]]) + '\n' \
                                               '${0}$'.format(f[names[0]]) + '\n' \
                                                                             '{0}:'.format(n[names[1]]) + '\n' \
                                                                                                          '${0}$'.format(
            f[names[1]]) + '\n' \
                           'KDE Bandwidth:' + '\n' \
                                              '${0}$'.format(bw)
        ax.text(1.025, 0.6475, textstr, transform=ax.transAxes, fontsize=14, bbox=props)

    plt.gcf().clear()
    ax = sns.distplot(v, kde=False, norm_hist=True)
    topNames, topParams = best_fit_distribution(v, ax)
    topDists = {i: getattr(st, topNames[i]) for i in range(N)}
    pdfs = get_pdfs(topDists, topParams, N)

    for i in pdfs:
        paramNames = (topDists[i].shapes + ', loc, scale').split(', ') if topDists[i].shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(paramNames, topParams)])
        ax = pdfs[i].plot(lw=2, label=topNames[i] + '\n' + param_str, legend=True)

    density, bandwidth = density_kde(v)
    density.plot(lw=2, label="Gaussian Kernel Density Estimate", legend=True)
    plot_best_fit_function_string(ax, topParams, topNames, bandwidth)
    format_plot(v.label, v, 'Distribution Fit', plt, ax)
    export_plot(ax, DistributionFitPlotsDir, v.label + DistributionFitPlotSuffix)


# LOOK INTO SNS.COUNTPLOT
def compare_all_stats(stats):
    """
    stats0: pandas DataFrame
    """
    for stat in STATS:
        plot_stats_comparison(stats, stat)


def generate_report_plots(df, dtypes):
    categoricals = {}
    numericals = {}
    variables = []
    dfCat = []  # this will be the version of df with only the categorical
    dfNum = []  # this will be the version of df with only the numerical
    for col, label in zip(df, dtypes):
        if label == 'Categorical':
            categoricals[col] = collect_stats(df[col], label)
            dfCat.append(col)
            plot_bars(df[col],
                      categoricals[col])  # plot_scatter(df[col], fit=False)  # plot_stats(df[col], stats0=['Mode'])

        elif label == 'Numerical':
            numericals[col] = collect_stats(df[col], label)
            dfNum.append(col)
            plot_histogram(df[col])
            plot_scatter(df[col])
            plot_box(df[col], numericals[col])
            plot_violin(df[col], numericals[col])
        variables.append(col)
    if 'Numerical' in dtypes:
        stats_corr_cov(df[dfNum], numericals)
    return variables


def stats_corr_cov(df, stats):
    # stats0, corr, cov = collect_stats(df)
    # make_all_plots(df, stats0, corr, cov)
    corr, cov = get_corr_cov(df)
    plot_corrcov(corr, cov)
    compare_all_stats(stats)


def add_plots(figure, x, y, type='line', lineWidth=None, colors=(None,)):
    """
    figure: Bokeh figure
    x: list
    y: list of list
    return: list of Bokeh lines
    """
    if lineWidth == None:
        lineWidth = 0.1
    if type == 'line':
        f = figure.line
    elif type == 'scatter':
        f = figure.scatter
    elif type == 'bar':
        f = figure.vbars
    if colors != (None,) and colors != [None]:
        colorSet = colors
    else:
        colorSet = Colors.COLOR_SET
    lines = []
    for j, column in enumerate(y):
        # color = '#' + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])
        color = colorSet[j % len(colorSet)]
        if type == 'bar':
            try:
                lines.append(f(x, top=y.loc[:, column], width=0.1, fill_color=color))
            except Exception as e:
                print(e)
        else:
            lines.append(f(x, y.loc[:, column], line_width=lineWidth, line_color=color))
    return lines


def make_hover(xLabel, yLabel):
    """
    xLabel: str
    yLabel: str
    return: Bokeh HoverTool
    """
    hover = HoverTool(tooltips=[(xLabel, '$x'), (yLabel, '$y')], mode='vline')
    return hover


def make_figure(title, tools, xLabel, yLabel, xKeys, yKeys, xRange=None, yRange=None, bound=False):
    p = figure(title=title, tools=tools, toolbar_location='above', x_axis_label=xLabel, y_axis_label=yLabel, logo=None,
               sizing_mode='stretch_both', x_range=Range1d(xRange[0], xRange[1], bounds='auto') if bound else xRange,
               y_range=Range1d(yRange[0], yRange[1], bounds='auto') if bound else yRange)
    set_style(p)

    if xKeys != None:
        p.xaxis.bounds = (min(xKeys, key=xKeys.get), max(xKeys, key=xKeys.get))
        p.xaxis.ticker = [key for key in xKeys]
        lozoya.data_api.html_api.html_apicrawl.formatter = FuncTickFormatter(code="""
            var labels = %s;
            return labels[tick];""" % xKeys)
        p.xaxis.major_label_orientation = -pi / 3

    if yKeys != None:
        p.yaxis.bounds = (min(yKeys, key=yKeys.get), max(yKeys, key=yKeys.get))
        p.yaxis.ticker = [key for key in yKeys]
        lozoya.data_api.html_api.html_apicrawl.formatter = FuncTickFormatter(code="""
            var labels = %s;
            return labels[tick];""" % yKeys)

    return p


def make_contour_plot(file, X, Y, boundaries, titles, xLabels, yLabels, names=None, lineWidths=None, legends=None,
                      types=None, xKeys=(None,), yKeys=(None,), colors=None):
    temporaryFile = os.path.join(PLOTS_DIR, 'temp.html')
    output_file(temporaryFile)
    if legends == None:
        legends = [(True,) for i in range(len(Y))]
    if types == None:
        types = [('line',) for i in range(len(Y))]
    if names == None:
        names = determine_names(Y, legends)
    if colors == None:
        colors = [[[None] for j in range(len(Y[i]))] for i in range(len(Y))]

    plotList = []
    xRange, yRange, data = boundaries[0], boundaries[1], boundaries[2]

    for i in range(len(Y)):
        linesList = []
        for j in range(len(Y[i])):
            if j == 0:
                p = make_figure(title=titles[i], tools=make_tools(xLabels[i], yLabels[i]), xLabel=xLabels[i],
                                yLabel=yLabels[i], xKeys=xKeys[i], yKeys=yKeys[i], xRange=xRange, yRange=yRange,
                                bound=True)
                make_image(p, xRange, yRange, data, Colors.COLOR_SET[:len(colors[i])])
            lines = add_plots(p, x=X[i][j].astype(float), y=Y[i][j].astype(float), type=types[i][j],
                              lineWidth=lineWidths[i][j], colors=colors[i][j])
            if legends[i][j]:
                linesList.append(lines)
        set_legend(p, linesList, names[i])
        plotList.append(p)
    lay = row(plotList, sizing_mode='stretch_both')
    save(lay)
    _fix_html(temporaryFile, file)


def make_tabbed_plot(file, X, Y, titles, xLabels, yLabels, names=None, lineWidths=None, legends=None, types=None,
                     xKeys=(None,), yKeys=(None,)):
    """
    X: list of lists
    Y: list of lists
    names: list lists of str to specify the label to use for each line in the legend
    titles: list of str labels to use as title for each figure
    xLabels: list of str labels to use as x axis for each figure
    yLabels: list of str labels to use as y axis for each figure
    type: list lists of str 'line', 'histogram', 'scatter' to specify the type of line
    lineWidths: list of lists of int to specify the width of each line
    legends: list of lists of booleans to determine whether to represent lines in the legend
    return:
    """
    output_file(file)
    if legends == None:
        legends = [(True,) for i in range(len(Y))]
    if types == None:
        types = [('line',) for i in range(len(Y))]
    if names == None:
        names = []
        for i, y in enumerate(Y):
            for j, dF in enumerate(y):
                if legends[i][j]:
                    if j > 0:
                        names[i].append([str(name) for name in dF.columns])
                    else:
                        names.append([[str(name) for name in dF.columns]])
    tabsList = []
    for i in range(len(Y)):
        linesList = []
        for j in range(len(Y[i])):
            if j == 0:
                p = make_figure(title=titles[i], tools=make_tools(xLabels[i], yLabels[i]), xLabel=xLabels[i],
                                yLabel=yLabels[i], xKeys=xKeys[i], yKeys=yKeys[i])

            lines = add_plots(p, X[i][j].astype(float), Y[i][j].astype(float), type=types[i][j],
                              lineWidth=lineWidths[i][j])
            if legends[i][j]:
                linesList.append(lines)
        set_legend(p, linesList, names[i])
        lay = layout([[p]], sizing_mode='fixed')
        tabsList.append(Panel(child=lay, title=titles[i]))

    tabs = Tabs(tabs=tabsList)
    save(tabs)


def make_table(path, dataFrameList, titles):
    """
    Creates a bokeh html file containing a table containing the data in dataFrame
    path: str
    dataFrameList: list of pandas DataFrame
    titles: list of str
    return: None
    """
    output_file(path)
    tableList = []
    for dataFrame, title in zip(dataFrameList, titles):
        try:
            dataFrame = pd.DataFrame(dataFrame.values, index=[str(i) for i in dataFrame.index],
                                     columns=[str(col) for col in dataFrame.columns])
            source = ColumnDataSource(dataFrame.astype(str))
            names = [str(col) for col in dataFrame.columns]
            columns = [TableColumn(field=names[i], title=names[i]) for i in range(len(names))]
            t = DataTable(source=source, columns=columns, fit_columns=True, reorderable=True)
        except Exception as e:
            print(e)  # tableList.append(t)  # lay = Panel(child=t, title=title)
    # lay = layout(t, sizing_mode='stretch_both')
    save(t)


def make_tabbed_table(path, dataFrameList, titles):
    """
    Creates a bokeh html file containing a table containing the data in dataFrame
    path: str
    dataFrameList: list of pandas DataFrame
    titles: list of str
    return: None
    """
    output_file(path)
    tabsList = []
    for dataFrame, title in zip(dataFrameList, titles):
        source = ColumnDataSource(dataFrame)
        names = dataFrame.columns
        columns = [TableColumn(field=names[i], title=names[i]) for i in range(len(names))]
        data_table = DataTable(source=source, columns=columns, sizing_mode='stretch_both')
        data_table.reorderable = True
        data_table.fit_columns = True
        tabsList.append(Panel(child=data_table, title=title))
    tabs = Tabs(tabs=tabsList)
    save(tabs)


def make_tools(xLabels, yLabels):
    hover = make_hover(xLabels, yLabels)
    tools = [  # 'box_select',
        'lasso_select',  # 'box_zoom',
        'wheel_zoom',  # 'crosshair',
        hover, 'pan',  # 'poly_select',
        'undo', 'redo', 'reset', 'save',  # 'tap',
        # 'zoom_in',
        # 'zoom_out'
    ]
    return tools


def set_legend(figure, lines, names):
    """
    figure: Bokeh figure
    lines: list of lists of Bokeh lines
    names: list of lists of str
    return: None
    """
    items = make_legend_items(lines, names)
    legend = Legend(items=items)
    legend.background_fill_color = BACKGROUND_COLOR_LIGHTERER
    legend.border_line_alpha = 0.5
    legend.border_line_color = BACKGROUND_COLOR_LIGHTER
    legend.border_line_width = 5
    # legend.label_text_font = ''
    # legend.label_text_font_style = ''
    legend.label_text_color = CONTRASTED_COLOR
    legend.margin = 10
    legend.padding = 10
    legend.spacing = 10
    figure.add_layout(legend, 'right')
    figure.legend.click_policy = 'hide'


def set_style(figure):
    figure.axis.axis_label_standoff = 20
    figure.axis.axis_label_text_color = CONTRASTED_COLOR
    figure.axis.axis_line_width = 1
    figure.axis.axis_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_in = 5
    figure.axis.major_tick_out = 10
    figure.axis.minor_tick_line_color = CONTRASTED_COLOR
    figure.axis.minor_tick_in = -2
    figure.axis.minor_tick_out = 7
    figure.axis.major_label_text_color = CONTRASTED_COLOR

    figure.background_fill_color = GRAPH_BACKGROUND_COLOR
    # figure.background_fill_alpha = 0.5
    figure.border_fill_color = BACKGROUND_COLOR_LIGHT

    figure.grid.grid_line_alpha = 0.5
    figure.grid.grid_line_dash = [2, 4]
    figure.grid.grid_line_color = CONTRASTED_COLOR

    figure.min_border_bottom = 100
    figure.min_border_left = 50
    figure.min_border_right = 50
    figure.min_border_top = 100

    # figure.outline_line_width = 5
    # figure.outline_line_alpha = 0.5
    figure.outline_line_color = CONTRASTED_COLOR

    figure.title.text_color = CONTRASTED_COLOR  # figure.title.text_font = ''  # figure.title.text_font_style = ''


def _fix_html(temporaryPath, path):
    q = ['', '']
    with open(path, 'w') as fw:
        fw.truncate()
        with open(temporaryPath, 'r') as f:
            for line in f:
                q[0] = q[1]
                q[1] = line
                if q[1].strip() == 'width: 90%;' and q[0].strip() == 'body {':
                    fw.write(re.sub('90', '100', q[1]))
                else:
                    fw.write(line)


def add_plots(figure, x, y, type='line', lineWidth=None, colors=(None,)):
    """
    figure: Bokeh figure
    x: list
    y: list of list
    return: list of Bokeh lines
    """
    if lineWidth == None:
        lineWidth = 0.1
    if type == 'line':
        f = figure.line
    elif type == 'scatter':
        f = figure.scatter
    elif type == 'bar':
        f = figure.vbars
    if colors != (None,) and colors != [None]:
        colorSet = colors
    else:
        colorSet = COLOR_SET
    lines = []
    for j, column in enumerate(y):
        # color = '#' + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])
        color = colorSet[j % len(colorSet)]
        if type == 'bar':
            try:
                lines.append(f(x, top=y.loc[:, column], width=0.1, fill_color=color))
            except Exception as e:
                print(e)
        else:
            lines.append(f(x, y.loc[:, column], line_width=lineWidth, line_color=color))
    return lines


def set_style(figure):
    figure.axis.axis_label_standoff = 20
    figure.axis.axis_label_text_color = CONTRASTED_COLOR
    figure.axis.axis_line_width = 1
    figure.axis.axis_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_in = 5
    figure.axis.major_tick_out = 10
    figure.axis.minor_tick_line_color = CONTRASTED_COLOR
    figure.axis.minor_tick_in = -2
    figure.axis.minor_tick_out = 7
    figure.axis.major_label_text_color = CONTRASTED_COLOR

    figure.background_fill_color = GRAPH_BACKGROUND_COLOR
    # figure.background_fill_alpha = 0.5
    figure.border_fill_color = BACKGROUND_COLOR_LIGHT

    figure.grid.grid_line_alpha = 0.5
    figure.grid.grid_line_dash = [2, 4]
    figure.grid.grid_line_color = CONTRASTED_COLOR

    figure.min_border_bottom = 100
    figure.min_border_left = 50
    figure.min_border_right = 50
    figure.min_border_top = 100

    # figure.outline_line_width = 5
    # figure.outline_line_alpha = 0.5
    figure.outline_line_color = CONTRASTED_COLOR

    figure.title.text_color = CONTRASTED_COLOR  # figure.title.text_font = ''  # figure.title.text_font_style = ''


def add_plots(figure, x, y, colorSet, type='line', lineWidth=None, colors=(None,)):
    """
    figure: Bokeh figure
    x: list
    y: list of list
    return: list of Bokeh lines
    """
    if lineWidth == None:
        lineWidth = 0.1
    if type == 'line':
        f = figure.line
    elif type == 'scatter':
        f = figure.scatter
    elif type == 'bar':
        f = figure.vbars
    if colors != (None,) and colors != [None]:
        colorSet = colors
    else:
        colorSet = colorSet
    lines = []
    for j, column in enumerate(y):
        # color = '#' + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])
        color = colorSet[j % len(colorSet)]
        if type == 'bar':
            try:
                lines.append(f(x, top=y.loc[:, column], width=0.1, fill_color=color))
            except Exception as e:
                print(e)
        else:
            lines.append(f(x, y.loc[:, column], line_width=lineWidth, line_color=color))
    return lines


def make_contour_plot(file, X, Y, boundaries, titles, xLabels, yLabels, path, colorSet, names=None, lineWidths=None,
                      legends=None, types=None, xKeys=(None,), yKeys=(None,), colors=None):
    temporaryFile = os.path.join(path, 'temp.html')
    output_file(temporaryFile)
    if legends == None:
        legends = [(True,) for i in range(len(Y))]
    if types == None:
        types = [('line',) for i in range(len(Y))]
    if names == None:
        names = determine_names(Y, legends)
    if colors == None:
        colors = [[[None] for j in range(len(Y[i]))] for i in range(len(Y))]

    plotList = []
    xRange, yRange, data = boundaries[0], boundaries[1], boundaries[2]

    for i in range(len(Y)):
        linesList = []
        for j in range(len(Y[i])):
            if j == 0:
                p = make_figure(title=titles[i], tools=make_tools(xLabels[i], yLabels[i]), xLabel=xLabels[i],
                                yLabel=yLabels[i], xKeys=xKeys[i], yKeys=yKeys[i], xRange=xRange, yRange=yRange,
                                bound=True)
                make_image(p, xRange, yRange, data, colorSet[:len(colors[i])])
            lines = add_plots(p, x=X[i][j].astype(float), y=Y[i][j].astype(float), type=types[i][j],
                              lineWidth=lineWidths[i][j], colors=colors[i][j])
            if legends[i][j]:
                linesList.append(lines)
        set_legend(p, linesList, names[i])
        plotList.append(p)
    lay = row(plotList, sizing_mode='stretch_both')
    save(lay)
    _fix_html(temporaryFile, file)


def make_plot(file, X, Y, titles, xLabels, yLabels, path, names=None, lineWidths=None, legends=None, types=None,
              xKeys=(None,), yKeys=(None,), colors=None):
    """
    X: list of lists
    Y: list of lists
    names: list lists of str to specify the label to use for each line in the legend
    titles: list of str labels to use as title for each figure
    xLabels: list of str labels to use as x axis for each figure
    yLabels: list of str labels to use as y axis for each figure
    type: list lists of str 'line', 'histogram', 'scatter' to specify the type of line
    lineWidths: list of lists of int to specify the width of each line
    legends: list of lists of booleans to determine whether to represent lines in the legend
    colors: list of lists of lists
    return:
    """
    temporaryFile = os.path.join(path, 'temp.html')
    output_file(temporaryFile)
    if legends == None:
        legends = [(True,) for i in range(len(Y))]
    if types == None:
        types = [('line',) for i in range(len(Y))]
    if names == None:
        names = determine_names(Y, legends)
    if colors == None:
        colors = [[[None] for j in range(len(Y[i]))] for i in range(len(Y))]

    plotList = []
    for i in range(len(Y)):
        linesList = []
        for j in range(len(Y[i])):
            if j == 0:
                p = make_figure(title=titles[i], tools=make_tools(xLabels[i], yLabels[i]), xLabel=xLabels[i],
                                yLabel=yLabels[i], xKeys=xKeys[i], yKeys=yKeys[i])
            lines = add_plots(p, x=X[i][j].astype(float), y=Y[i][j].astype(float), type=types[i][j],
                              lineWidth=lineWidths[i][j], colors=colors[i][j])
            if legends[i][j]:
                linesList.append(lines)
        set_legend(p, linesList, names[i])
        plotList.append(p)
    lay = row(plotList, sizing_mode='stretch_both')
    save(lay)
    _fix_html(temporaryFile, file)


def set_legend(figure, lines, names, backgroundColorLighterer, backgroundColorLight, contrastedColor):
    """
    figure: Bokeh figure
    lines: list of lists of Bokeh lines
    names: list of lists of str
    return: None
    """
    items = make_legend_items(lines, names)
    legend = Legend(items=items)
    legend.background_fill_color = backgroundColorLighterer
    legend.border_line_alpha = 0.5
    legend.border_line_color = backgroundColorLight
    legend.border_line_width = 5
    # legend.label_text_font = ''
    # legend.label_text_font_style = ''
    legend.label_text_color = contrastedColor
    legend.margin = 10
    legend.padding = 10
    legend.spacing = 10
    figure.add_layout(legend, 'right')
    figure.legend.click_policy = 'hide'


def set_style(figure, contrastedColor, graphBackgroundColor, backgroundColorLight):
    figure.axis.axis_label_standoff = 20
    figure.axis.axis_label_text_color = contrastedColor
    figure.axis.axis_line_width = 1
    figure.axis.axis_line_color = contrastedColor
    figure.axis.major_tick_line_color = contrastedColor
    figure.axis.major_tick_in = 5
    figure.axis.major_tick_out = 10
    figure.axis.minor_tick_line_color = contrastedColor
    figure.axis.minor_tick_in = -2
    figure.axis.minor_tick_out = 7
    figure.axis.major_label_text_color = contrastedColor
    figure.background_fill_color = graphBackgroundColor
    # figure.background_fill_alpha = 0.5
    figure.border_fill_color = backgroundColorLight
    figure.grid.grid_line_alpha = 0.5
    figure.grid.grid_line_dash = [2, 4]
    figure.grid.grid_line_color = contrastedColor
    figure.min_border_bottom = 100
    figure.min_border_left = 50
    figure.min_border_right = 50
    figure.min_border_top = 100
    # figure.outline_line_width = 5
    # figure.outline_line_alpha = 0.5
    figure.outline_line_color = contrastedColor
    figure.title.text_color = contrastedColor  # figure.title.text_font = ''  # figure.title.text_font_style = ''


def s_round(arr, acc=4):
    """
    arr: numpy array
    return: numpy array
    """
    m = []
    try:
        for i in range(len(arr)):
            m.append(round_sigfigs(arr[i], acc))
        return np.array(m)
    except Exception as e:
        print(e)
        pass

    try:
        return round_sigfigs(arr, acc)
    except Exception as e:
        print(e)
        pass

    try:
        return sym.N(arr, acc)
    except Exception as e:
        print(e)
        pass


def format_plot(col, v, plotStyle, plot, ax):
    """
    col: string
    colMin: number
    colMax: number
    plotStyle: string
    plot0: Matplotlib plt
    """
    plot.title(col + ' ' + plotStyle, fontsize='18', fontweight='bold', fontname='Georgia')
    if plotStyle != 'Statistics Summary':
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1.)
        ax.spines['left'].set_linewidth(1.)

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_linewidth(1.)
        ax.spines['right'].set_linewidth(1.)

        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')
    # plt.xlim(v.min() - 1, v.max() + 1)
    if ax and plotStyle != 'Statistics Summary':
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_color('0.5')
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

    if plotStyle == 'Bar Plot':
        plot.ylabel('Frequency')

    elif plotStyle == 'Box Plot':
        pass

    elif plotStyle == 'Distribution Fit':
        plot.ylabel('Frequency')
        plot.xlim(np.min(v) - 1, np.max(v) + 1)

    elif plotStyle == 'Histogram':
        plot.ylabel('Frequency')

    elif plotStyle == 'Scatter Plot':
        plot.xlim(np.min(v.index.values) - 1,
                  np.max(v.index.values) + 1)  # plt.gca().set_aspect('equal', adjustable='box')


    elif plotStyle == 'Statistics Summary':
        pass

    elif plotStyle == 'Violin Plot':
        plt.ylabel('Frequency')


def plot_scatter(v, fit=True):
    """
    df: pandas Series (vector)
    fit: bool
    return: void
    """
    # sns.set_palette("pastel")
    # bins = np.arange(stats0[ col ][ 'min' ]-0.5,stats0[ col ][ 'max' ] + 1.5, step=dist_min(df[col]))
    plt.gcf().clear()
    scatterPlot = sns.regplot(np.arange(len(v)), v, order=2, fit_reg=False, label=v.label)
    # scatterPlot(color='#b9cfe7', markersize=5, markeredgewidth=1, markeredgecolor='#A54218', markerfacecolor='#4AA0C0')
    if fit: pass  # plot_regression_and_ci(v, scatterPlot)
    format_plot(v.label, v, 'Scatter Plot', plt, scatterPlot)
    export_plot(scatterPlot, ScatterPlotsDir, v.label + ScatterPlotSuffix)


def plot_violin(v, stats=None):
    plt.gcf().clear()
    violinPlot = sns.violinplot(x=v)
    if stats != None:
        plot_violin_stats(v, stats, violinPlot)
    format_plot(v.label, v, 'Violin Plot', plt, violinPlot)
    export_plot(violinPlot, ViolinPlotsDir, v.label + ViolinPlotSuffix)


def plot_histogram(plt, index, curve, color, lims, norm=False):
    plt.hist(curve, alpha=0.5, bins=np.arange(lims[0][1] + 1), normed=norm, align='left',
             facecolor=GRAPH_COLORS[color][index % len(GRAPH_COLORS[color])])
    """
    # Setting the font and style
    matplotlib.rcParams['font.serif'] = "Times New Roman"
    matplotlib.rcParams['font.family'] = "serif"
    self.font = {'fontname': 'Times New Roman'}
    plt.style.use(style)
    Bar_Chart_3D
    #TODO
    xpos = [1, 1, 2, 3, 5, 6, 3, 5, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    ypos = y_input_data
    zpos = np.zeros_like(xpos)
    # Construct arrays with the dimensions for the bars.
    dx, dy = 0.05 * np.ones_like(zpos)
    dz = z_input_data
    # Creating the plot0
    self.graph2.bar3d(xpos, ypos, zpos, dx, dy, dz, color='teal')
    generate_graph
    # Placing the axis labels and the title on the graph2
    plt.xlabel(self.x_label, **self.font)
    plt.ylabel(self.y_label, **self.font)
    generage_graph_3d
    # Placing the axis labels and the title on the graph2
    self.bar_chart_3d.set_xlabel(self.x_label)
    self.bar_chart_3d.set_ylabel(self.y_label)
    self.bar_chart_3d.set_zlabel(self.z_label)
    generate_subplot_3d
    # Creating the figure
    self.bar_chart_3d = plt.figure(figure).add_subplot(111, projection='3d')
    return self.bar_chart_3d
    """


def set_style(figure):
    figure.axis.axis_label_standoff = 20
    figure.axis.axis_label_text_color = CONTRASTED_COLOR
    figure.axis.axis_line_width = 1
    figure.axis.axis_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_line_color = CONTRASTED_COLOR
    figure.axis.major_tick_in = 5
    figure.axis.major_tick_out = 10
    figure.axis.minor_tick_line_color = CONTRASTED_COLOR
    figure.axis.minor_tick_in = -2
    figure.axis.minor_tick_out = 7
    figure.axis.major_label_text_color = CONTRASTED_COLOR

    figure.background_fill_color = GRAPH_BACKGROUND_COLOR
    # figure.background_fill_alpha = 0.5
    figure.border_fill_color = BACKGROUND_COLOR_LIGHT

    figure.grid.grid_line_alpha = 0.5
    figure.grid.grid_line_dash = [2, 4]
    figure.grid.grid_line_color = CONTRASTED_COLOR

    figure.min_border_bottom = 100
    figure.min_border_left = 50
    figure.min_border_right = 50
    figure.min_border_top = 100

    # figure.outline_line_width = 5
    # figure.outline_line_alpha = 0.5
    figure.outline_line_color = CONTRASTED_COLOR

    figure.title.text_color = CONTRASTED_COLOR  # figure.title.text_font = ''  # figure.title.text_font_style = ''


class Plotter:
    def __init__(self, style='bmh'):
        # Setting the font and style
        matplotlib.rcParams['font.serif'] = "Times New Roman"
        matplotlib.rcParams['font.family'] = "serif"
        self.font = {'fontname': 'Times New Roman'}
        plt.style.use(style)

    def Plot(self, figure, input_data, subplot, step_size=1, legend_labels_list=[], starting_year=1,
             x_label='Time (Years)', y_label='State', title='Monte Carlo Simulation2 of Item Value'):
        # Setting labels and style
        self.y_axis, self.step_size, self.starting_year, self.x_label, self.y_label, self.title = input_data, step_size, starting_year, x_label, y_label, title
        self.figure = figure
        self.size = 0
        self.line_styles = ['.--', '*--', 'x--', '^--']  # The plotter wraps around these line style

        # Creating the plot0
        plt.figure(self.figure)
        plt.subplot(subplot)

        # This for loop is plotting all the points in each list in the input_data list.
        for i in range(self.y_axis.__len__()):
            self.x_axis = []
            self.x_axis = np.arange(self.starting_year, self.starting_year + self.y_axis[0].__len__(), self.step_size)
            if legend_labels_list.__len__() != 0:
                plt.plot(self.x_axis, self.y_axis[i], self.line_styles[i % self.line_styles.__len__()],
                         label=legend_labels_list[i % legend_labels_list.__len__()], linewidth=0.5)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)  # Placing a legend beside the plot0
            else:
                plt.plot(self.x_axis, self.y_axis[i], self.line_styles[i % self.line_styles.__len__()], linewidth=0.5)

        self.generate_graph()

    def Histogram(self, figure, input_data, x_label='State', y_label='Frequency', title='Frequency Distribution'):
        # Setting labels
        self.y_axis, self.x_label, self.y_label, self.title = input_data, x_label, y_label, title
        self.figure = figure

        # Creating the figure
        plt.figure(self.figure)
        self.title = str('Frequency Distribution at Year ' + str(int(self.figure) - 1))

        # Creating the plot0
        plt.hist(self.y_axis, bins=20, facecolor='pink')
        self.generate_graph()

    def Bar_Chart_3D(self, graph, y_input_data, z_input_data, iterations, x_label='Time (Years)', y_label='State',
                     z_label='Frequency', title='Frequency Distribution'):
        # Setting labels
        self.y_axis, self.z_axis, self.x_label, self.y_label, self.z_label, self.title = y_input_data, z_input_data, x_label, y_label, z_label, title
        self.iterations, self.graph = iterations, graph
        self.title = 'Frequency Distribution'

        # TODO
        xpos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        ypos = y_input_data
        zpos = np.zeros_like(xpos)

        # Construct arrays with the dimensions for the bars.
        dx, dy = 0.05 * np.ones_like(zpos)
        dz = z_input_data

        # Creating the plot0
        self.graph.bar3d(xpos, ypos, zpos, dx, dy, dz, color='teal')

        self.generate_graph()

    def generate_graph(self):
        # Placing the axis labels and the title on the graph2
        plt.xlabel(self.x_label, **self.font)
        plt.ylabel(self.y_label, **self.font)
        plt.title(self.title, **self.font)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # This and the preceding line ensure a full screen display upon rendering

    def generage_graph_3d(self):
        # Placing the axis labels and the title on the graph2
        self.bar_chart_3d.set_xlabel(self.x_label)
        self.bar_chart_3d.set_ylabel(self.y_label)
        self.bar_chart_3d.set_zlabel(self.z_label)
        plt.title(self.title)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # This and the preceding line ensure a full screen display upon rendering

    def generate_subplot_3d(self, figure):
        # Creating the figure
        self.bar_chart_3d = plt.figure(figure).add_subplot(111, projection='3d')
        return self.bar_chart_3d

    def display(self):
        plt.grid(True)
        plt.show()


def decimalize(val, acc, lim=False):
    if not lim:
        m = '%.4e'
    else:
        m = '%.2e'
    return m % Decimal(round(val, acc)) if len(str(round(val, acc))) > acc else round(val, acc)


def dist_min(col):
    minDist = 1
    c = sorted(col)
    for i in range(len(c) - 1):
        dist = abs(c[i] - c[i + 1])
        if dist < minDist and dist != 0:
            minDist = dist
    return minDist * 10


# DEPRECATED
def fractionize(val):
    # calibration constants
    sigfigs = 5
    roundfigs = sigfigs - 1
    sigfigs2 = 3

    def get_exponent(val, returnVal=False):
        ss = ('{0}').format(val)
        if ('E' in ss):
            e = 'E'
        elif ('e' in ss):
            e = 'e'
        else:
            return ss
        mantissa, exp = ss.split(e)
        if returnVal == True:
            return float(exp), mantissa
        return float(exp)

    def de_exponentize(val):
        ss = ('{0}').format(val)
        if 'E' in ss or 'e' in ss:
            ex, mantissa = get_exponent(val, returnVal=True)
            if mantissa == ss:
                return val
            m = round(float(mantissa), abs(int(ex)) + 1)
            b = m * (10 ** ex)
            return b
        return val

    def cancel_scino(a, b):
        if 'e' in a and 'e' not in b:
            return a, b
        elif 'e' not in a and 'e' in b:
            n = b.split('e')
            b1 = n[0]
            a1 = a + 'e-' + n[1]
            return a1, b1
        elif 'e' in a and 'e' in b:
            m = a.split('e')
            n = b.split('e')
            r = int(m[1]) - int(n[1])
            a1 = m[0] + 'e+' + str(r)
            b1 = n[0]
            return a1, b1
        else:
            return a, b

    def check_exponent_signs(a, b):
        if 'e' in a:
            a0, a1 = a.split('e')
            if '-+' in a1 or '+-' in a1:
                a = a0 + 'e-' + a1[2:]

        if 'e' in b:
            b0, b1 = b.split('e')
            if '-+' in b1 or '+-' in b1:
                b = b0 + 'e-' + b1[2:]
        return a, b

    def check_redundant_denominator(a, b):
        b0 = b.rstrip('0')
        if b0 == '1':
            if 'e' in a:
                a0, a1 = a.split('e')
                asign, arest = a1[0], a1[1:]
                arest = int(arest) - (len(b) - len(b0))
                if arest < 0: asign = ''
                a = a0 + 'e' + asign + str(arest)
                b = ''
        return a, b

    def check_redundant_exponent(a, b):
        if 'e' in a:
            a0, a1 = a.split('e')
            asign = a1[0]
            arest = a1[1:]
            if arest == '0':
                a = a0
            elif arest == '1':
                if asign == '+':
                    a = str(round(float(a0) * 10, len(a0)))
                else:
                    a = str(round(float(a0) / 10, len(a0)))

        if 'e' in b:
            b0, b1 = b.split('e')
            bsign = b1[0]
            brest = b1[1:]
            if brest == '0':
                b = b0
            elif brest == '1':
                if bsign == '+':
                    b = str(round(float(b0) * 10, len(b0)))
                else:
                    b = str(round(float(b0) / 10, len(b0)))
        return a, b

    v0 = de_exponentize(val)

    v0 = sym.nsimplify(v0)
    ag = 0
    for char in str(v0):
        if char == '/' or char == '*' or char == '(':
            ag += 1
    if ag < 2:
        v = v0
    else:
        if str(val)[0:2] == '0.':
            art = str(val)[2:]
            ad = 1 + len(art) - len(art.lstrip('0'))
            art = round(val, ad)
        else:
            art = round(val, sigfigs)

        v = art

    if r'/' in str(v):
        a, b = str(v).split(r'/')
        if a[0] == '-':
            asign = '-'
            a = a[1:]
        else:
            asign = ''
        if b[0] == '_':
            b = b[1:]
            bsign = '-'
        else:
            bsign = ''
        # NUMERATOR
        acc = len(a)
        if acc > sigfigs:
            a = a[0] + '.' + a[1:roundfigs] + 'e+' + str(acc - 1)
        # DENOMINATOR
        # check trailing 0
        trail = len(b) - len(b.rstrip('0'))
        if trail > sigfigs2:
            b = b.rstrip('0') + '000' + 'e+' + str(trail - sigfigs2)
        else:
            acc = len(b)
            if acc > sigfigs:
                b = b[0] + '.' + b[1:roundfigs] + 'e+' + str(acc - 1)
        a, b = cancel_scino(a, b)
        a, b = check_exponent_signs(a, b)
        a, b = check_redundant_exponent(a, b)
        a, b = check_redundant_denominator(a, b)
        a, b = check_redundant_exponent(a, b)

        a = asign + a
        b = bsign + b

        """if 'e' in a:
            a = '(' + a + ')'
        if 'e' in b:
            b = '(' + b + ')'
            """

        if b != '':
            if 'e' in a:
                a, ex = a.split('e')
                return str(r'\frac{' + a + r'}{' + b + '}') + r'\times 10^{' + ex + '}'
            return asign + str(r'\frac{' + a + r'}{' + b + '}')
        else:
            if 'e' in a:
                a, ex = a.split('e')
                return str(a) + r'\times 10^{' + ex + '}'
            return a
    return v


def get_function_string(fam, regParams, col='x'):
    f = ''
    c = 1
    tol = 10
    r = 10
    if fam == 'Exponential':
        return fit_string_exponential(f, regParams)
    elif fam == 'Logarithmic':
        return fit_string_logarithmic(f, regParams)
    elif fam == 'Sinusoidal':
        return fit_string_sinusoidal(regParams)
    return fit_string_polynomial(c, f, r, regParams, tol)  # if fam == 'Polynomial' or fam == 'Linear':


def fit_string_polynomial(c, f, r, regParams, tol):
    for rP in regParams:
        if not (-1 / (10 ** tol) < rP < 1 / (10 ** tol)) or c == 1:
            rP = fractionize(abs(rP))
            rP0 = str(rP)
            if c != len(regParams):
                n = len(regParams) - 1 - (c - 1)
                if n > 1:
                    f += r'{0}'.format(
                        rP0 if rP0.rstrip('0').rstrip('.') != '1' else '') + r'\mathbf{' + 'x' + '}' + '^{0}'.format(n)
                else:
                    f += '{0}'.format(rP0 if rP0.rstrip('0').rstrip('.') != '1' else '') + r'\mathbf{' + 'x' + '}'
                if c == 1 and regParams[0] < 0:
                    f = '-' + f
            else:
                f += '{0}'.format(rP0, r)
                break
        if c < len(regParams):
            if regParams[c] > 1 / (10 ** tol):
                f += '+'
            elif regParams[c] < -1 / (10 ** tol):
                f += '-'
        c += 1
    return f


def fit_string_exponential(f, regParams):
    """if regParams[1] < 0:
            addi = '-' + fractionize(regParams[1])  # str(abs(round(regParams[1], 3)))
        elif regParams[1] > 0:
            addi = '+' + fractionize(regParams[1])  # str(abs(round(regParams[1], 3)))
        else:"""
    addi = ''
    A = fractionize(regParams[0])
    B = fractionize(regParams[1])
    f = r'{0}\times exp[{1}({3})]{2}'.format(A, B, addi, 'x')
    return f


def fit_string_sinusoidal(regParams):
    if regParams[2] < -1e-3:
        addi0 = '-' + str(fractionize(regParams[2]))  # str(abs(round(regParams[1], 3)))
    elif regParams[2] > 1e-3:
        addi0 = '+' + str(fractionize(regParams[2]))  # str(abs(round(regParams[1], 3)))
    else:
        addi0 = ''
    if regParams[3] < -1e-3:
        addi1 = '-' + str(fractionize(regParams[3]))  # str(abs(round(regParams[2], 3)))
    elif regParams[3] > 1e-3:
        addi1 = '+' + str(fractionize(regParams[3]))  # str(abs(round(regParams[2], 3)))
    else:
        addi1 = ''
    if round(regParams[0], 4) == 1:
        A = ''
    else:
        # A = str(round(regParams[0], 3)) + r'\times '
        A = str(fractionize(regParams[0])) + r'\times '
    B = str(fractionize(regParams[1]))
    f = A + r'sin[{0}'.format(B)
    # f += r'\mathbf{' + 'x' + '}' + r'{0}]{1}'.format(addi0, addi1)
    f += r'\mathbf{' + 'x' + '}' + r'{0}]{1}'.format(addi0, addi1)
    return f


def fit_string_logarithmic(f, regParams):
    B = str(fractionize(abs(regParams[1])))
    if float(regParams[1]) < 0:
        B = '-' + B
    elif float(regParams[1]) > 0:
        B = '+' + B
    else:
        B = ''
    A = str(fractionize(regParams[0]))
    if A == '1':
        A = ''
    else:
        A = A + r'\times'
    f = A + 'ln[\mathbf{' + 'x' + r'}]' + B
    return f


def s_round(arr, acc=6):
    """
    arr: numpy array
    return: numpy array
    """
    m = []
    try:
        for i in range(len(arr)):
            m.append(float(sym.N(arr[i], acc)))
        return np.array(m)
    except Exception as e:
        print(e)
        return sym.N(arr, acc)


def tryer(args, i):
    """
    args: list or dictionary to interface7
    i: int, position to interface7 in args
    return: value in args[i] if it exists, else None
    """
    try:
        return args[i]
    except:
        return None


# DEPRECATED
def fractionize(val):
    # calibration constants
    sigfigs = 5
    roundfigs = sigfigs - 1
    sigfigs2 = 3
    v0 = de_exponentize(val)
    v0 = sym.nsimplify(v0)
    ag = 0
    for char in str(v0):
        if char == '/' or char == '*' or char == '(':
            ag += 1
    if ag < 2:
        v = v0
    else:
        if str(val)[0:2] == '0.':
            art = str(val)[2:]
            ad = 1 + len(art) - len(art.lstrip('0'))
            art = round(val, ad)
        else:
            art = round(val, sigfigs)

        v = art

    if r'/' in str(v):
        a, b = str(v).split(r'/')
        if a[0] == '-':
            asign = '-'
            a = a[1:]
        else:
            asign = ''
        if b[0] == '_':
            b = b[1:]
            bsign = '-'
        else:
            bsign = ''
        # NUMERATOR
        acc = len(a)
        if acc > sigfigs:
            a = a[0] + '.' + a[1:roundfigs] + 'e+' + str(acc - 1)
        # DENOMINATOR
        # check trailing 0
        trail = len(b) - len(b.rstrip('0'))
        if trail > sigfigs2:
            b = b.rstrip('0') + '000' + 'e+' + str(trail - sigfigs2)
        else:
            acc = len(b)
            if acc > sigfigs:
                b = b[0] + '.' + b[1:roundfigs] + 'e+' + str(acc - 1)
        a, b = cancel_scino(a, b)
        a, b = check_exponent_signs(a, b)
        a, b = check_redundant_exponent(a, b)
        a, b = check_redundant_denominator(a, b)
        a, b = check_redundant_exponent(a, b)

        a = asign + a
        b = bsign + b

        """if 'e' in a:
            a = '(' + a + ')'
        if 'e' in b:
            b = '(' + b + ')'
            """

        if b != '':
            if 'e' in a:
                a, ex = a.split('e')
                return str(r'\frac{' + a + r'}{' + b + '}') + r'\times 10^{' + ex + '}'
            return asign + str(r'\frac{' + a + r'}{' + b + '}')
        else:
            if 'e' in a:
                a, ex = a.split('e')
                return str(a) + r'\times 10^{' + ex + '}'
            return a
    return v


def headerize(headerString, headerLevel):
    return str("<h{0}>{1}</h{0}>").format(headerLevel, headerString)


def get_function_string(fam, regParams, col='x'):
    f = ''
    c = 1
    tol = 10
    r = 10
    if fam == 'Polynomial' or fam == 'Linear':
        for rP in regParams:
            if not (-1 / (10 ** tol) < rP < 1 / (10 ** tol)) or c == 1:
                rP = fractionize(abs(rP))
                rP0 = str(rP)
                if c != len(regParams):
                    n = len(regParams) - 1 - (c - 1)
                    if n > 1:
                        f += r'{0}'.format(rP0 if rP0.rstrip('0').rstrip(
                            '.') != '1' else '') + r'\mathbf{' + 'x' + '}' + '^{0}'.format(n)
                    else:
                        f += '{0}'.format(rP0 if rP0.rstrip('0').rstrip('.') != '1' else '') + r'\mathbf{' + 'x' + '}'
                    if c == 1 and regParams[0] < 0:
                        f = '-' + f
                else:
                    f += '{0}'.format(rP0, r)
                    break
            if c < len(regParams):
                if regParams[c] > 1 / (10 ** tol):
                    f += '+'
                elif regParams[c] < -1 / (10 ** tol):
                    f += '-'
            c += 1

    elif fam == 'Exponential':
        f = fit_string_exponential(f, regParams)
    elif fam == 'Logarithmic':
        f = fit_string_logarithmic(f, regParams)
    elif fam == 'Sinusoidal':
        f = fit_string_sinusoidal(f, regParams)
    return f


def sub_space(num):
    if num == 0:
        return ""
    s = "</br>"
    for i in range(num):
        s += "</br>"
    return s


def make_symbol(args, i, symSim=False):
    """
    args: list or dictionary to interface
    i: int, position to interface in args
    return: value in args[i] if it exists, else None
    """
    try:
        if symSim:
            return sym.N(args[i])
        else:
            if args[i] - int(args[i]) == 0:
                return int(args[i])
            return args[i]
    except:
        return None


# DECORATOR
def log(func):
    @wraps(func)
    def make_log(*args, **kwargs):
        print("Calling function: " + str(func.__name__))
        print("With arguments: " + str(args))
        print("With keywords: " + str(kwargs))
        return func(*args, **kwargs)

    return make_log


def timer(func):
    @wraps(func)
    def time_func(*args, **kwargs):
        startTime = time.time()
        vals = func(*args, **kwargs)
        endTime = time.time() - startTime
        print('Calling function {0} took {1}'.format(func.__name__, endTime))
        return vals

    return time_func


# DEPRECATED
def fractionize(val):
    # calibration constants
    sigfigs = 5
    roundfigs = sigfigs - 1
    sigfigs2 = 3
    v0 = de_exponentize(val)
    v0 = sym.nsimplify(v0)
    ag = 0
    for char in str(v0):
        if char == '/' or char == '*' or char == '(':
            ag += 1
    if ag < 2:
        v = v0
    else:
        if str(val)[0:2] == '0.':
            art = str(val)[2:]
            ad = 1 + len(art) - len(art.lstrip('0'))
            art = round(val, ad)
        else:
            art = round(val, sigfigs)

        v = art

    if r'/' in str(v):
        a, b = str(v).split(r'/')
        if a[0] == '-':
            asign = '-'
            a = a[1:]
        else:
            asign = ''
        if b[0] == '_':
            b = b[1:]
            bsign = '-'
        else:
            bsign = ''
        # NUMERATOR
        acc = len(a)
        if acc > sigfigs:
            a = a[0] + '.' + a[1:roundfigs] + 'e+' + str(acc - 1)
        # DENOMINATOR
        # check trailing 0
        trail = len(b) - len(b.rstrip('0'))
        if trail > sigfigs2:
            b = b.rstrip('0') + '000' + 'e+' + str(trail - sigfigs2)
        else:
            acc = len(b)
            if acc > sigfigs:
                b = b[0] + '.' + b[1:roundfigs] + 'e+' + str(acc - 1)
        a, b = cancel_scino(a, b)
        a, b = check_exponent_signs(a, b)
        a, b = check_redundant_exponent(a, b)
        a, b = check_redundant_denominator(a, b)
        a, b = check_redundant_exponent(a, b)

        a = asign + a
        b = bsign + b

        """if 'e' in a:
            a = '(' + a + ')'
        if 'e' in b:
            b = '(' + b + ')'
            """

        if b != '':
            if 'e' in a:
                a, ex = a.split('e')
                return str(r'\frac{' + a + r'}{' + b + '}') + r'\times 10^{' + ex + '}'
            return asign + str(r'\frac{' + a + r'}{' + b + '}')
        else:
            if 'e' in a:
                a, ex = a.split('e')
                return str(a) + r'\times 10^{' + ex + '}'
            return a
    return v


def get_function_string(fam, regParams, col='x'):
    f = ''
    c = 1
    tol = 10
    r = 10
    if fam == 'Polynomial' or fam == 'Linear':
        for rP in regParams:
            if not (-1 / (10 ** tol) < rP < 1 / (10 ** tol)) or c == 1:
                rP = fractionize(abs(rP))
                rP0 = str(rP)
                if c != len(regParams):
                    n = len(regParams) - 1 - (c - 1)
                    if n > 1:
                        f += r'{0}'.format(rP0 if rP0.rstrip('0').rstrip(
                            '.') != '1' else '') + r'\mathbf{' + 'x' + '}' + '^{0}'.format(n)
                    else:
                        f += '{0}'.format(rP0 if rP0.rstrip('0').rstrip('.') != '1' else '') + r'\mathbf{' + 'x' + '}'
                    if c == 1 and regParams[0] < 0:
                        f = '-' + f
                else:
                    f += '{0}'.format(rP0, r)
                    break
            if c < len(regParams):
                if regParams[c] > 1 / (10 ** tol):
                    f += '+'
                elif regParams[c] < -1 / (10 ** tol):
                    f += '-'
            c += 1

    elif fam == 'Exponential':
        f = fit_string_exponential(f, regParams)
    elif fam == 'Logarithmic':
        f = fit_string_logarithmic(f, regParams)
    elif fam == 'Sinusoidal':
        f = fit_string_sinusoidal(f, regParams)
    return f


def s_round(arr, acc=4):
    """
    arr: numpy array
    return: numpy array
    """
    m = []
    try:
        for i in range(len(arr)):
            m.append(round_sigfigs(arr[i], acc))
        return np.array(m)
    except Exception as e:
        pass

    try:
        return round_sigfigs(arr, acc)
    except Exception as e:
        pass

    try:
        return sym.N(arr, acc)
    except Exception as e:
        pass
    return arr


def sub_space(num):
    if num == 0:
        return ""
    s = "</br>"
    for i in range(num):
        s += "</br>"
    return s


def make_symbol(args, i, symSim=False):
    """
    args: list or dictionary to interface7
    i: int, position to interface7 in args
    return: value in args[i] if it exists, else None
    """
    try:
        if symSim:
            return sym.N(args[i])
        else:
            if args[i] - int(args[i]) == 0:
                return int(args[i])
            return args[i]
    except:
        pass
    try:
        return sym.Symbol(args[i])
    except:
        return None


def clear_directories(directory):
    for subdir, dirs, files in os.walk(directory):
        for folder in dirs:
            f = os.path.join(subdir, folder)
            for file in os.listdir(f):
                path = os.path.join(f, file)
                try:
                    if os.path.isfile(path):
                        os.unlink(path)  # elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)


# DECORATOR
def log(func):
    @wraps(func)
    def make_log(*args, **kwargs):
        print("Calling function: {}.".format(str(func.__name__)))
        print("With arguments: {}.".format(str(args)))
        print("With keywords: {}.".format(str(kwargs)))
        return func(*args, **kwargs)

    return make_log


def timer(func):
    @wraps(func)
    def time_func(*args, **kwargs):
        startTime = time.time()
        vals = func(*args, **kwargs)
        endTime = time.time() - startTime
        print('Function {0} took {1} seconds.'.format(func.__name__, endTime))
        return vals

    return time_func


def enum_dict(keys, value):
    """
    Creates a dictionary containing identical values for each key.
    keys: list
    value: anything
    return: dictionary
    """
    return {i: value for i in range(keys)}


# DEPRECATED
def fractionize(val):
    # calibration constants
    sigfigs = 5
    roundfigs = sigfigs - 1
    sigfigs2 = 3
    v0 = de_exponentize(val)
    v0 = sym.nsimplify(v0)
    ag = 0
    for char in str(v0):
        if char == '/' or char == '*' or char == '(':
            ag += 1
    if ag < 2:
        v = v0
    else:
        if str(val)[0:2] == '0.':
            art = str(val)[2:]
            ad = 1 + len(art) - len(art.lstrip('0'))
            art = round(val, ad)
        else:
            art = round(val, sigfigs)

        v = art

    if r'/' in str(v):
        a, b = str(v).split(r'/')
        if a[0] == '-':
            asign = '-'
            a = a[1:]
        else:
            asign = ''
        if b[0] == '_':
            b = b[1:]
            bsign = '-'
        else:
            bsign = ''
        # NUMERATOR
        acc = len(a)
        if acc > sigfigs:
            a = a[0] + '.' + a[1:roundfigs] + 'e+' + str(acc - 1)
        # DENOMINATOR
        # check trailing 0
        trail = len(b) - len(b.rstrip('0'))
        if trail > sigfigs2:
            b = b.rstrip('0') + '000' + 'e+' + str(trail - sigfigs2)
        else:
            acc = len(b)
            if acc > sigfigs:
                b = b[0] + '.' + b[1:roundfigs] + 'e+' + str(acc - 1)
        a, b = cancel_scino(a, b)
        a, b = check_exponent_signs(a, b)
        a, b = check_redundant_exponent(a, b)
        a, b = check_redundant_denominator(a, b)
        a, b = check_redundant_exponent(a, b)

        a = asign + a
        b = bsign + b

        """if 'e' in a:
            a = '(' + a + ')'
        if 'e' in b:
            b = '(' + b + ')'
            """

        if b != '':
            if 'e' in a:
                a, ex = a.split('e')
                return str(r'\frac{' + a + r'}{' + b + '}') + r'\times 10^{' + ex + '}'
            return asign + str(r'\frac{' + a + r'}{' + b + '}')
        else:
            if 'e' in a:
                a, ex = a.split('e')
                return str(a) + r'\times 10^{' + ex + '}'
            return a
    return v


def get_function_string(fam, regParams, col='x'):
    f = ''
    c = 1
    tol = 10
    r = 10
    if fam == 'Polynomial' or fam == 'Linear':
        for rP in regParams:
            if not (-1 / (10 ** tol) < rP < 1 / (10 ** tol)) or c == 1:
                rP = fractionize(abs(rP))
                rP0 = str(rP)
                if c != len(regParams):
                    n = len(regParams) - 1 - (c - 1)
                    if n > 1:
                        f += r'{0}'.format(rP0 if rP0.rstrip('0').rstrip(
                            '.') != '1' else '') + r'\mathbf{' + 'x' + '}' + '^{0}'.format(n)
                    else:
                        f += '{0}'.format(rP0 if rP0.rstrip('0').rstrip('.') != '1' else '') + r'\mathbf{' + 'x' + '}'
                    if c == 1 and regParams[0] < 0:
                        f = '-' + f
                else:
                    f += '{0}'.format(rP0, r)
                    break
            if c < len(regParams):
                if regParams[c] > 1 / (10 ** tol):
                    f += '+'
                elif regParams[c] < -1 / (10 ** tol):
                    f += '-'
            c += 1

    elif fam == 'Exponential':
        f = fit_string_exponential(f, regParams)
    elif fam == 'Logarithmic':
        f = fit_string_logarithmic(f, regParams)
    elif fam == 'Sinusoidal':
        f = fit_string_sinusoidal(f, regParams)
    return f


@jit
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


def sub_space(num):
    if num == 0:
        return ""
    s = "</br>"
    for i in range(num):
        s += "</br>"
    return s


def thread(target, args, q=None):
    if type(q) == type(None):
        q1 = queue.Queue()
    else:
        q1 = q

    def wrapper():
        q.put(target(*args))

    t = threading.Thread(target=wrapper)
    t.start()
    t.join()
    return q1


def text_formatter(text, width=100):
    """
    Iterates through each character in text
    and store the number of characters that have been
    iterated over so far in a counter variable.
    If the number of characters exceeds the width,
    a new line character is inserted and the counter resets.
    text: str
    width: int
    return: str
    """
    line = 1
    p = ''
    for e, char in enumerate(text):
        if e != len(text) + 1:
            p += (text[e])
            if (e > line * width and text[e] == ' '):
                line += 1
                p += '\n'
    return p


# DEPRECATED
def fractionize(val):
    # calibration constants
    sigfigs = 5
    roundfigs = sigfigs - 1
    sigfigs2 = 3
    v0 = de_exponentize(val)
    v0 = sym.nsimplify(v0)
    ag = 0
    for char in str(v0):
        if char == '/' or char == '*' or char == '(':
            ag += 1
    if ag < 2:
        v = v0
    else:
        if str(val)[0:2] == '0.':
            art = str(val)[2:]
            ad = 1 + len(art) - len(art.lstrip('0'))
            art = round(val, ad)
        else:
            art = round(val, sigfigs)

        v = art

    if r'/' in str(v):
        a, b = str(v).split(r'/')
        if a[0] == '-':
            asign = '-'
            a = a[1:]
        else:
            asign = ''
        if b[0] == '_':
            b = b[1:]
            bsign = '-'
        else:
            bsign = ''
        # NUMERATOR
        acc = len(a)
        if acc > sigfigs:
            a = a[0] + '.' + a[1:roundfigs] + 'e+' + str(acc - 1)
        # DENOMINATOR
        # check trailing 0
        trail = len(b) - len(b.rstrip('0'))
        if trail > sigfigs2:
            b = b.rstrip('0') + '000' + 'e+' + str(trail - sigfigs2)
        else:
            acc = len(b)
            if acc > sigfigs:
                b = b[0] + '.' + b[1:roundfigs] + 'e+' + str(acc - 1)
        a, b = cancel_scino(a, b)
        a, b = check_exponent_signs(a, b)
        a, b = check_redundant_exponent(a, b)
        a, b = check_redundant_denominator(a, b)
        a, b = check_redundant_exponent(a, b)

        a = asign + a
        b = bsign + b

        """if 'e' in a:
            a = '(' + a + ')'
        if 'e' in b:
            b = '(' + b + ')'
            """

        if b != '':
            if 'e' in a:
                a, ex = a.split('e')
                return str(r'\frac{' + a + r'}{' + b + '}') + r'\times 10^{' + ex + '}'
            return asign + str(r'\frac{' + a + r'}{' + b + '}')
        else:
            if 'e' in a:
                a, ex = a.split('e')
                return str(a) + r'\times 10^{' + ex + '}'
            return a
    return v


def get_function_string(fam, regParams, col='x'):
    f = ''
    c = 1
    tol = 10
    r = 10
    if fam == 'Polynomial' or fam == 'Linear':
        for rP in regParams:
            if not (-1 / (10 ** tol) < rP < 1 / (10 ** tol)) or c == 1:
                rP = fractionize(abs(rP))
                rP0 = str(rP)
                if c != len(regParams):
                    n = len(regParams) - 1 - (c - 1)
                    if n > 1:
                        f += r'{0}'.format(rP0 if rP0.rstrip('0').rstrip(
                            '.') != '1' else '') + r'\mathbf{' + 'x' + '}' + '^{0}'.format(n)
                    else:
                        f += '{0}'.format(rP0 if rP0.rstrip('0').rstrip('.') != '1' else '') + r'\mathbf{' + 'x' + '}'
                    if c == 1 and regParams[0] < 0:
                        f = '-' + f
                else:
                    f += '{0}'.format(rP0, r)
                    break
            if c < len(regParams):
                if regParams[c] > 1 / (10 ** tol):
                    f += '+'
                elif regParams[c] < -1 / (10 ** tol):
                    f += '-'
            c += 1

    elif fam == 'Exponential':
        f = fit_string_exponential(f, regParams)
    elif fam == 'Logarithmic':
        f = fit_string_logarithmic(f, regParams)
    elif fam == 'Sinusoidal':
        f = fit_string_sinusoidal(f, regParams)
    return f


def sub_space(num):
    if num == 0:
        return ""
    s = "</br>"
    for i in range(num):
        s += "</br>"
    return s


def collect_files(filesDir):
    _files = []
    for dir, subdir, files in os.walk(filesDir):
        for file in files:
            _files.append(os.path.join(dir, file))
    return _files


def text_formatter(text, width=100):
    """
    Iterates through each character in text
    and store the number of characters that have been
    iterated over so far in a counter variable.
    If the number of characters exceeds the width,
    a new line character is inserted and the counter resets.
    text: str
    width: int
    return: str
    """
    line = 1
    p = ''
    for e, char in enumerate(text):
        if e != len(text) + 1:
            p += (text[e])
            if (e > line * width and text[e] == ' '):
                line += 1
                p += '\n'
    return p


def is_number(string):
    """
    Determines whether a value is numerical or not.
    For test, if '3.12' is passed, True will be returned.
    If 'a' is passed, False will be returned.
    string: str
    return: bool - True or False
    """
    try:
        float(string)
        return True
    except ValueError as e:
        pass

    try:
        unicodedata.numeric(string)
        return True
    except (TypeError, ValueError) as e:
        pass
    return False


def get_function_string(fam, regParams, col='x'):
    f = ''
    c = 1
    tol = 10
    r = 10
    if fam == 'Polynomial' or fam == 'Linear':
        for rP in regParams:
            if not (-1 / (10 ** tol) < rP < 1 / (10 ** tol)) or c == 1:
                rP = fractionize(abs(rP))
                rP0 = str(rP)
                if c != len(regParams):
                    n = len(regParams) - 1 - (c - 1)
                    if n > 1:
                        f += r'{0}'.format(rP0 if rP0.rstrip('0').rstrip(
                            '.') != '1' else '') + r'\mathbf{' + 'x' + '}' + '^{0}'.format(n)
                    else:
                        f += '{0}'.format(rP0 if rP0.rstrip('0').rstrip('.') != '1' else '') + r'\mathbf{' + 'x' + '}'
                    if c == 1 and regParams[0] < 0:
                        f = '-' + f
                else:
                    f += '{0}'.format(rP0, r)
                    break
            if c < len(regParams):
                if regParams[c] > 1 / (10 ** tol):
                    f += '+'
                elif regParams[c] < -1 / (10 ** tol):
                    f += '-'
            c += 1

    elif fam == 'Exponential':
        f = fit_string_exponential(f, regParams)
    elif fam == 'Logarithmic':
        f = fit_string_logarithmic(f, regParams)
    elif fam == 'Sinusoidal':
        f = fit_string_sinusoidal(f, regParams)
    return f


def text_formatter(text, width=100):
    """
    Iterates through each character in text
    and store the number of characters that have been
    iterated over so far in a counter variable.
    If the number of characters exceeds the width,
    a new line character is inserted and the counter resets.
    text: str
    width: int
    return: str
    """
    line = 1
    p = ''
    for e, char in enumerate(text):
        if e != len(text) + 1:
            p += (text[e])
            if (e > line * width and text[e] == ' '):
                line += 1
                p += '\n'
    return p


def get_function_string(fam, regParams, col='x'):
    f = ''
    c = 1
    tol = 10
    r = 10
    if fam == 'Polynomial' or fam == 'Linear':
        for rP in regParams:
            if not (-1 / (10 ** tol) < rP < 1 / (10 ** tol)) or c == 1:
                rP = fractionize(abs(rP))
                rP0 = str(rP)
                if c != len(regParams):
                    n = len(regParams) - 1 - (c - 1)
                    if n > 1:
                        f += r'{0}'.format(rP0 if rP0.rstrip('0').rstrip(
                            '.') != '1' else '') + r'\mathbf{' + 'x' + '}' + '^{0}'.format(n)
                    else:
                        f += '{0}'.format(rP0 if rP0.rstrip('0').rstrip('.') != '1' else '') + r'\mathbf{' + 'x' + '}'
                    if c == 1 and regParams[0] < 0:
                        f = '-' + f
                else:
                    f += '{0}'.format(rP0, r)
                    break
            if c < len(regParams):
                if regParams[c] > 1 / (10 ** tol):
                    f += '+'
                elif regParams[c] < -1 / (10 ** tol):
                    f += '-'
            c += 1

    elif fam == 'Exponential':
        f = fit_string_exponential(f, regParams)
    elif fam == 'Logarithmic':
        f = fit_string_logarithmic(f, regParams)
    elif fam == 'Sinusoidal':
        f = fit_string_sinusoidal(f, regParams)
    return f


def substitute_kwargs(kwarg1, kwarg2):
    if (not kwarg1) and (not kwarg2):
        print("error")
        k1 = kwarg1
        k2 = kwargs2

    elif not kwarg1:
        k1 = kwarg2
        k2 = kwarg2

    elif not kwarg2:
        k1 = kwarg1
        k2 = kwarg1

    else:
        k1 = kwarg1
        k2 = kwarg2
    return k1, k2


def add_extension(fileName, extension):
    if ('.{}'.format(extension) not in fileName) or (extension not in fileName):
        return '{0}.{1}'.format(fileName, extension)
    else:
        return fileName


def text_formatter(text, width=100):
    """
    Iterates through each character in text
    and store the number of characters that have been
    iterated over so far in a counter variable.
    If the number of characters exceeds the width,
    a new line character is inserted and the counter resets.
    text: str
    width: int
    return: str
    """
    line = 1
    p = ''
    for e, char in enumerate(text):
        if e != len(text) + 1:
            p += (text[e])
            if (e > line * width and text[e] == ' '):
                line += 1
                p += '\n'
    return p


def get_function_string(fam, regParams, col='x'):
    f = ''
    c = 1
    tol = 10
    r = 10
    if fam == 'Polynomial' or fam == 'Linear':
        for rP in regParams:
            if not (-1 / (10 ** tol) < rP < 1 / (10 ** tol)) or c == 1:
                rP = fractionize(abs(rP))
                rP0 = str(rP)
                if c != len(regParams):
                    n = len(regParams) - 1 - (c - 1)
                    if n > 1:
                        f += r'{0}'.format(rP0 if rP0.rstrip('0').rstrip(
                            '.') != '1' else '') + r'\mathbf{' + 'x' + '}' + '^{0}'.format(n)
                    else:
                        f += '{0}'.format(rP0 if rP0.rstrip('0').rstrip('.') != '1' else '') + r'\mathbf{' + 'x' + '}'
                    if c == 1 and regParams[0] < 0:
                        f = '-' + f
                else:
                    f += '{0}'.format(rP0, r)
                    break
            if c < len(regParams):
                if regParams[c] > 1 / (10 ** tol):
                    f += '+'
                elif regParams[c] < -1 / (10 ** tol):
                    f += '-'
            c += 1

    elif fam == 'Exponential':
        f = fit_string_exponential(f, regParams)
    elif fam == 'Logarithmic':
        f = fit_string_logarithmic(f, regParams)
    elif fam == 'Sinusoidal':
        f = fit_string_sinusoidal(f, regParams)
    return f


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


def text_formatter(text, width=100):
    """
    Iterates through each character in text
    and store the number of characters that have been
    iterated over so far in a counter variable.
    If the number of characters exceeds the width,
    a new line character is inserted and the counter resets.
    text: str
    width: int
    return: str
    """
    line = 1
    p = ''
    for e, char in enumerate(text):
        if e != len(text) + 1:
            p += (text[e])
            if (e > line * width and text[e] == ' '):
                line += 1
                p += '\n'
    return p


def get_function_string(fam, regParams, col='x'):
    f = ''
    c = 1
    tol = 10
    r = 10
    if fam == 'Polynomial' or fam == 'Linear':
        for rP in regParams:
            if not (-1 / (10 ** tol) < rP < 1 / (10 ** tol)) or c == 1:
                rP = fractionize(abs(rP))
                rP0 = str(rP)
                if c != len(regParams):
                    n = len(regParams) - 1 - (c - 1)
                    if n > 1:
                        f += r'{0}'.format(rP0 if rP0.rstrip('0').rstrip(
                            '.') != '1' else '') + r'\mathbf{' + 'x' + '}' + '^{0}'.format(n)
                    else:
                        f += '{0}'.format(rP0 if rP0.rstrip('0').rstrip('.') != '1' else '') + r'\mathbf{' + 'x' + '}'
                    if c == 1 and regParams[0] < 0:
                        f = '-' + f
                else:
                    f += '{0}'.format(rP0, r)
                    break
            if c < len(regParams):
                if regParams[c] > 1 / (10 ** tol):
                    f += '+'
                elif regParams[c] < -1 / (10 ** tol):
                    f += '-'
            c += 1

    elif fam == 'Exponential':
        f = fit_string_exponential(f, regParams)
    elif fam == 'Logarithmic':
        f = fit_string_logarithmic(f, regParams)
    elif fam == 'Sinusoidal':
        f = fit_string_sinusoidal(f, regParams)
    return f


def text_formatter(text, width=100):
    """
    Iterates through each character in text
    and store the number of characters that have been
    iterated over so far in a counter variable.
    If the number of characters exceeds the width,
    a new line character is inserted and the counter resets.
    text: str
    width: int
    return: str
    """
    line = 1
    p = ''
    for e, char in enumerate(text):
        if e != len(text) + 1:
            p += (text[e])
            if (e > line * width and text[e] == ' '):
                line += 1
                p += '\n'
    return p


def get_function_string(fam, regParams, col='x'):
    f = ''
    c = 1
    tol = 10
    r = 10
    if fam == 'Polynomial' or fam == 'Linear':
        for rP in regParams:
            if not (-1 / (10 ** tol) < rP < 1 / (10 ** tol)) or c == 1:
                rP = fractionize(abs(rP))
                rP0 = str(rP)
                if c != len(regParams):
                    n = len(regParams) - 1 - (c - 1)
                    if n > 1:
                        f += r'{0}'.format(rP0 if rP0.rstrip('0').rstrip(
                            '.') != '1' else '') + r'\mathbf{' + 'x' + '}' + '^{0}'.format(n)
                    else:
                        f += '{0}'.format(rP0 if rP0.rstrip('0').rstrip('.') != '1' else '') + r'\mathbf{' + 'x' + '}'
                    if c == 1 and regParams[0] < 0:
                        f = '-' + f
                else:
                    f += '{0}'.format(rP0, r)
                    break
            if c < len(regParams):
                if regParams[c] > 1 / (10 ** tol):
                    f += '+'
                elif regParams[c] < -1 / (10 ** tol):
                    f += '-'
            c += 1

    elif fam == 'Exponential':
        f = fit_string_exponential(f, regParams)
    elif fam == 'Logarithmic':
        f = fit_string_logarithmic(f, regParams)
    elif fam == 'Sinusoidal':
        f = fit_string_sinusoidal(f, regParams)
    return f


# DEPRECATED
def fractionize(val):
    # calibration constants
    sigfigs = 5
    roundfigs = sigfigs - 1
    sigfigs2 = 3
    v0 = de_exponentize(val)
    v0 = sym.nsimplify(v0)
    ag = 0
    for char in str(v0):
        if char == '/' or char == '*' or char == '(':
            ag += 1
    if ag < 2:
        v = v0
    else:
        if str(val)[0:2] == '0.':
            art = str(val)[2:]
            ad = 1 + len(art) - len(art.lstrip('0'))
            art = round(val, ad)
        else:
            art = round(val, sigfigs)

        v = art

    if r'/' in str(v):
        a, b = str(v).split(r'/')
        if a[0] == '-':
            asign = '-'
            a = a[1:]
        else:
            asign = ''
        if b[0] == '_':
            b = b[1:]
            bsign = '-'
        else:
            bsign = ''
        # NUMERATOR
        acc = len(a)
        if acc > sigfigs:
            a = a[0] + '.' + a[1:roundfigs] + 'e+' + str(acc - 1)
        # DENOMINATOR
        # check trailing 0
        trail = len(b) - len(b.rstrip('0'))
        if trail > sigfigs2:
            b = b.rstrip('0') + '000' + 'e+' + str(trail - sigfigs2)
        else:
            acc = len(b)
            if acc > sigfigs:
                b = b[0] + '.' + b[1:roundfigs] + 'e+' + str(acc - 1)
        a, b = cancel_scino(a, b)
        a, b = check_exponent_signs(a, b)
        a, b = check_redundant_exponent(a, b)
        a, b = check_redundant_denominator(a, b)
        a, b = check_redundant_exponent(a, b)

        a = asign + a
        b = bsign + b

        """if 'e' in a:
            a = '(' + a + ')'
        if 'e' in b:
            b = '(' + b + ')'
            """

        if b != '':
            if 'e' in a:
                a, ex = a.split('e')
                return str(r'\frac{' + a + r'}{' + b + '}') + r'\times 10^{' + ex + '}'
            return asign + str(r'\frac{' + a + r'}{' + b + '}')
        else:
            if 'e' in a:
                a, ex = a.split('e')
                return str(a) + r'\times 10^{' + ex + '}'
            return a
    return v


def get_function_string(fam, regParams, col='x'):
    f = ''
    c = 1
    tol = 10
    r = 10
    if fam == 'Polynomial' or fam == 'Linear':
        for rP in regParams:
            if not (-1 / (10 ** tol) < rP < 1 / (10 ** tol)) or c == 1:
                rP = fractionize(abs(rP))
                rP0 = str(rP)
                if c != len(regParams):
                    n = len(regParams) - 1 - (c - 1)
                    if n > 1:
                        f += r'{0}'.format(rP0 if rP0.rstrip('0').rstrip(
                            '.') != '1' else '') + r'\mathbf{' + 'x' + '}' + '^{0}'.format(n)
                    else:
                        f += '{0}'.format(rP0 if rP0.rstrip('0').rstrip('.') != '1' else '') + r'\mathbf{' + 'x' + '}'
                    if c == 1 and regParams[0] < 0:
                        f = '-' + f
                else:
                    f += '{0}'.format(rP0, r)
                    break
            if c < len(regParams):
                if regParams[c] > 1 / (10 ** tol):
                    f += '+'
                elif regParams[c] < -1 / (10 ** tol):
                    f += '-'
            c += 1

    elif fam == 'Exponential':
        f = fit_string_exponential(f, regParams)
    elif fam == 'Logarithmic':
        f = fit_string_logarithmic(f, regParams)
    elif fam == 'Sinusoidal':
        f = fit_string_sinusoidal(f, regParams)
    return f


def sub_space(num):
    if num == 0:
        return ""
    s = "</br>"
    for i in range(num):
        s += "</br>"
    return s


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
    else:
        return 0  # Can't take the log of 0


def plot_streamlines(fig, ax, U, u, v, density, atm, xlim, ylim):
    x0, x1 = xlim
    y0, y1 = ylim
    Y, X = np.ogrid[y0:y1:complex(0, 10), x0:x1:complex(0, 10)]
    speed = np.sqrt(u(X, Y) ** 2 + v(X, Y) ** 2)
    pressure = (0.5 * density * (U ** 2 - speed ** 2) + atm) / 1000
    strm = ax.streamplot(X, Y, u(X, Y), v(X, Y), density=1, color=pressure, linewidth=1, cmap='Spectral_r', zorder=-1)
    fig.colorbar(strm.lines)


def plot_cylinders(ax, radii, centroids):
    for radius, centroid in zip(radii, centroids):
        c = plt.Circle(centroid, radius=radius, edgecolor='black', facecolor='none', linewidth=1, )
        ax.add_patch(c)
        c = plt.Circle(centroid, radius=radius, facecolor='grey', alpha=1, )
        ax.add_patch(c)


def format_axes(ax):
    ax.set_aspect(
        'equal')  # ax.figure.subplots_adjust(bottom=0, top=1, left=0, right=1)  # ax.xaxis.set_ticks([])  # ax.yaxis.set_ticks([])


"""

def write_chapter(jobDir=None, jobName=None, templatePath='', **kwargs):

    report_generator0 Structure:
        CHAPTER (Chapters can contain any number of Sections)
            SECTION (Sections can contain any number of Subsections)
                SUBSECTION (Subsections contain content of Figure, Caption, or Paragraph)


    outputPath = os.path.join(jobDir,
                              jobName,
                              'DescriptiveStatistics.html')

    chapterTitle = 'Descriptive Statistics'

    templateVars = {'chapterTitle':
                    chapterTitle,
                'chapterAbstract':
                    ReportGenerator.generate_textstruct(chapterAbstract),
                'chapterSections':
                    ReportGenerator.generate_sentences(chapterSections),
                'heavyBreak':
                    False}
    htmlOut = ReportGenerator.render_HTML(templatePath=templatePath,
                                          templateVars=templateVars,
                                          outputPath=outputPath)


def plots_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The following is a brief description of each of the plots presented in this report'
    ]
    p0['sentence 1'] = [
        'The descriptions indicate how the plots are meant to be interpreted'
    ]

    return [p0]

def box_plot_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The box plots <em>left</em> whisker shows the <em>lowest</em> value in the distribution.'
    ]
    p0['sentence 1'] = [
        'Similarly, the <em>right</em> whisker shows the <em>highest</em> value in the data.'
    ]
    p0['sentence 1'] = [
        'The <em>left</em> edge of the box is the <em>25th Percentile</em>'
    ]
    p0['sentence 2'] = [
        'The <em>right</em> edge of the box is the <em>75th Percentile</em>'
    ]

    p0['sentence 3'] = [
        'Lastly, the <em>line</em> in box is the <em>5oth Percentile</em>, or the <em>median</em>.'
    ]
    return[p0]

def histogram_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The Histogram shows the frequency distribution of the data.'
    ]

    p0['sentence 1'] = [
        'Histograms rule!'
    ]
    return[p0]

def violin_plot_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The Violin Plot shows the frequency distribution of the data and has a mini box plot0.'
    ]

    p0['sentence 1'] = [
        'Violin plot are the future!'
    ]
    return[p0]

def scatter_plot_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The scatter plot0 shows the data.'
    ]

    p0['sentence 1'] = [
        'These plots show a best fit Regression6 when a significant level of correlation is found'
    ]

    return [p0]


def box_plot_section():
    boxPlot = ReportGenerator.Section('Box Plot')
    boxPlotDescription = ReportGenerator.Subsection('Description')
    boxPlotDescription.insert_content(boxPlot_description())
    boxPlotExample = ReportGenerator.Subsection('Example')
    boxplot.insert_subsections([boxPlotDescription,
                             boxPlotExample])
    return boxPlot


def histogram_section():
    histogram = ReportGenerator.Section('Histogram')
    histogramDescription = ReportGenerator.Subsection('Description')
    histogramExample = ReportGenerator.Subsection('Example')
    histogram.insert_subsections([histogramDescription,
                               histogramExample])
    return histogram


def violin_plot_section():
    violin = ReportGenerator.Section('Violin')
    violinDescription = ReportGenerator.Subsection('Description')
    violinExample = ReportGenerator.Subsection('Example')
    violin.insert_subsections([violinDescription,
                             violinExample])
    return violin


def scatter_plot_section():
    scatter = ReportGenerator.Section('Scatter Plot')
    scatterDescription = ReportGenerator.Subsection('Description')
    scatterExample = ReportGenerator.Subsection('Example')
    scatter.insert_subsections([scatterDescription,
                             scatterExample])
    return scatter


def write_chapter():
    chapter = ReportGenerator.Chapter('plot')
    boxPlot = box_plot_section()_section()
    histogram = histogram_section()_section()
    scatterPlot = scatter_section()_section()
    violinPlot = violin_plot_section()_section()
    kurtosis = kurtosis_section()
    chapter.insert_sections([bo])
    print(chapter)


write_chapter()
"""

"""x = Plotter()
x.Bar_Chart_3D(1, [0,5,1,2,3,5,
                  2,1,3,5,6,3,
                  6,2,3,5,2,1,
                  1,3,6,3,1,2], [1], 5)
x.Show()"""

"""
            # Setting the font and style
            matplotlib.rcParams['font.serif'] = "Times New Roman"
            matplotlib.rcParams['font.family'] = "serif"
            self.font = {'fontname': 'Times New Roman'}
            plt.style.use(style)

        Bar_Chart_3D
            #TODO
            xpos = [1, 1, 2, 3, 5, 6, 3, 5, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            ypos = y_input_data
            zpos = np.zeros_like(xpos)
            # Construct arrays with the dimensions for the bars.
            dx, dy = 0.05 * np.ones_like(zpos)
            dz = z_input_data
            # Creating the plot0
            self.graph2.bar3d(xpos, ypos, zpos, dx, dy, dz, color='teal')

        generate_graph
        # Placing the axis labels and the title on the graph2
            plt.xlabel(self.x_label, **self.font)
            plt.ylabel(self.y_label, **self.font)

        generage_graph_3d
            # Placing the axis labels and the title on the graph2
            self.bar_chart_3d.set_xlabel(self.x_label)
            self.bar_chart_3d.set_ylabel(self.y_label)
            self.bar_chart_3d.set_zlabel(self.z_label)

        generate_subplot_3d
        # Creating the figure
            self.bar_chart_3d = plt.figure(figure).add_subplot(111, projection='3d')
            return self.bar_chart_3d
"""

"""
            # Setting the font and style
            matplotlib.rcParams['font.serif'] = "Times New Roman"
            matplotlib.rcParams['font.family'] = "serif"
            self.font = {'fontname': 'Times New Roman'}
            plt.style.use(style)

        Bar_Chart_3D
            #TODO
            xpos = [1, 1, 2, 3, 5, 6, 3, 5, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            ypos = y_input_data
            zpos = np.zeros_like(xpos)
            # Construct arrays with the dimensions for the bars.
            dx, dy = 0.05 * np.ones_like(zpos)
            dz = z_input_data
            # Creating the plot0
            self.graph2.bar3d(xpos, ypos, zpos, dx, dy, dz, color='teal')

        generate_graph
        # Placing the axis labels and the title on the graph2
            plt.xlabel(self.x_label, **self.font)
            plt.ylabel(self.y_label, **self.font)

        generage_graph_3d
            # Placing the axis labels and the title on the graph2
            self.bar_chart_3d.set_xlabel(self.x_label)
            self.bar_chart_3d.set_ylabel(self.y_label)
            self.bar_chart_3d.set_zlabel(self.z_label)

        generate_subplot_3d
        # Creating the figure
            self.bar_chart_3d = plt.figure(figure).add_subplot(111, projection='3d')
            return self.bar_chart_3d
"""

"""
            # Setting the font and style
            matplotlib.rcParams['font.serif'] = "Times New Roman"
            matplotlib.rcParams['font.family'] = "serif"
            self.font = {'fontname': 'Times New Roman'}
            plt.style.use(style)

        Bar_Chart_3D
            #TODO
            xpos = [1, 1, 2, 3, 5, 6, 3, 5, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            ypos = y_input_data
            zpos = np.zeros_like(xpos)
            # Construct arrays with the dimensions for the bars.
            dx, dy = 0.05 * np.ones_like(zpos)
            dz = z_input_data
            # Creating the plot0
            self.graph2.bar3d(xpos, ypos, zpos, dx, dy, dz, color='teal')

        generate_graph
        # Placing the axis labels and the title on the graph2
            plt.xlabel(self.x_label, **self.font)
            plt.ylabel(self.y_label, **self.font)

        generage_graph_3d
            # Placing the axis labels and the title on the graph2
            self.bar_chart_3d.set_xlabel(self.x_label)
            self.bar_chart_3d.set_ylabel(self.y_label)
            self.bar_chart_3d.set_zlabel(self.z_label)

        generate_subplot_3d
        # Creating the figure
            self.bar_chart_3d = plt.figure(figure).add_subplot(111, projection='3d')
            return self.bar_chart_3d
"""
'''
