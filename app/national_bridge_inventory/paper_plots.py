import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from settings import numericalCols, con, conditionCols

plt.switch_backend('agg')
import seaborn as sns

import settings as gc
import util

np.random.seed(1)


def plet(data, title, plotType):
    sns.set_context('paper')
    if plotType != 'pair_plot':
        sns.set_style(gc.snsStyle)
    sns.set_palette(gc.colors)
    plt.legend(fontsize=7, loc='best', frameon=False, borderaxespad=5)
    plt.title(title, fontsize='9', fontweight='bold', fontname='Georgia')

    if plotType == 'dist':
        lot = sns.distplot(data, kde=False, norm_hist=True)
        plt.ylabel('Probability')
        plt.xlabel('Condition')
        plt.xlim(np.min(data) - 1, np.max(data) + 1)

    elif plotType == 'frequency_bar':
        freqs = []
        for col in data:
            freq = data[col].value_counts()
            freqs.append(freq)
        d = pd.concat(freqs, axis=1)
        d['Condition'] = d.index
        d = pd.melt(d, id_vars='Condition', var_name='Item', value_name='Frequency')
        lot = sns.factorplot(x='Condition', y='Frequency', hue='Item', data=d, kind='bar')
        plt.ylabel('Frequency')
        plt.xlabel('Condition')
        # plt.ylim(np.min(data.index.values) - 1, np.max(data.index.values) + 1)
        plt.title(title, fontsize='9', fontweight='bold', fontname='Georgia')
        savePath = os.path.join(gc.plotsDir, 'frequency_bar', title)
        lot.savefig(
            savePath,
            bbox_inches='tight'
        )

    elif plotType == 'frequency_violin':
        d = pd.melt(data, var_name='Item', value_name='Condition')
        lot = sns.violinplot(x='Item', y='Condition', data=d)
        plt.ylabel('Condition')
        plt.xlabel('Item')
        # plt.ylim(np.min(data.index.values) - 1, np.max(data.index.values) + 1)
        savePath = os.path.join(gc.plotsDir, 'frequency_violin', title)
        lot = lot.get_figure()
        lot.savefig(
            savePath,
            bbox_inches='tight'
        )


    elif plotType == 'scatter':
        sns.scatterplot(data=data, legend=False)
        lot = sns.lineplot(data=data)
        plt.xlim(np.min(data.index.values) - 1, np.max(data.index.values) + 1)
        plt.xlabel('Year')
        plt.ylabel('Number of Bridges')
        fig = lot.get_figure()
        savePath = os.path.join(gc.plotsDir, 'scatter', title)
        fig.savefig(
            savePath,
            bbox_inches='tight'
        )


    elif plotType == 'corr':
        mask = np.zeros_like(data, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        lot = sns.heatmap(
            data, annot=True, linewidths=0.5, cmap=cmap,
            center=0, mask=mask, square=True, fmt='.1g',
            vmin=-1, vmax=1, annot_kws={"size": 6}, cbar_kws={"shrink": .5}
        )
        savePath = os.path.join(gc.plotsDir, 'heatmap', title)
        fig = lot.get_figure()
        fig.savefig(
            savePath,
            bbox_inches='tight'
        )

    elif plotType == 'cov':
        mask = np.zeros_like(data, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        lot = sns.heatmap(
            data, annot=True, linewidths=10, cmap=cmap,
            center=0, mask=mask, square=True, fmt='.1g',
            annot_kws={"size": 15}, cbar_kws={"shrink": .5}
        )
        savePath = os.path.join(gc.plotsDir, 'heatmap', title)
        fig = lot.get_figure()
        fig.savefig(
            savePath,
            bbox_inches='tight'
        )

    elif plotType == 'pair_plot':
        lot = sns.PairGrid(data)
        lot.map_upper(sns.regplot)
        lot.map_lower(sns.kdeplot)
        lot.map_diag(sns.kdeplot, lw=3, legend=False)
        sns.set_context('paper')
        sns.set_palette(gc.colors)
        plt.legend(fontsize=7, loc='best', frameon=False, borderaxespad=5)
        plt.title(title, fontsize='9', fontweight='bold', fontname='Georgia')
        sns.set_style(gc.snsStyle)
        savePath = os.path.join(gc.plotsDir, 'pair_plot', title)
        lot.savefig(
            savePath,
            bbox_inches='tight'
        )

    # fig = lot.get_figure()

    plt.gcf().clear()


def corr_cov():
    '''
    FOR HEATMAPS
    '''
    for f in util.get_files(gc.trainingDir):
        print(f)
        sf = util.strip_ext(util.path_end(f))
        year = sf[-4:]
        title = 'Texas Bridge Correlation Heatmap in {}'.format(year)
        data = pd.read_csv(
            f,
            usecols=numericalCols,
            na_values=['N']
        ).dropna(axis=0).rename(columns=con)
        plet(data.corr(), title, 'corr')

        title = 'Texas Bridge Covariance Heatmap in {}'.format(year)
        data = pd.read_csv(
            f,
            usecols=conditionCols,
            na_values=['N']
        ).dropna(axis=0).rename(columns=con)
        plet(data.cov(), title, 'cov')


def frequency():
    '''
    FOR FREQUENCY DISTRIBUTION OF CONDITION RATINGS
    '''
    for f in util.get_files(gc.trainingDir):
        print(f)
        sf = util.strip_ext(util.path_end(f))
        year = sf[-4:]
        title = 'Texas Bridge Condition Ratings in {}'.format(year)
        data = pd.read_csv(
            f,
            usecols=conditionCols,
            na_values=['N']
        ).dropna(axis=0).astype('I').rename(columns=con)
        plet(data, title, 'frequency_bar')
        # plet(data, title, 'frequency_violin')
        # print(data.describe())


def count_bridges():
    '''
    FOR COUNTING THE BRIDGES
    '''
    for f in util.get_files(gc.trainingDir):
        print(f)
        data = pd.read_csv(
            f,
            usecols=conditionCols, )
        print(len(data), len(data.dropna(axis=0)))


def incomplete_bridges():
    '''
    FOR PLOTTING INCOMPLETE BRIDGES
    '''
    data = pd.read_csv(os.path.join('misc', 'incomplete_bridges.csv'))
    data.set_index('Year', inplace=True)
    title = 'Texas Bridge Information'
    plet(data, title, 'scatter')


def full_time_history():
    '''
    FOR PLOTTING TIME HISTORY OF ONE BRIDGE
    '''
    datar = []
    for f in util.get_files(gc.trainingDir):
        print(f)
        sf = util.strip_ext(util.path_end(f))
        year = sf[-4:]
        datar.append(
            pd.read_csv(
                f,
                usecols=['STRUCTURE_NUMBER_008', 'DECK_COND_058', 'YEAR_RECONSTRUCTED_106'],
                na_values=['N']
            ).set_index('STRUCTURE_NUMBER_008').dropna(axis=0).rename(columns=con).add_suffix(
                year
            )
        )
    data = pd.concat(datar, axis=1, join='inner').dropna(axis=0).astype('I')
    for i in range(len(data)):
        if i % 500 == 0:
            if int(list(data.iloc[i, :])[1]) < 1992:
                print('-----------------------------------------')
                print(list(data.iloc[i, :]))
                print('-----------------------------------------')
    print(len(data))


def pair_plot():
    datar = []
    for f in util.get_files(gc.trainingDir):
        sf = util.strip_ext(util.path_end(f))
        year = sf[-4:]
        data = pd.read_csv(
            f,
            usecols=['STRUCTURE_NUMBER_008', 'DECK_COND_058', 'ADT_029', 'PERCENT_ADT_TRUCK_109'],
            na_values=['N']
        ).set_index('STRUCTURE_NUMBER_008').dropna(axis=0).rename(columns=con).add_suffix(year)
        print(data)
        title = 'Texas Bridge Condition Ratings in {}'.format(year)
        plet(data, title, 'pair_plot')
        datar.append(data)

    '''for i in range(len(datar)):
        if i > 0:
            title = 'Texas Bridge Condition Ratings in {}'.format(i)
            current = datar[i]
            previous = datar[i - 1]
            df = pd.concat([current, previous], axis=1, join='inner')
            print(df)
            plet(df, title, 'pair_plot')'''


'''
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print(cosine_similarity(np.array([1, 2, 3]).reshape(1, -1),
                        np.array([4, 5, 6]).reshape(1, -1)
                        ))
print(cosine_similarity(np.array([2, 1, 3]).reshape(1, -1),
                        np.array([5, 4, 6]).reshape(1, -1)
                        ))
'''

# pair_plot()
full_time_history()
