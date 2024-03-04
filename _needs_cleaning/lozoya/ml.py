r'''
from __future__ import division, print_function

# import matplotlib
# matplotlib.use('Qt4Agg')
import mpl_toolkits.mplot3d.axes3d as p3
from Utilities.Paths import *
from bokeh.io import show
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition, svm
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, KMeans, MeanShift, \
    SpectralClustering
from sklearn.datasets import load_digits, make_circles, make_classification, make_moons
from sklearn.datasets.samples_generator import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, SGDClassifier, SGDRegressor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, QuantileTransformer, RobustScaler, \
    StandardScaler
from sklearn.svm import LinearSVR, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import data_api as dh

path = r'C:\Users\cocodell\PycharmProjects\Big\Database\Hernando Desoto\CR1000 - Wireless_VW_Results.dat'
features = ('DisplacementTempDirect(1)', 'Displacementdirect(1)')
data = DataOptimizer.optimize_dataFrame(SyntheticData.data)
dtypes = DataProcessor.get_dtypes(data)
StaticPlotGenerator.generate_all_plots(data, dtypes)
X, y, X_plot = generate_sample_data()

# print(X)
# print(y)
models = grid_search()
fitted_models, fit_time = fitting(models, X, y)
y_models, prediction_time = predicting(fitted_models, X_plot)
plot_stuff(models, y_models, features, fit_time, prediction_time)

rng = np.random.RandomState()
train_size = 100

REGRESSORS = {
    'Adaptive Boost':         AdaBoostRegressor, 'Decision Tree': DecisionTreeRegressor, 'Elastic Net': ElasticNet,
    'Gaussian Process':       GaussianProcessRegressor, 'Nearest Neighbors': KNeighborsRegressor,
    'Kernel Ridge':           KernelRidge, 'Multilayer Perceptron': MLPRegressor,
    'Random Forest':          RandomForestRegressor, 'Stochastic Gradient Descent': SGDRegressor,
    'Support Vector Machine': SVR
}

h = 0.0002  # step size in the mesh

names, classifiers = get_names(), get_classifiers()
X, y = set_data()
linearly_separable = (X, y)

datasets = get_datasets()

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # just plot0 the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")

    plot_test_train(ax, X_train, X_test, y_train, y_test, cm_bright, xx, yy)

    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        Z = plot_decision_boundary(clf, xx, yy)
        plot_scatter_and_color(ds_cnt, ax, Z, xx, yy, cm, X_train, X_test, y_train, y_test, cm_bright)

        i += 1

plt.tight_layout()
plt.show()


# TODO mlpr takes too long and doesn't work
def cross_validation_regression_models(checks):
    models = {}
    if checks['Adaptive Boost']:
        abr = GridSearchCV(
            AdaBoostRegressor(), cv=5,
            param_grid={
                'learning_rate': (0.5, 1,), 'loss': ('linear', 'square', 'exponential'),
                'n_estimators':  (1, 5),  # 'random_state': (None,)
            }
        )
        models['Adaptive Boost'] = abr

    if checks['Decision Tree']:
        dtr = GridSearchCV(
            DecisionTreeRegressor(), cv=5, param_grid={
                'criterion':                ('mse',), 'max_depth': (None,),
                'max_features':             (None, 'auto', 'sqrt', 'log2'),
                'max_leaf_nodes':           (2, 4, 6),
                'min_samples_leaf':         (1, 10, 100, .15, .3, .45),
                # min_samples_split=(options['dtrMinimumSamplesSplit']),
                'min_weight_fraction_leaf': (0,),
                'presort':                  (False,), 'random_state': (None,),
                'splitter':                 ('best', 'random')
            }
        )
        models['Decision Tree'] = dtr

    if checks['Elastic Net']:
        enr = GridSearchCV(
            ElasticNet(), cv=5,
            param_grid={
                'alpha':     (0.1, 1), 'fit_intercept': (True, False), 'l1_ratio': (0.25, 0.5),
                'normalize': (True, False), 'positive': (True, False),
                # 'random_state': (None,),
                'selection': ('cyclic', 'random'), 'tol': (0.01,), 'warm_start': (True, False)
            }
        )
        models['Elastic Net'] = enr

    if checks['Gaussian Process']:
        gpr = GridSearchCV(
            GaussianProcessRegressor(), cv=5,
            param_grid={'alpha': (0.1, 0.5), 'normalize_y': (True, False)}
        )
        models['Gaussian Process'] = gpr

    if checks['Nearest Neighbors']:
        knnr = GridSearchCV(
            KNeighborsRegressor(), cv=5,
            param_grid={
                'algorithm':   ('ball_tree', 'kd_tree', 'brute', 'auto',), 'leaf_size': (30,),
                'n_neighbors': (5,), 'p': (2,), 'weights': ('uniform', 'distance'),
            }
        )
        models['Nearest Neighbors'] = knnr

    if checks['Kernel Ridge']:
        krr = GridSearchCV(
            KernelRidge(), cv=5,
            param_grid={
                'alpha':  (3,), 'coef0': (0,), 'degree': (1, 2, 3), 'gamma': (.25, .5, .75),
                'kernel': ('linear', 'rbf', 'poly')
            }
        )
        models['Kernel Ridge'] = krr

    if checks['Multilayer Perceptron']:
        mlpr = GridSearchCV(
            MLPRegressor(), cv=5,
            param_grid={
                'activation':          ('identity', 'logistic', 'tanh', 'relu'),  # 'alpha': (0.0001,),
                # 'beta_1': (0.5,),
                # 'beta_2': (0.999,),
                # 'early_stopping':(True,False),
                # 'epsilon': (0.00001,),
                # 'hidden_layer_sizes': (100,),
                # 'learning_rate':('constant','invscaling','adaptive'),
                # 'learning_rate_init':(0.001,),
                # 'momentum': (0.5, 0.5),
                # 'nesterovs_momentum':(True,False),
                # 'power_t': (0.5, 0.1),
                'solver':              ('lbfgs', 'sgd', 'adam'),  # 'shuffle':(True,False),
                # 'tol':(0.01,),
                'validation_fraction': (0.5, 0.9),  # 'warm_start':(True,False),
            }
        )
        models['Multilayer Perceptron'] = mlpr

    if checks['Random Forest']:
        rfr = GridSearchCV(
            RandomForestRegressor(n_jobs=-1), cv=5,
            param_grid={
                'bootstrap2':        (True,), 'criterion': ('mse',), 'max_features': ('auto',),
                'max_depth':        (None,), 'max_leaf_nodes': (None,), 'min_samples_split': (2,),
                'min_samples_leaf': (1,), 'min_weight_fraction_leaf': (0,),
                'n_estimators':     (10,), 'oob_score': (False,), 'random_state': (None,),
                'warm_start':       (False,)
            }
        )
        models['Random Forest'] = rfr

    if checks['Stochastic Gradient Descent']:
        sgdr = GridSearchCV(
            SGDRegressor(), cv=5, param_grid={
                'average': (True, False), 'fit_intercept': (True, False),
                'loss':    ('squared_loss', 'huber', 'epsilon_insensitive',
                            'squared_epsilon_insensitive'),
                'penalty': ('none', 'l1', 'l2', 'elasticnet'),
                'power_t': (0.1, 0.5)  # 'alpha': (),
                # 'L1_ratio': (),
                # 'max_iter': (),
                # 'tol': (),
                # 'shuffle': (True, False),
                # 'learning_rate': (0.1, 1),
                # 'eta0': (),
                # 'warm_start': (True, False),
            }
        )
        models['Stochastic Gradient Descent'] = sgdr

    if checks['Support Vector Machine']:
        svmr = GridSearchCV(
            SVR(), cv=5, param_grid={  # 'C': (1, 100),
                # 'coef0': (0, 1),
                # 'degree': (1, 2),
                # 'epsilon': (0.1, 0.5),
                # 'gamma': ('auto',),
                'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),  # 'max_iter': (1, 10, 20,),
                # 'shrinking': (True, False),
                # 'tol': (1e-2,),
            }
        )
        models['Support Vector Machine'] = svmr

    return models


def predicting(models, X_plot, x):
    y_models = []
    for model in models:
        y_models.append(models[model].predict(X_plot))
    prediction = pd.DataFrame(np.array(y_models).T, columns=models, index=x)
    return prediction


def regression(models, x, y, p=None):
    if models:
        a = np.array(x)
        X = np.array([[value] for value in a]).astype(float)
        fitted_models = fitting(models, X, y.as_matrix().astype(float))
        results = predicting(fitted_models, np.linspace(0, p, p)[:, None] if p else X, x)
        return results


def regression_models(checks, options):
    models = {}

    if checks['Adaptive Boost']:
        abr = AdaBoostRegressor(
            learning_rate=options['abrLearningRate'], loss=options['abrLoss'],
            n_estimators=options['abrEstimators']
        )
        # random_state=options['abrRandomState']
        models['Adaptive Boost'] = abr

    if checks['Decision Tree']:
        if options['dtrMaximumDepth'] == 0:
            options['dtrMaximumDepth'] = None

        if options['dtrMaximumLeafNodes'] == 0:
            options['dtrMaximumLeafNodes'] = None

        options['dtrRandomState'] = int(options['dtrRandomState']) if options['dtrRandomState'] else None

        dtr = DecisionTreeRegressor(
            criterion=options['dtrCriterion'], max_depth=options['dtrMaximumDepth'],
            max_features=options['dtrMaximumFeatures'],
            max_leaf_nodes=options['dtrMaximumLeafNodes'],
            min_samples_leaf=options['dtrMinimumSamplesLeaf'],
            # min_samples_split=(options['dtrMinimumSamplesSplit']),
            min_weight_fraction_leaf=options['dtrMinimumWeightFractionLeaf'],
            random_state=options['dtrRandomState'], presort=options['dtrPresort'],
            splitter=options['dtrSplitter']
        )
        models['Decision Tree'] = dtr

    if checks['Elastic Net']:
        enr = ElasticNet(
            alpha=options['enrAlpha'],  # fit_intercept=options['enrFitIntercept'],
            l1_ratio=options['enrL1Ratio'], normalize=options['enrNormalize'],
            positive=options['enrPositive'],  # random_state=(None,),
            selection=options['enrSelection'], tol=options['enrTolerance'],
            warm_start=options['enrWarmStart']
        )
        models['Elastic Net'] = enr

    if checks['Gaussian Process']:
        if not options['gprAlpha']:
            options['gprAlpha'] = 1e-10
        else:
            options['gprAlpha'] = float(options['gprAlpha'])

        gpr = GaussianProcessRegressor(
            alpha=options['gprAlpha'],  # 'gprKernel': self.GPkernelCombo.currentText(),
            # 'gprOptimizer': self.GPoptimizerCombo.currentText(),
            normalize_y=options['gprNormalize'], )
        models['Gaussian Process'] = gpr

    if checks['Nearest Neighbors']:
        knnr = KNeighborsRegressor(
            algorithm=options['knnrAlgorithm'], leaf_size=int(options['knnrLeafSize']),
            n_neighbors=options['knnrNumberOfNeighbors'], p=options['knnrMinkowskiPower'],
            weights=options['knnrWeightsFunction']
        )
        models['Nearest Neighbors'] = knnr

    if checks['Kernel Ridge']:
        if options['krrGamma'] == 0:
            options['krrGamma'] = None

        options['krrPolynomialDegree'] = int(options['krrPolynomialDegree'])

        krr = KernelRidge(
            alpha=options['krrAlpha'], coef0=options['krrCoefficient0'],
            degree=(options['krrPolynomialDegree']), gamma=options['krrGamma'],
            kernel=options['krrKernel']
        )

        models['Kernel Ridge'] = krr

    if checks['Multilayer Perceptron']:
        if not options['mlprHiddenLayerSizes']:
            options['mlprHiddenLayerSizes'] = (100,)
        else:
            options['mlprHiddenLayerSizes'] = int(options['mlprHiddenLayerSizes'])

        options['mlprRandomState'] = int(options['mlprRandomState']) if options['mlprRandomState'] else None

        mlpr = MLPRegressor(
            activation=options['mlprActivationFunction'], alpha=options['mlprPenaltyParameter'],
            # batch_size=options['mlprBatchSize'],
            beta_1=options['mlprFirstMomentExponentialDecay'],
            beta_2=options['mlprSecondMomentExponentialDecay'],
            early_stopping=options['mlprEarlyStopping'], epsilon=options['mlprNumericalStability'],
            hidden_layer_sizes=options['mlprHiddenLayerSizes'],
            learning_rate=options['mlprLearningRate'],
            learning_rate_init=options['mlprInitialLearningRate'],
            max_iter=options['mlprMaximumIterations'], momentum=options['mlprMomentum'],
            nesterovs_momentum=options['mlprNesterovsMomentum'],
            power_t=options['mlprPowerForInverseLearningRate'], random_state=options['mlprRandomState'],
            shuffle=options['mlprShuffle'], solver=options['mlprWeightOptimizationSolver'],
            tol=options['mlprTolerance'], validation_fraction=options['mlprValidationFraction'],
            warm_start=options['mlprWarmStart']
        )
        models['Multilayer Perceptron'] = mlpr

    if checks['Random Forest']:
        if options['rfrMaximumDepth'] == 0:
            options['rfrMaximumDepth'] = None

        if options['rfrMaximumLeafNodes'] == 0:
            options['rfrMaximumLeafNodes'] = None

        options['rfrRandomState'] = int(options['rfrRandomState']) if options['rfrRandomState'] else None

        rfr = RandomForestRegressor(
            bootstrap2=options['rfrBootstrap'], criterion=options['rfrCriterion'],
            max_depth=options['rfrMaximumDepth'], max_features=options['rfrMaximumFeatures'],
            max_leaf_nodes=options['rfrMaximumLeafNodes'],
            min_samples_leaf=options['rfrMinimumSamplesAtLeaf'],
            min_samples_split=options['rfrMinimumSamplesSplit'],
            min_weight_fraction_leaf=options['rfrMinimumSumWeightedFraction'],
            n_estimators=options['rfrNumberOfTrees'], oob_score=options['rfrOutOfBagSamples'],
            random_state=options['rfrRandomState'], warm_start=options['rfrWarmStart'], )

        models['Random Forest'] = rfr

    if checks['Stochastic Gradient Descent']:
        if options['sgdrTolerance'] == 0:
            options['sgdrTolerance'] = None

        sgdr = SGDRegressor(
            alpha=options['sgdrAlpha'], average=options['sgdrAverage'], eta0=options['sgdrEta0'],
            fit_intercept=options['sgdrFitIntercept'], learning_rate=options['sgdrLearningRate'],
            loss=options['sgdrLoss'], l1_ratio=options['sgdrL1Ratio'],
            max_iter=options['sgdrMaxIterations'], penalty=options['sgdrPenalty'],
            power_t=options['sgdrPowerT'], shuffle=options['sgdrShuffle'], tol=options['sgdrTolerance'],
            warm_start=options['sgdrWarmStart']
        )
        models['Stochastic Gradient Descent'] = sgdr

    if checks['Support Vector Machine']:
        options['svmrPolynomialDegree'] = int(options['svmrPolynomialDegree'])

        if options['svmrGamma'] == 0:
            options['svmrGamma'] = 'auto'

        svmr = SVR(
            C=options['svmrC'], epsilon=options['svmrEpsilon'], kernel=options['svmrKernel'],
            degree=options['svmrPolynomialDegree'], gamma=options['svmrGamma'],
            coef0=options['svmrCoefficient0'], shrinking=options['svmrShrinking'], tol=options['svmrTolerance'],
            # cache_size=options['svmrCacheSize'],
            max_iter=options['svmrMaximumIterations']
        )
        models['Support Vector Machine'] = svmr

    if checks['Support Vector Machine']:
        options['svrPolynomialDegree'] = int(options['svrPolynomialDegree'])

        if options['svrGamma'] == 0:
            options['svrGamma'] = 'auto'

        svr = SVR(
            C=options['svrC'], epsilon=options['svrEpsilon'], kernel=options['svrKernel'],
            degree=options['svrPolynomialDegree'], gamma=options['svrGamma'], coef0=options['svrCoefficient0'],
            shrinking=options['svrShrinking'], tol=options['svrTolerance'],  # cache_size=options['svrCacheSize'],
            max_iter=options['svrMaximumIterations']
        )
        models['Support Vector Machine'] = svr

    return models


# #############################################################################
def generate_sample_data():
    # Generate sample data
    X = 5 * rng.rand(500, 1)
    y = np.sin(X).ravel()
    # Add noise to targets
    noise = 10
    y[::noise] += 3 * (0.5 - rng.rand(X.shape[0] // noise))
    X_plot = np.linspace(0, 5, 10000)[:, None]
    return X, y, X_plot


def generate_data(path, features, step=1):
    X = pd.read_csv(path, dtype='str')
    clean_X = clean_data(X)[::step]
    clean_X.fillna(0, inplace=True)
    X = pd.DataFrame(clean_X[features[0]]).as_matrix().astype(float)
    y = clean_X[features[1]].astype(float).values
    X_plot = np.linspace(0, max(X)[0], 100)[:, None]
    return X, y, X_plot


# #############################################################################
def grid_search():
    # Fit Regression6 model
    lsvr = GridSearchCV(LinearSVR(), cv=5, param_grid={})
    svs = GridSearchCV(
        SVR(kernel='rbf', gamma='auto'), cv=5,
        param_grid={"C": [1e0], "gamma": np.logspace(-2, 2, 5), "degree": [1, 2]}
    )
    svr = GridSearchCV(
        SVR(kernel='sigmoid', gamma='auto'), cv=5,
        param_grid={  # "kernel": ['linear', 'rbf', 'sigmoid'],
            "gamma": np.logspace(-2, 2, 5)
        }
    )
    if False == True:
        kr = GridSearchCV(
            KernelRidge(kernel='rbf', gamma=0.1), cv=5,
            param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)}
        )

        dtr = GridSearchCV(
            DecisionTreeRegressor(), cv=5,
            param_grid={
                "max_depth":    [2, 5, 20], "min_samples_leaf": [5, 10, 20],
                "max_features": ['auto', 'sqrt', 'log2']
            }
        )

        nnr = GridSearchCV(
            MLPRegressor(), cv=5, param_grid={
                "activation": ['identity', 'logistic', 'tanh', 'relu'],
                "solver":     ['lbfgs', 'sgd', 'adam'],
                # "learning_rate": ['constant', 'invscaling', 'adaptive'],
                # "alpha": [0.01, 0.001, 0.0001],
                "max_iter":   [100]
            }
        )
    knn = GridSearchCV(KNeighborsRegressor(), cv=5, param_grid={})

    rfr = GridSearchCV(
        RandomForestRegressor(n_jobs=-1), cv=5,
        param_grid={
            "n_estimators": [2, 5, 10, 1000], "max_features": ['auto', 'sqrt', 'log2'],
            "criterion":    ['mse', 'mae']
        }
    )

    # models = {"LSVR": lsvr, "SVR": svr, "KR": kr, "RFR": rfr, "DTR": dtr, "NNR": nnr}
    # models = {"LSVR": lsvr, "SVR": svr, "KNN": knn, "RFR": rfr}
    models = {"RFR": rfr, "SVR": svr}
    return models


###########################################################################
def fitting(models, X, y):
    m = models
    fit_time = []
    for i, model in enumerate(models):
        t0 = time.time()
        m[model].fit(X[:train_size], y[:train_size])
        fit_time.append(
            time.time() - t0
        )  # print(model + "complexity and bandwidth selected and model fitted in %.3f s" % fit_time[i])
    return m, fit_time


############################################################################
def predicting(models, X_plot):
    prediction_time = []
    y_models = []
    for i, model in enumerate(models):
        t0 = time.time()
        y_models.append(models[model].predict(X_plot))
        prediction_time.append(
            time.time() - t0
        )  # print(model + " prediction for %d inputs in %.3f s" % (X_plot.shape[0], prediction_time[i]))
    return y_models, prediction_time


# #############################################################################
def plot_stuff(models, y_models, features, fit_time, predict_time):
    colors = ('r', 'g', 'b', 'y', 'c', 'm')
    # Look at the results
    sv_ind = models['SVR'].best_estimator_.support_
    plt.scatter(X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors', zorder=2, edgecolors=(0, 0, 0))
    plt.scatter(X, y, c='k', label='data', zorder=1, edgecolors=(0, 0, 0))

    for i, model, y_model in zip(range(len(models)), models, y_models):
        plt.plot(
            X_plot, y_model, c=colors[i],
            label=model + ' (fit: %.3fs, predict: %.3fs)' % (fit_time[i], predict_time[i])
        )

    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('Regression6 Comparison')
    plt.legend()
    plt.show()


# TODO mlpr takes too long and doesn't work
def cross_validation_models(checks):
    models = {}
    if 'Adaptive Boost' in checks:
        abr = GridSearchCV(
            AdaBoostRegressor(), cv=5,
            param_grid={
                'learning_rate': (0.5, 1,), 'loss': ('linear', 'square', 'exponential'),
                'n_estimators':  (1, 5),  # 'random_state': (None,)
            }
        )
        models['Adaptive Boost'] = abr

    if 'Decision Tree' in checks:
        dtr = GridSearchCV(
            DecisionTreeRegressor(), cv=5, param_grid={
                'criterion':                ('mse',), 'max_depth': (None,),
                'max_features':             (None, 'auto', 'sqrt', 'log2'),
                'max_leaf_nodes':           (2, 4, 6),
                'min_samples_leaf':         (1, 10, 100, .15, .3, .45),
                # min_samples_split=(options['dtrMinimumSamplesSplit']),
                'min_weight_fraction_leaf': (0,),
                'presort':                  (False,), 'random_state': (None,),
                'splitter':                 ('best', 'random')
            }
        )
        models['Decision Tree'] = dtr

    if 'Elastic Net' in checks:
        enr = GridSearchCV(
            ElasticNet(), cv=5,
            param_grid={
                'alpha':     (0.1, 1), 'fit_intercept': (True, False), 'l1_ratio': (0.25, 0.5),
                'normalize': (True, False), 'positive': (True, False),
                # 'random_state': (None,),
                'selection': ('cyclic', 'random'), 'tol': (0.01,), 'warm_start': (True, False)
            }
        )
        models['Elastic Net'] = enr

    if 'Gaussian Process' in checks:
        gpr = GridSearchCV(
            GaussianProcessRegressor(), cv=5,
            param_grid={'alpha': (0.1, 0.5), 'normalize_y': (True, False)}
        )
        models['Gaussian Process'] = gpr

    if 'Nearest Neighbors' in checks:
        knnr = GridSearchCV(
            KNeighborsRegressor(), cv=5,
            param_grid={
                'algorithm':   ('ball_tree', 'kd_tree', 'brute', 'auto',), 'leaf_size': (30,),
                'n_neighbors': (5,), 'p': (2,), 'weights': ('uniform', 'distance'),
            }
        )
        models['Nearest Neighbors'] = knnr

    if 'Kernel Ridge' in checks:
        krr = GridSearchCV(
            KernelRidge(), cv=5,
            param_grid={
                'alpha':  (3,), 'coef0': (0,), 'degree': (1, 2, 3), 'gamma': (.25, .5, .75),
                'kernel': ('linear', 'rbf', 'poly')
            }
        )
        models['Kernel Ridge'] = krr

    if 'Multilayer Perceptron' in checks:
        mlpr = GridSearchCV(
            MLPRegressor(), cv=5,
            param_grid={
                'activation':          ('identity', 'logistic', 'tanh', 'relu'),  # 'alpha': (0.0001,),
                # 'beta_1': (0.5,),
                # 'beta_2': (0.999,),
                # 'early_stopping':(True,False),
                # 'epsilon': (0.00001,),
                # 'hidden_layer_sizes': (100,),
                # 'learning_rate':('constant','invscaling','adaptive'),
                # 'learning_rate_init':(0.001,),
                # 'momentum': (0.5, 0.5),
                # 'nesterovs_momentum':(True,False),
                # 'power_t': (0.5, 0.1),
                'solver':              ('lbfgs', 'sgd', 'adam'),  # 'shuffle':(True,False),
                # 'tol':(0.01,),
                'validation_fraction': (0.5, 0.9),  # 'warm_start':(True,False),
            }
        )
        models['Multilayer Perceptron'] = mlpr

    if 'Random Forest' in checks:
        rfr = GridSearchCV(
            RandomForestRegressor(n_jobs=-1), cv=5,
            param_grid={
                'bootstrap2':        (True,), 'criterion': ('mse',), 'max_features': ('auto',),
                'max_depth':        (None,), 'max_leaf_nodes': (None,), 'min_samples_split': (2,),
                'min_samples_leaf': (1,), 'min_weight_fraction_leaf': (0,),
                'n_estimators':     (10,), 'oob_score': (False,), 'random_state': (None,),
                'warm_start':       (False,)
            }
        )
        models['Random Forest'] = rfr

    if 'Stochastic Gradient Descent' in checks:
        sgdr = GridSearchCV(
            SGDRegressor(), cv=5, param_grid={
                'average': (True, False), 'fit_intercept': (True, False),
                'loss':    ('squared_loss', 'huber', 'epsilon_insensitive',
                            'squared_epsilon_insensitive'),
                'penalty': ('none', 'l1', 'l2', 'elasticnet'),
                'power_t': (0.1, 0.5)  # 'alpha': (),
                # 'L1_ratio': (),
                # 'max_iter': (),
                # 'tol': (),
                # 'shuffle': (True, False),
                # 'learning_rate': (0.1, 1),
                # 'eta0': (),
                # 'warm_start': (True, False),
            }
        )
        models['Stochastic Gradient Descent'] = sgdr

    if 'Support Vector Machine' in checks:
        svmr = GridSearchCV(
            SVR(), cv=5, param_grid={  # 'C': (1, 100),
                # 'coef0': (0, 1),
                # 'degree': (1, 2),
                # 'epsilon': (0.1, 0.5),
                # 'gamma': ('auto',),
                'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),  # 'max_iter': (1, 10, 20,),
                # 'shrinking': (True, False),
                # 'tol': (1e-2,),
            }
        )
        models['Support Vector Machine'] = svmr

    return models


def models(checks, options):
    models = {}

    if 'Adaptive Boost' in checks:
        settings = options['Adaptive Boost']
        abr = AdaBoostRegressor(**settings)
        models['Adaptive Boost'] = abr

    if 'Decision Tree' in checks:
        settings = options['Decision Tree']
        if settings['max_depth'] == 0:
            settings['max_depth'] = None

        if settings['max_leaf_nodes'] == 0:
            settings['max_leaf_nodes'] = None

        dtr = DecisionTreeRegressor(**settings)
        models['Decision Tree'] = dtr

    if 'Elastic Net' in checks:
        enr = ElasticNet(
            alpha=options['enrAlpha'],  # fit_intercept=options['enrFitIntercept'],
            l1_ratio=options['enrL1Ratio'], normalize=options['enrNormalize'],
            positive=options['enrPositive'],  # random_state=(None,),
            selection=options['enrSelection'], tol=options['enrTolerance'],
            warm_start=options['enrWarmStart']
        )
        models['Elastic Net'] = enr

    if 'Gaussian Process' in checks:
        if not options['gprAlpha']:
            options['gprAlpha'] = 1e-10
        else:
            options['gprAlpha'] = float(options['gprAlpha'])

        gpr = GaussianProcessRegressor(
            alpha=options['gprAlpha'],  # 'gprKernel': self.GPkernelCombo.currentText(),
            # 'gprOptimizer': self.GPoptimizerCombo.currentText(),
            normalize_y=options['gprNormalize'], )
        models['Gaussian Process'] = gpr

    if 'Nearest Neighbors' in checks:
        knnr = KNeighborsRegressor(
            algorithm=options['knnrAlgorithm'], leaf_size=int(options['knnrLeafSize']),
            n_neighbors=options['knnrNumberOfNeighbors'], p=options['knnrMinkowskiPower'],
            weights=options['knnrWeightsFunction']
        )
        models['Nearest Neighbors'] = knnr

    if 'Kernel Ridge' in checks:
        if options['krrGamma'] == 0:
            options['krrGamma'] = None

        options['krrPolynomialDegree'] = int(options['krrPolynomialDegree'])

        krr = KernelRidge(
            alpha=options['krrAlpha'], coef0=options['krrCoefficient0'],
            degree=(options['krrPolynomialDegree']), gamma=options['krrGamma'],
            kernel=options['krrKernel']
        )

        models['Kernel Ridge'] = krr

    if 'Multilayer Perceptron' in checks:
        if not options['mlprHiddenLayerSizes']:
            options['mlprHiddenLayerSizes'] = (100,)
        else:
            options['mlprHiddenLayerSizes'] = int(options['mlprHiddenLayerSizes'])

        options['mlprRandomState'] = int(options['mlprRandomState']) if options['mlprRandomState'] else None

        mlpr = MLPRegressor(
            activation=options['mlprActivationFunction'], alpha=options['mlprPenaltyParameter'],
            # batch_size=options['mlprBatchSize'],
            beta_1=options['mlprFirstMomentExponentialDecay'],
            beta_2=options['mlprSecondMomentExponentialDecay'],
            early_stopping=options['mlprEarlyStopping'], epsilon=options['mlprNumericalStability'],
            hidden_layer_sizes=options['mlprHiddenLayerSizes'],
            learning_rate=options['mlprLearningRate'],
            learning_rate_init=options['mlprInitialLearningRate'],
            max_iter=options['mlprMaximumIterations'], momentum=options['mlprMomentum'],
            nesterovs_momentum=options['mlprNesterovsMomentum'],
            power_t=options['mlprPowerForInverseLearningRate'], random_state=options['mlprRandomState'],
            shuffle=options['mlprShuffle'], solver=options['mlprWeightOptimizationSolver'],
            tol=options['mlprTolerance'], validation_fraction=options['mlprValidationFraction'],
            warm_start=options['mlprWarmStart']
        )
        models['Multilayer Perceptron'] = mlpr
    if 'Random Forest' in checks:
        if options['rfrMaximumDepth'] == 0:
            options['rfrMaximumDepth'] = None
        if options['rfrMaximumLeafNodes'] == 0:
            options['rfrMaximumLeafNodes'] = None
        options['rfrRandomState'] = int(options['rfrRandomState']) if options['rfrRandomState'] else None
        rfr = RandomForestRegressor(
            bootstrap2=options['rfrBootstrap'], criterion=options['rfrCriterion'],
            max_depth=options['rfrMaximumDepth'], max_features=options['rfrMaximumFeatures'],
            max_leaf_nodes=options['rfrMaximumLeafNodes'],
            min_samples_leaf=options['rfrMinimumSamplesAtLeaf'],
            min_samples_split=options['rfrMinimumSamplesSplit'],
            min_weight_fraction_leaf=options['rfrMinimumSumWeightedFraction'],
            n_estimators=options['rfrNumberOfTrees'], oob_score=options['rfrOutOfBagSamples'],
            random_state=options['rfrRandomState'], warm_start=options['rfrWarmStart'], )
        models['Random Forest'] = rfr
    if 'Stochastic Gradient Descent' in checks:
        if options['sgdrTolerance'] == 0:
            options['sgdrTolerance'] = None
        sgdr = SGDRegressor(
            alpha=options['sgdrAlpha'], average=options['sgdrAverage'], eta0=options['sgdrEta0'],
            fit_intercept=options['sgdrFitIntercept'], learning_rate=options['sgdrLearningRate'],
            loss=options['sgdrLoss'], l1_ratio=options['sgdrL1Ratio'],
            max_iter=options['sgdrMaxIterations'], penalty=options['sgdrPenalty'],
            power_t=options['sgdrPowerT'], shuffle=options['sgdrShuffle'], tol=options['sgdrTolerance'],
            warm_start=options['sgdrWarmStart']
        )
        models['Stochastic Gradient Descent'] = sgdr
    if 'Support Vector Machine' in checks:
        options['svmrPolynomialDegree'] = int(options['svmrPolynomialDegree'])
        if options['svmrGamma'] == 0:
            options['svmrGamma'] = 'auto'
        svmr = SVR(
            C=options['svmrC'], epsilon=options['svmrEpsilon'], kernel=options['svmrKernel'],
            degree=options['svmrPolynomialDegree'], gamma=options['svmrGamma'],
            coef0=options['svmrCoefficient0'], shrinking=options['svmrShrinking'], tol=options['svmrTolerance'],
            # cache_size=options['svmrCacheSize'],
            max_iter=options['svmrMaximumIterations']
        )
        models['Support Vector Machine'] = svmr
    return models


def run(dir, index, column, models, mode):
    data, xKeys, yKeys = dh.process_regression_data(dir, index, column, mode, )
    results = regression(models, x=data.index, y=data)
    data = pd.DataFrame(data, columns=(column,), index=data.index)
    x = ((data.index, data.index),)
    y = ((results, data),)
    return x, y, xKeys, yKeys


def plot_regression(x, y, xLabel, yLabel, xKeys, yKeys):
    make_plot(
        REGRESSION_GRAPH, X=x, Y=y, titles=('Regression6',), xLabels=(xLabel,), yLabels=(yLabel,),
        lineWidths=((1, 1),), legends=((True, True),), types=(('line', 'scatter'),), xKeys=(xKeys,),
        yKeys=(yKeys,)
    )


# TODO mlpr takes too long and doesn't work
def cross_validation_models(checks):
    models = {}
    if 'Adaptive Boost' in checks:
        abr = GridSearchCV(
            AdaBoostRegressor(), cv=5,
            param_grid={
                'learning_rate': (0.5, 1,), 'loss': ('linear', 'square', 'exponential'),
                'n_estimators':  (1, 5),
            }
        )

        models['Adaptive Boost'] = abr

    if 'Decision Tree' in checks:
        dtr = GridSearchCV(
            DecisionTreeRegressor(), cv=5, param_grid={
                'criterion':                ('mse',), 'max_depth': (None,),
                'max_features':             (None, 'auto', 'sqrt', 'log2'),
                'max_leaf_nodes':           (2, 4, 6),
                'min_samples_leaf':         (1, 10, 100, .15, .3, .45),
                # min_samples_split=(options['dtrMinimumSamplesSplit']),
                'min_weight_fraction_leaf': (0,),
                'presort':                  (False,), 'random_state': (None,),
                'splitter':                 ('best', 'random')
            }
        )
        models['Decision Tree'] = dtr

    if 'Elastic Net' in checks:
        enr = GridSearchCV(
            ElasticNet(), cv=5,
            param_grid={
                'alpha':     (0.1, 1), 'fit_intercept': (True, False), 'l1_ratio': (0.25, 0.5),
                'normalize': (True, False), 'positive': (True, False),
                'selection': ('cyclic', 'random'), 'tol': (0.01,), 'warm_start': (True, False)
            }
        )
        models['Elastic Net'] = enr

    if 'Gaussian Process' in checks:
        gpr = GridSearchCV(
            GaussianProcessRegressor(), cv=5,
            param_grid={'alpha': (0.1, 0.5), 'normalize_y': (True, False)}
        )
        models['Gaussian Process'] = gpr

    if 'Nearest Neighbors' in checks:
        knnr = GridSearchCV(
            KNeighborsRegressor(), cv=5,
            param_grid={
                'algorithm':   ('ball_tree', 'kd_tree', 'brute', 'auto',), 'leaf_size': (30,),
                'n_neighbors': (5,), 'p': (2,), 'weights': ('uniform', 'distance'),
            }
        )
        models['Nearest Neighbors'] = knnr

    if 'Kernel Ridge' in checks:
        krr = GridSearchCV(
            KernelRidge(), cv=5,
            param_grid={
                'alpha':  (3,), 'coef0': (0,), 'degree': (1, 2, 3), 'gamma': (.25, .5, .75),
                'kernel': ('linear', 'rbf', 'poly')
            }
        )
        models['Kernel Ridge'] = krr

    if 'Multilayer Perceptron' in checks:
        mlpr = GridSearchCV(
            MLPRegressor(), cv=5,
            param_grid={
                'activation':          ('identity', 'logistic', 'tanh', 'relu'),  # 'alpha': (0.0001,),
                # 'beta_1': (0.5,),
                # 'beta_2': (0.999,),
                # 'early_stopping':(True,False),
                # 'epsilon': (0.00001,),
                # 'hidden_layer_sizes': (100,),
                # 'learning_rate':('constant','invscaling','adaptive'),
                # 'learning_rate_init':(0.001,),
                # 'momentum': (0.5, 0.5),
                # 'nesterovs_momentum':(True,False),
                # 'power_t': (0.5, 0.1),
                'solver':              ('lbfgs', 'sgd', 'adam'),  # 'shuffle':(True,False),
                # 'tol':(0.01,),
                'validation_fraction': (0.5, 0.9),  # 'warm_start':(True,False),
            }
        )
        models['Multilayer Perceptron'] = mlpr

    if 'Random Forest' in checks:
        rfr = GridSearchCV(
            RandomForestRegressor(n_jobs=-1), cv=5,
            param_grid={
                'bootstrap2':        (True,), 'criterion': ('mse',), 'max_features': ('auto',),
                'max_depth':        (None,), 'max_leaf_nodes': (None,), 'min_samples_split': (2,),
                'min_samples_leaf': (1,), 'min_weight_fraction_leaf': (0,),
                'n_estimators':     (10,), 'oob_score': (False,), 'warm_start': (False,)
            }
        )
        models['Random Forest'] = rfr

    if 'Stochastic Gradient Descent' in checks:
        sgdr = GridSearchCV(
            SGDRegressor(), cv=5, param_grid={
                'average':  (True, False), 'fit_intercept': (True, False),
                'loss':     ('squared_loss', 'huber', 'epsilon_insensitive',
                             'squared_epsilon_insensitive'),
                'penalty':  ('none', 'l1', 'l2', 'elasticnet'),
                'power_t':  (0.1, 0.5),  # 'alpha': (),
                # 'L1_ratio': (),
                'max_iter': (1000,), 'tol': (1e-3,),
                # 'shuffle': (True, False),
                # 'learning_rate': (0.1, 1),
                # 'eta0': (),
                # 'warm_start': (True, False),
            }
        )
        models['Stochastic Gradient Descent'] = sgdr

    if 'Support Vector Machine' in checks:
        svmr = GridSearchCV(
            SVR(), cv=5, param_grid={  # 'C': (1, 100),
                # 'coef0': (0, 1),
                # 'degree': (1, 2),
                # 'epsilon': (0.1, 0.5),
                # 'gamma': ('auto',),
                'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),  # 'max_iter': (1, 10, 20,),
                # 'shrinking': (True, False),
                # 'tol': (1e-2,),
            }
        )
        models['Support Vector Machine'] = svmr

    return models


def regression(models, index, x, y, p=None):
    if models:
        fitted_models = MachineLearner.fitting(models, x, y)
        results = MachineLearner.predicting(fitted_models, np.linspace(0, p, p)[:, None] if p else x, index)
        return results


def models(checks, options):
    if 'Decision Tree' in checks:
        settings = options['Decision Tree']
        if settings['max_depth'] == 0:
            settings['max_depth'] = None
        if settings['max_leaf_nodes'] == 0:
            settings['max_leaf_nodes'] = None

    if 'Gaussian Process' in checks:
        settings = options['Gaussian Process']
        if settings['alpha'] == 0:
            settings['alpha'] = 1e-10
        else:
            settings['alpha'] = float(settings['alpha'])

    if 'Kernel Ridge' in checks:
        settings = options['Kernel Ridge']
        if settings['gamma'] == 0:
            settings['gamma'] = None
        settings['degree'] = int(settings['degree'])

    if 'Multilayer Perceptron' in checks:
        settings = options['Multilayer Perceptron']
        if not settings['hidden_layer_sizes']:
            settings['hidden_layer_sizes'] = (100,)
        else:
            settings['hidden_layer_sizes'] = int(settings['hidden_layer_sizes'])

    if 'Stochastic Gradient Descent' in checks:
        settings = options['Stochastic Gradient Descent']
        if settings['tol'] == 0:
            settings['tol'] = None

    if 'Support Vector Machine' in checks:
        settings = options['Support Vector Machine']
        settings['degree'] = int(settings['degree'])

        if settings['gamma'] == 0:
            settings['gamma'] = 'auto'

    models = MachineLearner.get_models(REGRESSORS, checks, options)
    return models


def run(widget, dir, index, column, target, models, scaler, mode):
    x, y, xKeys, yKeys = DataProcessor.process_regression(dir, index, column, target, mode)
    widget.progressBar.setValue(25)
    scaledX, sclr = Scalers.scale(scaler, x)
    widget.progressBar.setValue(50)
    results = DataProcessor.sort_data(regression(models, index=y.index, x=scaledX, y=y.as_matrix().astype(float)))
    widget.progressBar.setValue(75)
    return x, y, results, xKeys, yKeys


def predicting(models, X_plot):
    y_models = []
    for model in models:
        y_models.append(models[model].predict(X_plot))
    y_models = pd.DataFrame(np.array(y_models).T, columns=models)
    return y_models


def regression(models, x, y, p):
    if models:
        a = np.array(x)
        X = np.array([[value] for value in a]).astype(float)
        Y = y.as_matrix().astype(float)
        X_plot = np.linspace(0, p, p)[:, None]
        fitted_models = fitting(models, X, Y)
        y_models = predicting(fitted_models, X_plot)
        return y_models


def cross_validation_regression_models(checks):
    models = {}
    if checks['lsvr']:
        models['lsvr'] = GridSearchCV(LinearSVR(), cv=5, param_grid={})
    if checks['svs']:
        models['svs'] = GridSearchCV(
            SVR(kernel='rbf', gamma='auto'), cv=5,
            param_grid={"C": [1e0], "gamma": np.logspace(-2, 2, 5), "degree": [1, 2]}
        )
    if checks['svr']:
        models['svr'] = GridSearchCV(
            SVR(kernel='sigmoid', gamma='auto'), cv=5,
            param_grid={  # "kernel": ['linear', 'rbf', 'sigmoid'],
                "gamma": np.logspace(-2, 2, 5)
            }
        )
    if checks['kr']:
        models['kr'] = GridSearchCV(
            KernelRidge(kernel='rbf', gamma=0.1), cv=5,
            param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)}
        )
    if checks['dtr']:
        models['dtr'] = GridSearchCV(
            DecisionTreeRegressor(), cv=5,
            param_grid={
                "max_depth":    [2, 5, 20], "min_samples_leaf": [5, 10, 20],
                "max_features": ['auto', 'sqrt', 'log2']
            }
        )
    if checks['nnr']:
        models['nnr'] = GridSearchCV(
            MLPRegressor(), cv=5,
            param_grid={
                "activation": ['identity', 'logistic', 'tanh', 'relu'],
                "solver":     ['lbfgs', 'sgd', 'adam'],
                # "learning_rate": ['constant', 'invscaling', 'adaptive'],
                # "alpha": [0.01, 0.001, 0.0001],
                "max_iter":   [100]
            }, )
    if checks['knn']:
        models['knn'] = GridSearchCV(KNeighborsRegressor(), cv=5, param_grid={})
    if checks['rfr']:
        models['rfr'] = GridSearchCV(
            RandomForestRegressor(n_jobs=-1), cv=5, param_grid={  # "n_estimators": [1, 5],
                "max_features": ['auto', 'sqrt', 'log2'],  # "criterion": ['mse', 'mae']
            }, )
    return models


def regression_models(checks, options):
    models = {}
    if checks['svr']:
        if not options['svrC']:
            options['svrC'] = 1
        else:
            options['svrC'] = float(options['svrC'])
        if not options['svrEpsilon']:
            options['svrEpsilon'] = 0.1
        else:
            options['svrEpsilon'] = float(options['svrEpsilon'])
        if not options['svrGamma']:
            options['svrGamma'] = 'auto'
        else:
            options['svrGamma'] = float(options['svrGamma'])
        if not options['svrCoefficient0']:
            options['svrCoefficient0'] = 0.0
        else:
            options['svrCoefficient0'] = float(options['svrCoefficient0'])
        if not options['svrTolerance']:
            options['svrTolerance'] = 1e-3
        else:
            options['svrTolerance'] = float(options['svrTolerance'])
        if not options['svrCacheSize']:
            options['svrCacheSize'] = 1.0
        else:
            options['svrCacheSize'] = float(options['svrCacheSize'])
        if not options['svrMaximumIterations']:
            options['svrMaximumIterations'] = 1.0 * 100
        else:
            options['svrMaximumIterations'] = float(options['svrMaximumIterations'])
        svr = SVR(
            C=options['svrC'], epsilon=options['svrEpsilon'], kernel=options['svrKernel'],
            degree=int(options['svrPolynomialDegree']), gamma=options['svrGamma'],
            coef0=options['svrCoefficient0'], shrinking=options['svrShrinking'], tol=options['svrTolerance'],
            cache_size=options['svrCacheSize'], max_iter=options['svrMaximumIterations'], )
        models['svr'] = svr
    if checks['kr']:
        models = {}
        if checks['kr']:
            if not options['krAlpha']:
                options['krAlpha'] = 1
            else:
                options['krAlpha'] = float(options['krAlpha'])
            if options['krGamma']:
                options['krGamma'] = float(options['krGamma'])
            options['krPolynomialDegree'] = int(options['krPolynomialDegree'])
            if not options['krCoef0']:
                options['krCoef0'] = 1
            else:
                options['krCoef0'] = float(options['krCoef0'])
        kr = KernelRidge(
            alpha=options['krAlpha'], kernel=options['krKernel'], gamma=options['krGamma'],
            degree=(options['krPolynomialDegree']), coef0=options['krCoef0'], )
        models['kr'] = kr
    if checks['dtr']:
        models = {}
        if checks['dtr']:
            if options['dtrMaxDepth']:
                options['dtrMaxDepth'] = float(options['dtrMaxDepth'])
            if not options['dtrMinSamplesSplit']:
                options['dtMinSamplesSplit'] = 2
            else:
                options['dtrMinSamplesSplit'] = float(options['dtrMinSamplesSplit'])
            if not options['dtrMinSamplesLeaf']:
                options['dtrMinSamplesLeaf'] = 1
            else:
                options['dtrMinSamplesLeaf'] = float(options['dtrMinSamplesLeaf'])
            if not options['dtrMinWeightFractionLeaf']:
                options['dtrMinWeightFractionLeaf'] = 0
            else:
                options['dtrMinWeightFractionLeaf'] = float(options['dtrMinWeightFractionLeaf'])
            if options['dtrMaxFeatures']:
                options['dtrMaxFeatures'] = float(options['dtrMaxFeatures'])
            if options['dtrRandomState']:
                options['dtrRandomState'] = int(options['dtrRandomState'])
            if options['dtrMaxLeafNodes']:
                options['dtrMaxLeafNodes'] = int(options['dtrMaxLeafNodes'])
            if not options['dtrMinImpurityDecrease']:
                options['dtrMinImpurityDecrease'] = 0
            else:
                options['dtrMinImpurityDecrease'] = float(options['dtrMinImpurityDecrease'])
        dtr = DecisionTreeRegressor(
            criterion=options['dtrCriterion'], splitter=options['dtrSplitter'],
            max_depth=options['dtrMaxDepth'], min_samples_split=(options['dtrMinSamplesSplit']),
            min_samples_leaf=options['dtrMinSamplesLeaf'],
            min_weight_fraction_leaf=options['dtrMinWeightFractionLeaf'],
            max_features=options['dtrMaxFeatures'], random_state=options['dtrRandomState'],
            max_leaf_nodes=options['dtrMaxLeafNodes'],
            min_impurity_decrease=options['dtrMinImpurityDecrease'],
            presort=options['dtrPresort'], )
        models['dtr'] = dtr
    if checks['rfr']:
        if not options['rfrNumberOfTrees']:
            options['rfrNumberOfTrees'] = 10
        else:
            options['rfrNumberOfTrees'] = int(options['rfrNumberOfTrees'])
        if options['rfrMaximumDepth']:
            options['rfrMaximumDepth'] = int(options['rfrMaximumDepth'])
        if not options['rfrMinimumSamplesSplit']:
            options['rfrMinimumSamplesSplit'] = 2
        else:
            options['rfrMinimumSamplesSplit'] = float(options['rfrMinimumSamplesSplit'])
        if not options['rfrMinimumSamplesAtLeaf']:
            options['rfrMinimumSamplesAtLeaf'] = 1
        else:
            options['rfrMinimumSamplesAtLeaf'] = float(options['rfrMinimumSamplesAtLeaf'])
        if not options['rfrMinimumSumWeightedFraction']:
            options['rfrMinimumSumWeightedFraction'] = 0
        else:
            options['rfrMinimumSumWeightedFraction'] = float(options['rfrMinimumSumWeightedFraction'])
        if options['rfrMaximumLeafNodes']:
            options['rfrMaximumLeafNodes'] = int(options['rfrMaximumLeafNodes'])
        if options['RandomState']:
            options['RandomState'] = int(options['RandomState'])
        rfr = RandomForestRegressor(
            n_estimators=options['rfrNumberOfTrees'], criterion=options['rfrCriterion'],
            max_features=options['rfrMaximumFeatures'], max_depth=options['rfrMaximumDepth'],
            min_samples_split=options['rfrMinimumSamplesSplit'],
            min_samples_leaf=options['rfrMinimumSamplesAtLeaf'],
            min_weight_fraction_leaf=options['rfrMinimumSumWeightedFraction'],
            max_leaf_nodes=options['rfrMaximumLeafNodes'], bootstrap2=options['rfrBootstrap'],
            oob_score=options['rfrOutOfBagSamples'], random_state=options['RandomState'],
            warm_start=options['rfrWarmStart'], )
        models['rfr'] = rfr
    return models


# TODO mlpr takes too long and doesn't work
def cross_validation_models(checks):
    models = {}
    if 'Adaptive Boost' in checks:
        abr = GridSearchCV(
            AdaBoostRegressor(), cv=5,
            param_grid={
                'learning_rate': (0.5, 1,), 'loss': ('linear', 'square', 'exponential'),
                'n_estimators':  (1, 5),
            }
        )

        models['Adaptive Boost'] = abr

    if 'Decision Tree' in checks:
        dtr = GridSearchCV(
            DecisionTreeRegressor(), cv=5, param_grid={
                'criterion':                ('mse',), 'max_depth': (None,),
                'max_features':             (None, 'auto', 'sqrt', 'log2'),
                'max_leaf_nodes':           (2, 4, 6),
                'min_samples_leaf':         (1, 10, 100, .15, .3, .45),
                # min_samples_split=(options['dtrMinimumSamplesSplit']),
                'min_weight_fraction_leaf': (0,),
                'presort':                  (False,), 'random_state': (None,),
                'splitter':                 ('best', 'random')
            }
        )
        models['Decision Tree'] = dtr

    if 'Elastic Net' in checks:
        enr = GridSearchCV(
            ElasticNet(), cv=5,
            param_grid={
                'alpha':     (0.1, 1), 'fit_intercept': (True, False), 'l1_ratio': (0.25, 0.5),
                'normalize': (True, False), 'positive': (True, False),
                'selection': ('cyclic', 'random'), 'tol': (0.01,), 'warm_start': (True, False)
            }
        )
        models['Elastic Net'] = enr

    if 'Gaussian Process' in checks:
        gpr = GridSearchCV(
            GaussianProcessRegressor(), cv=5,
            param_grid={'alpha': (0.1, 0.5), 'normalize_y': (True, False)}
        )
        models['Gaussian Process'] = gpr

    if 'Nearest Neighbors' in checks:
        knnr = GridSearchCV(
            KNeighborsRegressor(), cv=5,
            param_grid={
                'algorithm':   ('ball_tree', 'kd_tree', 'brute', 'auto',), 'leaf_size': (30,),
                'n_neighbors': (5,), 'p': (2,), 'weights': ('uniform', 'distance'),
            }
        )
        models['Nearest Neighbors'] = knnr

    if 'Kernel Ridge' in checks:
        krr = GridSearchCV(
            KernelRidge(), cv=5,
            param_grid={
                'alpha':  (3,), 'coef0': (0,), 'degree': (1, 2, 3), 'gamma': (.25, .5, .75),
                'kernel': ('linear', 'rbf', 'poly')
            }
        )
        models['Kernel Ridge'] = krr

    if 'Multilayer Perceptron' in checks:
        mlpr = GridSearchCV(
            MLPRegressor(), cv=5,
            param_grid={
                'activation':          ('identity', 'logistic', 'tanh', 'relu'),  # 'alpha': (0.0001,),
                # 'beta_1': (0.5,),
                # 'beta_2': (0.999,),
                # 'early_stopping':(True,False),
                # 'epsilon': (0.00001,),
                # 'hidden_layer_sizes': (100,),
                # 'learning_rate':('constant','invscaling','adaptive'),
                # 'learning_rate_init':(0.001,),
                # 'momentum': (0.5, 0.5),
                # 'nesterovs_momentum':(True,False),
                # 'power_t': (0.5, 0.1),
                'solver':              ('lbfgs', 'sgd', 'adam'),  # 'shuffle':(True,False),
                # 'tol':(0.01,),
                'validation_fraction': (0.5, 0.9),  # 'warm_start':(True,False),
            }
        )
        models['Multilayer Perceptron'] = mlpr

    if 'Random Forest' in checks:
        rfr = GridSearchCV(
            RandomForestRegressor(n_jobs=-1), cv=5,
            param_grid={
                'bootstrap2':        (True,), 'criterion': ('mse',), 'max_features': ('auto',),
                'max_depth':        (None,), 'max_leaf_nodes': (None,), 'min_samples_split': (2,),
                'min_samples_leaf': (1,), 'min_weight_fraction_leaf': (0,),
                'n_estimators':     (10,), 'oob_score': (False,), 'warm_start': (False,)
            }
        )
        models['Random Forest'] = rfr

    if 'Stochastic Gradient Descent' in checks:
        sgdr = GridSearchCV(
            SGDRegressor(), cv=5, param_grid={
                'average': (True, False), 'fit_intercept': (True, False),
                'loss':    ('squared_loss', 'huber', 'epsilon_insensitive',
                            'squared_epsilon_insensitive'),
                'penalty': ('none', 'l1', 'l2', 'elasticnet'),
                'power_t': (0.1, 0.5)  # 'alpha': (),
                # 'L1_ratio': (),
                # 'max_iter': (),
                # 'tol': (),
                # 'shuffle': (True, False),
                # 'learning_rate': (0.1, 1),
                # 'eta0': (),
                # 'warm_start': (True, False),
            }
        )
        models['Stochastic Gradient Descent'] = sgdr

    if 'Support Vector Machine' in checks:
        svmr = GridSearchCV(
            SVR(), cv=5, param_grid={  # 'C': (1, 100),
                # 'coef0': (0, 1),
                # 'degree': (1, 2),
                # 'epsilon': (0.1, 0.5),
                # 'gamma': ('auto',),
                'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),  # 'max_iter': (1, 10, 20,),
                # 'shrinking': (True, False),
                # 'tol': (1e-2,),
            }
        )
        models['Support Vector Machine'] = svmr

    return models


def models(checks, options):
    models = {}

    if 'Adaptive Boost' in checks:
        settings = options['Adaptive Boost']
        abr = AdaBoostRegressor(**settings)
        models['Adaptive Boost'] = abr

    if 'Decision Tree' in checks:
        settings = options['Decision Tree']
        if settings['max_depth'] == 0:
            settings['max_depth'] = None

        if settings['max_leaf_nodes'] == 0:
            settings['max_leaf_nodes'] = None

        dtr = DecisionTreeRegressor(**settings)
        models['Decision Tree'] = dtr

    if 'Elastic Net' in checks:
        settings = options['Elastic Net']
        enr = ElasticNet(**settings)
        models['Elastic Net'] = enr

    if 'Gaussian Process' in checks:
        settings = options['Gaussian Process']
        if settings['alpha'] == 0:
            settings['alpha'] = 1e-10
        else:
            settings['alpha'] = float(settings['alpha'])

        gpr = GaussianProcessRegressor(**settings)
        models['Gaussian Process'] = gpr

    if 'Nearest Neighbors' in checks:
        settings = options['Nearest Neighbors']
        knnr = KNeighborsRegressor(**settings)
        models['Nearest Neighbors'] = knnr

    if 'Kernel Ridge' in checks:
        settings = options['Kernel Ridge']
        if settings['gamma'] == 0:
            settings['gamma'] = None

        settings['degree'] = int(settings['degree'])

        krr = KernelRidge(**settings)

        models['Kernel Ridge'] = krr

    if 'Multilayer Perceptron' in checks:
        settings = options['Multilayer Perceptron']
        if not settings['hidden_layer_sizes']:
            settings['hidden_layer_sizes'] = (100,)
        else:
            settings['hidden_layer_sizes'] = int(settings['hidden_layer_sizes'])

        mlpr = MLPRegressor(**settings)
        models['Multilayer Perceptron'] = mlpr

    if 'Random Forest' in checks:
        settings = options['Random Forest']
        rfr = RandomForestRegressor(**settings)
        models['Random Forest'] = rfr

    if 'Stochastic Gradient Descent' in checks:
        settings = options['Stochastic Gradient Descent']
        if settings['tol'] == 0:
            settings['tol'] = None

        sgdr = SGDRegressor(**settings)
        models['Stochastic Gradient Descent'] = sgdr

    if 'Support Vector Machine' in checks:
        settings = options['Support Vector Machine']
        settings['degree'] = int(settings['degree'])

        if settings['gamma'] == 0:
            settings['gamma'] = 'auto'

        svmr = SVR(**settings)
        models['Support Vector Machine'] = svmr

    return models


def run(dir, index, column, models, scaler, mode):
    data, xKeys, yKeys = dh.process_regression_data(dir, index, column, mode)
    if scaler != None:
        data = pd.DataFrame(data.values, index=data.index)
        data.reset_index(inplace=True)
        scaled = (scaler.fit_transform((data.as_matrix())))
        data = pd.DataFrame(scaled)
        data.set_index(0, drop=True, inplace=True)
        data = data.iloc[:, 0]
        print(data)
    results = dh.sort_data(regression(models, x=data.index, y=data))
    data = pd.DataFrame(data, columns=(column,), index=data.index)
    x = ((data.index, data.index),)
    y = ((results, data),)
    # y = ((data, results),)
    return x, y, xKeys, yKeys


def generate_sample_data():
    # Generate sample data
    X = 5 * rng.rand(500, 1)
    y = np.sin(X).ravel()
    # Add noise to targets
    noise = 10
    y[::noise] += 3 * (0.5 - rng.rand(X.shape[0] // noise))
    X_plot = np.linspace(0, 5, 10000)[:, None]
    return X, y, X_plot


def generate_data(path, features, step=1):
    X = pd.read_csv(path, dtype='str')
    clean_X = dh.clean_data(X)[::step]
    clean_X.fillna(0, inplace=True)
    X = pd.DataFrame(clean_X[features[0]]).as_matrix().astype(float)
    y = clean_X[features[1]].astype(float).values
    X_plot = np.linspace(0, max(X)[0], 100)[:, None]
    return X, y, X_plot


def predicting(models, X_plot):
    prediction_time = []
    y_models = []
    for i, model in enumerate(models):
        t0 = time.time()
        y_models.append(models[model].predict(X_plot))
        prediction_time.append(time.time() - t0)
    return y_models, prediction_time


def regression(models, x, y):
    if models:
        X, y, X_plot = generate_sample_data()  # generate_data(path, (x,y))
        fitted_models, fit_time = fitting(models, X, y)
        y_models, prediction_time = predicting(fitted_models, X_plot)
        return y_models  # plot_stuff(models, y_models, features, fit_time, prediction_time)


"""def plot_stuff(models, y_models, features, fit_time, predict_time):
    colors = ('r', 'g', 'b', 'y', 'c', 'm')
    # Look at the results
    sv_ind = models['SVR'].best_estimator_.support_
    plt.scatter(X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors',
                zorder=1, edgecolors=(0, 0, 0))
    plt.scatter(X, y, c='k', label='data', zorder=1,
                edgecolors=(0, 0, 0))

    for i, model, y_model in zip(range(len(models)), models, y_models):
        plt.plot0(X_plot, y_model, c=colors[i], label=model + ' (fit: %.3fs, predict: %.3fs)' % (fit_time[i], predict_time[i]))

    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('Regression6 Comparison')
    plt.legend()
    plt.show()"""


# TODO mlpr takes too long and doesn't work
def cross_validation_regression_models(checks):
    models = {}
    if checks['Adaptive Boost']:
        abr = GridSearchCV(
            AdaBoostRegressor(), cv=5,
            param_grid={
                'learning_rate': (0.5, 1,), 'loss': ('linear', 'square', 'exponential'),
                'n_estimators':  (1, 5),  # 'random_state': (None,)
            }
        )

        models['Adaptive Boost'] = abr

    if checks['Decision Tree']:
        dtr = GridSearchCV(
            DecisionTreeRegressor(), cv=5, param_grid={
                'criterion':                ('mse',), 'max_depth': (None,),
                'max_features':             (None, 'auto', 'sqrt', 'log2'),
                'max_leaf_nodes':           (2, 4, 6),
                'min_samples_leaf':         (1, 10, 100, .15, .3, .45),
                # min_samples_split=(options['dtrMinimumSamplesSplit']),
                'min_weight_fraction_leaf': (0,),
                'presort':                  (False,), 'random_state': (None,),
                'splitter':                 ('best', 'random')
            }
        )
        models['Decision Tree'] = dtr

    if checks['Elastic Net']:
        enr = GridSearchCV(
            ElasticNet(), cv=5,
            param_grid={
                'alpha':     (0.1, 1), 'fit_intercept': (True, False), 'l1_ratio': (0.25, 0.5),
                'normalize': (True, False), 'positive': (True, False),
                # 'random_state': (None,),
                'selection': ('cyclic', 'random'), 'tol': (0.01,), 'warm_start': (True, False)
            }
        )
        models['Elastic Net'] = enr

    if checks['Gaussian Process']:
        gpr = GridSearchCV(
            GaussianProcessRegressor(), cv=5,
            param_grid={'alpha': (0.1, 0.5), 'normalize_y': (True, False)}
        )
        models['Gaussian Process'] = gpr

    if checks['Nearest Neighbors']:
        knnr = GridSearchCV(
            KNeighborsRegressor(), cv=5,
            param_grid={
                'algorithm':   ('ball_tree', 'kd_tree', 'brute', 'auto',), 'leaf_size': (30,),
                'n_neighbors': (5,), 'p': (2,), 'weights': ('uniform', 'distance'),
            }
        )
        models['Nearest Neighbors'] = knnr

    if checks['Kernel Ridge']:
        krr = GridSearchCV(
            KernelRidge(), cv=5,
            param_grid={
                'alpha':  (3,), 'coef0': (0,), 'degree': (1, 2, 3), 'gamma': (.25, .5, .75),
                'kernel': ('linear', 'rbf', 'poly')
            }
        )
        models['Kernel Ridge'] = krr

    if checks['Multilayer Perceptron']:
        mlpr = GridSearchCV(
            MLPRegressor(), cv=5,
            param_grid={
                'activation':          ('identity', 'logistic', 'tanh', 'relu'), 'alpha': (0.0001,),
                'beta_1':              (0.9,), 'beta_2': (0.999,),  # 'early_stopping':(True,False),
                'epsilon':             (0.00001,), 'hidden_layer_sizes': (100,),
                # 'learning_rate':('constant','invscaling','adaptive'),
                # 'learning_rate_init':(0.001,),
                'momentum':            (0.9, 0.5),  # 'nesterovs_momentum':(True,False),
                'power_t':             (0.5, 0.1), 'solver': ('lbfgs', 'sgd', 'adam'),
                # 'shuffle':(True,False),
                # 'tol':(0.01,),
                'validation_fraction': (0.1, 0.5),  # 'warm_start':(True,False),
            }
        )
        models['Multilayer Perceptron'] = mlpr

    if checks['Random Forest']:
        rfr = GridSearchCV(
            RandomForestRegressor(n_jobs=-1), cv=5,
            param_grid={
                'bootstrap2':        (True,), 'criterion': ('mse',), 'max_features': ('auto',),
                'max_depth':        (None,), 'max_leaf_nodes': (None,), 'min_samples_split': (2,),
                'min_samples_leaf': (1,), 'min_weight_fraction_leaf': (0,),
                'n_estimators':     (10,), 'oob_score': (False,), 'random_state': (None,),
                'warm_start':       (False,)
            }
        )
        models['Random Forest'] = rfr

    if checks['Stochastic Gradient Descent']:
        sgdr = GridSearchCV(
            SGDRegressor(), cv=5, param_grid={
                'average': (True, False), 'fit_intercept': (True, False),
                'loss':    ('squared_loss', 'huber', 'epsilon_insensitive',
                            'squared_epsilon_insensitive'),
                'penalty': ('none', 'l1', 'l2', 'elasticnet'),
                'power_t': (0.1, 0.5)  # 'alpha': (),
                # 'L1_ratio': (),
                # 'max_iter': (),
                # 'tol': (),
                # 'shuffle': (True, False),
                # 'learning_rate': (0.1, 1),
                # 'eta0': (),
                # 'warm_start': (True, False),
            }
        )
        models['Stochastic Gradient Descent'] = sgdr

    if checks['Support Vector Machine']:
        svr = GridSearchCV(
            SVR(), cv=5, param_grid={  # 'C': (1, 100),
                # 'coef0': (0, 1),
                # 'degree': (1, 2),
                # 'epsilon': (0.1, 0.5),
                # 'gamma': ('auto',),
                'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),  # 'max_iter': (1, 10, 20,),
                # 'shrinking': (True, False),
                # 'tol': (1e-2,),
            }
        )
        models['Support Vector Machine'] = svr

    return models


def run(widget, dir, index, column, target, models, scaler, mode):
    x, y, xKeys, yKeys = dh.process_regression(dir, index, column, target, mode)
    widget.progressBar.setValue(25)
    scaledX, sclr = Scalers.scale(scaler, x)
    widget.progressBar.setValue(50)
    results = dh.sort_data(regression(models, index=y.index, x=scaledX, y=y.as_matrix().astype(float)))
    widget.progressBar.setValue(75)
    return x, y, results, xKeys, yKeys


def plot_regression(data, results, xLabel, yLabel, xKeys, yKeys, file, title=None):
    x = ((data.index, data.index),)
    y = ((data, results),)
    make_plot(
        file, X=x, Y=y, titles=(title,), xLabels=(xLabel,), yLabels=(yLabel,), lineWidths=((1, 1),),
        legends=((True, True),), types=(('scatter', 'line'),), xKeys=(xKeys,), yKeys=(yKeys,),
        colors=(((CONTRASTED_COLOR,), (None,)),)
    )


def regression(models, x, y, p=None):
    if models:
        a = np.array(x)
        X = np.array([[value] for value in a]).astype(float)
        print(X)
        print(y.as_matrix().astype(float))
        fitted_models = fitting(models, X, y.as_matrix().astype(float))
        results = predicting(fitted_models, np.linspace(0, p, p)[:, None] if p else X, x)
        return results


def plot_regression(x, y, xLabel, yLabel, xKeys, yKeys):
    make_plot(
        REGRESSION_GRAPH, X=x, Y=y, titles=('Regression6',), xLabels=(xLabel,), yLabels=(yLabel,),
        lineWidths=((1, 1),), legends=((True, True),), types=(('scatter', 'line'),), xKeys=(xKeys,),
        yKeys=(yKeys,), colors=(((CONTRASTED_COLOR,), (None,)),)
    )


def regression_models(checks, options):
    models = {}

    if 'Adaptive Boost' in checks:
        abr = AdaBoostRegressor(
            learning_rate=options['abrLearningRate'], loss=options['abrLoss'],
            n_estimators=options['abrEstimators']
        )
        # random_state=options['abrRandomState']
        models['Adaptive Boost'] = abr

    if 'Decision Tree' in checks:
        if options['dtrMaximumDepth'] == 0:
            options['dtrMaximumDepth'] = None

        if options['dtrMaximumLeafNodes'] == 0:
            options['dtrMaximumLeafNodes'] = None

        options['dtrRandomState'] = int(options['dtrRandomState']) if options['dtrRandomState'] else None

        dtr = DecisionTreeRegressor(
            criterion=options['dtrCriterion'], max_depth=options['dtrMaximumDepth'],
            max_features=options['dtrMaximumFeatures'],
            max_leaf_nodes=options['dtrMaximumLeafNodes'],
            min_samples_leaf=options['dtrMinimumSamplesLeaf'],
            # min_samples_split=(options['dtrMinimumSamplesSplit']),
            min_weight_fraction_leaf=options['dtrMinimumWeightFractionLeaf'],
            random_state=options['dtrRandomState'], presort=options['dtrPresort'],
            splitter=options['dtrSplitter']
        )
        models['Decision Tree'] = dtr

    if 'Elastic Net' in checks:
        enr = ElasticNet(
            alpha=options['enrAlpha'],  # fit_intercept=options['enrFitIntercept'],
            l1_ratio=options['enrL1Ratio'], normalize=options['enrNormalize'],
            positive=options['enrPositive'],  # random_state=(None,),
            selection=options['enrSelection'], tol=options['enrTolerance'],
            warm_start=options['enrWarmStart']
        )
        models['Elastic Net'] = enr

    if 'Gaussian Process' in checks:
        if not options['gprAlpha']:
            options['gprAlpha'] = 1e-10
        else:
            options['gprAlpha'] = float(options['gprAlpha'])

        gpr = GaussianProcessRegressor(
            alpha=options['gprAlpha'],  # 'gprKernel': self.GPkernelCombo.currentText(),
            # 'gprOptimizer': self.GPoptimizerCombo.currentText(),
            normalize_y=options['gprNormalize'], )
        models['Gaussian Process'] = gpr

    if 'Nearest Neighbors' in checks:
        knnr = KNeighborsRegressor(
            algorithm=options['knnrAlgorithm'], leaf_size=int(options['knnrLeafSize']),
            n_neighbors=options['knnrNumberOfNeighbors'], p=options['knnrMinkowskiPower'],
            weights=options['knnrWeightsFunction']
        )
        models['Nearest Neighbors'] = knnr

    if 'Kernel Ridge' in checks:
        if options['krrGamma'] == 0:
            options['krrGamma'] = None

        options['krrPolynomialDegree'] = int(options['krrPolynomialDegree'])

        krr = KernelRidge(
            alpha=options['krrAlpha'], coef0=options['krrCoefficient0'],
            degree=(options['krrPolynomialDegree']), gamma=options['krrGamma'],
            kernel=options['krrKernel']
        )

        models['Kernel Ridge'] = krr

    if 'Multilayer Perceptron' in checks:
        if not options['mlprHiddenLayerSizes']:
            options['mlprHiddenLayerSizes'] = (100,)
        else:
            options['mlprHiddenLayerSizes'] = int(options['mlprHiddenLayerSizes'])

        options['mlprRandomState'] = int(options['mlprRandomState']) if options['mlprRandomState'] else None

        mlpr = MLPRegressor(
            activation=options['mlprActivationFunction'], alpha=options['mlprPenaltyParameter'],
            # batch_size=options['mlprBatchSize'],
            beta_1=options['mlprFirstMomentExponentialDecay'],
            beta_2=options['mlprSecondMomentExponentialDecay'],
            early_stopping=options['mlprEarlyStopping'], epsilon=options['mlprNumericalStability'],
            hidden_layer_sizes=options['mlprHiddenLayerSizes'],
            learning_rate=options['mlprLearningRate'],
            learning_rate_init=options['mlprInitialLearningRate'],
            max_iter=options['mlprMaximumIterations'], momentum=options['mlprMomentum'],
            nesterovs_momentum=options['mlprNesterovsMomentum'],
            power_t=options['mlprPowerForInverseLearningRate'], random_state=options['mlprRandomState'],
            shuffle=options['mlprShuffle'], solver=options['mlprWeightOptimizationSolver'],
            tol=options['mlprTolerance'], validation_fraction=options['mlprValidationFraction'],
            warm_start=options['mlprWarmStart']
        )
        models['Multilayer Perceptron'] = mlpr

    if 'Random Forest' in checks:
        if options['rfrMaximumDepth'] == 0:
            options['rfrMaximumDepth'] = None

        if options['rfrMaximumLeafNodes'] == 0:
            options['rfrMaximumLeafNodes'] = None

        options['rfrRandomState'] = int(options['rfrRandomState']) if options['rfrRandomState'] else None

        rfr = RandomForestRegressor(
            bootstrap2=options['rfrBootstrap'], criterion=options['rfrCriterion'],
            max_depth=options['rfrMaximumDepth'], max_features=options['rfrMaximumFeatures'],
            max_leaf_nodes=options['rfrMaximumLeafNodes'],
            min_samples_leaf=options['rfrMinimumSamplesAtLeaf'],
            min_samples_split=options['rfrMinimumSamplesSplit'],
            min_weight_fraction_leaf=options['rfrMinimumSumWeightedFraction'],
            n_estimators=options['rfrNumberOfTrees'], oob_score=options['rfrOutOfBagSamples'],
            random_state=options['rfrRandomState'], warm_start=options['rfrWarmStart'], )

        models['Random Forest'] = rfr

    if 'Stochastic Gradient Descent' in checks:
        if options['sgdrTolerance'] == 0:
            options['sgdrTolerance'] = None

        sgdr = SGDRegressor(
            alpha=options['sgdrAlpha'], average=options['sgdrAverage'], eta0=options['sgdrEta0'],
            fit_intercept=options['sgdrFitIntercept'], learning_rate=options['sgdrLearningRate'],
            loss=options['sgdrLoss'], l1_ratio=options['sgdrL1Ratio'],
            max_iter=options['sgdrMaxIterations'], penalty=options['sgdrPenalty'],
            power_t=options['sgdrPowerT'], shuffle=options['sgdrShuffle'], tol=options['sgdrTolerance'],
            warm_start=options['sgdrWarmStart']
        )
        models['Stochastic Gradient Descent'] = sgdr

    if 'Support Vector Machine' in checks:
        options['svmrPolynomialDegree'] = int(options['svmrPolynomialDegree'])

        if options['svmrGamma'] == 0:
            options['svmrGamma'] = 'auto'

        svmr = SVR(
            C=options['svmrC'], epsilon=options['svmrEpsilon'], kernel=options['svmrKernel'],
            degree=options['svmrPolynomialDegree'], gamma=options['svmrGamma'],
            coef0=options['svmrCoefficient0'], shrinking=options['svmrShrinking'], tol=options['svmrTolerance'],
            # cache_size=options['svmrCacheSize'],
            max_iter=options['svmrMaximumIterations']
        )
        models['Support Vector Machine'] = svmr

    return models


# TODO mlpr takes too long and doesn't work
def cross_validation_models(models):
    _models = {}
    if 'Adaptive Boost' in models:
        _models['Adaptive Boost'] = GridSearchCV(
            AdaBoostRegressor(), cv=5, param_grid={
                'learning_rate': (0.5, 1,),
                'loss':          ('linear', 'square',
                                  'exponential'),
                'n_estimators':  (1, 5),
            }
        )

    if 'Decision Tree' in models:
        _models['Decision Tree'] = GridSearchCV(
            DecisionTreeRegressor(), cv=5,
            param_grid={
                'criterion':                ('mse',), 'max_depth': (None,),
                'max_features':             (None, 'auto', 'sqrt', 'log2'),
                'max_leaf_nodes':           (2, 4, 6),
                'min_samples_leaf':         (1, 10, 100, .15, .3, .45),
                # min_samples_split=(options['dtrMinimumSamplesSplit']),
                'min_weight_fraction_leaf': (0,), 'presort': (False,),
                'random_state':             (None,), 'splitter': ('best', 'random')
            }
        )

    if 'Elastic Net' in models:
        _models['Elastic Net'] = GridSearchCV(
            ElasticNet(), cv=5,
            param_grid={
                'alpha':    (0.1, 1), 'fit_intercept': (True, False),
                'l1_ratio': (0.25, 0.5), 'normalize': (True, False),
                'positive': (True, False), 'selection': ('cyclic', 'random'),
                'tol':      (0.01,), 'warm_start': (True, False)
            }
        )

    if 'Gaussian Process' in models:
        _models['Gaussian Process'] = GridSearchCV(
            GaussianProcessRegressor(), cv=5,
            param_grid={'alpha': (0.1, 0.5), 'normalize_y': (True, False)}
        )

    if 'Nearest Neighbors' in models:
        _models['Nearest Neighbors'] = GridSearchCV(
            KNeighborsRegressor(), cv=5,
            param_grid={
                'algorithm': ('ball_tree', 'kd_tree', 'brute', 'auto',),
                'leaf_size': (30,), 'n_neighbors': (5,), 'p': (2,),
                'weights':   ('uniform', 'distance'),
            }
        )

    if 'Kernel Ridge' in models:
        _models['Kernel Ridge'] = GridSearchCV(
            KernelRidge(), cv=5,
            param_grid={
                'alpha':  (3,), 'coef0': (0,), 'degree': (1, 2, 3),
                'gamma':  (.25, .5, .75),
                'kernel': ('linear', 'rbf', 'poly')
            }
        )

    if 'Multilayer Perceptron' in models:
        _models['Multilayer Perceptron'] = GridSearchCV(
            MLPRegressor(), cv=5, param_grid={
                'activation':          ('identity', 'logistic', 'tanh', 'relu'),  # 'alpha': (0.0001,),
                # 'beta_1': (0.5,),
                # 'beta_2': (0.999,),
                # 'early_stopping':(True,False),
                # 'epsilon': (0.00001,),
                # 'hidden_layer_sizes': (100,),
                # 'learning_rate':('constant','invscaling','adaptive'),
                # 'learning_rate_init':(0.001,),
                # 'momentum': (0.5, 0.5),
                # 'nesterovs_momentum':(True,False),
                # 'power_t': (0.5, 0.1),
                'solver':              ('lbfgs', 'sgd', 'adam'),  # 'shuffle':(True,False),
                # 'tol':(0.01,),
                'validation_fraction': (0.5, 0.9),  # 'warm_start':(True,False),
            }
        )

    if 'Random Forest' in models:
        rfr = GridSearchCV(
            RandomForestRegressor(n_jobs=-1), cv=5,
            param_grid={
                'bootstrap2':        (True,), 'criterion': ('mse',), 'max_features': ('auto',),
                'max_depth':        (None,), 'max_leaf_nodes': (None,), 'min_samples_split': (2,),
                'min_samples_leaf': (1,), 'min_weight_fraction_leaf': (0,),
                'n_estimators':     (10,), 'oob_score': (False,), 'warm_start': (False,)
            }
        )
        _models['Random Forest'] = rfr

    if 'Stochastic Gradient Descent' in models:
        _models['Stochastic Gradient Descent'] = GridSearchCV(
            SGDRegressor(), cv=5,
            param_grid={
                'average':       (True, False),
                'fit_intercept': (True, False), 'loss': (
                    'squared_loss', 'huber', 'epsilon_insensitive',
                    'squared_epsilon_insensitive'),
                'penalty':       ('none', 'l1', 'l2', 'elasticnet'),
                'power_t':       (0.1, 0.5),  # 'alpha': (),
                # 'L1_ratio': (),
                'max_iter':      (1000,), 'tol': (1e-3,),
                # 'shuffle': (True, False),
                # 'learning_rate': (0.1, 1),
                # 'eta0': (),
                # 'warm_start': (True, False),
            }
        )

    if 'Support Vector Machine' in models:
        _models['Support Vector Machine'] = GridSearchCV(
            SVR(), cv=5, param_grid={  # 'C': (1, 100),
                # 'coef0': (0, 1),
                # 'degree': (1, 2),
                # 'epsilon': (0.1, 0.5),
                # 'gamma': ('auto',),
                'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),  # 'max_iter': (1, 10, 20,),
                # 'shrinking': (True, False),
                # 'tol': (1e-2,),
            }
        )

    return _models


def regression(models, index, x, y, p=None):
    if models:
        fittedModels = MachineLearner.fit(models, x, y)
        results = MachineLearner.predict(fittedModels, np.linspace(0, p, p)[:, None] if p else x, index)
        return results


def adjust_model_settings(models, options):
    """
    Model Settings Preprocessing Utility Function
    Adjusts the settings for each model to avoid exceptions, also handles special cases.
    Returns a version of each model in models that has the corresponding settings set.
    models: list of str
    options: dict of dict, outer dict key is model name, inner dict is settings for corresponding model
    return:
    """
    if 'Decision Tree' in models:
        settings = options['Decision Tree']
        if settings['max_depth'] == 0:
            settings['max_depth'] = None
        if settings['max_leaf_nodes'] == 0:
            settings['max_leaf_nodes'] = None

    if 'Gaussian Process' in models:
        settings = options['Gaussian Process']
        if settings['alpha'] == 0:
            settings['alpha'] = 1e-10
        else:
            settings['alpha'] = float(settings['alpha'])

    if 'Kernel Ridge' in models:
        settings = options['Kernel Ridge']
        if settings['gamma'] == 0:
            settings['gamma'] = None
        settings['degree'] = int(settings['degree'])

    if 'Multilayer Perceptron' in models:
        settings = options['Multilayer Perceptron']
        if not settings['hidden_layer_sizes']:
            settings['hidden_layer_sizes'] = (100,)
        else:
            settings['hidden_layer_sizes'] = int(settings['hidden_layer_sizes'])

    if 'Stochastic Gradient Descent' in models:
        settings = options['Stochastic Gradient Descent']
        if settings['tol'] == 0:
            settings['tol'] = None

    if 'Support Vector Machine' in models:
        settings = options['Support Vector Machine']
        settings['degree'] = int(settings['degree'])

        if settings['gamma'] == 0:
            settings['gamma'] = 'auto'

    _models = MachineLearner.adjust_settings(algorithms=REGRESSORS, models=models, settings=options)
    return _models


def run(dir, columns, index, target, models, scaler='None', mode='series'):
    """
    dir: str, path to directory where to look for data
    index: str, name of column to sort data with respect to
    column: list of str, names of columns to be used in Regression6,
            all columns will be training inputs except for the
            column designated in the target argument
    target: str, name of column to use as training target
    models: list of str, names of Regression6 models to use
    scaler: str, name of scaler to use:
            'None', 'MinMax', 'MaxAbs', 'QuantileTransformer', 'Robust', 'Standard'
    mode: str, 'series' or 'parallel'
    return:
    """
    x, y, xKeys, yKeys = DataProcessor.process_regression(dir, index, columns, target, mode)
    scaledX, _scaler = Scalers.scale(scaler, x)
    results = DataProcessor.sort_data(regression(models, index=y.index, x=scaledX, y=y.as_matrix().astype(float)))
    return x, y, results, xKeys, yKeys


"""
EXAMPLE
x = run(r'Z:\Family\LoPar Technologies\LoParDataSets\CollectedData\Iris',
        ['Id', 'SepalLengthCm'],
        'Id',
        'SepalLengthCm',
        {'Adaptive Boost': AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear',
                                             n_estimators=50, random_state=None)},
        None,
        'series')

print(x[1])
"""


def plot_regression(data, results, xLabel, yLabel, xKeys, yKeys, file, title=None):
    x = ((data.index, data.index),)
    y = ((data, results),)
    DynamicPlotGenerator.make_plot(
        file, X=x, Y=y, titles=(title,), xLabels=(xLabel,), yLabels=(yLabel,),
        lineWidths=((1, 1),), legends=((True, True),), types=(('scatter', 'line'),),
        xKeys=(xKeys,), yKeys=(yKeys,), colors=(((Colors.CONTRASTED_COLOR,), (None,)),)
    )


"""
EXAMPLE
x = run(r'Z:\Family\LoPar Technologies\LoParDataSets\CollectedData\Iris',
        ['Id', 'SepalLengthCm'],
        'Id',
        'SepalLengthCm',
        {'Adaptive Boost': AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear',
                                             n_estimators=50, random_state=None)},
        None,
        'series')

print(x[1])
"""


def plot_regression(data, results, xLabel, yLabel, xKeys, yKeys, file, title=None):
    x = ((data.index, data.index),)
    y = ((data, results),)
    DynamicPlotGenerator.make_plot(
        file, X=x, Y=y, titles=(title,), xLabels=(xLabel,), yLabels=(yLabel,),
        lineWidths=((1, 1),), legends=((True, True),), types=(('scatter', 'line'),),
        xKeys=(xKeys,), yKeys=(yKeys,), colors=(((CONTRASTED_COLOR,), (None,)),)
    )


def regression(models, index, x, y, p=None):
    if models:
        fitted_models = fitting(models, x, y)
        results = predicting(fitted_models, np.linspace(0, p, p)[:, None] if p else x, index)
        return results


# CLASSIFICATION
X = pd.read_csv(DATABASE, usecols=('G1', 'G2', 'G3'))
X.fillna(0, inplace=True)
y = X['G3']
X.drop(['G3'], axis=1, inplace=True)
X = pd.read_csv(DATABASE, usecols=('G1', 'G2', 'G3'))
X.fillna(0, inplace=True)
y = X['G3']
X.drop(['G3'], axis=1, inplace=True)

CLASSIFIERS = {
    'Adaptive Boost':              AdaBoostClassifier, 'Decision Tree': DecisionTreeClassifier,
    'Gaussian Process':            GaussianProcessClassifier, 'Naive Bayes': GaussianNB,
    'Nearest Neighbors':           KNeighborsClassifier, 'Multilayer Perceptron': MLPClassifier,
    'Quadratic Discriminant':      QuadraticDiscriminantAnalysis, 'Random Forest': RandomForestClassifier,
    'Stochastic Gradient Descent': SGDClassifier, 'Support Vector Machine': SVC
}


# TODO mlpc takes too long and doesn't work
def cross_validation_models(checks):
    models = {}
    if 'Adaptive Boost' in checks:
        abc = GridSearchCV(
            AdaBoostClassifier(), cv=5,
            param_grid={
                'learning_rate': (0.5, 1,), 'loss': ('linear', 'square', 'exponential'),
                'n_estimators':  (1, 5),  # 'random_state': (None,)
            }
        )

        models['Adaptive Boost'] = abc

    if 'Decision Tree' in checks:
        dtc = GridSearchCV(
            DecisionTreeClassifier(), cv=5, param_grid={
                'criterion':                ('mse',), 'max_depth': (None,),
                'max_features':             (None, 'auto', 'sqrt', 'log2'),
                'max_leaf_nodes':           (2, 4, 6),
                'min_samples_leaf':         (1, 10, 100, .15, .3, .45),
                # min_samples_split=(options['dtcMinimumSamplesSplit']),
                'min_weight_fraction_leaf': (0,),
                'presort':                  (False,), 'random_state': (None,),
                'splitter':                 ('best', 'random')
            }
        )
        models['Decision Tree'] = dtc

    if 'Gaussian Process' in checks:
        gpc = GridSearchCV(
            GaussianProcessClassifier(), cv=5,
            param_grid={'alpha': (0.1, 0.5), 'normalize_y': (True, False)}
        )
        models['Gaussian Process'] = gpc

    if 'Nearest Neighbors' in checks:
        knnc = GridSearchCV(
            KNeighborsClassifier(), cv=5,
            param_grid={
                'algorithm':   ('ball_tree', 'kd_tree', 'brute', 'auto',), 'leaf_size': (30,),
                'n_neighbors': (5,), 'p': (2,), 'weights': ('uniform', 'distance'),
            }
        )
        models['Nearest Neighbors'] = knnc

    if 'Multilayer Perceptron' in checks:
        mlpc = GridSearchCV(
            MLPClassifier(), cv=5,
            param_grid={
                'activation':          ('identity', 'logistic', 'tanh', 'relu'), 'alpha': (0.0001,),
                'beta_1':              (0.9,), 'beta_2': (0.999,),  # 'early_stopping':(True,False),
                'epsilon':             (0.00001,), 'hidden_layer_sizes': (100,),
                # 'learning_rate':('constant','invscaling','adaptive'),
                # 'learning_rate_init':(0.001,),
                'momentum':            (0.9, 0.5),  # 'nesterovs_momentum':(True,False),
                'power_t':             (0.5, 0.1), 'solver': ('lbfgs', 'sgd', 'adam'),
                # 'shuffle':(True,False),
                # 'tol':(0.01,),
                'validation_fraction': (0.1, 0.5),  # 'warm_start':(True,False),
            }
        )
        models['Multilayer Perceptron'] = mlpc

    if 'Naive Bayes' in checks:
        nbc = GridSearchCV(
            GaussianNB(), cv=5, param_grid={

            }
        )
        models['Naive Bayes'] = nbc

    if 'Quadratic Discriminant' in checks:
        krr = GridSearchCV(
            QuadraticDiscriminantAnalysis(), cv=5, param_grid={

            }
        )
        models['Quadratic Discriminant'] = krr

    if 'Random Forest' in checks:
        rfc = GridSearchCV(
            RandomForestClassifier(n_jobs=-1), cv=5,
            param_grid={
                'bootstrap2':        (True,), 'criterion': ('mse',), 'max_features': ('auto',),
                'max_depth':        (None,), 'max_leaf_nodes': (None,), 'min_samples_split': (2,),
                'min_samples_leaf': (1,), 'min_weight_fraction_leaf': (0,),
                'n_estimators':     (10,), 'oob_score': (False,), 'random_state': (None,),
                'warm_start':       (False,)
            }
        )
        models['Random Forest'] = rfc

    if 'Stochastic Gradient Descent' in checks:
        sgdc = GridSearchCV(
            SGDClassifier(), cv=5, param_grid={
                'average': (True, False), 'fit_intercept': (True, False),
                'loss':    ('squared_loss', 'huber', 'epsilon_insensitive',
                            'squared_epsilon_insensitive'),
                'penalty': ('none', 'l1', 'l2', 'elasticnet'),
                'power_t': (0.1, 0.5)  # 'alpha': (),
                # 'L1_ratio': (),
                # 'max_iter': (),
                # 'tol': (),
                # 'shuffle': (True, False),
                # 'learning_rate': (0.1, 1),
                # 'eta0': (),
                # 'warm_start': (True, False),
            }
        )
        models['Stochastic Gradient Descent'] = sgdc

    if 'Support Vector Machine' in checks:
        svmc = GridSearchCV(
            SVC(), cv=5, param_grid={  # 'C': (1, 100),
                # 'coef0': (0, 1),
                # 'degree': (1, 2),
                # 'epsilon': (0.1, 0.5),
                # 'gamma': ('auto',),
                'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),  # 'max_iter': (1, 10, 20,),
                # 'shrinking': (True, False),
                # 'tol': (1e-2,),
            }
        )
        models['Support Vector Machine'] = svmc

    return models


def get_decision_boundary2(clf, xx, yy):
    """
    Calculate the decision boundaries for a classifier.
    clf: sklearn classifier
    xx, yy: meshgrid ndarray"""
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return Z


def get_decision_boundary(models, xx, yy, x):
    """
    Calculate the decision boundaries for a classifier.
    clf: a classifier
    xx, yy: meshgrid ndarray
    """
    y_models = []
    for model in models:
        Z = models[model].predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        y_models.append(Z)
    prediction = y_models[0]
    return prediction


def classification(models, x, y, p=None):
    if models:
        try:
            # X = pd.DataFrame([[float(x.index[i]), float(x.values[i, 0])] for i in range(len(x))]).as_matrix()
            X = x.as_matrix()
            Y = y.as_matrix()
            fitted_models = MachineLearner.fitting(models, X, Y)
            # results = MachineLearner.predicting(fitted_models, np.linspace(0, p, p)[:, None] if p else X, x)
            X0, X1 = X[:, 0], X[:, 1]
            xRange = DataProcessor.get_range(X0, extension=1)
            yRange = DataProcessor.get_range(X1, extension=1)
            xx, yy = DataProcessor.make_meshgrid(xRange, yRange)
            decisionBoundary = get_decision_boundary(fitted_models, xx, yy, X[:, 0])
            results = (xRange, yRange, decisionBoundary)
            return results
        except Exception as e:
            print(e)


def models(checks, options):
    if 'Decision Tree' in checks:
        settings = options['Decision Tree']
        if settings['max_depth'] == 0:
            settings['max_depth'] = None
        if settings['max_leaf_nodes'] == 0:
            settings['max_leaf_nodes'] = None

    if 'Multilayer Perceptron' in checks:
        settings = options['Multilayer Perceptron']
        if not settings['hidden_layer_sizes']:
            settings['hidden_layer_sizes'] = (100,)
        else:
            settings['hidden_layer_sizes'] = int(settings['hidden_layer_sizes'])

    if 'Stochastic Gradient Descent' in checks:
        settings = options['Stochastic Gradient Descent']
        if settings['tol'] == 0:
            settings['tol'] = None

    if 'Support Vector Machine' in checks:
        settings = options['Support Vector Machine']
        settings['degree'] = int(settings['degree'])
        if settings['gamma'] == 0:
            settings['gamma'] = 'auto'

    models = MachineLearner.get_models(CLASSIFIERS, checks, options)
    return models


def get_decision_boundary2(clf, xx, yy):
    """
    Calculate the decision boundaries for a classifier.
    clf: a classifier
    xx, yy: meshgrid ndarray"""
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return Z


def classification(models, x, y, p=None):
    if models:
        try:
            # X = pd.DataFrame([[float(x.index[i]), float(x.values[i, 0])] for i in range(len(x))]).as_matrix()
            X = x.as_matrix()
            Y = y.as_matrix()
            fitted_models = fitting(models, X, Y)
            # results = predicting(fitted_models, np.linspace(0, p, p)[:, None] if p else X, x)
            X0, X1 = X[:, 0], X[:, 1]
            xRange = dh.get_range(X0, extension=1)
            yRange = dh.get_range(X1, extension=1)
            xx, yy = dh.make_meshgrid(xRange, yRange)
            decisionBoundary = get_decision_boundary(fitted_models, xx, yy, X[:, 0])
            results = (xRange, yRange, decisionBoundary)
            return results
        except Exception as e:
            print(e)


def models(checks, options):
    models = {}

    if 'Adaptive Boost' in checks:
        abc = AdaBoostClassifier(learning_rate=options['abcLearningRate'], n_estimators=options['abcEstimators'])
        # random_state=options['abcRandomState']
        models['Adaptive Boost'] = abc

    if 'Decision Tree' in checks:
        if options['dtcMaximumDepth'] == 0:
            options['dtcMaximumDepth'] = None

        if options['dtcMaximumLeafNodes'] == 0:
            options['dtcMaximumLeafNodes'] = None

        options['dtcRandomState'] = int(options['dtcRandomState']) if options['dtcRandomState'] else None

        dtc = DecisionTreeClassifier(
            criterion=options['dtcCriterion'], max_depth=options['dtcMaximumDepth'],
            max_features=options['dtcMaximumFeatures'],
            max_leaf_nodes=options['dtcMaximumLeafNodes'],
            min_samples_leaf=options['dtcMinimumSamplesLeaf'],
            # min_samples_split=(options['dtcMinimumSamplesSplit']),
            min_weight_fraction_leaf=options['dtcMinimumWeightFractionLeaf'],
            random_state=options['dtcRandomState'], presort=options['dtcPresort'],
            splitter=options['dtcSplitter']
        )
        models['Decision Tree'] = dtc

    if 'Gaussian Process' in checks:
        if not options['gpcAlpha']:
            options['gpcAlpha'] = 1e-10
        else:
            options['gpcAlpha'] = float(options['gpcAlpha'])

        gpc = GaussianProcessClassifier(
            copy_X_train=options['gpcCopyXTrain'], kernel=options['gpcKernel'],
            max_iter_predict=options['gpcMaximumIterations'],
            multi_class=options['gpcMultiClass'], n_jobs=options['gpcNumberofJobs'],
            n_restarts_optimizer=options['gpcNumberofRestarts'],
            warm_start=options['gpcWarmStart'], random_state=options['gpcRandomState']

        )
        models['Gaussian Process'] = gpc

    if 'Nearest Neighbors' in checks:
        knnc = KNeighborsClassifier(
            algorithm=options['knncAlgorithm'], leaf_size=int(options['knncLeafSize']),
            n_neighbors=options['knncNumberOfNeighbors'], p=options['knncMinkowskiPower'],
            weights=options['knncWeightsFunction']
        )
        models['Nearest Neighbors'] = knnc

    if 'Quadratic Discriminant' in checks:
        qdac = QuadraticDiscriminantAnalysis(
            priors=options['qdacPriors'],
            reg_param=options['qdacRegularizationParameter'],
            store_covariance=options['qdacStoreCovariance'],
            tol=options['qdacTolerance'], )

        models['Quadratic Discriminant'] = qdac

    if 'Multilayer Perceptron' in checks:
        if not options['mlpcHiddenLayerSizes']:
            options['mlpcHiddenLayerSizes'] = (100,)
        else:
            options['mlpcHiddenLayerSizes'] = int(options['mlpcHiddenLayerSizes'])

        options['mlpcRandomState'] = int(options['mlpcRandomState']) if options['mlpcRandomState'] else None

        mlpc = MLPClassifier(
            activation=options['mlpcActivationFunction'], alpha=options['mlpcPenaltyParameter'],
            # batch_size=options['mlpcBatchSize'],
            beta_1=options['mlpcFirstMomentExponentialDecay'],
            beta_2=options['mlpcSecondMomentExponentialDecay'],
            early_stopping=options['mlpcEarlyStopping'], epsilon=options['mlpcNumericalStability'],
            hidden_layer_sizes=options['mlpcHiddenLayerSizes'],
            learning_rate=options['mlpcLearningRate'],
            learning_rate_init=options['mlpcInitialLearningRate'],
            max_iter=options['mlpcMaximumIterations'], momentum=options['mlpcMomentum'],
            nesterovs_momentum=options['mlpcNesterovsMomentum'],
            power_t=options['mlpcPowerForInverseLearningRate'],
            random_state=options['mlpcRandomState'], shuffle=options['mlpcShuffle'],
            solver=options['mlpcWeightOptimizationSolver'], tol=options['mlpcTolerance'],
            validation_fraction=options['mlpcValidationFraction'], warm_start=options['mlpcWarmStart']
        )
        models['Multilayer Perceptron'] = mlpc

    if 'Naive Bayes' in checks:
        nbc = GaussianNB(

        )
        models['Naive Bayes'] = nbc

    if 'Random Forest' in checks:
        rfc = RandomForestClassifier()

        models['Random Forest'] = rfc

    if 'Stochastic Gradient Descent' in checks:
        if options['sgdcTolerance'] == 0:
            options['sgdcTolerance'] = None

        sgdc = SGDClassifier(
            alpha=options['sgdcAlpha'], average=options['sgdcAverage'], eta0=options['sgdcEta0'],
            fit_intercept=options['sgdcFitIntercept'], learning_rate=options['sgdcLearningRate'],
            loss=options['sgdcLoss'], l1_ratio=options['sgdcL1Ratio'],
            max_iter=options['sgdcMaxIterations'], penalty=options['sgdcPenalty'],
            power_t=options['sgdcPowerT'], shuffle=options['sgdcShuffle'],
            tol=options['sgdcTolerance'], warm_start=options['sgdcWarmStart']
        )
        models['Stochastic Gradient Descent'] = sgdc

    if 'Support Vector Machine' in checks:
        options['svmcPolynomialDegree'] = int(options['svmcPolynomialDegree'])

        if options['svmcGamma'] == 0:
            options['svmcGamma'] = 'auto'

        svmc = SVC(
            C=options['svmcC'], kernel=options['svmcKernel'], degree=options['svmcPolynomialDegree'],
            gamma=options['svmcGamma'], coef0=options['svmcCoefficient0'], shrinking=options['svmcShrinking'],
            tol=options['svmcTolerance'],  # cache_size=options['svmcCacheSize'],
            max_iter=options['svmcMaximumIterations']
        )
        models['Support Vector Machine'] = svmc

    return models


# TODO mlpc takes too long and doesn't work
def cross_validation_classification_models(checks):
    models = {}
    if checks['Adaptive Boost']:
        abc = GridSearchCV(
            AdaBoostClassifier(), cv=5,
            param_grid={
                'learning_rate': (0.5, 1,), 'loss': ('linear', 'square', 'exponential'),
                'n_estimators':  (1, 5),  # 'random_state': (None,)
            }
        )

        models['abc'] = abc

    if checks['Decision Tree']:
        dtc = GridSearchCV(
            DecisionTreeClassifier(), cv=5, param_grid={
                'criterion':                ('mse',), 'max_depth': (None,),
                'max_features':             (None, 'auto', 'sqrt', 'log2'),
                'max_leaf_nodes':           (2, 4, 6),
                'min_samples_leaf':         (1, 10, 100, .15, .3, .45),
                # min_samples_split=(options['dtcMinimumSamplesSplit']),
                'min_weight_fraction_leaf': (0,),
                'presort':                  (False,), 'random_state': (None,),
                'splitter':                 ('best', 'random')
            }
        )
        models['dtc'] = dtc

    if checks['Gaussian Process']:
        gpc = GridSearchCV(
            GaussianProcessClassifier(), cv=5,
            param_grid={'alpha': (0.1, 0.5), 'normalize_y': (True, False)}
        )
        models['gpc'] = gpc

    if checks['Nearest Neighbors']:
        knnc = GridSearchCV(
            KNeighborsClassifier(), cv=5,
            param_grid={
                'algorithm':   ('ball_tree', 'kd_tree', 'brute', 'auto',), 'leaf_size': (30,),
                'n_neighbors': (5,), 'p': (2,), 'weights': ('uniform', 'distance'),
            }
        )
        models['knnc'] = knnc

    if checks['Multilayer Perceptron']:
        mlpc = GridSearchCV(
            MLPClassifier(), cv=5,
            param_grid={
                'activation':          ('identity', 'logistic', 'tanh', 'relu'), 'alpha': (0.0001,),
                'beta_1':              (0.9,), 'beta_2': (0.999,),  # 'early_stopping':(True,False),
                'epsilon':             (0.00001,), 'hidden_layer_sizes': (100,),
                # 'learning_rate':('constant','invscaling','adaptive'),
                # 'learning_rate_init':(0.001,),
                'momentum':            (0.9, 0.5),  # 'nesterovs_momentum':(True,False),
                'power_t':             (0.5, 0.1), 'solver': ('lbfgs', 'sgd', 'adam'),
                # 'shuffle':(True,False),
                # 'tol':(0.01,),
                'validation_fraction': (0.1, 0.5),  # 'warm_start':(True,False),
            }
        )
        models['mlpc'] = mlpc

    if checks['Naive Bayes']:
        nbc = GridSearchCV(
            GaussianNB(), cv=5, param_grid={

            }
        )
        models['nbc'] = nbc

    if checks['Quadratic Discriminant']:
        krr = GridSearchCV(
            QuadraticDiscriminantAnalysis(), cv=5, param_grid={

            }
        )
        models['krr'] = krr

    if checks['Random Forest']:
        rfc = GridSearchCV(
            RandomForestClassifier(n_jobs=-1), cv=5,
            param_grid={
                'bootstrap2':        (True,), 'criterion': ('mse',), 'max_features': ('auto',),
                'max_depth':        (None,), 'max_leaf_nodes': (None,), 'min_samples_split': (2,),
                'min_samples_leaf': (1,), 'min_weight_fraction_leaf': (0,),
                'n_estimators':     (10,), 'oob_score': (False,), 'random_state': (None,),
                'warm_start':       (False,)
            }
        )
        models['rfc'] = rfc

    if checks['Stochastic Gradient Descent']:
        sgdc = GridSearchCV(
            SGDClassifier(), cv=5, param_grid={
                'average': (True, False), 'fit_intercept': (True, False),
                'loss':    ('squared_loss', 'huber', 'epsilon_insensitive',
                            'squared_epsilon_insensitive'),
                'penalty': ('none', 'l1', 'l2', 'elasticnet'),
                'power_t': (0.1, 0.5)  # 'alpha': (),
                # 'L1_ratio': (),
                # 'max_iter': (),
                # 'tol': (),
                # 'shuffle': (True, False),
                # 'learning_rate': (0.1, 1),
                # 'eta0': (),
                # 'warm_start': (True, False),
            }
        )
        models['sgdc'] = sgdc

    if checks['Support Vector Machine']:
        svmc = GridSearchCV(
            SVC(), cv=5, param_grid={  # 'C': (1, 100),
                # 'coef0': (0, 1),
                # 'degree': (1, 2),
                # 'epsilon': (0.1, 0.5),
                # 'gamma': ('auto',),
                'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),  # 'max_iter': (1, 10, 20,),
                # 'shrinking': (True, False),
                # 'tol': (1e-2,),
            }
        )
        models['svmc'] = svmc

    return models


def classification(models, x, y, p=None):
    if models:
        try:
            print(x)
            print(y)
            X = pd.DataFrame([[float(x.index[i]), float(x.values[i, 0])] for i in range(len(x))]).as_matrix()
            print(X)
            fitted_models = fitting(models, X, y.as_matrix().astype(float))
            results = predicting(fitted_models, np.linspace(0, p, p)[:, None] if p else X, x)
            return results
        except Exception as e:
            print(e)


def classification_models(checks, options):
    models = {}

    if checks['Adaptive Boost']:
        abc = AdaBoostClassifier(learning_rate=options['abcLearningRate'], n_estimators=options['abcEstimators'])
        # random_state=options['abcRandomState']
        models['abc'] = abc

    if checks['Decision Tree']:
        if options['dtcMaximumDepth'] == 0:
            options['dtcMaximumDepth'] = None

        if options['dtcMaximumLeafNodes'] == 0:
            options['dtcMaximumLeafNodes'] = None

        options['dtcRandomState'] = int(options['dtcRandomState']) if options['dtcRandomState'] else None

        dtc = DecisionTreeClassifier(
            criterion=options['dtcCriterion'], max_depth=options['dtcMaximumDepth'],
            max_features=options['dtcMaximumFeatures'],
            max_leaf_nodes=options['dtcMaximumLeafNodes'],
            min_samples_leaf=options['dtcMinimumSamplesLeaf'],
            # min_samples_split=(options['dtcMinimumSamplesSplit']),
            min_weight_fraction_leaf=options['dtcMinimumWeightFractionLeaf'],
            random_state=options['dtcRandomState'], presort=options['dtcPresort'],
            splitter=options['dtcSplitter']
        )
        models['dtc'] = dtc

    if checks['Gaussian Process']:
        if not options['gpcAlpha']:
            options['gpcAlpha'] = 1e-10
        else:
            options['gpcAlpha'] = float(options['gpcAlpha'])

        gpc = GaussianProcessClassifier(

        )
        models['gpc'] = gpc

    if checks['Nearest Neighbors']:
        knnc = KNeighborsClassifier(
            algorithm=options['knncAlgorithm'], leaf_size=int(options['knncLeafSize']),
            n_neighbors=options['knncNumberOfNeighbors'], p=options['knncMinkowskiPower'],
            weights=options['knncWeightsFunction']
        )
        models['knnc'] = knnc

    if checks['Quadratic Discriminant']:
        qdac = QuadraticDiscriminantAnalysis()

        models['qdac'] = qdac

    if checks['Multilayer Perceptron']:
        if not options['mlpcHiddenLayerSizes']:
            options['mlpcHiddenLayerSizes'] = (100,)
        else:
            options['mlpcHiddenLayerSizes'] = int(options['mlpcHiddenLayerSizes'])

        options['mlpcRandomState'] = int(options['mlpcRandomState']) if options['mlpcRandomState'] else None

        mlpc = MLPClassifier(
            activation=options['mlpcActivationFunction'], alpha=options['mlpcPenaltyParameter'],
            # batch_size=options['mlpcBatchSize'],
            beta_1=options['mlpcFirstMomentExponentialDecay'],
            beta_2=options['mlpcSecondMomentExponentialDecay'],
            early_stopping=options['mlpcEarlyStopping'], epsilon=options['mlpcNumericalStability'],
            hidden_layer_sizes=options['mlpcHiddenLayerSizes'],
            learning_rate=options['mlpcLearningRate'],
            learning_rate_init=options['mlpcInitialLearningRate'],
            max_iter=options['mlpcMaximumIterations'], momentum=options['mlpcMomentum'],
            nesterovs_momentum=options['mlpcNesterovsMomentum'],
            power_t=options['mlpcPowerForInverseLearningRate'],
            random_state=options['mlpcRandomState'], shuffle=options['mlpcShuffle'],
            solver=options['mlpcWeightOptimizationSolver'], tol=options['mlpcTolerance'],
            validation_fraction=options['mlpcValidationFraction'], warm_start=options['mlpcWarmStart']
        )
        models['mlpc'] = mlpc

    if checks['Naive Bayes']:
        nbc = GaussianNB(

        )
        models['nbc'] = nbc

    if checks['Random Forest']:
        rfc = RandomForestClassifier()

        models['rfc'] = rfc

    if checks['Stochastic Gradient Descent']:
        if options['sgdcTolerance'] == 0:
            options['sgdcTolerance'] = None

        sgdc = SGDClassifier(
            alpha=options['sgdcAlpha'], average=options['sgdcAverage'], eta0=options['sgdcEta0'],
            fit_intercept=options['sgdcFitIntercept'], learning_rate=options['sgdcLearningRate'],
            loss=options['sgdcLoss'], l1_ratio=options['sgdcL1Ratio'],
            max_iter=options['sgdcMaxIterations'], penalty=options['sgdcPenalty'],
            power_t=options['sgdcPowerT'], shuffle=options['sgdcShuffle'],
            tol=options['sgdcTolerance'], warm_start=options['sgdcWarmStart']
        )
        models['sgdc'] = sgdc

    if checks['Support Vector Machine']:
        options['svmcPolynomialDegree'] = int(options['svmcPolynomialDegree'])

        if options['svmcGamma'] == 0:
            options['svmcGamma'] = 'auto'

        svmc = SVC(
            C=options['svmcC'], kernel=options['svmcKernel'], degree=options['svmcPolynomialDegree'],
            gamma=options['svmcGamma'], coef0=options['svmcCoefficient0'], shrinking=options['svmcShrinking'],
            tol=options['svmcTolerance'],  # cache_size=options['svmcCacheSize'],
            max_iter=options['svmcMaximumIterations']
        )
        models['svmc'] = svmc

    return models


def classification(models, x, y, p=None):
    if models:
        try:
            # X = pd.DataFrame([[float(x.index[i]), float(x.values[i, 0])] for i in range(len(x))]).as_matrix()
            X = x.as_matrix()
            Y = y.as_matrix()
            fitted_models = MachineLearner.fitting(models, X, Y)
            # results = MachineLearner.predicting(fitted_models, np.linspace(0, p, p)[:, None] if p else X, x)
            X0, X1 = X[:, 0], X[:, 1]
            xRange = dh.get_range(X0, extension=1)
            yRange = dh.get_range(X1, extension=1)
            xx, yy = dh.make_meshgrid(xRange, yRange)
            decisionBoundary = get_decision_boundary(fitted_models, xx, yy, X[:, 0])
            results = (xRange, yRange, decisionBoundary)
            return results
        except Exception as e:
            print(e)


# TODO mlpc takes too long and doesn't work
def cross_validation_classification_models(checks):
    models = {}
    if checks['abc']:
        abc = GridSearchCV(
            AdaBoostClassifier(), cv=5,
            param_grid={
                'learning_rate': (0.5, 1,), 'loss': ('linear', 'square', 'exponential'),
                'n_estimators':  (1, 5),  # 'random_state': (None,)
            }
        )

        models['abc'] = abc

    if checks['dtc']:
        dtc = GridSearchCV(
            DecisionTreeClassifier(), cv=5, param_grid={
                'criterion':                ('mse',), 'max_depth': (None,),
                'max_features':             (None, 'auto', 'sqrt', 'log2'),
                'max_leaf_nodes':           (2, 4, 6),
                'min_samples_leaf':         (1, 10, 100, .15, .3, .45),
                # min_samples_split=(options['dtcMinimumSamplesSplit']),
                'min_weight_fraction_leaf': (0,),
                'presort':                  (False,), 'random_state': (None,),
                'splitter':                 ('best', 'random')
            }
        )
        models['dtc'] = dtc

    if checks['enr']:
        enr = GridSearchCV(
            GaussianNB(), cv=5,
            param_grid={
                'alpha':     (0.1, 1), 'fit_intercept': (True, False), 'l1_ratio': (0.25, 0.5),
                'normalize': (True, False), 'positive': (True, False),
                # 'random_state': (None,),
                'selection': ('cyclic', 'random'), 'tol': (0.01,), 'warm_start': (True, False)
            }
        )
        models['enr'] = enr

    if checks['gpc']:
        gpc = GridSearchCV(
            GaussianProcessClassifier(), cv=5,
            param_grid={'alpha': (0.1, 0.5), 'normalize_y': (True, False)}
        )
        models['gpc'] = gpc

    if checks['knnc']:
        knnc = GridSearchCV(
            KNeighborsClassifier(), cv=5,
            param_grid={
                'algorithm':   ('ball_tree', 'kd_tree', 'brute', 'auto',), 'leaf_size': (30,),
                'n_neighbors': (5,), 'p': (2,), 'weights': ('uniform', 'distance'),
            }
        )
        models['knnc'] = knnc

    if checks['qdac']:
        krr = GridSearchCV(
            QuadraticDiscriminantAnalysis(), cv=5, param_grid={

            }
        )
        models['krr'] = krr

    if checks['mlpc']:
        mlpc = GridSearchCV(
            MLPClassifier(), cv=5,
            param_grid={
                'activation':          ('identity', 'logistic', 'tanh', 'relu'), 'alpha': (0.0001,),
                'beta_1':              (0.9,), 'beta_2': (0.999,),  # 'early_stopping':(True,False),
                'epsilon':             (0.00001,), 'hidden_layer_sizes': (100,),
                # 'learning_rate':('constant','invscaling','adaptive'),
                # 'learning_rate_init':(0.001,),
                'momentum':            (0.9, 0.5),  # 'nesterovs_momentum':(True,False),
                'power_t':             (0.5, 0.1), 'solver': ('lbfgs', 'sgd', 'adam'),
                # 'shuffle':(True,False),
                # 'tol':(0.01,),
                'validation_fraction': (0.1, 0.5),  # 'warm_start':(True,False),
            }
        )
        models['mlpc'] = mlpc

    if checks['rfc']:
        rfc = GridSearchCV(
            RandomForestClassifier(n_jobs=-1), cv=5,
            param_grid={
                'bootstrap2':        (True,), 'criterion': ('mse',), 'max_features': ('auto',),
                'max_depth':        (None,), 'max_leaf_nodes': (None,), 'min_samples_split': (2,),
                'min_samples_leaf': (1,), 'min_weight_fraction_leaf': (0,),
                'n_estimators':     (10,), 'oob_score': (False,), 'random_state': (None,),
                'warm_start':       (False,)
            }
        )
        models['rfc'] = rfc

    if checks['sgdc']:
        sgdc = GridSearchCV(
            SGDClassifier(), cv=5, param_grid={
                'average': (True, False), 'fit_intercept': (True, False),
                'loss':    ('squared_loss', 'huber', 'epsilon_insensitive',
                            'squared_epsilon_insensitive'),
                'penalty': ('none', 'l1', 'l2', 'elasticnet'),
                'power_t': (0.1, 0.5)  # 'alpha': (),
                # 'L1_ratio': (),
                # 'max_iter': (),
                # 'tol': (),
                # 'shuffle': (True, False),
                # 'learning_rate': (0.1, 1),
                # 'eta0': (),
                # 'warm_start': (True, False),
            }
        )
        models['sgdc'] = sgdc

    if checks['svr']:
        svr = GridSearchCV(
            SVC(), cv=5, param_grid={  # 'C': (1, 100),
                # 'coef0': (0, 1),
                # 'degree': (1, 2),
                # 'epsilon': (0.1, 0.5),
                # 'gamma': ('auto',),
                'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),  # 'max_iter': (1, 10, 20,),
                # 'shrinking': (True, False),
                # 'tol': (1e-2,),
            }
        )
        models['svr'] = svr

    return models


def models(checks, options):
    models = {}

    if 'Adaptive Boost' in checks:
        settings = options['Adaptive Boost']
        abc = AdaBoostClassifier(**settings)
        models['Adaptive Boost'] = abc

    if 'Decision Tree' in checks:
        settings = options['Decision Tree']
        if settings['max_depth'] == 0:
            settings['max_depth'] = None

        if settings['max_leaf_nodes'] == 0:
            settings['max_leaf_nodes'] = None

        dtc = DecisionTreeClassifier(**settings)
        models['Decision Tree'] = dtc

    if 'Gaussian Process' in checks:
        settings = options['Gaussian Process']
        gpc = GaussianProcessClassifier(**settings)
        models['Gaussian Process'] = gpc

    if 'Multilayer Perceptron' in checks:
        settings = options['Multilayer Perceptron']
        if not settings['hidden_layer_sizes']:
            settings['hidden_layer_sizes'] = (100,)
        else:
            settings['hidden_layer_sizes'] = int(settings['hidden_layer_sizes'])
        mlpc = MLPClassifier(**settings)
        models['Multilayer Perceptron'] = mlpc

    if 'Naive Bayes' in checks:
        nbc = GaussianNB(

        )
        models['Naive Bayes'] = nbc

    if 'Nearest Neighbors' in checks:
        settings = options['Nearest Neighbors']
        knnc = KNeighborsClassifier(**settings)
        models['Nearest Neighbors'] = knnc

    if 'Quadratic Discriminant' in checks:
        settings = options['Quadratic Discriminant']
        qdac = QuadraticDiscriminantAnalysis(**settings)
        models['Quadratic Discriminant'] = qdac

    if 'Random Forest' in checks:
        settings = options['Random Forest']
        rfc = RandomForestClassifier(**settings)
        models['Random Forest'] = rfc

    if 'Stochastic Gradient Descent' in checks:
        settings = options['Stochastic Gradient Descent']
        if settings['tol'] == 0:
            settings['tol'] = None
        sgdc = SGDClassifier(**settings)
        models['Stochastic Gradient Descent'] = sgdc

    if 'Support Vector Machine' in checks:
        settings = options['Support Vector Machine']
        settings['degree'] = int(settings['degree'])
        if settings['gamma'] == 0:
            settings['gamma'] = 'auto'
        svmc = SVC(**settings)
        models['Support Vector Machine'] = svmc

    return models


# TODO mlpc takes too long and doesn't work
def cross_validation_classification_models(checks):
    models = {}
    if checks['abc']:
        abc = GridSearchCV(
            AdaBoostClassifier(), cv=5,
            param_grid={
                'learning_rate': (0.5, 1,), 'loss': ('linear', 'square', 'exponential'),
                'n_estimators':  (1, 5),  # 'random_state': (None,)
            }
        )

        models['abc'] = abc

    if checks['dtc']:
        dtc = GridSearchCV(
            DecisionTreeClassifier(), cv=5, param_grid={
                'criterion':                ('mse',), 'max_depth': (None,),
                'max_features':             (None, 'auto', 'sqrt', 'log2'),
                'max_leaf_nodes':           (2, 4, 6),
                'min_samples_leaf':         (1, 10, 100, .15, .3, .45),
                # min_samples_split=(options['dtcMinimumSamplesSplit']),
                'min_weight_fraction_leaf': (0,),
                'presort':                  (False,), 'random_state': (None,),
                'splitter':                 ('best', 'random')
            }
        )
        models['dtc'] = dtc

    if checks['enr']:
        enr = GridSearchCV(
            GaussianNB(), cv=5,
            param_grid={
                'alpha':     (0.1, 1), 'fit_intercept': (True, False), 'l1_ratio': (0.25, 0.5),
                'normalize': (True, False), 'positive': (True, False),
                # 'random_state': (None,),
                'selection': ('cyclic', 'random'), 'tol': (0.01,), 'warm_start': (True, False)
            }
        )
        models['enr'] = enr

    if checks['gpc']:
        gpc = GridSearchCV(
            GaussianProcessClassifier(), cv=5,
            param_grid={'alpha': (0.1, 0.5), 'normalize_y': (True, False)}
        )
        models['gpc'] = gpc

    if checks['knnc']:
        knnc = GridSearchCV(
            KNeighborsClassifier(), cv=5,
            param_grid={
                'algorithm':   ('ball_tree', 'kd_tree', 'brute', 'auto',), 'leaf_size': (30,),
                'n_neighbors': (5,), 'p': (2,), 'weights': ('uniform', 'distance'),
            }
        )
        models['knnc'] = knnc

    if checks['mlpc']:
        mlpc = GridSearchCV(
            MLPClassifier(), cv=5,
            param_grid={
                'activation':          ('identity', 'logistic', 'tanh', 'relu'), 'alpha': (0.0001,),
                'beta_1':              (0.9,), 'beta_2': (0.999,),  # 'early_stopping':(True,False),
                'epsilon':             (0.00001,), 'hidden_layer_sizes': (100,),
                # 'learning_rate':('constant','invscaling','adaptive'),
                # 'learning_rate_init':(0.001,),
                'momentum':            (0.9, 0.5),  # 'nesterovs_momentum':(True,False),
                'power_t':             (0.5, 0.1), 'solver': ('lbfgs', 'sgd', 'adam'),
                # 'shuffle':(True,False),
                # 'tol':(0.01,),
                'validation_fraction': (0.1, 0.5),  # 'warm_start':(True,False),
            }
        )
        models['mlpc'] = mlpc

    if checks['qdac']:
        krr = GridSearchCV(
            QuadraticDiscriminantAnalysis(), cv=5, param_grid={

            }
        )
        models['krr'] = krr

    if checks['rfc']:
        rfc = GridSearchCV(
            RandomForestClassifier(n_jobs=-1), cv=5,
            param_grid={
                'bootstrap2':        (True,), 'criterion': ('mse',), 'max_features': ('auto',),
                'max_depth':        (None,), 'max_leaf_nodes': (None,), 'min_samples_split': (2,),
                'min_samples_leaf': (1,), 'min_weight_fraction_leaf': (0,),
                'n_estimators':     (10,), 'oob_score': (False,), 'random_state': (None,),
                'warm_start':       (False,)
            }
        )
        models['rfc'] = rfc

    if checks['sgdc']:
        sgdc = GridSearchCV(
            SGDClassifier(), cv=5, param_grid={
                'average': (True, False), 'fit_intercept': (True, False),
                'loss':    ('squared_loss', 'huber', 'epsilon_insensitive',
                            'squared_epsilon_insensitive'),
                'penalty': ('none', 'l1', 'l2', 'elasticnet'),
                'power_t': (0.1, 0.5)  # 'alpha': (),
                # 'L1_ratio': (),
                # 'max_iter': (),
                # 'tol': (),
                # 'shuffle': (True, False),
                # 'learning_rate': (0.1, 1),
                # 'eta0': (),
                # 'warm_start': (True, False),
            }
        )
        models['sgdc'] = sgdc

    if checks['svmc']:
        svmc = GridSearchCV(
            SVC(), cv=5, param_grid={  # 'C': (1, 100),
                # 'coef0': (0, 1),
                # 'degree': (1, 2),
                # 'epsilon': (0.1, 0.5),
                # 'gamma': ('auto',),
                'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),  # 'max_iter': (1, 10, 20,),
                # 'shrinking': (True, False),
                # 'tol': (1e-2,),
            }
        )
        models['svmc'] = svmc

    return models


def classification_models(checks, options):
    models = {}

    if checks['abc']:
        abc = AdaBoostClassifier(learning_rate=options['abcLearningRate'], n_estimators=options['abcEstimators'])
        # random_state=options['abcRandomState']
        models['abc'] = abc

    if checks['dtc']:
        if options['dtcMaximumDepth'] == 0:
            options['dtcMaximumDepth'] = None

        if options['dtcMaximumLeafNodes'] == 0:
            options['dtcMaximumLeafNodes'] = None

        options['dtcRandomState'] = int(options['dtcRandomState']) if options['dtcRandomState'] else None

        dtc = DecisionTreeClassifier(
            criterion=options['dtcCriterion'], max_depth=options['dtcMaximumDepth'],
            max_features=options['dtcMaximumFeatures'],
            max_leaf_nodes=options['dtcMaximumLeafNodes'],
            min_samples_leaf=options['dtcMinimumSamplesLeaf'],
            # min_samples_split=(options['dtcMinimumSamplesSplit']),
            min_weight_fraction_leaf=options['dtcMinimumWeightFractionLeaf'],
            random_state=options['dtcRandomState'], presort=options['dtcPresort'],
            splitter=options['dtcSplitter']
        )
        models['dtc'] = dtc

    if checks['nbc']:
        enr = GaussianNB(

        )
        models['enr'] = enr

    if checks['gpc']:
        if not options['gpcAlpha']:
            options['gpcAlpha'] = 1e-10
        else:
            options['gpcAlpha'] = float(options['gpcAlpha'])

        gpc = GaussianProcessClassifier(

        )
        models['gpc'] = gpc

    if checks['knnc']:
        knnc = KNeighborsClassifier(
            algorithm=options['knncAlgorithm'], leaf_size=int(options['knncLeafSize']),
            n_neighbors=options['knncNumberOfNeighbors'], p=options['knncMinkowskiPower'],
            weights=options['knncWeightsFunction']
        )
        models['knnc'] = knnc

    if checks['qdac']:
        qdac = QuadraticDiscriminantAnalysis()

        models['qdac'] = qdac

    if checks['mlpc']:
        if not options['mlpcHiddenLayerSizes']:
            options['mlpcHiddenLayerSizes'] = (100,)
        else:
            options['mlpcHiddenLayerSizes'] = int(options['mlpcHiddenLayerSizes'])

        options['mlpcRandomState'] = int(options['mlpcRandomState']) if options['mlpcRandomState'] else None

        mlpc = MLPClassifier(
            activation=options['mlpcActivationFunction'], alpha=options['mlpcPenaltyParameter'],
            # batch_size=options['mlpcBatchSize'],
            beta_1=options['mlpcFirstMomentExponentialDecay'],
            beta_2=options['mlpcSecondMomentExponentialDecay'],
            early_stopping=options['mlpcEarlyStopping'], epsilon=options['mlpcNumericalStability'],
            hidden_layer_sizes=options['mlpcHiddenLayerSizes'],
            learning_rate=options['mlpcLearningRate'],
            learning_rate_init=options['mlpcInitialLearningRate'],
            max_iter=options['mlpcMaximumIterations'], momentum=options['mlpcMomentum'],
            nesterovs_momentum=options['mlpcNesterovsMomentum'],
            power_t=options['mlpcPowerForInverseLearningRate'],
            random_state=options['mlpcRandomState'], shuffle=options['mlpcShuffle'],
            solver=options['mlpcWeightOptimizationSolver'], tol=options['mlpcTolerance'],
            validation_fraction=options['mlpcValidationFraction'], warm_start=options['mlpcWarmStart']
        )
        models['mlpc'] = mlpc

    if checks['rfc']:
        rfc = RandomForestClassifier()

        models['rfc'] = rfc

    if checks['sgdc']:
        if options['sgdcTolerance'] == 0:
            options['sgdcTolerance'] = None

        sgdc = SGDClassifier(
            alpha=options['sgdcAlpha'], average=options['sgdcAverage'], eta0=options['sgdcEta0'],
            fit_intercept=options['sgdcFitIntercept'], learning_rate=options['sgdcLearningRate'],
            loss=options['sgdcLoss'], l1_ratio=options['sgdcL1Ratio'],
            max_iter=options['sgdcMaxIterations'], penalty=options['sgdcPenalty'],
            power_t=options['sgdcPowerT'], shuffle=options['sgdcShuffle'],
            tol=options['sgdcTolerance'], warm_start=options['sgdcWarmStart']
        )
        models['sgdc'] = sgdc

    if checks['svmc']:
        options['svmcPolynomialDegree'] = int(options['svmcPolynomialDegree'])

        if options['svmcGamma'] == 0:
            options['svmcGamma'] = 'auto'

        svmc = SVC(
            C=options['svmcC'], kernel=options['svmcKernel'], degree=options['svmcPolynomialDegree'],
            gamma=options['svmcGamma'], coef0=options['svmcCoefficient0'], shrinking=options['svmcShrinking'],
            tol=options['svmcTolerance'],  # cache_size=options['svmcCacheSize'],
            max_iter=options['svmcMaximumIterations']
        )
        models['svmc'] = svmc

    if checks['svc']:
        options['svcPolynomialDegree'] = int(options['svrPolynomialDegree'])

        if options['svcGamma'] == 0:
            options['svcGamma'] = 'auto'

        svc = SVC(
            C=options['svcC'], kernel=options['svcKernel'], degree=options['svcPolynomialDegree'],
            gamma=options['svcGamma'], coef0=options['svcCoefficient0'], shrinking=options['svcShrinking'],
            tol=options['svcTolerance'],  # cache_size=options['svrCacheSize'],
            max_iter=options['svcMaximumIterations']
        )
        models['svc'] = svc

    return models


def set_data():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return X, y


def get_datasets():
    return [make_moons(noise=0.3, random_state=0), make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]


def get_names():
    return ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process", "Decision Tree", "Random Forest",
            "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]


def get_classifiers():
    return [KNeighborsClassifier(3), SVC(kernel="linear", C=0.025), SVC(gamma=2, C=1),
            GaussianProcessClassifier(1.0 * RBF(1.0)), DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), MLPClassifier(alpha=1),
            AdaBoostClassifier(), GaussianNB(), QuadraticDiscriminantAnalysis()]


def plot_test_train(ax, X_train, X_test, y_train, y_test, cm_bright, xx, yy):
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())


def plot_decision_boundary(clf, xx, yy):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        return clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        return clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]


def plot_scatter_and_color(ds_cnt, ax, Z, xx, yy, cm, X_train, X_test, y_train, y_test, cm_bright):
    # Put the result into a color plot0
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
    plot_settings(ax, xx, yy)
    if ds_cnt == 0:
        ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')


def plot_settings(ax, xx, yy):
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())


h = 0.01  # step size in the mesh

names, classifiers = get_names(), get_classifiers()
X, y = set_data()
linearly_separable = (X, y)

datasets = get_datasets()

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # just plot0 the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")

    plot_test_train(ax, X_train, X_test, y_train, y_test, cm_bright, xx, yy)

    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        Z = plot_decision_boundary(clf, xx, yy)
        plot_scatter_and_color(ds_cnt, ax, Z, xx, yy, cm, X_train, X_test, y_train, y_test, cm_bright)

        i += 1

plt.tight_layout()
plt.show()


def classification(models, x, y, p=None):
    if models:
        try:
            X = pd.DataFrame([[float(x.index[i]), float(x.values[i, 0])] for i in range(len(x))]).as_matrix()
            fitted_models = fitting(models, X, y.as_matrix().astype(float))
            results = predicting(fitted_models, np.linspace(0, p, p)[:, None] if p else X, x)
            return results
        except Exception as e:
            print(e)


def set_data():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return X, y


def get_classifiers():
    classifiers = {  # 'Nearest Neighbors': KNeighborsClassifier(),
        # 'Support Vector Machine': SVC(),
        'Gaussian Process': GaussianProcessClassifier(),  # 'Decision Tree': DecisionTreeClassifier(),
        # 'Random Forest': RandomForestClassifier(),
        # 'Multilayer Perceptron': MLPClassifier(),
        # 'Adaptive Boost': AdaBoostClassifier(),
        # 'Naive Bayes': GaussianNB(),
        # 'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis()
    }
    return classifiers


def plot_data(ax, X, y, cm_bright):
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')


def plot_scatter_and_color(ax, Z, xx, yy, cm, X_train, X_test, y_train, y_test, cm_bright, score, name):
    # Put the result into a color plot0
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    plot_data(ax, X, y, cm_bright)

    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')


def classify(classifiers, X):
    i = 2
    xx, yy = get_mesh(X)
    for clf in classifiers:
        ax = plt.subplot(1, len(classifiers) + 1, i)
        classifiers[clf].fit(X_train, y_train)
        score = classifiers[clf].score(X_test, y_test)
        Z = plot_decision_boundary(classifiers[clf], xx, yy)
        plot_scatter_and_color(ax, Z, xx, yy, cm, X_train, X_test, y_train, y_test, cm_bright, score, clf)
        i += 1


def get_mesh(X):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


classifiers = get_classifiers()
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#00FFFF'])
DATABASE = os.path.abspath('C:\\Users\\frano\PycharmProjects\Big\Database\ClassTest\ClassTest.txt')
X = pd.read_csv(DATABASE, usecols=('G1', 'G2', 'G3'), nrows=20)
X.fillna(0, inplace=True)
y = X['G3']
X.drop(['G3'], axis=1, inplace=True)
X = X.as_matrix()
y = y.as_matrix()
print(X, y)
h = 0.01  # step size in the mesh

figure = plt.figure(figsize=(27, 9))

# X, y = set_data()
# X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8, random_state=42)

# just plot0 the dataset first
ax = plt.subplot(1, len(classifiers) + 1, 1)
ax.set_title("Input data")
plot_data(ax, X, y, cm_bright)

classify(classifiers, X)
plt.show()


def categorical_to_numerical_labels(X=None, y=None):
    if type(X) == type(pd.DataFrame()):
        newX = []
        for column in X:
            unique = X[column].unique()
            if any(isinstance(item, str) for item in unique):
                x = []
                labels = {val: i for i, val in enumerate(X[column].unique())}
                for val in X[column]:
                    for label in labels:
                        if val == label:
                            x.append(labels[label])
                newX.append(x)
            else:
                newX.append(X[column].tolist())
        return pd.DataFrame(newX)
    if type(y) == type(pd.DataFrame()):
        newY = []
        labels = {val: i for i, val in enumerate(y.unique())}
        for val in y:
            for label in labels:
                if val == label:
                    newY.append(labels[label])
        return pd.Series(newY)


def classify(classifiers, X):
    i = 2
    xx, yy = get_mesh(X)
    for clf in classifiers:
        ax = plt.subplot(1, len(classifiers) + 1, i)
        classifiers[clf].fit(X_train, y_train)
        score = classifiers[clf].score(X_test, y_test)
        Z = plot_decision_boundary(classifiers[clf], xx, yy)
        plot_scatter_and_color(ax, Z, xx, yy, cm, X_train, X_test, y_train, y_test, cm_bright, score, clf)
        i += 1


def get_classifiers():
    classifiers = {  # 'Nearest Neighbors': KNeighborsClassifier(),
        # 'Support Vector Machine': SVC(),
        'Gaussian Process': GaussianProcessClassifier(),  # 'Decision Tree': DecisionTreeClassifier(),
        # 'Random Forest': RandomForestClassifier(),
        # 'Multilayer Perceptron': MLPClassifier(),
        # 'Adaptive Boost': AdaBoostClassifier(),
        # 'Naive Bayes': GaussianNB(),
        # 'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis()
    }
    return classifiers


def get_datasets():
    return [make_moons(noise=0.3, random_state=0), make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]


def get_eigenvalues(data, n=None):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n)
    pca.fit(data)
    var = pca.explained_variance_ratio_
    return var


def get_names():
    return ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process", "Decision Tree", "Random Forest",
            "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot0 in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), )
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_data(ax, X, y, cm_bright):
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')


def plot_settings(ax, xx, yy):
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())


def plot_test_train(ax, X_train, X_test, y_train, y_test, cm_bright, xx, yy):
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
    plot_settings(ax, xx, yy)


def process_data(data, label, columns=None):
    """
    Reads a file and splits the data into a label pandas Series and a pandas DataFrame to use in sklearn
    file: string
    label: string
    columns: list of strings
    returns: pandas DataFrame
    """
    y = data[label]
    if columns:
        X = data[list((columns))]
    else:
        X = data.drop([label], axis=1)
    return X.values, y


def process_data(file, label, columns=None):
    """
    Reads a file and splits the data into a label pandas Series and a pandas DataFrame to use in sklearn
    file: string
    label: string
    columns: list of strings
    returns: pandas DataFrame
    """
    X = pd.read_csv(file)
    X.fillna(0, inplace=True)
    y = X[label].values
    if columns:
        X = X[list((columns,))].values
    else:
        X = X.drop([label], axis=1).values
    return X, y


def nig_fig(model, start, stop, iterations):
    print("\n" + str(start) + " to " + str(stop))
    for _ in range(iterations):
        a = random.randrange(start, stop)
        prediction = clf.predict((a, a))
        print(a, prediction[0])


def set_data():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return X, y


DATABASE = os.path.abspath('C:\\Users\\frano\PycharmProjects\PASS simplified\Database\Alcohol\mat.txt')
X = pd.read_csv(DATABASE, usecols=('G1', 'G2', 'G3'))
X.fillna(0, inplace=True)
y = X['G3']
X.drop(['G3'], axis=1, inplace=True)


def set_data():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return X, y


def get_datasets():
    return [make_moons(noise=0.3, random_state=0), make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]


def get_names():
    return ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process", "Decision Tree", "Random Forest",
            "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]


def plot_settings(ax, xx, yy):
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())


h = 0.01  # step size in the mesh

names, classifiers = get_names(), get_classifiers()
X, y = set_data()
linearly_separable = (X, y)

datasets = get_datasets()

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # just plot0 the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")

    plot_test_train(ax, X_train, X_test, y_train, y_test, cm_bright, xx, yy)

    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        Z = plot_decision_boundary(clf, xx, yy)
        plot_scatter_and_color(ds_cnt, ax, Z, xx, yy, cm, X_train, X_test, y_train, y_test, cm_bright)

        i += 1

plt.tight_layout()
plt.show()

h = 0.01  # step size in the mesh

names, classifiers = get_names(), get_classifiers()
X, y = set_data()
linearly_separable = (X, y)

datasets = get_datasets()

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=42)
    xx, yy = get_mesh(X)
    # just plot0 the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    plot_test_train(ax, X_train, X_test, y_train, y_test, cm_bright, xx, yy)
    i += 1
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        Z = plot_decision_boundary(clf, xx, yy)
        plot_scatter_and_color(ds_cnt, ax, Z, xx, yy, cm, X_train, X_test, y_train, y_test, cm_bright)
        i += 1

plt.tight_layout()
plt.show()

clf = SVC()

plt.figure()
t0 = time.time()
X, y = process_data('creditcard small.csv', 'Class', 'Time')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

clf.fit(X_train, y_train)

plt.plot(X)
plt.show()
print(clf.score(X_test, y_test))
print(time.time() - t0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

# #############################################################################
# Visualize the results on PCA-reduced data
reduced_data = PCA(n_components=2).fit_transform(data)
ks = [15, 11]
m = [[i[k] for k in ks] for i in data]
print(m)
print(reduced_data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(m)

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot0
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.imshow(
    Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired,
    aspect='auto', origin='lower'
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

print(__doc__)

# Code source: Gaël Varoquaux
# License: BSD 2 clause


np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(
        X[y == label, 0].mean(), X[y == label, 1].mean() + 1.5, X[y == label, 2].mean(), name,
        horizontalalignment='center', bbox=dict(alpha=.5, edgecolor='w', facecolor='w')
    )
# Reorder the labels to have colors matching the aibase results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

# import data
iris = datasets.load_iris()
X = iris.data[:, :2]  # only take first 1 features
y = iris.target
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
plt.figure(2, figsize=(8, 6))
plt.clf()
# plot0 training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', )
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
# plot0 first 2 PCA dimensions
fig = plt.figure(1, figsize=(11, 7))
ax = Axes3D(fig, elev=-150, azim=110)
xReduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(xReduced[:, 0], xReduced[:, 1], xReduced[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40, )
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.zaxis.set_ticklabels([])
plt.show()

# Generate data (swiss roll dataset)
n_samples = 2500
noise = 0.12358132134
X, _ = make_swiss_roll(n_samples, noise)
# Make it thinner
X[:, 1] *= .12358132134
# Compute clustering
print("Compute unstructured hierarchical clustering...")
st = time.time()
ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X)
elapsed_time = time.time() - st
label = ward.labels_
print("Elapsed time: %.2fs" % elapsed_time)
print("Number of points: %i" % label.size)
# Plot result
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(
        X[label == l, 0], X[label == l, 1], X[label == l, 2], color=plt.cm.jet(np.float(l) / np.max(label + 1)),
        s=20, edgecolor='k'
    )
plt.title('Without connectivity constraints (time %.2fs)' % elapsed_time)
# Define the structure A of the data. Here a 10 nearest neighbors
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
# Compute clustering
print("Compute structured hierarchical clustering...")
st = time.time()
ward = AgglomerativeClustering(n_clusters=6, connectivity=connectivity, linkage='ward', ).fit(X)
elapsed_time = time.time() - st
label = ward.labels_
print("Elapsed time: %.2fs" % elapsed_time)
print("Number of points: %i" % label.size)
# Plot result
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(
        X[label == l, 0], X[label == l, 1], X[label == l, 2], color=plt.cm.jet(float(l) / np.max(label + 1)),
        s=20, edgecolor='k'
    )
plt.title('With connectivity constraints (time %.2fs)' % elapsed_time)
plt.show()

digits = datasets.load_digits()
# print(digits.data)
# print(digits.target)
# print(digits.images[0])
clf = svm.SVC(gamma=0.011, C=300)
print(len(digits.data))
x, y = digits.data[:-13], digits.target[:-13]
clf.fit(x, y)
print('Prediction:', clf.predict(digits.data[-13]))
plt.imshow(digits.images[-13], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target
data = pd.read_csv('Iris.csv')
data.fillna(0, inplace=True)
data = (categorical_to_numerical_labels(X=data))
eigenvalues = get_eigenvalues(data)
print(eigenvalues)
# X, y = process_data(data, 'Species', features)
features = ('SepalLengthCm', 'SepalWidthCm')
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot0 the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C), svm.LinearSVC(C=C), svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)
# title for the plots
titles = ('SVC with linear kernel', 'LinearSVC (linear kernel)', 'SVC with RBF kernel',
          'SVC with polynomial (degree 2) kernel',)
# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
plt.show()

# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target
data = pd.read_csv('Iris.csv')
data.fillna(0, inplace=True)
data = (categorical_to_numerical_labels(X=data))
eigenvalues = get_eigenvalues(data)
print(eigenvalues)
# X, y = process_data(data, 'Species', features)
features = ('SepalLengthCm', 'SepalWidthCm')
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot0 the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C), svm.LinearSVC(C=C), svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)
# title for the plots
titles = ('SVC with linear kernel', 'LinearSVC (linear kernel)', 'SVC with RBF kernel',
          'SVC with polynomial (degree 2) kernel')
# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
plt.show()
classifiers = get_classifiers()
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#00FFFF'])
DATABASE = os.path.abspath('C:\\Users\\frano\PycharmProjects\Big\Database\ClassTest\ClassTest.txt')
X = pd.read_csv(DATABASE, usecols=('G1', 'G2', 'G3'), nrows=20)
X.fillna(0, inplace=True)
y = X['G3']
X.drop(['G3'], axis=1, inplace=True)
X = X.as_matrix()
y = y.as_matrix()
print(X, y)
h = 0.01  # step size in the mesh

figure = plt.figure(figsize=(27, 9))

# X, y = set_data()
# X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8, random_state=42)

# just plot0 the dataset first
ax = plt.subplot(1, len(classifiers) + 1, 1)
ax.set_title("Input data")
plot_data(ax, X, y, cm_bright)

classify(classifiers, X)
plt.show()

# load  iris dataset
dataset = datasets.load_iris()
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

iris = load_iris()
# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target
# **"Features"** are also known as predictors, inputs, or attributes. The **"response"** is also known as the target, label, or output.
# check the shapes of X and y
print(X.shape)
print(y.shape)
# **"Observations"** are also known as samples, instances, or records.
# examine the first 5 rows of the feature matrix (including the feature names)
pd.DataFrame(X, columns=iris.feature_names).head()
# examine the response vector
print(y)
# In order to **build a model**, the features must be **numeric**, and every observation must have the **same features in the same order**.
# instantiate the model (with the default parameters)
knn = KNeighborsClassifier()
# fit the model with data (occurs in-place)
knn.fit(X, y)
# predict the response for a new observation
knn.predict([[3, 5, 4, 2]])
# test text for model training (SMS messages)
simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']
# From the [scikit-learn 1](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction):
# > Text Analysis is a major application field for machine learning algorithms. However the raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect **numerical feature vectors with a fixed size** rather than the **raw text documents with variable length**.
# We will use [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) to "convert text into a matrix of token counts":
# import and instantiate CountVectorizer (with the default parameters)
vect = CountVectorizer()
# learn the 'vocabulary' of the training data (occurs in-place)
vect.fit(simple_train)
# examine the fitted vocabulary
vect.get_feature_names()
# transform training data into a 'document-term matrix'
simple_train_dtm = vect.transform(simple_train)
# convert sparse matrix to a dense matrix
simple_train_dtm.toarray()
# examine the vocabulary and document-term matrix together
pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
# From the [scikit-learn 1](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction):
# > In this scheme, features and samples are defined as follows:
# > - Each individual token occurrence frequency (normalized or not) is treated as a **feature**.
# > - The vector of all the token frequencies for a given document is considered a multivariate **sample**.
# > A **corpus of documents** can thus be represented by a matrix with **one row per document** and **one column per token** (e.g. word) occurring in the corpus.
# > We call **vectorization** the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the **Bag of Words** or "Bag of n-grams" representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.
# check the type of the document-term matrix
type(simple_train_dtm)
# examine the sparse matrix contents
print(simple_train_dtm)
# From the [scikit-learn 1](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction):
# > As most documents will typically use a very small subset of the words used in the corpus, the resulting matrix will have **many feature values that are zeros** (typically more than 99% of them).
# > For instance, a collection of 10,000 short text documents (such as emails) will use a vocabulary with a size in the order of 100,000 unique words in total while each document will use 100 to 1000 unique words individually.
# > In order to be able to **store such a matrix in memory** but also to **speed up operations**, implementations will typically use a **sparse representation** such as the implementations available in the `scipy.sparse` package.
# test text for model testing
simple_test = ["please don't call me"]
# In order to **make a prediction**, the new observation must have the **same features as the training observations**, both in number and meaning.
# transform testing data into a document-term matrix (using existing vocabulary)
simple_test_dtm = vect.transform(simple_test)
simple_test_dtm.toarray()
# examine the vocabulary and document-term matrix together
pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names())
# **Summary:**
# - `vect.fit(train)` **learns the vocabulary** of the training data
# - `vect.transform(train)` uses the **fitted vocabulary** to build a document-term matrix from the training data
# - `vect.transform(test)` uses the **fitted vocabulary** to build a document-term matrix from the testing data (and **ignores tokens** it hasn't seen before)
# ## Part 2: Reading a text-based dataset into pandas
# read file into pandas using a relative path
path = 'data/sms.tsv'
sms = pd.read_table(path, header=None, names=['label', 'message'])
# alternative: read file into pandas from a URL
# url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
# sms = pd.read_table(url, header=None, names=['label', 'message'])
# examine the shape
print(sms.shape)
# examine the first 10 rows
print(sms.head(10))
# examine the object_oriented distribution
sms.name.value_counts()
# convert label to a numerical variable
sms['label_num'] = sms.name.map({'ham': 0, 'spam': 1})
# check that the conversion worked
sms.head(10)
# how to define X and y (from the iris data) for use with a MODEL
X = iris.data
y = iris.target
print(X.shape)
print(y.shape)
# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
X = sms.message
y = sms.label_num
print(X.shape)
print(y.shape)
# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# ## Part 3: Vectorizing our dataset
# instantiate the vectorizer
vect = CountVectorizer()
# learn training data vocabulary, then use it to create a document-term matrix
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
# equivalently: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)
# examine the document-term matrix
X_train_dtm
# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm
# ## Part 5: Building and evaluating a model
# We will use [multinomial Naive Bayes](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html):
# > The multinomial Naive Bayes classifier is suitable for classification0 with **discrete features** (e.g., word counts for text classification0). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.
# import and instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()
# train the model using X_train_dtm
nb.fit(X_train_dtm, y_train)
# make object_oriented predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)
# calculate accuracy of object_oriented predictions
metrics.accuracy_score(y_test, y_pred_class)
# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)
# print message text for the false positives (ham incorrectly classified as spam)
# print message text for the false negatives (spam incorrectly classified as ham)
# test false negative
X_test[3132]
# calculate predicted probabilities for X_test_dtm (poorly calibrated)
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob
# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)
# ## Part 6: Comparing models
# We will compare multinomial Naive Bayes with [logistic Regression6](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression):
# > Logistic Regression6, despite its name, is a **linear model for classification0** rather than Regression6. Logistic Regression6 is also known in the literature as logit Regression6, maximum-entropy classification0 (MaxEnt) or the log-linear classifier. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.
# import and instantiate a logistic Regression6 model
logreg = LogisticRegression()
# train the model using X_train_dtm
logreg.fit(X_train_dtm, y_train)
# make object_oriented predictions for X_test_dtm
y_pred_class = logreg.predict(X_test_dtm)
# calculate predicted probabilities for X_test_dtm (well calibrated)
y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]
y_pred_prob
# calculate accuracy
metrics.accuracy_score(y_test, y_pred_class)
# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)
# ## Part 3: Examining a model for further insight
# We will examine the our **trained Naive Bayes model** to calculate the approximate **"spamminess" of each token**.
# store the vocabulary of X_train
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)
# examine the first 50 tokens
print(X_train_tokens[0:50])
# examine the last 50 tokens
print(X_train_tokens[-50:])
# Naive Bayes counts the number of times each token appears in each object_oriented
nb.feature_count_
# rows represent classes, columns represent tokens
nb.feature_count_.shape
# number of times each token appears across all HAM messages
ham_token_count = nb.feature_count_[0, :]
ham_token_count
# number of times each token appears across all SPAM messages
spam_token_count = nb.feature_count_[1, :]
spam_token_count
# create a DataFrame of tokens with their separate ham and spam counts
tokens = pd.DataFrame({'token': X_train_tokens, 'ham': ham_token_count, 'spam': spam_token_count}).set_index('token')
tokens.head()
# examine 5 random DataFrame rows
tokens.sample(5, random_state=6)
# Naive Bayes counts the number of observations in each object_oriented
nb.class_count_
# Before we can calculate the "spamminess" of each token, we need to avoid **dividing by zero** and account for the **object_oriented imbalance**.
# add 1 to ham and spam counts to avoid dividing by 0
tokens['ham'] = tokens.ham + 1
tokens['spam'] = tokens.spam + 1
tokens.sample(5, random_state=6)
# convert the ham and spam counts into frequencies
tokens['ham'] = tokens.ham / nb.class_count_[0]
tokens['spam'] = tokens.spam / nb.class_count_[1]
tokens.sample(5, random_state=6)
# calculate the ratio of spam-to-ham for each token
tokens['spam_ratio'] = tokens.spam / tokens.ham
tokens.sample(5, random_state=6)
# examine the DataFrame sorted by spam_ratio
# note: use sort() instead of sort_values() for pandas 0.16.1 and earlier
tokens.sort_values('spam_ratio', ascending=False)
# look up the spam_ratio for a given token
tokens.loc['dating', 'spam_ratio']
# ## Part 5: Practicing this workflow on another dataset
# Please open the **`exercise.ipynb`** notebook (or the **`exercise.py`** script).
# ## Part 5: Tuning the vectorizer (discussion)
# Thus far, we have been using the default parameters of [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html):
# show default parameters for CountVectorizer
vect
# However, the vectorizer is worth tuning, just like a model is worth tuning! Here are a few parameters that you might want to tune:
# - **stop_words:** string {'english'}, list, or None (default)
#     - If 'english', a built-in stop word list for English is used.
#     - If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
#     - If None, no stop words will be used.
# remove English stop words
vect = CountVectorizer(stop_words='english')
# - **ngram_range:** tuple (min_n, max_n), default=(1, 1)
#     - The lower and upper boundary of the range of n-values for different n-grams to be extracted.
#     - All values of n such that min_n <= n <= max_n will be used.
# include 1-grams and 1-grams
vect = CountVectorizer(ngram_range=(1, 2))
# - **max_df:** float in range [0.0, 1.0] or int, default=1.0
#     - When building the vocabulary, ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
#     - If float, the parameter represents a proportion of documents.
#     - If integer, the parameter represents an absolute count.
# ignore terms that appear in more than 50% of the documents
vect = CountVectorizer(max_df=0.5)
# - **min_df:** float in range [0.0, 1.0] or int, default=1
#     - When building the vocabulary, ignore terms that have a document frequency strictly lower than the given threshold. (This value is also called "cut-off" in the literature.)
#     - If float, the parameter represents a proportion of documents.
#     - If integer, the parameter represents an absolute count.
# only keep terms that appear in at least 1 documents
vect = CountVectorizer(min_df=2)
# **Guidelines for tuning CountVectorizer:**
# - Use your knowledge of the **problem** and the **text**, and your understanding of the **tuning parameters**, to help you decide what parameters to tune and how to tune them.
# - **experiment**, and let the data tell you the best approach!


# Generate sample data
np.random.seed(0)
batch_size = 47
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)
# Compute clustering with Means
k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0
# Compute clustering with MiniBatchKMeans
mbk = MiniBatchKMeans(
    init='k-means++', n_clusters=3, batch_size=batch_size, n_init=10, max_no_improvement=10,
    verbose=0
)
t0 = time.time()
mbk.fit(X)
t_mini_batch = time.time() - t0
# Plot result
fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
# We want to have the same colors for the same aibase from the
# MiniBatchKMeans and the KMeans algorithm. Let's pair the aibase centers per
# closest one.
k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)
order = pairwise_distances_argmin(k_means_cluster_centers, mbk_means_cluster_centers)
# KMeans
ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' % (t_batch, k_means.inertia_))
# MiniBatchKMeans
ax = fig.add_subplot(1, 3, 2)
for k, col in zip(range(n_clusters), colors):
    my_members = mbk_means_labels == order[k]
    cluster_center = mbk_means_cluster_centers[order[k]]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
ax.set_title('MiniBatchKMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' % (t_mini_batch, mbk.inertia_))
# Initialise the different array to all False
different = (mbk_means_labels == 4)
ax = fig.add_subplot(1, 3, 3)
for k in range(n_clusters):
    different += ((k_means_labels == k) != (mbk_means_labels == order[k]))
identic = np.logical_not(different)
ax.plot(X[identic, 0], X[identic, 1], 'w', markerfacecolor='#bbbbbb', marker='.')
ax.plot(X[different, 0], X[different, 1], 'w', markerfacecolor='m', marker='.')
ax.set_title('Difference')
ax.set_xticks(())
ax.set_yticks(())
plt.show()

DATABASE = os.path.abspath('C:\\Users\\frano\PycharmProjects\PASS simplified\Database\Alcohol\mat.txt')
X = pd.read_csv(DATABASE, usecols=('G1', 'G2', 'G3'))
X.fillna(0, inplace=True)
y = X['G3']
X.drop(['G3'], axis=1, inplace=True)

print(__doc__)

# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target


def get_eigenvalues(data, n=None):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n)
    pca.fit(data)
    var = pca.explained_variance_ratio_
    return var


data = pd.read_csv('Iris.csv')
data.fillna(0, inplace=True)
data = (categorical_to_numerical_labels(X=data))
eigenvalues = get_eigenvalues(data)
print(eigenvalues)
# X, y = process_data(data, 'Species', features)


features = ('SepalLengthCm', 'SepalWidthCm')

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot0 the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C), svm.LinearSVC(C=C), svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = (
    'SVC with linear kernel', 'LinearSVC (linear kernel)', 'SVC with RBF kernel',
    'SVC with polynomial (degree 2) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plt.show()

print(__doc__)


def unimportant(ax):
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot0 in
    Parameters:
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns:
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy):
    """Plot the decision boundaries for a classifier.

    Parameters:
    ax: matplotlib axes object
    clf: a classifier
    xx, yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    return out


# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target
data = pd.read_csv(r'C:\Users\frano\PycharmProjects\Big\iris')
X = data.iloc[:, :2].as_matrix()
y = data.iloc[:, 2].as_matrix()

models = (svm.SVC(kernel='linear'), svm.LinearSVC(), svm.SVC(kernel='rbf', gamma=0.7), svm.SVC(kernel='poly', degree=3))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel', 'LinearSVC (linear kernel)', 'SVC with RBF kernel',
          'SVC with polynomial (degree 2) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    unimportant(ax)

plt.show()


def test_markov_chain():
    initialState = pd.Series([1, 0, 0])
    transitionMatrix = pd.DataFrame([[0.5, 0.4, 0.1], [0.5, 0.2, 0.3], [0.1, 0.3, 0.6]])
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
    z = np.linalg.solve(x, y)
    print(z)
    for i in range(1000):
        initialState = initialState.dot(transitionMatrix)
    print(initialState)


def test_bokeh_0():
    output_file("../../app/graph/bars.html")
    data = pd.DataFrame([[1, 3], [2, 4], [1, 4]], columns=('index', 'day'))
    p = figure()
    p.circle(x=data.iloc[:, 0], y=data.iloc[:, 1])
    labels = {0: 'a', 1: 'b', 2: 'c'}
    lozoya.data_api.html_api.html_apicrawl.formatter = FuncTickFormatter(
        code="""
        var labels = %s;
        return labels[tick]""" % labels
    )
    show(p)


def test_bokeh_1():
    N = 1
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(x, y)
    d = xx
    p = figure(x_range=(0, 10), y_range=(0, 10))
    # must give a vector of image data for image parameter
    p.image(image=[d], x=0, y=0, dw=10, dh=10, palette=['rgb(0,0,0,0.2)'])
    output_file("image.html", title="image.py test")
    show(p)  # open a browser


initialState = pd.Series([1, 0, 0])
transitionMatrix = pd.DataFrame([[0.5, 0.4, 0.1], [0.5, 0.2, 0.3], [0.1, 0.3, 0.6]])
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
z = np.linalg.solve(x, y)
print(z)

for i in range(1000):
    initialState = initialState.dot(transitionMatrix)

print(initialState)


def unimportant(ax):
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot0 in
    Parameters:
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional
    Returns:
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy):
    """Plot the decision boundaries for a classifier.
    Parameters:
    ax: matplotlib axes object
    clf: a classifier
    xx, yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    return out


X, y, _, _ = dh.process_classification_data('', '', '', '')
models = (svm.SVC(kernel='linear'), svm.LinearSVC(), svm.SVC(kernel='rbf', gamma=0.7), svm.SVC(kernel='poly', degree=3))
models = (clf.fit(X, y) for clf in models)
# title for the plots
titles = ('SVC with linear kernel', 'LinearSVC (linear kernel)', 'SVC with RBF kernel',
          'SVC with polynomial (degree 2) kernel')
# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
plt.show()

# For detailed comments on animation and the techniqes used here, see
# the wiki entry http://www.scipy.org/Cookbook/Matplotlib/Animations


ITERS = 1000


class BlitQT(FigureCanvas):

    def __init__(self):
        FigureCanvas.__init__(self, Figure())

        self.ax = self.figure.add_subplot(111)
        self.ax.grid()
        self.draw()

        self.oldSize = self.ax.bbox.width, self.ax.bbox.height
        self.axBackground = self.copy_from_bbox(self.ax.bbox)
        self.cnt = 0

        self.x = np.arange(0, 2 * np.pi, 0.01)
        self.sin_line, = self.ax.plot(self.x, np.sin(self.x), animated=True)
        self.cos_line, = self.ax.plot(self.x, np.cos(self.x), animated=True)
        self.draw()

        self.tstart = time.time()
        self.startTimer(10)

    def timerEvent(self, evt):
        currentSize = self.ax.bbox.width, self.ax.bbox.height
        if self.oldSize != currentSize:
            self.oldSize = currentSize
            self.ax.clear()
            self.ax.grid()
            self.draw()
            self.axBackground = self.copy_from_bbox(self.ax.bbox)

        self.restore_region(self.axBackground)

        # update the data
        self.sin_line.set_ydata(np.sin(self.x + self.cnt / 10.0))
        self.cos_line.set_ydata(np.cos(self.x + self.cnt / 10.0))
        # just draw the animated artist
        self.ax.draw_artist(self.sin_line)
        self.ax.draw_artist(self.cos_line)
        # just redraw the axes rectangle
        self.blit(self.ax.bbox)

        if self.cnt == 0:
            # TODO: this shouldn't be necessary, but if it is excluded the
            # canvas outside the axes is not initially painted.
            self.draw()
        if self.cnt == ITERS:
            # print the timing info and quit
            print('FPS:', ITERS / (time.time() - self.tstart))
            sys.exit()
        else:
            self.cnt += 1


# app = QtGui.QApplication(sys.argv)
# widget = BlitQT()
# widget.show()
# sys.exit(app.exec_())


if __name__ == "__main__":
    class Main:
        def __init__(self):
            self.app = QtWidgets.QApplication(sys.argv)
            BlitQT().show()
            sys.exit(self.app.exec_())


    Main()

N = 1
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
xx, yy = np.meshgrid(x, y)
d = xx

p = figure(x_range=(0, 10), y_range=(0, 10))

# must give a vector of image data for image parameter
p.image(image=[d], x=0, y=0, dw=10, dh=10, palette=['rgb(0,0,0,0.2)'])

output_file("image.html", title="image.py test")

show(p)  # open a browser

DATABASE = os.path.abspath('C:\\Users\\frano\PycharmProjects\PASS simplified\Database\Alcohol\mat.txt')
X = pd.read_csv(DATABASE, usecols=('G1', 'G2', 'G3'))
X.fillna(0, inplace=True)
y = X['G3']
X.drop(['G3'], axis=1, inplace=True)


def set_data():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return X, y


def get_datasets():
    return [make_moons(noise=0.3, random_state=0), make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]


def get_names():
    return ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process", "Decision Tree", "Random Forest",
            "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]


def plot_settings(ax, xx, yy):
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())


# CLUSTERING
def predicting(models, X_plot, index):
    y_models = []
    for model in models:
        y_models.append(models[model].predict(X_plot))
    prediction = pd.DataFrame(np.array(y_models).T, columns=models, index=index)
    return prediction


def run(widget, dir, index, column, target, models, scaler, mode):
    x, y, xKeys, yKeys = dh.process_regression(dir, index, column, target, mode)
    widget.progressBar.setValue(25)
    scaledX, sclr = scale(scaler, x)
    widget.progressBar.setValue(50)
    results = dh.sort_data(regression(models, index=y.index, x=scaledX, y=y.as_matrix().astype(float)))
    widget.progressBar.setValue(75)
    return x, y, results, xKeys, yKeys


def models(checks, options):
    models = {}

    if 'Affinity Propagation' in checks:
        settings = options['Affinity Propagation']
        apc = AffinityPropagation(**settings)
        models['Affinity Propagation'] = apc

    if 'Agglomerative Clustering' in checks:
        settings = options['Agglomerative Clustering']
        acc = AgglomerativeClustering(**settings)
        models['Agglomerative Clustering'] = acc

    if 'Birch Clustering' in checks:
        settings = options['Birch Clustering']
        bcc = Birch(**settings)
        models['Birch Clustering'] = bcc

    if 'DBSCAN' in checks:
        settings = options['DBSCAN']
        dbc = DBSCAN(**settings)
        models['DBSCAN'] = dbc

    if 'Gaussian Mixtures' in checks:
        settings = options['Gaussian Mixtures']
        gmc = GaussianMixture(**settings)
        models['Gaussian Mixtures'] = gmc

    if 'K-Means' in checks:
        settings = options['K-Means']
        kmc = KMeans(**settings)
        models['K-Means'] = kmc

    if 'Mean Shift' in checks:
        settings = options['Mean Shift']
        msc = MeanShift(**settings)
        models['Mean Shift'] = msc

    if 'Spectral Clustering' in checks:
        settings = options['Spectral Clustering']
        scc = SpectralClustering(**settings)
        models['Spectral Clustering'] = scc

    if 'Ward Hierarchical Clustering' in checks:
        settings = options['Ward Hierarchical Clustering']
        whc = AgglomerativeClustering(**settings)
        models['Ward Hierarchical Clustering'] = whc

    return models


# CROSS VALIDATION
def regression(checks):
    models = {}
    if checks['lsvr']:
        lsvr = GridSearchCV(LinearSVR(), cv=5, param_grid={})
        models['lsvr'] = lsvr

    if checks['svs']:
        svs = GridSearchCV(
            SVR(kernel='rbf', gamma='auto'), cv=5,
            param_grid={"C": [1e0], "gamma": np.logspace(-2, 2, 5), "degree": [1, 2]}
        )
        models['svs'] = svs

    if checks['svr']:
        svr = GridSearchCV(
            SVR(kernel='sigmoid', gamma='auto'), cv=5,
            param_grid={  # "kernel": ['linear', 'rbf', 'sigmoid'],
                "gamma": np.logspace(-2, 2, 5)
            }
        )
        models['svr'] = svr

    if checks['kr']:
        kr = GridSearchCV(
            KernelRidge(kernel='rbf', gamma=0.1), cv=5,
            param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)}
        )
        models['kr'] = kr

    if checks['dtr']:
        dtr = GridSearchCV(
            DecisionTreeRegressor(), cv=5,
            param_grid={
                "max_depth":    [2, 5, 20], "min_samples_leaf": [5, 10, 20],
                "max_features": ['auto', 'sqrt', 'log2']
            }
        )
        models['dtr'] = dtr

    if checks['nnr']:
        nnr = GridSearchCV(
            MLPRegressor(), cv=5, param_grid={
                "activation": ['identity', 'logistic', 'tanh', 'relu'],
                "solver":     ['lbfgs', 'sgd', 'adam'],
                # "learning_rate": ['constant', 'invscaling', 'adaptive'],
                # "alpha": [0.01, 0.001, 0.0001],
                "max_iter":   [100]
            }
        )
        models['nnr'] = nnr

    if checks['knn']:
        knn = GridSearchCV(KNeighborsRegressor(), cv=5, param_grid={})
        models['knn'] = knn

    if checks['rfr']:
        rfr = GridSearchCV(
            RandomForestRegressor(n_jobs=-1), cv=5, param_grid={  # "n_estimators": [1, 5],
                "max_features": ['auto', 'sqrt', 'log2'],  # "criterion": ['mse', 'mae']
            }
        )
        models['rfr'] = rfr

    return models


# SCALING
def scaler(scaler):
    if scaler == 'None':
        return None
    elif scaler == 'MinMax':
        return MinMaxScaler(copy=False)
    elif scaler == 'MaxAbs':
        return MaxAbsScaler(copy=False)
    elif scaler == 'Normalizer':
        return Normalizer(copy=False)
    elif scaler == 'QuantileTransformer':
        return QuantileTransformer(copy=False)
    elif scaler == 'Robust':
        return RobustScaler(copy=False)
    elif scaler == 'Standard':
        return StandardScaler(copy=False)


def scale(scaler, data):
    if scaler != None:
        scaled = scaler.fit_transform(data.astype(float).as_matrix())
        d = pd.DataFrame(scaled, index=data.index, columns=data.columns, dtype=float).as_matrix()
        return d, scaler
    return data.astype(float).as_matrix(), None


def unscale(scaler, data):
    if scaler != None:
        d = data
        for j, result in enumerate(data.columns):
            k = pd.DataFrame(data.iloc[:, j].values, index=data.index, columns=(1,))
            k.reset_index(inplace=True)
            unscaled = scaler.inverse_transform(k.as_matrix())
            k = pd.DataFrame(unscaled)
            k.set_index(0, drop=True, inplace=True)
            for i, value in enumerate(data.index):
                d.iloc[i, j] = k.iloc[i, 0]
        return d
    return data

'''
