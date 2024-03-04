from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, QuantileTransformer, RobustScaler, \
    StandardScaler  # , PowerTransformer

from regression.forms import *
from regression.models import *

x = 'ts-model-x-{}'
y = 'ts-model-y-{}'
algorithms = 'ts-model-algorithms-{}'
settings = 'ts-model-settings-{}'
scaler = 'ts-model-scaler-{}'
yContainer = 'ts-model-y-container-{}'
mlModels = {
    'LinearRegression':               LinearRegression,
    'GaussianProcessRegression':      GaussianProcessRegressor,
    'KNearestNeighborsRegression':    KNeighborsRegressor,
    'RandomForestRegression':         RandomForestRegressor,
    'SupportVectorMachineRegression': SVR,
}
modelAbbreviations = {
    'LinearRegression':               'lr',
    'GaussianProcessRegression':      'gp',
    'KNearestNeighborsRegression':    'knn',
    'RandomForestRegression':         'rf',
    'SupportVectorMachineRegression': 'svm'
}
modelClasses = {
    'Regression':              TSRegressionModel,
    'Classification':          None,
    'Clustering':              None,
    'DimensionalityReduction': None
}
modelNames = {
    'LinearRegression':               'Linear',
    'GaussianProcessRegression':      'Gaussian Process',
    'KNearestNeighborsRegression':    'K Nearest Neighbors',
    'RandomForestRegression':         'Random Forest',
    'SupportVectorMachineRegression': 'Support Vector Machine'
}
modelSettingsForms = {
    'LinearRegression':               RegressionLinearForm,
    'GaussianProcessRegression':      RegressionGPForm,
    'KNearestNeighborsRegression':    RegressionKNNForm,
    'RandomForestRegression':         RegressionRFForm,
    'SupportVectorMachineRegression': RegressionSVMForm,
}
scalers = {
    'None':           None,
    'MaxAbsScaler':   MaxAbsScaler,
    'MinMaxScaler':   MinMaxScaler,
    # 'Normalizer':          Normalizer,
    # 'QuantileTransformer': QuantileTransformer,
    'RobustScaler':   RobustScaler,
    'StandardScaler': StandardScaler,
    # 'PowerTransformer': PowerTransformer,
}
scalerChoices = (
    ('None', 'None'),
    ('MaxAbsScaler', 'Max Abs'),
    ('MinMaxScaler', 'Min Max'),
    # ('Normalizer', 'Normalize'),
    # ('QuantileTransformer', 'Quantile Transform'),
    ('RobustScaler', 'Robust'),
    ('StandardScaler', 'Standard'),
    # ('PowerTransformer', 'Power Transform'),
)
modelChoicesClassification = {
    '': ''
}

kwargTranslation = {
    # LINEAR REGRESSION
    'lrFitIntercept':          'fit_intercept',
    'lrNormalize':             'normalize',
    # GAUSSIAN PROCESS REGRESSION
    'gpKernel':                'kernel',
    'gpAlpha':                 'alpha',
    'gpOptimizer':             'optimizer',
    'gpNRestartsOptimizer':    'n_restarts_optimizer',
    'gpNormalizeY':            'normalize_y',
    'gpRandomState':           'random_state',
    # K NEAREST NEIGHBORS
    'knnNNeighbors':           'n_neighbors',
    'knnWeights':              'weights',
    'knnAlgorithm':            'algorithm',
    'knnLeafSize':             'leaf_size',
    'knnP':                    'p',
    'knnDistanceMetric':       'metric',
    # RANDOM FOREST
    'rfNEstimators':           'n_estimators',
    'rfCriterion':             'criterion',
    'rfMaxDepth':              'max_depth',
    'rfMinSamplesSplit':       'min_samples_split',
    'rfMinSamplesLeaf':        'min_samples_leaf',
    'rfMinWeightFractionLeaf': 'min_weight_fraction_leaf',
    'rfMaxFeatures':           'max_features',
    'rfMaxLeafNodes':          'max_leaf_nodes',
    'rfMinImpurityDecrease':   'min_impurity_decrease',
    'rfBootstrap':             'bootstrap2',
    'rfOutOfBag':              'oob_score',
    'rfRandomState':           'random_state',
    'rfWarmStart':             'warm_start',
    # SUPPORT VECTOR MACHINE
    'svmKernel':               'kernel',
    'svmDegree':               'degree',
    'svmGamma':                'gamma',
    'svmCoef0':                'coef0',
    'svmTolerance':            'tol',
    'svmC':                    'C',
    'svmEpsilon':              'epsilon',
    'svmShrinking':            'shrinking',
    'svmMaximumIterations':    'max_iter'
}
# K NEAREST NEIGHBORS
knnDistanceMetricChoices = (
    ('euclidean', 'Euclidean'),
    ('manhattan', 'Manhattan'),
    ('chebyshev', 'Chebyshev'),
    ('minkowski', 'Minkowski'),
)
knnAlgorithmChoices = (
    ('ball_tree', 'Ball Tree'),
    ('kd_tree', 'KD Tree'),
    ('brute', 'Brute Force'),
    ('auto', 'Automatic'),
)
knnWeightsChoices = (
    ('uniform', 'Uniform'),
    ('distance', 'Distance'),
)
# RANDOM FOREST
rfCriterionChoices = (
    ('mse', 'Mean Squared Error'),
    ('mae', 'Mean Absolute Error'),
)
# SUPPORT VECTOR MACHINE
svmKernelChoices = (
    ('rbf', 'Radial Basis Function'),
    ('linear', 'Linear'),
    ('poly', 'Polynomial'),
    ('sigmoid', 'Sigmoid'),
    # ('precomputed', 'Precomputed'),
)
# K NEAREST NEIGHBORS
knnWeightsChoices = (
    ('uniform', 'Uniform'),
    ('distance', 'Distance'),
)
# RANDOM FOREST
rfCriterionChoices = (
    ('mse', 'Mean Squared Error'),
    ('mae', 'Mean Absolute Error'),
)
# K NEAREST NEIGHBORS
knnWeightsChoices = (
    ('uniform', 'Uniform'),
    ('distance', 'Distance'),
)
# RANDOM FOREST
rfCriterionChoices = (
    ('mse', 'Mean Squared Error'),
    ('mae', 'Mean Absolute Error'),
)
modelChoices = [(key, modelNames[key]) for key in sorted(modelNames.keys())]
