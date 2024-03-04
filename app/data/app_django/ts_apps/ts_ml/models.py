import django.db.models
import app.data.app_django.tensorstone.models as ts_general_models
import configuration
import lozoya.data
import lozoya.text


class TSModelML(ts_general_models.TSModelAppSingleInput):
    """
    Basic class for machine learning app.
    Contains properties and methods required by all machine learning apps.
    Child classes must override the model property with the appropriate model.
    """
    modelillo = None
    x = django.db.models.TextField(
        blank=True,
        null=True,
    )
    y = django.db.models.TextField(
        blank=True,
        null=True,
    )
    scaler = django.db.models.CharField(
        default='None',
        max_length=15,
        choices=configuration.scalerChoices,
    )
    yFit = django.db.models.TextField(
        blank=True,
        null=True,
    )

    def set_outdated(self):
        self.yFit = None
        self.save()

    def set_uptodate(self):
        self.save()

    def fit_model(self):
        data = lozoya.data.get_xy(
            path=self.input.output,
            parameters={
                'x': lozoya.text.str_to_list(self.x),
                'y': self.y,
            }
        )
        X, y = data['x'].values, data['y'].values
        if self.scaler:
            _scaler = configuration.scalers[self.scaler]()
            X = _scaler.fit_transform(X)
        self.yFit = self.modelillo.fit(X, y)
        self.set_uptodate()


class TSModelMLClassification(TSModelML):
    pass


class TSModelMLClustering(TSModelML):
    pass


class TSModelMLDimensionalityReduction(TSModelML):
    pass


class TSModelMLRegression(TSModelML):
    def __init__(self, *args, **kwargs):
        super(TSModelMLRegression, self).__init__(*args, **kwargs)
        self.objectType = 'regression'


class TSModelMLDeepNeuralNetwork(TSModelML):
    pass


import uuid

import django.db
import ts_ml.models
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


class TSBaseClassification(BaseMLApp):
    def __init__(self, *args, **kwargs):
        super(TSBaseClassification, self).__init__(*args, **kwargs)
        self.objectType = 'classification'


class TSBaseClustering(BaseMLApp):
    def __init__(self, *args, **kwargs):
        super(TSBaseClustering, self).__init__(*args, **kwargs)
        self.objectType = 'clustering'


class TSBaseML(ts_general_models.TSModelAppSingleInput):
    """
    Basic class for machine learning app.
    Contains properties and methods required by all machine learning apps.
    Child classes must override the model property with the appropriate model.
    """
    modelillo = None
    x = django.db.models.TextField(blank=True, null=True, )
    y = django.db.models.TextField(blank=True, null=True, )
    scaler = django.db.models.CharField(default='None', max_length=15, choices=_deprecated.py.choices.scalerChoices, )
    yFit = django.db.models.TextField(blank=True, null=True, )

    def set_outdated(self):
        self.yFit = None
        self.save()

    def set_uptodate(self):
        self.save()

    def fit_model(self):
        data = ts_util.processor.get_xy(
            path=self.input.output,
            parameters={
                'x': lozoya.text.str_to_list(self.x),
                'y': self.y,
            }
        )
        X, y = data['x'].values, data['y'].values
        if self.scaler:
            _scaler = _deprecated.py.ml_general.scalers[self.scaler]()
            X = _scaler.fit_transform(X)
        self.yFit = self.modelillo.fit(X, y)
        self.set_uptodate()


class TSBaseRegression(TSBaseML):
    def __init__(self, *args, **kwargs):
        super(TSBaseRegression, self).__init__(*args, **kwargs)
        self.objectType = 'regression'


class TSRegressionModel(django.db.Model):
    """
    lr = Linear Regression
    gp = Gaussian Process
    rf = Random Forest
    svm = Support Vector Machine
    """
    modelID = django.db.models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, )
    dashboardID = django.db.models.UUIDField(editable=False)
    n = django.db.models.IntegerField()
    appType = django.db.models.TextField(default='regression', editable=False)
    testDataFile = django.db.models.CharField(max_length=100)
    trainDataFile = django.db.models.CharField(max_length=100)
    scaler = django.db.models.CharField(default='None', max_length=15, )
    x = django.db.models.TextField()
    y = django.db.models.TextField()
    algorithms = django.db.models.TextField()
    plotDim = django.db.models.IntegerField(default=2)
    plotx = django.db.models.CharField(max_length=100, default='')
    ploty = django.db.models.CharField(max_length=100, default='')
    plotz = django.db.models.CharField(max_length=100, default='')
    fit = django.db.models.BooleanField(default=False)

    # LINEAR REGRESSION
    lrFitIntercept = django.db.models.BooleanField(default=True, verbose_name="Fit Intercept", )
    lrNormalize = django.db.models.BooleanField(default=False, verbose_name="Normalize", )

    # GAUSSIAN PROCESS REGRESSION
    gpAlpha = django.db.models.FloatField(default=1e-10, verbose_name="Alpha", )
    gpNRestartsOptimizer = django.db.models.IntegerField(default=0, verbose_name="n Optimizer Restarts", )
    gpNormalizeY = django.db.models.BooleanField(default=False, verbose_name="Normalize Y", )
    gpRandomState = django.db.models.IntegerField(null=True, blank=True, verbose_name="Random State", )

    # K NEAREST NEIGHBORS
    knnNNeighbors = django.db.models.IntegerField(default=5, verbose_name='Neighbors', )
    knnWeights = django.db.models.CharField(max_length=20, default='uniform', verbose_name='Weights', )
    knnAlgorithm = django.db.models.CharField(max_length=20, default='auto', verbose_name='Algorithm', )
    knnLeafSize = django.db.models.IntegerField(default=30, verbose_name='Leaf Size', )
    knnP = django.db.models.IntegerField(default=2, verbose_name='Minkowski Power', )
    knnDistanceMetric = django.db.models.CharField(max_length=20, default='minkowski', verbose_name='Distance Metric', )

    # RANDOM FOREST REGRESSION
    rfNEstimators = django.db.models.IntegerField(default=10, verbose_name="n Estimators", )
    rfCriterion = django.db.models.CharField(default='mse', max_length=20, verbose_name="Criterion", )
    rfMaxDepth = django.db.models.IntegerField(null=True, verbose_name="Maximum Depth", )
    rfMinSamplesSplit = django.db.models.FloatField(default=1e-3, verbose_name="Minimum Samples Split", )
    rfMinSamplesLeaf = django.db.models.FloatField(default=1e-3, verbose_name="Minimum Samples Leaf", )
    rfMinWeightFractionLeaf = django.db.models.FloatField(default=0.0, verbose_name="Minimum Weight Fraction Leaf", )
    rfMaxFeatures = django.db.models.FloatField(null=True, verbose_name="Maximum Features", )
    rfMaxLeafNodes = django.db.models.IntegerField(null=True, verbose_name="Maximum Leaf Nodes", )
    rfMinImpurityDecrease = django.db.models.FloatField(default=0.0, verbose_name="Minimum Impurity Decrease", )
    rfBootstrap = django.db.models.BooleanField(default=True, verbose_name="Bootstrap", )
    rfOutOfBag = django.db.models.BooleanField(default=False, verbose_name="Out of Bag Samples", )
    rfRandomState = django.db.models.IntegerField(null=True, verbose_name="Random State", )
    rfWarmStart = django.db.models.BooleanField(default=False, verbose_name="Warm Start", )

    # SUPPORT VECTOR MACHINE
    svmKernel = django.db.models.CharField(default='rbf', max_length=20, verbose_name="Kernel", )
    svmDegree = django.db.models.IntegerField(default=3, verbose_name="Degree", )
    svmGamma = django.db.models.FloatField(default=1, verbose_name="Gamma", )
    svmCoef0 = django.db.models.FloatField(default=1, verbose_name="Coefficient 0", )
    svmTolerance = django.db.models.FloatField(default=1e-3, verbose_name="Tolerance", )
    svmC = django.db.models.FloatField(default=1.0, verbose_name="C", )
    svmEpsilon = django.db.models.FloatField(default=0.1, verbose_name="Epsilon", )
    svmShrinking = django.db.models.BooleanField(default=True, verbose_name="Shrinking", )
    svmMaximumIterations = django.db.models.IntegerField(default=100, verbose_name="Maximum Iterations", )

    class Meta:
        verbose_name = 'Regression Model'
        verbose_name_plural = 'Regression Models'

    def set_outdated(self):
        self.fit = False
        self.save()

    def set_uptodate(self):
        self.save()


class TSRegressionModel(django.db.Model):
    """
    lr = Linear Regression
    gp = Gaussian Process
    rf = Random Forest
    svm = Support Vector Machine
    """
    id = django.db.models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    testDataFile = django.db.models.CharField(max_length=100)
    trainDataFile = django.db.models.CharField(max_length=100)
    scaler = django.db.models.CharField(default='None', max_length=15)
    x = django.db.models.TextField()
    y = django.db.models.TextField()
    algorithms = django.db.models.TextField()


class TSRegressionLinear(ts_ml.models.TSBaseRegression):
    modelillo = LinearRegression
    lrFitIntercept = django.db.models.BooleanField(default=True, verbose_name="Fit Intercept", )
    lrNormalize = django.db.models.BooleanField(default=False, verbose_name="Normalize", )

    class Meta:
        verbose_name = 'Regression Linear'
        verbose_name_plural = 'Regression Linear'


class TSRegressionGP(ts_ml.models.TSBaseRegression):
    modelillo = GaussianProcessRegressor
    gpAlpha = django.db.models.FloatField(default=1e-10, verbose_name="Alpha", )
    gpNRestartsOptimizer = django.db.models.IntegerField(default=0, verbose_name="n Optimizer Restarts", )
    gpNormalizeY = django.db.models.BooleanField(default=False, verbose_name="Normalize Y", )
    gpRandomState = django.db.models.IntegerField(null=True, blank=True, verbose_name="Random State", )

    class Meta:
        verbose_name = 'Regression Gaussian Process'
        verbose_name_plural = 'Regression Gaussian Process'


class TSRegressionKNN(ts_ml.models.TSBaseRegression):
    modelillo = KNeighborsRegressor
    knnNNeighbors = django.db.models.IntegerField(default=5, verbose_name='Neighbors', )
    knnWeights = django.db.models.CharField(max_length=20, default='uniform', verbose_name='Weights', )
    knnAlgorithm = django.db.models.CharField(max_length=20, default='auto', verbose_name='Algorithm', )
    knnLeafSize = django.db.models.IntegerField(default=30, verbose_name='Leaf Size', )
    knnP = django.db.models.IntegerField(default=2, verbose_name='Minkowski Power', )
    knnDistanceMetric = django.db.models.CharField(max_length=20, default='minkowski', verbose_name='Distance Metric', )

    class Meta:
        verbose_name = 'Regression K Nearest Neighbors'
        verbose_name_plural = 'Regression K Nearest Neighbors'


class TSRegressionRF(ts_ml.models.TSBaseRegression):
    modelillo = RandomForestRegressor
    rfNEstimators = django.db.models.IntegerField(default=10, verbose_name="n Estimators", )
    rfCriterion = django.db.models.CharField(default='mse', max_length=20, verbose_name="Criterion", )
    rfMaxDepth = django.db.models.IntegerField(null=True, verbose_name="Maximum Depth", )
    rfMinSamplesSplit = django.db.models.FloatField(default=1e-3, verbose_name="Minimum Samples Split", )
    rfMinSamplesLeaf = django.db.models.FloatField(default=1e-3, verbose_name="Minimum Samples Leaf", )
    rfMinWeightFractionLeaf = django.db.models.FloatField(default=0.0, verbose_name="Minimum Weight Fraction Leaf", )
    rfMaxFeatures = django.db.models.FloatField(null=True, verbose_name="Maximum Features", )
    rfMaxLeafNodes = django.db.models.IntegerField(null=True, verbose_name="Maximum Leaf Nodes", )
    rfMinImpurityDecrease = django.db.models.FloatField(default=0.0, verbose_name="Minimum Impurity Decrease", )
    rfBootstrap = django.db.models.BooleanField(default=True, verbose_name="Bootstrap", )
    rfOutOfBag = django.db.models.BooleanField(default=False, verbose_name="Out of Bag Samples", )
    rfRandomState = django.db.models.IntegerField(null=True, verbose_name="Random State", )
    rfWarmStart = django.db.models.BooleanField(default=False, verbose_name="Warm Start", )

    class Meta:
        verbose_name = 'Regression Random Forest'
        verbose_name_plural = 'Regression Random Forest'


class TSRegressionSVM(ts_ml.models.TSBaseRegression):
    modelillo = SVR
    svmKernel = django.db.models.CharField(default='rbf', max_length=20, verbose_name="Kernel", )
    svmDegree = django.db.models.IntegerField(default=3, verbose_name="Degree", )
    svmGamma = django.db.models.FloatField(default=1, verbose_name="Gamma", )
    svmCoef0 = django.db.models.FloatField(default=1, verbose_name="Coefficient 0", )
    svmTolerance = django.db.models.FloatField(default=1e-3, verbose_name="Tolerance", )
    svmC = django.db.models.FloatField(default=1.0, verbose_name="C", )
    svmEpsilon = django.db.models.FloatField(default=0.1, verbose_name="Epsilon", )
    svmShrinking = django.db.models.BooleanField(default=True, verbose_name="Shrinking", )
    svmMaximumIterations = django.db.models.IntegerField(default=100, verbose_name="Maximum Iterations", )

    class Meta:
        verbose_name = 'Regression Support Vector Machine'
        verbose_name_plural = 'Regression Support Vector Machine'


objectTypes = {

}
propertyDict = {

}


# class BaseClassificationApp(BaseMLApp):
#     def __init__(self, *args, **kwargs):
#         super(BaseClassificationApp, self).__init__(*args, **kwargs)
#         self.objectType = 'classification'
#
#
# class BaseClusteringApp(BaseMLApp):
#     def __init__(self, *args, **kwargs):
#         super(BaseClusteringApp, self).__init__(*args, **kwargs)
#         self.objectType = 'clustering'

# class BaseClassificationApp(BaseMLApp):
#     def __init__(self, *args, **kwargs):
#         super(BaseClassificationApp, self).__init__(*args, **kwargs)
#         self.objectType = 'classification'
#
#
# class BaseClusteringApp(BaseMLApp):
#     def __init__(self, *args, **kwargs):
#         super(BaseClusteringApp, self).__init__(*args, **kwargs)
#         self.objectType = 'clustering'

class TSRegressionModel(django.db.models.Model):
    """
    lr = Linear Regression
    gp = Gaussian Process
    rf = Random Forest
    svm = Support Vector Machine
    """
    id = django.db.models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    testDataFile = django.db.models.CharField(max_length=100)
    trainDataFile = django.db.models.CharField(max_length=100)
    scaler = django.db.models.CharField(
        default='None',
        max_length=15
    )
    x = django.db.models.TextField()
    y = django.db.models.TextField()
    algorithms = django.db.models.TextField()
    # LINEAR REGRESSION
    lrFitIntercept = django.db.models.BooleanField(
        default=True,
        verbose_name="Fit Intercept"
    )
    lrNormalize = django.db.models.BooleanField(
        default=False,
        verbose_name="Normalize"
    )
    # GAUSSIAN PROCESS REGRESSION
    gpAlpha = django.db.models.FloatField(
        default=1e-10,
        verbose_name="Alpha"
    )
    gpNRestartsOptimizer = django.db.models.IntegerField(
        default=0,
        verbose_name="n Optimizer Restarts"
    )
    gpNormalizeY = django.db.models.BooleanField(
        default=False,
        verbose_name="Normalize Y"
    )
    gpRandomState = django.db.models.IntegerField(
        null=True,
        blank=True,
        verbose_name="Random State"
    )
    # K NEAREST NEIGHBORS
    knnNNeighbors = django.db.models.IntegerField(
        default=5,
        verbose_name='Neighbors'
    )
    knnWeights = django.db.models.CharField(
        max_length=20,
        default='uniform',
        verbose_name='Weights'
    )
    knnAlgorithm = django.db.models.CharField(
        max_length=20,
        default='auto',
        verbose_name='Algorithm'
    )
    knnLeafSize = django.db.models.IntegerField(
        default=30,
        verbose_name='Leaf Size'
    )
    knnP = django.db.models.IntegerField(
        default=2,
        verbose_name='Minkowski Power'
    )
    knnDistanceMetric = django.db.models.CharField(
        max_length=20,
        default='minkowski',
        verbose_name='Distance Metric'
    )
    # RANDOM FOREST REGRESSION
    rfNEstimators = django.db.models.IntegerField(
        default=10,
        verbose_name="n Estimators"
    )
    rfCriterion = django.db.models.CharField(
        default='mse',
        max_length=20,
        verbose_name="Criterion"
    )
    rfMaxDepth = django.db.models.IntegerField(
        null=True,
        verbose_name="Maximum Depth"
    )
    rfMinSamplesSplit = django.db.models.FloatField(
        default=1e-3,
        verbose_name="Minimum Samples Split"
    )
    rfMinSamplesLeaf = django.db.models.FloatField(
        default=1e-3,
        verbose_name="Minimum Samples Leaf"
    )
    rfMinWeightFractionLeaf = django.db.models.FloatField(
        default=0.0,
        verbose_name="Minimum Weight Fraction Leaf"
    )
    rfMaxFeatures = django.db.models.FloatField(
        null=True,
        verbose_name="Maximum Features"
    )
    rfMaxLeafNodes = django.db.models.IntegerField(
        null=True,
        verbose_name="Maximum Leaf Nodes"
    )
    rfMinImpurityDecrease = django.db.models.FloatField(
        default=0.0,
        verbose_name="Minimum Impurity Decrease"
    )
    rfBootstrap = django.db.models.BooleanField(
        default=True,
        verbose_name="Bootstrap"
    )
    rfOutOfBag = django.db.models.BooleanField(
        default=False,
        verbose_name="Out of Bag Samples"
    )
    rfRandomState = django.db.models.IntegerField(
        null=True,
        verbose_name="Random State"
    )
    rfWarmStart = django.db.models.BooleanField(
        default=False,
        verbose_name="Warm Start"
    )
    # SUPPORT VECTOR MACHINE
    svmKernel = django.db.models.CharField(
        default='rbf',
        max_length=20,
        verbose_name="Kernel"
    )
    svmDegree = django.db.models.IntegerField(
        default=3,
        verbose_name="Degree"
    )
    svmGamma = django.db.models.FloatField(
        default=1,
        verbose_name="Gamma"
    )
    svmCoef0 = django.db.models.FloatField(
        default=1,
        verbose_name="Coefficient 0"
    )
    svmTolerance = django.db.models.FloatField(
        default=1e-3,
        verbose_name="Tolerance"
    )
    svmC = django.db.models.FloatField(
        default=1.0,
        verbose_name="C"
    )
    svmEpsilon = django.db.models.FloatField(
        default=0.1,
        verbose_name="Epsilon"
    )
    svmShrinking = django.db.models.BooleanField(
        default=True,
        verbose_name="Shrinking"
    )
    svmMaximumIterations = django.db.models.IntegerField(
        default=100,
        verbose_name="Maximum Iterations"
    )

    class Meta:
        verbose_name = 'Regression Model'
        verbose_name_plural = 'Regression Models'
