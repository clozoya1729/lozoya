import sklearn.cluster
import sklearn.datasets
import sklearn.decomposition
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

try:
    import sklearn.ensemble
    import sklearn.covariance
    import sklearn.discriminant_analysis
    import sklearn.gaussian_process
    import sklearn.isotonic
    import sklearn.impute
    import sklearn.kernel_ridge
    import sklearn.naive_bayes
    import sklearn.neural_network
except:
    pass


class SklearnModels:
    covariance = [
        sklearn.covariance.EmpiricalCovariance,
        sklearn.covariance.EllipticEnvelope,
        sklearn.covariance.GraphicalLasso,
        sklearn.covariance.LedoitWolf,
        sklearn.covariance.MinCovDet,
        sklearn.covariance.OAS,
        sklearn.covariance.ShrunkCovariance,
    ]
    clustering = [
        sklearn.cluster.AffinityPropagation,
        sklearn.cluster.AgglomerativeClustering,
        sklearn.cluster.Birch,
        sklearn.cluster.DBSCAN,
        sklearn.cluster.HDBSCAN,
        sklearn.cluster.FeatureAgglomeration,
        sklearn.cluster.KMeans,
        sklearn.cluster.BisectingKMeans,
        sklearn.cluster.MiniBatchKMeans,
        sklearn.cluster.MeanShift,
        sklearn.cluster.OPTICS,
        sklearn.cluster.SpectralClustering,
        sklearn.cluster.SpectralBiclustering,
        sklearn.cluster.SpectralCoclustering,
    ]
    dataset = [
        sklearn.datasets.load_breast_cancer,
        sklearn.datasets.load_diabetes,
        sklearn.datasets.load_digits,
        # sklearn.datasets.load_files,
        sklearn.datasets.load_iris,
        sklearn.datasets.load_linnerud,
        # sklearn.datasets.load_sample_image,
        sklearn.datasets.load_wine,

    ]
    samplesGenerator = [
        # sklearn.datasets.make_biclusters,
        sklearn.datasets.make_blobs,
        # sklearn.datasets.make_checkerboard,
        sklearn.datasets.make_circles,
        sklearn.datasets.make_classification,
        sklearn.datasets.make_friedman1,
        sklearn.datasets.make_friedman2,
        sklearn.datasets.make_friedman3,
        sklearn.datasets.make_gaussian_quantiles,
        sklearn.datasets.make_hastie_10_2,
        # sklearn.datasets.make_low_rank_matrix,
        sklearn.datasets.make_moons,
        # sklearn.datasets.make_multilabel_classification,
        sklearn.datasets.make_regression,
        # sklearn.datasets.make_s_curve,
        # sklearn.datasets.make_sparse_coded_signal,
        # sklearn.datasets.make_sparse_spd_matrix,
        # sklearn.datasets.make_sparse_uncorrelated,
        # sklearn.datasets.make_spd_matrix,
        # sklearn.datasets.make_swiss_roll,
    ]
    decomposition = [
        sklearn.decomposition.DictionaryLearning,
        sklearn.decomposition.FactorAnalysis,
        sklearn.decomposition.FastICA,
        sklearn.decomposition.IncrementalPCA,
        sklearn.decomposition.KernelPCA,
        sklearn.decomposition.LatentDirichletAllocation,
        sklearn.decomposition.MiniBatchDictionaryLearning,
        sklearn.decomposition.MiniBatchSparsePCA,
        sklearn.decomposition.NMF,
        sklearn.decomposition.MiniBatchNMF,
        sklearn.decomposition.PCA,
        sklearn.decomposition.SparsePCA,
        sklearn.decomposition.SparseCoder,
        sklearn.decomposition.TruncatedSVD,
    ]
    preprocessing = [
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis,
        sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis,
        sklearn.preprocessing.MaxAbsScaler,
        sklearn.preprocessing.MinMaxScaler,
        sklearn.preprocessing.RobustScaler,
        sklearn.preprocessing.StandardScaler,
        sklearn.preprocessing.QuantileTransformer,
        sklearn.preprocessing.PowerTransformer,
        sklearn.preprocessing.Binarizer,
        sklearn.preprocessing.Normalizer,
        sklearn.preprocessing.OneHotEncoder,
        sklearn.preprocessing.OrdinalEncoder,
        sklearn.preprocessing.TargetEncoder,
        sklearn.preprocessing.KBinsDiscretizer,
        sklearn.impute.SimpleImputer,
        sklearn.impute.KNNImputer,
        sklearn.impute.MissingIndicator,
        sklearn.preprocessing.PolynomialFeatures,
        sklearn.preprocessing.SplineTransformer,
    ]
    classification = [
        sklearn.ensemble.AdaBoostClassifier,
        sklearn.tree.DecisionTreeClassifier,
        sklearn.neighbors.KNeighborsClassifier,
        sklearn.gaussian_process.GaussianProcessClassifier,
        sklearn.naive_bayes.GaussianNB,
        sklearn.ensemble.GradientBoostingClassifier,
        sklearn.neural_network.MLPClassifier,
        sklearn.ensemble.RandomForestClassifier,
        sklearn.linear_model.SGDClassifier,
        sklearn.svm.SVC,
    ]
    regression = [
        sklearn.ensemble.AdaBoostRegressor,
        sklearn.linear_model.ARDRegression,
        sklearn.ensemble.BaggingRegressor,
        sklearn.linear_model.BayesianRidge,
        sklearn.tree.DecisionTreeRegressor,
        sklearn.linear_model.ElasticNet,
        sklearn.tree.ExtraTreeRegressor,
        sklearn.ensemble.ExtraTreesRegressor,
        sklearn.linear_model.HuberRegressor,
        sklearn.isotonic.IsotonicRegression,
        sklearn.gaussian_process.GaussianProcessRegressor,
        sklearn.ensemble.GradientBoostingRegressor,
        sklearn.ensemble.HistGradientBoostingRegressor,
        sklearn.kernel_ridge.KernelRidge,
        sklearn.neighbors.KNeighborsRegressor,
        sklearn.linear_model.Lars,
        sklearn.linear_model.Lasso,
        sklearn.linear_model.LassoLars,
        sklearn.linear_model.LinearRegression,
        sklearn.svm.LinearSVR,
        sklearn.linear_model.LogisticRegression,
        sklearn.neural_network.MLPRegressor,
        sklearn.linear_model.OrthogonalMatchingPursuit,
        sklearn.svm.NuSVR,
        sklearn.linear_model.PassiveAggressiveRegressor,
        sklearn.linear_model.Perceptron,
        sklearn.linear_model.QuantileRegressor,
        sklearn.ensemble.RandomForestRegressor,
        sklearn.linear_model.RANSACRegressor,
        sklearn.linear_model.Ridge,
        sklearn.linear_model.SGDRegressor,
        sklearn.svm.SVR,
        sklearn.linear_model.TheilSenRegressor,
        sklearn.linear_model.TweedieRegressor,
    ]
