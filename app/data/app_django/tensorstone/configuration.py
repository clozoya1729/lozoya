import datetime
import os
import sys

import numpy as np
from classification.models import ClassificationModel
from clustering.models import ClusteringModel
from dataprep.models import DataPrepModel
from dimreduction.models import DimReductionModel
from distfit.models import DistFitModel
from dnn.models import DNNModel
from plotter.models import PlotModel
from regression.models import RegressionModel
from stats_analysis.models import StatsAnalysisModel

objectTypes = [
    ('file', 'File'),
    ('generator', 'Generator'),
    ('regression', 'Regression'),
    ('plot', 'Plot'),
]
appTypes = (
    ('plot', 'Plot'),
    ('classification0', 'Classification'),
    ('clustering', 'Clustering'),
    ('dimreduction', 'Dimensionality Reduction'),
    ('regression', 'Regression'),
    ('dnn', 'Deep Neural Network'),
    ('stats', 'Statistics'),
    ('distfit', 'Distribution Fitting'),
    ('dataprep', 'Data Preparation'),
)
objectTypes = (
    ('plot', 'Plot'),
    ('classification', 'Classification'),
    ('clustering', 'Clustering'),
    ('dimreduction', 'Dimensionality Reduction'),
    ('regression', 'regression'),
    ('dnn', 'Deep Neural Network'),
    ('stats', 'Statistics'),
    ('distfit', 'Distribution Fitting'),
    ('dataprep', 'Data Preparation'),
)
appTypes = {
    'regression':      RegressionModel,
    'plot':            PlotModel,
    'dataprep':        DataPrepModel,
    'distfit':         DistFitModel,
    'classification0': ClassificationModel,
    'clustering':      ClusteringModel,
    'dimreduction':    DimReductionModel,
    'dnn':             DNNModel,
    'stats':           StatsAnalysisModel,
}

# ML
modelX = 'ts-model-x-{}'
modelY = 'ts-model-y-{}'
modelSelect = 'ts-model-select-{}'
modelScaler = 'ts-model-scaler-{}'
yContainer = 'ts-model-y-container-{}'
x = 'ts-model-x-{}'
y = 'ts-model-y-{}'
algorithms = 'ts-model-algorithms-{}'
scaler = 'ts-model-scaler-{}'
settings = 'ts-model-settings-{}'
# PLOTTER
axArea = 'ax-area-{}'
axSelect = 'ax-select-{}-{}'
dimSelect = 'dim-select-{}'
plot = 'ts-plot-{}'
plotContainer = 'plot-container-{}'
plotType = 'plot-type-{}'
axes = 'plot{}-{}'
dimensions = 'plotDim-{}'
plotArea = 'plot-area-{}'
accordion = 'plot-accordion-{}'
# GENERAL
appInput = 'app-input-{}'
appOutput = 'app-output-{}'
appReplace = 'app-replace-{}'
appDelete = 'app-delete-{}'
appDistrict = 'app-district-{}'
appAccordion = 'app-accordion-{}'
appName = 'app-{}'
appHeading = 'app-heading-{}'
appArea = 'app-area'
appZone = 'app-zone-{}'
options = 'options-{}'
optionsButton = 'options-button-{}'
settings = 'settings-{}'
settingsButton = 'settings-button-{}'
optionsTitle = 'options-title-{}'
settingsTitle = 'settings-title-{}'
closeOptions = 'close-options-{}'
closeSettings = 'close-settings-{}'
optionsBody = 'options-body-{}'
settingsBody = 'settings-body-{}'
removeButton = 'remove-button-{}'
generalIDs = [
    appInput,
    appOutput,
    appReplace,
    appDistrict,
    appZone,
    appAccordion,
    appName,
    appHeading,
]
sidebarIDs = [
    options,
    optionsButton,
    optionsTitle,
    settings,
    settingsButton,
    settingsTitle,
    closeOptions,
    closeSettings,
    optionsBody,
    settingsBody,
]
# ML
# ML
modelX = 'ts-model-x-{}'
modelY = 'ts-model-y-{}'
modelScaler = 'ts-model-scaler-{}'
yContainer = 'ts-model-y-container-{}'
settings = 'ts-model-settings-{}'
modelAlgorithms = 'ts-model-algorithms-{}'
modelScaler = 'ts-model-scaler-{}'
yContainer = 'ts-model-y-container-{}'
mlIDs = [
    modelX,
    modelY,
    modelAlgorithms,
    settings,
    modelScaler,
    yContainer,
]
modelSettings = 'ts-model-settings-{}'
# PLOTTER
axArea = 'ax-area-{}'
axSelect = 'ax-select-{}-{}'
dimArea = 'dim-area-{}'
dimSelect = 'dime-select-{}'
plot = 'ts-plot-{}'
plotType = 'plot-type-{}'
plotTypeArea = 'plot-type-area-{}'
allIDs = {
    'regression': mlIDs,
}
homeRoot = 'home'
homeTemplate = '{}/home2.html'.format(homeRoot)
infoRoot = 'info'
faq = '{}/faq.html'.format(infoRoot)
legalRoot = 'legal'
credits = '{}/credits1.html'.format(legalRoot)
privacy = '{}/privacy3.html'.format(legalRoot)
terms = '{}/terms3.html'.format(legalRoot)
analysisRoot = 'analysis'
analysisForm = '{}/form.html'.format(analysisRoot)
analysisApp = '{}/app.html'.format(analysisRoot)
analysisPlot = '{}/plot.html'.format(analysisRoot)
dataprepRoot = 'dataprep'
dataprepForm = '{}/form.html'.format(dataprepRoot)
dataprepApp = '{}/app.html'.format(dataprepRoot)
# HOME
homeRoot = 'home'
homeTemplate = '{}/home2.html'.format(homeRoot)
# INFO
infoRoot = 'info'
faq = '{}/faq.html'.format(infoRoot)
services = '{}/services0.html'.format(infoRoot)
# LEGAL
legalRoot = 'legal'
credits = '{}/credits.html'.format(legalRoot)
privacy = '{}/privacy0.html'.format(legalRoot)
terms = '{}/terms0.html'.format(legalRoot)
# ANALYSIS
analysisRoot = 'analysis'
analysisForm = '{}/form0.html'.format(analysisRoot)
analysisApp = '{}/app.html'.format(analysisRoot)
# DATA PREP
dataprepRoot = 'dataprep'
dataprepForm = '{}/form0.html'.format(dataprepRoot)
dataprepApp = '{}/app.html'.format(dataprepRoot)
# APPS
appsRoot = 'apps'
appContainer = '{}/app_container.html'.format(appsRoot)
plotApp = '{}/plot.html'.format(appsRoot)
regressionApp = '{}/regression.html'.format(appsRoot)
# MENUS
menusRoot = 'menus0'
sidebar = '{}/sidebar.html'.format(menusRoot)
fileSidebar = '{}/file.html'.format(menusRoot)
appOptions = '{}/app-options.html'.format(menusRoot)
regressionSettings = '{}/regression_settings.html'.format(menusRoot)
# BUTTONS
buttonsRoot = 'buttons'
editFileButton = '{}/edit_file.html'.format(buttonsRoot)
removeAppButton = '{}/remove_app.html'.format(buttonsRoot)
removeFileButton = '{}/remove_file.html'.format(buttonsRoot)
regressionButton = '{}/regression.html'.format(buttonsRoot)
server = os.path.dirname(os.path.dirname(sys.argv[0]))
website = 'data'
drivePath = server  # 'mnt'
uploadsPath = os.path.join(drivePath, website, 'uploads')
fitPath = os.path.join(uploadsPath, '{}', 'fit.csv')
templatesPath = os.path.join(drivePath, website, 'templates')
"""
TEMPLATE ROOTS
"""
# APP
serverRoot = 'server'
dashboardRoot = 'dashboard'
generalRoot = 'general'
# GENERAL WEBSITE
formsRoot = 'forms'
infoRoot = 'info'
"""
GENERAL WEBSITE TEMPLATE PATHS
"""
# PAGES
homeTemplate = '{}/home2.html'.format(generalRoot)
# INFO
about = '{}/{}/about2.html'.format(generalRoot, infoRoot)
tutorials = '{}/{}/tutorials.html'.format(generalRoot, infoRoot)
privacy = '{}/{}/privacy0.html'.format(generalRoot, infoRoot)
terms = '{}/{}/terms0.html'.format(generalRoot, infoRoot)
# FORMS
createPrivate = '{}/{}/create-private.html'.format(generalRoot, formsRoot)
createPublic = '{}/{}/create-public-dashboard.html'.format(generalRoot, formsRoot)
"""
APP TEMPLATE PATHS
"""
# DASHBOARD
dashboardBody = '{}/body.html'.format(dashboardRoot)
dashboardNavbar = '{}/navbar.html'.format(dashboardRoot)
website = 'data-django'
drivePath = server  # 'mnt'
uploadsPath = os.path.join(drivePath, website, 'uploads')
templatesPath = os.path.join(drivePath, website, 'templates')
fitPath = os.path.join(uploadsPath, '{}', 'fit.csv')
now = datetime.datetime.now()
# STATS CONFIG
funcFams = {'Exponential': lambda x, a, b: a * np.exp(b * x), 'Logarithmic': lambda x, a, b: a * np.log(x) + b}
sigfigs = 2
N = 1
centralTendenciesTableStats = ['Mean', 'Median']
dispersionTableStats = ['Standard Deviation', 'Variance', 'Skew', 'Kurtosis']
iqrTableStats = ['Lower IQR', 'First Quartile', 'Median', 'Third Quartile', 'Upper IQR']
# PATHS
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
# FUNCTION STRINGS ------------------------------------------------------
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
    'johnsonsb':     'Johnson SB', 'johnsonsu': 'Johnson SU', 'kappa4': 'Kappa 4', 'ksone': 'ksone',
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
# STATISTICS ------------------------------------------------------------
Quantiles = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1]
website = 'data'
drivePath = server  # 'mnt'
uploadsPath = os.path.join(drivePath, website, 'uploads')
templatesPath = os.path.join(drivePath, website, 'templates')
fitPath = os.path.join(uploadsPath, '{}', 'fit.csv')
"""
TEMPLATE ROOTS
"""
# APP
serverRoot = 'server'
dashboardRoot = 'dashboard'
appsRoot = 'app'
generalRoot = 'data'
# GENERAL WEBSITE
buttonsRoot = 'buttons'
formsRoot = 'forms'
infoRoot = 'info'
legalRoot = 'legal'
menusRoot = 'menus'
mlRoot = 'app/ts_ml'
plotRoot = 'app/plot'
tutorialsRoot = 'tutorials'
"""
GENERAL WEBSITE TEMPLATE PATHS
"""
# PAGES
homeTemplate = '{}/home2.html'.format(generalRoot)
about = '{}/{}/about2.html'.format(generalRoot, infoRoot)
faq = '{}/{}/faq.html'.format(generalRoot, infoRoot)
future = '{}/{}/future.html'.format(generalRoot, infoRoot)
credits = '{}/{}/credits1.html'.format(generalRoot, legalRoot)
privacy = '{}/{}/privacy3.html'.format(generalRoot, legalRoot)
terms = '{}/{}/terms3.html'.format(generalRoot, legalRoot)
tutorials = 'data/tutorials/tutorials.html'.format(generalRoot, tutorialsRoot)
# FORMS
createPrivate = '{}/{}/create-private.html'.format(generalRoot, formsRoot)
createPublic = '{}/{}/create-public.html'.format(generalRoot, formsRoot)
login = '{}/{}/login.html'.format(generalRoot, formsRoot)
resetPasswordRequest = '{}/{}/reset-password-request.html'.format(generalRoot, formsRoot)
signup = '{}/{}/signup.html'.format(generalRoot, formsRoot)
"""
APP TEMPLATE PATHS
"""
# SERVER
editFileButton = '{}/edit_file.html'.format(buttonsRoot)
removeFileButton = '{}/remove_file.html'.format(buttonsRoot)
fileSidebar = '{}/file.html'.format(menusRoot)
# DASHBOARD
dashboardBody = '{}/dashboard-body1.html'.format(dashboardRoot)
appBody = '{}/app-body.html'.format(appsRoot)
appContainer = '{}/app-container.html'.format(appsRoot)
appDeleteButton = '{}/remove_app.html'.format(buttonsRoot)
dashboardNavbar = '{}/navbar/navbar.html'.format(dashboardRoot)
insertAppBtn = '{}/navbar/insert-app-btn1.html'.format(dashboardRoot)
sidebar = '{}/sidebar/sidebar.html'.format(menusRoot)
sidebarNavbar = '{}/sidebar/sidebar-navbar.html'.format(dashboardRoot)
insertAppModal = '{}/insert-app-modal/menu.html'.format(dashboardRoot)
insertAppFooter = '{}/insert-app-modal/footer.html'.format(dashboardRoot)
insertAppBody = '{}/insert-app-modal/body.html'.format(dashboardRoot)
# ML TEMPLATES
regressionPlot = '{}/regression-plot-body.html'.format(mlRoot)
mlButtons = '{}/app-buttons.html'.format(mlRoot)
mlSidebarBody = '{}/sidebar-body.html'.format(mlRoot)
regressionAppMenu = '{}/regression-sidebar-app-menu.html'.format(mlRoot)
regressionHPMenu = '{}/regression-sidebar-hp-menu.html'.format(mlRoot)
regressionSettingsMenu = '{}/regression-sidebar-settings-menu.html'.format(mlRoot)
regressionButton = '{}/regression.html'.format(buttonsRoot)
# PLOT TEMPLATES
plotBody = '{}/plot-body.html'.format(plotRoot)
plotSettings = '{}/plot-settings.html'.format(plotRoot)
plotButtons = '{}/plot-button-toolbar.html'.format(plotRoot)
scatterPlot = '{}/scatter.html'.format(plotRoot)
histogramPlot = '{}/plot-body.html'.format(plotRoot)
website = 'data'
drivePath = server  # 'mnt'
uploadsPath = os.path.join(drivePath, website, 'uploads')
templatesPath = os.path.join(drivePath, website, 'templates')
fitPath = os.path.join(uploadsPath, '{}', 'fit.csv')
'''
TEMPLATE ROOTS
'''
# APP
serverRoot = 'server'
dashboardRoot = 'dashboard'
appsRoot = 'app'
generalRoot = 'general'
# GENERAL WEBSITE
buttonsRoot = 'buttons'
formsRoot = 'forms'
infoRoot = 'info'
legalRoot = 'legal'
menusRoot = 'menus'
mlRoot = 'app/ml'
plotRoot = 'app/plot'
tutorialsRoot = 'tutorials'
'''
GENERAL WEBSITE TEMPLATE PATHS
'''
# PAGES
homeTemplate = '{}/home2.html'.format(generalRoot)
about = '{}/{}/about2.html'.format(generalRoot, infoRoot)
faq = '{}/{}/faq.html'.format(generalRoot, infoRoot)
future = '{}/{}/future.html'.format(generalRoot, infoRoot)
credits = '{}/{}/credits1.html'.format(generalRoot, legalRoot)
privacy = '{}/{}/privacy3.html'.format(generalRoot, legalRoot)
terms = '{}/{}/terms3.html'.format(generalRoot, legalRoot)
tutorials = 'general/tutorials/tutorials.html'.format(generalRoot, tutorialsRoot)
# FORMS
createPrivate = '{}/{}/create-private.html'.format(generalRoot, formsRoot)
createPublic = '{}/{}/create-public.html'.format(generalRoot, formsRoot)
login = '{}/{}/login.html'.format(generalRoot, formsRoot)
resetPasswordRequest = '{}/{}/reset-password-request.html'.format(generalRoot, formsRoot)
signup = '{}/{}/signup.html'.format(generalRoot, formsRoot)
'''
APP TEMPLATE PATHS
'''
# SERVER
editFileButton = '{}/edit_file.html'.format(buttonsRoot)
removeFileButton = '{}/remove_file.html'.format(buttonsRoot)
fileSidebar = '{}/file.html'.format(menusRoot)
# DASHBOARD
dashboardBody = '{}/dashboard-body1.html'.format(dashboardRoot)
appBody = '{}/app-body.html'.format(appsRoot)
appContainer = '{}/app-container.html'.format(appsRoot)
appDeleteButton = '{}/remove_app.html'.format(buttonsRoot)
dashboardNavbar = '{}/navbar/navbar.html'.format(dashboardRoot)
insertAppBtn = '{}/navbar/insert-app-btn1.html'.format(dashboardRoot)
sidebar = '{}/sidebar/sidebar.html'.format(menusRoot)
sidebarNavbar = '{}/sidebar/sidebar-navbar.html'.format(dashboardRoot)
insertAppModal = '{}/insert-app-modal/menu.html'.format(dashboardRoot)
insertAppFooter = '{}/insert-app-modal/footer.html'.format(dashboardRoot)
insertAppBody = '{}/insert-app-modal/body.html'.format(dashboardRoot)
# ML TEMPLATES
regressionPlot = '{}/regression-plot-body.html'.format(mlRoot)
mlButtons = '{}/app-buttons.html'.format(mlRoot)
mlSidebarBody = '{}/sidebar-body.html'.format(mlRoot)
regressionAppMenu = '{}/regression-sidebar-app-menu.html'.format(mlRoot)
regressionHPMenu = '{}/regression-sidebar-hp-menu.html'.format(mlRoot)
regressionSettingsMenu = '{}/regression-sidebar-settings-menu.html'.format(mlRoot)
regressionButton = '{}/regression.html'.format(buttonsRoot)
# PLOT TEMPLATES
plotBody = '{}/plot-body.html'.format(plotRoot)
plotSettings = '{}/plot-settings.html'.format(plotRoot)
plotButtons = '{}/plot-button-toolbar.html'.format(plotRoot)
scatterPlot = '{}/scatter.html'.format(plotRoot)
histogramPlot = '{}/plot-body.html'.format(plotRoot)
website = 'data'
drivePath = server  # 'mnt'
uploadsPath = os.path.join(drivePath, website, 'uploads')
templatesPath = os.path.join(drivePath, website, 'templates')
fitPath = os.path.join(uploadsPath, '{}', 'fit.csv')
"""
TEMPLATE ROOTS
"""
# APP
serverRoot = 'server'
dashboardRoot = 'dashboard'
appsRoot = 'app'
generalRoot = 'data'
# GENERAL WEBSITE
buttonsRoot = 'buttons'
formsRoot = 'forms'
infoRoot = 'info'
legalRoot = 'legal'
menusRoot = 'menus'
mlRoot = 'app/ts_ml'
plotRoot = 'app/plot'
tutorialsRoot = 'tutorials'
"""
GENERAL WEBSITE TEMPLATE PATHS
"""
# PAGES
homeTemplate = '{}/home2.html'.format(generalRoot)
about = '{}/{}/about2.html'.format(generalRoot, infoRoot)
faq = '{}/{}/faq.html'.format(generalRoot, infoRoot)
future = '{}/{}/future.html'.format(generalRoot, infoRoot)
credits = '{}/{}/credits1.html'.format(generalRoot, legalRoot)
privacy = '{}/{}/privacy3.html'.format(generalRoot, legalRoot)
terms = '{}/{}/terms3.html'.format(generalRoot, legalRoot)
tutorials = 'data/tutorials/tutorials.html'.format(generalRoot, tutorialsRoot)
# FORMS
createPrivate = '{}/{}/create-private.html'.format(generalRoot, formsRoot)
createPublic = '{}/{}/create-public.html'.format(generalRoot, formsRoot)
login = '{}/{}/login.html'.format(generalRoot, formsRoot)
resetPasswordRequest = '{}/{}/reset-password-request.html'.format(generalRoot, formsRoot)
signup = '{}/{}/signup.html'.format(generalRoot, formsRoot)
"""
APP TEMPLATE PATHS
"""
# SERVER
editFileButton = '{}/edit_file.html'.format(buttonsRoot)
removeFileButton = '{}/remove_file.html'.format(buttonsRoot)
fileSidebar = '{}/file.html'.format(menusRoot)
# DASHBOARD
dashboardBody = '{}/dashboard-body1.html'.format(dashboardRoot)
appBody = '{}/app-body.html'.format(appsRoot)
appContainer = '{}/app-container.html'.format(appsRoot)
appDeleteButton = '{}/remove_app.html'.format(buttonsRoot)
dashboardNavbar = '{}/navbar.html'.format(dashboardRoot)
# ML TEMPLATES
regressionPlot = '{}/regression-plot-body.html'.format(mlRoot)
mlButtons = '{}/app-buttons.html'.format(mlRoot)
mlSidebarBody = '{}/sidebar-body.html'.format(mlRoot)
regressionAppMenu = '{}/regression-sidebar-app-menu.html'.format(mlRoot)
regressionHPMenu = '{}/regression-sidebar-hp-menu.html'.format(mlRoot)
regressionSettingsMenu = '{}/regression-sidebar-settings-menu.html'.format(mlRoot)
regressionButton = '{}/regression.html'.format(buttonsRoot)
# PLOT TEMPLATES
plotBody = '{}/plot-body.html'.format(plotRoot)
plotSettings = '{}/plot-settings.html'.format(plotRoot)
plotButtons = '{}/plot-button-toolbar.html'.format(plotRoot)
scatterPlot = '{}/scatter.html'.format(plotRoot)
histogramPlot = '{}/plot-body.html'.format(plotRoot)
website = 'data'
drivePath = server  # 'mnt'
uploadsPath = os.path.join(drivePath, website, 'uploads')
templatesPath = os.path.join(drivePath, website, 'templates')
fitPath = os.path.join(uploadsPath, '{}', 'fit.csv')
"""
TEMPLATE ROOTS
"""
# APP
serverRoot = 'server'
dashboardRoot = 'dashboard'
appsRoot = 'app'
generalRoot = 'data'
# GENERAL WEBSITE
buttonsRoot = 'buttons'
formsRoot = 'forms'
infoRoot = 'info'
legalRoot = 'legal'
menusRoot = 'menus'
mlRoot = 'app/ts_ml'
plotRoot = 'app/plot'
tutorialsRoot = 'tutorials'
"""
GENERAL WEBSITE TEMPLATE PATHS
"""
# PAGES
homeTemplate = '{}/home2.html'.format(generalRoot)
about = '{}/{}/about2.html'.format(generalRoot, infoRoot)
faq = '{}/{}/faq.html'.format(generalRoot, infoRoot)
future = '{}/{}/future.html'.format(generalRoot, infoRoot)
credits = '{}/{}/credits1.html'.format(generalRoot, legalRoot)
privacy = '{}/{}/privacy3.html'.format(generalRoot, legalRoot)
terms = '{}/{}/terms3.html'.format(generalRoot, legalRoot)
tutorials = 'data/tutorials/tutorials.html'.format(generalRoot, tutorialsRoot)
# FORMS
createPrivate = '{}/{}/create-private.html'.format(generalRoot, formsRoot)
createPublic = '{}/{}/create-public.html'.format(generalRoot, formsRoot)
login = '{}/{}/login.html'.format(generalRoot, formsRoot)
resetPasswordRequest = '{}/{}/reset-password-request.html'.format(generalRoot, formsRoot)
signup = '{}/{}/signup.html'.format(generalRoot, formsRoot)

"""
APP TEMPLATE PATHS
"""
# SERVER
editFileButton = '{}/edit_file.html'.format(buttonsRoot)
removeFileButton = '{}/remove_file.html'.format(buttonsRoot)
fileSidebar = '{}/file.html'.format(menusRoot)
# DASHBOARD
dashboardBody = '{}/body.html'.format(dashboardRoot)
appBody = '{}/app-body.html'.format(appsRoot)
appContainer = '{}/app-container.html'.format(appsRoot)
appDeleteButton = '{}/remove_app.html'.format(buttonsRoot)
dashboardNavbar = '{}/navbar.html'.format(dashboardRoot)
# ML TEMPLATES
regressionPlot = '{}/regression-plot-body.html'.format(mlRoot)
mlButtons = '{}/app-buttons.html'.format(mlRoot)
mlSidebarBody = '{}/sidebar-body.html'.format(mlRoot)
regressionAppMenu = '{}/regression-sidebar-app-menu.html'.format(mlRoot)
regressionHPMenu = '{}/regression-sidebar-hp-menu.html'.format(mlRoot)
regressionSettingsMenu = '{}/regression-sidebar-settings-menu.html'.format(mlRoot)
regressionButton = '{}/regression.html'.format(buttonsRoot)
# PLOT TEMPLATES
plotBody = '{}/plot-body.html'.format(plotRoot)
plotSettings = '{}/plot-settings.html'.format(plotRoot)
plotButtons = '{}/plot-button-toolbar.html'.format(plotRoot)
scatterPlot = '{}/scatter.html'.format(plotRoot)
histogramPlot = '{}/plot-body.html'.format(plotRoot)
website = 'data'
drivePath = server  # 'mnt'
uploadsPath = os.path.join(drivePath, website, 'uploads')
templatesPath = os.path.join(drivePath, website, 'templates')
"""
GENERAL WEBSITE TEMPLATE PATHS
"""
# PAGES
homeTemplate = '{}/home2.html'.format(generalRoot)
# INFO
about = '{}/{}/about2.html'.format(generalRoot, infoRoot)
tutorials = '{}/{}/tutorials.html'.format(generalRoot, infoRoot)
privacy = '{}/{}/privacy3.html'.format(generalRoot, infoRoot)
terms = '{}/{}/terms3.html'.format(generalRoot, infoRoot)
# FORMS
createPrivate = '{}/{}/create-private.html'.format(generalRoot, formsRoot)
createPublic = '{}/{}/create-public-dashboard.html'.format(generalRoot, formsRoot)
"""
APP TEMPLATE PATHS
"""
# DASHBOARD
dashboardBody = '{}/body.html'.format(dashboardRoot)
dashboardNavbar = '{}/navbar.html'.format(dashboardRoot)
# objectTypes = (
#     ('plot', 'Plot'),
#     ('classification', 'Classification'),
#     ('clustering', 'Clustering'),
#     ('dimreduction', 'Dimensionality Reduction'),
#     ('regression', 'regression'),
#     ('dnn', 'Deep Neural Network'),
#     ('stats', 'Statistics'),
#     ('distfit', 'Distribution Fitting'),
#     ('dataprep', 'Data Preparation'),
# )
objectTypes = [
    ('file', 'File'),
    ('generator', 'Generator'),
    ('mathfunction', 'Math Function'),
    ('regression', 'Regression'),
    ('plot', 'Plot'),
]
colors = ["#00AFBB",
          "#ff9f00",
          "#CC79A7",
          "#009E73",
          "#66ccff",  # Not as distinguishable with #00AFBB for colorblind
          "#F0E442"]

# PLOTTER
axArea = 'ax-area-{}'
axSelect = 'ax-select-{}-{}'
dimSelect = 'dim-select-{}'
plot = 'ts-plot-{}'
plotContainer = 'plot-container-{}'
plotType = 'plot-type-{}'
axes = 'plot{}-{}'
dimensions = 'plotDim-{}'
plotArea = 'plot-area-{}'
accordion = 'plot-accordion-{}'
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
