import os
import sys

# AI LIBRARY - sklearn or keras
aiLib = 'keras'

# DIRECTORY SETTINGS
testingDir = 'testing'
trainingDir = 'data/1992'
plotsDir = r'E:\github2\_output\national_bridge_inventory_plots'
modelsDir = {'keras': 'keras_models', 'sklearn': 'sklearn_models', }
modelExt = {'keras': 'h5', 'sklearn': 'joblib', }
drive = '{}{}'.format(os.path.dirname(os.path.dirname(sys.argv[0])), os.sep)

lat = 'LAT_016'
long = 'LONG_017'
deck = 'DECK_COND_058'
superstructure = 'SUPERSTRUCTURE_COND_059'
substructure = 'SUBSTRUCTURE_COND_060'
channel = 'CHANNEL_COND_061'
culvert = 'CULVERT_COND_062'

# AI SETTINGS
epochs = 100
batchSize = 32  # * 21  # 672

# print('== GLOBAL SETTINGS ==\n'
#       ' o aiLib: {}\n'
#       ' o epochs: {}\n'
#       ' o batchSize: {}\n'
#       '====================='.format(aiLib, epochs, batchSize, ))

# PLOT CONFIG
snsStyle = {
    'lines.linewidth':       1,
    'figure.figsize':        (1, 1),
    'figure.facecolor':      'white',
    'font.family':           ['sans-serif', ],
    'font.sans-serif':       ['Arial',
                              'DejaVu Sans',
                              'Liberation Sans',
                              'Bitstream Vera Sans',
                              'sans-serif'],

    'axes.axisbelow':        True,
    'axes.edgecolor':        'black',
    'axes.facecolor':        'white',
    'axes.grid':             True,
    'axes.labelcolor':       'black',
    'axes.spines.bottom':    True,
    'axes.spines.left':      True,
    'axes.spines.right':     True,
    'axes.spines.top':       True,

    'grid.color':            '.8',
    'grid.linestyle':        '-',
    'image.cmap':            'rocket',
    'lines.solid_capstyle':  'round',
    'patch.edgecolor':       'black',
    'patch.force_edgecolor': True,
    'text.color':            'black',

    'xtick.bottom':          True,
    'xtick.color':           'black',
    'xtick.direction':       'out',
    'xtick.top':             False,
    'ytick.color':           'black',
    'ytick.direction':       'out',
    'ytick.left':            True,
    'ytick.right':           False,
}
colors = ["#00AFBB", "#ff9f00", "#CC79A7", "#009E73", "#F0E442#66ccff"]
el = {
    'STRUCTURE_KIND_043A': 10,
    'STRUCTURE_TYPE_043B': 23,
    'DECK_COND_058':       10,
}

categoricalCols = [
    # 'STRUCTURE_KIND_043A',  # 10 element vector
    # 'STRUCTURE_TYPE_043B',  # 23 element vector
    # 'STRUCTURE_FLARED_035',
    # 'DECK_STRUCTURE_TYPE_107',
    # SURFACE_TYPE_108A',
    # 'MEMBRANE_TYPE_108B',
    # 'SERVICE_LEVEL_005C',
    # 'DECK_PROTECTION_108C',
    # 'PIER_PROTECTION_111',
    # 'DESIGN_LOAD_031',
]
numericalCols = [
    # 'ADT_029',
    # 'YEAR_ADT_030',
    # 'DECK_WIDTH_MT_052',
    # 'MAX_SPAN_LEN_MT_048',
    # 'PERCENT_ADT_TRUCK_109',
    # 'YEAR_BUILT_027',
    # 'YEAR_RECONSTRUCTED_106',
    'LAT_016',
    'LONG_017',
    # 'DEGREES_SKEW_034',
    # 'MIN_VERT_CLR_010',
]
con = {
    'DECK_COND_058':           'Deck',
    'SUPERSTRUCTURE_COND_059': 'Superstructure',
    'SUBSTRUCTURE_COND_060':   'Substructure',
    'CHANNEL_COND_061':        'Channel',
    'CULVERT_COND_062':        'Culvert',
    'ADT_029':                 'ADT',
    # 'YEAR_ADT_030': '',
    'PERCENT_ADT_TRUCK_109':   '% ADT Trucks',
    'YEAR_BUILT_027':          'Year Built',
    'YEAR_RECONSTRUCTED_106':  'Year Reconstructed',
    'LAT_016':                 'Latitude',
    'LONG_017':                'Longitude',
    'DECK_WIDTH_MT_052':       'Deck Width (m)',
    'MAX_SPAN_LEN_MT_048':     'Max Span Length (m)',
    'DEGREES_SKEW_034':        'Skew (degrees)',
    'MIN_VERT_CLR_010':        'Min Vertical Clearance (m)',
    # 'DESIGN_LOAD_031': 'Design Load',
}
conditionCols = ['DECK_COND_058', 'SUPERSTRUCTURE_COND_059', 'SUBSTRUCTURE_COND_060']

classWeight = {
    0: 50,
    1: 50,
    2: 50,
    3: 20,
    4: 10,
    5: 9,
    6: 5,
    7: 2,
    8: 1,
    9: 10
}
