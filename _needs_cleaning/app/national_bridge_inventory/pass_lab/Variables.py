# mcarmona@utep.edu
# cm = count matrix
# iS = initial state
# iSV = initial state vector
# f_s = frequency summary
# mc_s_p = markov chain settings path
# pv = probability vector
# RSP = report settings path
# sv = state vector
# tm = transition matrix
# vina = valid item names
# vinu = valid item numbers
# vs = valid states
# xi = x initial
# xf = x final
# fp = file path
# DATABASE
FOLDER = 'YEAR'
FILE = 'STATE_CODE_001'
ID = 'STRUCTURE_NUMBER_008'
LATITUDE = 'LAT_016'
LONGITUDE = 'LONG_017'
ENCODING = 'latin1'

# GRAPHS
GRAPH_TITLES = ("Transition Probabilities", "Raw Deterioration", "Simulation Deterioration", "Raw Frequency (per Year)",
                "Simulation Frequency (per Year)", "Raw Frequency (per State)", "Simulation Frequency (per State)",
                "3D Frequency")
AXIS_LABELS = (("Time (Years)", "Probability"), ("Time (Years)", "State"),
               ("Time (Years)", "Frequency"), ("State", "Frequency"))
