import configuration
import lozoya.gui

app = lozoya.gui_api.TSApp(
    name=configuration.name,
    root=configuration.root,
)
app.exec()

r'''
MASTERPATH = "C:\\Users\\frano\PycharmProjects"
import os

directory_in_str = "C:\\Users\\frano\PycharmProjects\BridgeDataQuery\Database\Information\Years\\2009"
directory = os.fsencode(directory_in_str)

states = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
    "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
    "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY", "72": "PR"
}
codes = ["01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16", "17", "18", "19", "20", "21",
         "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
         "40", "41", "42", "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56", "72"]

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    for i in codes:
        if filename.startswith(str(i)):
            print(i)  # print(os.path.join(directory, filename))
            new_name = states[i] + "09"
            old_file = directory_in_str + os.path.sep + filename
            new_file = directory_in_str + os.path.sep + new_name + ".txt"

            os.rename(old_file, new_file)
            break

import os

MASTERPATH = "C:\\Users\\frano\PycharmProjects"
directory_in_str = "C:\\Users\\frano\PycharmProjects\BridgeDataQuery\Database\Information\Years\\2009"
directory = os.fsencode(directory_in_str)
states = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
    "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
    "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY", "72": "PR"
}
codes = ["01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16", "17", "18", "19", "20", "21",
         "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
         "40", "41", "42", "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56", "72"]

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    for i in codes:
        if filename.startswith(str(i)):
            print(i)  # print(os.path.join(directory, filename))
            new_name = states[i] + "09"
            old_file = directory_in_str + os.path.sep + filename
            new_file = directory_in_str + os.path.sep + new_name + ".txt"
            os.rename(old_file, new_file)
            break

# DATABASE
FOLDER = 'YEAR'
FILE = 'STATE_CODE_001'
ID = 'STRUCTURE_NUMBER_008'
LATITUDE = 'LAT_016'
LONGITUDE = 'LONG_017'
ENCODING = 'latin1'
SEPARATOR = '	'
HEADERS = ['SOS_VoterID', 'idnumber', 'voter_status', 'party_code', 'lastname', 'firstname', 'middlename', 'namesuffix',
           'streetnumber', 'streetbuilding', 'streetpredir', 'streetname', 'streettype', 'streetpostdir', 'unit_type',
           'unit', 'city', 'zip', 'zip4', 'mail1', 'mail2', 'mail3', 'mailcity', 'mailstate', 'mailzip', 'mailzip4',
           'sex', 'birthdate', 'eligible_date', 'effective_date', 'precinct', 'precsub', 'COUNTY', 'U.S. REP.',
           'ST. SENATE', 'ST. REP.', 'COMMISSIONER', 'CITY', 'CITY SINGLE MEMBER', 'SCHOOL', 'SCHOOL SINGLE MEMBER',
           'PRECINCT', 'MUNICIPAL UTILITY DISTRICT', 'WATER', 'district_13', 'FRESH WATER SUPPLY',
           'WATER SINGLE MEMBER', 'COMMUNITY COLLEGE', 'COMMUNITY COLLEGE SINGL MEMBER', 'EMERGENCY SERVICES',
           'CITY PROPOSED', 'HOSPITAL', 'JUSTICE OF THE PEACE', 'district_22', 'STATE BOARD OF EDU', 'district_24',
           'COUNTY COMMISSIONER PRECINCT', 'district_26', 'district_27', 'district_28', 'district_29', 'PRECINCT CHAIR',
           'district_31', 'district_32', 'district_33', 'district_34', 'district_35', 'district_36', 'district_37',
           'district_38', 'district_39', 'district_40', 'district_41', 'district_42', 'district_43', 'district_44',
           'district_45', 'district_46', 'district_47', 'district_48', 'district_49', 'district_50', 'election_code1',
           'vote_type1', 'party_code1', 'election_code2', 'vote_type2', 'party_code2', 'election_code3', 'vote_type3',
           'party_code3', 'election_code4', 'vote_type4', 'party_code4', 'election_code5', 'vote_type5', 'party_code5',
           'election_code6', 'vote_type6', 'party_code6', 'election_code7', 'vote_type7', 'party_code7',
           'election_code8', 'vote_type8', 'party_code8', 'election_code9', 'vote_type9', 'party_code9',
           'election_code10', 'vote_type10', 'party_code10', 'election_code11', 'vote_type11', 'party_code11',
           'election_code12', 'vote_type12', 'party_code12', 'election_code13', 'vote_type13', 'party_code13',
           'election_code14', 'vote_type14', 'party_code14', 'election_code15', 'vote_type15', 'party_code15',
           'election_code16', 'vote_type16', 'party_code16', 'election_code17', 'vote_type17', 'party_code17',
           'election_code18', 'vote_type18', 'party_code18', 'election_code19', 'vote_type19', 'party_code19',
           'election_code20', 'vote_type20', 'party_code20', 'election_code21', 'vote_type21', 'party_code21',
           'election_code22', 'vote_type22', 'party_code22', 'election_code23', 'vote_type23', 'party_code23',
           'election_code24', 'vote_type24', 'party_code24', 'election_code25', 'vote_type25', 'party_code25',
           'election_code26', 'vote_type26', 'party_code26', 'election_code27', 'vote_type27', 'party_code27',
           'election_code28', 'vote_type28', 'party_code28', 'election_code29', 'vote_type29', 'party_code29',
           'election_code30', 'vote_type30', 'party_code30', 'election_code31', 'vote_type31', 'party_code31',
           'election_code32', 'vote_type32', 'party_code32', 'election_code33', 'vote_type33', 'party_code33',
           'election_code34', 'vote_type34', 'party_code34', 'election_code35', 'vote_type35', 'party_code35',
           'election_code36', 'vote_type36', 'party_code36', 'election_code37', 'vote_type37', 'party_code37',
           'election_code38', 'vote_type38', 'party_code38', 'election_code39', 'vote_type39', 'party_code39',
           'election_code40', 'vote_type40', 'party_code40', 'registration_date', 'party_affiliation_date',
           'last_activity_date', 'precinct_group', 'phone_number', 'ID_Compliant', 'Absentee_Category',
           'Absentee_Category_Date', 'Ethnicity']

# GRAPHS
GRAPH_TITLES = (
    "Transition Probabilities", "Raw Deterioration", "Simulation2 Deterioration", "Raw Frequency (per Year)",
    "Simulation2 Frequency (per Year)", "Raw Frequency (per State)", "Simulation2 Frequency (per State)",
    "3D Frequency")
AXIS_LABELS = (
    ("Time (Years)", "Probability"), ("Time (Years)", "State"), ("Time (Years)", "Frequency"), ("State", "Frequency"))

import os

import scipy.stats as st

# DATABASE
ID = None  # 'STRUCTURE_NUMBER_008'
LATITUDE = 'point_latitude'
LONGITUDE = 'point_longitude'
ENCODING = 'latin1'

# GRAPHS
GRAPH_TITLES = (
    "Transition Probabilities", "Raw Deterioration", "Simulation2 Deterioration", "Raw Frequency (per Year)",
    "Simulation2 Frequency (per Year)", "Raw Frequency (per State)", "Simulation2 Frequency (per State)",
    "3D Frequency")
AXIS_LABELS = (
    ("Time (Years)", "Probability"), ("Time (Years)", "State"), ("Time (Years)", "Frequency"), ("State", "Frequency"))

from Utilities33.Dependencies import *

# PATHS
ROOT = os.getcwd()

DATABASE = ROOT + '\Database\Years\\'
VALUES = ROOT + r'\Interface\InterfaceUtilities\Labels\parameterValues.txt'
STATES = ROOT + r'\Database\Information\allStates.txt'
STATE_CODES = ROOT + r'\Database\Information\allStateCodes.txt'
ITEM_NAMES = ROOT + r'\Interface\InterfaceUtilities\Labels\ItemNames.txt'
temp_RSP = ROOT + '\\Utilities14\Preferences\Temporary_Report_Preferences.txt'
RSP = ROOT + '\\Utilities14\Preferences\Report_Preferences.txt'

# DATABASE
FOLDER = 'YEAR'
FILE = 'STATE_CODE_001'
ID = 'STRUCTURE_NUMBER_008'
LATITUDE = 'LAT_016'
LONGITUDE = 'LONG_017'
ENCODING = 'latin1'

# GRAPHS
GRAPH_TITLES = (
    "Transition Probabilities", "Raw Deterioration", "Simulation2 Deterioration", "Raw Frequency (per Year)",
    "Simulation2 Frequency (per Year)", "Raw Frequency (per State)", "Simulation2 Frequency (per State)",
    "3D Frequency")
AXIS_LABELS = (
    ("Time (Years)", "Probability"), ("Time (Years)", "State"), ("Time (Years)", "Frequency"), ("State", "Frequency"))

HEADERS = collect_headers(DATABASE)
# ENUMERATE OF FEATURES IN DATABASE
FEATURES = {}
with open(ITEM_NAMES, "r") as f:
    for i, line in enumerate(f):
        FEATURES[i] = line.strip()

# List of items that can transition through states
tmp = []
with open(VALUES, "r") as parameterValuesFile:
    with open(ITEM_NAMES, "r") as parameterNamesFile:
        for name, value in zip(csv.reader(parameterNamesFile), (csv.reader(parameterValuesFile))):
            if name[0] != FOLDER and name[0] != FILE and "None" not in value:
                tmp.append(name[0])
LEARNABLE = [FEATURES[i] for i in FEATURES if (FEATURES[i]) in tmp]

# DATABASE
ENCODING = None
ID = None
LATITUDE = None
LONGITUDE = None

SEPARATOR = '	'
HEADERS = ['SOS_VoterID', 'idnumber', 'voter_status', 'party_code', 'lastname', 'firstname', 'middlename', 'namesuffix',
           'streetnumber', 'streetbuilding', 'streetpredir', 'streetname', 'streettype', 'streetpostdir', 'unit_type',
           'unit', 'city', 'zip', 'zip4', 'mail1', 'mail2', 'mail3', 'mailcity', 'mailstate', 'mailzip', 'mailzip4',
           'sex', 'birthdate', 'eligible_date', 'effective_date', 'precinct', 'precsub', 'COUNTY', 'U.S. REP.',
           'ST. SENATE', 'ST. REP.', 'COMMISSIONER', 'CITY', 'CITY SINGLE MEMBER', 'SCHOOL', 'SCHOOL SINGLE MEMBER',
           'PRECINCT', 'MUNICIPAL UTILITY DISTRICT', 'WATER', 'district_13', 'FRESH WATER SUPPLY',
           'WATER SINGLE MEMBER', 'COMMUNITY COLLEGE', 'COMMUNITY COLLEGE SINGL MEMBER', 'EMERGENCY SERVICES',
           'CITY PROPOSED', 'HOSPITAL', 'JUSTICE OF THE PEACE', 'district_22', 'STATE BOARD OF EDU', 'district_24',
           'COUNTY COMMISSIONER PRECINCT', 'district_26', 'district_27', 'district_28', 'district_29', 'PRECINCT CHAIR',
           'district_31', 'district_32', 'district_33', 'district_34', 'district_35', 'district_36', 'district_37',
           'district_38', 'district_39', 'district_40', 'district_41', 'district_42', 'district_43', 'district_44',
           'district_45', 'district_46', 'district_47', 'district_48', 'district_49', 'district_50', 'election_code1',
           'vote_type1', 'party_code1', 'election_code2', 'vote_type2', 'party_code2', 'election_code3', 'vote_type3',
           'party_code3', 'election_code4', 'vote_type4', 'party_code4', 'election_code5', 'vote_type5', 'party_code5',
           'election_code6', 'vote_type6', 'party_code6', 'election_code7', 'vote_type7', 'party_code7',
           'election_code8', 'vote_type8', 'party_code8', 'election_code9', 'vote_type9', 'party_code9',
           'election_code10', 'vote_type10', 'party_code10', 'election_code11', 'vote_type11', 'party_code11',
           'election_code12', 'vote_type12', 'party_code12', 'election_code13', 'vote_type13', 'party_code13',
           'election_code14', 'vote_type14', 'party_code14', 'election_code15', 'vote_type15', 'party_code15',
           'election_code16', 'vote_type16', 'party_code16', 'election_code17', 'vote_type17', 'party_code17',
           'election_code18', 'vote_type18', 'party_code18', 'election_code19', 'vote_type19', 'party_code19',
           'election_code20', 'vote_type20', 'party_code20', 'election_code21', 'vote_type21', 'party_code21',
           'election_code22', 'vote_type22', 'party_code22', 'election_code23', 'vote_type23', 'party_code23',
           'election_code24', 'vote_type24', 'party_code24', 'election_code25', 'vote_type25', 'party_code25',
           'election_code26', 'vote_type26', 'party_code26', 'election_code27', 'vote_type27', 'party_code27',
           'election_code28', 'vote_type28', 'party_code28', 'election_code29', 'vote_type29', 'party_code29',
           'election_code30', 'vote_type30', 'party_code30', 'election_code31', 'vote_type31', 'party_code31',
           'election_code32', 'vote_type32', 'party_code32', 'election_code33', 'vote_type33', 'party_code33',
           'election_code34', 'vote_type34', 'party_code34', 'election_code35', 'vote_type35', 'party_code35',
           'election_code36', 'vote_type36', 'party_code36', 'election_code37', 'vote_type37', 'party_code37',
           'election_code38', 'vote_type38', 'party_code38', 'election_code39', 'vote_type39', 'party_code39',
           'election_code40', 'vote_type40', 'party_code40', 'registration_date', 'party_affiliation_date',
           'last_activity_date', 'precinct_group', 'phone_number', 'ID_Compliant', 'Absentee_Category',
           'Absentee_Category_Date', 'Ethnicity']

# DATABASE
ENCODING = 'utf-5'
ID = None
LATITUDE = 'latitude'
LONGITUDE = 'longitude'

SEPARATOR = ','
HEADERS = ['col1', 'col2']

# DATABASE
ENCODING = None
ID = None
LATITUDE = None
LONGITUDE = None

SEPARATOR = ','
HEADERS = ['col1', 'col2']

# DATABASE
FOLDER = 'YEAR'
FILE = 'STATE_CODE_001'
ID = 'STRUCTURE_NUMBER_008'
LATITUDE = 'LAT_016'
LONGITUDE = 'LONG_017'
ENCODING = 'latin1'

# GRAPHS
GRAPH_TITLES = (
    "Transition Probabilities", "Raw Deterioration", "Simulation2 Deterioration", "Raw Frequency (per Year)",
    "Simulation2 Frequency (per Year)", "Raw Frequency (per State)", "Simulation2 Frequency (per State)",
    "3D Frequency")
AXIS_LABELS = (
    ("Time (Years)", "Probability"), ("Time (Years)", "State"), ("Time (Years)", "Frequency"), ("State", "Frequency"))

from Utilities.Dependencies import *

# PATHS
ROOT = os.getcwd()

DATABASE = ROOT + '\Database\Years\\'
VALUES = ROOT + r'\Interface\InterfaceUtilities\Labels\parameterValues.txt'
STATES = ROOT + r'\Database\Information\allStates.txt'
STATE_CODES = ROOT + r'\Database\Information\allStateCodes.txt'
ITEM_NAMES = ROOT + r'\Interface\InterfaceUtilities\Labels\ItemNames.txt'
temp_RSP = ROOT + '\\Utilities14\Preferences\Temporary_Report_Preferences.txt'
RSP = ROOT + '\\Utilities14\Preferences\Report_Preferences.txt'

# DATABASE
FOLDER = 'YEAR'
FILE = 'STATE_CODE_001'
ID = 'STRUCTURE_NUMBER_008'
LATITUDE = 'LAT_016'
LONGITUDE = 'LONG_017'
ENCODING = 'latin1'

# GRAPHS
GRAPH_TITLES = (
    "Transition Probabilities", "Raw Deterioration", "Simulation2 Deterioration", "Raw Frequency (per Year)",
    "Simulation2 Frequency (per Year)", "Raw Frequency (per State)", "Simulation2 Frequency (per State)",
    "3D Frequency")
AXIS_LABELS = (
    ("Time (Years)", "Probability"), ("Time (Years)", "State"), ("Time (Years)", "Frequency"), ("State", "Frequency"))


def collect_headers(dir):
    """
    Create a file containing all unique
    column names in the entire database
    """
    headers = [FOLDER]
    for folder in os.listdir(dir):
        path = dir + '/' + folder
        for filepath in os.listdir(path):
            try:
                with open(path + '/' + filepath, newline='') as file:
                    header = next(csv.reader(file))
                for column in header:
                    if column not in headers:
                        headers.append(column)
            except:
                pass

    with open(ITEM_NAMES, "w") as file:
        for header in headers:
            file.write(header + '\n')

    return headers


HEADERS = collect_headers(DATABASE)
# ENUMERATE OF FEATURES IN DATABASE
FEATURES = {}
with open(ITEM_NAMES, "r") as f:
    for i, line in enumerate(f):
        FEATURES[i] = line.strip()

# List of items that can transition through states
tmp = []
with open(VALUES, "r") as parameterValuesFile:
    with open(ITEM_NAMES, "r") as parameterNamesFile:
        for name, value in zip(csv.reader(parameterNamesFile), (csv.reader(parameterValuesFile))):
            if name[0] != FOLDER and name[0] != FILE and "None" not in value:
                tmp.append(name[0])
LEARNABLE = [FEATURES[i] for i in FEATURES if (FEATURES[i]) in tmp]

from Utilities.Dependencies import *

# PATHS
ROOT = os.getcwd()
ICONS = ROOT + r'\Interface\InterfaceUtilities\Icons'
DATABASE = ROOT + '\Database\Years\\'
VALUES = ROOT + r'\Interface\InterfaceUtilities\Labels\parameterValues.txt'
STATES = ROOT + r'\Database\Information\allStates.txt'
STATE_CODES = ROOT + r'\Database\Information\allStateCodes.txt'
ITEM_NAMES = ROOT + r'\Interface\InterfaceUtilities\Labels\ItemNames.txt'
temp_RSP = ROOT + '\\Utilities14\Preferences\Temporary_Report_Preferences.txt'
RSP = ROOT + '\\Utilities14\Preferences\Report_Preferences.txt'
WINDOW_ICON = ICONS + r'\window.png'

# DATABASE
FOLDER = 'YEAR'
FILE = 'STATE_CODE_001'
ID = 'STRUCTURE_NUMBER_008'
LATITUDE = 'LAT_016'
LONGITUDE = 'LONG_017'
ENCODING = 'latin1'

# GRAPHS
GRAPH_TITLES = (
    "Transition Probabilities", "Raw Deterioration", "Simulation2 Deterioration", "Raw Frequency (per Year)",
    "Simulation2 Frequency (per Year)", "Raw Frequency (per State)", "Simulation2 Frequency (per State)",
    "3D Frequency")
AXIS_LABELS = (
    ("Time (Years)", "Probability"), ("Time (Years)", "State"), ("Time (Years)", "Frequency"), ("State", "Frequency"))


def collect_headers(dir):
    """
    Create a file containing all unique
    column names in the entire database
    """
    headers = [FOLDER]
    for folder in os.listdir(dir):
        path = dir + '/' + folder
        for filepath in os.listdir(path):
            try:
                with open(path + '/' + filepath, newline='') as file:
                    header = next(csv.reader(file))
                for column in header:
                    if column not in headers:
                        headers.append(column)
            except:
                pass

    with open(ITEM_NAMES, "w") as file:
        for header in headers:
            file.write(header + '\n')


collect_headers(DATABASE)
# ENUMERATE OF FEATURES IN DATABASE
FEATURES = {}
with open(ITEM_NAMES, "r") as f:
    for i, line in enumerate(f):
        FEATURES[i] = line.strip()

# List of items that can transition through states
tmp = []
with open(VALUES, "r") as parameterValuesFile:
    with open(ITEM_NAMES, "r") as parameterNamesFile:
        for name, value in zip(csv.reader(parameterNamesFile), (csv.reader(parameterValuesFile))):
            if name[0] != FOLDER and name[0] != FILE and "None" not in value:
                tmp.append(name[0])
LEARNABLE = [FEATURES[i] for i in FEATURES if (FEATURES[i]) in tmp]

# DATABASE
ENCODING = 'utf-5'
ID = 'STRUCTURE_NUMBER_008'
LATITUDE = 'LAT_016'
LONGITUDE = 'LONG_017'

SEPARATOR = ','
HEADERS = ['col1', 'col2']

import csv
import os

# PATHS
MASTER = "C:\\Users\\frano\PycharmProjects"
ROOT = r'\BigData'
DATABASE = MASTER + ROOT + '\Database\Years\\'
VALUES = MASTER + ROOT + r'\Interface\InterfaceUtilities\Labels\parameterValues.txt'
STATES = MASTER + ROOT + r'\Database\Information\allStates.txt'
STATE_CODES = MASTER + ROOT + r'\Database\Information\allStateCodes.txt'
ITEM_NAMES = MASTER + ROOT + r'\Interface\InterfaceUtilities\Labels\ItemNames.txt'
temp_RSP = MASTER + ROOT + '\\Utilities14\Preferences\Temporary_Report_Preferences.txt'
RSP = MASTER + ROOT + '\\Utilities14\Preferences\Report_Preferences.txt'

# DATABASE
FOLDER = 'YEAR'
FILE = 'STATE_CODE_001'
ID = 'STRUCTURE_NUMBER_008'
LATITUDE = 'LAT_016'
LONGITUDE = 'LONG_017'
ENCODING = 'latin1'

# GRAPHS
GRAPH_TITLES = (
    "Transition Probabilities", "Raw Deterioration", "Simulation2 Deterioration", "Raw Frequency (per Year)",
    "Simulation2 Frequency (per Year)", "Raw Frequency (per State)", "Simulation2 Frequency (per State)",
    "3D Frequency")
AXIS_LABELS = (
    ("Time (Years)", "Probability"), ("Time (Years)", "State"), ("Time (Years)", "Frequency"), ("State", "Frequency"))


def collect_headers(dir):
    """
    Create a file containing all unique
    column names in the entire database
    """
    headers = [FOLDER]
    for folder in os.listdir(dir):
        path = dir + '/' + folder
        for filepath in os.listdir(path):
            try:
                with open(path + '/' + filepath, newline='') as file:
                    header = next(csv.reader(file))
                for column in header:
                    if column not in headers:
                        headers.append(column)
            except:
                pass

    with open(ITEM_NAMES, "w") as file:
        for header in headers:
            file.write(header + '\n')


collect_headers(DATABASE)
# ENUMERATE OF FEATURES IN DATABASE
FEATURES = {}
with open(ITEM_NAMES, "r") as f:
    for i, line in enumerate(f):
        FEATURES[i] = line.strip()

# List of items that can transition through states
tmp = []
with open(VALUES, "r") as parameterValuesFile:
    with open(ITEM_NAMES, "r") as parameterNamesFile:
        for name, value in zip(csv.reader(parameterNamesFile), (csv.reader(parameterValuesFile))):
            if name[0] != FOLDER and name[0] != FILE and "None" not in value:
                tmp.append(name[0])
LEARNABLE = [FEATURES[i] for i in FEATURES if (FEATURES[i]) in tmp]

# DATABASE
ID = None  # 'STRUCTURE_NUMBER_008'
LATITUDE = 'LAT_016'
LONGITUDE = 'LONG_017'
ENCODING = 'latin1'

# GRAPHS
GRAPH_TITLES = (
    "Transition Probabilities", "Raw Deterioration", "Simulation2 Deterioration", "Raw Frequency (per Year)",
    "Simulation2 Frequency (per Year)", "Raw Frequency (per State)", "Simulation2 Frequency (per State)",
    "3D Frequency")
AXIS_LABELS = (
    ("Time (Years)", "Probability"), ("Time (Years)", "State"), ("Time (Years)", "Frequency"), ("State", "Frequency"))

from Utilities32.Dependencies import *

# PATHS
ROOT = os.getcwd()
ICONS = ROOT + r'\Interface\InterfaceUtilities\Icons'
DATABASE = ROOT + '\Database\Years\\'
VALUES = ROOT + r'\Interface\InterfaceUtilities\Labels\parameterValues.txt'
STATES = ROOT + r'\Database\Information\allStates.txt'
STATE_CODES = ROOT + r'\Database\Information\allStateCodes.txt'
ITEM_NAMES = ROOT + r'\Interface\InterfaceUtilities\Labels\ItemNames.txt'
temp_RSP = ROOT + '\\Utilities14\Preferences\Temporary_Report_Preferences.txt'
RSP = ROOT + '\\Utilities14\Preferences\Report_Preferences.txt'
WINDOW_ICON = ICONS + r'\window.png'

# DATABASE
FOLDER = 'YEAR'
FILE = 'STATE_CODE_001'
ID = 'STRUCTURE_NUMBER_008'
LATITUDE = 'LAT_016'
LONGITUDE = 'LONG_017'
ENCODING = 'latin1'

# GRAPHS
GRAPH_TITLES = (
    "Transition Probabilities", "Raw Deterioration", "Simulation2 Deterioration", "Raw Frequency (per Year)",
    "Simulation2 Frequency (per Year)", "Raw Frequency (per State)", "Simulation2 Frequency (per State)",
    "3D Frequency")
AXIS_LABELS = (
    ("Time (Years)", "Probability"), ("Time (Years)", "State"), ("Time (Years)", "Frequency"), ("State", "Frequency"))


def collect_headers(dir):
    """
    Create a file containing all unique
    column names in the entire database
    """
    headers = [FOLDER]
    for folder in os.listdir(dir):
        path = dir + '/' + folder
        for filepath in os.listdir(path):
            try:
                with open(path + '/' + filepath, newline='') as file:
                    header = next(csv.reader(file))
                for column in header:
                    if column not in headers:
                        headers.append(column)
            except:
                pass

    with open(ITEM_NAMES, "w") as file:
        for header in headers:
            file.write(header + '\n')


collect_headers(DATABASE)
# ENUMERATE OF FEATURES IN DATABASE
FEATURES = {}
with open(ITEM_NAMES, "r") as f:
    for i, line in enumerate(f):
        FEATURES[i] = line.strip()

# List of items that can transition through states
tmp = []
with open(VALUES, "r") as parameterValuesFile:
    with open(ITEM_NAMES, "r") as parameterNamesFile:
        for name, value in zip(csv.reader(parameterNamesFile), (csv.reader(parameterValuesFile))):
            if name[0] != FOLDER and name[0] != FILE and "None" not in value:
                tmp.append(name[0])
LEARNABLE = [FEATURES[i] for i in FEATURES if (FEATURES[i]) in tmp]

import csv

# PATHS
MASTER = "C:\\Users\\frano\PycharmProjects"
ROOT = r'\untitled'
ICONS = MASTER + ROOT + r'\Interface\InterfaceUtilities\Icons'
DATABASE = MASTER + ROOT + '\Database\Years\\'
VALUES = MASTER + ROOT + r'\Interface\InterfaceUtilities\Labels\parameterValues.txt'
STATES = MASTER + ROOT + r'\Database\Information\allStates.txt'
STATE_CODES = MASTER + ROOT + r'\Database\Information\allStateCodes.txt'
ITEM_NAMES = MASTER + ROOT + r'\Interface\InterfaceUtilities\Labels\ItemNames.txt'
temp_RSP = MASTER + ROOT + '\\Utilities14\Preferences\Temporary_Report_Preferences.txt'
RSP = MASTER + ROOT + '\\Utilities14\Preferences\Report_Preferences.txt'
WINDOW_ICON = ICONS + r'\window.png'

# DATABASE
FOLDER = 'YEAR'
FILE = 'STATE_CODE_001'
ID = 'STRUCTURE_NUMBER_008'
LATITUDE = 'LAT_016'
LONGITUDE = 'LONG_017'
ENCODING = 'latin1'

# GRAPHS
GRAPH_TITLES = (
    "Transition Probabilities", "Raw Deterioration", "Simulation2 Deterioration", "Raw Frequency (per Year)",
    "Simulation2 Frequency (per Year)", "Raw Frequency (per State)", "Simulation2 Frequency (per State)",
    "3D Frequency")
AXIS_LABELS = (
    ("Time (Years)", "Probability"), ("Time (Years)", "State"), ("Time (Years)", "Frequency"), ("State", "Frequency"))


def collect_headers(dir):
    """
    Create a file containing all unique
    column names in the entire database
    """
    headers = [FOLDER]
    for folder in os.listdir(dir):
        path = dir + '/' + folder
        for filepath in os.listdir(path):
            try:
                with open(path + '/' + filepath, newline='') as file:
                    header = next(csv.reader(file))
                for column in header:
                    if column not in headers:
                        headers.append(column)
            except:
                pass

    with open(ITEM_NAMES, "w") as file:
        for header in headers:
            file.write(header + '\n')


collect_headers(DATABASE)
# ENUMERATE OF FEATURES IN DATABASE
FEATURES = {}
with open(ITEM_NAMES, "r") as f:
    for i, line in enumerate(f):
        FEATURES[i] = line.strip()

# List of items that can transition through states
tmp = []
with open(VALUES, "r") as parameterValuesFile:
    with open(ITEM_NAMES, "r") as parameterNamesFile:
        for name, value in zip(csv.reader(parameterNamesFile), (csv.reader(parameterValuesFile))):
            if name[0] != FOLDER and name[0] != FILE and "None" not in value:
                tmp.append(name[0])
LEARNABLE = [FEATURES[i] for i in FEATURES if (FEATURES[i]) in tmp]
# STATISTICS ------------------------------------------------------------

Quantiles = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1]

# FUNCTION STRINGS ------------------------------------------------------
DISTRIBUTIONS = [  # st.alpha,
    st.anglit, st.arcsine,  # st.beta,
    # st.betaprime,
    st.bradford, st.burr, st.burr12, st.cauchy,  # st.chi,
    # st.chi2,
    st.cosine,  # st.dgamma,
    st.dweibull,  # st.erlang,
    st.expon,  # st.exponnorm,
    st.exponweib, st.exponpow,  # st.f,
    st.fatiguelife, st.fisk, st.foldcauchy, st.foldnorm,  # st.frechet_r,
    # st.frechet_l,
    st.genlogistic, st.genpareto,  # st.gennorm,
    # st.genexpon,
    # st.genextreme,
    # st.gausshyper,
    # st.gamma,
    # st.gengamma,
    # st.genhalflogistic,
    # st.gilbrat,
    # st.gompertz,
    # st.gumbel_r,
    # st.gumbel_l,
    st.halfcauchy, st.halflogistic, st.halfnorm,  # st.halfgennorm,
    st.hypsecant,  # st.invgamma,
    st.invgauss, st.invweibull, st.johnsonsb,  # st.johnsonsu,
    # st.ksone,
    # st.kstwobign,
    st.laplace, st.levy, st.levy_l,  # st.levy_stable,
    st.logistic,  # st.loggamma,
    # st.loglaplace,
    # st.lognorm,
    st.lomax, st.maxwell, st.mielke,  # st.moyal,
    # st.nakagami,
    # st.ncx2,
    # st.ncf,
    # st.nct,
    st.norm, st.pareto,  # st.pearson3,
    st.powerlaw,  # st.powerlognorm,
    # st.powernorm,
    # st.rdist,
    # st.reciprocal,
    st.rayleigh,  # st.rice,
    st.recipinvgauss, st.semicircular,  # st.t,
    # st.triang,
    st.truncexpon,  # st.truncnorm,
    # st.tukeylambda,
    # st.uniform,
    # st.vonmises,
    # st.vonmises_line,
    st.wald, st.weibull_min, st.weibull_max,  # st.wrapcauchy
]

DISTRIBUTIONNAMES = ['Alpha', 'Anglit', 'Arcsine', 'beta', 'Beta Prime', 'Bradford', 'Burr Type III', 'Burr Type XII',
                     'Cauchy', 'Chi', 'Chi-Squared', 'Cosine', 'Double Gamma', 'Double Weibull', 'Erlang',
                     'Exponential', 'Exponentially Modified Normal', 'Exponentiated Weibull', 'Exponential Power', 'F',
                     'Fatigue-life (Birnbaum-Saunders', 'Fisk', 'Folded Cauchy', 'Folded Normal', 'Frechet_r',
                     'Frechet_l', 'Generalized Logistic', 'Generalized Pareto', 'Geneneralized Normal',
                     'Generalized Exponential', 'Generalized Extreme', 'Gauss Hypergeometric', 'Gamma',
                     'Generalized Gamma', 'Generalized Half-Logistic', 'Gilbrat', 'Gompertz(or Truncated Gumbel)',
                     'Right-Skewed Gumbel', 'Left-Skewed Gumbel', 'Half-Cauchy', 'Half-Logistic', 'Half-Normal',
                     'The Upper Half of a Generalized Normal', 'Hyperbolic Secant', 'Inverted Gamma',
                     'Inverse Gaussian', 'Inverted Weibull', 'Johnson SB', 'Johnson SU', 'ksone', 'kstwobign',
                     'Laplace', 'Levy', 'Left-Skewed Levy', 'Levy_Stable', 'Logistic', 'Log Gamma', 'Log-Laplace',
                     'Lognormal', 'Lomax (Pareto of the Second Kind)', 'Maxwell', 'Mielke\'s Beta-Kappa', 'Moyal',
                     'Nakagami', 'Non-Central Chi-Squared', 'Non-Central F Distribution', 'Non-Central Student’s T',
                     'Normal', 'Pareto', 'Pearson Type III', 'Power-Function', 'Power Log-Normal', 'Power Normal',
                     'R-Distributed', 'Reciprocal', 'Rayleigh', 'Rice', 'Reciprocal Inverse Gaussian', 'Semicircular',
                     'Student\'s T', 'Triangular', 'Truncated Exponential', 'Truncated Normal', 'Tukey-Lambda',
                     'Uniform', 'Von Mises', 'Von Mises', 'Wald', 'Weibull Minimum', 'Weibull Maximum',
                     'Wrapped Cauchy']

"""
DISTRIBUTIONNAMES = {
    'alpha': 'Alpha',
    'anglit': ,
    'arcsine': ,
    'beta': ,
    'betaprime': ,
    'bradford': ,
    'burr': ,
    'cauchy': ,
    'chi': ,
    'chi2': ,
    'cosine': ,
    'dgamma': ,
    'dweibull': ,
    'erlang': ,
    'expon': ,
    'exponnorm': ,
    'exponweib': , 
    'exponpow': , 
    'f': , 
    'fatiguelife': , 
    'fisk': ,
    'foldcauchy': , 
    'foldnorm': , 
    'frechet_r': , 
    'frechet_l': ,
    'genlogistic': , 
    'genpareto': , 
    'gennorm': , 
    'genexpon': ,
    'genextreme': , 
    'gausshyper': , 
    'gamma': , 
    'gengamma': ,
    'genhalflogistic': , 
    'gilbrat': , 
    'gompertz': , 
    'gumbel_r': ,
    'gumbel_l': , 
    'halfcauchy': , 
    'halflogistic': , 
    'halfnorm': ,
    'halfgennorm': , 
    'hypsecant': , 
    'invgamma': , 
    'invgauss': ,
    'invweibull': , 
    'johnsonsb': , 
    'johnsonsu': , 
    'ksone': ,
    'kstwobign': , 
    'laplace': , 
    'levy': , 
    'levy_l': , 
    'levy_stable': ,
    'logistic': , 
    'loggamma': , 
    'loglaplace': , 
    'lognorm': ,
    'lomax': , 
    'maxwell': , 
    'mielke': , 
    'nakagami': , 
    'ncx2': , 
    'ncf': ,
    'nct': , 
    'norm': , 
    'pareto': , 
    'pearson3': , 
    'powerlaw': ,
    'powerlognorm': , 
    'powernorm': , 
    'rdist': , 
    'reciprocal': ,
    'rayleigh': , 
    'rice': , 
    'recipinvgauss': , 
    'semicircular': ,
    't': , 
    'triang': , 
    'truncexpon': , 
    'truncnorm': ,
    'tukeylambda': , 
    'uniform': , 
    'vonmises': , 
    'vonmises_line': ,
    'wald': , 
    'weibull_min': , 
    'weibull_max': , 
    'wrapcauchy': 
}"""
from pathlib import Path

p = str(Path(__file__).parents[1])

dire = os.path.join(p, os.path.join('HTML2', 'FunctionStrings'))

# PLOT DIRECTORIES AND FILE NAMES ---------------------------------------

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

PlotDirs = [BarPlotsDir, BoxPlotsDir, DistributionPlotsDir, DistributionFitPlotsDir, HeatmapPlotsDir, ScatterPlotsDir,
            TablePlotsDir, ViolinPlotsDir, StatComparisonDir]

BarPlotSuffix = '_bar'
BoxPlotSuffix = '_box'
DistributionPlotSuffix = '_distribution'
DistributionFitPlotSuffix = '_distributionFit'
ScatterPlotSuffix = '_scatter'
TablePlotSuffix = '_table'
ViolinPlotSuffix = '_violin'

# STATISTICS ------------------------------------------------------------

Quantiles = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1]

# REPORT GENERATOR ------------------------------------------------------

beginDoc = str('<!DOCTYPE html>\n<html>\n<link rel="stylesheet" href="ReportCSS4.css">\n<body>')
endDoc = str('</body>\n</html>')
indent = "&emsp;&emsp;&emsp;&emsp;"
subSpaceP = "</br></br></br>"

from Utilities.Dependencies import *

# SYSTEM
SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
HAND = 'hand2'
PTR = 'left_ptr'
# NUMERICAL
CURRENT_YEAR = datetime.datetime.now().year - 1
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
INDICATORS = ('--', '-')
NA_VALUES = ['N']
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
UNITS = ('km', 'm', 'mi', 'ft')
from Utilities33.Variables import *

# SYSTEM
SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
HAND = 'hand2'
PTR = 'left_ptr'

# EXTENSIONS
KML_EXT = '.kml'
QRY_EXT = '.qry'
TXT_EXT = '.txt'
XLSX_EXT = '.xlsx'

# EXTENSION TUPLES
ALL_FILES = 'All files (*.*)'
KML_FILES = 'kml files (*.kml)'
QRY_FILES = 'Query files (*.qry)'
TXT_FILES = 'Text files (*.txt)'
XLSX_FILES = 'xlsx files (*.xlsx)'

# NUMERICAL
CURRENT_YEAR = datetime.datetime.now().year - 1
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
INDICATORS = ('--', '-')
NA_VALUES = ['N']
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
UNITS = ('km', 'm', 'mi', 'ft')

# GRAPHS
GRAPH_COLORS = (
    ('#3CE1E0', '#56A71E', '#7CD530', '#B6ED15', '#E6F70E', '#F5AB23', '#F06D12', '#B43519', '#891A1A', '#000000'), (
        'maroon', 'darkred', 'brown', 'firebrick', 'crimson', 'indianred', 'lightcoral', 'salmon', 'rosybrown',
        'darksalmon', 'lightsalmon', 'tomato', 'orangered', 'darkorange', 'orange'), (
        'darkcyan', 'teal', 'steelblue', 'cadetblue', 'lightslategrey', 'skyblue', 'turquoise', 'lightseagreen',
        'darkturquoise'))
GRAPH_FILL_STYLES = ('none',)  # top, bottom, right, left, full: markerfacecoloralt='gray'
GRAPH_MARKERS = ('o',)

# DESCRIPTIONS
ICONS = ROOT + r'\Interface\InterfaceUtilities\Icons'

WINDOW_ICON = ICONS + r'\window.png'

OPEN_FOLDER_DESCRIPTION = 'Open a folder from which to extract files.'
OPEN_FOLDER_ICON = ICONS + r'\openFolder.png'

SAVE_FOLDER_DESCRIPTION = 'Open a folder to save files in.'
SAVE_FOLDER_ICON = ICONS + r'\saveFolder.png'

OPEN_FILE_DESCRIPTION = 'Open file.'
OPEN_FILE_ICON = ICONS + r'\openFile.png'

SAVE_FILE_DESCRIPTION = 'Save file.'
SAVE_FILE_ICON = ICONS + r'\saveFile.png'

EXIT_DESCRIPTION = 'Exit application.'
EXIT_ICON = ICONS + r'\exit.png'

# Data encapsulates interface7, geosearch, machine learning, and sentiment analysis
DATABASE_DESCRIPTION = 'Database: Set your database and its variables.'
DATABASE_ICON = ICONS + r'\database.png'

SEARCH_DESCRIPTION = 'search Engine: Navigate your database and specify criteria for extracting data.'
SEARCH_ICON = ICONS + r'\search.png'

GEO_SEARCH_DESCRIPTION = 'Geographic search: Explore your database and produce geographical information system visualizations.'
GEO_SEARCH_ICON = ICONS + r'\geoSearch.png'

MACHINE_LEARNING_DESCRIPTION = 'Machine Learning: Perform statistical analysis on your data and visualize your results.'
MACHINE_LEARNING_ICON = ICONS + r'\machineLearning.png'

SENTIMENT_DESCRIPTION = 'Sentiment Analysis: Discover popular opinion.'
SENTIMENT_ICON = ICONS + r'\sentiment.png'

# Assist encapsulates updates and help
UPDATES_DESCRIPTION = 'Updates: Check for updates.'
UPDATES_ICON = ICONS + r'\updates.png'

HELP_DESCRIPTION = 'Help2: Browse 1 for software information and examples.'
HELP_ICON = ICONS + r'\help.png'

# Planner encapsulates schedule and tasks
PLANNER_DESCRIPTION = 'Planner: Adjust variables used within the planner such as date and time settings.'
PLANNER_ICON = ICONS + r'\plannerSettings.png'

SCHEDULE_DESCRIPTION = 'Schedule: Organize events and develop schedules.'
SCHEDULE_ICON = ICONS + r'\schedule.png'

TASKS_DESCRIPTION = 'Tasks: Delegate and monitor tasks.'
TASKS_ICON = ICONS + r'\tasks.png'

RESOURCES_DESCRIPTION = 'Resources: Manage your inventories.'
RESOURCES_ICON = ICONS + r'\resources.png'

FINANCE_DESCRIPTION = 'Finance: Prepare budgets and keep track of transactions.'
FINANCE_ICON = ICONS + r'\finance.png'

BACKGROUND_IMAGE = ICONS + r'\background.png'

DOWN_ARROW_ICON = ICONS + r'downArrow.png'

BACKGROUND_COLOR_DARK = r'rgb(25,25,25)'
BACKGROUND_COLOR_LIGHT = r'rgb(49,49,49)'
BORDER_COLOR = r'rgb(70,70,70)'
FONT_COLOR = r'rgb(255,255,255)'

BUTTON_COLOR = r'rgb(55,55,55)'

BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'

LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(32,33,30), stop: 0.5 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(22,23,20), stop: 0.5 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'
WIDGET_STYLE = """
            QWidget
            {
                color: """ + FONT_COLOR + """;
                background-color: """ + BACKGROUND_COLOR_DARK + """;
                selection-background-color:""" + BACKGROUND_COLOR_LIGHT + """;
                selection-color: """ + FONT_COLOR + """;
                background-clip: border;
                border-image: none;
                outline: 0;
            }
            """
ENTRY_STYLE = """
            QWidget
            {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
                border: 2px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            }
            """
WINDOW_STYLE = """
            QMainWindow {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);

            }
            QMainWindow::separator {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + BACKGROUND_COLOR_LIGHT + """;
                width: 10px; /* when vertical */
                height: 10px; /* when horizontal */
            }

            QMainWindow::separator:hover {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: black;
            }

            """

CHECK_STYLE = """
            QCheckBox {
                spacing: 5px;
            }

            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }

            QCheckBox::indicator:unchecked {
            }

            QCheckBox::indicator:unchecked:hover {
            }

            QCheckBox::indicator:unchecked:pressed {
            }

            QCheckBox::indicator:checked {
            }

            QCheckBox::indicator:checked:hover {
            }

            QCheckBox::indicator:checked:pressed {
            }

            QCheckBox::indicator:indeterminate:hover {
            }

            QCheckBox::indicator:indeterminate:pressed {
            }
            """

COMBO_STYLE = """
            QComboBox {
                background: """ + LIGHT_GRADIENT + """;
                color: rgb(255,255,255);
                border: 3px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                padding: 1px 1px 1px 3px;
                min-width: 6px;
            }

            QComboBox:!editable::hover {
                background: """ + DARK_GRADIENT + """; 
            }

            QComboBox:editable {
                background: """ + DARK_GRADIENT + """;         
            }

            /* QComboBox gets the "on" state when the popup is open */
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: """ + DARK_GRADIENT + """;
            }

            QComboBox:on { /* shift the text when the popup opens */
                padding-top: 3px;
                padding-left: 4px;
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 10px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: """ + BORDER_STYLE + """;
                border-top-right-radius: """ + BORDER_RADIUS + """;
                border-bottom-right-radius: """ + BORDER_RADIUS + """;
            }

            QComboBox::down-arrow {

            }

            QComboBox::down-arrow:on { /* shift the arrow when popup is open */
                top: 1px;
                left: 1px;
            }
            """

FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """
LABEL_STYLE = """
            QLabel {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;
                padding: 2px;
            }
            """

MENUBAR_STYLE = """
            QMenuBar {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid rgb(0,0,0);
            }

            QMenuBar::item {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
            }

            QMenuBar::item::selected {
                background: """ + BACKGROUND_COLOR_DARK + """;
            }

            QMenu {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid #000;           
            }

            QMenu::item::selected {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
            }
            """

SCROLL_STYLE = """
            QScrollBar:vertical {
                 background: """ + BACKGROUND_COLOR_DARK + """;
                 width: 15px;
                 margin: 22px 0 22px 0;
             }
             QScrollBar::handle:vertical {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::handle:vertical:pressed {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::add-line:vertical {
                 background: """ + LIGHT_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: bottom;
                 subcontrol-origin: margin;
             }

             QScrollBar::sub-line:vertical {
                 background: """ + DARK_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: top;
                 subcontrol-origin: margin;
             }
             QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                 border: 2px solid """ + DARK_GRADIENT + """;
                 width: 3px;
                 height: 3px;
                 background: black;
             }
            QScrollBar::up-arrow:vertical:pressed, QScrollBar::down-arrow:vertical:pressed {
                 border: 2px solid """ + LIGHT_GRADIENT + """;
                 width: 3px;
                 height: 3px;
                 background: black;
             }

             QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                 background: none;
             }
             """
SPLITTER_STYLE = """
            QSplitter {
                background: """ + BACKGROUND_COLOR_DARK + """;
            }
            QSplitter::handle {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
            }

            QSplitter::handle:hover {
                background: white;
            }


            QSplitter::handle:horizontal {
                width: 8px;
            }

            QSplitter::handle:vertical {
                height: 8px;
            }

            QSplitter::handle:pressed {
                background-color: """ + DARK_GRADIENT + """;
            }
            """

TOOLBAR_STYLE = """ 
            QToolBar, QToolButton, QToolTip { 
                background: rgb(56,60,55);
                background: """ + LIGHT_GRADIENT + """;

                color: """ + FONT_COLOR + """;
                spacing: 3px; /* spacing between items in the tool bar */
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;

            } 
            QToolBar {
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }


            QToolButton:hover {
                background: """ + DARK_GRADIENT + """;
                border: 0px;
            }

            QToolBar::handle {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                border: 1px solid rgb(100,100,100);
            } 
            """

# TODO and use os module
# The paths generator1 must read the entire database, collecting the
# necessary data to fill the lists in the list generator1 as well as information in the paths above.
# This process will occur each time a different database is used and a current working database
# file will be created to save the configurations yielded by the aforementioned procedure so that
# it doesn't need to be performed each time the program starts

from Utilities19.Dependencies import *

# SYSTEM
SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
HAND = 'hand2'
PTR = 'left_ptr'
# NUMERICAL
CURRENT_YEAR = datetime.datetime.now().year - 1
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
INTERVALS = {'from to': '--', 'between': '-'}
NA_VALUES = ['N']
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt, '==': operator.eq}
INVERTED_OPERATORS = {operator.ge: '>=', operator.le: '<=', operator.gt: '>', operator.lt: '<', operator.eq: '=='}
OPERATORS_WORDS = {
    'greaterthanorequalto': '>=', 'lessthanorequalto': '<=', 'greaterthan': '>', 'lessthan': '<',
    'equal'               : '=='
}

UNITS = ('km', 'm', 'mi', 'ft')

CLASSIFIERS_NAMES = (
    'Adaptive Boost', 'Decision Tree', 'Naive Bayes', 'Gaussian Process', 'Nearest Neighbors', 'Multilayer Perceptron',
    'Quadratic Discriminant', 'Random Forest', 'Stochastic Gradient Descent', 'Support Vector Machine')
CLUSTERERS_NAMES = ('',)
DIMENSIONALITY_REDUCERS_NAMES = ('',)
REGRESSORS_NAMES = (
    'Adaptive Boost', 'Decision Tree', 'Elastic Net', 'Gaussian Process', 'Nearest Neighbors', 'Kernel Ridge',
    'Multilayer Perceptron', 'Random Forest', 'Stochastic Gradient Descent', 'Support Vector Machine')

from Utilities.Dependencies import *
from Utilities.Variables import *

# SYSTEM
SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
HAND = 'hand2'
PTR = 'left_ptr'

# PATHS
ROOT = os.getcwd()
DATABASE = ROOT + '\Database\Years\\'
VALUES = ROOT + r'\Interface\InterfaceUtilities\Labels\parameterValues.txt'
STATES = ROOT + r'\Database\Information\allStates.txt'
STATE_CODES = ROOT + r'\Database\Information\allStateCodes.txt'
ITEM_NAMES = ROOT + r'\Interface\InterfaceUtilities\Labels\ItemNames.txt'
temp_RSP = ROOT + '\\Utilities14\Preferences\Temporary_Report_Preferences.txt'
RSP = ROOT + '\\Utilities14\Preferences\Report_Preferences.txt'
TEMP_SAVE = ROOT + "\Database\\temp.txt"
BASE_MAP = ROOT + r'\Graphs\Maps\baseMap.html'
BASE_GRAPH = ROOT + r'\Graphs\Main.html'


def collect_headers(dir):
    """
    Create a file containing all unique
    column names in the entire database
    """
    headers = [FOLDER]
    for folder in os.listdir(dir):
        path = dir + '/' + folder
        for filepath in os.listdir(path):
            try:
                with open(path + '/' + filepath, newline='') as file:
                    header = next(csv.reader(file))
                for column in header:
                    if column not in headers:
                        headers.append(column)
            except:
                pass

    with open(ITEM_NAMES, "w") as file:
        for header in headers:
            file.write(header + '\n')

    return headers


HEADERS = collect_headers(DATABASE)
# ENUMERATE OF FEATURES IN DATABASE
FEATURES = {}
with open(ITEM_NAMES, "r") as f:
    for i, line in enumerate(f):
        FEATURES[i] = line.strip()

# List of items that can transition through states
tmp = []
with open(VALUES, "r") as parameterValuesFile:
    with open(ITEM_NAMES, "r") as parameterNamesFile:
        for name, value in zip(csv.reader(parameterNamesFile), (csv.reader(parameterValuesFile))):
            if name[0] != FOLDER and name[0] != FILE and "None" not in value:
                tmp.append(name[0])
LEARNABLE = [FEATURES[i] for i in FEATURES if (FEATURES[i]) in tmp]

# EXTENSIONS
CSV_EXT = '.csv'
KML_EXT = '.kml'
PDF_EXT = '.pdf'
QRY_EXT = '.qry'
TXT_EXT = '.txt'
XLSX_EXT = '.xlsx'

# EXTENSION TUPLES
ALL_FILES = 'All files (*.*)'
CSV_FILES = 'csv files (*.csv);;'
KML_FILES = 'kml files (*.kml);;'
PDF_FILES = 'pdf files (*.pdf);;'
QRY_FILES = 'Query files (*.qry);;'
TXT_FILES = 'text files (*.txt);;'
XLSX_FILES = 'xlsx files (*.xlsx);;'

# NUMERICAL
CURRENT_YEAR = datetime.datetime.now().year - 1
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
INDICATORS = ('--', '-')
NA_VALUES = ['N']
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
UNITS = ('km', 'm', 'mi', 'ft')

# GRAPHS
GRAPH_COLORS = (
    ('#3CE1E0', '#56A71E', '#7CD530', '#B6ED15', '#E6F70E', '#F5AB23', '#F06D12', '#B43519', '#891A1A', '#000000'), (
        'maroon', 'darkred', 'brown', 'firebrick', 'crimson', 'indianred', 'lightcoral', 'salmon', 'rosybrown',
        'darksalmon', 'lightsalmon', 'tomato', 'orangered', 'darkorange', 'orange'), (
        'darkcyan', 'teal', 'steelblue', 'cadetblue', 'lightslategrey', 'skyblue', 'turquoise', 'lightseagreen',
        'darkturquoise'))
GRAPH_FILL_STYLES = ('none',)  # top, bottom, right, left, full: markerfacecoloralt='gray'
GRAPH_MARKERS = ('o',)

# DESCRIPTIONS
ICONS = ROOT + r'\Interface\InterfaceUtilities\Icons'

WINDOW_ICON = ICONS + r'\window.png'

OPEN_FOLDER_DESCRIPTION = 'Open a folder from which to extract files.'
OPEN_FOLDER_ICON = ICONS + r'\openFolder.png'

SAVE_FOLDER_DESCRIPTION = 'Open a folder to save files in.'
SAVE_FOLDER_ICON = ICONS + r'\saveFolder.png'

OPEN_FILE_DESCRIPTION = 'Open file.'
OPEN_FILE_ICON = ICONS + r'\openFile.png'

SAVE_FILE_DESCRIPTION = 'Save file.'
SAVE_FILE_ICON = ICONS + r'\saveFile.png'

EXIT_DESCRIPTION = 'Exit application.'
EXIT_ICON = ICONS + r'\exit.png'

# Data encapsulates interface7, geosearch, machine learning, and sentiment analysis
DATABASE_DESCRIPTION = 'Database: Set your database and its variables.'
DATABASE_ICON = ICONS + r'\database.png'

SEARCH_DESCRIPTION = 'search Engine: Navigate your database and specify criteria for extracting data.'
SEARCH_ICON = ICONS + r'\search.png'

GEO_SEARCH_DESCRIPTION = 'Geographic search: Explore your database and produce geographical information system visualizations.'
GEO_SEARCH_ICON = ICONS + r'\geoSearch.png'

MACHINE_LEARNING_DESCRIPTION = 'Machine Learning: Perform statistical analysis on your data and visualize your results.'
MACHINE_LEARNING_ICON = ICONS + r'\machineLearning.png'

SENTIMENT_DESCRIPTION = 'Sentiment Analysis: Discover popular opinion.'
SENTIMENT_ICON = ICONS + r'\sentiment.png'

# Assist encapsulates updates and help
UPDATES_DESCRIPTION = 'Updates: Check for updates.'
UPDATES_ICON = ICONS + r'\updates.png'

HELP_DESCRIPTION = 'Help2: Browse 1 for software information and examples.'
HELP_ICON = ICONS + r'\help.png'

# Planner encapsulates schedule and tasks
PLANNER_DESCRIPTION = 'Planner: Adjust variables used within the planner such as date and time settings.'
PLANNER_ICON = ICONS + r'\plannerSettings.png'

SCHEDULE_DESCRIPTION = 'Schedule: Organize events and develop schedules.'
SCHEDULE_ICON = ICONS + r'\schedule.png'

TASKS_DESCRIPTION = 'Tasks: Delegate and monitor tasks.'
TASKS_ICON = ICONS + r'\tasks.png'

RESOURCES_DESCRIPTION = 'Resources: Manage your inventories.'
RESOURCES_ICON = ICONS + r'\resources.png'

FINANCE_DESCRIPTION = 'Finance: Prepare budgets and keep track of transactions.'
FINANCE_ICON = ICONS + r'\finance.png'

BACKGROUND_IMAGE = ICONS + r'\background.png'

DOWN_ARROW_ICON = ICONS + r'downArrow.png'

BACKGROUND_COLOR_DARK = r'rgb(25,25,25)'
BACKGROUND_COLOR_LIGHT = r'rgb(49,49,49)'
BORDER_COLOR = r'rgb(70,70,70)'
FONT_COLOR = r'rgb(255,255,255)'

BUTTON_COLOR = r'rgb(55,55,55)'

BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'

LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(32,33,30), stop: 0.5 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(22,23,20), stop: 0.5 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'
WIDGET_STYLE = """
            QWidget
            {
                color: """ + FONT_COLOR + """;
                background-color: """ + BACKGROUND_COLOR_DARK + """;
                selection-background-color:""" + BACKGROUND_COLOR_LIGHT + """;
                selection-color: """ + FONT_COLOR + """;
                background-clip: border;
                border-image: none;
                outline: 0;
            }
            """
ENTRY_STYLE = """
            QWidget
            {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
                border: 2px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            }
            """
WINDOW_STYLE = """
            QMainWindow {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);

            }
            QMainWindow::separator {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + BACKGROUND_COLOR_LIGHT + """;
                width: 10px; /* when vertical */
                height: 10px; /* when horizontal */
            }

            QMainWindow::separator:hover {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: black;
            }

            """

CHECK_STYLE = """
            QCheckBox {
                spacing: 5px;
            }

            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }

            QCheckBox::indicator:unchecked {
            }

            QCheckBox::indicator:unchecked:hover {
            }

            QCheckBox::indicator:unchecked:pressed {
            }

            QCheckBox::indicator:checked {
            }

            QCheckBox::indicator:checked:hover {
            }

            QCheckBox::indicator:checked:pressed {
            }

            QCheckBox::indicator:indeterminate:hover {
            }

            QCheckBox::indicator:indeterminate:pressed {
            }
            """

COMBO_STYLE = """
            QComboBox {
                background: """ + LIGHT_GRADIENT + """;
                color: rgb(255,255,255);
                border: 3px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                padding: 1px 1px 1px 3px;
                min-width: 6px;
            }

            QComboBox:!editable::hover {
                background: """ + DARK_GRADIENT + """; 
            }

            QComboBox:editable {
                background: """ + DARK_GRADIENT + """;         
            }

            /* QComboBox gets the "on" state when the popup is open */
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: """ + DARK_GRADIENT + """;
            }

            QComboBox:on { /* shift the text when the popup opens */
                padding-top: 3px;
                padding-left: 4px;
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 10px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: """ + BORDER_STYLE + """;
                border-top-right-radius: """ + BORDER_RADIUS + """;
                border-bottom-right-radius: """ + BORDER_RADIUS + """;
            }

            QComboBox::down-arrow {

            }

            QComboBox::down-arrow:on { /* shift the arrow when popup is open */
                top: 1px;
                left: 1px;
            }
            """

FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """
LABEL_STYLE = """
            QLabel {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;
                padding: 2px;
            }
            """

MENUBAR_STYLE = """
            QMenuBar {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid rgb(0,0,0);
            }

            QMenuBar::item {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
            }

            QMenuBar::item::selected {
                background: """ + BACKGROUND_COLOR_DARK + """;
            }

            QMenu {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid #000;           
            }

            QMenu::item::selected {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
            }
            """

SCROLL_STYLE = """
            QScrollBar:vertical {
                 background: """ + BACKGROUND_COLOR_DARK + """;
                 width: 15px;
                 margin: 22px 0 22px 0;
             }
             QScrollBar::handle:vertical {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::handle:vertical:pressed {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::add-line:vertical {
                 background: """ + LIGHT_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: bottom;
                 subcontrol-origin: margin;
             }

             QScrollBar::sub-line:vertical {
                 background: """ + DARK_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: top;
                 subcontrol-origin: margin;
             }
             QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                 border: 2px solid """ + DARK_GRADIENT + """;
                 width: 3px;
                 height: 3px;
                 background: black;
             }
            QScrollBar::up-arrow:vertical:pressed, QScrollBar::down-arrow:vertical:pressed {
                 border: 2px solid """ + LIGHT_GRADIENT + """;
                 width: 3px;
                 height: 3px;
                 background: black;
             }

             QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                 background: none;
             }
             """
SPLITTER_STYLE = """
            QSplitter {
                background: """ + BACKGROUND_COLOR_DARK + """;
            }
            QSplitter::handle {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
            }

            QSplitter::handle:hover {
                background: white;
            }


            QSplitter::handle:horizontal {
                width: 8px;
            }

            QSplitter::handle:vertical {
                height: 8px;
            }

            QSplitter::handle:pressed {
                background-color: """ + DARK_GRADIENT + """;
            }
            """

TOOLBAR_STYLE = """ 
            QToolBar, QToolButton, QToolTip { 
                background: rgb(56,60,55);
                background: """ + LIGHT_GRADIENT + """;

                color: """ + FONT_COLOR + """;
                spacing: 3px; /* spacing between items in the tool bar */
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;

            } 
            QToolBar {
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }


            QToolButton:hover {
                background: """ + DARK_GRADIENT + """;
                border: 0px;
            }

            QToolBar::handle {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                border: 1px solid rgb(100,100,100);
            } 
            """

# TODO and use os module
# The paths generator1 must read the entire database, collecting the
# necessary data to fill the lists in the list generator1 as well as information in the paths above.
# This process will occur each time a different database is used and a current working database
# file will be created to save the configurations yielded by the aforementioned procedure so that
# it doesn't need to be performed each time the program starts

import ctypes
import datetime
import operator

from Utilities.MenuData import *
from Utilities.Variables import *

# SYSTEM
SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
HAND = 'hand2'
PTR = 'left_ptr'

# EXTENSIONS
KML_EXT = '.kml'
QRY_EXT = '.qry'
TXT_EXT = '.txt'
XLSX_EXT = '.xlsx'

# EXTENSION TUPLES
ALL_FILES = ('All files', '*.*')
KML_FILES = ('KML files', '*.kml')
QRY_FILES = ('Query files', '*.qry')

# NUMERICAL
CURRENT_YEAR = datetime.datetime.now().year - 1
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
INDICATORS = ('--', '-')
NA_VALUES = ['N']
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
UNITS = ('km', 'm', 'mi', 'ft')

# GRAPHS
GRAPH_COLORS = (
    ('#3CE1E0', '#56A71E', '#7CD530', '#B6ED15', '#E6F70E', '#F5AB23', '#F06D12', '#B43519', '#891A1A', '#000000'), (
        'maroon', 'darkred', 'brown', 'firebrick', 'crimson', 'indianred', 'lightcoral', 'salmon', 'rosybrown',
        'darksalmon', 'lightsalmon', 'tomato', 'orangered', 'darkorange', 'orange'), (
        'darkcyan', 'teal', 'steelblue', 'cadetblue', 'lightslategrey', 'skyblue', 'turquoise', 'lightseagreen',
        'darkturquoise'))
GRAPH_FILL_STYLES = ('none',)  # top, bottom, right, left, full: markerfacecoloralt='gray'
GRAPH_MARKERS = ('o',)

# DESCRIPTIONS
OPEN_FOLDER_DESCRIPTION = 'Open a folder from which to extract files.'
OPEN_FOLDER_ICON = ICONS + r'\openFolder.png'

SAVE_FOLDER_DESCRIPTION = 'Open a folder to save files in.'
SAVE_FOLDER_ICON = ICONS + r'\saveFolder.png'

OPEN_FILE_DESCRIPTION = 'Open file.'
OPEN_FILE_ICON = ICONS + r'\openFile.png'

SAVE_FILE_DESCRIPTION = 'Save file.'
SAVE_FILE_ICON = ICONS + r'\saveFile.png'

EXIT_DESCRIPTION = 'Exit application.'
EXIT_ICON = ICONS + r'\exit.png'

# Data encapsulates interface7, geosearch, machine learning, and sentiment analysis
DATABASE_DESCRIPTION = 'Database: Set your database and its variables.'
DATABASE_ICON = ICONS + r'\database.png'

SEARCH_DESCRIPTION = 'search Engine: Navigate your database and specify criteria for extracting data.'
SEARCH_ICON = ICONS + r'\search.png'

GEO_SEARCH_DESCRIPTION = 'Geographic search: Explore your database and produce geographical information system visualizations.'
GEO_SEARCH_ICON = ICONS + r'\geoSearch.png'

MACHINE_LEARNING_DESCRIPTION = 'Machine Learning: Perform statistical analysis on your data and visualize your results.'
MACHINE_LEARNING_ICON = ICONS + r'\machineLearning.png'

SENTIMENT_DESCRIPTION = 'Sentiment Analysis: Discover popular opinion.'
SENTIMENT_ICON = ICONS + r'\sentiment.png'

# Assist encapsulates updates and help
UPDATES_DESCRIPTION = 'Updates: Check for updates.'
UPDATES_ICON = ICONS + r'\updates.png'

HELP_DESCRIPTION = 'Help2: Browse 1 for software information and examples.'
HELP_ICON = ICONS + r'\help.png'

# Planner encapsulates schedule and tasks
PLANNER_DESCRIPTION = 'Planner: Adjust variables used within the planner such as date and time settings.'
PLANNER_ICON = ICONS + r'\plannerSettings.png'

SCHEDULE_DESCRIPTION = 'Schedule: Organize events and develop schedules.'
SCHEDULE_ICON = ICONS + r'\schedule.png'

TASKS_DESCRIPTION = 'Tasks: Delegate and monitor tasks.'
TASKS_ICON = ICONS + r'\tasks.png'

RESOURCES_DESCRIPTION = 'Resources: Manage your inventories.'
RESOURCES_ICON = ICONS + r'\resources.png'

BACKGROUND_IMAGE = ICONS + r'\background.png'

DOWN_ARROW_ICON = ICONS + r'downArrow.png'

BACKGROUND_COLOR_DARK = r'rgb(20,20,20)'
BACKGROUND_COLOR_LIGHT = r'rgb(49,49,49)'
BORDER_COLOR = r'rgb(100,100,100)'
WINDOW_STYLE = """
            color: rgb(255,255,255);
            QMainWindow {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(/interface7/InterfaceUtilities0/icon2/background.png);

            }
            QMainWindow::separator {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                width: 10px; /* when vertical */
                height: 10px; /* when horizontal */
            }

            QMainWindow::separator:hover {
                background: BACKGROUND_COLOR_LIGHT;
                color: rgb(0,0,0);
            }

            """
COMBO_STYLE = """QComboBox {
                border: 1px solid """ + BORDER_COLOR + """;
                background: """ + BACKGROUND_COLOR_DARK + """;
                border-radius: 3px;
                padding: 1px 18px 1px 3px;
                min-width: 6px;
                }

                QComboBox:editable {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                }

                QComboBox:!editable, QComboBox::drop-down:editable {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                         stop: 0 """ + BACKGROUND_COLOR_LIGHT + """, stop: 0.3 """ + BACKGROUND_COLOR_LIGHT + """,
                                         stop: 0.5 """ + BACKGROUND_COLOR_LIGHT + """, stop: 1.0 """ + BACKGROUND_COLOR_LIGHT + """);
                }

                /* QComboBox gets the "on" state when the popup is open */
                QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                        stop: 0 """ + BACKGROUND_COLOR_DARK + """, stop: 0.3 """ + BACKGROUND_COLOR_DARK + """,
                                        stop: 0.5 """ + BACKGROUND_COLOR_DARK + """, stop: 1.0 """ + BACKGROUND_COLOR_DARK + """);
                }

                QComboBox:on { /* shift the text when the popup opens */
                padding-top: 3px;
                padding-left: 4px;
                }

                QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 10px;

                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: solid; /* just a single line */
                border-top-right-radius: 3px; /* same radius as the QComboBox */
                border-bottom-right-radius: 3px;
                }

                QComboBox::down-arrow {
                image: url(downArrow.png);
                }

                QComboBox::down-arrow:on { /* shift the arrow when popup is open */
                top: 1px;
                left: 1px;
                }"""

FRAME_STYLE = """
            QFrame, QLabel, QToolTip {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                border: 1px solid """ + BORDER_COLOR + """;
                border-radius: 4px;
                padding: 2px;
            }"""
MENUBAR_STYLE = """
            QMenuBar {
                background-color: rgb(49,49,49);
                color: rgb(255,255,255);
                border: 1px solid #000;
            }

            QMenuBar::item {
                background-color: rgb(49,49,49);
                color: rgb(255,255,255);
            }

            QMenuBar::item::selected {
                background-color: rgb(30,30,30);
            }

            QMenu {
                background-color: rgb(49,49,49);
                color: rgb(255,255,255);
                border: 1px solid #000;           
            }

            QMenu::item::selected {
                background-color: rgb(30,30,30);
            }
            """
SPLITTER_STYLE = """
                QSplitter::handle {
                    background-color: """ + BACKGROUND_COLOR_LIGHT + """;
                }

                QSplitter::handle:horizontal {
                    width: 4px;
                }

                QSplitter::handle:vertical {
                    height: 4px;
                }

                QSplitter::handle:pressed {
                    background-color: """ + BACKGROUND_COLOR_DARK + """;
                }
                """
STATUSBAR_STYLE = """
                QStatusBar {
                    background: rgb(49,49,49);
                    border: 1px solid rgb(100,100,100);
                }

                QStatusBar::item {
                    border: 3px solid rgb(50,75,60);
                    border-radius: 3px;

                }"""
TOOLBAR_STYLE = """ 
                QToolBar { 
                    background: rgb(55,60,55);
                    spacing: 3px; /* spacing between items in the tool bar */
                    border: 1px solid """ + BORDER_COLOR + """;

                } 
                QToolBar::handle {
                    image: url(handle.png);
                    background: rgb(45,50,48);
                    border: 1px solid rgb(100,100,100);
                } 
                """

# TODO and use os module
# The paths generator1 must read the entire database, collecting the
# necessary data to fill the lists in the list generator1 as well as information in the paths above.
# This process will occur each time a different database is used and a current working database
# file will be created to save the configurations yielded by the aforementioned procedure so that
# it doesn't need to be performed each time the program starts

from Utilities13.Dependencies import *
from Utilities13.Variables import *

# SYSTEM
SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
HAND = 'hand2'
PTR = 'left_ptr'

# PATHS
ROOT = os.getcwd()
DATABASE = ROOT + '\Database\Years\\'
VALUES = ROOT + r'\Interface\InterfaceUtilities\Labels\parameterValues.txt'
STATES = ROOT + r'\Database\Information\allStates.txt'
STATE_CODES = ROOT + r'\Database\Information\allStateCodes.txt'
ITEM_NAMES = ROOT + r'\Interface\InterfaceUtilities\Labels\ItemNames.txt'
temp_RSP = ROOT + '\\Utilities14\Preferences\Temporary_Report_Preferences.txt'
RSP = ROOT + '\\Utilities14\Preferences\Report_Preferences.txt'
TEMP_SAVE = ROOT + "\Database\\temp.txt"
BASE_MAP = ROOT + r'\Graphs\Maps\baseMap.html'
BASE_GRAPH = ROOT + r'\Graphs\Main.html'


def collect_headers(dir):
    """
    Create a file containing all unique
    column names in the entire database
    """
    headers = [FOLDER]
    for folder in os.listdir(dir):
        path = dir + '/' + folder
        for filepath in os.listdir(path):
            try:
                with open(path + '/' + filepath, newline='') as file:
                    header = next(csv.reader(file))
                for column in header:
                    if column not in headers:
                        headers.append(column)
            except:
                pass

    with open(ITEM_NAMES, "w") as file:
        for header in headers:
            file.write(header + '\n')

    return headers


HEADERS = collect_headers(DATABASE)
# ENUMERATE OF FEATURES IN DATABASE
FEATURES = {}
with open(ITEM_NAMES, "r") as f:
    for i, line in enumerate(f):
        FEATURES[i] = line.strip()

# List of items that can transition through states
tmp = []
with open(VALUES, "r") as parameterValuesFile:
    with open(ITEM_NAMES, "r") as parameterNamesFile:
        for name, value in zip(csv.reader(parameterNamesFile), (csv.reader(parameterValuesFile))):
            if name[0] != FOLDER and name[0] != FILE and "None" not in value:
                tmp.append(name[0])
LEARNABLE = [FEATURES[i] for i in FEATURES if (FEATURES[i]) in tmp]

# EXTENSIONS
CSV_EXT = '.csv'
KML_EXT = '.kml'
PDF_EXT = '.pdf'
QRY_EXT = '.qry'
TXT_EXT = '.txt'
XLSX_EXT = '.xlsx'

# EXTENSION TUPLES
ALL_FILES = 'All files (*.*)'
CSV_FILES = 'csv files (*.csv);;'
KML_FILES = 'kml files (*.kml);;'
PDF_FILES = 'pdf files (*.pdf);;'
QRY_FILES = 'Query files (*.qry);;'
TXT_FILES = 'text files (*.txt);;'
XLSX_FILES = 'xlsx files (*.xlsx);;'

# NUMERICAL
CURRENT_YEAR = datetime.datetime.now().year - 1
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
INDICATORS = ('--', '-')
NA_VALUES = ['N']
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
UNITS = ('km', 'm', 'mi', 'ft')

# GRAPHS
GRAPH_COLORS = (
    ('#3CE1E0', '#56A71E', '#7CD530', '#B6ED15', '#E6F70E', '#F5AB23', '#F06D12', '#B43519', '#891A1A', '#000000'), (
        'maroon', 'darkred', 'brown', 'firebrick', 'crimson', 'indianred', 'lightcoral', 'salmon', 'rosybrown',
        'darksalmon', 'lightsalmon', 'tomato', 'orangered', 'darkorange', 'orange'), (
        'darkcyan', 'teal', 'steelblue', 'cadetblue', 'lightslategrey', 'skyblue', 'turquoise', 'lightseagreen',
        'darkturquoise'))
GRAPH_FILL_STYLES = ('none',)  # top, bottom, right, left, full: markerfacecoloralt='gray'
GRAPH_MARKERS = ('o',)

# DESCRIPTIONS
ICONS = ROOT + r'\Interface\InterfaceUtilities\Icons'

WINDOW_ICON = ICONS + r'\window.png'

OPEN_FOLDER_DESCRIPTION = 'Open a folder from which to extract files.'
OPEN_FOLDER_ICON = ICONS + r'\openFolder.png'

SAVE_FOLDER_DESCRIPTION = 'Open a folder to save files in.'
SAVE_FOLDER_ICON = ICONS + r'\saveFolder.png'

OPEN_FILE_DESCRIPTION = 'Open file.'
OPEN_FILE_ICON = ICONS + r'\openFile.png'

SAVE_FILE_DESCRIPTION = 'Save file.'
SAVE_FILE_ICON = ICONS + r'\saveFile.png'

EXIT_DESCRIPTION = 'Exit application.'
EXIT_ICON = ICONS + r'\exit.png'

# Data encapsulates interface7, geosearch, machine learning, and sentiment analysis
DATABASE_DESCRIPTION = 'Database: Set your database and its variables.'
DATABASE_ICON = ICONS + r'\database.png'

SEARCH_DESCRIPTION = 'search Engine: Navigate your database and specify criteria for extracting data.'
SEARCH_ICON = ICONS + r'\search.png'

GEO_SEARCH_DESCRIPTION = 'Geographic search: Explore your database and produce geographical information system visualizations.'
GEO_SEARCH_ICON = ICONS + r'\geoSearch.png'

MACHINE_LEARNING_DESCRIPTION = 'Machine Learning: Perform statistical analysis on your data and visualize your results.'
MACHINE_LEARNING_ICON = ICONS + r'\machineLearning.png'

SENTIMENT_DESCRIPTION = 'Sentiment Analysis: Discover popular opinion.'
SENTIMENT_ICON = ICONS + r'\sentiment.png'

# Assist encapsulates updates and help
UPDATES_DESCRIPTION = 'Updates: Check for updates.'
UPDATES_ICON = ICONS + r'\updates.png'

HELP_DESCRIPTION = 'Help2: Browse 1 for software information and examples.'
HELP_ICON = ICONS + r'\help.png'

# Planner encapsulates schedule and tasks
PLANNER_DESCRIPTION = 'Planner: Adjust variables used within the planner such as date and time settings.'
PLANNER_ICON = ICONS + r'\plannerSettings.png'

SCHEDULE_DESCRIPTION = 'Schedule: Organize events and develop schedules.'
SCHEDULE_ICON = ICONS + r'\schedule.png'

TASKS_DESCRIPTION = 'Tasks: Delegate and monitor tasks.'
TASKS_ICON = ICONS + r'\tasks.png'

RESOURCES_DESCRIPTION = 'Resources: Manage your inventories.'
RESOURCES_ICON = ICONS + r'\resources.png'

FINANCE_DESCRIPTION = 'Finance: Prepare budgets and keep track of transactions.'
FINANCE_ICON = ICONS + r'\finance.png'

BACKGROUND_IMAGE = ICONS + r'\background.png'

DOWN_ARROW_ICON = ICONS + r'downArrow.png'

BACKGROUND_COLOR_DARK = r'rgb(25,25,25)'
BACKGROUND_COLOR_LIGHT = r'rgb(49,49,49)'
BORDER_COLOR = r'rgb(70,70,70)'
FONT_COLOR = r'rgb(255,255,255)'

BUTTON_COLOR = r'rgb(55,55,55)'

BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'

LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(32,33,30), stop: 0.5 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(22,23,20), stop: 0.5 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'
WIDGET_STYLE = """
            QWidget
            {
                color: """ + FONT_COLOR + """;
                background-color: """ + BACKGROUND_COLOR_DARK + """;
                selection-background-color:""" + BACKGROUND_COLOR_LIGHT + """;
                selection-color: """ + FONT_COLOR + """;
                background-clip: border;
                border-image: none;
                outline: 0;
            }
            """
ENTRY_STYLE = """
            QWidget
            {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
                border: 2px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            }
            """
WINDOW_STYLE = """
            QMainWindow {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);

            }
            QMainWindow::separator {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + BACKGROUND_COLOR_LIGHT + """;
                width: 10px; /* when vertical */
                height: 10px; /* when horizontal */
            }

            QMainWindow::separator:hover {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: black;
            }

            """

CHECK_STYLE = """
            QCheckBox {
                spacing: 5px;
            }

            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }

            QCheckBox::indicator:unchecked {
            }

            QCheckBox::indicator:unchecked:hover {
            }

            QCheckBox::indicator:unchecked:pressed {
            }

            QCheckBox::indicator:checked {
            }

            QCheckBox::indicator:checked:hover {
            }

            QCheckBox::indicator:checked:pressed {
            }

            QCheckBox::indicator:indeterminate:hover {
            }

            QCheckBox::indicator:indeterminate:pressed {
            }
            """

COMBO_STYLE = """
            QComboBox {
                background: """ + LIGHT_GRADIENT + """;
                color: rgb(255,255,255);
                border: 3px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                padding: 1px 1px 1px 3px;
                min-width: 6px;
            }

            QComboBox:!editable::hover {
                background: """ + DARK_GRADIENT + """; 
            }

            QComboBox:editable {
                background: """ + DARK_GRADIENT + """;         
            }

            /* QComboBox gets the "on" state when the popup is open */
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: """ + DARK_GRADIENT + """;
            }

            QComboBox:on { /* shift the text when the popup opens */
                padding-top: 3px;
                padding-left: 4px;
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 10px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: """ + BORDER_STYLE + """;
                border-top-right-radius: """ + BORDER_RADIUS + """;
                border-bottom-right-radius: """ + BORDER_RADIUS + """;
            }

            QComboBox::down-arrow {

            }

            QComboBox::down-arrow:on { /* shift the arrow when popup is open */
                top: 1px;
                left: 1px;
            }
            """

FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """
LABEL_STYLE = """
            QLabel {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;
                padding: 2px;
            }
            """

MENUBAR_STYLE = """
            QMenuBar {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid rgb(0,0,0);
            }

            QMenuBar::item {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
            }

            QMenuBar::item::selected {
                background: """ + BACKGROUND_COLOR_DARK + """;
            }

            QMenu {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid #000;           
            }

            QMenu::item::selected {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
            }
            """

SCROLL_STYLE = """
            QScrollBar:vertical {
                 background: """ + BACKGROUND_COLOR_DARK + """;
                 width: 15px;
                 margin: 22px 0 22px 0;
             }
             QScrollBar::handle:vertical {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::handle:vertical:pressed {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::add-line:vertical {
                 background: """ + LIGHT_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: bottom;
                 subcontrol-origin: margin;
             }

             QScrollBar::sub-line:vertical {
                 background: """ + DARK_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: top;
                 subcontrol-origin: margin;
             }
             QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                 border: 2px solid """ + DARK_GRADIENT + """;
                 width: 3px;
                 height: 3px;
                 background: black;
             }
            QScrollBar::up-arrow:vertical:pressed, QScrollBar::down-arrow:vertical:pressed {
                 border: 2px solid """ + LIGHT_GRADIENT + """;
                 width: 3px;
                 height: 3px;
                 background: black;
             }

             QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                 background: none;
             }
             """
SPLITTER_STYLE = """
            QSplitter {
                background: """ + BACKGROUND_COLOR_DARK + """;
            }
            QSplitter::handle {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
            }

            QSplitter::handle:hover {
                background: white;
            }


            QSplitter::handle:horizontal {
                width: 8px;
            }

            QSplitter::handle:vertical {
                height: 8px;
            }

            QSplitter::handle:pressed {
                background-color: """ + DARK_GRADIENT + """;
            }
            """

TOOLBAR_STYLE = """ 
            QToolBar, QToolButton, QToolTip { 
                background: rgb(56,60,55);
                background: """ + LIGHT_GRADIENT + """;

                color: """ + FONT_COLOR + """;
                spacing: 3px; /* spacing between items in the tool bar */
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;

            } 
            QToolBar {a
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }


            QToolButton:hover {
                background: """ + DARK_GRADIENT + """;
                border: 0px;
            }

            QToolBar::handle {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                border: 1px solid rgb(100,100,100);
            } 
            """

# TODO and use os module
# The paths generator1 must read the entire database, collecting the
# necessary data to fill the lists in the list generator1 as well as information in the paths above.
# This process will occur each time a different database is used and a current working database
# file will be created to save the configurations yielded by the aforementioned procedure so that
# it doesn't need to be performed each time the program starts

# SYSTEM
SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
HAND = 'hand2'
PTR = 'left_ptr'
# NUMERICAL
CURRENT_YEAR = datetime.datetime.now().year - 1
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
INTERVALS = {'from to': '--', 'between': '-'}
NA_VALUES = ['N']
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt, '==': operator.eq}
INVERTED_OPERATORS = {operator.ge: '>=', operator.le: '<=', operator.gt: '>', operator.lt: '<', operator.eq: '=='}
OPERATORS_WORDS = {
    'greaterthanorequalto': '>=', 'lessthanorequalto': '<=', 'greaterthan': '>', 'lessthan': '<',
    'equal'               : '=='
}

UNITS = ('km', 'm', 'mi', 'ft')

CLASSIFIERS_NAMES = (
    'Adaptive Boost', 'Decision Tree', 'Elastic Net', 'Gaussian Process', 'Nearest Neighbors', 'Multilayer Perceptron',
    'Quadratic Discriminant', 'Random Forest', 'Stochastic Gradient Descent', 'Support Vector Machine')
CLUSTERERS_NAMES = ('',)
DIMENSIONALITY_REDUCERS_NAMES = ('',)
REGRESSORS_NAMES = (
    'Adaptive Boost', 'Decision Tree', 'Elastic Net', 'Gaussian Process', 'Nearest Neighbors', 'Kernel Ridge',
    'Multilayer Perceptron', 'Random Forest', 'Stochastic Gradient Descent', 'Support Vector Machine')
import ctypes
import datetime
import operator

from Utilities8.MenuData import *
from Utilities8.Variables import *

# SYSTEM
SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
HAND = 'hand2'
PTR = 'left_ptr'

# EXTENSIONS
KML_EXT = '.kml'
QRY_EXT = '.qry'
TXT_EXT = '.txt'
XLSX_EXT = '.xlsx'

# EXTENSION TUPLES
ALL_FILES = ('All files', '*.*')
KML_FILES = ('KML files', '*.kml')
QRY_FILES = ('Query files', '*.qry')

# NUMERICAL
CURRENT_YEAR = datetime.datetime.now().year - 1
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
INDICATORS = ('--', '-')
NA_VALUES = ['N']
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
UNITS = ('km', 'm', 'mi', 'ft')

# GRAPHS
GRAPH_COLORS = (
    ('#3CE1E0', '#56A71E', '#7CD530', '#B6ED15', '#E6F70E', '#F5AB23', '#F06D12', '#B43519', '#891A1A', '#000000'), (
        'maroon', 'darkred', 'brown', 'firebrick', 'crimson', 'indianred', 'lightcoral', 'salmon', 'rosybrown',
        'darksalmon', 'lightsalmon', 'tomato', 'orangered', 'darkorange', 'orange'), (
        'darkcyan', 'teal', 'steelblue', 'cadetblue', 'lightslategrey', 'skyblue', 'turquoise', 'lightseagreen',
        'darkturquoise'))
GRAPH_FILL_STYLES = ('none',)  # top, bottom, right, left, full: markerfacecoloralt='gray'
GRAPH_MARKERS = ('o',)

# DESCRIPTIONS
OPEN_FOLDER_DESCRIPTION = 'Open a folder from which to extract files.'
OPEN_FOLDER_ICON = ICONS + r'\openFolder.png'

SAVE_FOLDER_DESCRIPTION = 'Open a folder to save files in.'
SAVE_FOLDER_ICON = ICONS + r'\saveFolder.png'

OPEN_FILE_DESCRIPTION = 'Open file.'
OPEN_FILE_ICON = ICONS + r'\openFile.png'

SAVE_FILE_DESCRIPTION = 'Save file.'
SAVE_FILE_ICON = ICONS + r'\saveFile.png'

EXIT_DESCRIPTION = 'Exit application.'
EXIT_ICON = ICONS + r'\exit.png'

# Data encapsulates interface7, geosearch, machine learning, and sentiment analysis
DATABASE_DESCRIPTION = 'Database: Set your database and its variables.'
DATABASE_ICON = ICONS + r'\database.png'

SEARCH_DESCRIPTION = 'search Engine: Navigate your database and specify criteria for extracting data.'
SEARCH_ICON = ICONS + r'\search.png'

GEO_SEARCH_DESCRIPTION = 'Geographic search: Explore your database and produce geographical information system visualizations.'
GEO_SEARCH_ICON = ICONS + r'\geoSearch.png'

MACHINE_LEARNING_DESCRIPTION = 'Machine Learning: Perform statistical analysis on your data and visualize your results.'
MACHINE_LEARNING_ICON = ICONS + r'\machineLearning.png'

SENTIMENT_DESCRIPTION = 'Sentiment Analysis: Discover popular opinion.'
SENTIMENT_ICON = ICONS + r'\sentiment.png'

# Assist encapsulates updates and help
UPDATES_DESCRIPTION = 'Updates: Check for updates.'
UPDATES_ICON = ICONS + r'\updates.png'

HELP_DESCRIPTION = 'Help2: Browse 1 for software information and examples.'
HELP_ICON = ICONS + r'\help.png'

# Planner encapsulates schedule and tasks
PLANNER_DESCRIPTION = 'Planner: Adjust variables used within the planner such as date and time settings.'
PLANNER_ICON = ICONS + r'\plannerSettings.png'

SCHEDULE_DESCRIPTION = 'Schedule: Organize events and develop schedules.'
SCHEDULE_ICON = ICONS + r'\schedule.png'

TASKS_DESCRIPTION = 'Tasks: Delegate and monitor tasks.'
TASKS_ICON = ICONS + r'\tasks.png'

RESOURCES_DESCRIPTION = 'Resources: Manage your inventories.'
RESOURCES_ICON = ICONS + r'\resources.png'

BACKGROUND_IMAGE = ICONS + r'\background.png'

DOWN_ARROW_ICON = ICONS + r'downArrow.png'

BACKGROUND_COLOR_DARK = r'rgb(20,20,20)'
BACKGROUND_COLOR_LIGHT = r'rgb(49,49,49)'
BORDER_COLOR = r'rgb(100,100,100)'
WINDOW_STYLE = """
            color: rgb(255,255,255);
            QMainWindow {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(/interface7/InterfaceUtilities0/icon2/background.png);

            }
            QMainWindow::separator {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                width: 10px; /* when vertical */
                height: 10px; /* when horizontal */
            }

            QMainWindow::separator:hover {
                background: BACKGROUND_COLOR_LIGHT;
                color: rgb(0,0,0);
            }

            """
COMBO_STYLE = """QComboBox {
                border: 1px solid """ + BORDER_COLOR + """;
                background: """ + BACKGROUND_COLOR_DARK + """;
                border-radius: 3px;
                padding: 1px 18px 1px 3px;
                min-width: 6px;
                }

                QComboBox:editable {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                }

                QComboBox:!editable, QComboBox::drop-down:editable {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                         stop: 0 """ + BACKGROUND_COLOR_LIGHT + """, stop: 0.3 """ + BACKGROUND_COLOR_LIGHT + """,
                                         stop: 0.5 """ + BACKGROUND_COLOR_LIGHT + """, stop: 1.0 """ + BACKGROUND_COLOR_LIGHT + """);
                }

                /* QComboBox gets the "on" state when the popup is open */
                QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                        stop: 0 """ + BACKGROUND_COLOR_DARK + """, stop: 0.3 """ + BACKGROUND_COLOR_DARK + """,
                                        stop: 0.5 """ + BACKGROUND_COLOR_DARK + """, stop: 1.0 """ + BACKGROUND_COLOR_DARK + """);
                }

                QComboBox:on { /* shift the text when the popup opens */
                padding-top: 3px;
                padding-left: 4px;
                }

                QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 10px;

                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: solid; /* just a single line */
                border-top-right-radius: 3px; /* same radius as the QComboBox */
                border-bottom-right-radius: 3px;
                }

                QComboBox::down-arrow {
                image: url(downArrow.png);
                }

                QComboBox::down-arrow:on { /* shift the arrow when popup is open */
                top: 1px;
                left: 1px;
                }"""

FRAME_STYLE = """
            QFrame, QLabel, QToolTip {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                border: 1px solid """ + BORDER_COLOR + """;
                border-radius: 4px;
                padding: 2px;
            }"""
MENUBAR_STYLE = """
            QMenuBar {
                background-color: rgb(49,49,49);
                color: rgb(255,255,255);
                border: 1px solid #000;
            }

            QMenuBar::item {
                background-color: rgb(49,49,49);
                color: rgb(255,255,255);
            }

            QMenuBar::item::selected {
                background-color: rgb(30,30,30);
            }

            QMenu {
                background-color: rgb(49,49,49);
                color: rgb(255,255,255);
                border: 1px solid #000;           
            }

            QMenu::item::selected {
                background-color: rgb(30,30,30);
            }
            """
SPLITTER_STYLE = """
                QSplitter::handle {
                    background-color: """ + BACKGROUND_COLOR_LIGHT + """;
                }

                QSplitter::handle:horizontal {
                    width: 4px;
                }

                QSplitter::handle:vertical {
                    height: 4px;
                }

                QSplitter::handle:pressed {
                    background-color: """ + BACKGROUND_COLOR_DARK + """;
                }
                """
STATUSBAR_STYLE = """
                QStatusBar {
                    background: rgb(49,49,49);
                    border: 1px solid rgb(100,100,100);
                }

                QStatusBar::item {
                    border: 3px solid rgb(50,75,60);
                    border-radius: 3px;

                }"""
TOOLBAR_STYLE = """ 
                QToolBar { 
                    background: rgb(55,60,55);
                    spacing: 3px; /* spacing between items in the tool bar */
                    border: 1px solid """ + BORDER_COLOR + """;

                } 
                QToolBar::handle {
                    image: url(handle.png);
                    background: rgb(45,50,48);
                    border: 1px solid rgb(100,100,100);
                } 
                """

# TODO and use os module
# The paths generator1 must read the entire database, collecting the
# necessary data to fill the lists in the list generator1 as well as information in the paths above.
# This process will occur each time a different database is used and a current working database
# file will be created to save the configurations yielded by the aforementioned procedure so that
# it doesn't need to be performed each time the program starts
import datetime
import operator

from Utilities.Preferences import MasterPath

MasterPath_Object = MasterPath.MasterPath()

# DATABASE
FOLDER = 'YEAR'
FILE = 'STATE_CODE_001'
ID = 'STRUCTURE_NUMBER_008'
LATITUDE = 'LAT_016'
LONGITUDE = 'LONG_017'

# NUMERICAL
UNITS = ('--', 'km', 'm', 'mi', 'ft')
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
INDICATORS = ('--', '-')
NA_VALUES = ['N']
CURRENT_YEAR = datetime.datetime.now().year - 1

# PATHS
MASTER = MasterPath_Object.getMasterPath()
ROOT = r'\BigData'
DATABASE = MASTER + ROOT + '\Database\Years\\'
VALUES = ROOT + r'\Interface\InterfaceUtilities\Labels\parameterValues.txt'
NAMES = ROOT + r'\Interface\InterfaceUtilities\Labels\ItemNames.txt'
STATES = ROOT + r'\Database\Information\allStates.txt'
STATE_CODES = ROOT + r'\Database\Information\allStateCodes.txt'
NUMBERS = ROOT + r'\Interface\InterfaceUtilities\Labels\parameterNumbers.txt'
ITEM_NAMES = ROOT + r'\Interface\InterfaceUtilities\Labels\ItemNames.txt'
temp_RSP = "\BigData\\Utilities14\Preferences\Temporary_Report_Preferences.txt"
RSP = "\BigData\\Utilities14\Preferences\Report_Preferences.txt"
MCSP = "\BigData\\Utilities14\Preferences\Markov_Chain_Preferences.txt"

# EXTENSION TUPLES
ALL_EXT = ("All files", "*.*")
QRY_EXT = ("Query files", "*.qry")

# GRAPHS
GRAPH_MARKERS = ('o',)
GRAPH_FILL_STYLES = ('none',)  # top, bottom, right, left, full: markerfacecoloralt='gray'
GRAPH_COLORS = (
    ('#3CE1E0', '#56A71E', '#7CD530', '#B6ED15', '#E6F70E', '#F5AB23', '#F06D12', '#B43519', '#891A1A', '#000000'), (
        'maroon', 'darkred', 'brown', 'firebrick', 'crimson', 'indianred', 'lightcoral', 'salmon', 'rosybrown',
        'darksalmon', 'lightsalmon', 'tomato', 'orangered', 'darkorange', 'orange'), (
        'darkcyan', 'teal', 'steelblue', 'cadetblue', 'lightslategrey', 'skyblue', 'turquoise', 'lightseagreen',
        'darkturquoise'))

# TODO and use os module
# The paths generator1 must read the entire database, collecting the
# necessary data to fill the lists in the list generator1 as well as information in the paths above.
# This process will occur each time a different database is used and a current working database
# file will be created to save the configurations yielded by the aforementioned procedure so that
# it doesn't need to be performed each time the program starts

if __name__ == "Utilities14.Constants":
    from Utilities29.Dependencies import *
    from Utilities29.Descriptions import *
    from Utilities29.Extensions import *
    from Utilities29.Icons import *
    from Utilities29.Paths import *
    from Utilities29.Variables import *


    def get_separator(header):
        counters = []
        max = 0
        s = ','

        for separator in SEPARATORS:
            counter = 0
            for character in header:
                if separator == character:
                    counter += 1
            counters.append(counter)
            if counter > max:
                max = counter
                s = separator

        return s


    def collect_headers(dir):
        """
        Create a file containing all unique
        column names in the entire database
        """
        headers = []
        separator = None
        for subdir, dirs, files in os.walk(dir):
            for file in files:
                if os.path.splitext(file)[1] != '.h5' and os.path.splitext(file)[0] != 'temp':
                    try:
                        with open(os.path.join(subdir, file), newline='') as f:
                            header = next(f)
                            if not separator:
                                separator = get_separator(header)

                        with open(os.path.join(subdir, file), newline='') as f:
                            header = next(csv.reader(f, delimiter=separator))

                        for column in header:
                            if column not in headers:
                                headers.append(column)
                    except:
                        pass

        with open(ITEM_NAMES, "w") as file:
            for header in headers:
                file.write(header + '\n')

        return headers, separator


    def optimize_dataset(path):
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1] != HDF_EXT and os.path.splitext(file)[0] != 'temp' and \
                        os.path.splitext(file)[0] + HDF_EXT not in files:
                    df = pd.read_csv(os.path.join(subdir, file), dtype='str', sep=SEPARATOR, keep_default_na=False)
                    df.to_hdf(os.path.join(subdir, os.path.splitext(file)[0] + HDF_EXT), os.path.splitext(file)[0])


    # SYSTEM
    SMALL_FONT = ("Verdana", 8)
    MEDIUM_FONT = ("Verdana", 10)
    LARGE_FONT = ("Verdana", 12)
    user32 = ctypes.windll.user32
    WIDTH = user32.GetSystemMetrics(0)
    HEIGHT = user32.GetSystemMetrics(1)
    HAND = 'hand2'
    PTR = 'left_ptr'
    SEPARATORS = (',', '\t', '|', '/', '\\', '.', ';', ':', '-', ' ')
    HEADERS, SEPARATOR = collect_headers(
            DATABASE
    )  # TODO SEPARATOR MUST UPDATE A DROP DOWN LIST VALUE IN THE DATABASE MENU
    # optimize_dataset(DATABASE)
    # NUMERICAL
    CURRENT_YEAR = datetime.datetime.now().year - 1
    EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
    INDICATORS = ('--', '-')
    NA_VALUES = ['N']
    OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
    UNITS = ('km', 'm', 'mi', 'ft')

from Utilities.Variables import *

# SYSTEM
SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
HAND = 'hand2'
PTR = 'left_ptr'

# EXTENSIONS
KML_EXT = '.kml'
QRY_EXT = '.qry'
TXT_EXT = '.txt'
XLSX_EXT = '.xlsx'

# EXTENSION TUPLES
ALL_FILES = 'All files (*.*)'
KML_FILES = 'kml files (*.kml)'
QRY_FILES = 'Query files (*.qry)'
TXT_FILES = 'Text files (*.txt)'
XLSX_FILES = 'xlsx files (*.xlsx)'

# NUMERICAL
CURRENT_YEAR = datetime.datetime.now().year - 1
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
INDICATORS = ('--', '-')
NA_VALUES = ['N']
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
UNITS = ('km', 'm', 'mi', 'ft')

# GRAPHS
GRAPH_COLORS = (
    ('#3CE1E0', '#56A71E', '#7CD530', '#B6ED15', '#E6F70E', '#F5AB23', '#F06D12', '#B43519', '#891A1A', '#000000'), (
        'maroon', 'darkred', 'brown', 'firebrick', 'crimson', 'indianred', 'lightcoral', 'salmon', 'rosybrown',
        'darksalmon', 'lightsalmon', 'tomato', 'orangered', 'darkorange', 'orange'), (
        'darkcyan', 'teal', 'steelblue', 'cadetblue', 'lightslategrey', 'skyblue', 'turquoise', 'lightseagreen',
        'darkturquoise'))
GRAPH_FILL_STYLES = ('none',)  # top, bottom, right, left, full: markerfacecoloralt='gray'
GRAPH_MARKERS = ('o',)

# DESCRIPTIONS
ICONS = ROOT + r'\Interface\InterfaceUtilities\Icons'

WINDOW_ICON = ICONS + r'\window.png'

OPEN_FOLDER_DESCRIPTION = 'Open a folder from which to extract files.'
OPEN_FOLDER_ICON = ICONS + r'\openFolder.png'

SAVE_FOLDER_DESCRIPTION = 'Open a folder to save files in.'
SAVE_FOLDER_ICON = ICONS + r'\saveFolder.png'

OPEN_FILE_DESCRIPTION = 'Open file.'
OPEN_FILE_ICON = ICONS + r'\openFile.png'

SAVE_FILE_DESCRIPTION = 'Save file.'
SAVE_FILE_ICON = ICONS + r'\saveFile.png'

EXIT_DESCRIPTION = 'Exit application.'
EXIT_ICON = ICONS + r'\exit.png'

# Data encapsulates interface7, geosearch, machine learning, and sentiment analysis
DATABASE_DESCRIPTION = 'Database: Set your database and its variables.'
DATABASE_ICON = ICONS + r'\database.png'

SEARCH_DESCRIPTION = 'search Engine: Navigate your database and specify criteria for extracting data.'
SEARCH_ICON = ICONS + r'\search.png'

GEO_SEARCH_DESCRIPTION = 'Geographic search: Explore your database and produce geographical information system visualizations.'
GEO_SEARCH_ICON = ICONS + r'\geoSearch.png'

MACHINE_LEARNING_DESCRIPTION = 'Machine Learning: Perform statistical analysis on your data and visualize your results.'
MACHINE_LEARNING_ICON = ICONS + r'\machineLearning.png'

SENTIMENT_DESCRIPTION = 'Sentiment Analysis: Discover popular opinion.'
SENTIMENT_ICON = ICONS + r'\sentiment.png'

# Assist encapsulates updates and help
UPDATES_DESCRIPTION = 'Updates: Check for updates.'
UPDATES_ICON = ICONS + r'\updates.png'

HELP_DESCRIPTION = 'Help2: Browse 1 for software information and examples.'
HELP_ICON = ICONS + r'\help.png'

# Planner encapsulates schedule and tasks
PLANNER_DESCRIPTION = 'Planner: Adjust variables used within the planner such as date and time settings.'
PLANNER_ICON = ICONS + r'\plannerSettings.png'

SCHEDULE_DESCRIPTION = 'Schedule: Organize events and develop schedules.'
SCHEDULE_ICON = ICONS + r'\schedule.png'

TASKS_DESCRIPTION = 'Tasks: Delegate and monitor tasks.'
TASKS_ICON = ICONS + r'\tasks.png'

RESOURCES_DESCRIPTION = 'Resources: Manage your inventories.'
RESOURCES_ICON = ICONS + r'\resources.png'

FINANCE_DESCRIPTION = 'Finance: Prepare budgets and keep track of transactions.'
FINANCE_ICON = ICONS + r'\finance.png'

BACKGROUND_IMAGE = ICONS + r'\background.png'

DOWN_ARROW_ICON = ICONS + r'downArrow.png'

BACKGROUND_COLOR_DARK = r'rgb(25,25,25)'
BACKGROUND_COLOR_LIGHT = r'rgb(49,49,49)'
BORDER_COLOR = r'rgb(70,70,70)'
FONT_COLOR = r'rgb(255,255,255)'

BUTTON_COLOR = r'rgb(55,55,55)'

BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'

LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(32,33,30), stop: 0.5 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(22,23,20), stop: 0.5 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'
WIDGET_STYLE = """
            QWidget
            {
                color: """ + FONT_COLOR + """;
                background-color: """ + BACKGROUND_COLOR_DARK + """;
                selection-background-color:""" + BACKGROUND_COLOR_LIGHT + """;
                selection-color: """ + FONT_COLOR + """;
                background-clip: border;
                border-image: none;
                outline: 0;
            }
            """
ENTRY_STYLE = """
            QWidget
            {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
                border: 2px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
            }
            """
WINDOW_STYLE = """
            QMainWindow {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);

            }
            QMainWindow::separator {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + BACKGROUND_COLOR_LIGHT + """;
                width: 10px; /* when vertical */
                height: 10px; /* when horizontal */
            }

            QMainWindow::separator:hover {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: black;
            }

            """

CHECK_STYLE = """
            QCheckBox {
                spacing: 5px;
            }

            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }

            QCheckBox::indicator:unchecked {
            }

            QCheckBox::indicator:unchecked:hover {
            }

            QCheckBox::indicator:unchecked:pressed {
            }

            QCheckBox::indicator:checked {
            }

            QCheckBox::indicator:checked:hover {
            }

            QCheckBox::indicator:checked:pressed {
            }

            QCheckBox::indicator:indeterminate:hover {
            }

            QCheckBox::indicator:indeterminate:pressed {
            }
            """

COMBO_STYLE = """
            QComboBox {
                background: """ + LIGHT_GRADIENT + """;
                color: rgb(255,255,255);
                border: 3px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                padding: 1px 1px 1px 3px;
                min-width: 6px;
            }

            QComboBox:!editable::hover {
                background: """ + DARK_GRADIENT + """; 
            }

            QComboBox:editable {
                background: """ + DARK_GRADIENT + """;         
            }

            /* QComboBox gets the "on" state when the popup is open */
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: """ + DARK_GRADIENT + """;
            }

            QComboBox:on { /* shift the text when the popup opens */
                padding-top: 3px;
                padding-left: 4px;
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 10px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: """ + BORDER_STYLE + """;
                border-top-right-radius: """ + BORDER_RADIUS + """;
                border-bottom-right-radius: """ + BORDER_RADIUS + """;
            }

            QComboBox::down-arrow {

            }

            QComboBox::down-arrow:on { /* shift the arrow when popup is open */
                top: 1px;
                left: 1px;
            }
            """

FRAME_STYLE = """
            QFrame {
                background: """ + BACKGROUND_COLOR_DARK + """;
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }
            """
LABEL_STYLE = """
            QLabel {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;
                padding: 2px;
            }
            """

MENUBAR_STYLE = """
            QMenuBar {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid rgb(0,0,0);
            }

            QMenuBar::item {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
            }

            QMenuBar::item::selected {
                background: """ + BACKGROUND_COLOR_DARK + """;
            }

            QMenu {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid #000;           
            }

            QMenu::item::selected {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
            }
            """

SCROLL_STYLE = """
            QScrollBar:vertical {
                 background: """ + BACKGROUND_COLOR_DARK + """;
                 width: 15px;
                 margin: 22px 0 22px 0;
             }
             QScrollBar::handle:vertical {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::handle:vertical:pressed {
                 background: """ + DARK_GRADIENT + """;
                 min-height: 20px;
             }
             QScrollBar::add-line:vertical {
                 background: """ + LIGHT_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: bottom;
                 subcontrol-origin: margin;
             }

             QScrollBar::sub-line:vertical {
                 background: """ + DARK_GRADIENT + """;
                 height: 20px;
                 subcontrol-position: top;
                 subcontrol-origin: margin;
             }
             QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                 border: 2px solid """ + DARK_GRADIENT + """;
                 width: 3px;
                 height: 3px;
                 background: black;
             }
            QScrollBar::up-arrow:vertical:pressed, QScrollBar::down-arrow:vertical:pressed {
                 border: 2px solid """ + LIGHT_GRADIENT + """;
                 width: 3px;
                 height: 3px;
                 background: black;
             }

             QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                 background: none;
             }
             """
SPLITTER_STYLE = """
            QSplitter {
                background: """ + BACKGROUND_COLOR_DARK + """;
            }
            QSplitter::handle {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
            }

            QSplitter::handle:hover {
                background: white;
            }


            QSplitter::handle:horizontal {
                width: 8px;
            }

            QSplitter::handle:vertical {
                height: 8px;
            }

            QSplitter::handle:pressed {
                background-color: """ + DARK_GRADIENT + """;
            }
            """

TOOLBAR_STYLE = """ 
            QToolBar, QToolButton, QToolTip { 
                background: rgb(56,60,55);
                background: """ + LIGHT_GRADIENT + """;

                color: """ + FONT_COLOR + """;
                spacing: 3px; /* spacing between items in the tool bar */
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;

            } 
            QToolBar {
                background-image: url(./interface7/InterfaceUtilities0/icon2/background.png);
            }


            QToolButton:hover {
                background: """ + DARK_GRADIENT + """;
                border: 0px;
            }

            QToolBar::handle {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                border: 1px solid rgb(100,100,100);
            } 
            """

# TODO and use os module
# The paths generator1 must read the entire database, collecting the
# necessary data to fill the lists in the list generator1 as well as information in the paths above.
# This process will occur each time a different database is used and a current working database
# file will be created to save the configurations yielded by the aforementioned procedure so that
# it doesn't need to be performed each time the program starts
from Utilities32.Variables import *

# SYSTEM
SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
HAND = 'hand2'
PTR = 'left_ptr'

# EXTENSIONS
KML_EXT = '.kml'
QRY_EXT = '.qry'
TXT_EXT = '.txt'
XLSX_EXT = '.xlsx'

# EXTENSION TUPLES
ALL_FILES = 'All files (*.*)'
KML_FILES = 'kml files (*.kml)'
QRY_FILES = 'Query files (*.qry)'
TXT_FILES = 'Text files (*.txt)'
XLSX_FILES = 'xlsx files (*.xlsx)'

# NUMERICAL
CURRENT_YEAR = datetime.datetime.now().year - 1
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
INDICATORS = ('--', '-')
NA_VALUES = ['N']
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
UNITS = ('km', 'm', 'mi', 'ft')

# GRAPHS
GRAPH_COLORS = (
    ('#3CE1E0', '#56A71E', '#7CD530', '#B6ED15', '#E6F70E', '#F5AB23', '#F06D12', '#B43519', '#891A1A', '#000000'), (
        'maroon', 'darkred', 'brown', 'firebrick', 'crimson', 'indianred', 'lightcoral', 'salmon', 'rosybrown',
        'darksalmon', 'lightsalmon', 'tomato', 'orangered', 'darkorange', 'orange'), (
        'darkcyan', 'teal', 'steelblue', 'cadetblue', 'lightslategrey', 'skyblue', 'turquoise', 'lightseagreen',
        'darkturquoise'))
GRAPH_FILL_STYLES = ('none',)  # top, bottom, right, left, full: markerfacecoloralt='gray'
GRAPH_MARKERS = ('o',)

# DESCRIPTIONS
OPEN_FOLDER_DESCRIPTION = 'Open a folder from which to extract files.'
OPEN_FOLDER_ICON = ICONS + r'\openFolder.png'

SAVE_FOLDER_DESCRIPTION = 'Open a folder to save files in.'
SAVE_FOLDER_ICON = ICONS + r'\saveFolder.png'

OPEN_FILE_DESCRIPTION = 'Open file.'
OPEN_FILE_ICON = ICONS + r'\openFile.png'

SAVE_FILE_DESCRIPTION = 'Save file.'
SAVE_FILE_ICON = ICONS + r'\saveFile.png'

EXIT_DESCRIPTION = 'Exit application.'
EXIT_ICON = ICONS + r'\exit.png'

# Data encapsulates interface7, geosearch, machine learning, and sentiment analysis
DATABASE_DESCRIPTION = 'Database: Set your database and its variables.'
DATABASE_ICON = ICONS + r'\database.png'

SEARCH_DESCRIPTION = 'search Engine: Navigate your database and specify criteria for extracting data.'
SEARCH_ICON = ICONS + r'\search.png'

GEO_SEARCH_DESCRIPTION = 'Geographic search: Explore your database and produce geographical information system visualizations.'
GEO_SEARCH_ICON = ICONS + r'\geoSearch.png'

MACHINE_LEARNING_DESCRIPTION = 'Machine Learning: Perform statistical analysis on your data and visualize your results.'
MACHINE_LEARNING_ICON = ICONS + r'\machineLearning.png'

SENTIMENT_DESCRIPTION = 'Sentiment Analysis: Discover popular opinion.'
SENTIMENT_ICON = ICONS + r'\sentiment.png'

# Assist encapsulates updates and help
UPDATES_DESCRIPTION = 'Updates: Check for updates.'
UPDATES_ICON = ICONS + r'\updates.png'

HELP_DESCRIPTION = 'Help2: Browse 1 for software information and examples.'
HELP_ICON = ICONS + r'\help.png'

# Planner encapsulates schedule and tasks
PLANNER_DESCRIPTION = 'Planner: Adjust variables used within the planner such as date and time settings.'
PLANNER_ICON = ICONS + r'\plannerSettings.png'

SCHEDULE_DESCRIPTION = 'Schedule: Organize events and develop schedules.'
SCHEDULE_ICON = ICONS + r'\schedule.png'

TASKS_DESCRIPTION = 'Tasks: Delegate and monitor tasks.'
TASKS_ICON = ICONS + r'\tasks.png'

RESOURCES_DESCRIPTION = 'Resources: Manage your inventories.'
RESOURCES_ICON = ICONS + r'\resources.png'

BACKGROUND_IMAGE = ICONS + r'\background.png'

DOWN_ARROW_ICON = ICONS + r'downArrow.png'

BACKGROUND_COLOR_DARK = r'rgb(30,30,30)'
BACKGROUND_COLOR_LIGHT = r'rgb(49,49,49)'
BORDER_COLOR = r'rgb(70,70,70)'
FONT_COLOR = r'rgb(255,255,255)'

BUTTON_COLOR = r'rgb(55,55,55)'

BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'

LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(32,33,30), stop: 0.5 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(22,23,20), stop: 0.5 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'

WINDOW_STYLE = """
            QMainWindow {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;

            }
            QMainWindow::separator {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: rgb(255,255,255);
                width: 10px; /* when vertical */
                height: 10px; /* when horizontal */
            }

            QMainWindow::separator:hover {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: rgb(255,255,255);
            }

            """

COMBO_STYLE = """
            QComboBox {
                background: """ + LIGHT_GRADIENT + """;
                color: rgb(255,255,255);
                border: 3px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                padding: 1px 1px 1px 3px;
                min-width: 6px;
            }

            QComboBox:!editable::hover {
                background: """ + DARK_GRADIENT + """; 
            }

            QComboBox:editable {
                background: """ + DARK_GRADIENT + """;         
            }

            /* QComboBox gets the "on" state when the popup is open */
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: """ + DARK_GRADIENT + """;
            }

            QComboBox:on { /* shift the text when the popup opens */
                padding-top: 3px;
                padding-left: 4px;
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 10px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: """ + BORDER_STYLE + """;
                border-top-right-radius: """ + BORDER_RADIUS + """;
                border-bottom-right-radius: """ + BORDER_RADIUS + """;
            }

            QComboBox::down-arrow {
            }

            QComboBox::down-arrow:on { /* shift the arrow when popup is open */
                top: 1px;
                left: 1px;
            }
            """

FRAME_STYLE = """
            QFrame, QLabel, QToolTip {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;
                padding: 2px;
            }"""

MENUBAR_STYLE = """
            QMenuBar {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid rgb(0,0,0);
            }

            QMenuBar::item {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
            }

            QMenuBar::item::selected {
                background: """ + BACKGROUND_COLOR_DARK + """;
            }

            QMenu {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid #000;           
            }

            QMenu::item::selected {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
            }
            """
SPLITTER_STYLE = """
            QSplitter::handle {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
            }

            QSplitter::handle:horizontal {
                width: 4px;
            }

            QSplitter::handle:vertical {
                height: 4px;
            }

            QSplitter::handle:pressed {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
            }
            """

STATUSBAR_STYLE = """
            QStatusBar {
                background: rgb(49,49,49);
                color: """ + FONT_COLOR + """;
                border: 1px solid rgb(100,100,100);
            }

            QStatusBar::item {
                border: """ + BORDER_WIDTH + """ """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;

            }"""

TOOLBAR_STYLE = """ 
            QToolBar, QToolButton, QToolTip { 
                background: rgb(56,60,55);
                background: """ + LIGHT_GRADIENT + """;

                color: """ + FONT_COLOR + """;
                spacing: 3px; /* spacing between items in the tool bar */
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;

            } 

            QToolButton:hover {
                background: """ + DARK_GRADIENT + """;
                border: 0px;
            }

            QToolBar::handle {
                background: rgb(45,50,48);
                border: 1px solid rgb(100,100,100);
            } 
            """

# TODO and use os module
# The paths generator1 must read the entire database, collecting the
# necessary data to fill the lists in the list generator1 as well as information in the paths above.
# This process will occur each time a different database is used and a current working database
# file will be created to save the configurations yielded by the aforementioned procedure so that
# it doesn't need to be performed each time the program starts
import ctypes
import datetime
import operator

# SYSTEM
SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
HAND = 'hand2'
PTR = 'left_ptr'

# EXTENSIONS
KML_EXT = '.kml'
QRY_EXT = '.qry'
TXT_EXT = '.txt'
XLSX_EXT = '.xlsx'

# EXTENSION TUPLES
ALL_FILES = ('All files', '*.*')
KML_FILES = ('KML files', '*.kml')
QRY_FILES = ('Query files', '*.qry')

# NUMERICAL
CURRENT_YEAR = datetime.datetime.now().year - 1
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
INDICATORS = ('--', '-')
NA_VALUES = ['N']
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
UNITS = ('--', 'km', 'm', 'mi', 'ft')

# GRAPHS
GRAPH_COLORS = (
    ('#3CE1E0', '#56A71E', '#7CD530', '#B6ED15', '#E6F70E', '#F5AB23', '#F06D12', '#B43519', '#891A1A', '#000000'), (
        'maroon', 'darkred', 'brown', 'firebrick', 'crimson', 'indianred', 'lightcoral', 'salmon', 'rosybrown',
        'darksalmon', 'lightsalmon', 'tomato', 'orangered', 'darkorange', 'orange'), (
        'darkcyan', 'teal', 'steelblue', 'cadetblue', 'lightslategrey', 'skyblue', 'turquoise', 'lightseagreen',
        'darkturquoise'))
GRAPH_FILL_STYLES = ('none',)  # top, bottom, right, left, full: markerfacecoloralt='gray'
GRAPH_MARKERS = ('o',)

# TODO and use os module
# The paths generator1 must read the entire database, collecting the
# necessary data to fill the lists in the list generator1 as well as information in the paths above.
# This process will occur each time a different database is used and a current working database
# file will be created to save the configurations yielded by the aforementioned procedure so that
# it doesn't need to be performed each time the program starts
if __name__ == "Utilities14.Constants":
    from Utilities.Dependencies import *
    from Utilities.Descriptions import *
    from Utilities.Extensions import *
    from Utilities.Icons import *
    from Utilities.Paths import *
    from Utilities.Variables import *


    def get_separator(header):
        counters = []
        max = 0
        s = ','

        for separator in SEPARATORS:
            counter = 0
            for character in header:
                if separator == character:
                    counter += 1
            counters.append(counter)
            if counter > max:
                max = counter
                s = separator

        return s


    def collect_headers(dir):
        """
        Create a file containing all unique
        column names in the entire database
        """
        headers = []
        separator = None
        for subdir, dirs, files in os.walk(dir):
            for file in files:
                if os.path.splitext(file)[1] != '.h5' and os.path.splitext(file)[0] != 'temp':
                    try:
                        with open(os.path.join(subdir, file), newline='') as f:
                            header = next(f)
                            if not separator:
                                separator = get_separator(header)

                        with open(os.path.join(subdir, file), newline='') as f:
                            header = next(csv.reader(f, delimiter=separator))

                        for column in header:
                            if column not in headers:
                                headers.append(column)
                    except:
                        pass

        with open(ITEM_NAMES, "w") as file:
            for header in headers:
                file.write(header + '\n')

        return headers, separator


    def optimize_dataset(path):
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1] != HDF_EXT and os.path.splitext(file)[0] != 'temp' and \
                        os.path.splitext(file)[0] + HDF_EXT not in files:
                    df = pd.read_csv(os.path.join(subdir, file), dtype='str', sep=SEPARATOR, keep_default_na=False)
                    df.to_hdf(os.path.join(subdir, os.path.splitext(file)[0] + HDF_EXT), os.path.splitext(file)[0])


    # SYSTEM
    SMALL_FONT = ("Verdana", 8)
    MEDIUM_FONT = ("Verdana", 10)
    LARGE_FONT = ("Verdana", 12)
    user32 = ctypes.windll.user32
    WIDTH = user32.GetSystemMetrics(0)
    HEIGHT = user32.GetSystemMetrics(1)
    HAND = 'hand2'
    PTR = 'left_ptr'
    SEPARATORS = (',', '\t', '|', '/', '\\', '.', ';', ':', '-', ' ')
    HEADERS, SEPARATOR = collect_headers(
            DATABASE
    )  # TODO SEPARATOR MUST UPDATE A DROP DOWN LIST VALUE IN THE DATABASE MENU
    # optimize_dataset(DATABASE)
    # NUMERICAL
    CURRENT_YEAR = datetime.datetime.now().year - 1
    EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
    INDICATORS = ('--', '-')
    NA_VALUES = ['N']
    OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
    UNITS = ('km', 'm', 'mi', 'ft')
import ctypes
import datetime
import operator

from Utilities7.MenuData import *
from Utilities7.Variables import *

# SYSTEM
SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
HAND = 'hand2'
PTR = 'left_ptr'

# EXTENSIONS
KML_EXT = '.kml'
QRY_EXT = '.qry'
TXT_EXT = '.txt'
XLSX_EXT = '.xlsx'

# EXTENSION TUPLES
ALL_FILES = ('All files', '*.*')
KML_FILES = ('KML files', '*.kml')
QRY_FILES = ('Query files', '*.qry')

# NUMERICAL
CURRENT_YEAR = datetime.datetime.now().year - 1
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
INDICATORS = ('--', '-')
NA_VALUES = ['N']
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
UNITS = ('--', 'km', 'm', 'mi', 'ft')

# GRAPHS
GRAPH_COLORS = (
    ('#3CE1E0', '#56A71E', '#7CD530', '#B6ED15', '#E6F70E', '#F5AB23', '#F06D12', '#B43519', '#891A1A', '#000000'), (
        'maroon', 'darkred', 'brown', 'firebrick', 'crimson', 'indianred', 'lightcoral', 'salmon', 'rosybrown',
        'darksalmon', 'lightsalmon', 'tomato', 'orangered', 'darkorange', 'orange'), (
        'darkcyan', 'teal', 'steelblue', 'cadetblue', 'lightslategrey', 'skyblue', 'turquoise', 'lightseagreen',
        'darkturquoise'))
GRAPH_FILL_STYLES = ('none',)  # top, bottom, right, left, full: markerfacecoloralt='gray'
GRAPH_MARKERS = ('o',)

# TODO and use os module
# The paths generator1 must read the entire database, collecting the
# necessary data to fill the lists in the list generator1 as well as information in the paths above.
# This process will occur each time a different database is used and a current working database
# file will be created to save the configurations yielded by the aforementioned procedure so that
# it doesn't need to be performed each time the program starts
# SYSTEM
SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
HAND = 'hand2'
PTR = 'left_ptr'
# NUMERICAL
CURRENT_YEAR = datetime.datetime.now().year - 1
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
INDICATORS = ('--', '-')
INTERVALS = {'from to': '--', 'between': '-'}
NA_VALUES = ['N']
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt, '==': operator.eq}
INVERTED_OPERATORS = {operator.ge: '>=', operator.le: '<=', operator.gt: '>', operator.lt: '<', operator.eq: '=='}
OPERATORS_WORDS = {
    'greaterthanorequalto': '>=', 'lessthanorequalto': '<=', 'greaterthan': '>', 'lessthan': '<',
    'equal'               : '=='
}
UNITS = ('km', 'm', 'mi', 'ft')
# MACHINE LEARNING
CLASSIFIERS_NAMES = (
    'Adaptive Boost', 'Decision Tree', 'Naive Bayes', 'Gaussian Process', 'Nearest Neighbors', 'Multilayer Perceptron',
    'Quadratic Discriminant', 'Random Forest', 'Stochastic Gradient Descent', 'Support Vector Machine')
CLUSTERERS_NAMES = ('',)
DIMENSIONALITY_REDUCERS_NAMES = ('',)
REGRESSORS_NAMES = (
    'Adaptive Boost', 'Decision Tree', 'Elastic Net', 'Gaussian Process', 'Nearest Neighbors', 'Kernel Ridge',
    'Multilayer Perceptron', 'Random Forest', 'Stochastic Gradient Descent', 'Support Vector Machine')
SAMPLER_NAMES = ('Latin Hypercube', 'Monte Carlo')
SCALER_NAMES = ('None', 'MinMax', 'MaxAbs', 'Normalizer', 'QuantileTransformer', 'Robust', 'Standard')
from Utilities.Variables import *

# SYSTEM
SMALL_FONT = ("Verdana", 8)
MEDIUM_FONT = ("Verdana", 10)
LARGE_FONT = ("Verdana", 12)
user32 = ctypes.windll.user32
WIDTH = user32.GetSystemMetrics(0)
HEIGHT = user32.GetSystemMetrics(1)
HAND = 'hand2'
PTR = 'left_ptr'

# EXTENSIONS
KML_EXT = '.kml'
QRY_EXT = '.qry'
TXT_EXT = '.txt'
XLSX_EXT = '.xlsx'

# EXTENSION TUPLES
ALL_FILES = 'All files (*.*)'
KML_FILES = 'kml files (*.kml)'
QRY_FILES = 'Query files (*.qry)'
TXT_FILES = 'Text files (*.txt)'
XLSX_FILES = 'xlsx files (*.xlsx)'

# NUMERICAL
CURRENT_YEAR = datetime.datetime.now().year - 1
EARTH_RADIUS = {'km': 6373, 'm': 6373000, 'mi': 3960, 'ft': 3950 * 5280}
INDICATORS = ('--', '-')
NA_VALUES = ['N']
OPERATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
UNITS = ('km', 'm', 'mi', 'ft')

# GRAPHS
GRAPH_COLORS = (
    ('#3CE1E0', '#56A71E', '#7CD530', '#B6ED15', '#E6F70E', '#F5AB23', '#F06D12', '#B43519', '#891A1A', '#000000'), (
        'maroon', 'darkred', 'brown', 'firebrick', 'crimson', 'indianred', 'lightcoral', 'salmon', 'rosybrown',
        'darksalmon', 'lightsalmon', 'tomato', 'orangered', 'darkorange', 'orange'), (
        'darkcyan', 'teal', 'steelblue', 'cadetblue', 'lightslategrey', 'skyblue', 'turquoise', 'lightseagreen',
        'darkturquoise'))
GRAPH_FILL_STYLES = ('none',)  # top, bottom, right, left, full: markerfacecoloralt='gray'
GRAPH_MARKERS = ('o',)

# DESCRIPTIONS
OPEN_FOLDER_DESCRIPTION = 'Open a folder from which to extract files.'
OPEN_FOLDER_ICON = ICONS + r'\openFolder.png'

SAVE_FOLDER_DESCRIPTION = 'Open a folder to save files in.'
SAVE_FOLDER_ICON = ICONS + r'\saveFolder.png'

OPEN_FILE_DESCRIPTION = 'Open file.'
OPEN_FILE_ICON = ICONS + r'\openFile.png'

SAVE_FILE_DESCRIPTION = 'Save file.'
SAVE_FILE_ICON = ICONS + r'\saveFile.png'

EXIT_DESCRIPTION = 'Exit application.'
EXIT_ICON = ICONS + r'\exit.png'

# Data encapsulates interface7, geosearch, machine learning, and sentiment analysis
DATABASE_DESCRIPTION = 'Database: Set your database and its variables.'
DATABASE_ICON = ICONS + r'\database.png'

SEARCH_DESCRIPTION = 'search Engine: Navigate your database and specify criteria for extracting data.'
SEARCH_ICON = ICONS + r'\search.png'

GEO_SEARCH_DESCRIPTION = 'Geographic search: Explore your database and produce geographical information system visualizations.'
GEO_SEARCH_ICON = ICONS + r'\geoSearch.png'

MACHINE_LEARNING_DESCRIPTION = 'Machine Learning: Perform statistical analysis on your data and visualize your results.'
MACHINE_LEARNING_ICON = ICONS + r'\machineLearning.png'

SENTIMENT_DESCRIPTION = 'Sentiment Analysis: Discover popular opinion.'
SENTIMENT_ICON = ICONS + r'\sentiment.png'

# Assist encapsulates updates and help
UPDATES_DESCRIPTION = 'Updates: Check for updates.'
UPDATES_ICON = ICONS + r'\updates.png'

HELP_DESCRIPTION = 'Help2: Browse 1 for software information and examples.'
HELP_ICON = ICONS + r'\help.png'

# Planner encapsulates schedule and tasks
PLANNER_DESCRIPTION = 'Planner: Adjust variables used within the planner such as date and time settings.'
PLANNER_ICON = ICONS + r'\plannerSettings.png'

SCHEDULE_DESCRIPTION = 'Schedule: Organize events and develop schedules.'
SCHEDULE_ICON = ICONS + r'\schedule.png'

TASKS_DESCRIPTION = 'Tasks: Delegate and monitor tasks.'
TASKS_ICON = ICONS + r'\tasks.png'

RESOURCES_DESCRIPTION = 'Resources: Manage your inventories.'
RESOURCES_ICON = ICONS + r'\resources.png'

BACKGROUND_IMAGE = ICONS + r'\background.png'

DOWN_ARROW_ICON = ICONS + r'downArrow.png'

BACKGROUND_COLOR_DARK = r'rgb(30,30,30)'
BACKGROUND_COLOR_LIGHT = r'rgb(49,49,49)'
BORDER_COLOR = r'rgb(70,70,70)'
FONT_COLOR = r'rgb(255,255,255)'

BUTTON_COLOR = r'rgb(55,55,55)'

BORDER_RADIUS = r'3px'
BORDER_WIDTH = r'3px'
BORDER_STYLE = r'double'

LIGHT_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(32,33,30), stop: 0.5 rgb(56,58,55), stop: 0.5 rgb(61,63,60), stop: 1.0 rgb(46,50,48));'
DARK_GRADIENT = r'qlineargradient(x1: 0.3, y1: 1, x2: -0.2, y2: -0.5, stop: 0.05 rgb(22,23,20), stop: 0.5 rgb(90,90,90), stop: 0.5 rgb(114,115,113), stop: 1.0 rgb(46,50,48));'

WINDOW_STYLE = """
            QMainWindow {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;

            }
            QMainWindow::separator {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: rgb(255,255,255);
                width: 10px; /* when vertical */
                height: 10px; /* when horizontal */
            }

            QMainWindow::separator:hover {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: rgb(255,255,255);
            }

            """

COMBO_STYLE = """
            QComboBox {
                background: """ + LIGHT_GRADIENT + """;
                color: rgb(255,255,255);
                border: 3px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                padding: 1px 1px 1px 3px;
                min-width: 6px;
            }

            QComboBox:!editable::hover {
                background: """ + DARK_GRADIENT + """; 
            }

            QComboBox:editable {
                background: """ + DARK_GRADIENT + """;         
            }

            /* QComboBox gets the "on" state when the popup is open */
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: """ + DARK_GRADIENT + """;
            }

            QComboBox:on { /* shift the text when the popup opens */
                padding-top: 3px;
                padding-left: 4px;
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 10px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: """ + BORDER_STYLE + """;
                border-top-right-radius: """ + BORDER_RADIUS + """;
                border-bottom-right-radius: """ + BORDER_RADIUS + """;
            }

            QComboBox::down-arrow {
            }

            QComboBox::down-arrow:on { /* shift the arrow when popup is open */
                top: 1px;
                left: 1px;
            }
            """

FRAME_STYLE = """
            QFrame, QLabel, QToolTip {
                background: """ + BACKGROUND_COLOR_DARK + """;
                color: """ + FONT_COLOR + """;
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;
                padding: 2px;
            }"""

MENUBAR_STYLE = """
            QMenuBar {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid rgb(0,0,0);
            }

            QMenuBar::item {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
            }

            QMenuBar::item::selected {
                background: """ + BACKGROUND_COLOR_DARK + """;
            }

            QMenu {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
                border: 1px solid #000;           
            }

            QMenu::item::selected {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
            }
            """
SPLITTER_STYLE = """
            QSplitter::handle {
                background: """ + BACKGROUND_COLOR_LIGHT + """;
                color: """ + FONT_COLOR + """;
            }

            QSplitter::handle:horizontal {
                width: 4px;
            }

            QSplitter::handle:vertical {
                height: 4px;
            }

            QSplitter::handle:pressed {
                background-color: """ + BACKGROUND_COLOR_DARK + """;
            }
            """

STATUSBAR_STYLE = """
            QStatusBar {
                background: rgb(49,49,49);
                color: """ + FONT_COLOR + """;
                border: 1px solid rgb(100,100,100);
            }

            QStatusBar::item {
                border: """ + BORDER_WIDTH + """ """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;
                border-radius: """ + BORDER_RADIUS + """;

            }"""

TOOLBAR_STYLE = """ 
            QToolBar, QToolButton, QToolTip { 
                background: rgb(56,60,55);
                background: """ + LIGHT_GRADIENT + """;

                color: """ + FONT_COLOR + """;
                spacing: 3px; /* spacing between items in the tool bar */
                border: 1px """ + BORDER_STYLE + """ """ + BORDER_COLOR + """;

            } 

            QToolButton:hover {
                background: """ + DARK_GRADIENT + """;
                border: 0px;
            }

            QToolBar::handle {
                background: rgb(45,50,48);
                border: 1px solid rgb(100,100,100);
            } 
            """


# TODO and use os module
# The paths generator1 must read the entire database, collecting the
# necessary data to fill the lists in the list generator1 as well as information in the paths above.
# This process will occur each time a different database is used and a current working database
# file will be created to save the configurations yielded by the aforementioned procedure so that
# it doesn't need to be performed each time the program starts


class ListGenerator:
    def __init__(self, parent):
        self.numbers, self.names, self.entryList, self.names_numbers = {}, {}, {}, {}
        self.parent = parent

    def setSearchMenuLists(self, populator):
        self.populator = populator
        with open(
                self.parent.MASTERPATH + "\BridgeDataQuery\Interface\InterfaceUtilities\Labels\parameterNumbers.txt",
                "r"
                ) as parameterNumbersFile:
            with open(self.parent.MASTERPATH + "\BridgeDataQuery\Database\ItemNames.txt", "r") as f:
                k = 0
                for line, line_num in (zip(f, parameterNumbersFile)):
                    self.entryList[line.strip()] = ttk.Entry(self.populator.frame.interior)
                    self.numbers[line.strip()] = line_num.strip()
                    self.names[k] = line.strip()
                    k += 1

    def getParameterNumbers(self):
        return self.numbers

    def getParameterNames(self):
        return self.names

    def getEntryList(self):
        return self.entryList

    def getParameterNumbersDictionary(self):
        # self.parameterNumbersDictionary = {}
        # for m in range(self.numbers.__len__()):
        # self.parameterNumbersDictionary[self.numbers[m]] = m + 1

        return self.numbers  # self.parameterNumbersDictionary

    def getParameterNamesWithRespectToNumber(self):
        return {self.numbers[number]: self.names[number] for number in self.numbers}

    def setNonNoneValueList(self):
        # A list of items that can transition through states
        self.nonNoneValueList, self.nonNoneNumber, self.nonNoneName = [], [], []
        path1 = r"\BridgeDataQuery\Interface\InterfaceUtilities\Labels\parameterValues.txt"
        path2 = r"\BridgeDataQuery\Database\ItemNames.txt"

        with open(self.parent.MASTERPATH + path1, "r") as parameterValuesFile:
            with open(self.parent.MASTERPATH + path2, "r") as parameterNamesFile:
                for name, value in zip(csv.reader(parameterNamesFile), (csv.reader(parameterValuesFile))):
                    if name[0] != 'YEAR' and name[0] != "STATE_CODE_001" and "None" not in value:
                        self.nonNoneValueList.append(name[0])

        print(self.nonNoneValueList)
        print(self.names)
        print(self.numbers)

        self.nonNoneName = [self.names[i] for i in self.names if (self.names[i]) in self.nonNoneValueList]
        self.nonNoneNumber = [self.numbers[name] for name in self.numbers if (name) in self.nonNoneValueList]

        print(self.nonNoneName)
        print(self.nonNoneNumber)

        """# A list of items that can transition through states
        self.nonNoneValueList, self.nonNoneNumber, self.nonNoneName = [], [], []
        with open(self.parent.MASTERPATH + "\\BridgeDataQuery\Interface\InterfaceUtilities\Labels\parameterValues.txt", "r") as parameterValuesFile:
            for omega, row in enumerate(csv.reader(parameterValuesFile), start=1):
                if omega != 1 and omega != 1 and "None" not in row:
                    self.nonNoneValueList.append(omega)
        print(self.nonNoneValueList)
        self.nonNoneName = [self.names[name] for name in self.names if (name + 1) in self.nonNoneValueList]
        self.nonNoneNumber = [self.numbers[name] for name in self.numbers if (name + 1) in self.nonNoneValueList]"""

    def getNonNoneValueList(self):
        return (self.nonNoneValueList)

    def getNonNoneName(self):
        return (self.nonNoneName)

    def getNonNoneNumber(self):
        return (self.nonNoneNumber)

    # A list of items corresponding to a specific item in the nonNoneValueList
    def valid_lists(self, index):
        self.index = index
        self.valid_values = ''
        with open(
                self.parent.MASTERPATH + "\BridgeDataQuery\Interface\InterfaceUtilities\Labels\parameterValues.txt",
                "r"
                ) as parameterValuesFile:
            for omega, row in enumerate(csv.reader(parameterValuesFile), start=1):
                if omega == int(self.index):
                    self.valid_values = row
                    break

    def get_valid_values_list(self):
        return self.valid_values

    def state_code_state(self):
        self.states = {}
        states_path = 'C:\\Users\\frano\PycharmProjects\BridgeDataQuery\Database\Information\\allStates.txt'
        state_codes_path = 'C:\\Users\\frano\PycharmProjects\BridgeDataQuery\Database\Information\\allStateCodes.txt'
        with open(states_path) as states:
            with open(state_codes_path) as state_codes:
                for state, state_code in zip(states, state_codes):
                    self.states[state_code.strip()] = state.strip()  # print(self.states)

        return self.states


class GenerateBridge:
    def __init__(self, query, fileStructure, items_in_bridge):
        self.query, self.fileStructure, self.items_in_bridge = query, fileStructure, items_in_bridge
        self.bridgeItems = {}
        omega = 0

        for item in self.fileStructure:
            self.bridgeItems[item.lstrip("0")] = self.items_in_bridge[
                omega]  # POPULATE THE BRIDGE WITH THE ITEMS FROM THE FILE
            omega += 1

    def __del__(self):
        pass

    def getBridgeItems(self):
        return self.bridgeItems  # RETURN THE LIST OF ITEMS ASSOCIATED WITH BRIDGE


import tkinter.ttk as ttk


class ListGenerator:
    def __init__(self, parent):
        self.allStates, self.allStateCodes, self.numbers, self.names, self.entryList, self.names_numbers = [], {}, {}, {}, {}, {}
        self.states_to_search = []
        self.parent = parent

        # PARAMETER NUMBERS
        with open(
                self.parent.MASTERPATH + "\BridgeDataQuery\Interface\InterfaceUtilities\Labels\parameterNumbers.txt",
                "r"
                ) as parameterNumbersFile:
            for k, line in enumerate(parameterNumbersFile):
                self.numbers[k] = line.strip()

        # A LIST OF ALL STATES
        with open(self.parent.MASTERPATH + '\BridgeDataQuery\Database\Information\\allStates.txt', 'r') as allStates:
            self.allStates = [line.strip() for line in allStates]

        # A LIST OF ALL STATE CODES
        with open(
                self.parent.MASTERPATH + '\BridgeDataQuery\Database\Information\\allStateCodes.txt',
                'r'
                ) as allStateCodes:
            for i, line in enumerate(allStateCodes):
                self.allStateCodes[self.allStates[i]] = line.strip()

    def getParameterNumbers(self):
        return self.numbers

    def getParameterNames(self, populator):
        self.populator = populator
        with open(
                self.parent.MASTERPATH + "\BridgeDataQuery\Interface\InterfaceUtilities\Labels\parameterNames.txt",
                "r"
                ) as parameterNamesFile:
            for k, line in enumerate(parameterNamesFile):
                if k == self.parent.number_of_items:
                    break
                self.names[k], self.entryList[line.strip()] = line.strip(), ttk.Entry(self.populator.frame.interior)

        return self.names

    def getEntryList(self):
        return self.entryList

    def getAllStates(self):
        return self.allStates

    def getAllStateCodes(self):
        return self.allStateCodes

    def getParameterNumbersDictionary(self):
        self.parameterNumbersDictionary = {}
        for m in range(self.numbers.__len__()):
            self.parameterNumbersDictionary[self.numbers[m]] = m + 1

        return self.parameterNumbersDictionary

    def getParameterNamesWithRespectToNumber(self):
        return {self.numbers[number]: self.names[number] for number in self.numbers}

    def setSearchList(self, query):
        self.query = query
        self.search = {}
        # This for loop is generating the list of items to be searched.
        for item in self.numbers:
            self.search[self.numbers[item]] = '' if self.names[item] == '(Reserved)' else self.query[self.names[item]]

    def getSearchList(self):
        return self.search

    def setListOfStatesToSearch(self, states_to_search):
        self.states_to_search = states_to_search

    def get_list_of_states_to_search(self):
        return self.states_to_search

    # A list of items that transition states
    def setNonNoneValueList(self):
        self.nonNoneValueList, self.nonNoneNumber, self.nonNoneName = [], [], []

        with open(
                self.parent.MASTERPATH + "\\BridgeDataQuery\Interface\INterfaceUtilities\Labels\parameterValues.txt",
                "r"
                ) as parameterValuesFile:

            for omega, row in enumerate(csv.reader(parameterValuesFile), start=1):
                if omega != 1 and omega != 2 and "None" not in row:
                    self.nonNoneValueList.append(omega)

        self.nonNoneName = [self.names[name] for name in self.names if (name + 1) in self.nonNoneValueList]
        self.nonNoneNumber = [self.numbers[name] for name in self.numbers if (name + 1) in self.nonNoneValueList]

    def getNonNoneValueList(self):
        return (self.nonNoneValueList)

    def getNonNoneName(self):
        return (self.nonNoneName)

    def getNonNoneNumber(self):
        return (self.nonNoneNumber)

    # A list of items corresponding to a specific item in the nonNoneValueList
    def valid_lists(self, index):
        self.index = index
        self.valid_values = ''
        with open(
                self.parent.MASTERPATH + "\\BridgeDataQuery\Interface\InterfaceUtilities\Labels\parameterValues.txt",
                "r"
                ) as parameterValuesFile:
            for omega, row in enumerate(csv.reader(parameterValuesFile), start=1):
                if omega == int(self.index):
                    self.valid_values = row
                    break

    def get_valid_values_list(self):
        return self.valid_values


class GenerateBridge:
    def __init__(self, Search, items_in_bridge):
        self.Search = Search
        self.items_in_bridge = items_in_bridge
        self.bridge = {}

        for omega, item in enumerate(self.Search.fileStructure):
            self.bridge[item.lstrip("0")] = self.items_in_bridge[
                omega]  # POPULATE THE BRIDGE WITH THE ITEMS FROM THE FILE

    def __del__(self):
        pass

    def get_bridge(self):
        return self.bridge  # RETURN THE LIST OF ITEMS ASSOCIATED WITH BRIDGE


"""object_oriented StatesInYear:

    def __init__(self):

        years = []                  #This will hold a list of the years for which data is available
        starting_year = 1992        #This is the first year in the database
        number_of_years = 25        #The number of years in the database
        for i in range(number_of_years):    #Populating the list with the years
            years.append(i + starting_year)

        self.statesInYear = {}
        self.states = {}
        self.bridges = {}
        self.items = {}

        with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\statesList.txt", "r") as statesList:
            with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\bridgesList.txt", "r") as bridgesList:
                with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\GUI\parameterNames.txt","r") as itemsList:
                    for year in years:
                        for state in statesList:
                            for bridge in bridgesList:
                                for item in itemsList:
                                    self.items[item.strip()] = 0
                                self.bridges[bridge.strip()] = self.items
                            self.states[state.strip()] = self.bridges
                        self.statesInYear[year] = self.states

        with open("report.txt", "w") as report:
            report.write(str(self.statesInYear))


    #print(years[i])
siy = StatesInYear()
"""

"""
k = 0
object_oriented Items_In_Bridge:
    def __init__(self):
        pass
    def itemsList(self):
        self.itemsInBridge
        return itemsInBridge

yearsList = [1992, 1993, 1994]
statesList = ['TX', 'AL', 'MI', 'WY']
list_of_bridges = ['Asuna Bridge', 'Clinsimp Bridges',
                   'Takis Bridges', 'Bricken Bridge', "Hubart's Bridge",
                   'Akai Bridge', 'Friendship Bridge', 'Wal-Mart Bridge',
                   'Bose Bridge', 'Samsung Bridge', 'Converse Bridge']
itemsList = ['Age', 'Material', 'Span']
itemValueList = ['', 0, 1, 1, 2, 3]
years = {}
statesInYear = {}
bridgesInState = {}
#itemsInBridge = {}
#for year in years:
    #for state in statesList:
bridgesList = []


for i in range(random.randrange(1, 10)):
    bridgesList.append(list_of_bridges[random.randrange(10)])

itemsInBridge={ bridge:{ key:[] for key in bridgesList} for bridge in bridgesInState}
print(itemsInBridge)
"""
"""
for i in range(random.randrange(1, 10)):
    bridgesList.append(list_of_bridges[random.randrange(10)])
    print(bridgesList)
for bridge in bridgesList:
    for item in itemsList:
        #itemsInBridge[item] = itemValueList[random.randrange(5)]

    bridgesInState[bridge] = itemsInBridge
    k += 1
    print(itemsInBridge)
print (bridgesInState)"""

"""
names=["lloyd", "alice", "tyler"]
keys=["homework", "quizzes", "tests"]
dic={ name.capitalize():{ key:[] for key in keys} for name in names}
print(dic)"""


class years:
    def __init__(self):
        print('years')


class states(years):
    def __init__(self):
        print('states')


class bridges(states):
    def __init__(self):
        print('bridges')


class items(bridges):
    def __init__(self):
        print('items')


states()
years()

"""class StatesInYear:

    def __init__(self):

        years = []                  #This will hold a list of the years for which data is available
        starting_year = 1992        #This is the first year in the database
        number_of_years = 25        #The number of years in the database
        for i in range(number_of_years):    #Populating the list with the years
            years.append(i + starting_year)

        self.statesInYear = {}
        self.states = {}
        self.bridges = {}
        self.items = {}

        with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\statesList.txt", "r") as statesList:
            with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\bridgesList.txt", "r") as bridgesList:
                with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\GUI\parameterNames.txt","r") as itemsList:
                    for year in years:
                        for state in statesList:
                            for bridge in bridgesList:
                                for item in itemsList:
                                    self.items[item.strip()] = 0
                                self.bridges[bridge.strip()] = self.items
                            self.states[state.strip()] = self.bridges
                        self.statesInYear[year] = self.states

        with open("report.txt", "w") as report:
            report.write(str(self.statesInYear))


    #print(years[i])
siy = StatesInYear()
"""

"""
k = 0
class Items_In_Bridge:
    def __init__(self):
        pass
    def itemsList(self):
        self.itemsInBridge
        return itemsInBridge

yearsList = [1992, 1993, 1994]
statesList = ['TX', 'AL', 'MI', 'WY']
list_of_bridges = ['Asuna Bridge', 'Clinsimp Bridges',
                   'Takis Bridges', 'Bricken Bridge', "Hubart's Bridge",
                   'Akai Bridge', 'Friendship Bridge', 'Wal-Mart Bridge',
                   'Bose Bridge', 'Samsung Bridge', 'Converse Bridge']
itemsList = ['Age', 'Material', 'Span']
itemValueList = ['', 0, 1, 2, 3, 4]
years = {}
statesInYear = {}
bridgesInState = {}
#itemsInBridge = {}
#for year in years:
    #for state in statesList:
bridgesList = []


for i in range(random.randrange(1, 10)):
    bridgesList.append(list_of_bridges[random.randrange(10)])

itemsInBridge={ bridge:{ key:[] for key in bridgesList} for bridge in bridgesInState}
print(itemsInBridge)
"""
"""
for i in range(random.randrange(1, 10)):
    bridgesList.append(list_of_bridges[random.randrange(10)])
    print(bridgesList)
for bridge in bridgesList:
    for item in itemsList:
        #itemsInBridge[item] = itemValueList[random.randrange(5)]

    bridgesInState[bridge] = itemsInBridge
    k += 1
    print(itemsInBridge)
print (bridgesInState)"""

"""
names=["lloyd", "alice", "tyler"]
keys=["homework", "quizzes", "tests"]
dic={ name.capitalize():{ key:[] for key in keys} for name in names}
print(dic)"""


class years:
    def __init__(self):
        print('years')


class states(years):
    def __init__(self):
        print('states')


class bridges(states):
    def __init__(self):
        print('bridges')


class items(bridges):
    def __init__(self):
        print('items')


import os


def collect_headers(dir):
    """
    Create a file containing all unique
    column names in the entire database
    """
    headers = [FOLDER]
    for folder in os.listdir(dir):
        path = dir + '/' + folder
        for filepath in os.listdir(path):
            try:
                with open(path + '/' + filepath, newline='') as file:
                    header = next(csv.reader(file))
                for column in header:
                    if column not in headers:
                        headers.append(column)
            except:
                pass

    with open(ITEM_NAMES, "w") as file:
        for header in headers:
            file.write(header + '\n')

    return headers


starting_year = 1992
years = {}
states = []
stateCodes = {}


# THIS CLASS CREATES A YEAR OBJECT
class Year:
    def __init__(self, year):
        self.year = year
        self.DictOfStatesInYear = {}

    def getYear(self):
        return self.year

    """def setDictOfStatesInYear(self, DictOfStatesInYear):
        self.DictOfStatesInYear = DictOfStatesInYear"""

    """def getListOfStatesInYear(self):
        return self.DictOfStatesInYear"""

    def addYearToDictOfYears(self):
        years[self.year] = self.DictOfStatesInYear


# THIS CLASS CREATES A STATE OBJECT
class State(Year):
    def __init__(self, state, stateCode):
        self.state = state
        self.stateCode = stateCode
        self.DictOfBridgesInState = {}

    def getState(self):
        return self.state

    def getStateCode(self):
        return self.stateCode

    """def setListOfBridgesInState(self, ListOfBridgesInState):
        self.ListOfBridgesInState = ListOfBridgesInState"""

    def addStateToDictOfStatesInYear(self, temporaryYearObject):
        temporaryYearObject.DictOfStatesInYear[self.state] = self.DictOfBridgesInState


# THIS CLASS CREATES A BRIDGE OBJECT
class Bridge(State):
    def __init__(self, bridge):
        self.bridge = bridge
        self.DictOfItemsInBridge = {}

    def getBridge(self):
        return self.bridge

    """def setDictOfItemsInBridge(self,DictOfItemsInBridge):
        self.DictOfItemsOfBridge = DictOfItemsInBridge"""

    def addBridgeToDictOfBridgesInState(self, temporaryStateObject):
        temporaryStateObject.DictOfBridgesInState[self.bridge] = self.DictOfItemsInBridge


# THIS CLASS CREATES AN ITEM OBJECT
class Item(Bridge):
    def __init__(self, item):
        self.item = item

    def getItem(self):
        return self.item

    def setValueOfItemInDictOfItemsInBridge(self, temporaryBridgeObject):
        temporaryBridgeObject.DictOfItemsInBridge[self.item] = random.randrange(6)

    def getValueOfItemInDictOfItemsInBridge(self, temporaryBridgeObject):
        return temporaryBridgeObject.DictOfItemsInBridge[self.item]

    def setYearValue(self, temporaryBridgeObject, temporaryYearObject):
        temporaryBridgeObject.DictOfItemsInBridge[self.item] = temporaryYearObject.getYear()

    def setStateCodeValue(self, temporaryBridgeObject, temporaryStateObject):
        temporaryBridgeObject.DictOfItemsInBridge[self.item] = temporaryStateObject.getStateCode()

    def setStructureNumberValue(self, temporaryBridgeObject):
        temporaryBridgeObject.DictOfItemsInBridge[self.item] = temporaryBridgeObject.getBridge()


# THIS CLASS IS DISCRIMINATING AGAINST THE DATA THAT DOES NOT MATCH THE QUERY
class Reducer:
    def __init__(self, MASTERPATH, parameterNames, parameterNumbers, query):
        self.parameterNames = parameterNames
        self.parameterNumbers = parameterNumbers
        self.query = query
        self.initial_year_iteration = 0
        self.years_to_iterate = 25
        self.found = False
        self.bridge_to_search = None

        with open(MASTERPATH + '\BridgeDataQuery\Database\Information\\allStates.txt', 'r') as allStates:
            for line in allStates:
                states.append(line.strip())

        with open(MASTERPATH + '\BridgeDataQuery\Database\Information\\allStateCodes.txt', 'r') as allStateCodes:
            i = 0
            for line in allStateCodes:
                stateCodes[states[i]] = line.strip()
                i += 1

                # REDUCE YEARS
        try:
            if (int(self.query['Year']) >= starting_year) and (int(self.query['Year']) < 2017):
                self.initial_year_iteration = int(self.query['Year']) - starting_year
                self.years_to_iterate = self.initial_year_iteration + 1
            else:
                print(self.query['Year'] + " is not a valid year. search aborted.")
                return
        except:
            pass

            # REDUCE STATES
        with open(MASTERPATH + "\\BridgeDataQuery\Database\Information\statesList.txt", "w") as statesList:
            with open(MASTERPATH + "\\BridgeDataQuery\Database\Information\\stateCodesList.txt", "w") as stateCodesList:
                if self.query['State Code'] is not '':
                    for state in stateCodes:
                        if self.query['State Code'] == stateCodes[state]:
                            statesList.write(state)
                            stateCodesList.write(self.query['State Code'])
                            self.found = True
                    if not self.found:
                        print(self.query['State Code'] + " is not a valid State Code. search aborted.")
                        return
                else:
                    for state in stateCodes:
                        statesList.write(state + "\n")
                        stateCodesList.write(stateCodes[state] + "\n")

                        # REDUCE BRIDGES
        if self.query['Structure Number'] is not '':
            self.bridge_to_search = (self.query['Structure Number'])
        else:
            pass

            # REDUCE ITEMS
        """for q in range(116):
            if q == 0 or q == 1 or q == 5:
                if self.query[q] is not '': pass"""

        DatabaseNavigator(
            MASTERPATH, self.query, self.parameterNames, self.parameterNumbers,
            self.initial_year_iteration, self.years_to_iterate, self.bridge_to_search
            )


# THIS CLASS CREATES THE DATABASE WHICH WILL BE REPORTED AFTER DISCRIMINATION
class DatabaseNavigator:
    def __init__(
            self, MASTERPATH, query, parameterNames, parameterNumbers, initial_year_iteration, years_to_iterate,
            bridge_to_search
            ):
        years.clear()
        self.query = query
        self.parameterNames = parameterNames
        self.parameterNumbers = parameterNumbers
        self.discriminate_bridge = False
        self.include = True
        self.initial_year_iteration = initial_year_iteration
        self.years_to_iterate = years_to_iterate
        self.bridge_to_search = bridge_to_search
        if self.bridge_to_search:
            self.discriminate_bridge = True

        # CYCLE THROUGH EACH YEAR
        for j in range(self.initial_year_iteration, self.years_to_iterate):
            temporaryYearObject = Year(j + starting_year)
            # Open list of states to validate discrimination
            with open(MASTERPATH + "\\BridgeDataQuery\Database\Information\\statesList.txt", "r") as statesList:
                # CYCLE THROUGH EACH STATE WITHIN THE CURRENT YEAR
                for state in statesList:
                    temporaryStateObject = State(state.strip(), stateCodes[state.strip()])

                    # CYCLE THROUGH EACH BRIDGE WITHIN THE CURRENT STATE
                    for k in range(random.randrange(3) + 3):
                        self.include = True
                        temporaryBridgeObject = Bridge(
                                'Bridge' + str(random.randrange(3) + 3)
                        )  # + str(random.randrange(100)))
                        if temporaryBridgeObject.getBridge() != self.bridge_to_search and self.discriminate_bridge:
                            continue
                        # CYCLE THROUGH EACH ITEM WITHIN THE CURRENT BRIDGE
                        for item in parameterNumbers:
                            temporaryItemObject = Item("Item " + parameterNumbers[item] + ": " + parameterNames[item])

                            if parameterNames[item] == 'Year':
                                temporaryItemObject.setYearValue(temporaryBridgeObject, temporaryYearObject)
                            elif parameterNames[item] == 'State Code':
                                temporaryItemObject.setStateCodeValue(temporaryBridgeObject, temporaryStateObject)
                            elif parameterNames[item] == 'Structure Number':
                                temporaryItemObject.setStructureNumberValue(temporaryBridgeObject)
                            else:
                                temporaryItemObject.setValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)
                            # print("Value in object: " + str(temporaryItemObject.getValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)))
                            # print("Value in query: " + str(query[temporaryItemObject.getItem()]))
                            # print("Match: " + str(str(temporaryItemObject.getValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)) == str(query[temporaryItemObject.getItem()])))
                            # print("Condition 1: " + str((query[temporaryItemObject.getItem()] is not '')))
                            # print("Condition 1: " + str(str(temporaryItemObject.getValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)).strip() != str(query[temporaryItemObject.getItem()]).strip()))
                            try:
                                if (query[temporaryItemObject.getItem()] is not ''):
                                    if (str(
                                            temporaryItemObject.getValueOfItemInDictOfItemsInBridge(
                                                    temporaryBridgeObject
                                            )
                                    ).strip() != str(
                                            query[temporaryItemObject.getItem()]
                                    ).strip()):
                                        self.include = False
                                        continue
                            except:
                                pass
                        # print("Include: " + str(self.include))
                        if self.include:
                            temporaryBridgeObject.addBridgeToDictOfBridgesInState(temporaryStateObject)

                    temporaryStateObject.addStateToDictOfStatesInYear(temporaryYearObject)

                temporaryYearObject.addYearToDictOfYears()

        for year in years:
            print(str(year) + ": " + str(years[year]))
        # print(years[1992]['TX']['Bridge0']['State Code'])
        writeReport(MASTERPATH)


class writeReport:
    # WRITE REPORT
    def __init__(self, MASTERPATH):
        with open(MASTERPATH + "\\BridgeDataQuery\Reports\\report.txt", "w") as report:
            for year in years:
                report.write(str(year) + ": " + str(years[year]))
            print("report_generator0 has been generated.")


"""
states.append(State(1999))
states[0].printYear()
states[0].setState('TX')
states[0].getState()#
states[0].getYear()#
states.append(State(2000))
states[1].setState('AL')
states[1].getYear()#
states[1].getState()#
states[0].getYear()#

bridges.append(Bridge(2002))
bridges[0].setState('AL')
bridges[0].setBridge('Main5 Bridge')
bridges[0].getBridge()#
bridges[0].getYear()#
bridges[0].getState()#

bridges.append(Bridge(2015))
bridges[1].setState('CO')
bridges[1].setBridge('Bridge 54')
bridges[1].getBridge()#
bridges[1].getYear()#
bridges[1].getState()#



object_oriented Human:
    def __init__(self, height, weight, maxH):
        self.height = height
        self.weight = weight
        self.maxH = maxH

    def setHeight(self, height):
        self.height = height

    def getHeight(self):
        return self.height

    def getMaxH(self):
        return self.maxH


Keren = Human(20, 1, 160)
Christian = Human(15, 0.3, 175)

for i in range(200):
    Keren.setHeight(Keren.getMaxH()*(i/200) + Keren.height)
    Christian.setHeight(Christian.getMaxH()*(i/200) + Christian.height)
    print("Keren's height: " + str(Keren.getHeight()))
    print("Christian's height: " + str(Christian.getHeight()))"""

############################################################################
"""
for j in range(24):
    temporaryYearObject = Year(j+1992)

    for i in range(1):

        if j is 0:
            if i is 0:
                temporaryStateObject = State('TX')
                for k in range(random.randrange(2) + 1):
                    temporaryBridgeObject = Bridge('Bridge' + str(k*10*random.randrange(5)) + str(k*101*random.randrange(5)))
                    for i in range(2):
                        temporaryItemObject = Item('Item ' + str(i))
                        temporaryItemObject.setValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)
                    temporaryBridgeObject.addBridgeToDictOfBridgesInState(temporaryStateObject)

            elif i is 1:
                temporaryStateObject = State('NM')
                for k in range(random.randrange(2) + 1):
                    temporaryBridgeObject = Bridge('Bridge' + str(k*10*random.randrange(5)) + str(101*random.randrange(5)))
                    for i in range(2):
                        temporaryItemObject = Item('Item ' + str(i))
                        temporaryItemObject.setValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)
                    temporaryBridgeObject.addBridgeToDictOfBridgesInState(temporaryStateObject)
        elif j is 1:
            if i is 0:
                temporaryStateObject = State('CO')
                for k in range(random.randrange(2) + 1):
                    temporaryBridgeObject = Bridge('Bridge' + str(k*10*random.randrange(5)) + str(101)*random.randrange(5))
                    for i in range(2):
                        temporaryItemObject = Item('Item ' + str(i))
                        temporaryItemObject.setValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)
                    temporaryBridgeObject.addBridgeToDictOfBridgesInState(temporaryStateObject)
            elif i is 1:
                temporaryStateObject = State('AL')
                for k in range(random.randrange(2) + 1):
                    temporaryBridgeObject = Bridge('Bridge' + str(k*10*random.randrange(5)) + str(101*random.randrange(5)))
                    for i in range(2):
                        temporaryItemObject = Item('Item ' + str(i))
                        temporaryItemObject.setValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)
                    temporaryBridgeObject.addBridgeToDictOfBridgesInState(temporaryStateObject)
        elif j is 1:
            if i is 0:
                temporaryStateObject = State('LA')
                for k in range(random.randrange(2) + 1):
                    temporaryBridgeObject = Bridge('Bridge' + str(k*10*random.randrange(5)) + str(101*random.randrange(5)))
                    for i in range(2):
                        temporaryItemObject = Item('Item ' + str(i))
                        temporaryItemObject.setValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)
                    temporaryBridgeObject.addBridgeToDictOfBridgesInState(temporaryStateObject)
            elif i is 1:
                temporaryStateObject = State('WY')
                for k in range(random.randrange(2) + 1):
                    temporaryBridgeObject = Bridge('Bridge' + str(k*10*random.randrange(5)) + str(101*random.randrange(5)))
                    for i in range(2):
                        temporaryItemObject = Item('Item ' + str(i))
                        temporaryItemObject.setValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)
                    temporaryBridgeObject.addBridgeToDictOfBridgesInState(temporaryStateObject)

        temporaryStateObject.addStateToDictOfStatesInYear(temporaryYearObject)

    temporaryYearObject.addYearToDictOfYears()"""

import random

starting_year = 1992
years = {}
states = []
stateCodes = {}


# THIS CLASS CREATES A STATE OBJECT
class State(Year):
    def __init__(self, state, stateCode):
        self.state = state
        self.stateCode = stateCode
        self.DictOfBridgesInState = {}

    def getState(self):
        return self.state

    def getStateCode(self):
        return self.stateCode

    """def setListOfBridgesInState(self, ListOfBridgesInState):
        self.ListOfBridgesInState = ListOfBridgesInState"""

    def addStateToDictOfStatesInYear(self, temporaryYearObject):
        temporaryYearObject.DictOfStatesInYear[self.state] = self.DictOfBridgesInState


# THIS CLASS CREATES A BRIDGE OBJECT
class Bridge(State):
    def __init__(self, bridge):
        self.bridge = bridge
        self.DictOfItemsInBridge = {}

    def getBridge(self):
        return self.bridge

    """def setDictOfItemsInBridge(self,DictOfItemsInBridge):
        self.DictOfItemsOfBridge = DictOfItemsInBridge"""

    def addBridgeToDictOfBridgesInState(self, temporaryStateObject):
        temporaryStateObject.DictOfBridgesInState[self.bridge] = self.DictOfItemsInBridge


# THIS CLASS IS DISCRIMINATING AGAINST THE DATA THAT DOES NOT MATCH THE QUERY
class Reducer:
    def __init__(self, MASTERPATH, parameterNames, parameterNumbers, query):
        self.parameterNames = parameterNames
        self.parameterNumbers = parameterNumbers
        self.query = query
        self.initial_year_iteration = 0
        self.years_to_iterate = 25
        self.found = False
        self.bridge_to_search = None

        with open(MASTERPATH + '\BridgeDataQuery\Database\Information\\allStates.txt', 'r') as allStates:
            for line in allStates:
                states.append(line.strip())

        with open(MASTERPATH + '\BridgeDataQuery\Database\Information\\allStateCodes.txt', 'r') as allStateCodes:
            i = 0
            for line in allStateCodes:
                stateCodes[states[i]] = line.strip()
                i += 1

                # REDUCE YEARS
        try:
            if (int(self.query['Year']) >= starting_year) and (int(self.query['Year']) < 2017):
                self.initial_year_iteration = int(self.query['Year']) - starting_year
                self.years_to_iterate = self.initial_year_iteration + 1
            else:
                print(self.query['Year'] + " is not a valid year. Search aborted.")
                return
        except:
            pass

            # REDUCE STATES
        with open(MASTERPATH + "\\BridgeDataQuery\Database\Information\statesList.txt", "w") as statesList:
            with open(MASTERPATH + "\\BridgeDataQuery\Database\Information\\stateCodesList.txt", "w") as stateCodesList:
                if self.query['State Code'] is not '':
                    for state in stateCodes:
                        if self.query['State Code'] == stateCodes[state]:
                            statesList.write(state)
                            stateCodesList.write(self.query['State Code'])
                            self.found = True
                    if not self.found:
                        print(self.query['State Code'] + " is not a valid State Code. Search aborted.")
                        return
                else:
                    for state in stateCodes:
                        statesList.write(state + "\n")
                        stateCodesList.write(stateCodes[state] + "\n")

                        # REDUCE BRIDGES
        if self.query['Structure Number'] is not '':
            self.bridge_to_search = (self.query['Structure Number'])
        else:
            pass

            # REDUCE ITEMS
        """for q in range(116):
            if q == 0 or q == 1 or q == 8:
                if self.query[q] is not '': pass"""

        DatabaseNavigator(
            MASTERPATH, self.query, self.parameterNames, self.parameterNumbers,
            self.initial_year_iteration, self.years_to_iterate, self.bridge_to_search
            )


class writeReport:
    # WRITE REPORT
    def __init__(self, MASTERPATH):
        with open(MASTERPATH + "\\BridgeDataQuery\Reports\\report.txt", "w") as report:
            for year in years:
                report.write(str(year) + ": " + str(years[year]))
            print("Report has been generated.")


"""
states.append(State(1999))
states[0].printYear()
states[0].setState('TX')
states[0].getState()#
states[0].getYear()#
states.append(State(2000))
states[1].setState('AL')
states[1].getYear()#
states[1].getState()#
states[0].getYear()#

bridges.append(Bridge(2002))
bridges[0].setState('AL')
bridges[0].setBridge('Main Bridge')
bridges[0].getBridge()#
bridges[0].getYear()#
bridges[0].getState()#

bridges.append(Bridge(2015))
bridges[1].setState('CO')
bridges[1].setBridge('Bridge 54')
bridges[1].getBridge()#
bridges[1].getYear()#
bridges[1].getState()#



class Human:
    def __init__(self, height, weight, maxH):
        self.height = height
        self.weight = weight
        self.maxH = maxH

    def setHeight(self, height):
        self.height = height

    def getHeight(self):
        return self.height

    def getMaxH(self):
        return self.maxH


Keren = Human(20, 1, 160)
Christian = Human(15, 0.7, 175)

for i in range(200):
    Keren.setHeight(Keren.getMaxH()*(i/200) + Keren.height)
    Christian.setHeight(Christian.getMaxH()*(i/200) + Christian.height)
    print("Keren's height: " + str(Keren.getHeight()))
    print("Christian's height: " + str(Christian.getHeight()))"""

############################################################################
"""
for j in range(24):
    temporaryYearObject = Year(j+1992)

    for i in range(2):

        if j is 0:
            if i is 0:
                temporaryStateObject = State('TX')
                for k in range(random.randrange(3) + 1):
                    temporaryBridgeObject = Bridge('Bridge' + str(k*10*random.randrange(5)) + str(k*101*random.randrange(5)))
                    for i in range(3):
                        temporaryItemObject = Item('Item ' + str(i))
                        temporaryItemObject.setValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)
                    temporaryBridgeObject.addBridgeToDictOfBridgesInState(temporaryStateObject)

            elif i is 1:
                temporaryStateObject = State('NM')
                for k in range(random.randrange(3) + 1):
                    temporaryBridgeObject = Bridge('Bridge' + str(k*10*random.randrange(5)) + str(101*random.randrange(5)))
                    for i in range(3):
                        temporaryItemObject = Item('Item ' + str(i))
                        temporaryItemObject.setValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)
                    temporaryBridgeObject.addBridgeToDictOfBridgesInState(temporaryStateObject)
        elif j is 1:
            if i is 0:
                temporaryStateObject = State('CO')
                for k in range(random.randrange(3) + 1):
                    temporaryBridgeObject = Bridge('Bridge' + str(k*10*random.randrange(5)) + str(101)*random.randrange(5))
                    for i in range(3):
                        temporaryItemObject = Item('Item ' + str(i))
                        temporaryItemObject.setValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)
                    temporaryBridgeObject.addBridgeToDictOfBridgesInState(temporaryStateObject)
            elif i is 1:
                temporaryStateObject = State('AL')
                for k in range(random.randrange(3) + 1):
                    temporaryBridgeObject = Bridge('Bridge' + str(k*10*random.randrange(5)) + str(101*random.randrange(5)))
                    for i in range(3):
                        temporaryItemObject = Item('Item ' + str(i))
                        temporaryItemObject.setValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)
                    temporaryBridgeObject.addBridgeToDictOfBridgesInState(temporaryStateObject)
        elif j is 2:
            if i is 0:
                temporaryStateObject = State('LA')
                for k in range(random.randrange(3) + 1):
                    temporaryBridgeObject = Bridge('Bridge' + str(k*10*random.randrange(5)) + str(101*random.randrange(5)))
                    for i in range(3):
                        temporaryItemObject = Item('Item ' + str(i))
                        temporaryItemObject.setValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)
                    temporaryBridgeObject.addBridgeToDictOfBridgesInState(temporaryStateObject)
            elif i is 1:
                temporaryStateObject = State('WY')
                for k in range(random.randrange(3) + 1):
                    temporaryBridgeObject = Bridge('Bridge' + str(k*10*random.randrange(5)) + str(101*random.randrange(5)))
                    for i in range(3):
                        temporaryItemObject = Item('Item ' + str(i))
                        temporaryItemObject.setValueOfItemInDictOfItemsInBridge(temporaryBridgeObject)
                    temporaryBridgeObject.addBridgeToDictOfBridgesInState(temporaryStateObject)

        temporaryStateObject.addStateToDictOfStatesInYear(temporaryYearObject)

    temporaryYearObject.addYearToDictOfYears()"""

import csv


class AllowableValueListGenerator:
    def __init__(self, MASTERPATH, itemName, index):
        self.itemName = itemName
        self.index = index
        self.allowableValues = ''

        with open(MASTERPATH + "\\BridgeDataQuery\Interface\Labels\parameterValues.txt", "r") as parameterValuesFile:
            omega = 0
            reader = csv.reader(parameterValuesFile)
            for row in reader:
                if omega == self.index:
                    self.allowableValues = row
                omega += 1

    def getAllowableValuesList(self):
        return self.allowableValues


class ListGenerator:
    def __init__(self, parent):
        self.parent = parent


states()
years()

class NBIBridge:
    """def __init__(self, state):
        self.state = state
        self.bridgesInState = {}
        with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\bridgesList.txt",
                  "w") as bridgesList:
            for i in range(random.randrange(3)):
                # if (self.items[item.strip()] == self.query[self.k]) and self.query[self.k] is not '':
                bridgesList.write(self.state + " BRIDGE " + str(random.randrange(5)) + str(random.randrange(5)) + str(
                    random.randrange(5)) + str(random.randrange(5)) + "\n")
                # else:
                # self.items[item.strip()] = 0
                # self.k += 1
                # with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\bridgesList.txt", "r") as bridgesList:
                # print(bridgesList.read())
        with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\GUI\\bridgesList.txt", "r") as itemsList:
            for item in itemsList:
                self.itemsInBridge[item.strip()] = random.randrange(5)"""
    """def __init__(self, state):
        self.state = state
        self.bridgesInState = {}
        with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\bridgesList.txt",
                  "w") as bridgesList:
            for i in range(random.randrange(4)):
                # if (self.items[item.strip()] == self.query[self.k]) and self.query[self.k] is not '':
                bridgesList.write(self.state + " BRIDGE " + str(random.randrange(5)) + str(random.randrange(5)) + str(
                    random.randrange(5)) + str(random.randrange(5)) + "\n")
                # else:
                # self.items[item.strip()] = 0
                # self.k += 1
                # with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\bridgesList.txt", "r") as bridgesList:
                # print(bridgesList.read())
        with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\GUI\\bridgesList.txt", "r") as itemsList:
            for item in itemsList:
                self.itemsInBridge[item.strip()] = random.randrange(5)"""

    def bridgesInState(self, state):
        # self.bridge = bridge
        self.state = state

        with open(
            "C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\bridgesList.txt",
            "w"
        ) as bridgesList:
            for i in range(random.randrange(4) + 1):
                # if (self.items[item.strip()] == self.query[self.k]) and self.query[self.k] is not '':
                bridgesList.write(
                    self.state + " BRIDGE " + str(random.randrange(5)) + str(random.randrange(5)) + str(
                        random.randrange(5)
                    ) + str(
                        random.randrange(
                            5
                        )
                    ) + "\n"
                )  # else:  # self.items[item.strip()] = 0  # self.k += 1  # with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\bridgesList.txt", "r") as bridgesList:  # print(bridgesList.read())

    def __init__(self, query, fileStructure, items):
        self.query = query
        self.fileStructure = fileStructure
        self.items = items
        self.bridgeItems = {}
        omega = 0
        for item in self.fileStructure:
            self.bridgeItems[item.lstrip("0")] = self.items[omega]  # POPULATE THE BRIDGE WITH THE ITEMS FROM THE FILE
            omega += 1

    def getBridgeItems(self):
        return self.bridgeItems  # RETURN THE LIST OF ITEMS ASSOCIATED WITH BRIDGE

    def __init__(self, query, fileStructure, items):
        self.query = query
        self.fileStructure = fileStructure
        self.items = items
        self.bridgeItems = {}
        omega = 0
        for item in self.fileStructure:
            self.bridgeItems[item.lstrip("0")] = self.items[omega]  # POPULATE THE BRIDGE WITH THE ITEMS FROM THE FILE
            omega += 1

    def getBridgeItems(self):
        return self.bridgeItems  # RETURN THE LIST OF ITEMS ASSOCIATED WITH BRIDGE


class NBIBridgeItems:
    def __init__(self, bridge):
        self.bridge = bridge
        self.itemsInBridge = {}
        with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\GUI\parameterNames.txt", "r") as itemsList:
            for item in itemsList:
                self.itemsInBridge[item.strip()] = random.randrange(5)

    """def itemsInBridge(self):
        self.bridge = bridge
        self.itemsInBridge = {}
        #itemsInBridge.clear()
        with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\GUI\parameterNames.txt", "r") as itemsList:
            for item in itemsList:

                #if (self.items[item.strip()] == self.query[self.k]) and self.query[self.k] is not '':
                self.itemsInBridge[item.strip()] = random.randrange(5)
                #else:
                    #self.items[item.strip()] = 0
                #self.k += 1"""


class NBIData:
    def __init__(self, Search):
        self.Search = Search
        self.GUI = self.Search.GUI
        self.util = self.Search.GUI.util
        self.years_to_search = []
        self.found = False

    def __init__(self, query, starting_year, current_year, years_to_iterate, MASTERPATH, listGenerator):
        self.query, self.starting_year, self.current_year, self.years_to_iterate, self.MASTERPATH, self.listGenerator = query, starting_year, current_year, years_to_iterate, MASTERPATH, listGenerator
        self.years_to_search = []
        self.found = False

    def __init__(self):
        self.query = []
        self.found = False
        self.evaluate(self.query)

    def __init__(self):
        self.query = []
        self.found = False
        self.evaluate(self.query)

    def evaluate(self, query):

        self.statesInYear = {}
        self.states = {}
        self.bridgesInfo = {}
        self.yearInfo = {}
        self.query = query
        with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\\temp.txt", "r") as loadFile:
            for line in loadFile:
                self.query.append(line.strip())

        # CHECK IF YEAR IS SPECIFIED

        try:
            if (int(self.query[0]) >= starting_year) and (int(self.query[0]) < 2017):
                years[int(self.query[0]) - starting_year] = int(self.query[0])
                years.clear()
                years.append(int(self.query[0]))
            else:
                print(self.query[0] + " is not a valid year. Search aborted.")
                return
        except:
            pass

        # CHECK IF STATE IS SPECIFIED
        with open(
            "C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\allStates.txt",
            "r"
        ) as allStates:
            with open(
                "C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\statesList.txt",
                "w"
            ) as statesList:
                if self.query[1] is not '':
                    for line in allStates:
                        if self.query[1] == line.strip():
                            statesList.write(self.query[1] + "\n")
                            self.found = True
                    if self.found == False:
                        print(self.query[1] + " is not a valid state. Search aborted.")
                        return
                else:
                    for line in allStates:
                        statesList.write(line.strip() + "\n")

        # CHECK IF BRIDGE IS SPECIFIED
        if self.query[8] is not '':
            with open(
                "C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\bridgesList.txt",
                "w"
            ) as bridgesList:
                bridgesList.write(self.query[8] + "\n")
        else:
            with open(
                "C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\allStateBridges.txt",
                "r"
            ) as allStateBridges:
                with open(
                    "C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\bridgesList.txt",
                    "w"
                ) as bridgesList:
                    for line in allStateBridges:
                        bridgesList.write(line.strip() + "\n")

        # EXTRACT THE MATCHING DATA FROM THE DATABASE

        with open(
            "C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\statesList.txt",
            "r"
        ) as statesList:
            for year in years:  # Each year is a dictionary
                print(year)
                for state in statesList:  # Each key in the year dictionary is a state
                    print(state)
                    x = Bridges_In_State()
                    x.bridgesInState(state.strip())  # Update the list of bridges in the state

                    with open(
                        "C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\bridgesList.txt",
                        "r"
                    ) as bridgesList:

                        for bridge in bridgesList:  # Each bridge contains a dictionary of items
                            itemsInBridge = Items_In_Bridge(bridge)
                            self.bridgesInfo[
                                bridge.strip()] = itemsInBridge.itemsInBridge  # Each value in the item dictionary is the numerical symbol associated with the key
                            print(str(year) + ": " + bridge + ": " + str(itemsInBridge.itemsInBridge))
                    self.states[state.strip()] = self.bridgesInfo

                statesInYear = States_In_Year(year, self.states)
                self.yearInfo[year] = statesInYear.statesInYear

                self.statesInYear = self.yearInfo
                self.year  # print(self.yearInfo)  # print(year)

        # WRITE REPORT
        with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Reports\\report.txt", "w") as report:
            report.write(str(self.statesInYear))
            print("Report has been generated.")

        """#EXTRACT ITEM ID
        for year in self.years:
            for state in self.states:
                for bridge in self.bridges:
                    for item in self.items:
                        self.ExtractItemID(year, state, bridge, item)

    def ExtractItemID(self, year, state, bridge, item):
        self.year = self.years[year]
        self.state = self.states[state] #assign the state in which to search
        self.bridge = self.bridges[bridge]
        self. item = self.items[item]"""

    def evaluate(self, query):

        self.statesInYear = {}
        self.states = {}
        self.bridgesInfo = {}
        self.yearInfo = {}
        self.query = query
        with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\\temp.txt", "r") as loadFile:
            for line in loadFile:
                self.query.append(line.strip())

        # CHECK IF YEAR IS SPECIFIED

        try:
            if (int(self.query[0]) >= starting_year) and (int(self.query[0]) < 2017):
                years[int(self.query[0]) - starting_year] = int(self.query[0])
                years.clear()
                years.append(int(self.query[0]))
            else:
                print(self.query[0] + " is not a valid year. search aborted.")
                return
        except:
            pass

        # CHECK IF STATE IS SPECIFIED
        with open(
            "C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\allStates.txt",
            "r"
        ) as allStates:
            with open(
                "C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\statesList.txt",
                "w"
            ) as statesList:
                if self.query[1] is not '':
                    for line in allStates:
                        if self.query[1] == line.strip():
                            statesList.write(self.query[1] + "\n")
                            self.found = True
                    if self.found == False:
                        print(self.query[1] + " is not a valid state. search aborted.")
                        return
                else:
                    for line in allStates:
                        statesList.write(line.strip() + "\n")

        # CHECK IF BRIDGE IS SPECIFIED
        if self.query[8] is not '':
            with open(
                "C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\bridgesList.txt",
                "w"
            ) as bridgesList:
                bridgesList.write(self.query[8] + "\n")
        else:
            with open(
                "C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\allStateBridges.txt",
                "r"
            ) as allStateBridges:
                with open(
                    "C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\bridgesList.txt",
                    "w"
                ) as bridgesList:
                    for line in allStateBridges:
                        bridgesList.write(line.strip() + "\n")

        # EXTRACT THE MATCHING DATA FROM THE DATABASE

        with open(
            "C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\statesList.txt",
            "r"
        ) as statesList:
            for year in years:  # Each year is a dictionary
                print(year)
                for state in statesList:  # Each key in the year dictionary is a state
                    print(state)
                    x = Bridges_In_State()
                    x.bridgesInState(state.strip())  # Update the list of bridges in the state

                    with open(
                        "C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Database\Information\\bridgesList.txt",
                        "r"
                    ) as bridgesList:

                        for bridge in bridgesList:  # Each bridge contains a dictionary of items
                            itemsInBridge = Items_In_Bridge(bridge)
                            self.bridgesInfo[
                                bridge.strip()] = itemsInBridge.itemsInBridge  # Each value in the item dictionary is the numerical symbol associated with the key
                            print(str(year) + ": " + bridge + ": " + str(itemsInBridge.itemsInBridge))
                    self.states[state.strip()] = self.bridgesInfo

                statesInYear = States_In_Year(year, self.states)
                self.yearInfo[year] = statesInYear.statesInYear

                self.statesInYear = self.yearInfo
                self.year  # print(self.yearInfo)  # print(year)

        # WRITE REPORT
        with open("C:\\Users\\frano\PycharmProjects\\BridgeDataQuery\Reports\\report.txt", "w") as report:
            report.write(str(self.statesInYear))
            print("report_generator0 has been generated.")

        """#EXTRACT ITEM ID
        for year in self.years:
            for state in self.states:
                for bridge in self.bridges:
                    for item in self.items:
                        self.ExtractItemID(year, state, bridge, item)

    def ExtractItemID(self, year, state, bridge, item):
        self.year = self.years[year]
        self.state = self.states[state] #assign the state in which to interface7
        self.bridge = self.bridges[bridge]
        self. item = self.items[item]"""

    def ReduceStates(self, listGenerator):
        allStates = self.listGenerator.getAllStates()
        allStateCodes = self.listGenerator.getAllStateCodes()
        list_of_states_to_search = []
        # list_of_codes_to_search = []

        if self.query['State Code'] is not '':  # CHECK IF STATE WAS SPECIFIED
            if not ("," in list(self.query['State Code'])):  # QUERY IS SINGLE STATE
                for state in allStateCodes:
                    if self.query['State Code'] == allStateCodes[state]:
                        list_of_states_to_search.append(state)
                        # list_of_codes_to_search.append(self.query['State Code'] + "\n")
                        self.found = True
                        break

                if not self.found:
                    print(self.query['State Code'] + " is not a valid State Code. search aborted.")
                    return

            else:  # QUERY IS MORE THAN ONE STATE
                self.states = re.findall(r".?\w+.?", self.query['State Code'])
                for stateSearch in self.states:
                    stripState = stateSearch.strip()
                    stripState = stripState.rstrip(",")
                    omega = 0
                    for state in allStateCodes:
                        if stripState == allStateCodes[state]:
                            list_of_states_to_search.append(allStates[omega])
                            # list_of_codes_to_search.append(stripState + "\n")
                            self.found = True
                            break
                        omega += 1

        else:
            for state in allStateCodes:
                list_of_states_to_search.append(
                    state + "\n"
                )  # list_of_codes_to_search.append(allStateCodes[state] + "\n")

        listGenerator.setListOfStatesToSearch(list_of_states_to_search)

    def ReduceStates(self):
        allStates = self.Search.GUI.util.lists.getAllStates()
        allStateCodes = self.Search.GUI.util.lists.getAllStateCodes()
        list_of_states_to_search = []
        # list_of_codes_to_search = []

        if self.Search.GUI.populator.query['State Code'] is not '':  # CHECK IF STATE WAS SPECIFIED
            if not ("," in list(self.Search.GUI.populator.query['State Code'])):  # QUERY IS SINGLE STATE
                for state in allStateCodes:
                    if self.Search.GUI.populator.query['State Code'] == allStateCodes[state]:
                        list_of_states_to_search.append(state)
                        # list_of_codes_to_search.append(self.query['State Code'] + "\n")
                        self.found = True
                        break

                if not self.found:
                    print(self.Search.GUI.populator.query['State Code'] + " is not a valid State Code. search aborted.")
                    return

            else:  # QUERY IS MORE THAN ONE STATE
                self.states = re.findall(r".?\w+.?", self.Search.GUI.populator.query['State Code'])
                for stateSearch in self.states:
                    stripState = stateSearch.strip()
                    stripState = stripState.rstrip(",")
                    omega = 0
                    for state in allStateCodes:
                        if stripState == allStateCodes[state]:
                            list_of_states_to_search.append(allStates[omega])
                            # list_of_codes_to_search.append(stripState + "\n")
                            self.found = True
                            break
                        omega += 1

        else:
            for state in allStateCodes:
                list_of_states_to_search.append(
                    state + "\n"
                )  # list_of_codes_to_search.append(allStateCodes[state] + "\n")

        self.Search.GUI.util.lists.setListOfStatesToSearch(list_of_states_to_search)

    def ReduceYears(self):
        if (self.query['Year']) is not '':  # CHECK IF YEAR WAS SPECIFIED
            if not ("," in list(self.query['Year'])):
                if (int(self.query['Year']) >= self.starting_year) and (
                    int(self.query['Year']) <= self.current_year):  # DETERMINE WHETHER SPECIFIED YEAR IS VALID
                    self.initial_year_iteration = int(
                        self.query['Year']
                    )  # SET THE STARTING AND ENDING POINTS OF THE SEARCH (YEARS)
                    self.years_to_iterate = self.initial_year_iteration + 1
                    for j in range(self.initial_year_iteration, self.years_to_iterate):
                        self.years_to_search.append(j)
                else:
                    print(self.query['Year'] + " is not a valid year. search aborted.")
                    return
            else:
                self.years = re.findall(r".?\w+.?", self.query['Year'])
                for year in self.years:
                    self.years_to_search.append(year.lstrip("0").strip().rstrip(","))
        else:
            for j in range(self.starting_year, self.current_year + 1):
                self.years_to_search.append(j)

        return self.years_to_search

    def ReduceYears(self):
        if (self.Search.GUI.populator.query['Year']) is not '':  # CHECK IF YEAR WAS SPECIFIED
            if not ("," in list(self.Search.GUI.populator.query['Year'])):
                if (int(self.Search.GUI.populator.query['Year']) >= self.Search.starting_year) and (int(
                    self.Search.GUI.populator.query[
                        'Year']
                ) <= self.Search.current_year):  # DETERMINE WHETHER SPECIFIED YEAR IS VALID
                    self.initial_year_iteration = int(
                        self.Search.GUI.populator.query[
                            'Year']
                    )  # SET THE STARTING AND ENDING POINTS OF THE SEARCH (YEARS)
                    self.years_to_iterate = self.initial_year_iteration + 1
                    for j in range(self.initial_year_iteration, self.years_to_iterate):
                        self.years_to_search.append(j)
                else:
                    print(self.Search.GUI.populator.query['Year'] + " is not a valid year. search aborted.")
                    return
            else:
                self.years = re.findall(r".?\w+.?", self.Search.GUI.populator.query['Year'])
                for year in self.years:
                    self.years_to_search.append(year.lstrip("0").strip().rstrip(","))
        else:
            for j in range(self.Search.starting_year, self.Search.current_year + 1):
                self.years_to_search.append(j)

        return self.years_to_search


class NBIStates:
    def __init__(self, year, states):
        self.year = year
        self.states = states
        self.statesInYear = {}
        self.statesInYear[year] = self.states

    def __init__(self, year, states):
        self.year = year
        self.states = states
        self.statesInYear = {}
        self.statesInYear[year] = self.states

class RadialSearch:
    def __init__(self, latitude, longitude, radius):
        self.centroid = (latitude, longitude)
        self.r = radius

    def interpret_query(self):
        pass

    def create_kml_file(self):
        pass

    def insert_query_results_into_kml_file(self):
        pass


class GeographicSearchMenu(SubMenu):
    def populate(self):

        self.util.name(
            container=self.top, text=GS_ML, coords=((1, 1), (3, 1), (1, 4), (2, 4), (3, 4), (4, 1)),
            columnspan=[1] * 6, sticky=['w'] * 6
        )

        self.util.entry(
            container=self.top, width=[15] * 5, coords=((2, 1), (1, 5), (2, 5), (3, 5), (4, 2)),
            columnspan=[1] * 5, sticky=['we'] * 5
        )
        try:
            for entry in self.util.entries:
                entry.insert(0, '')
            self.util.entries[0].insert(self.parent.filepath)
        except:
            pass

        self.util.separator(container=self.top, orient='h', coords=(6, 1), columnspan=5, sticky='we')
        self.util.button(
            container=self.top, text=GS_BT,
            commands=[lambda: self.util.select_folder(self), self.cancel, self.run,
                      lambda: self.util.save_file(fileExt=KML_FILES, index=4, caller=self)], cursor=HAND,
            coords=((2, 2), (7, 1), (7, 5), (4, 5)), sticky=['e'] * 4
        )
        self.units = self.util.get_var(self.top, 'str', UNITS[0])
        self.util.list_menu(
            container=self.top, variable=self.units, contents=UNITS, command=self.units_selection,
            coords=(3, 2), span=(1, 2), sticky='we'
        )

    def units_selection(self, value):
        self.units.update(value)


class GeographicSearchMenu(SubMenu):
    def run(self):
        try:
            for i in self.util.entries[1:3]:
                int(i.get())
        except:
            print("Enter a value for all parameters.")
            return
        indices, coordinates = gather_coordinates(self.util.entries[0].get())
        indices, coordinates = interpret_query(
            centroid=(int(self.util.entries[2].get()), int(self.util.entries[3].get())), indices=indices,
            coordinates=coordinates, radius=int(self.util.entries[1].get()), units=self.units.get()
        )
        create_kml(indices, coordinates, self.util.entries[4].get())

    def populate(self):
        # label
        self.util.name(
            container=self.top, text=GS_ML, row=[1, 3, 1, 2, 3, 4], column=[1] * 2 + [4] * 3 + [1],
            columnspan=[1] * 6, sticky=['w'] * 6
        )
        # Entries (File Paths, Iterations)
        self.util.entry(
            container=self.top, width=[15] * 5, row=[2, 1, 2, 3, 4], column=[1] + [5] * 3 + [2],
            columnspan=[1] * 5, sticky=['we'] * 5
        )

        try:
            for entry in self.util.entries:
                entry.insert(0, '')
            self.util.entries[0].insert(self.parent.filepath)
        except:
            pass

        # Separators
        self.util.separator(container=self.top, orient='h', row=6, column=1, columnspan=5, sticky='we')
        # Buttons
        self.util.button(
            container=self.top, text=GS_BT, commands=[lambda: select_folder(self), self.cancel, self.run,
                                                      lambda: save_file(self, KML_FILES, 4)], cursor=HAND,
            row=[2, 7, 7, 4], column=[2, 1, 5, 5], sticky=['e'] * 4
        )

        # Create a Tkinter variable to store the item
        self.units = StringVar(self.top)
        # Dictionary with options
        self.units.update(UNITS[0])  # set the default option
        # List
        listMenu1 = ttk.OptionMenu(self.top, self.units, *UNITS, command=self.units_selection)
        listMenu1.grid(row=3, column=2, columnspan=2, sticky='we')

    def units_selection(self, value):
        self.units.update(value)


'''

from __future__ import unicode_literals

import sys

import matplotlib

matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QDial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import datetime

from Utilities.Preferences import MasterPath

current_year = datetime.datetime.now().year - 1
MasterPath_Object = MasterPath.MasterPath()

PROGRAM_NAME = 'Minus Tau'  # os.path.basename(sys.argv[0])


def make_dial(layout, text, stepSize, default, min, max, row, col):
    groupBox = QtWidgets.QGroupBox(text)
    label = QtWidgets.QLabel()
    form = QtWidgets.QFormLayout()
    dial = QDial()
    dial.setMaximumSize(50, 50)
    dial.setNotchesVisible(True)
    dial.setSingleStep(stepSize)
    dial.setValue(default)
    dial.setMinimum(min)
    dial.setMaximum(max)
    dial.setWrapping(False)
    label.setText(str(default))
    form.addWidget(dial)
    form.addWidget(label)
    groupBox.setLayout(form)
    layout.addWidget(groupBox, row, col)
    return dial, label


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=10, dpi=1000, layout=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.ax.grid()
        self.draw()


class DynamicMplCanvas(MplCanvas):
    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)
        self.samplingRate = 200
        self.nyquistFrequency = self.samplingRate / 2
        self.layout = kwargs['layout']
        self.x = np.array([0])
        self.y = np.array([0])
        # self.kalmanSignal = np.array([0])
        # self.kalmanP = 1
        self.set_controls()
        # timer = QtCore.QTimer(self)
        # timer.timeout.connect(self.update_figure)
        # timer.start(0)
        self.oldSize = self.ax.bbox.width, self.ax.bbox.height
        self.axBackground = self.copy_from_bbox(self.ax.bbox)
        self.cnt = 0
        # self.plot0 = self.ax.plot0(self.x, self.y, animated=True)[0]
        self.filteredPlot = self.ax.plot(self.x, self.y, animated=True)[0]
        self.kalmanPlot = self.ax.plot(self.x, self.y, animated=True)[0]
        self.draw()
        self.startTimer(10)

    def set_controls(self):
        self.filterGroupBox = QtWidgets.QGroupBox('Frequency Filter')
        self.filterGrid = QtWidgets.QGridLayout()
        self.kalmanGroupBox = QtWidgets.QGroupBox('Kalman Filter')
        self.kalmanGrid = QtWidgets.QGridLayout()

        self.highpassFrequencyDial, self.highpassFrequencyLabel = make_dial(
            self.filterGrid, 'Highpass Frequency',
            stepSize=1, default=0, min=0,
            max=int(self.nyquistFrequency / 2), row=0,
            col=0
        )

        self.notchFrequencyDial, self.notchFrequencyLabel = make_dial(
            self.filterGrid, 'Notch Frequency', stepSize=1,
            default=0, min=0, max=int(self.nyquistFrequency),
            row=0, col=1
        )

        self.lowpassFrequencyDial, self.lowpassFrequencyLabel = make_dial(
            self.filterGrid, 'Lowpass Frequency',
            stepSize=1,
            default=int(self.nyquistFrequency),
            min=int(self.nyquistFrequency / 2),
            max=int(self.nyquistFrequency), row=0, col=2
        )

        self.kalmanQDial, self.kalmanQLabel = make_dial(
            self.kalmanGrid, 'Q', stepSize=1, default=5, min=0, max=1000,
            row=0, col=0
        )  # 0.5
        self.kalmanRDial, self.kalmanRLabel = make_dial(
            self.kalmanGrid, 'R', stepSize=1, default=50, min=0, max=10000,
            row=0, col=1
        )  # 5

        self.filterGroupBox.setLayout(self.filterGrid)
        self.layout.addWidget(self.filterGroupBox, 1, 0)

        self.kalmanGroupBox.setLayout(self.kalmanGrid)
        self.layout.addWidget(self.kalmanGroupBox, 2, 0)

    def scroll(self):
        if len(self.x) > self.samplingRate:
            self.x = self.x[1:]
            # self.kalmanSignal = self.kalmanSignal[1:]
            self.y = self.y[1:]

    def get_signal(self, frequencies, amplitudes):
        y = 0
        for frequency, amplitude in zip(frequencies, amplitudes):
            y += amplitude * np.sin(2 * np.pi * frequency * self.x)
        return y

    def update_dial_labels(self):
        self.highpassFrequencyLabel.setText(str(self.highpassFrequencyDial.value()))
        self.notchFrequencyLabel.setText(str(self.notchFrequencyDial.value()))
        self.lowpassFrequencyLabel.setText(str(self.lowpassFrequencyDial.value()))
        self.kalmanQLabel.setText(str(self.kalmanQDial.value() / 10))
        self.kalmanRLabel.setText(str(self.kalmanRDial.value() / 10))

    def timerEvent(self, evt):
        self.x = np.append(self.x, [self.x[-1] + 1 / self.samplingRate])
        self.y = self.get_signal(frequencies=[20], amplitudes=[1])

        self.scroll()
        self.update_dial_labels()
        currentSize = self.ax.bbox.width, self.ax.bbox.height
        if self.oldSize != currentSize:
            self.oldSize = currentSize
            self.ax.clear()
            self.ax.grid()
            self.draw()
            self.axBackground = self.copy_from_bbox(self.ax.bbox)

        self.restore_region(self.axBackground)
        # self.plot0.set_xdata(self.x)
        # self.plot0.set_ydata(self.y)
        # self.ax.draw_artist(self.plot0)

        y1 = highpass_filter(self.y, self.samplingRate, self.highpassFrequencyDial.value(), )
        y1 = notch_filter(y1, self.samplingRate, self.notchFrequencyDial.value(), )

        y1 = lowpass_filter(y1, self.samplingRate, self.lowpassFrequencyDial.value(), )
        self.filteredY = self.y[:int(self.samplingRate / 6) - 1]
        self.filteredY = np.append(self.filteredY, y1[len(self.filteredY):])
        self.filteredPlot.set_xdata(self.x)
        self.filteredPlot.set_ydata(self.filteredY)
        self.ax.draw_artist(self.filteredPlot)
        self.blit(self.ax.bbox)
        self.draw()

        '''self.raw, self.kalmanSignal, self.kalmanP = Filter.kalman_filter(
            self.raw, y,
            p0=self.kalmanP,
            q=self.kalmanQDial.value()/10,
            r=self.kalmanRDial.value()/10)'''
        # self.ax.plot0(self.x, y)
        # self.ax.plot0(self.x, y0)
        # self.ax.plot0(self.x, self.y)
        # self.ax.plot0(self.x, self.kalmanSignal)
        dx = max(self.x) - min(self.x)
        self.ax.set_xlim(min(self.x) + dx / 6, max(self.x))
        self.ax.set_ylim(min(self.y), max(self.y))  # self.draw()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle(PROGRAM_NAME)
        self.mainWidget = QtWidgets.QWidget(self)
        self.lay = QtWidgets.QGridLayout(self.mainWidget)
        self.canvas = DynamicMplCanvas(self.mainWidget, width=5, height=4, dpi=100, layout=self.lay)
        self.lay.addWidget(self.canvas, 0, 0)
        self.mainWidget.setFocus()
        self.setCentralWidget(self.mainWidget)

    def quit(self):
        self.close()


qApp = QtWidgets.QApplication(sys.argv)
window = ApplicationWindow()
window.setWindowTitle("%s" % PROGRAM_NAME)
window.show()
sys.exit(qApp.exec_())
# qApp.exec_()

import datetime

from Utilities.Preferences import MasterPath

number_of_items = 141
current_year = datetime.datetime.now().year - 1
MasterPath_Object = MasterPath.MasterPath()
MASTERPATH = MasterPath_Object.getMasterPath()


class BridgeDataQuery:
    def __init__(self, master):
        self.master = master
        master.title("Christian's Machine")
        end = 4

        self.frame = VerticalScrolledFrame.VerticalScrolledFrame(master)
        self.frame.grid(column=end)
        global listGenerator
        listGenerator = ListGenerator.ListGenerator(MASTERPATH, self.frame, number_of_items)

        menubar = Menu(master)

        # create a pulldown menu, and add it to the menu bar
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open search", command=self.load)
        filemenu.add_command(label="Save search As", command=self.save)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.destroy)
        menubar.add_cascade(label="File", menu=filemenu)

        # PREFERENCES OPTIONS
        preferences = Menu(menubar, tearoff=0)
        preferences.add_command(label="report_generator0 Options", command=self.report_settings)
        preferences.add_command(label="Main5 Path", command=self.main_path)
        menubar.add_cascade(label="Preferences", menu=preferences)

        # MACHINE LEARNING OPTIONS
        machine_learning = Menu(menubar, tearoff=0)
        machine_learning_tools = Menu(menubar, tearoff=0)
        machine_learning_tools_generate = Menu(menubar, tearoff=0)
        machine_learning_settings = Menu(menubar, tearoff=0)

        machine_learning_settings.add_command(label="Markov Chain", command=self.markov_chain_options)
        # machine_learning_settings.add_command(label="Neural Network", command=self.machine_options)
        # achine_learning_settings.add_command(label="Genetic Algorithm", command=self.machine_options)
        # machine_learning_settings.add_command(label="Support Vector Machine", command=self.machine_options)

        machine_learning_tools_generate.add_command(label="Random Vector", command=self.generate_vector)
        machine_learning_tools_generate.add_command(label="Random Matrix", command=self.generate_matrix)

        machine_learning.add_cascade(label="Settings", menu=machine_learning_settings)
        machine_learning_tools.add_cascade(label="Generate", menu=machine_learning_tools_generate)
        machine_learning.add_cascade(label="Tools", menu=machine_learning_tools)
        menubar.add_cascade(label="Machine Learning", menu=machine_learning)

        # PLOTTER OPTIONS
        plotter_settings = Menu(menubar, tearoff=0)
        plotter_settings.add_command(label="Create Plot", command=self.create_plot)
        plotter_settings.add_command(label="Plotter Settings", command=self.plotter_settings)
        menubar.add_cascade(label="Plotter", menu=plotter_settings)

        # HELP MENU
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.load)
        menubar.add_cascade(label="Help2", menu=helpmenu)

        # display the menu
        master.config(menu=menubar)

        # DECLARING THE LISTS AND DICTIONARIES THAT WILL HOLD INFORMATION
        self.parameterLabel, self.parameterLabel2 = [], []  # This list will hold the labels
        self.label_text_List, self.label_text_List2 = [], []  # This list will hold the text which will go on each label
        self.parameterNames = {}  # This will hold the name of each parameter
        self.parameterNumbers = {}  # This will hold the name of each parameter
        self.inputValidationInformation = {}  # This will hold a dictionary in which keys are item name and values are lists of allowable digits
        self.entryList = {}  # This will hold the name of each entry text field as well as the field itself
        self.entries = []
        self.query = {}
        # POPULATING PARAMETER NUMBERS
        self.parameterNumbers = listGenerator.getParameterNumbers()

        # POPULATING THE DICTIONARY WHICH WILL HOLD THE ENTRY TEXT FIELDS AND GENERATING THE PARAMETER NAMES LIST
        self.parameterNames = listGenerator.getParameterNames()
        self.entryList = listGenerator.getEntryList()
        self.files = 'single'
        self.filename = 'report'

        # POPULATING THE DICTIONARY WHICH WILL HOLD THE VALIDATION INFORMATION
        for i in range(number_of_items):
            temporaryAllowableValuesObject = AllowableValueListGenerator.AllowableValueListGenerator(
                MASTERPATH,
                self.parameterNames[
                    i], i
            )
            self.inputValidationInformation[self.parameterNumbers[
                i]] = temporaryAllowableValuesObject.getAllowableValuesList()  # "LIST OF ALLOWABLE values HERE"
            # SETTING THE ENTRY TEXT FIELDS
            if self.parameterNames[i + 0] != '(Reserved)':
                self.entry = self.entryList[self.parameterNames[i + 0]]
                self.entries.append(self.entry)
                self.entry.grid(row=i + 3, column=3, columnspan=1)
                self.entryList[self.parameterNames[i]].insert(0, '')
            # POPULATING THE LISTS WHICH WILL HOLD THE LABELS, LABEL TEXT, AND TEXT ENTRY FIELDS
            self.label_text_List.append(StringVar())
            self.parameterLabel.append(Label(self.frame.interior, textvariable=self.label_text_List[i]))
            self.label_text_List2.append(StringVar())
            self.parameterLabel2.append(Label(self.frame.interior, textvariable=self.label_text_List2[i]))
            # SETTING THE LABELS AND TEXT ENTRY FIELDS IN THE MENU
            self.parameterName = self.parameterNames[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel[i]  # Set the parameter label from the parameter label list
            self.label_text = self.label_text_List[i]
            self.label_text.set(self.parameterName)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 3, column=1, columnspan=1, sticky=W)  # Set the label location
            # SETTING THE LABELS OF ITEM NUMBER
            self.parameterNumber = self.parameterNumbers[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel2[i]  # Set the parameter label from the parameter label list
            self.label_text2 = self.label_text_List2[i]
            self.label_text2.set(self.parameterNumber)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 3, column=0, columnspan=1, sticky=W)  # Set the label location

        # SEPARATOR
        separator = ttk.Separator(self.master, orient=HORIZONTAL)
        separator.grid(row=1, column=0, columnspan=5, sticky=W + E)

        # SETTING THE BUTTONS
        self.search_button = Button(master, text="search", command=self.search, cursor='hand2')
        self.search_button.grid(row=2, column=4, sticky=E)
        self.clear_button = Button(master, text="Clear", command=self.clear, cursor='hand2')
        self.clear_button.grid(row=2, column=0, sticky=W)

        master.mainloop()

    def save(self):
        path = tkf.asksaveasfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            if path[path.__len__() - 4] is not ".":
                path = path + ".txt"

            with open(path, "w") as saveFile:
                for i in range(number_of_items):
                    saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")
        except:
            return

    def load(self):
        path = tkf.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            with open(path, "r") as loadFile:
                for i in range(number_of_items):
                    self.entryList[self.parameterNames[i + 0]].delete(0, END)
                i = 0
                for line in loadFile:
                    self.entryList[self.parameterNames[i]].insert(i, line.strip())
                    i += 1
        except:
            return

    def search(self):
        self.name_report()

    def search_resume(self):
        with open(MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "w") as saveFile:
            for i in range(number_of_items):
                saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")

        with open(MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "r") as loadFile:
            i = 0
            for line in loadFile:
                if str(self.parameterNames[i]) == '(Reserved)':
                    self.query[self.parameterNames[i] + "_" + str(i)] = (line.strip())
                else:
                    self.query[self.parameterNames[i]] = (line.strip())
                i += 1

        if self.dataValidator():
            Search.InitiateSearch(
                MASTERPATH, self.parameterNames, self.parameterNumbers, self.query, listGenerator,
                self.files, self.filename
            )


class CreateWindow:
    def __init__(self, window):
        self.root = Tk()
        self.setMaster(self.root)
        self.set_gui(window)
        self.run()

    def set_gui(self, window):
        if window == 0:
            self.my_gui = BridgeDataQuery(self.root)

    def setMaster(self, master):
        # global root
        self.root = master

    def getMaster(self):
        return self.root

    def run(self):
        self.root.mainloop()


CreateWindow = CreateWindow(0)

import datetime

from Utilities.Preferences import MasterPath

number_of_items = 141
current_year = datetime.datetime.now().year - 1
MasterPath_Object = MasterPath.MasterPath()
MASTERPATH = MasterPath_Object.getMasterPath()


class BridgeDataQuery:
    def load(self):
        path = tkf.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            with open(path, "r") as loadFile:
                for i in range(number_of_items):
                    self.entryList[self.parameterNames[i + 0]].delete(0, END)
                i = 0
                for line in loadFile:
                    self.entryList[self.parameterNames[i]].insert(i, line.strip())
                    i += 1
        except:
            return

    def search(self):
        self.name_report()

    def search_resume(self):
        with open(MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "w") as saveFile:
            for i in range(number_of_items):
                saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")

        with open(MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "r") as loadFile:
            i = 0
            for line in loadFile:
                if str(self.parameterNames[i]) == '(Reserved)':
                    self.query[self.parameterNames[i] + "_" + str(i)] = (line.strip())
                else:
                    self.query[self.parameterNames[i]] = (line.strip())
                i += 1

        if self.dataValidator():
            Search.InitiateSearch(
                MASTERPATH, self.parameterNames, self.parameterNumbers, self.query, listGenerator,
                self.files, self.filename
            )


CreateWindow = CreateWindow(0)

import datetime

from lozoya.compute.text_parser import InitiateSearch
from Utilities import MasterPath

number_of_items = 141
current_year = datetime.datetime.now().year - 1
MasterPath_Object = MasterPath.MasterPath()
MASTERPATH = MasterPath_Object.getMasterPath()


class BridgeDataQuery:
    def load(self):
        path = tkf.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            with open(path, "r") as loadFile:
                for i in range(number_of_items):
                    self.entryList[self.parameterNames[i + 0]].delete(0, END)
                i = 0
                for line in loadFile:
                    self.entryList[self.parameterNames[i]].insert(i, line.strip())
                    i += 1
        except:
            return

    def search(self):
        self.name_report()

    def search_resume(self):
        with open(MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "w") as saveFile:
            for i in range(number_of_items):
                saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")

        with open(MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "r") as loadFile:
            i = 0
            for line in loadFile:
                if str(self.parameterNames[i]) == '(Reserved)':
                    self.query[self.parameterNames[i] + "_" + str(i)] = (line.strip())
                else:
                    self.query[self.parameterNames[i]] = (line.strip())
                i += 1

        if self.dataValidator():
            InitiateSearch.InitiateSearch(
                MASTERPATH, self.parameterNames, self.parameterNumbers, self.query,
                listGenerator, self.files, self.filename
            )

    def clear(self):
        for i in range(number_of_items):
            self.entryList[self.parameterNames[i + 0]].delete(0, END)

    def dataValidator(self):
        for i in range(number_of_items):
            if self.parameterNames[i] == '(Reserved)':
                pass
            else:
                if self.query[self.parameterNames[i]] != '':
                    if 'None' not in self.inputValidationInformation[self.parameterNumbers[i]]:
                        if True:  # self.query[self.parameterNames[i]] in self.inputValidationInformation[self.parameterNumbers[i]]:
                            pass
                        else:
                            print(
                                str(self.query[self.parameterNames[i]]) + " is not a valid " + str(
                                    self.parameterNames[i]
                                )
                            )  # + ". search Aborted.")
                            pass
        return True

    def name_report(self):
        NameReportsMenu = NameReport.NameReport(self.master, MASTERPATH, self)

    def report_settings(self):
        ReportSettingsMenu = ReportSettings.ReportSettings(self.master, MASTERPATH)

    def machine_options(self):
        MachineOptionMenu = MarkovChainMenu.MachineLearningSettings(self.master)

    def markov_chain_options(self):
        MarkovChainSettingsMenu = MarkovChainSettings.MarkovChainSettings(self.master, MASTERPATH)

    def create_plot(self):
        CreatePlotMenu = PlotterSettings.Plot(self.master, MASTERPATH)

    def plotter_settings(self):
        PlotterSettingsMenu = PlotterSettings.PlotterSettings(self.master, MASTERPATH)

    def main_path(self):
        MainPathMenu = MainPathSettings.MainPathSettings(self.master, MASTERPATH)

    def generate_vector(self):
        GenerateSettingsMenu = GeneratorSettings.SettingsGenerator(self.master, MASTERPATH, 0)

    def generate_matrix(self):
        GenerateSettingsMenu = GeneratorSettings.SettingsGenerator(self.master, MASTERPATH, 1)

    def set_filename(self, filename):
        self.filename = filename


CreateWindow = CreateWindow(0)

import datetime

from Utilities import MasterPath

number_of_items = 141
current_year = datetime.datetime.now().year - 1
MasterPath_Object = MasterPath.MasterPath()
MASTERPATH = MasterPath_Object.getMasterPath()


class BridgeDataQuery:
    def __init__(self, master):
        self.master = master
        master.title("Christian's Machine")
        end = 4
        self.frame = VerticalScrolledFrame.VerticalScrolledFrame(master)
        self.frame.grid(column=end)
        global listGenerator
        listGenerator = ListGenerator.ListGenerator(MASTERPATH, self.frame, number_of_items)
        menubar = Menu(master)
        # create a pulldown menu, and add it to the menu bar
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open search", command=self.load)
        filemenu.add_command(label="Save search As", command=self.save)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        # PREFERENCES OPTIONS
        preferences = Menu(menubar, tearoff=0)
        preferences.add_command(label="report_generator0 Options", command=self.report_settings)
        preferences.add_command(label="Main5 Path", command=self.main_path)
        menubar.add_cascade(label="Preferences", menu=preferences)
        # MACHINE LEARNING OPTIONS
        machine_learning = Menu(menubar, tearoff=0)
        machine_learning_tools = Menu(menubar, tearoff=0)
        machine_learning_tools_generate = Menu(menubar, tearoff=0)
        machine_learning_settings = Menu(menubar, tearoff=0)
        machine_learning_settings.add_command(label="Markov Chain", command=self.markov_chain_options)
        # machine_learning_settings.add_command(label="Neural Network", command=self.machine_options)
        # achine_learning_settings.add_command(label="Genetic Algorithm", command=self.machine_options)
        # machine_learning_settings.add_command(label="Support Vector Machine", command=self.machine_options)
        machine_learning_tools_generate.add_command(label="Random Vector", command=self.generate_vector)
        machine_learning_tools_generate.add_command(label="Random Matrix", command=self.generate_matrix)
        machine_learning.add_cascade(label="Settings", menu=machine_learning_settings)
        machine_learning_tools.add_cascade(label="Generate", menu=machine_learning_tools_generate)
        machine_learning.add_cascade(label="Tools", menu=machine_learning_tools)
        menubar.add_cascade(label="Machine Learning", menu=machine_learning)
        # PLOTTER OPTIONS
        plotter_settings = Menu(menubar, tearoff=0)
        plotter_settings.add_command(label="Create Plot", command=self.create_plot)
        plotter_settings.add_command(label="Plotter Settings", command=self.plotter_settings)
        menubar.add_cascade(label="Plotter", menu=plotter_settings)
        # HELP MENU
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.load)
        menubar.add_cascade(label="Help2", menu=helpmenu)
        # display the menu
        master.config(menu=menubar)
        # DECLARING THE LISTS AND DICTIONARIES THAT WILL HOLD INFORMATION
        self.parameterLabel, self.parameterLabel2 = [], []  # This list will hold the labels
        self.label_text_List, self.label_text_List2 = [], []  # This list will hold the text which will go on each label
        self.parameterNames = {}  # This will hold the name of each parameter
        self.parameterNumbers = {}  # This will hold the name of each parameter
        self.inputValidationInformation = {}  # This will hold a dictionary in which keys are item name and values are lists of allowable digits
        self.entryList = {}  # This will hold the name of each entry text field as well as the field itself
        self.entries = []
        self.query = {}
        # POPULATING PARAMETER NUMBERS
        self.parameterNumbers = listGenerator.getParameterNumbers()
        # POPULATING THE DICTIONARY WHICH WILL HOLD THE ENTRY TEXT FIELDS AND GENERATING THE PARAMETER NAMES LIST
        self.parameterNames = listGenerator.getParameterNames()
        self.entryList = listGenerator.getEntryList()
        self.files = 'single'
        self.filename = 'report'
        # POPULATING THE DICTIONARY WHICH WILL HOLD THE VALIDATION INFORMATION
        for i in range(number_of_items):
            temporaryAllowableValuesObject = AllowableValueListGenerator.AllowableValueListGenerator(
                MASTERPATH,
                self.parameterNames[
                    i], i
            )
            self.inputValidationInformation[self.parameterNumbers[
                i]] = temporaryAllowableValuesObject.getAllowableValuesList()  # "LIST OF ALLOWABLE values HERE"
            # SETTING THE ENTRY TEXT FIELDS
            if self.parameterNames[i + 0] != '(Reserved)':
                self.entry = self.entryList[self.parameterNames[i + 0]]
                self.entries.append(self.entry)
                self.entry.grid(row=i + 3, column=3, columnspan=1)
                self.entryList[self.parameterNames[i]].insert(0, '')
            # POPULATING THE LISTS WHICH WILL HOLD THE LABELS, LABEL TEXT, AND TEXT ENTRY FIELDS
            self.label_text_List.append(StringVar())
            self.parameterLabel.append(Label(self.frame.interior, textvariable=self.label_text_List[i]))
            self.label_text_List2.append(StringVar())
            self.parameterLabel2.append(Label(self.frame.interior, textvariable=self.label_text_List2[i]))
            # SETTING THE LABELS AND TEXT ENTRY FIELDS IN THE MENU
            self.parameterName = self.parameterNames[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel[i]  # Set the parameter label from the parameter label list
            self.label_text = self.label_text_List[i]
            self.label_text.set(self.parameterName)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 3, column=1, columnspan=1, sticky=W)  # Set the label location
            # SETTING THE LABELS OF ITEM NUMBER
            self.parameterNumber = self.parameterNumbers[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel2[i]  # Set the parameter label from the parameter label list
            self.label_text2 = self.label_text_List2[i]
            self.label_text2.set(self.parameterNumber)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 3, column=0, columnspan=1, sticky=W)  # Set the label location
        # SEPARATOR
        separator = ttk.Separator(self.master, orient=HORIZONTAL)
        separator.grid(row=1, column=0, columnspan=5, sticky=W + E)
        # SETTING THE BUTTONS
        self.search_button = Button(master, text="search", command=self.search, cursor='hand2')
        self.search_button.grid(row=2, column=4, sticky=E)
        self.clear_button = Button(master, text="Clear", command=self.clear, cursor='hand2')
        self.clear_button.grid(row=2, column=0, sticky=W)
        master.mainloop()

    def save(self):
        path = tkf.asksaveasfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            if path[path.__len__() - 4] is not ".":
                path = path + ".txt"

            with open(path, "w") as saveFile:
                for i in range(number_of_items):
                    saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")
        except:
            return

    def load(self):
        path = tkf.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            with open(path, "r") as loadFile:
                for i in range(number_of_items):
                    self.entryList[self.parameterNames[i + 0]].delete(0, END)
                i = 0
                for line in loadFile:
                    self.entryList[self.parameterNames[i]].insert(i, line.strip())
                    i += 1
        except:
            return

    def search(self):
        self.name_report()

    def search_resume(self):
        with open(MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "w") as saveFile:
            for i in range(number_of_items):
                saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")

        with open(MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "r") as loadFile:
            i = 0
            for line in loadFile:
                if str(self.parameterNames[i]) == '(Reserved)':
                    self.query[self.parameterNames[i] + "_" + str(i)] = (line.strip())
                else:
                    self.query[self.parameterNames[i]] = (line.strip())
                i += 1

        if self.dataValidator():
            Search.InitiateSearch(
                MASTERPATH, self.parameterNames, self.parameterNumbers, self.query, listGenerator,
                self.files, self.filename
            )

    def clear(self):
        for i in range(number_of_items):
            self.entryList[self.parameterNames[i + 0]].delete(0, END)

    def dataValidator(self):
        for i in range(number_of_items):
            if self.parameterNames[i] == '(Reserved)':
                pass
            else:
                if self.query[self.parameterNames[i]] != '':
                    if 'None' not in self.inputValidationInformation[self.parameterNumbers[i]]:
                        if True:  # self.query[self.parameterNames[i]] in self.inputValidationInformation[self.parameterNumbers[i]]:
                            pass
                        else:
                            print(
                                str(self.query[self.parameterNames[i]]) + " is not a valid " + str(
                                    self.parameterNames[i]
                                )
                            )  # + ". search Aborted.")
                            pass
        return True

    def name_report(self):
        NameReportsMenu = NameReport.NameReport(self.master, MASTERPATH, self)

    def report_settings(self):
        ReportSettingsMenu = ReportSettings.ReportSettings(self.master, MASTERPATH)

    def machine_options(self):
        MachineOptionMenu = MarkovChainMenu.MachineLearningSettings(self.master)

    def markov_chain_options(self):
        MarkovChainSettingsMenu = MarkovChainSettings.MarkovChainSettings(self.master, MASTERPATH)

    def create_plot(self):
        CreatePlotMenu = PlotterSettings.Plot(self.master, MASTERPATH)

    def plotter_settings(self):
        PlotterSettingsMenu = PlotterSettings.PlotterSettings(self.master, MASTERPATH)

    def main_path(self):
        MainPathMenu = MainPathSettings.MainPathSettings(self.master, MASTERPATH)

    def generate_vector(self):
        GenerateSettingsMenu = GeneratorSettings.SettingsGenerator(self.master, MASTERPATH, 0)

    def generate_matrix(self):
        GenerateSettingsMenu = GeneratorSettings.SettingsGenerator(self.master, MASTERPATH, 1)

    def set_filename(self, filename):
        self.filename = filename


CreateWindow = CreateWindow(0)

import datetime

from Utilities.Preferences import MasterPath

current_year = datetime.datetime.now().year - 1
MasterPath_Object = MasterPath.MasterPath()


class BridgeDataQuery:
    def __init__(self, master):
        self.master = master

        # TODO FIX SCROLLBAR
        self.frame = VerticalScrolledFrame.VerticalScrolledFrame(master)
        self.frame.grid(column=4, sticky=NSEW)

        self.MASTERPATH = MasterPath_Object.getMasterPath()
        self.number_of_items = 142
        self.list_generator = ListGenerator.ListGenerator(self)
        menubar = Menu(master)

        # create a pulldown menu, and add it to the menu bar
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open search", command=self.load)
        filemenu.add_command(label="Save search As", command=self.save)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.destroy)
        menubar.add_cascade(label="File", menu=filemenu)

        # PREFERENCES OPTIONS
        preferences = Menu(menubar, tearoff=0)
        preferences.add_command(label="Plotter Settings", command=self.plotter_settings)
        preferences.add_command(label="Markov Chain Settings", command=self.markov_chain_settings)
        preferences.add_command(label="report_generator0 Settings", command=self.report_settings)
        preferences.add_command(label="Main5 Path", command=self.main_path)
        menubar.add_cascade(label="Preferences", menu=preferences)

        # MACHINE LEARNING OPTIONS
        machine_learning = Menu(menubar, tearoff=0)
        machine_learning_tools = Menu(menubar, tearoff=0)
        machine_learning_tools_generate = Menu(menubar, tearoff=0)
        machine_learning_settings = Menu(menubar, tearoff=0)

        machine_learning_tools_generate.add_command(label="Initial State", command=self.generate_vector)
        machine_learning_tools_generate.add_command(label="Transition Matrix", command=self.generate_matrix)

        machine_learning.add_command(label="Markov Chain", command=self.markov_chain_menu)
        machine_learning_tools.add_cascade(label="Generate Random", menu=machine_learning_tools_generate)
        machine_learning.add_cascade(label="Tools", menu=machine_learning_tools)
        menubar.add_cascade(label="Machine Learning", menu=machine_learning)

        # PLOTTER OPTIONS
        plotter_settings = Menu(menubar, tearoff=0)
        plotter_settings.add_command(label="Create Plot", command=self.create_plot)
        menubar.add_cascade(label="Plotter", menu=plotter_settings)

        # HELP MENU
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.load)
        menubar.add_cascade(label="Help2", menu=helpmenu)

        # display the menu
        master.config(menu=menubar)

        # DECLARING THE LISTS AND DICTIONARIES THAT WILL HOLD INFORMATION
        self.parameterLabel, self.parameterLabel2 = [], []  # This list will hold the labels
        self.label_text_List, self.label_text_List2 = [], []  # This list will hold the text which will go on each label
        self.parameterNames, self.parameterNumbers = {}, {}  # This will hold the name/number of each parameter
        self.inputValidationInformation = {}  # This will hold a dictionary in which keys are item name and values are lists of allowable digits
        self.entryList = {}  # This will hold the name of each entry text field as well as the field itself
        self.entries = []
        self.query = {}
        # POPULATING PARAMETER NUMBERS
        self.parameterNumbers = self.list_generator.getParameterNumbers()

        # POPULATING THE DICTIONARY WHICH WILL HOLD THE ENTRY TEXT FIELDS AND GENERATING THE PARAMETER NAMES LIST
        self.parameterNames = self.list_generator.getParameterNames()

        self.entryList = self.list_generator.getEntryList()
        self.files = 'single'
        self.filename = 'report'

        # POPULATING THE DICTIONARY WHICH WILL HOLD THE VALIDATION INFORMATION
        for i in range(self.number_of_items):
            temporaryAllowableValuesObject = AllowableValueListGenerator.AllowableValueListGenerator(
                self.MASTERPATH,
                self.parameterNames[
                    i], i
            )
            self.inputValidationInformation[self.parameterNumbers[
                i]] = temporaryAllowableValuesObject.getAllowableValuesList()  # "LIST OF ALLOWABLE values HERE"
            # SETTING THE ENTRY TEXT FIELDS
            if self.parameterNames[i + 0] != '(Reserved)':
                self.entry = self.entryList[self.parameterNames[i + 0]]
                self.entries.append(self.entry)
                self.entry.grid(row=i + 3, column=3, columnspan=1)
                self.entryList[self.parameterNames[i]].insert(0, '')
            # POPULATING THE LISTS WHICH WILL HOLD THE LABELS, LABEL TEXT, AND TEXT ENTRY FIELDS
            self.label_text_List.append(StringVar())
            self.parameterLabel.append(Label(self.frame.interior, textvariable=self.label_text_List[i]))
            self.label_text_List2.append(StringVar())
            self.parameterLabel2.append(Label(self.frame.interior, textvariable=self.label_text_List2[i]))
            # SETTING THE LABELS AND TEXT ENTRY FIELDS IN THE MENU
            self.parameterName = self.parameterNames[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel[i]  # Set the parameter label from the parameter label list
            self.label_text = self.label_text_List[i]
            self.label_text.set(self.parameterName)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 3, column=1, columnspan=1, sticky=W)  # Set the label location
            # SETTING THE LABELS OF ITEM NUMBER
            self.parameterNumber = self.parameterNumbers[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel2[i]  # Set the parameter label from the parameter label list
            self.label_text2 = self.label_text_List2[i]
            self.label_text2.set(self.parameterNumber)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 3, column=0, columnspan=1, sticky=W)  # Set the label location

        # SEPARATOR
        separator = ttk.Separator(self.master, orient=HORIZONTAL)
        separator.grid(row=1, column=0, columnspan=5, sticky=W + E)

        # SETTING THE BUTTONS
        self.search_button = Button(master, text="search", command=self.search, cursor='hand2')
        self.search_button.grid(row=2, column=4, sticky=SE)
        self.clear_button = Button(master, text="Clear", command=self.clear, cursor='hand2')
        self.clear_button.grid(row=2, column=0, sticky=SW)

        # self.list_generator.getParameterNamesWithRespectToNumber()

        # self.master.mainloop()


class CreateWindow:
    def __init__(self, Main, window):
        self.Main = Main
        self.Main.master.title("Christian's Machine")
        # self.root = Tk()
        # self.Main5.master.iconbitmap('C:\\Users\\frano\PycharmProjects\BridgeDataQuery\interface7\InterfaceUtilities0\icon2\i.ico')
        self.setMaster(self.Main.master)
        self.set_gui(window)
        self.run()

    def set_gui(self, window):
        if window == 0:
            self.my_gui = BridgeDataQuery(self.Main.master)

    def setMaster(self, master):
        # global root
        self.root = master

    def getMaster(self):
        return self.root

    def run(self):
        self.Main.master.mainloop()


class BridgeDataQuery:
    def __init__(self, master):
        self.master = master
        # TODO FIX SCROLLBAR
        self.frame = VerticalScrolledFrame.VerticalScrolledFrame(master)
        self.frame.grid(column=4, sticky=NSEW)
        self.MASTERPATH = MasterPath_Object.getMasterPath()
        self.number_of_items = 142
        self.list_generator = ListGenerator.ListGenerator(self)
        menubar = Menu(master)
        # create a pulldown menu, and add it to the menu bar
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open search", command=self.load)
        filemenu.add_command(label="Save search As", command=self.save)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        # PREFERENCES OPTIONS
        preferences = Menu(menubar, tearoff=0)
        preferences.add_command(label="Plotter Settings", command=self.plotter_settings)
        preferences.add_command(label="Markov Chain Settings", command=self.markov_chain_settings)
        preferences.add_command(label="report_generator0 Settings", command=self.report_settings)
        preferences.add_command(label="Main5 Path", command=self.main_path)
        menubar.add_cascade(label="Preferences", menu=preferences)
        # MACHINE LEARNING OPTIONS
        machine_learning = Menu(menubar, tearoff=0)
        machine_learning_tools = Menu(menubar, tearoff=0)
        machine_learning_tools_generate = Menu(menubar, tearoff=0)
        machine_learning_settings = Menu(menubar, tearoff=0)
        machine_learning_tools_generate.add_command(label="Initial State", command=self.generate_vector)
        machine_learning_tools_generate.add_command(label="Transition Matrix", command=self.generate_matrix)
        machine_learning.add_command(label="Markov Chain", command=self.markov_chain_menu)
        machine_learning_tools.add_cascade(label="Generate Random", menu=machine_learning_tools_generate)
        machine_learning.add_cascade(label="Tools", menu=machine_learning_tools)
        menubar.add_cascade(label="Machine Learning", menu=machine_learning)
        # PLOTTER OPTIONS
        plotter_settings = Menu(menubar, tearoff=0)
        plotter_settings.add_command(label="Create Plot", command=self.create_plot)
        menubar.add_cascade(label="Plotter", menu=plotter_settings)
        # HELP MENU
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.load)
        menubar.add_cascade(label="Help2", menu=helpmenu)
        # display the menu
        master.config(menu=menubar)
        '''# DECLARING THE LISTS AND DICTIONARIES THAT WILL HOLD INFORMATION
        self.parameterLabel, self.parameterLabel2 = [], []  # This list will hold the labels
        self.label_text_List, self.label_text_List2 = [], []  # This list will hold the text which will go on each label
        self.parameterNames, self.parameterNumbers = {}, {}  # This will hold the name/number of each parameter
        self.inputValidationInformation = {}  # This will hold a dictionary in which keys are item name and values are lists of allowable digits
        self.entryList = {}  # This will hold the name of each entry text field as well as the field itself
        self.entries = []
        self.query = {}
        # POPULATING PARAMETER NUMBERS
        self.parameterNumbers = self.list_generator.getParameterNumbers()
        # POPULATING THE DICTIONARY WHICH WILL HOLD THE ENTRY TEXT FIELDS AND GENERATING THE PARAMETER NAMES LIST
        self.parameterNames = self.list_generator.getParameterNames()
        self.entryList = self.list_generator.getEntryList()
        self.files = 'single'
        self.filename = 'report'
        # POPULATING THE DICTIONARY WHICH WILL HOLD THE VALIDATION INFORMATION
        for i in range(self.number_of_items):
            temporaryAllowableValuesObject = AllowableValueListGenerator.AllowableValueListGenerator(self.MASTERPATH,
                                                                                                     self.parameterNames[
                                                                                                         i], i)
            self.inputValidationInformation[self.parameterNumbers[
                i]] = temporaryAllowableValuesObject.getAllowableValuesList()  # "LIST OF ALLOWABLE values HERE"
            # SETTING THE ENTRY TEXT FIELDS
            if self.parameterNames[i + 0] != '(Reserved)':
                self.entry = self.entryList[self.parameterNames[i + 0]]
                self.entries.append(self.entry)
                self.entry.grid(row=i + 2, column=2, columnspan=1)
                self.entryList[self.parameterNames[i]].insert(0, '')
            # POPULATING THE LISTS WHICH WILL HOLD THE LABELS, LABEL TEXT, AND TEXT ENTRY FIELDS
            self.label_text_List.append(StringVar())
            self.parameterLabel.append(Label(self.frame.interior, textvariable=self.label_text_List[i]))
            self.label_text_List2.append(StringVar())
            self.parameterLabel2.append(Label(self.frame.interior, textvariable=self.label_text_List2[i]))
            # SETTING THE LABELS AND TEXT ENTRY FIELDS IN THE MENU
            self.parameterName = self.parameterNames[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel[i]  # Set the parameter label from the parameter label list
            self.label_text = self.label_text_List[i]
            self.label_text.set(self.parameterName)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 2, column=1, columnspan=1, sticky=W)  # Set the label location
            # SETTING THE LABELS OF ITEM NUMBER
            self.parameterNumber = self.parameterNumbers[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel2[i]  # Set the parameter label from the parameter label list
            self.label_text2 = self.label_text_List2[i]
            self.label_text2.set(self.parameterNumber)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 2, column=0, columnspan=1, sticky=W)  # Set the label location
'''
        # SEPARATOR
        separator = ttk.Separator(self.master, orient=HORIZONTAL)
        separator.grid(row=1, column=0, columnspan=5, sticky=W + E)
        # SETTING THE BUTTONS
        self.search_button = Button(master, text="search", command=self.search, cursor='hand2')
        self.search_button.grid(row=2, column=4, sticky=SE)
        self.clear_button = Button(master, text="Clear", command=self.clear, cursor='hand2')
        self.clear_button.grid(
            row=2, column=0,
            sticky=SW
        )  # self.list_generator.getParameterNamesWithRespectToNumber()  # self.master.mainloop()

    def save(self):
        path = tkf.asksaveasfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            if path[path.__len__() - 4] is not ".":
                path = path + ".txt"

            with open(path, "w") as saveFile:
                for i in range(self.number_of_items):
                    pass  # saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")
        except:
            return

    def load(self):
        path = tkf.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            with open(path, "r") as loadFile:
                for i in range(self.number_of_items):
                    pass  # self.entryList[self.parameterNames[i + 0]].delete(0, END)
                for i, line in enumerate(loadFile):
                    pass  # self.entryList[self.parameterNames[i]].insert(i, line.strip())

        except:
            return

    def search(self):
        self.name_report()

    def search_resume(self):

        with open(self.MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "w") as saveFile:
            for i in range(self.number_of_items):
                pass  # saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")

        with open(self.MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "r") as loadFile:
            for i, line in enumerate(loadFile):
                '''if str(self.parameterNames[i]) == '(Reserved)':
                    self.query[self.parameterNames[i] + "_" + str(i)] = (line.strip())
                else:
                    self.query[self.parameterNames[i]] = (line.strip())'''

        if self.dataValidator():
            self.list_generator.setSearchList(self.query)
            Search.Search(self)

    def clear(self):
        for i in range(self.number_of_items):
            pass  # self.entryList[self.parameterNames[i + 0]].delete(0, END)

    def dataValidator(self):
        '''for i in range(self.number_of_items):
            if self.parameterNames[i] == '(Reserved)':
                pass
            else:
                if self.query[self.parameterNames[i]] != '':
                    if 'None' not in self.inputValidationInformation[self.parameterNumbers[i]]:
                        if True:  # self.query[self.parameterNames[i]] in self.inputValidationInformation[self.parameterNumbers[i]]:
                            pass
                        else:
                            print(str(self.query[self.parameterNames[i]]) + " is not a valid " + str(
                                self.parameterNames[i]))  # + ". search Aborted.")
                            pass'''
        return True

    def name_report(self):
        NameReport.NameReport(self)

    def report_settings(self):
        ReportSettings.ReportSettings(self)

    def markov_chain_menu(self):
        MarkovChainMenu.MarkovChainMenu(self)

    def markov_chain_settings(self):
        MarkovChainSettings.MarkovChainSettings(self.master, self.MASTERPATH)

    def create_plot(self):
        PlotterSettings.Plot(self.master, self.MASTERPATH)

    def plotter_settings(self):
        PlotterSettings.PlotterSettings(self.master, self.MASTERPATH)

    def main_path(self):
        MainPathSettings.MainPathSettings(self.master, self.MASTERPATH)

    def generate_vector(self):
        GeneratorSettings.SettingsGenerator(self.master, self.MASTERPATH, 0)

    def generate_matrix(self):
        GeneratorSettings.SettingsGenerator(self.master, self.MASTERPATH, 1)

    def set_filepath(self, filepath):
        self.filepath = filepath


import datetime

from Misc import AllowableValueListGenerator
from Utilities13 import MasterPath

number_of_items = 141
current_year = datetime.datetime.now().year - 1
MasterPath_Object = MasterPath.MasterPath()
MASTERPATH = MasterPath_Object.getMasterPath()


class BridgeDataQuery:
    def load(self):
        path = tkf.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            with open(path, "r") as loadFile:
                for i in range(number_of_items):
                    self.entryList[self.parameterNames[i + 0]].delete(0, END)
                i = 0
                for line in loadFile:
                    self.entryList[self.parameterNames[i]].insert(i, line.strip())
                    i += 1
        except:
            return

    def search(self):
        self.name_report()

    def search_resume(self):
        with open(MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "w") as saveFile:
            for i in range(number_of_items):
                saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")

        with open(MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "r") as loadFile:
            i = 0
            for line in loadFile:
                if str(self.parameterNames[i]) == '(Reserved)':
                    self.query[self.parameterNames[i] + "_" + str(i)] = (line.strip())
                else:
                    self.query[self.parameterNames[i]] = (line.strip())
                i += 1

        if self.dataValidator():
            Search.InitiateSearch(
                MASTERPATH, self.parameterNames, self.parameterNumbers, self.query, listGenerator,
                self.files, self.filename
            )

    def clear(self):
        for i in range(number_of_items):
            self.entryList[self.parameterNames[i + 0]].delete(0, END)

    def dataValidator(self):
        for i in range(number_of_items):
            if self.parameterNames[i] == '(Reserved)':
                pass
            else:
                if self.query[self.parameterNames[i]] != '':
                    if 'None' not in self.inputValidationInformation[self.parameterNumbers[i]]:
                        if True:  # self.query[self.parameterNames[i]] in self.inputValidationInformation[self.parameterNumbers[i]]:
                            pass
                        else:
                            print(
                                str(self.query[self.parameterNames[i]]) + " is not a valid " + str(
                                    self.parameterNames[i]
                                )
                            )  # + ". search Aborted.")
                            pass
        return True

    def name_report(self):
        NameReportsMenu = NameReport.NameReport(self.master, MASTERPATH, self)

    def report_settings(self):
        ReportSettingsMenu = ReportSettings.ReportSettings(self.master, MASTERPATH)

    def machine_options(self):
        MachineOptionMenu = MarkovChainMenu.MachineLearningSettings(self.master)

    def markov_chain_options(self):
        MarkovChainSettingsMenu = MarkovChainSettings.MarkovChainSettings(self.master, MASTERPATH)

    def create_plot(self):
        CreatePlotMenu = PlotterSettings.Plot(self.master, MASTERPATH)

    def plotter_settings(self):
        PlotterSettingsMenu = PlotterSettings.PlotterSettings(self.master, MASTERPATH)

    def main_path(self):
        MainPathMenu = MainPathSettings.MainPathSettings(self.master, MASTERPATH)

    def generate_vector(self):
        GenerateSettingsMenu = GeneratorSettings.SettingsGenerator(self.master, MASTERPATH, 0)

    def generate_matrix(self):
        GenerateSettingsMenu = GeneratorSettings.SettingsGenerator(self.master, MASTERPATH, 1)

    def set_filename(self, filename):
        self.filename = filename


CreateWindow = CreateWindow(0)

import datetime
import tkinter.ttk as ttk
from tkinter import *
from tkinter import filedialog as tkf

from Utilities13 import MasterPath

current_year = datetime.datetime.now().year - 1
MasterPath_Object = MasterPath.MasterPath()


class BridgeDataQuery:
    def __init__(self, master):
        self.master = master

        # TODO FIX SCROLLBAR
        self.frame = VerticalScrolledFrame.VerticalScrolledFrame(master)
        self.frame.grid(column=4, sticky=NSEW)

        self.MASTERPATH = MasterPath_Object.getMasterPath()
        self.number_of_items = 142
        self.list_generator = ListGenerator.ListGenerator(self)
        menubar = Menu(master)

        # create a pulldown menu, and add it to the menu bar
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open search", command=self.load)
        filemenu.add_command(label="Save search As", command=self.save)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.destroy)
        menubar.add_cascade(label="File", menu=filemenu)

        # PREFERENCES OPTIONS
        preferences = Menu(menubar, tearoff=0)
        preferences.add_command(label="Plotter Settings", command=self.plotter_settings)
        preferences.add_command(label="Markov Chain Settings", command=self.markov_chain_settings)
        preferences.add_command(label="report_generator0 Settings", command=self.report_settings)
        preferences.add_command(label="Main5 Path", command=self.main_path)
        menubar.add_cascade(label="Preferences", menu=preferences)

        # MACHINE LEARNING OPTIONS
        machine_learning = Menu(menubar, tearoff=0)
        machine_learning_tools = Menu(menubar, tearoff=0)
        machine_learning_tools_generate = Menu(menubar, tearoff=0)
        machine_learning_settings = Menu(menubar, tearoff=0)

        machine_learning_tools_generate.add_command(label="Initial State", command=self.generate_vector)
        machine_learning_tools_generate.add_command(label="Transition Matrix", command=self.generate_matrix)

        machine_learning.add_command(label="Markov Chain", command=self.markov_chain_menu)
        machine_learning_tools.add_cascade(label="Generate Random", menu=machine_learning_tools_generate)
        machine_learning.add_cascade(label="Tools", menu=machine_learning_tools)
        menubar.add_cascade(label="Machine Learning", menu=machine_learning)

        # PLOTTER OPTIONS
        plotter_settings = Menu(menubar, tearoff=0)
        plotter_settings.add_command(label="Create Plot", command=self.create_plot)
        menubar.add_cascade(label="Plotter", menu=plotter_settings)

        # HELP MENU
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.load)
        menubar.add_cascade(label="Help2", menu=helpmenu)

        # display the menu
        master.config(menu=menubar)

        '''# DECLARING THE LISTS AND DICTIONARIES THAT WILL HOLD INFORMATION
        self.parameterLabel, self.parameterLabel2 = [], []  # This list will hold the labels
        self.label_text_List, self.label_text_List2 = [], []  # This list will hold the text which will go on each label
        self.parameterNames, self.parameterNumbers = {}, {}  # This will hold the name/number of each parameter
        self.inputValidationInformation = {}  # This will hold a dictionary in which keys are item name and values are lists of allowable digits
        self.entryList = {}  # This will hold the name of each entry text field as well as the field itself
        self.entries = []
        self.query = {}
        # POPULATING PARAMETER NUMBERS
        self.parameterNumbers = self.list_generator.getParameterNumbers()

        # POPULATING THE DICTIONARY WHICH WILL HOLD THE ENTRY TEXT FIELDS AND GENERATING THE PARAMETER NAMES LIST
        self.parameterNames = self.list_generator.getParameterNames()

        self.entryList = self.list_generator.getEntryList()
        self.files = 'single'
        self.filename = 'report'

        # POPULATING THE DICTIONARY WHICH WILL HOLD THE VALIDATION INFORMATION
        for i in range(self.number_of_items):
            temporaryAllowableValuesObject = AllowableValueListGenerator.AllowableValueListGenerator(self.MASTERPATH,
                                                                                                     self.parameterNames[
                                                                                                         i], i)
            self.inputValidationInformation[self.parameterNumbers[
                i]] = temporaryAllowableValuesObject.getAllowableValuesList()  # "LIST OF ALLOWABLE values HERE"
            # SETTING THE ENTRY TEXT FIELDS
            if self.parameterNames[i + 0] != '(Reserved)':
                self.entry = self.entryList[self.parameterNames[i + 0]]
                self.entries.append(self.entry)
                self.entry.grid(row=i + 2, column=2, columnspan=1)
                self.entryList[self.parameterNames[i]].insert(0, '')
            # POPULATING THE LISTS WHICH WILL HOLD THE LABELS, LABEL TEXT, AND TEXT ENTRY FIELDS
            self.label_text_List.append(StringVar())
            self.parameterLabel.append(Label(self.frame.interior, textvariable=self.label_text_List[i]))
            self.label_text_List2.append(StringVar())
            self.parameterLabel2.append(Label(self.frame.interior, textvariable=self.label_text_List2[i]))
            # SETTING THE LABELS AND TEXT ENTRY FIELDS IN THE MENU
            self.parameterName = self.parameterNames[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel[i]  # Set the parameter label from the parameter label list
            self.label_text = self.label_text_List[i]
            self.label_text.set(self.parameterName)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 2, column=1, columnspan=1, sticky=W)  # Set the label location
            # SETTING THE LABELS OF ITEM NUMBER
            self.parameterNumber = self.parameterNumbers[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel2[i]  # Set the parameter label from the parameter label list
            self.label_text2 = self.label_text_List2[i]
            self.label_text2.set(self.parameterNumber)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 2, column=0, columnspan=1, sticky=W)  # Set the label location
'''
        # SEPARATOR
        separator = ttk.Separator(self.master, orient=HORIZONTAL)
        separator.grid(row=1, column=0, columnspan=5, sticky=W + E)

        # SETTING THE BUTTONS
        self.search_button = Button(master, text="search", command=self.search, cursor='hand2')
        self.search_button.grid(row=2, column=4, sticky=SE)
        self.clear_button = Button(master, text="Clear", command=self.clear, cursor='hand2')
        self.clear_button.grid(row=2, column=0, sticky=SW)

        # self.list_generator.getParameterNamesWithRespectToNumber()

        # self.master.mainloop()


import datetime

from Utilities.Preferences import MasterPath

number_of_items = 141
current_year = datetime.datetime.now().year - 1
MasterPath_Object = MasterPath.MasterPath()
MASTERPATH = MasterPath_Object.getMasterPath()


class BridgeDataQuery:
    def __init__(self, master):
        self.master = master
        master.title("Christian's Machine")
        end = 4

        self.frame = VerticalScrolledFrame.VerticalScrolledFrame(master)
        self.frame.grid(column=end)
        global listGenerator
        listGenerator = ListGenerator.ListGenerator(MASTERPATH, self.frame, number_of_items)

        menubar = Menu(master)

        # create a pulldown menu, and add it to the menu bar
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open search", command=self.load)
        filemenu.add_command(label="Save search As", command=self.save)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.destroy)
        menubar.add_cascade(label="File", menu=filemenu)

        # PREFERENCES OPTIONS
        preferences = Menu(menubar, tearoff=0)
        preferences.add_command(label="report_generator0 Options", command=self.report_settings)
        preferences.add_command(label="Main5 Path", command=self.main_path)
        menubar.add_cascade(label="Preferences", menu=preferences)

        # MACHINE LEARNING OPTIONS
        machine_learning = Menu(menubar, tearoff=0)
        machine_learning_tools = Menu(menubar, tearoff=0)
        machine_learning_tools_generate = Menu(menubar, tearoff=0)
        machine_learning_settings = Menu(menubar, tearoff=0)

        machine_learning_settings.add_command(label="Markov Chain", command=self.markov_chain_options)
        # machine_learning_settings.add_command(label="Neural Network", command=self.machine_options)
        # achine_learning_settings.add_command(label="Genetic Algorithm", command=self.machine_options)
        # machine_learning_settings.add_command(label="Support Vector Machine", command=self.machine_options)

        machine_learning_tools_generate.add_command(label="Random Vector", command=self.generate_vector)
        machine_learning_tools_generate.add_command(label="Random Matrix", command=self.generate_matrix)

        machine_learning.add_cascade(label="Settings", menu=machine_learning_settings)
        machine_learning_tools.add_cascade(label="Generate", menu=machine_learning_tools_generate)
        machine_learning.add_cascade(label="Tools", menu=machine_learning_tools)
        menubar.add_cascade(label="Machine Learning", menu=machine_learning)

        # PLOTTER OPTIONS
        plotter_settings = Menu(menubar, tearoff=0)
        plotter_settings.add_command(label="Create Plot", command=self.create_plot)
        plotter_settings.add_command(label="Plotter Settings", command=self.plotter_settings)
        menubar.add_cascade(label="Plotter", menu=plotter_settings)

        # HELP MENU
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.load)
        menubar.add_cascade(label="help", menu=helpmenu)

        # display the menu
        master.config(menu=menubar)

        # DECLARING THE LISTS AND DICTIONARIES THAT WILL HOLD INFORMATION
        self.parameterLabel, self.parameterLabel2 = [], []  # This list will hold the labels
        self.label_text_List, self.label_text_List2 = [], []  # This list will hold the text which will go on each label
        self.parameterNames = {}  # This will hold the name of each parameter
        self.parameterNumbers = {}  # This will hold the name of each parameter
        self.inputValidationInformation = {}  # This will hold a dictionary in which keys are item name and values are lists of allowable digits
        self.entryList = {}  # This will hold the name of each entry text field as well as the field itself
        self.entries = []
        self.query = {}
        # POPULATING PARAMETER NUMBERS
        self.parameterNumbers = listGenerator.getParameterNumbers()

        # POPULATING THE DICTIONARY WHICH WILL HOLD THE ENTRY TEXT FIELDS AND GENERATING THE PARAMETER NAMES LIST
        self.parameterNames = listGenerator.getParameterNames()
        self.entryList = listGenerator.getEntryList()
        self.files = 'single'
        self.filename = 'report'

        # POPULATING THE DICTIONARY WHICH WILL HOLD THE VALIDATION INFORMATION
        for i in range(number_of_items):
            temporaryAllowableValuesObject = AllowableValueListGenerator.AllowableValueListGenerator(
                MASTERPATH,
                self.parameterNames[
                    i], i
            )
            self.inputValidationInformation[self.parameterNumbers[
                i]] = temporaryAllowableValuesObject.getAllowableValuesList()  # "LIST OF ALLOWABLE values HERE"
            # SETTING THE ENTRY TEXT FIELDS
            if self.parameterNames[i + 0] != '(Reserved)':
                self.entry = self.entryList[self.parameterNames[i + 0]]
                self.entries.append(self.entry)
                self.entry.grid(row=i + 3, column=3, columnspan=1)
                self.entryList[self.parameterNames[i]].insert(0, '')
            # POPULATING THE LISTS WHICH WILL HOLD THE LABELS, LABEL TEXT, AND TEXT ENTRY FIELDS
            self.label_text_List.append(StringVar())
            self.parameterLabel.append(Label(self.frame.interior, textvariable=self.label_text_List[i]))
            self.label_text_List2.append(StringVar())
            self.parameterLabel2.append(Label(self.frame.interior, textvariable=self.label_text_List2[i]))
            # SETTING THE LABELS AND TEXT ENTRY FIELDS IN THE MENU
            self.parameterName = self.parameterNames[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel[i]  # Set the parameter label from the parameter label list
            self.label_text = self.label_text_List[i]
            self.label_text.set(self.parameterName)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 3, column=1, columnspan=1, sticky=W)  # Set the label location
            # SETTING THE LABELS OF ITEM NUMBER
            self.parameterNumber = self.parameterNumbers[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel2[i]  # Set the parameter label from the parameter label list
            self.label_text2 = self.label_text_List2[i]
            self.label_text2.set(self.parameterNumber)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 3, column=0, columnspan=1, sticky=W)  # Set the label location

        # SEPARATOR
        separator = ttk.Separator(self.master, orient=HORIZONTAL)
        separator.grid(row=1, column=0, columnspan=5, sticky=W + E)

        # SETTING THE BUTTONS
        self.search_button = Button(master, text="search", command=self.search, cursor='hand2')
        self.search_button.grid(row=2, column=4, sticky=E)
        self.clear_button = Button(master, text="Clear", command=self.clear, cursor='hand2')
        self.clear_button.grid(row=2, column=0, sticky=W)

        master.mainloop()

    def save(self):
        path = tkf.asksaveasfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            if path[path.__len__() - 4] is not ".":
                path = path + ".txt"

            with open(path, "w") as saveFile:
                for i in range(number_of_items):
                    saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")
        except:
            return

    def load(self):
        path = tkf.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            with open(path, "r") as loadFile:
                for i in range(number_of_items):
                    self.entryList[self.parameterNames[i + 0]].delete(0, END)
                i = 0
                for line in loadFile:
                    self.entryList[self.parameterNames[i]].insert(i, line.strip())
                    i += 1
        except:
            return

    def search(self):
        self.name_report()

    def search_resume(self):
        with open(MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "w") as saveFile:
            for i in range(number_of_items):
                saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")

        with open(MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "r") as loadFile:
            i = 0
            for line in loadFile:
                if str(self.parameterNames[i]) == '(Reserved)':
                    self.query[self.parameterNames[i] + "_" + str(i)] = (line.strip())
                else:
                    self.query[self.parameterNames[i]] = (line.strip())
                i += 1

        if self.dataValidator():
            Search.InitiateSearch(
                MASTERPATH, self.parameterNames, self.parameterNumbers, self.query, listGenerator,
                self.files, self.filename
            )

    def clear(self):
        for i in range(number_of_items):
            self.entryList[self.parameterNames[i + 0]].delete(0, END)

    def dataValidator(self):
        for i in range(number_of_items):
            if self.parameterNames[i] == '(Reserved)':
                pass
            else:
                if self.query[self.parameterNames[i]] != '':
                    if 'None' not in self.inputValidationInformation[self.parameterNumbers[i]]:
                        if True:  # self.query[self.parameterNames[i]] in self.inputValidationInformation[self.parameterNumbers[i]]:
                            pass
                        else:
                            print(
                                str(self.query[self.parameterNames[i]]) + " is not a valid " + str(
                                    self.parameterNames[i]
                                )
                            )  # + ". search Aborted.")
                            pass
        return True

    def name_report(self):
        NameReportsMenu = ReportMenu.NameReport(self.master, MASTERPATH, self)

    def report_settings(self):
        ReportSettingsMenu = ReportSettings.ReportSettings(self.master, MASTERPATH)

    def machine_options(self):
        MachineOptionMenu = MarkovChainMenu.MachineLearningSettings(self.master)

    def markov_chain_options(self):
        MarkovChainSettingsMenu = MarkovChainSettings.MarkovChainSettings(self.master, MASTERPATH)

    def create_plot(self):
        CreatePlotMenu = PlotterSettings.Plot(self.master, MASTERPATH)

    def plotter_settings(self):
        PlotterSettingsMenu = PlotterSettings.PlotterSettings(self.master, MASTERPATH)

    def main_path(self):
        MainPathMenu = MainPathSettings.MainPathSettings(self.master, MASTERPATH)

    def generate_vector(self):
        GenerateSettingsMenu = GeneratorSettings.SettingsGenerator(self.master, MASTERPATH, 0)

    def generate_matrix(self):
        GenerateSettingsMenu = GeneratorSettings.SettingsGenerator(self.master, MASTERPATH, 1)

    def set_filename(self, filename):
        self.filename = filename


CreateWindow = CreateWindow(0)

import datetime

from Interface.SubMenus import ReportMenu
from Utilities.Preferences import MasterPath

current_year = datetime.datetime.now().year - 1
MasterPath_Object = MasterPath.MasterPath()


class BridgeDataQuery:
    def __init__(self, master):
        self.master = master

        # TODO FIX SCROLLBAR
        self.frame = VerticalScrolledFrame.VerticalScrolledFrame(master)
        self.frame.grid(column=4, sticky=NSEW)

        self.MASTERPATH = MasterPath_Object.getMasterPath()
        self.number_of_items = 142
        self.list_generator = ListGenerator.ListGenerator(self)
        menubar = Menu(master)

        # create a pulldown menu, and add it to the menu bar
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open search", command=self.load)
        filemenu.add_command(label="Save search As", command=self.save)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.destroy)
        menubar.add_cascade(label="File", menu=filemenu)

        # PREFERENCES OPTIONS
        preferences = Menu(menubar, tearoff=0)
        preferences.add_command(label="Plotter Settings", command=self.plotter_settings)
        preferences.add_command(label="Markov Chain Settings", command=self.markov_chain_settings)
        preferences.add_command(label="report_generator0 Settings", command=self.report_settings)
        preferences.add_command(label="Main5 Path", command=self.main_path)
        menubar.add_cascade(label="Preferences", menu=preferences)

        # MACHINE LEARNING OPTIONS
        machine_learning = Menu(menubar, tearoff=0)
        machine_learning_tools = Menu(menubar, tearoff=0)
        machine_learning_tools_generate = Menu(menubar, tearoff=0)
        machine_learning_settings = Menu(menubar, tearoff=0)

        machine_learning_tools_generate.add_command(label="Initial State", command=self.generate_vector)
        machine_learning_tools_generate.add_command(label="Transition Matrix", command=self.generate_matrix)

        machine_learning.add_command(label="Markov Chain", command=self.markov_chain_menu)
        machine_learning_tools.add_cascade(label="Generate Random", menu=machine_learning_tools_generate)
        machine_learning.add_cascade(label="Tools", menu=machine_learning_tools)
        menubar.add_cascade(label="Machine Learning", menu=machine_learning)

        # PLOTTER OPTIONS
        plotter_settings = Menu(menubar, tearoff=0)
        plotter_settings.add_command(label="Create Plot", command=self.create_plot)
        menubar.add_cascade(label="Plotter", menu=plotter_settings)

        # HELP MENU
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.load)
        menubar.add_cascade(label="help", menu=helpmenu)

        # display the menu
        master.config(menu=menubar)

        # DECLARING THE LISTS AND DICTIONARIES THAT WILL HOLD INFORMATION
        self.parameterLabel, self.parameterLabel2 = [], []  # This list will hold the labels
        self.label_text_List, self.label_text_List2 = [], []  # This list will hold the text which will go on each label
        self.parameterNames, self.parameterNumbers = {}, {}  # This will hold the name/number of each parameter
        self.inputValidationInformation = {}  # This will hold a dictionary in which keys are item name and values are lists of allowable digits
        self.entryList = {}  # This will hold the name of each entry text field as well as the field itself
        self.entries = []
        self.query = {}
        # POPULATING PARAMETER NUMBERS
        self.parameterNumbers = self.list_generator.getParameterNumbers()

        # POPULATING THE DICTIONARY WHICH WILL HOLD THE ENTRY TEXT FIELDS AND GENERATING THE PARAMETER NAMES LIST
        self.parameterNames = self.list_generator.getParameterNames()

        self.entryList = self.list_generator.getEntryList()
        self.files = 'single'
        self.filename = 'report'

        # POPULATING THE DICTIONARY WHICH WILL HOLD THE VALIDATION INFORMATION
        for i in range(self.number_of_items):
            temporaryAllowableValuesObject = AllowableValueListGenerator.AllowableValueListGenerator(
                self.MASTERPATH,
                self.parameterNames[
                    i], i
            )
            self.inputValidationInformation[self.parameterNumbers[
                i]] = temporaryAllowableValuesObject.getAllowableValuesList()  # "LIST OF ALLOWABLE values HERE"
            # SETTING THE ENTRY TEXT FIELDS
            if self.parameterNames[i + 0] != '(Reserved)':
                self.entry = self.entryList[self.parameterNames[i + 0]]
                self.entries.append(self.entry)
                self.entry.grid(row=i + 3, column=3, columnspan=1)
                self.entryList[self.parameterNames[i]].insert(0, '')
            # POPULATING THE LISTS WHICH WILL HOLD THE LABELS, LABEL TEXT, AND TEXT ENTRY FIELDS
            self.label_text_List.append(StringVar())
            self.parameterLabel.append(Label(self.frame.interior, textvariable=self.label_text_List[i]))
            self.label_text_List2.append(StringVar())
            self.parameterLabel2.append(Label(self.frame.interior, textvariable=self.label_text_List2[i]))
            # SETTING THE LABELS AND TEXT ENTRY FIELDS IN THE MENU
            self.parameterName = self.parameterNames[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel[i]  # Set the parameter label from the parameter label list
            self.label_text = self.label_text_List[i]
            self.label_text.set(self.parameterName)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 3, column=1, columnspan=1, sticky=W)  # Set the label location
            # SETTING THE LABELS OF ITEM NUMBER
            self.parameterNumber = self.parameterNumbers[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel2[i]  # Set the parameter label from the parameter label list
            self.label_text2 = self.label_text_List2[i]
            self.label_text2.set(self.parameterNumber)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 3, column=0, columnspan=1, sticky=W)  # Set the label location

        # SEPARATOR
        separator = ttk.Separator(self.master, orient=HORIZONTAL)
        separator.grid(row=1, column=0, columnspan=5, sticky=W + E)

        # SETTING THE BUTTONS
        self.search_button = Button(master, text="search", command=self.search, cursor='hand2')
        self.search_button.grid(row=2, column=4, sticky=SE)
        self.clear_button = Button(master, text="Clear", command=self.clear, cursor='hand2')
        self.clear_button.grid(row=2, column=0, sticky=SW)

        # self.list_generator.getParameterNamesWithRespectToNumber()

        # self.master.mainloop()

    def save(self):
        path = tkf.asksaveasfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            if path[path.__len__() - 4] is not ".":
                path = path + ".txt"

            with open(path, "w") as saveFile:
                for i in range(self.number_of_items):
                    pass  # saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")
        except:
            return

    def load(self):
        path = tkf.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            with open(path, "r") as loadFile:
                for i in range(self.number_of_items):
                    pass  # self.entryList[self.parameterNames[i + 0]].delete(0, END)
                for i, line in enumerate(loadFile):
                    pass  # self.entryList[self.parameterNames[i]].insert(i, line.strip())

        except:
            return

    def search(self):
        self.name_report()

    def search_resume(self):

        with open(self.MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "w") as saveFile:
            for i in range(self.number_of_items):
                pass  # saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")

        with open(self.MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "r") as loadFile:
            for i, line in enumerate(loadFile):
                '''if str(self.parameterNames[i]) == '(Reserved)':
                    self.query[self.parameterNames[i] + "_" + str(i)] = (line.strip())
                else:
                    self.query[self.parameterNames[i]] = (line.strip())'''

        if self.dataValidator():
            self.list_generator.setSearchList(self.query)
            Search.Search(self)

    def clear(self):
        for i in range(self.number_of_items):
            pass  # self.entryList[self.parameterNames[i + 0]].delete(0, END)

    def dataValidator(self):
        '''for i in range(self.number_of_items):
            if self.parameterNames[i] == '(Reserved)':
                pass
            else:
                if self.query[self.parameterNames[i]] != '':
                    if 'None' not in self.inputValidationInformation[self.parameterNumbers[i]]:
                        if True:  # self.query[self.parameterNames[i]] in self.inputValidationInformation[self.parameterNumbers[i]]:
                            pass
                        else:
                            print(str(self.query[self.parameterNames[i]]) + " is not a valid " + str(
                                self.parameterNames[i]))  # + ". search Aborted.")
                            pass'''
        return True

    def name_report(self):
        ReportMenu.NameReport(self)

    def report_settings(self):
        ReportSettings.ReportSettings(self)

    def markov_chain_menu(self):
        MarkovChainMenu.MarkovChainMenu(self)

    def markov_chain_settings(self):
        MarkovChainSettings.MarkovChainSettings(self.master, self.MASTERPATH)

    def create_plot(self):
        PlotterSettings.Plot(self.master, self.MASTERPATH)

    def plotter_settings(self):
        PlotterSettings.PlotterSettings(self.master, self.MASTERPATH)

    def main_path(self):
        MainPathSettings.MainPathSettings(self.master, self.MASTERPATH)

    def generate_vector(self):
        GeneratorSettings.SettingsGenerator(self.master, self.MASTERPATH, 0)

    def generate_matrix(self):
        GeneratorSettings.SettingsGenerator(self.master, self.MASTERPATH, 1)

    def set_filepath(self, filepath):
        self.filepath = filepath


import datetime

from Utilities.GlobalUtilities import MasterPath
from z__legacy import AllowableValueListGenerator

number_of_items = 141
current_year = datetime.datetime.now().year - 1
MasterPath_Object = MasterPath.MasterPath()
MASTERPATH = MasterPath_Object.getMasterPath()


class BridgeDataQuery:
    def __init__(self, master):
        self.master = master
        master.title("Christian's Machine")
        end = 4

        self.frame = VerticalScrolledFrame.VerticalScrolledFrame(master)
        self.frame.grid(column=end)
        global listGenerator
        listGenerator = ListGenerator.ListGenerator(MASTERPATH, self.frame, number_of_items)

        menubar = Menu(master)

        # create a pulldown menu, and add it to the menu bar
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Search", command=self.load)
        filemenu.add_command(label="Save Search As", command=self.save)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.destroy)
        menubar.add_cascade(label="File", menu=filemenu)

        # PREFERENCES OPTIONS
        preferences = Menu(menubar, tearoff=0)
        preferences.add_command(label="Report Options", command=self.report_settings)
        preferences.add_command(label="Main Path", command=self.main_path)
        menubar.add_cascade(label="Preferences", menu=preferences)

        # MACHINE LEARNING OPTIONS
        machine_learning = Menu(menubar, tearoff=0)
        machine_learning_tools = Menu(menubar, tearoff=0)
        machine_learning_tools_generate = Menu(menubar, tearoff=0)
        machine_learning_settings = Menu(menubar, tearoff=0)

        machine_learning_settings.add_command(label="Markov Chain", command=self.markov_chain_options)
        # machine_learning_settings.add_command(label="Neural Network", command=self.machine_options)
        # achine_learning_settings.add_command(label="Genetic Algorithm", command=self.machine_options)
        # machine_learning_settings.add_command(label="Support Vector Machine", command=self.machine_options)

        machine_learning_tools_generate.add_command(label="Random Vector", command=self.generate_vector)
        machine_learning_tools_generate.add_command(label="Random Matrix", command=self.generate_matrix)

        machine_learning.add_cascade(label="Settings", menu=machine_learning_settings)
        machine_learning_tools.add_cascade(label="Generate", menu=machine_learning_tools_generate)
        machine_learning.add_cascade(label="Tools", menu=machine_learning_tools)
        menubar.add_cascade(label="Machine Learning", menu=machine_learning)

        # PLOTTER OPTIONS
        plotter_settings = Menu(menubar, tearoff=0)
        plotter_settings.add_command(label="Create Plot", command=self.create_plot)
        plotter_settings.add_command(label="Plotter Settings", command=self.plotter_settings)
        menubar.add_cascade(label="Plotter", menu=plotter_settings)

        # HELP MENU
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.load)
        menubar.add_cascade(label="Help", menu=helpmenu)

        # display the menu
        master.config(menu=menubar)

        # DECLARING THE LISTS AND DICTIONARIES THAT WILL HOLD INFORMATION
        self.parameterLabel, self.parameterLabel2 = [], []  # This list will hold the labels
        self.label_text_List, self.label_text_List2 = [], []  # This list will hold the text which will go on each label
        self.parameterNames = {}  # This will hold the name of each parameter
        self.parameterNumbers = {}  # This will hold the name of each parameter
        self.inputValidationInformation = {}  # This will hold a dictionary in which keys are item name and values are lists of allowable digits
        self.entryList = {}  # This will hold the name of each entry text field as well as the field itself
        self.entries = []
        self.query = {}
        # POPULATING PARAMETER NUMBERS
        self.parameterNumbers = listGenerator.getParameterNumbers()

        # POPULATING THE DICTIONARY WHICH WILL HOLD THE ENTRY TEXT FIELDS AND GENERATING THE PARAMETER NAMES LIST
        self.parameterNames = listGenerator.getParameterNames()
        self.entryList = listGenerator.getEntryList()
        self.files = 'single'
        self.filename = 'report'

        # POPULATING THE DICTIONARY WHICH WILL HOLD THE VALIDATION INFORMATION
        for i in range(number_of_items):
            temporaryAllowableValuesObject = AllowableValueListGenerator.AllowableValueListGenerator(
                MASTERPATH,
                self.parameterNames[
                    i], i
            )
            self.inputValidationInformation[self.parameterNumbers[
                i]] = temporaryAllowableValuesObject.getAllowableValuesList()  # "LIST OF ALLOWABLE values HERE"
            # SETTING THE ENTRY TEXT FIELDS
            if self.parameterNames[i + 0] != '(Reserved)':
                self.entry = self.entryList[self.parameterNames[i + 0]]
                self.entries.append(self.entry)
                self.entry.grid(row=i + 3, column=3, columnspan=1)
                self.entryList[self.parameterNames[i]].insert(0, '')
            # POPULATING THE LISTS WHICH WILL HOLD THE LABELS, LABEL TEXT, AND TEXT ENTRY FIELDS
            self.label_text_List.append(StringVar())
            self.parameterLabel.append(Label(self.frame.interior, textvariable=self.label_text_List[i]))
            self.label_text_List2.append(StringVar())
            self.parameterLabel2.append(Label(self.frame.interior, textvariable=self.label_text_List2[i]))
            # SETTING THE LABELS AND TEXT ENTRY FIELDS IN THE MENU
            self.parameterName = self.parameterNames[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel[i]  # Set the parameter label from the parameter label list
            self.label_text = self.label_text_List[i]
            self.label_text.set(self.parameterName)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 3, column=1, columnspan=1, sticky=W)  # Set the label location
            # SETTING THE LABELS OF ITEM NUMBER
            self.parameterNumber = self.parameterNumbers[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel2[i]  # Set the parameter label from the parameter label list
            self.label_text2 = self.label_text_List2[i]
            self.label_text2.set(self.parameterNumber)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 3, column=0, columnspan=1, sticky=W)  # Set the label location

        # SEPARATOR
        separator = ttk.Separator(self.master, orient=HORIZONTAL)
        separator.grid(row=1, column=0, columnspan=5, sticky=W + E)

        # SETTING THE BUTTONS
        self.search_button = Button(master, text="Search", command=self.search, cursor='hand2')
        self.search_button.grid(row=2, column=4, sticky=E)
        self.clear_button = Button(master, text="Clear", command=self.clear, cursor='hand2')
        self.clear_button.grid(row=2, column=0, sticky=W)

        master.mainloop()

    def save(self):
        path = tkf.asksaveasfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            if path[path.__len__() - 4] is not ".":
                path = path + ".txt"

            with open(path, "w") as saveFile:
                for i in range(number_of_items):
                    saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")
        except:
            return

    def load(self):
        path = tkf.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            with open(path, "r") as loadFile:
                for i in range(number_of_items):
                    self.entryList[self.parameterNames[i + 0]].delete(0, END)
                i = 0
                for line in loadFile:
                    self.entryList[self.parameterNames[i]].insert(i, line.strip())
                    i += 1
        except:
            return

    def search(self):
        self.name_report()

    def search_resume(self):
        with open(MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "w") as saveFile:
            for i in range(number_of_items):
                saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")

        with open(MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "r") as loadFile:
            i = 0
            for line in loadFile:
                if str(self.parameterNames[i]) == '(Reserved)':
                    self.query[self.parameterNames[i] + "_" + str(i)] = (line.strip())
                else:
                    self.query[self.parameterNames[i]] = (line.strip())
                i += 1

        if self.dataValidator():
            Search.InitiateSearch(
                MASTERPATH, self.parameterNames, self.parameterNumbers, self.query, listGenerator,
                self.files, self.filename
            )

    def clear(self):
        for i in range(number_of_items):
            self.entryList[self.parameterNames[i + 0]].delete(0, END)

    def dataValidator(self):
        for i in range(number_of_items):
            if self.parameterNames[i] == '(Reserved)':
                pass
            else:
                if self.query[self.parameterNames[i]] != '':
                    if 'None' not in self.inputValidationInformation[self.parameterNumbers[i]]:
                        if True:  # self.query[self.parameterNames[i]] in self.inputValidationInformation[self.parameterNumbers[i]]:
                            pass
                        else:
                            print(
                                str(self.query[self.parameterNames[i]]) + " is not a valid " + str(
                                    self.parameterNames[i]
                                )
                            )  # + ". Search Aborted.")
                            pass
        return True

    def name_report(self):
        NameReportsMenu = NameReport.NameReport(self.master, MASTERPATH, self)

    def report_settings(self):
        ReportSettingsMenu = ReportSettings.ReportSettings(self.master, MASTERPATH)

    def machine_options(self):
        MachineOptionMenu = MarkovChainMenu.MachineLearningSettings(self.master)

    def markov_chain_options(self):
        MarkovChainSettingsMenu = MarkovChainSettings.MarkovChainSettings(self.master, MASTERPATH)

    def create_plot(self):
        CreatePlotMenu = PlotterSettings.Plot(self.master, MASTERPATH)

    def plotter_settings(self):
        PlotterSettingsMenu = PlotterSettings.PlotterSettings(self.master, MASTERPATH)

    def main_path(self):
        MainPathMenu = MainPathSettings.MainPathSettings(self.master, MASTERPATH)

    def generate_vector(self):
        GenerateSettingsMenu = GeneratorSettings.SettingsGenerator(self.master, MASTERPATH, 0)

    def generate_matrix(self):
        GenerateSettingsMenu = GeneratorSettings.SettingsGenerator(self.master, MASTERPATH, 1)

    def set_filename(self, filename):
        self.filename = filename


CreateWindow = CreateWindow(0)

import datetime
import tkinter.ttk as ttk
from tkinter import *
from tkinter import filedialog as tkf

from Interface import NameReport
from Interface.InterfaceUtilities import ReportSettings, MarkovChainMenu, VerticalScrolledFrame
from Interface.InterfaceUtilities.misc import GeneratorSettings, MarkovChainSettings, PlotterSettings, MainPathSettings
from Utilities.Generators import ListGenerator
from Utilities.GlobalUtilities import MasterPath

from search import Search

current_year = datetime.datetime.now().year - 1
MasterPath_Object = MasterPath.MasterPath()


class BridgeDataQuery:
    def __init__(self, master):
        self.master = master

        # TODO FIX SCROLLBAR
        self.frame = VerticalScrolledFrame.VerticalScrolledFrame(master)
        self.frame.grid(column=4, sticky=NSEW)

        self.MASTERPATH = MasterPath_Object.getMasterPath()
        self.number_of_items = 142
        self.list_generator = ListGenerator.ListGenerator(self)
        menubar = Menu(master)

        # create a pulldown menu, and add it to the menu bar
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Search", command=self.load)
        filemenu.add_command(label="Save Search As", command=self.save)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.destroy)
        menubar.add_cascade(label="File", menu=filemenu)

        # PREFERENCES OPTIONS
        preferences = Menu(menubar, tearoff=0)
        preferences.add_command(label="Plotter Settings", command=self.plotter_settings)
        preferences.add_command(label="Markov Chain Settings", command=self.markov_chain_settings)
        preferences.add_command(label="Report Settings", command=self.report_settings)
        preferences.add_command(label="Main Path", command=self.main_path)
        menubar.add_cascade(label="Preferences", menu=preferences)

        # MACHINE LEARNING OPTIONS
        machine_learning = Menu(menubar, tearoff=0)
        machine_learning_tools = Menu(menubar, tearoff=0)
        machine_learning_tools_generate = Menu(menubar, tearoff=0)
        machine_learning_settings = Menu(menubar, tearoff=0)

        machine_learning_tools_generate.add_command(label="Initial State", command=self.generate_vector)
        machine_learning_tools_generate.add_command(label="Transition Matrix", command=self.generate_matrix)

        machine_learning.add_command(label="Markov Chain", command=self.markov_chain_menu)
        machine_learning_tools.add_cascade(label="Generate Random", menu=machine_learning_tools_generate)
        machine_learning.add_cascade(label="Tools", menu=machine_learning_tools)
        menubar.add_cascade(label="Machine Learning", menu=machine_learning)

        # PLOTTER OPTIONS
        plotter_settings = Menu(menubar, tearoff=0)
        plotter_settings.add_command(label="Create Plot", command=self.create_plot)
        menubar.add_cascade(label="Plotter", menu=plotter_settings)

        # HELP MENU
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.load)
        menubar.add_cascade(label="Help", menu=helpmenu)

        # display the menu
        master.config(menu=menubar)

        '''# DECLARING THE LISTS AND DICTIONARIES THAT WILL HOLD INFORMATION
        self.parameterLabel, self.parameterLabel2 = [], []  # This list will hold the labels
        self.label_text_List, self.label_text_List2 = [], []  # This list will hold the text which will go on each label
        self.parameterNames, self.parameterNumbers = {}, {}  # This will hold the name/number of each parameter
        self.inputValidationInformation = {}  # This will hold a dictionary in which keys are item name and values are lists of allowable digits
        self.entryList = {}  # This will hold the name of each entry text field as well as the field itself
        self.entries = []
        self.query = {}
        # POPULATING PARAMETER NUMBERS
        self.parameterNumbers = self.list_generator.getParameterNumbers()

        # POPULATING THE DICTIONARY WHICH WILL HOLD THE ENTRY TEXT FIELDS AND GENERATING THE PARAMETER NAMES LIST
        self.parameterNames = self.list_generator.getParameterNames()

        self.entryList = self.list_generator.getEntryList()
        self.files = 'single'
        self.filename = 'report'

        # POPULATING THE DICTIONARY WHICH WILL HOLD THE VALIDATION INFORMATION
        for i in range(self.number_of_items):
            temporaryAllowableValuesObject = AllowableValueListGenerator.AllowableValueListGenerator(self.MASTERPATH,
                                                                                                     self.parameterNames[
                                                                                                         i], i)
            self.inputValidationInformation[self.parameterNumbers[
                i]] = temporaryAllowableValuesObject.getAllowableValuesList()  # "LIST OF ALLOWABLE values HERE"
            # SETTING THE ENTRY TEXT FIELDS
            if self.parameterNames[i + 0] != '(Reserved)':
                self.entry = self.entryList[self.parameterNames[i + 0]]
                self.entries.append(self.entry)
                self.entry.grid(row=i + 3, column=3, columnspan=1)
                self.entryList[self.parameterNames[i]].insert(0, '')
            # POPULATING THE LISTS WHICH WILL HOLD THE LABELS, LABEL TEXT, AND TEXT ENTRY FIELDS
            self.label_text_List.append(StringVar())
            self.parameterLabel.append(Label(self.frame.interior, textvariable=self.label_text_List[i]))
            self.label_text_List2.append(StringVar())
            self.parameterLabel2.append(Label(self.frame.interior, textvariable=self.label_text_List2[i]))
            # SETTING THE LABELS AND TEXT ENTRY FIELDS IN THE MENU
            self.parameterName = self.parameterNames[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel[i]  # Set the parameter label from the parameter label list
            self.label_text = self.label_text_List[i]
            self.label_text.set(self.parameterName)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 3, column=1, columnspan=1, sticky=W)  # Set the label location
            # SETTING THE LABELS OF ITEM NUMBER
            self.parameterNumber = self.parameterNumbers[i]  # Set the parameter name from the parameter names list
            self.label = self.parameterLabel2[i]  # Set the parameter label from the parameter label list
            self.label_text2 = self.label_text_List2[i]
            self.label_text2.set(self.parameterNumber)  # set the text of the parameter label from the label text list
            self.label.grid(row=i + 3, column=0, columnspan=1, sticky=W)  # Set the label location
'''
        # SEPARATOR
        separator = ttk.Separator(self.master, orient=HORIZONTAL)
        separator.grid(row=1, column=0, columnspan=5, sticky=W + E)

        # SETTING THE BUTTONS
        self.search_button = Button(master, text="Search", command=self.search, cursor='hand2')
        self.search_button.grid(row=2, column=4, sticky=SE)
        self.clear_button = Button(master, text="Clear", command=self.clear, cursor='hand2')
        self.clear_button.grid(row=2, column=0, sticky=SW)

        # self.list_generator.getParameterNamesWithRespectToNumber()

        # self.master.mainloop()

    def save(self):
        path = tkf.asksaveasfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            if path[path.__len__() - 4] is not ".":
                path = path + ".txt"

            with open(path, "w") as saveFile:
                for i in range(self.number_of_items):
                    pass  # saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")
        except:
            return

    def load(self):
        path = tkf.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        try:
            with open(path, "r") as loadFile:
                for i in range(self.number_of_items):
                    pass  # self.entryList[self.parameterNames[i + 0]].delete(0, END)
                for i, line in enumerate(loadFile):
                    pass  # self.entryList[self.parameterNames[i]].insert(i, line.strip())

        except:
            return

    def search(self):
        self.name_report()

    def search_resume(self):

        with open(self.MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "w") as saveFile:
            for i in range(self.number_of_items):
                pass  # saveFile.write(self.entryList[self.parameterNames[i]].get() + "\n")

        with open(self.MASTERPATH + "\\BridgeDataQuery\Database\\temp.txt", "r") as loadFile:
            for i, line in enumerate(loadFile):
                '''if str(self.parameterNames[i]) == '(Reserved)':
                    self.query[self.parameterNames[i] + "_" + str(i)] = (line.strip())
                else:
                    self.query[self.parameterNames[i]] = (line.strip())'''

        if self.dataValidator():
            self.list_generator.setSearchList(self.query)
            Search.Search(self)

    def clear(self):
        for i in range(self.number_of_items):
            pass  # self.entryList[self.parameterNames[i + 0]].delete(0, END)

    def dataValidator(self):
        '''for i in range(self.number_of_items):
            if self.parameterNames[i] == '(Reserved)':
                pass
            else:
                if self.query[self.parameterNames[i]] != '':
                    if 'None' not in self.inputValidationInformation[self.parameterNumbers[i]]:
                        if True:  # self.query[self.parameterNames[i]] in self.inputValidationInformation[self.parameterNumbers[i]]:
                            pass
                        else:
                            print(str(self.query[self.parameterNames[i]]) + " is not a valid " + str(
                                self.parameterNames[i]))  # + ". Search Aborted.")
                            pass'''
        return True

    def name_report(self):
        NameReport.NameReport(self)

    def report_settings(self):
        ReportSettings.ReportSettings(self)

    def markov_chain_menu(self):
        MarkovChainMenu.MarkovChainMenu(self)

    def markov_chain_settings(self):
        MarkovChainSettings.MarkovChainSettings(self.master, self.MASTERPATH)

    def create_plot(self):
        PlotterSettings.Plot(self.master, self.MASTERPATH)

    def plotter_settings(self):
        PlotterSettings.PlotterSettings(self.master, self.MASTERPATH)

    def main_path(self):
        MainPathSettings.MainPathSettings(self.master, self.MASTERPATH)

    def generate_vector(self):
        GeneratorSettings.SettingsGenerator(self.master, self.MASTERPATH, 0)

    def generate_matrix(self):
        GeneratorSettings.SettingsGenerator(self.master, self.MASTERPATH, 1)

    def set_filepath(self, filepath):
        self.filepath = filepath


class CreateWindow:
    def __init__(self, Main, window):
        self.Main = Main
        self.Main.master.title("Christian's Machine")
        # self.root = Tk()
        # self.Main.master.iconbitmap('C:\\Users\\frano\PycharmProjects\BridgeDataQuery\Interface\InterfaceUtilities\Icons\i.ico')
        self.setMaster(self.Main.master)
        self.set_gui(window)
        self.run()

    def set_gui(self, window):
        if window == 0:
            self.my_gui = BridgeDataQuery(self.Main.master)

    def setMaster(self, master):
        # global root
        self.root = master

    def getMaster(self):
        return self.root

    def run(self):
        self.Main.master.mainloop()


'''from interface7.Menu import Menu


object_oriented PopulateReportMenu(Menu):
    def populate(self):
        self.settings = [True, False, False, True]
        """Load report preferences"""
        with open(self.util.MASTERPATH + self.util.r_s_p, "r") as report_settings:
            for i, line in enumerate(report_settings):
                self.settings[i] = True if line.strip() == 'True' else False

        if self.parent.indicator==0:
            # Entries
            self.util.entry(container=self.top, width=[20], row=[2], column=[1], columnspan=[1], sticky='e')
            self.bt,self.bcmds,self.br,self.bc,self.bs=["Browse","Run","Cancel"],[self.parent.browse,self.parent.run,self.cancel],[2,5,5],[3,3,1],['w','w','e']
            self.lt,self.lr,self.lc,self.lcs,self.ls=["CSV:","Dictionary:","Single file:","Multiple files:","File path:"],[1, 1, 1, 1, 2],[1, 2, 1, 2, 1],[2, 2, 1, 1, 1],['w']*5

        elif self.parent.indicator==1:
            self.bt,self.bcmds,self.br,self.bc,self.bs=["Apply","Cancel"],[self.parent.apply,self.cancel],[5,5],[3,1],['w','e']
            self.lt,self.lr,self.lc,self.lcs,self.ls=["CSV:","Dictionary:","Single file:","Multiple files:"],[1, 1, 1, 1],[1, 2, 1, 2],[2, 2, 1, 1],['w']*3

        # label
        self.util.label(self.top, text=self.lt, row=self.lr, column=self.lc, columnspan=self.lcs, sticky=self.ls)
        # Separators
        self.util.separator(container=self.top, orient='h', row=3, column=1, columnspan=3, sticky='we')
        # Buttons
        self.util.button(container=self.top, text=self.bt, commands=self.bcmds, cursor='hand2', row=self.br, column=self.bc, sticky=self.bs)
        # Check Buttons
        self.util.check_button(container=self.top, commands=[(lambda:self.toggle_check_button(0)),(lambda:self.toggle_check_button(1)),(lambda:self.toggle_check_button(1)),(lambda:self.toggle_check_button(2))], cursor='left_ptr', row=[1, 1, 1, 1], column=[1, 3, 1, 3], sticky=['e', 'e', 'e', 'e'])

        for i in range(3):
            if self.settings[i]:
                self.select_check_button(i)'''

'''import tkinter.ttk as ttk
from tkinter import *

from interface7.Menus.Menu import Menu


object_oriented PopulateMarkovChainMenu(Menu):
    def populate(self):
        self.settings = [True, False]
        self.paths = 1

        self.util.lists.setNonNoneValueList()
        # Create a Tkinter variable to store the item
        self.item = StringVar(self.top)
        self.i_s = StringVar(self.top)

        # Dictionary with options
        self.vina = self.util.lists.getNonNoneName()
        self.vinu = self.util.lists.getNonNoneNumber()
        self.item.set('5A')  # set the default option
        self.i_s.set('None')

        # Loading user preferences
        with open(self.util.MASTERPATH + self.util.mc_s_p, "r") as report_preferences:
            self.settings = [True if line.strip() == 'True' else False for line in report_preferences]

        # label
        self.util.label(container=self.top, text=["Input Paths:", "Item:", "Initial State:", "Iterations:"], row=[1, self.paths + 1, self.paths + 1, self.paths+2], column=[1]*3, columnspan=[1]*3, sticky=['w']*5)
        # Entries (File Paths, Iterations)
        self.util.entry(container=self.top, width=[15,15], row=[1, self.paths + 2], column=[2]*1, columnspan=[1,1], sticky=['we']*1)
        # ITEM OF INTEREST
        popupMenu1 = ttk.OptionMenu(self.top, self.item, *self.vinu, command=self.item_selection)
        popupMenu1.grid(row=self.paths + 1, column=1, columnspan=1, sticky='we')
        # INITIAL STATE
        self.item_selection('5A')
        # Separators
        self.util.separator(container=self.top, orient='h', row=self.paths + 5, column=1, columnspan=5, sticky='we')
        # Buttons
        self.util.button(container=self.top, text=["Run", "Cancel", "Open"], commands=[self.parent.run, self.cancel, self.parent.load], cursor='hand2', row=[self.paths + 6, self.paths+6, 1], column=[3, 1, 3], sticky=['e', 'e', 'e'])

    def item_selection(self, value):
        self.parameterNumbersDictionary = self.util.lists.getParameterNumbersDictionary()
        self.util.lists.valid_lists(self.parameterNumbersDictionary[str(self.item.get())])
        self.vs = self.util.lists.get_valid_values_list()
        self.i_s.set(self.vs[0])

        popupMenu2 = ttk.OptionMenu(self.top, self.i_s, *self.vs)
        popupMenu2.grid(row=self.paths + 1, column=1, columnspan=1, sticky='we')'''

from Utilities import MasterPath
import ctypes

from _misc._utilities import MasterPath, ListGenerator

user32 = ctypes.windll.user32

MasterPath_Object = MasterPath.MasterPath()


class c_lo_globalUtilities:
    def __init__(self):
        self.MASTERPATH = MasterPath_Object.getMasterPath()
        self.number_of_items = 142
        self.list_generator = ListGenerator.ListGenerator(self)
        # self.WIDTH = user32.GetSystemMetrics(0)
        # self.HEIGHT = user32.GetSystemMetrics(1)
        self.SMALL_FONT = ("Verdana", 8)
        self.MEDIUM_FONT = ("Verdana", 10)
        self.LARGE_FONT = ("Verdana", 12)

    def add_button(self, container, amount, text, commands, cursor, row, column, sticky):
        """Add a button"""
        for index, button in enumerate(range(amount)):
            Button(container, text=text[index], command=commands[index], cursor=cursor).grid(
                row=row[index],
                column=column[index],
                sticky=sticky[index]
            )

    def add_check_button(self, container, amount, commands, cursor, row, column, sticky):
        """Add a button"""
        for index, button in enumerate(range(amount)):
            Button(container, command=commands[index], cursor=cursor).grid(
                row=row[index],
                column=column[index],
                sticky=sticky[index]
            )

    def add_label(self, container, amount, text, row, column, columnspan, sticky):
        for index, label in enumerate(range(amount)):
            Label(container, text=text[index]).grid(
                row=row[index], column=column[index], columnspan=columnspan[index],
                sticky=sticky[index]
            )

    def add_separator(self, container, orient, row, column, columnspan, sticky):
        """Add a separator"""
        ttk.Separator(container, orient=orient).grid(row=row, column=column, columnspan=columnspan, sticky=sticky)

    def add_drop_down_menu(self, parent, menus, tearoff, labels, commands):
        "Add a drop down menu"
        menubar = Menu(parent.container)
        for i, menu in enumerate(menus):
            menu = Menu(menubar, tearoff=tearoff[i])
            index = 0
            for label in (labels[i]):
                if label == 'add_separator()':
                    menu.add_separator()
                else:
                    menu.add_command(label=label, command=commands[i][index])
                    index += 1
            menubar.add_cascade(label=menus[i], menu=menu)

        parent.config(menu=menubar)

    def set_active(self, top):
        top.grab_set()


import tkinter.ttk as ttk
from tkinter import *

from Utilities21 import MasterPath

# user32 = ctypes.windll.user32

MasterPath_Object = MasterPath.MasterPath()


class c_lo_globalUtilities:
    def __init__(self):
        self.MASTERPATH = MasterPath_Object.getMasterPath()
        self.number_of_items = 142
        self.list_generator = ListGenerator.ListGenerator(self)
        # self.WIDTH = user32.GetSystemMetrics(0)
        # self.HEIGHT = user32.GetSystemMetrics(1)
        self.SMALL_FONT = ("Verdana", 8)
        self.MEDIUM_FONT = ("Verdana", 10)
        self.LARGE_FONT = ("Verdana", 12)

    def add_button(self, container, amount, text, commands, cursor, row, column, sticky):
        """Add a button"""
        for index, button in enumerate(range(amount)):
            Button(container, text=text[index], command=commands[index], cursor=cursor).grid(
                row=row[index],
                column=column[index],
                sticky=sticky[index]
            )

    def add_check_button(self, container, amount, commands, cursor, row, column, sticky):
        """Add a button"""
        for index, button in enumerate(range(amount)):
            Button(container, command=commands[index], cursor=cursor).grid(
                row=row[index], column=column[index],
                sticky=sticky[index]
            )

    def add_label(self, container, amount, text, row, column, columnspan, sticky):
        for index, label in enumerate(range(amount)):
            Label(container, text=text[index]).grid(
                row=row[index], column=column[index], columnspan=columnspan[index],
                sticky=sticky[index]
            )

    def add_separator(self, container, orient, row, column, columnspan, sticky):
        """Add a separator"""
        ttk.Separator(container, orient=orient).grid(row=row, column=column, columnspan=columnspan, sticky=sticky)

    def add_drop_down_menu(self, parent, menus, tearoff, labels, commands):
        "Add a drop down menu"
        menubar = Menu(parent.container)
        for i, menu in enumerate(menus):
            menu = Menu(menubar, tearoff=tearoff[i])
            index = 0
            for label in (labels[i]):
                if label == 'add_separator()':
                    menu.add_separator()
                else:
                    menu.add_command(label=label, command=commands[i][index])
                    index += 1
            menubar.add_cascade(label=menus[i], menu=menu)

        parent.config(menu=menubar)

    def set_active(self, top):
        top.grab_set()


class c_lo_globalUtilities:
    def __init__(self):
        self.MASTERPATH = MasterPath_Object.getMasterPath()
        self.number_of_items = 142
        self.list_generator = ListGenerator.ListGenerator(self)
        self.WIDTH = user32.GetSystemMetrics(0)
        self.HEIGHT = user32.GetSystemMetrics(1)
        self.SMALL_FONT = ("Verdana", 8)
        self.MEDIUM_FONT = ("Verdana", 10)
        self.LARGE_FONT = ("Verdana", 12)


parameters, start, end = [], [], []


class textParser():

    def __init__(self):
        with open(
            'instructions.txt', 'r'
        ) as iFile:  # The parameters required for extraction are read from the "instructions" file
            for line in iFile:
                parameters.append(line.rstrip())
        self.selectFile()

    def selectFile(self):
        path = parameters[0]  # The parameters required for the extraction are read from the parameters array
        newPath = path + "Reduced.txt"
        path = path + ".txt"
        iterations = int(parameters[1])
        for i in range(2 * iterations):
            if not (i % 2):  # Reading the starting and ending points of each extraction
                start.append(parameters[i + 2])
            else:
                end.append(parameters[i + 2])
        self.openFile(path, newPath, iterations)

    def openFile(self, path, newPath, iterations):
        rFile = open(path, 'r')  # Open the file in read mode
        # with open(path, 'r') as rFile:
        self.readFile(newPath, rFile, iterations)  # Call the read file function with the opened file as the argument

    def readFile(self, newPath, rFile, iterations):
        fullData, reducedData = [], []
        for line in rFile:
            fullData.append(line.rstrip())  # Populate each index in the fullData array with a line from the file
        k = 0
        for j in range(iterations):
            s = start[j]
            e = end[j]
            for i in range(k, fullData.__len__()):
                if fullData[i] == s:  # The reducedData array will be populated once the specified section is found
                    while fullData[i] != e:
                        reducedData.append(fullData[i].rstrip())
                        i += 1
                    if e != end[end.__len__() - 1]:
                        break
                elif fullData[i] == end[
                    end.__len__() - 1]:  # When the specified ending section is found, the extraction process terminates
                    rFile.close()
                    self.createFile(newPath, reducedData)
                    return

            k = i - 3

    def createFile(self, newPath, reducedData):
        print('New file created.')
        wFile = open(newPath, "w")  # Create a new file to store the reduced data
        self.writeFile(
            reducedData, wFile
        )  # Call the write file function with the reduced data and nnew file as arguments

    def writeFile(self, reducedData, wFile):
        for i in range(reducedData.__len__()):
            wFile.write(reducedData[i] + "\n")  # Write the reduced data into the new file
            i += 1

        wFile.close()


tP = textParser()

parameters, start, end = [], [], []


class textParser():

    def __init__(self):
        with open(
            'instructions.txt', 'r'
        ) as iFile:  # The parameters required for extraction are read from the "instructions" file
            for line in iFile:
                parameters.append(line.rstrip())
        self.selectFile()

    def selectFile(self):

        path = parameters[0]  # The parameters required for the extraction are read from the parameters array
        path = path[:(path.__len__() - 4)]
        newPath = path + "Reduced.txt"
        path = path + ".txt"
        iterations = int(parameters[1])

        for i in range(2 * iterations):

            if not (i % 2):  # Reading the starting and ending points of each extraction
                start.append(parameters[i + 2])
            else:
                end.append(parameters[i + 2])

        self.openFile(path, newPath, iterations)

    def openFile(self, path, newPath, iterations):

        # rFile = open(path, 'r')                         #Open the file in read mode
        with open(path, 'r') as rFile:
            self.readFile(
                newPath, rFile, iterations
            )  # Call the read file function with the opened file as the argument

    def readFile(self, newPath, rFile, iterations):

        i = 0
        reducedData = []
        parsing = False

        for line in rFile:

            tempS = start[i]  # Store the current iteration's starting point
            tempE = end[i]  # Store the current iteration's ending point
            smallS = 0
            smallE = 0
            word = ''
            if (
                parsing == False and tempS in line and tempE in line):  # Look for the starting token in each line of the document
                parsing = True
                print(line)
                for w in range(line.__len__()):
                    if (line[w] == tempE[0]):
                        w += 1
                        if tempE.__len__() == 1:
                            smallE = w
                        else:
                            for k in range(1, tempE.__len__()):
                                if (line[w] == tempE[k]):
                                    w += 1
                                    k += 1

                                    if (k == tempE.__len__()):
                                        smallE = w
                                        break
                                else:
                                    break
                        if (k == tempE.__len__()):
                            break

                word = line[:smallE]
                reducedData.append(word.strip())
                parsing = False

                if (i < (iterations - 1)):
                    i += 1
                else:
                    self.createFile(newPath, reducedData)
                    return

            elif (
                parsing == False and tempS in line and tempE not in line):  # Look for the starting token in each line of the document
                parsing = True
                print("Start")
                for w in range(line.__len__()):
                    if (line[w] == tempS[0]):
                        smallS = w
                        w += 1
                        if tempS.__len__() == 1:
                            pass
                        else:
                            for k in range(1, (tempS.__len__())):
                                if (line[w] == tempS[k]):
                                    w += 1
                                    k += 1

                                    if (k == tempS.__len__()):
                                        break
                                else:
                                    break
                        if (k == tempS.__len__()):
                            break

                word = line[smallS:]
                reducedData.append(word.strip())

            elif (parsing == True and tempE not in line):
                reducedData.append(line)

            elif (parsing == True and tempE in line):
                print("end")
                for w in range(line.__len__()):
                    if (line[w] == tempE[0]):
                        w += 1
                        if tempE.__len__() == 1:
                            smallE = w
                        else:
                            for k in range(1, tempE.__len__()):
                                if (line[w] == tempE[k]):
                                    w += 1
                                    k += 1

                                    if (k == tempE.__len__()):
                                        smallE = w
                                        break
                                else:
                                    break
                        if (k == tempE.__len__()):
                            break

                word = line[:smallE]
                reducedData.append(word.strip())
                parsing = False

                if (i < (iterations - 1)):
                    i += 1
                else:
                    self.createFile(newPath, reducedData)
                    return

    def start_end(self, tempList, tempList_length, tKey):
        for j in range(tempList_length):
            if tempList[j] == tKey:
                return j

    def createFile(self, newPath, reducedData):

        print('New file created.')
        wFile = open(newPath, "w")  # Create a new file to store the reduced data
        self.writeFile(
            reducedData, wFile
        )  # Call the write file function with the reduced data and nnew file as arguments

    def writeFile(self, reducedData, wFile):
        wFile.write("---------------------------------\n")
        for i in range(reducedData.__len__()):
            wFile.write(reducedData[i] + "\n")  # Write the reduced data into the new file
            i += 1
        wFile.write("---------------------------------\n")
        wFile.close()


import tkinter.ttk as ttk
from tkinter import *
from tkinter import filedialog as tkf


class NameReport():
    def __init__(self, parent):
        self.top = Toplevel(self.parent.master)
        self.top.grab_set()
        self.top.title("report_generator0 Options")
        self.settings = [True, False, False, True]

        with open(
            self.parent.util.MASTERPATH + "\BridgeDataQuery\\Utilities14\Preferences\Report_Preferences.txt",
            "r"
        ) as report_preferences:
            index = 0
            for line in report_preferences:
                if line.strip() == 'True':
                    self.settings[index] = True
                elif line.strip() == 'False':
                    self.settings[index] = False
                index += 1

        # CSV EXPORT
        label1 = Label(self.top, text="Export as comma separated value (csv) file:")
        label1.grid(row=1, column=1, columnspan=3, sticky=W)

        self.csv_button = Checkbutton(self.top, command=self.toggle_csv, cursor='left_ptr')
        self.csv_button.grid(row=1, column=4, sticky=E)

        # MACHINE LEARNING EXPORT
        label2 = Label(self.top, text="Export as machine learning input file:")
        label2.grid(row=2, column=1, columnspan=3, sticky=W)

        self.machine_button = Checkbutton(self.top, command=self.toggle_machine, cursor='left_ptr')
        self.machine_button.grid(row=2, column=4, sticky=E)

        # SINGLE VS MULTIPLE FILE OUTPUT
        label3 = Label(self.top, text="Single file:")
        label3.grid(row=3, column=1, sticky=W)

        self.single_file_button = Checkbutton(self.top, command=self.toggle_single, cursor='left_ptr')
        self.single_file_button.grid(row=3, column=2, sticky=E)

        label3 = Label(self.top, text="Multiple Files:")
        label3.grid(row=3, column=3, sticky=W)

        self.multiple_file_button = Checkbutton(self.top, command=self.toggle_multiple, cursor='left_ptr')
        self.multiple_file_button.grid(row=3, column=4, sticky=E)

        # FILE PATH INPUT
        label4 = Label(self.top, text="File Path:")
        label4.grid(row=4, column=1, sticky=W)

        self.file_path_input = Entry(self.top, width=30)
        self.file_path_input.grid(row=4, column=2, columnspan=2, sticky=E)

        file_path_browse = Button(self.top, text="Browse", command=self.Browse, cursor='hand2')
        file_path_browse.grid(row=4, column=4, sticky=E)

        # SEPARATOR
        separator = ttk.Separator(self.top, orient=HORIZONTAL)
        separator.grid(row=5, column=1, columnspan=4, sticky=W + E)

        run_button = Button(self.top, text="Run", command=self.run, cursor='hand2')
        run_button.grid(row=6, column=3, sticky=E)

        cancel_button = Button(self.top, text="Cancel", command=self.cancel, cursor='hand2')
        cancel_button.grid(row=6, column=4, sticky=E)

        if self.settings[0]:
            self.csv_button.select()

        if self.settings[1]:
            self.machine_button.select()

        if self.settings[2]:
            self.single_file_button.select()

        if self.settings[3]:
            self.multiple_file_button.select()

        self.top.grab_set()


import csv
from tkinter import *

from Utilities.Generators.Constants import *


class ListGenerator:
    def __init__(self, parent):
        self.parent = parent

    def setSearchMenuLists(self, populator):
        """
        Lists to use for populating menu options
        """
        self.numbers, self.names, self.entryList = {}, {}, {}
        with open(MASTER + NUMBERS, "r") as parameterNumbersFile:
            with open(MASTER + ITEM_NAMES, "r") as f:
                k = 0
                for line, lineNum in (zip(f, parameterNumbersFile)):
                    self.entryList[line.strip()] = ttk.Entry(populator.frame.interior)
                    self.numbers[line.strip()] = lineNum.strip()
                    self.names[k] = line.strip()
                    k += 1

    def setNonNoneValueList(self):
        """
        List of items that can transition through states
        """
        self.nonNoneValueList, self.nonNoneName = [], []
        with open(MASTER + VALUES, "r") as parameterValuesFile:
            with open(MASTER + NAMES, "r") as parameterNamesFile:
                for name, value in zip(csv.reader(parameterNamesFile), (csv.reader(parameterValuesFile))):
                    if name[0] != FOLDER and name[0] != FILE and "None" not in value:
                        self.nonNoneValueList.append(name[0])
        self.nonNoneName = [self.names[i] for i in self.names if (self.names[i]) in self.nonNoneValueList]

    def get_valid_lists(self, index):
        """
        List of items corresponding to a specific item in the nonNoneValueList
        """
        self.validValues = ''
        with open(MASTER + VALUES, 'r') as parameterValuesFile:
            with open(MASTER + NAMES, 'r') as parameterNamesFile:
                for omega, row in zip(csv.reader(parameterNamesFile), csv.reader(parameterValuesFile)):
                    if omega[0] == index:
                        self.validValues = row
                        break
        return self.validValues

    def state_code_state(self):
        self.states = {}
        with open(MASTER + STATES, 'r') as states:
            with open(MASTER + STATE_CODES, 'r') as state_codes:
                for state, state_code in zip(states, state_codes):
                    self.states[state_code.strip()] = state.strip()
        return self.states


class ListGenerator:
    def __init__(self, parent):
        self.numbers, self.names, self.entryList, self.names_numbers = {}, {}, {}, {}
        self.parent = parent
        self.values_path = r"\BridgeDataQuery\Interface\InterfaceUtilities\Labels\parameterValues.txt"
        self.names_path = r"\BridgeDataQuery\Database\ItemNames.txt"

    def setSearchMenuLists(self, populator):
        self.populator = populator
        with open(
            self.parent.MASTERPATH + "\BridgeDataQuery\Interface\InterfaceUtilities\Labels\parameterNumbers.txt",
            "r"
        ) as parameterNumbersFile:
            with open(self.parent.MASTERPATH + "\BridgeDataQuery\Database\ItemNames.txt", "r") as f:
                k = 0
                for line, line_num in (zip(f, parameterNumbersFile)):
                    self.entryList[line.strip()] = ttk.Entry(self.populator.frame.interior)
                    self.numbers[line.strip()] = line_num.strip()
                    self.names[k] = line.strip()
                    k += 1

    def getParameterNumbers(self):
        return self.numbers

    def getParameterNames(self):
        return self.names

    def getEntryList(self):
        return self.entryList

    def getParameterNumbersDictionary(self):
        return self.numbers

    def getParameterNamesWithRespectToNumber(self):
        return {self.numbers[number]: self.names[number] for number in self.numbers}

    def setNonNoneValueList(self):
        # A list of items that can transition through states
        self.nonNoneValueList, self.nonNoneNumber, self.nonNoneName = [], [], []

        with open(self.parent.MASTERPATH + self.values_path, "r") as parameterValuesFile:
            with open(self.parent.MASTERPATH + self.names_path, "r") as parameterNamesFile:
                for name, value in zip(csv.reader(parameterNamesFile), (csv.reader(parameterValuesFile))):
                    if name[0] != 'YEAR' and name[0] != "STATE_CODE_001" and "None" not in value:
                        self.nonNoneValueList.append(name[0])

        self.nonNoneName = [self.names[i] for i in self.names if (self.names[i]) in self.nonNoneValueList]
        self.nonNoneNumber = [self.numbers[name] for name in self.numbers if (name) in self.nonNoneValueList]

    def getNonNoneValueList(self):
        return (self.nonNoneValueList)

    def getNonNoneName(self):
        return (self.nonNoneName)

    def getNonNoneNumber(self):
        return (self.nonNoneNumber)

    # A list of items corresponding to a specific item in the nonNoneValueList
    def valid_lists(self, index):
        self.index = index
        self.valid_values = ''
        with open(self.parent.MASTERPATH + self.values_path, "r") as parameterValuesFile:
            with open(self.parent.MASTERPATH + self.names_path, "r") as parameterNamesFile:
                for omega, row in zip(csv.reader(parameterNamesFile), csv.reader(parameterValuesFile)):
                    if omega[0] == self.index:
                        self.valid_values = row
                        break

    def get_valid_values_list(self):
        return self.valid_values

    def state_code_state(self):
        self.states = {}
        states_path = self.parent.MASTERPATH + '\BridgeDataQuery\Database\Information\\allStates.txt'
        state_codes_path = self.parent.MASTERPATH + '\BridgeDataQuery\Database\Information\\allStateCodes.txt'
        with open(states_path) as states:
            with open(state_codes_path) as state_codes:
                for state, state_code in zip(states, state_codes):
                    self.states[state_code.strip()] = state.strip()

        return self.states


class ListGenerator:
    def __init__(self, MASTERPATH, frame, number_of_items):
        self.frame = frame
        self.allStates, self.allStateCodes, self.parameterNumbers, self.parameterNames, self.entryList, self.parameterNamesWithRespectToNumber = [], {}, {}, {}, {}, {}
        self.statesToSearch = []
        self.MASTERPATH = MASTERPATH

        # PARAMETER NUMBERS
        with open(
            self.MASTERPATH + "\\BridgeDataQuery\Interface\Labels\parameterNumbers.txt",
            "r"
        ) as parameterNumbersFile:
            k = 0
            for line in parameterNumbersFile:
                self.parameterNumbers[k] = line.strip()
                k += 1

        # ENTRY LIST AND PARAMETER NAMES
        with open(self.MASTERPATH + "\\BridgeDataQuery\Interface\Labels\parameterNames.txt", "r") as parameterNamesFile:
            k = 0
            for line in parameterNamesFile:
                if k == number_of_items:
                    break
                self.parameterNames[k] = line.strip()
                self.entryList[line.strip()] = Entry(self.frame.interior)
                k += 1

        # A LIST OF ALL STATES
        with open(self.MASTERPATH + '\BridgeDataQuery\Database\Information\\allStates.txt', 'r') as allStates:
            self.allStates = [line.strip() for line in allStates]

        # A LIST OF ALL STATE CODES
        with open(self.MASTERPATH + '\BridgeDataQuery\Database\Information\\allStateCodes.txt', 'r') as allStateCodes:
            i = 0
            for line in allStateCodes:
                self.allStateCodes[self.allStates[i]] = line.strip()  # POPULATE A LIST OF ALL STATE CODES
                i += 1

    def getAllStates(self):
        return self.allStates

    def getAllStateCodes(self):
        return self.allStateCodes

    def getParameterNumbers(self):
        return self.parameterNumbers

    def getParameterNumbersDictionary(self):
        self.parameterNumbersDictionary = {}
        for m in range(self.parameterNumbers.__len__()):
            self.parameterNumbersDictionary[self.parameterNumbers[m]] = m + 1

        return self.parameterNumbersDictionary

    def getParameterNames(self):
        return self.parameterNames

    def getEntryList(self):
        return self.entryList

    def getParameterNamesWithRespectToNumber(self):
        self.parameterNamesWithRespectToNumber = {self.parameterNumbers[number]: self.parameterNames[number] for number
                                                  in self.parameterNumbers}
        return self.parameterNamesWithRespectToNumber

    def setSearchList(self, query):
        self.query = query
        self.search = {}
        # This for loop is generating the list of items to be searched.
        for item in self.parameterNumbers:
            if self.parameterNames[item] == '(Reserved)':
                self.search[self.parameterNumbers[item]] = ''
            else:
                self.search[self.parameterNumbers[item]] = self.query[self.parameterNames[item]]

    def getSearchList(self):
        return self.search

    def setListOfStatesToSearch(self, statesToSearch):
        self.statesToSearch = statesToSearch

    def getListOfStatesToSearch(self):
        return self.statesToSearch

    def setNonNoneValueList(self):
        self.nonNoneValueList, self.nonNoneNumber, self.nonNoneName = [], [], []

        with open(
            self.MASTERPATH + "\\BridgeDataQuery\Interface\Labels\parameterValues.txt",
            "r"
        ) as parameterValuesFile:
            omega = 1
            reader = csv.reader(parameterValuesFile)
            for row in reader:
                if omega != 1 and omega != 2 and "None" not in row:
                    self.nonNoneValueList.append(omega)

                omega += 1

        for name in self.parameterNames:
            if (name + 1) in self.nonNoneValueList:
                self.nonNoneName.append(self.parameterNames[name])
                self.nonNoneNumber.append(self.parameterNumbers[name])

    def getNonNoneValueList(self):
        return (self.nonNoneValueList)

    def getNonNoneName(self):
        return (self.nonNoneName)

    def getNonNoneNumber(self):
        return (self.nonNoneNumber)

    def AllowableValueListGenerator(self, index):
        self.index = index
        self.allowableValues = ''

        with open(
            self.MASTERPATH + "\\BridgeDataQuery\Interface\Labels\parameterValues.txt",
            "r"
        ) as parameterValuesFile:
            omega = 1
            reader = csv.reader(parameterValuesFile)
            for row in reader:
                if omega == int(self.index):
                    self.allowableValues = row
                    break
                omega += 1

    def getAllowableValuesList(self):
        return self.allowableValues


"""
Author: Christian Lozoya, 2017
"""

VALUES_PATH = r"\BridgeDataQuery\Interface\InterfaceUtilities\Labels\parameterValues.txt"
NAMES_PATH = r"\BridgeDataQuery\Interface\InterfaceUtilities\Labels\ItemNames.txt"
STATES_PATH = r'\BridgeDataQuery\Database\Information\allStates.txt'
STATE_CODES_PATH = r'\BridgeDataQuery\Database\Information\allStateCodes.txt'


class ListGenerator:
    def __init__(self, parent):
        self.numbers, self.names, self.entryList, self.namesNumbers = {}, {}, {}, {}
        self.parent = parent

    def setSearchMenuLists(self, populator):
        with open(
            self.parent.MASTERPATH + "\BridgeDataQuery\Interface\InterfaceUtilities\Labels\parameterNumbers.txt",
            "r"
        ) as parameterNumbersFile:
            with open(
                self.parent.MASTERPATH + "\BridgeDataQuery\Interface\InterfaceUtilities\Labels\ItemNames.txt",
                "r"
            ) as f:
                k = 0
                for line, lineNum in (zip(f, parameterNumbersFile)):
                    self.entryList[line.strip()] = ttk.Entry(populator.frame.interior)
                    self.numbers[line.strip()] = lineNum.strip()
                    self.names[k] = line.strip()
                    k += 1

    def get_parameter_numbers(self):
        return self.numbers

    def get_parameter_names(self):
        return self.names

    def get_entry_list(self):
        return self.entryList

    def setNonNoneValueList(self):
        # A list of items that can transition through states
        self.nonNoneValueList, self.nonNoneName = [], []

        with open(self.parent.MASTERPATH + VALUES_PATH, "r") as parameterValuesFile:
            with open(self.parent.MASTERPATH + NAMES_PATH, "r") as parameterNamesFile:
                for name, value in zip(csv.reader(parameterNamesFile), (csv.reader(parameterValuesFile))):
                    if name[0] != 'YEAR' and name[0] != "STATE_CODE_001" and "None" not in value:
                        self.nonNoneValueList.append(name[0])

        self.nonNoneName = [self.names[i] for i in self.names if (self.names[i]) in self.nonNoneValueList]

    def get_non_none_names(self):
        return (self.nonNoneName)

    # A list of items corresponding to a specific item in the nonNoneValueList
    def valid_lists(self, index):
        self.valid_values = ''
        with open(self.parent.MASTERPATH + VALUES_PATH, 'r') as parameterValuesFile:
            with open(self.parent.MASTERPATH + NAMES_PATH, 'r') as parameterNamesFile:
                for omega, row in zip(csv.reader(parameterNamesFile), csv.reader(parameterValuesFile)):
                    if omega[0] == index:
                        self.valid_values = row
                        break

    def get_valid_values_list(self):
        return self.valid_values

    def state_code_state(self):
        self.states = {}
        with open(self.parent.MASTERPATH + STATES_PATH, 'r') as states:
            with open(self.parent.MASTERPATH + STATE_CODES_PATH, 'r') as state_codes:
                for state, state_code in zip(states, state_codes):
                    self.states[state_code.strip()] = state.strip()

        return self.states


"""
Author: Christian Lozoya, 2017
"""

import tkinter.ttk as ttk


class ListGenerator:
    def __init__(self, parent):
        self.parent = parent

    def setSearchMenuLists(self, populator):
        """
        Lists to use for populating menu options
        """
        self.numbers, self.names, self.entryList = {}, {}, {}
        with open(MASTER + NUMBERS, "r") as parameterNumbersFile:
            with open(MASTER + ITEM_NAMES, "r") as f:
                k = 0
                for line, lineNum in (zip(f, parameterNumbersFile)):
                    self.entryList[line.strip()] = ttk.Entry(populator.frame.interior)
                    self.numbers[line.strip()] = lineNum.strip()
                    self.names[k] = line.strip()
                    k += 1


class ListGenerator:
    def __init__(self, parent):
        self.allStates, self.allStateCodes, self.numbers, self.names, self.entryList, self.names_numbers = [], {}, {}, {}, {}, {}
        self.states_to_search = []
        self.parent = parent

        # PARAMETER NUMBERS
        with open(
            self.parent.MASTERPATH + "\BridgeDataQuery\Interface\Labels\parameterNumbers.txt",
            "r"
        ) as parameterNumbersFile:
            for k, line in enumerate(parameterNumbersFile):
                self.numbers[k] = line.strip()

        # A LIST OF ALL STATES
        with open(self.parent.MASTERPATH + '\BridgeDataQuery\Database\Information\\allStates.txt', 'r') as allStates:
            self.allStates = [line.strip() for line in allStates]

        # A LIST OF ALL STATE CODES
        with open(
            self.parent.MASTERPATH + '\BridgeDataQuery\Database\Information\\allStateCodes.txt',
            'r'
        ) as allStateCodes:
            for i, line in enumerate(allStateCodes):
                self.allStateCodes[self.allStates[i]] = line.strip()

    def getParameterNumbers(self):
        return self.numbers

    def getParameterNames(self, populator):
        self.populator = populator
        with open(
            self.parent.MASTERPATH + "\BridgeDataQuery\Interface\Labels\parameterNames.txt",
            "r"
        ) as parameterNamesFile:
            for k, line in enumerate(parameterNamesFile):
                if k == self.parent.number_of_items:
                    break
                self.names[k], self.entryList[line.strip()] = line.strip(), Entry(self.populator.frame.interior)

        return self.names

        self.nonNoneValueList, self.nonNoneNumber, self.nonNoneName = [], [], []

        with open(
            self.parent.MASTERPATH + "\\BridgeDataQuery\Interface\Labels\parameterValues.txt",
            "r"
        ) as parameterValuesFile:

            for omega, row in enumerate(csv.reader(parameterValuesFile), start=1):
                if omega != 1 and omega != 2 and "None" not in row:
                    self.nonNoneValueList.append(omega)

        self.nonNoneName = [self.names[name] for name in self.names if (name + 1) in self.nonNoneValueList]
        self.nonNoneNumber = [self.numbers[name] for name in self.numbers if (name + 1) in self.nonNoneValueList]

    def getNonNoneValueList(self):
        return (self.nonNoneValueList)

    def getNonNoneName(self):
        return (self.nonNoneName)

    def getNonNoneNumber(self):
        return (self.nonNoneNumber)

    # A list of items corresponding to a specific item in the nonNoneValueList
    def valid_list_generator(self, index):
        self.index = index
        self.valid_values = ''
        with open(
            self.parent.MASTERPATH + "\\BridgeDataQuery\Interface\Labels\parameterValues.txt",
            "r"
        ) as parameterValuesFile:
            for omega, row in enumerate(csv.reader(parameterValuesFile), start=1):
                if omega == int(self.index):
                    self.valid_values = row
                    break

    def get_valid_values_list(self):
        return self.valid_values
