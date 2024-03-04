# import csv
# import gzip
# import json
# import random
# import re
# import shutil
# import time
# from abc import ABCMeta
# from datetime import datetime
# from uuid import UUID
# import unicodedata
# import difflib
#
# try:
#     from StringIO import StringIO
# except ImportError:
#     from io import StringIO
# import pandas as pd
# from PyQt5.QtWidgets import (QMessageBox)
# from matplotlib import pyplot as plt
# from numba import *
# from openpyxl import load_workbook
# from scipy.io import loadmat, wavfile
# from sklearn.svm import SVR
#
# from lozoya.__variable import *

# TODO WRITE BETTER DOCSTRING
# def get_all_subclasses(cls):
#     return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in get_all_subclasses(c)])
#
#


v = pd.Series([1, 1, 333, 364, 321, 386, 365, 389, 399, 999, 10002])
ENCODING = 'latin1'
FILE = 'STATE_CODE_001'
ID = 'STRUCTURE_NUMBER_008'
LATITUDE = 'LAT_016'
LONGITUDE = 'LONG_017'
MASTERPATH = 'C:\\Users\\frano\PycharmProjects'
NA_VALUES = ['N']
TXT_EXT = '.txt'
collect_headers("C:\\Users\\frano\PycharmProjects\BigData\Database\Years")
# List of items that can transition through states
tmp = []
with open(VALUES, "r") as parameterValuesFile:
    with open(ITEM_NAMES, "r") as parameterNamesFile:
        for name, value in zip(csv.reader(parameterNamesFile), (csv.reader(parameterValuesFile))):
            if "None" not in value:
                tmp.append(name[0])
LEARNABLE = [FEATURES[i] for i in FEATURES if (FEATURES[i]) in tmp]
id = 'STRUCTURE_NUMBER_008'
FOLDER = 'YEAR'

states = []
stateCodes = {}
years_to_search = []
bridgesInState = {}
itemsInBridge = {}
years = []  # This will hold a list of the years for which data is available
starting_year = 1992  # This is the first year in the database
number_of_years = 25  # The number of years in the database
current_year = datetime.datetime.now().year - 1
for i in range(number_of_years):  # Populating the list with the years
    years.append(i + starting_year)
incompleteDataFrame = pd.DataFrame(
    [[234, 'Rudy', '6', None], [np.nan, 193, 'mar co', '$'], [True, 16.89, None, ' '], [-5, False, not True, '666.1']]
)
print(incompleteDataFrame)
print("\n")
print(remove_data(incompleteDataFrame, 1, 'Rudy'))
print("\n")
print(remove_data(incompleteDataFrame, 3, '$'))

collect_stats(v, 'Numerical')
# ENUMERATE OF FEATURES IN DATABASE
FEATURES = {}
with open(ITEM_NAMES, "r") as f:
    for i, line in enumerate(f):
        FEATURES[i] = line.strip()


def _csv(path, **kwargs):
    print("lil foller")
    print(path)
    print(kwargs)
    try:
        cols = pd.read_csv(path, nrows=0)
        usecols = [c for c in kwargs['columns'] if c in cols.columns]
        return pd.read_csv(
            path,  # usecols=usecols,
            # dtype=kwargs['dtype'],
            # encoding=kwargs['encoding'],
            # na_values=kwargs['naValues'],
            # sep=kwargs['separator']
        )
    except Exception as e:
        print('1', e)
    try:
        return pd.read_csv(
            path,  # dtype=kwargs['dtype']
        )
    except Exception as e:
        print('1', e)

    try:
        f = open(path)
        df = pd.read_csv(f)
        f.close()
        return df
    except Exception as e:
        print('2', e)


# def _csv(path, **kwargs):
#     try:
#         cols = pd.read_csv(path, nrows=0)
#         usecols = [c for c in columns if c in cols.columns]
#         return pd.read_csv(
#             path, usecols=usecols, dtype=kwargs['dtype'], encoding=kwargs['encoding'],
#             na_values=kwargs['naValues'], sep=kwargs['separator']
#         )
#     except Exception as e:
#         pass
#     try:
#         return pd.read_csv(path, dtype=kwargs['dtype'])
#     except Exception as e:
#         pass


def _csv(path, **kwargs):
    try:
        cols = pd.read_csv(path, nrows=0)
        usecols = [c for c in kwargs['columns'] if c in cols.columns]
        return pd.read_csv(
            path,  # usecols=usecols,
            # dtype=kwargs['dtype'],
            # encoding=kwargs['encoding'],
            # na_values=kwargs['naValues'],
            # sep=kwargs['separator']
        )
    except Exception as e:
        pass  # print('1', e)
    try:
        return pd.read_csv(
            path,  # dtype=kwargs['dtype']
        )
    except Exception as e:
        pass  # print('2', e)

    try:
        f = open(path)
        df = pd.read_csv(f)
        f.close()
        return df
    except Exception as e:
        pass  # print('3', e)


def _csv(path, **kwargs):
    return pd.read_csv(path)


def _excel(path, **kwargs):
    return pd.read_excel(path)


def _hdf5(path, **kwargs):
    return pd.read_hdf(path)


def _mat(path, **kwargs):
    x = loadmat(path)
    cols = [i[0][0] for i in x['Header'][0][0][2]]
    df = pd.DataFrame(data=x['Data'], columns=cols)
    return df


def _wav(path, **kwargs):
    return pd.DataFrame(wavfile.read(path)[1])


def adjust_settings(algorithms, models, settings):
    """
    Returns a version of each model in models that has the corresponding settings set.
    algorithms: dict, key is str (name of algorithm), value is python object
    models: list of str, names of models to use
    settings: dict of dict, outer dict key is model name, inner dict is settings for corresponding model
    return: dict, key is model name, value is model
    """
    payload = {}
    for model in models:
        payload[model] = algorithms[model](**settings[model])
    return payload


def append_to_results(requiredIDs, newRequiredIDs, results, scenario, max):
    newResults = []
    for _ in requiredIDs:
        level, key, value = _.split(',')
        for object in scenario.json:
            if key in object:
                newResults.append(object)
                results.append(object)
    navigate_dict(newRequiredIDs, newResults, 1, max)
    return results, newRequiredIDs


def collect_headers(dir, save=True):
    """
    Create a file containing all unique
    column names in the entire database
    """
    headers = []
    separator = None
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[0] != 'temp':
                header, separator = get_header(os.path.join(subdir, file))
                for column in header:
                    if column not in headers:
                        headers.append(column)
    if save:
        save_headers(headers)
    return headers, separator


def collect_headers(dir):
    """
    Create a file containing all unique
    column names in the entire database
    """
    headers = []
    separator = None
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[0] != 'temp':
                header, separator = get_header(os.path.join(subdir, file))
                for column in header:
                    if column not in headers:
                        headers.append(column)

    save_headers(headers)

    return headers, separator


# TODO WRITE BETTER DOCSTRING
def clean_data(dataFrame, columns=None):
    """
    Clears incomplete data resulting in regular matrix
    dataFrame: pandas data frame
    columns: list
    """
    cln = dataFrame
    cols = columns if columns else dataFrame.columns
    for col in cols:
        try:
            cln = (cln[np.isfinite(cln[col].astype(float))])
        except Exception as e:
            cln = cln.dropna()
    return cln


def clean_data(dataFrame, columns):
    """
    Clears incomplete data resulting in regular matrix
    data is a pandas DataFrame and columns are the columns in data
    """
    cln = dataFrame
    for column in columns:
        cln = (cln[np.isfinite(cln[column].astype(float))])
    cln.columns = columns
    return cln


def clean_data(dataFrame, columns=None):
    """
    Clears incomplete data resulting in regular matrix
    dataFrame: pandas data frame
    columns: list
    """
    cln = dataFrame
    cols = columns if columns else dataFrame.columns
    for col in cols:
        try:
            cln = (cln[np.isfinite(cln[col].astype(float))])
        except:
            pass
    return cln


def clean_data(dataFrame, columns=None):
    """
    Clears incomplete data resulting in regular matrix
    dataFrame: pandas dataFrame
    columns: list
    """
    cln = dataFrame
    cols = columns if columns else dataFrame.columns
    for col in cols:
        try:
            cln = (cln[np.isfinite(cln[col].astype(float))])
        except Exception as e:
            cln = cln.dropna()
    return cln


def clear(entries, features):
    for i in range(entries.__len__()):
        entries[features[i + 0]].setText('')


def collect_headers(dir):
    headers = []
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
    with open("C:\\Users\\frano\PycharmProjects\BigData\\Utilities14\COLUMNSTEST", "w") as file:
        for header in headers:
            file.write(header + '\n')


def collect_unique_values(file):
    """
    Create a file where each row contains all of the unique
    values encountered in the database for each column name
    """
    pass


def collect_unique_values(file):
    """
    Create a file where each row contains all of the unique
    values encountered in the database for each column name
    """
    pass


def concat_data(dataFrame, columns, index=None):
    """
    Concatenate list of pandas Series into a DataFrame
    dataFrame: is a pandas DataFrame or a list
    columns: are the columns in data
    index: used for concatenating only specific columns
    """
    if index:
        cnc = pd.DataFrame(pd.concat([d[index] for d in dataFrame], axis=1))
    else:
        cnc = pd.DataFrame(pd.concat([d for d in dataFrame], axis=1))
    cnc.columns = columns
    return cnc


def concat_data(dataFrame, columns, index=None):


def concat_data(seriesList, columns=None, axis=1):
    """
    Concatenate list of pandas Series into a DataFrame
    seriesList: list of pandas Series
    columns: list
    index is used for concatenating only specific columns
    """
    cnc = pd.DataFrame(pd.concat(seriesList, axis=axis))
    cnc.columns = columns
    return cnc


def concat_data(seriesList, columns=None, axis=1):
    """
    Concatenate list of pandas Series into a DataFrame.
    Each Series will become a column of the resulting DataFrame if axis is set to 1.
    Otherwise, if axis is set to 0, the result will be a pandas DataFrame containing one single column
    and the values of each Series in the list will be joined with respect to their original sequence.
    For test, if seriesList is:
        [pandas.Series([1, 'a', 2]), pandas.Series(['f', None])]
    and axis is set to 1, the result will be:
        0 1
    0   1 f
    1   a None
    1   2 Nan
    otherwise, if axis is set to 0, the result will be:
        0
    0   1
    1   a
    1   2
    0   f
    1   None
    seriesList: list of pandas Series
    columns: list of str - indicates the names that the columns of the resulting DataFrame will be given.
    axis: int - indicates whether to concatenate by row or by column
    return: pandas DataFrame
    """
    cnc = pd.DataFrame(pd.concat(seriesList, axis=axis))
    if columns:
        cnc.columns = columns
    return cnc


def count_files(dir):
    """
    Count the number of files in a directory.
    dir: str - indicates the path to the directory in which to count the files
    return: int
    """
    return len([dir + f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])


def create_classified_matrix(dataFrame, classes, labelKeys):
    cols = classes.unique()
    m = []
    for col in cols:
        m.append(
            pd.DataFrame(
                dataFrame[classes.isin([col])].values,
                columns=['index', labelKeys[col] if labelKeys != None else col]
            )
        )
    return m


def database_setup(dir, encoding, id, latitude, longitude):
    vars = (encoding, id, latitude, longitude)
    headers, separator = collect_headers(dir)
    with open(VARIABLES, 'w') as f:
        f.write(
            "#DATABASE\n"
            "ENCODING=" + vars[0] + "\n"
                                    "ID=" + vars[1] + "\n"
                                                      "LATITUDE=" + vars[2] + "\n"
                                                                              "LONGITUDE=" + vars[
                3] + "\n" + "NA_VALUES=" + "['N']"
                                           "\nSEPARATOR=" + str(separator) + "\n"
                                                                             "HEADERS=" + str(headers)
        )
    args = []
    for var in vars:
        if var == "None":
            args.append('')
        else:
            args.append(var.replace("'", ""))

    with open(DATABASE_VARIABLES, 'w') as f:
        f.write(args[0] + "\n" + args[1] + "\n" + args[2] + "\n" + args[3] + "\n" + dir)


def database_setup(dir, encoding, id, latitude, longitude):
    vars = (encoding, id, latitude, longitude)
    headers, separator = collect_headers(dir)
    with open(VARIABLES, 'w') as f:
        f.write(
            "#DATABASE\n"
            "ENCODING=" + vars[0] + "\n"
                                    "ID=" + vars[1] + "\n"
                                                      "LATITUDE=" + vars[2] + "\n"
                                                                              "LONGITUDE=" + vars[3] + "\n"
                                                                                                       "\nSEPARATOR=" + str(
                separator
            ) + "\n"
                "HEADERS=" + str(headers)
        )
    args = []
    for var in vars:
        if var == "None":
            args.append('')
        else:
            args.append(var.replace("'", ""))
    with open(DATABASE_VARIABLES, 'w') as f:
        f.write(args[0] + "\n" + args[1] + "\n" + args[2] + "\n" + args[3] + "\n" + dir)


def data_field(dataFrame, mode):
    if len(dataFrame.columns) == 1:
        return pd.Series(dataFrame.iloc[:, 0].as_matrix(), index=dataFrame.index)
    if mode == 'series':
        dF = pd.DataFrame(dataFrame)
    if mode == 'parallel':
        dF = pd.DataFrame(dataFrame.T)
    indices = []
    values = []
    for i, index in enumerate(dF.index):
        for j, column in enumerate(dF.columns):
            if type(dF.iloc[i, j]) != type(np.nan):
                indices.append(index)
                values.append(dF.iloc[i, j])
    prepared = pd.Series(values, index=indices).astype(float)
    return prepared


# TODO must be done before exporting results, this function will prepare data to be presentable
def denumerify(x, y, xKeys, yKeys):
    pass


def export_data(dataFrame, fileName, csv=False, excel=False, pdf=False):
    path = os.path.splitext(fileName)[0] + ' interface7 results'
    if csv:
        dataFrame.to_csv(path_or_buf=path + TXT_EXT)
    if excel:
        try:  # excel workbook exists
            writer = pd.ExcelWriter(path + XLSX_EXT, engine='openpyxl')
            writer.book = load_workbook(path + XLSX_EXT)
            writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
            dataFrame.to_excel(writer, sheet_name=fileName)
            writer.save()
        except:  # excel workbook is created if it does not exist
            writer = pd.ExcelWriter(
                os.path.splitext(fileName)[0] + ' interface7 results' + XLSX_EXT,
                engine='xlsxwriter'
            )
            dataFrame.to_excel(writer, sheet_name=fileName)
            writer.save()
    if pdf:
        pass


def export_data(dataFrame, fileName, csv=False, excel=False, pdf=False):
    if csv:
        dataFrame.to_csv(path_or_buf=os.path.splitext(fileName)[0] + ' interface7 results' + TXT_EXT)
    if excel:
        try:  # excel workbook exists
            writer = pd.ExcelWriter(os.path.join(fileName) + XLSX_EXT, engine='openpyxl')
            writer.book = load_workbook(os.path.join(fileName) + XLSX_EXT)
            writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
            dataFrame.to_excel(writer, sheet_name=fileName)
            writer.save()
        except:  # excel workbook is created if it does not exist
            writer = pd.ExcelWriter(os.path.join(fileName) + XLSX_EXT, engine='xlsxwriter')
            dataFrame.to_excel(writer, sheet_name=fileName)
            writer.save()
    if pdf:
        pass


# def export_data(data, savePath, formats, suffix=' interface7 results'):
#     """
#
#     data: pandas DataFrame or numpy array
#     savePath: str
#     formats: list of str
#     :return:
#     """
#     path = os.path.splitext(savePath)[0] + suffix
#     fileName = os.path.basename(os.path.normpath(savePath))
#
#     if 'CSV' in formats:
#         data.to_csv(path_or_buf=path + extension['txt'], index=False)
#
#     if 'Excel' in formats:
#         try:  # excel workbook exists
#             writer = pd.ExcelWriter(path + extension['excel'], engine='openpyxl')
#             writer.book = load_workbook(path + extension['excel'])
#             writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
#             data.to_excel(writer, sheet_name=fileName)
#             writer.save()
#         except:  # excel workbook is created if it does not exist
#             writer = pd.ExcelWriter(
#                 os.path.splitext(fileName)[0] + ' interface7 results' + extension['excel'],
#                 engine='xlsxwriter'
#             )
#             dataFrame.to_excel(writer, sheet_name=fileName)
#             writer.save()
#
#     if 'PDF' in formats:
#         pass
#
#     if 'WAV' in formats:
#         try:
#             wavfile.write(savePath, rate=44100, data=data.as_matrix())
#         except:
#             wavfile.write(savePath, rate=44100, data=data)


# def export_data(dataFrame, savePath, csv=False, excel=False, pdf=False):
#     path = os.path.splitext(savePath)[0] + ' interface result'
#     fileName = os.path.basename(os.path.normpath(savePath))
#
#     if csv:
#         dataFrame.to_csv(path_or_buf=path + TXT_EXT, index=False)
#
#     if excel:
#         try:  # excel workbook exists
#             writer = pd.ExcelWriter(path + XLSX_EXT, engine='openpyxl')
#             writer.book = load_workbook(path + XLSX_EXT)
#             writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
#             dataFrame.to_excel(writer, sheet_name=fileName)
#             writer.save()
#         except:  # excel workbook is created if it does not exist
#             writer = pd.ExcelWriter(
#                 os.path.splitext(fileName)[0] + ' interface7 results' + XLSX_EXT,
#                 engine='xlsxwriter'
#             )
#             dataFrame.to_excel(writer, sheet_name=fileName)
#             writer.save()
#
#     if pdf:
#         pass


# TODO This function will be redesigned to work with a single list or dict of bool instead of multiple bool values
def export_data(dataFrame, savePath, format, suffix=''):
    """
    dataFrame: pandas DataFrame
    savePath: str
    formats: list of str
    :return:
    """
    try:
        path = os.path.splitext(savePath)[0] + suffix
        fileName = os.path.basename(os.path.normpath(savePath))

        if format == 'csv':
            dataFrame.to_csv(path_or_buf=path + '.txt', index=False)

        if format == 'xlsx':
            try:  # excel workbook exists
                writer = pd.ExcelWriter(path + '.xlsx', engine='openpyxl')
                writer.book = load_workbook(path + '.xlsx')
                writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
            except:  # excel workbook is created if it does not exist
                writer = pd.ExcelWriter(path + '.xlsx', engine='xlsxwriter')
            dataFrame.to_excel(writer, sheet_name=fileName[-15:-5], index=False, )
            writer.save()
        print(fileName[-15:-5])

    except Exception as e:
        print(e)


# TODO This function will be redesigned to function with a single list or dict of bool instead of multiple bool values
def export_data(dataFrame, savePath, csv=False, excel=False, pdf=False, suffix=' interface results'):
    path = os.path.splitext(savePath)[0] + suffix
    fileName = os.path.basename(os.path.normpath(savePath))

    if csv:
        dataFrame.to_csv(path_or_buf=path + TXT_EXT, index=False)

    if excel:
        try:  # excel workbook exists
            writer = pd.ExcelWriter(path + XLSX_EXT, engine='openpyxl')
            writer.book = load_workbook(path + XLSX_EXT)
            writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
            dataFrame.to_excel(writer, sheet_name=fileName)
            writer.save()
        except:  # excel workbook is created if it does not exist
            writer = pd.ExcelWriter(
                os.path.splitext(fileName)[0] + ' interface7 results' + XLSX_EXT,
                engine='xlsxwriter'
            )
            dataFrame.to_excel(writer, sheet_name=fileName)
            writer.save()

    if pdf:
        pass


# def export_data(data, savePath, formats, suffix=' interface7 results'):
#     """
#
#     data: pandas DataFrame or numpy array
#     savePath: str
#     formats: list of str
#     :return:
#     """
#     path = os.path.splitext(savePath)[0] + suffix
#     fileName = os.path.basename(os.path.normpath(savePath))
#     if 'CSV' in formats:
#         data.to_csv(path_or_buf=path + TXT_EXT, index=False)
#     if 'Excel' in formats:
#         try:  # excel workbook exists
#             writer = pd.ExcelWriter(path + XLSX_EXT, engine='openpyxl')
#             writer.book = load_workbook(path + XLSX_EXT)
#             writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
#             data.to_excel(writer, sheet_name=fileName)
#             writer.save()
#         except:  # excel workbook is created if it does not exist
#             writer = pd.ExcelWriter(
#                 os.path.splitext(fileName)[0] + ' interface7 results' + XLSX_EXT,
#                 engine='xlsxwriter'
#             )
#             dataFrame.to_excel(writer, sheet_name=fileName)
#             writer.save()
#     if 'PDF' in formats:
#         pass
#     if 'WAV' in formats:
#         try:
#             wavfile.write(savePath, rate=44100, data=data.as_matrix())
#         except:
#             wavfile.write(savePath, rate=44100, data=data)


# TODO This function will be redesigned to function with a single list or dict of bool instead of multiple bool values
def export_data(dataFrame, savePath, formats, suffix=' interface7 results'):
    """

    dataFrame: pandas DataFrame
    savePath: str
    formats: list of str
    :return:
    """
    path = os.path.splitext(savePath)[0] + suffix
    fileName = os.path.basename(os.path.normpath(savePath))
    if 'CSV' in formats:
        dataFrame.to_csv(path_or_buf=path + TXT_EXT, index=False)
    if 'Excel' in formats:
        try:  # excel workbook exists
            writer = pd.ExcelWriter(path + XLSX_EXT, engine='openpyxl')
            writer.book = load_workbook(path + XLSX_EXT)
            writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
            dataFrame.to_excel(writer, sheet_name=fileName)
            writer.save()
        except:  # excel workbook is created if it does not exist
            writer = pd.ExcelWriter(
                os.path.splitext(fileName)[0] + ' interface7 results' + XLSX_EXT,
                engine='xlsxwriter'
            )
            dataFrame.to_excel(writer, sheet_name=fileName)
            writer.save()
    if 'PDF' in formats:
        pass


def export_results(dataFrame, state, fld1, path, file, settings):
    if not dataFrame.empty:
        if settings[0]:
            dataFrame.to_csv(path_or_buf=path + file + '_' + str(state) + str(fld1) + '.txt')
        if settings[1]:
            try:  # excel workbook exists
                writer = pd.ExcelWriter(path + file + extension['excel'], engine='openpyxl')
                writer.book = load_workbook(path + file + extension['excel'])
                writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
                dataFrame.to_excel(writer, sheet_name=str(state) + str(fld1))
                writer.save()
            except:  # excel workbook is created if it does not exist
                writer = pd.ExcelWriter(f'{path} {file} {extension["excel"]}', engine='xlsxwriter')
                dataFrame.to_excel(writer, sheet_name=str(state) + str(fld1))
                writer.save()
    else:
        print("Empty Dataframe.")


def export_results(dataFrame, fileName, csv=True, excel=False, pdf=False):
    if csv:
        dataFrame.to_csv(path_or_buf=os.path.join(fileName) + ' interface7 results' + TXT_EXT)

    if excel:
        try:  # excel workbook exists
            writer = pd.ExcelWriter(os.path.join(fileName) + XLSX_EXT, engine='openpyxl')
            writer.book = load_workbook(os.path.join(fileName) + XLSX_EXT)
            writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
            dataFrame.to_excel(writer, sheet_name=fileName)
            writer.save()
        except:  # excel workbook is created if it does not exist
            writer = pd.ExcelWriter(os.path.join(fileName) + XLSX_EXT, engine='xlsxwriter')
            dataFrame.to_excel(writer, sheet_name=fileName)
            writer.save()

    if pdf:
        pass


# SEARCH
# TODO The problem is that the 'number' is a string instead of a digit.
# TODO The reason this is a problem is because there is no method for converting the dataframe to digit.
def filter_data(dataFrame, column, entry, tol):
    operator, number = inequality(entry, tol)
    if operator != None:
        if is_number(number):
            if float(number) == int(float(number)):
                number = int(float(number))
            filtered = pd.DataFrame(data=dataFrame[operator(dataFrame.astype(float)[column], number)])
        else:
            filtered = pd.DataFrame(data=dataFrame[operator(dataFrame.astype(str)[column], number)])
        return filtered
    operators, numbers = interval(entry, tol)
    if operators != (None, None):
        leftOperator = operators[0]
        rightOperator = operators[1]
        leftNumber = numbers[0]
        rightNumber = numbers[1]
        filtered = pd.DataFrame(
            data=dataFrame[leftOperator(dataFrame[column], leftNumber) & rightOperator(dataFrame[column], rightNumber)]
        )
        return filtered
    return None


def filter_data(dataFrame, column, entry, tol):
    operator, number = inequality(entry, tol)
    if operator != None:
        if is_number(number):
            if float(number) == int(float(number)):
                number = str(int(float(number)))
        filtered = pd.DataFrame(data=dataFrame[operator(dataFrame[column], number)])
        return filtered
    operators, numbers = interval(entry, tol)
    if operators != (None, None):
        leftOperator = operators[0]
        rightOperator = operators[1]
        leftNumber = numbers[0]
        rightNumber = numbers[1]
        filtered = pd.DataFrame(
            data=dataFrame[leftOperator(dataFrame[column], leftNumber) & rightOperator(dataFrame[column], rightNumber)]
        )
        return filtered
    return None


def filter_data(dataFrame, entry, collection, column, i):
    if not ('<' in i or '>' in i or '-' in i):
        return collection
    filtered = collection
    for k in (inequality(entry), interval(entry)):
        for r in k:
            operator = r[0][0]
            if len(r[0]) == 1:  # Inequalities
                temp_df = pd.DataFrame(data=dataFrame[operator(dataFrame[column].astype(float), float(r[1][0]))])
                filtered.append(temp_df)
            elif len(r[0]) == 2:  # Ranges
                operator2 = r[0][1]
                temp_df = pd.DataFrame(
                    data=dataFrame[
                        operator(dataFrame[column].astype(float), float(pd.DataFrame(r[1]).columns[0])) & operator2(
                            dataFrame[column].astype(float), float(pd.DataFrame(r[1]).columns[1])
                        )]
                )
                filtered.append(temp_df)
    return filtered


def filter_data(dataFrame, column, entry, tol):
    operator, number = inequality(entry, tol)
    print(operator)
    if operator != None:
        if is_number(number):
            if float(number) == int(float(number)):
                number = str(int(float(number)))
        filtered = pd.DataFrame(data=dataFrame[operator(dataFrame[column], number)])
        print(filtered)
        return filtered
    operators, numbers = interval(entry, tol)
    if operators != (None, None):
        leftOperator = operators[0]
        rightOperator = operators[1]
        leftNumber = numbers[0]
        rightNumber = numbers[1]
        filtered = pd.DataFrame(
            data=dataFrame[leftOperator(dataFrame[column], leftNumber) & rightOperator(dataFrame[column], rightNumber)]
        )
        return filtered
    return None


def filter_dataFrame(dataFrame, column, entry, tol):
    filters = reduce_expression(entry, tol)
    ops = []
    nums = []
    for filter in filters:
        if filter[0] != None:
            ops.append(filter[0])
        if filter[1] != None:
            nums.append(filter[1])
    if len(ops) == 2:
        filtered = pd.DataFrame(data=dataFrame[ops[0](dataFrame[column], nums[0]) & ops[1](dataFrame[column], nums[1])])
    elif len(ops) == 1:
        filtered = pd.DataFrame(data=dataFrame[ops[0](dataFrame[column], nums[0])])
    else:
        filtered = None
    return filtered


def fit(models, X, y):
    """
    Fit each model in models to X, y
    models: dict, key is name of model, value is model
    X: matrix or vector
    y: matrix or vector
    return:
    """
    _models = models
    for model in models:
        _models[model].fit(X, y)
    return _models


def get_columns(entries):
    return [entry for entry in entries if entries[entry].text() is not '']


def get_columns(entries, entryChecks):
    return [entry for entry in entries if entryChecks[entry].isChecked()]


def get_dtypes(df):
    """
    df: pandas DataFrame
    return: list of str
    """
    return ['Numerical' if (str(df[col].dtype) != 'category' and str(df[col].dtype) != 'object') else 'Categorical' for
            col in df.columns]


def get_entries(entries):
    return {entry: entries[entry].text() for entry in entries}


def get_entries(entries, entryChecks):
    return {entry: entries[entry].text() for entry in entries if entryChecks[entry].isChecked()}


def get_dtypes(path, cols=None):
    """
    df: pandas DataFrame
    return: list of str
    """
    df = pd.read_csv(path, usecols=cols)
    result = ['Numerical' if (str(df[col].dtype) != 'category' and str(df[col].dtype) != 'object') else 'Categorical'
              for col in df.columns]
    return result if len(result) > 1 else result[0]


def get_header(path, separator=None):
    try:  # CSV
        with open(path) as f:
            head = next(f)
        if not separator:
            separator = "'" + str(get_separator(head)) + "'"
            header = pd.read_csv(StringIO(head), sep=separator.strip("'"), dtype='str')
    except:
        pass
    try:
        header = pd.read_excel(path, nrows=0)
    except:
        pass
    try:
        header = pd.read_hdf(path, nrows=0)
    except:
        pass
    return header, separator


def get_histogram_data(path, x, y, yData):
    histogramData = pd.read_csv(path)
    histogramData.loc[:, x[0]] = yData  # x[0] or x?
    histogramData.rename(columns={x[0]: y}, inplace=True)  # x[0] or x?
    return histogramData


def get_models(models, checks, settings):
    payload = {}
    for c in checks:
        payload[c] = models[c](**settings[c])
    return payload


def get_operator(operatorString):
    """
    Returns the Python object representation of the
    mathematical inequality operator equivalent of the string 'operator'
    operator: str - '>', '>=', '<', or '<='
    return: inequality operator
    """
    for operator in OPERATORS:
        if str(operator) == operatorString:
            return OPERATORS[operator]
    return None


def get_range(x, extension=0):
    """
    Finds the minimum and maximum values of an iterable.
    For test, if the iterable 'x' is the tuple (2.1, 5, 5, 3), then the tuple (2.1, 5) will be returned.
    If extension is set to 1, then the tuple (1.1, 6) will be returned instead.
    x: iterable of numbers
    extension: number to subtract from the minimum and add to the maximum
    return: tuple of floats
    """
    x_min, x_max = x.min() - extension, x.max() + extension
    return (x_min, x_max)


def get_separator(header):
    """
    Guess separator symbol in a string based on frequency
    header: string
    """
    max = 0
    s = ','
    for separator in SEPARATORS:
        counter = 0
        for character in header:
            if separator == character:
                counter += 1
        if counter > max:
            max = counter
            s = separator
    return s


def get_xy(path, x, y):
    data = pd.read_csv(path, usecols=x + [y]).sort_values(x)
    xData = data[x]
    yData = data.loc[:, y]
    return xData, yData


def get_xy(path, parameters):
    x, y = parameters['x'], parameters['y']
    data = pd.read_csv(path, usecols=x + [y]).sort_values(x)
    return dict(x=data[x], y=data.loc[:, y], )


def get_xy_choices(path):
    data = pd.read_csv(path, nrows=1)
    ch = [(i, i) for i in data.columns]
    return ch


def get_xy_cols(path):
    data = pd.read_csv(path, nrows=1)
    x = [(i, i) for i in data.columns]
    y = x[1:]
    return x, y


def get_uniques(dataFrame, dropna=True):
    """
    Returns all unique values in a pandas DataFrame
    dataFrame: pandas DataFrame
    return: numpy ndarray
    """
    labels = []
    for column in dataFrame.columns:
        labels.append(pd.DataFrame(dataFrame.loc[:, column].unique()))
    if dropna:
        labels = pd.Series(join_data_list(labels, axis=0)[0]).dropna().unique()
    else:
        labels = pd.Series(join_data_list(labels, axis=0)[0]).unique()
    try:
        if dropna:
            labels = labels[~np.isnan(labels.astype(float))]
    except Exception as e:
        print(e)
    return labels


def get_uniques(dataFrame, dropna=True):
    """
    Returns all unique values in a pandas DataFrame
    dataFrame: pandas DataFrame
    return: numpy ndarray
    """
    labels = []
    for column in dataFrame.columns:
        labels.append(pd.DataFrame(dataFrame.loc[:, column].unique()))
    labels = pd.Series(join_data_list(labels, axis=0)[0]).unique()
    try:
        if dropna:
            labels = labels[~np.isnan(labels.astype(float))]
    except Exception as e:
        print(e)
    return labels


def get_uniques(dataFrame, dropna=True):
    """
    All unique values in a pandas DataFrame
    dataFrame: pandas DataFrame
    return: pandas Series
    """
    labels = []
    for column in dataFrame.columns:
        labels.append(pd.DataFrame(dataFrame.loc[:, column].unique()))
    labels = pd.Series(join_data_list(labels, axis=0)[0]).unique()
    if dropna:
        labels = labels[~np.isnan(labels.astype(float))]
    return labels


def get_uniques(dataFrame, dropna=True):
    """
    Returns all unique values in a pandas DataFrame
    dataFrame: pandas DataFrame
    return: numpy ndarray
    """
    labels = []
    for column in dataFrame.columns:
        labels.append(pd.DataFrame(dataFrame.loc[:, column].unique()))
    if dropna:
        labels = pd.Series(join_data_list(labels, axis=0)[0]).dropna().unique()
    else:
        labels = pd.Series(join_data_list(labels, axis=0)[0]).unique()
    return labels


def get_uniques(dataFrame, dropna=True):
    """
    Returns all unique values in a pandas DataFrame. The unique values will be
    sorted numerically or alphabetically, and values equivalent to None will be dropped.
    For test, if the dataFrame is:
        0 1 1
    0   a b Nan
    1   b c a
    1   c a b
    then the numpy ndarray [a b c] will be returned.
    dataFrame: pandas DataFrame
    return: numpy ndarray
    """
    labels = []
    for column in dataFrame.columns:
        labels.append(pd.DataFrame(dataFrame.loc[:, column].unique()))
    if dropna:
        labels = pd.Series(join_data_list(labels, axis=0)[0]).sort_values().dropna().unique()
    else:
        labels = pd.Series(join_data_list(labels, axis=0)[0]).sort_values().unique()
    return labels


def get_uniques(_list):
    """
    Returns a list containing all unique items
    found in the given _list.
    :param _list: list
    :return: list
    """
    uniques = []
    for _ in _list:
        if _ not in uniques:
            uniques.append(_)
    return uniques


def get_uniques(dataFrame, dropna=True):
    """
    Returns all unique values in a pandas DataFrame
    dataFrame: pandas DataFrame
    return: numpy ndarray
    """
    labels = []
    for column in dataFrame.columns:
        labels.append(pd.DataFrame(dataFrame.loc[:, column].unique()))
    if dropna:
        labels = pd.Series(join_data_list(labels, axis=0)[0]).sort_values().dropna().unique()
    else:
        labels = pd.Series(join_data_list(labels, axis=0)[0]).sort_values().unique()
    return labels


def inequality(entry):
    """
    Process inequalities in an entry string. Supports >, <, >=, <=
    entry: string
    """
    for value in entry:
        for operator in OPERATORS:
            if operator in value:
                r = re.findall(r'(?<=[' + operator + '])(\d*[.]?\d*$)', value)
                # verify_inequality_entry(r, value, field)
                yield ((OPERATORS[operator],), r)
                break


def interval(entry):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    entry: string
    """
    for value in entry:
        for i in range(INDICATORS.__len__()):
            if INDICATORS[i] in value:
                r = pd.read_csv(StringIO(value), sep=INDICATORS[i], engine='python')
                if INDICATORS[i] == '--':
                    yield ((OPERATORS['>='], OPERATORS['<=']), r)
                elif INDICATORS[i] == '-':
                    yield ((OPERATORS['>'], OPERATORS['<']), r)
                break


def interval(entry, field):
    for value in entry:
        for i in range(INDICATORS.__len__()):
            if INDICATORS[i] in value:
                r = pd.read_csv(StringIO(value), sep=INDICATORS[i], engine='python')
                if INDICATORS[i] == '-':
                    yield ((OPERATORS['>'], OPERATORS['<']), r)
                elif INDICATORS[i] == '--':
                    yield ((OPERATORS['>='], OPERATORS['<=']), r)
                break


def inequality(entry, field):
    for value in entry:
        for operator in OPERATORS:
            if operator in value:
                r = re.findall(r'(?<=[' + operator + '])(\d*[.]?\d*$)', value)
                yield ((OPERATORS[operator],), r)
                break


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False
    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True
    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            op = leftover
        else:
            op = None
        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False
    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True
    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            op = leftover
        else:
            op = None
        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False

    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True

    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            op = leftover
        else:
            op = None

        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False

    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True

    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            op = leftover
        else:
            op = None

        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False

    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True

    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            op = leftover
        else:
            op = None

        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False

    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True

    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            op = leftover
        else:
            op = None

        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False

    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True

    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            op = leftover
        else:
            op = None

        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False

    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True

    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            op = leftover
        else:
            op = None

        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False

    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True

    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            if leftover not in INTERVALS.values():
                opRegex = difflib.get_close_matches(
                    leftover, list(INTERVALS.keys()) + list(INTERVALS.values()),
                    cutoff=tol
                )
                if opRegex:
                    op = opRegex[0]
                    op = INTERVALS[op]
                else:
                    op = None

            else:
                op = leftover
        else:
            op = None

        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False

    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True

    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            op = leftover
        else:
            op = None

        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False

    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True

    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            op = leftover
        else:
            op = None

        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False

    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True

    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            if leftover not in INTERVALS.values():
                opRegex = difflib.get_close_matches(
                    leftover, list(INTERVALS.keys()) + list(INTERVALS.values()),
                    cutoff=tol
                )
                if opRegex:
                    op = opRegex[0]
                    op = INTERVALS[op]
                else:
                    op = None

            else:
                op = leftover
        else:
            op = None

        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False

    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True

    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            op = leftover
        else:
            op = None

        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False
    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True
    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            op = leftover
        else:
            op = None
        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def inequality(str, tol):
    """
    Process inequalities in an entry string.
    Supports <, >, <=, >=, or english equivalent.
    If a number is detected, an inequality operator will be searched for.
    If an inequality operator is not found, an english equivalent will be searched for.
    str: string
    tol: float
    return: tuple
    """
    string = str.lower()
    numberRegex = re.findall(r'(\d*[.]?\d*$)', string.strip())
    if numberRegex == ['']:
        numberRegex = re.findall(r'[A-Za-z]+$', string.strip())
    if numberRegex:
        number = numberRegex[0]
    else:
        number = None
    if number:
        leftover = re.sub(number, '', string).strip().replace(" ", "")
        if leftover:
            if leftover not in OPERATORS.keys():
                opRegex = difflib.get_close_matches(leftover, OPERATORS_WORDS.keys(), cutoff=tol)
                if opRegex:
                    op = opRegex[0]
                    op = OPERATORS_WORDS[op]
                    op = OPERATORS[op]
                else:
                    op = None
            else:
                op = OPERATORS[leftover]
            return op, number
        return None, number
    return None, None


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False

    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True

    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            op = leftover
        else:
            op = None

        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False

    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True

    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            op = leftover
        else:
            op = None

        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def interval(entry, tol):
    """
    Process inequalities in an entry string. Supports - (exclusive), -- (inclusive)
    str: string
    tol: float
    return: tuple
    """
    string = entry.lower()
    number1 = ''
    number2 = ''
    second = False

    for c in entry:
        if c.isdigit() or c == '.':
            if second == False:
                number1 += c
            else:
                number2 += c
        elif not c.isdigit() and len(number1) > 0:
            second = True

    if number1 and number2:
        leftover = re.sub(number1, '', string).strip().replace(" ", "")
        leftover = re.sub(number2, '', leftover).strip().replace(" ", "")
        numbers = (number1, number2)
        if leftover:
            op = leftover
        else:
            op = None

        if op:
            if op == '--':
                ops = (OPERATORS['>='], OPERATORS['<='])
            else:
                ops = (OPERATORS['>'], OPERATORS['<'])
            return ops, numbers
    return (None, None), (None, None)


def inequality(str, tol):
    """
    Process inequalities in an entry string.
    Supports <, >, <=, >=, or english equivalent.
    If a number is detected, an inequality operator will be searched for.
    If an inequality operator is not found, an english equivalent will be searched for.
    str: string
    tol: float
    return: tuple
    """
    string = str.lower()
    numberRegex = re.findall(r'(\d*[.]?\d*$)', string.strip())
    if numberRegex == ['']:
        numberRegex = re.findall(r'[A-Za-z]+$', string.strip())
    if numberRegex:
        number = numberRegex[0]
    else:
        number = None
    if number:
        leftover = re.sub(number, '', string).strip().replace(" ", "")
        if leftover in OPERATORS:
            op = OPERATORS[leftover]
            return op, number
        return None, number
    return None, None


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


def is_number(string):
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


def is_valid_uuid(string):
    """
    Check whether the given string contains a valid
    uuid. Checks uuid versions 1-5.
    If the string does not contain a valid uuid, returns None.
    If a valid uuid is found, the uuid version is returned.
    :param string:
    :return:
    """
    version = 0
    for v in [1, 2, 3, 4, 5]:
        try:
            obj = uuid.UUID(string, version=v)
            if str(obj) == string:
                version = v
                break
        except ValueError:
            pass
    return version


def join_data(dataFrame):
    """
    dataFrame is a pandas dataFrame or a list
    """
    if dataFrame:
        return pd.DataFrame(pd.concat([d for d in dataFrame], join='inner', axis=0))
    else:
        return pd.DataFrame()


def join_data_list(dataFrameList, axis=0):
    """
    Concatenates the DataFrames in the dataFrameList
    keeping only rows or columns (depending on axis) that match.
    For test, if dataFrameList contains
        0 1                 0 1
    0   a b             0   2 1
    1   2 1     and     3   a a
    3   1 1             5   b b
    if axis is 0 the result will be:
       0 1
    0  a b
    1  2 1
    3  1 1
    0  2 1
    3  a a
    5  b b
    otherwise, if axis is 1, the result will be:
       0 1 0 1
    0  a b 2 1
    3  1 1 a a
    data: list of pandas DataFrame
    axis: int - if 0, DataFrames will be concatenated along row, if 1 they will be concatenated by column
    return pandas DataFrame
    """
    return pd.DataFrame(pd.concat(dataFrameList, join='inner', axis=axis))


def join_data_list(dataFrameList, axis=0):
    """
    Concatenates the dataFrames in the data list
    keeping only rows or columns (depending on axis) that match
    data: list of pandas dataFrames
    axis: integer
    """
    if dataFrameList:
        return pd.DataFrame(pd.concat(dataFrameList, join='inner', axis=axis))
    else:
        return pd.DataFrame()


def line_reader(path):
    lines = []
    with open(path) as f:
        for line in f:
            lines.append(line.strip())
    return lines


def line_reader(path):
    """
    Creates a list containing all of the lines in a text file specified by the path.
    Each line will be stripped of trailing spaces.
    Each value in the list corresponds to each line of the file. For test, if the file
    has the following contents:
    Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    Pellentesque elementum dapibus ligula id fringilla.
    the resulting list will be:
    ['Lorem ipsum dolor sit amet, consectetur adipiscing elit.',
    'Pellentesque elementum dapibus ligula id fringilla.']
    path: str - indicates the path of the file to read
    return: list
    """
    with open(path) as f:
        lines = list(map(lambda x: x.strip(), f))
    return lines


# TODO this doesn't do anything
def load_query(entries):
    try:
        path = open_files(fileExt=QRY_FILES)
        with open(path, "r") as loadFile:
            for i in range(entries.__len__()):
                entries[FEATURES[i]].delete(0, entries[FEATURES[i]].get().__len__())
            for i, line in enumerate(loadFile):
                if line.strip():
                    entries[FEATURES[i]].insert(i, line.strip())
    except:
        return


def make_symmetric_meshgrid(xRange, yRange, h=0.05):
    """
    Create a mesh of points to plot in.
    x: tuple - lower and upper bounds of the x-axis of the mesh
    y: tuple - lower and upper bounds of the y-axis of the mesh
    h: float - stepsize for the mesh
    returns: tuple of ndarrays
    """
    xx, yy = np.meshgrid(np.arange(xRange[0], xRange[1], h), np.arange(yRange[0], yRange[1], h))
    return xx, yy


def merge_data_list(dataFrameList):
    """
    Joins DataFrames contained inside a list. The DataFrames must contain only one
    column, but cannot be Series.
    dataFrameList: list of single column pandas DataFrames
    return: pandas DataFrame
    """
    # if len(dataFrameList) == 1:
    #    return dataFrameList[0]
    uniques_indices_list = [get_uniques(pd.DataFrame(dataFrame.index)) for dataFrame in dataFrameList]
    uniques_indices = get_uniques(pd.DataFrame(uniques_indices_list))
    d = {u: [] for u in uniques_indices}
    max, largest = 0, 0
    newDFList = []
    for i, dF in enumerate(dataFrameList):
        dFLen = len(dF.index)
        if dFLen > max:
            max = dFLen
            largest = i
    for dF in dataFrameList:
        temp = [dF.iloc[i, 0] for i, _ in enumerate(dF.index)]
        while len(temp) < max:
            temp.append(np.nan)
        newDFList.append(pd.Series(temp, index=dataFrameList[largest].index))
    for series in newDFList:
        for row, val in zip(series.index, series):
            d[row].append(val)
    """for dataFrame in dataFrameList:
        for i, row in enumerate(dataFrame.index):
            d[row].append(dataFrame.iloc[i, 0])
    for l in d:
        if len(d[l]) > max: max = len(d[l])"""
    for l in d:
        if len(d[l]) != max:
            while len(d[l]) != max:
                d[l].append(np.nan)
    joined = pd.DataFrame(d).T
    return joined


def merge_data_list(dataFrameList):
    """
    Joins DataFrames contained inside a list. The DataFrames must contain only one
    column, but cannot be Series.
    dataFrameList: list of single column pandas DataFrames
    return: pandas DataFrame
    """
    # if len(dataFrameList) == 1:
    #    return dataFrameList[0]

    uniques_indices_list = [get_uniques(pd.DataFrame(dataFrame.index)) for dataFrame in dataFrameList]
    uniques_indices = get_uniques(pd.DataFrame(uniques_indices_list))
    d = {u: [] for u in uniques_indices}
    for dataFrame in dataFrameList:
        for i, row in enumerate(dataFrame.index):
            d[row].append(dataFrame.iloc[i, 0])

    max = 0
    for l in d:
        if len(d[l]) > max:
            max = len(d[l])

    for l in d:
        if len(d[l]) != max:
            while len(d[l]) != max:
                d[l].append(np.nan)
    joined = pd.DataFrame(d).T
    return joined


def merge_data_list(dataFrameList):
    """
    Joins dataFrames contained inside a list
    :param dataFrameList: list of pandas DataFrames
    :return:  pandas DataFrame
    """
    joined = pd.DataFrame()
    for i, dataFrame in enumerate(dataFrameList):
        dataFrame.columns = [str(dataFrame.columns.values[0]) + " " + str(i)]
        joined = joined.join(dataFrame, how='outer')
    return joined


def merge_multidata_list(dataFrameList):
    """
    Joins DataFrames contained inside a list. The DataFrames must contain only one
    column, but cannot be Series.
    dataFrameList: list of single column pandas DataFrames
    return: pandas DataFrame
    """
    # if len(dataFrameList) == 1:
    #    return dataFrameList[0]
    uniques_indices_list = [get_uniques(pd.DataFrame(dataFrame.index)) for dataFrame in dataFrameList]
    uniques_indices = get_uniques(pd.DataFrame(uniques_indices_list))
    d = {u: [] for u in uniques_indices}
    for dataFrame in dataFrameList:
        for i, row in enumerate(dataFrame.index):
            temp = []
            for j, column in enumerate(dataFrame.columns):
                temp.append(dataFrame.iloc[i, j])
            d[row].append(temp)
    max = 0
    for l in d:
        if len(d[l]) > max:
            max = len(d[l])
    for l in d:
        if len(d[l]) != max:
            while len(d[l]) != max:
                d[l].append(np.nan)
    joined = pd.DataFrame(d).T
    return joined


def name_file(path):
    c = 0
    for i, s in enumerate((list(path))):
        if s == '\\' or s == '/':
            c = i + 1
    return '/' + path[int(c):]


def navigate_dict(requiredIDs, dictionary, level, max):
    """
    Some items in the _json variable of a scenario are lists
    and some are dictionaries. This function is used to
    navigate the dictionaries.
    :param requiredIDs: list of IDs that must be found in the json and extracted.
    :param dictionary: the dictionary to navigate
    :param level: the current level of depth within the _json that navigation is in
    :param max: the maximum depth of the _json
    :return:
    """
    max = update_max(level, max)
    for key in dictionary:
        if type(dictionary[key]) == dict:
            max = navigate_dict(requiredIDs, dictionary[key], level + 1, max)
        elif type(dictionary[key]) == list:
            max = navigate_list(requiredIDs, dictionary[key], level + 1, max)
        # if key in ['parentUID']:
        #     _ids.append('{},{},{}'.format(level, key, dictionary[key]))
        elif is_valid_uuid(str(dictionary[key])):
            requiredIDs.append('{},{},{}'.format(level, key, dictionary[key]))
    return max


def navigate_list(requiredIDs, _list, level, max):
    """
    Some items in the _json variable of a scenario are lists
    and some are dictionaries. This function is used to
    navigate the lists.
    :param requiredIDs: list of IDs that must be found in the json and extracted.
    :param _list: the list to navigate
    :param level: the current level of depth within the _json that navigation is in
    :param max: the maximum depth of the _json
    :return:
    """
    max = update_max(level, max)
    for item in _list:
        if type(item) == dict:
            max = navigate_dict(requiredIDs, item, level + 1, max)
        elif type(item) == list:
            max = navigate_list(requiredIDs, item, level + 1, max)
    return max


def nonize(string):
    """
    Returns None if a string is empty. Otherwise, returns the string.
    string: str
    return: None or str
    """
    if string == '':
        return None
    else:
        return string


def normalize_axis(dataFrame, axis=0):
    """
    Normalize dataFrame by axis.
    dataFrame: pandas DataFrame
    axis: int - If 0, the rows will be normalized. If 1, the columns will be normalized
    """
    try:
        if axis == 0:
            nrm = dataFrame.div(dataFrame.sum(axis=1), axis=0)
        if axis == 1:
            nrm = dataFrame.div(dataFrame.sum(axis=0), axis=1)
        nrm.fillna(0, inplace=True)
        return nrm
    except Exception as e:
        print(e)


def normalize_rows(dataFrame, rows=False, columns=False):
    """
    Normalize dataFrame by row
    """
    try:
        if rows:
            nrm = dataFrame.div(dataFrame.sum(axis=1), axis=0)
        if columns:
            nrm = dataFrame.div(dataFrame.sum(axis=0), axis=1)
        nrm.fillna(0, inplace=True)
        return nrm
    except Exception as e:
        print(e)


def normalize_rows(dataFrame):
    """
    Normalize dataFrame by row
    """
    try:
        nrm = dataFrame.div(dataFrame.sum(axis=1), axis=0)
        nrm.fillna(0, inplace=True)
        return nrm
    except:
        print("Division by zero")


def normalize_string(string):
    """
    Removes trailing spaces or new line indicators from a string.
    Additionally, the string will be converted entirely into lower case.
    string: str
    return: str
    """
    s = string.strip().lower()
    return s


def numerify_index(dataFrame, keys=None, series=False):
    """
    Reindexes dataFrame with numeric indices corresponding to original index.
    If keys are passed, the inverse operation will be performed (will replace numeric index with original index).
    Otherwise, the keys will be returned along with the reindexed dataFrame so that the inverse operation may
    be performed.
    dataFrame: pandas DataFrame
    keys: dict
    return: pandas DataFrame, dict (only if keys is not passed)
    """
    try:
        if keys:
            inverseKeys = {keys[m]: m for m in keys}
            results = pd.DataFrame(dataFrame.values, index=[inverseKeys[m] for m in dataFrame.index])
            if series == True:
                results = pd.Series(results.iloc[:, 0].as_matrix(), index=results.index)
            return results
        else:
            for i in dataFrame.index:
                i = float(i)
            return dataFrame, None
    except:
        if not keys:
            keyDict = {}
            k = {}
            uniques = get_uniques(pd.DataFrame(dataFrame.index))
            for i, u in enumerate(uniques):
                keyDict[i] = u
                k[u] = i
            results = pd.DataFrame(dataFrame.values, index=[k[m] for m in dataFrame.index])
            if series == True:
                results = pd.Series(results.iloc[:, 0].as_matrix(), index=results.index)
            return results, keyDict


def numerify_index(dataFrame, series=False):
    """
    Reindexes dataFrame with numeric indices corresponding to original index.
    Keys will be returned along with the reindexed dataFrame so that the inverse operation may
    be performed. For test, if dataFrame is:
        0 1 1
    a   a b c
    b   1 1 18
    b   1 f Nan
    the result will be:
        0 1 1
    0   a b c
    1   1 1 18
    1   1 f Nan
    and the keys will be returned as {0: a, 1: b}.
    dataFrame: pandas DataFrame
    return: pandas DataFrame, dict
    """
    indices = [str(i) for i in dataFrame.index]
    if all(is_number(u) for u in indices):
        return dataFrame, None
    else:
        keyDict = {}
        k = {}
        uniques = get_uniques(pd.DataFrame(dataFrame.index))
        for i, u in enumerate(uniques):
            keyDict[i] = u
            k[u] = i
        results = pd.DataFrame(dataFrame.values, index=[k[m] for m in dataFrame.index])
        if series == True:
            results = pd.Series(results.iloc[:, 0].as_matrix(), index=results.index)
        return results, keyDict


def numerify_values(dataFrame, series=False):
    uniques = get_uniques(pd.DataFrame(dataFrame))
    if all(is_number(u) for u in uniques):
        return dataFrame.astype(float), None
    else:
        keyDict = {i: u for i, u in enumerate(uniques)}
        result = dataFrame.replace(uniques, [k for k in keyDict])
        return result, keyDict


def numerify_values(dataFrame, keys=None, series=False):
    uniques = get_uniques(dataFrame)
    try:
        for u in uniques:
            u = float(u)
        return dataFrame, None
    except:
        keyDict = {i: u for i, u in enumerate(uniques)}
        result = dataFrame.replace(uniques, [k for k in keyDict])
        return result, keyDict


def numerify_values(dataFrame, keys=None, series=False):
    uniques = get_uniques(pd.DataFrame(dataFrame))
    try:
        for u in uniques:
            u = float(u)
        return dataFrame, None
    except:
        keyDict = {i: u for i, u in enumerate(uniques)}
        result = dataFrame.replace(uniques, [k for k in keyDict])
        return result, keyDict


def numerify_values(dataFrame):
    """
    Numerifies dataFrame that contains non-numerical values. For test, if dataFrame is:
        0 1 1
    1   a b c
    1   b b a
    the result will be:
        0 1 1
    1   0 1 1
    1   1 1 0
    and the keys will be {0: a, 1: b, 1: c}.
    dataFrame: pandas DataFrame
    return: pandas DataFrame, dict
    """
    uniques = get_uniques(pd.DataFrame(dataFrame))
    if all(is_number(u) for u in uniques):
        return dataFrame.astype(float), None
    else:
        keyDict = {i: u for i, u in enumerate(uniques)}
        result = dataFrame.replace(uniques, [k for k in keyDict])
        return result, keyDict


def populate_results(scenario, n):
    max = 0
    k = len(scenario.json) - 1
    results = []
    randomIndices = []
    while len(randomIndices) != n:
        randomNumber = random.randint(0, k)
        if randomNumber not in randomIndices:
            randomIndices.append(randomNumber)
    for i in range(n):
        r = randomIndices[i]
        results.append(scenario.json[r])
    requiredIDs = []
    max = navigate_list(requiredIDs, results, 1, max)
    return results, requiredIDs, max


def optimize_series(v):
    unique = v.unique()
    percentUnique = len(unique) / len(v)
    if any(not is_number(x) for x in v) and percentUnique < 0.5:
        dtype = 'category'
    else:
        dtype = 'float32'
    try:
        d = v.astype(dtype)
    except Exception as e:
        d = v
        print(e)
    # print("pre dtype: {} | post dtype: {}".format(v.dtype, d.dtype))
    return d, dtype


def predict(models, X_plot, index):
    yModels = []
    for model in models:
        yModels.append(models[model].predict(X_plot))
    prediction = pd.DataFrame(np.array(yModels).T, columns=models, index=index)
    return prediction


def process_classification_data(dir, index, column, classLabel):
    if classLabel == '':
        columns = column
    else:
        columns = (column, classLabel)
    data = read_data_list(dir, index, columns)
    labels = []

    if classLabel != '':
        for dF in data:
            ls = [v for v in dF.loc[:, classLabel].values]
            labels += ls
            dF.drop(classLabel, axis=1, inplace=True)
        labels = pd.DataFrame(labels).iloc[:, 0]
        labels, labelKeys = numerify_values(labels)
    else:
        labelKeys = None

    data = join_data_list(data)
    data, xKeys = numerify_index(data)
    data, yKeys = numerify_values(data)
    data = data.reset_index().astype(float)
    labels = labels.astype(float)
    classifiedMatrix = create_classified_matrix(data, labels, labelKeys)
    return data, labels, xKeys, yKeys, labelKeys, classifiedMatrix


def process_classification_data(dir, index, column, classLabel):
    if classLabel == '':
        columns = column
    else:
        columns = (column, classLabel)
    data = read_data_list(dir, index, columns)
    labels = []
    if classLabel != '':
        for dF in data:
            ls = [v for v in dF.loc[:, classLabel].values]
            labels += ls
            dF.drop(classLabel, axis=1, inplace=True)
        labels = pd.DataFrame(labels).iloc[:, 0]
        labels, labelKeys = numerify_values(labels)
    else:
        labelKeys = None
    data = join_data_list(data)
    data, xKeys = numerify_index(data)
    data, yKeys = numerify_values(data)
    data = data.reset_index().astype(float)
    labels = labels.astype(float)
    classifiedMatrix = create_classified_matrix(data, labels, labelKeys)
    return data, labels, xKeys, yKeys, labelKeys, classifiedMatrix


def process_classification_data(dir, index, column, classLabel):
    if classLabel == '':
        columns = column
    else:
        columns = (column, classLabel)
    data = read_data_list(dir, index, columns)
    labels = []
    if classLabel != '':
        for dF in data:
            ls = [v for v in dF.loc[:, classLabel].values]
            labels += ls
            dF.drop(classLabel, axis=1, inplace=True)
        labels = pd.DataFrame(labels).iloc[:, 0]
        labels, labelKeys = numerify_values(labels)
    else:
        labelKeys = None
    data = join_data_list(data)
    data, xKeys = numerify_index(data)
    data, yKeys = numerify_values(data)
    data = data.reset_index().astype(float)
    labels = labels.astype(float)
    return data, labels, xKeys, yKeys, labelKeys


def process_classification_data(dir, index, column, classLabel):
    if classLabel == '':
        columns = column
    else:
        columns = (column, classLabel)
    data = read_data_list(dir, index, columns)
    labels = []

    if classLabel != '':
        for dF in data:
            ls = [v for v in dF.loc[:, classLabel].values]
            labels += ls
            dF.drop(classLabel, axis=1, inplace=True)
        labels = pd.DataFrame(labels).iloc[:, 0]
        labels, labelKeys = numerify_values(labels)
    else:
        labelKeys = None

    data = join_data_list(data)
    data, xKeys = numerify_index(data)
    data, yKeys = numerify_values(data)
    data = data.reset_index().astype(float)
    labels = labels.astype(float)
    classifiedMatrix = create_classified_matrix(data, labels, labelKeys)
    return data, labels, xKeys, yKeys, labelKeys, classifiedMatrix


def process_classification_data(dir, index, column, classLabel):
    if classLabel == '':
        columns = column
    else:
        columns = (column, classLabel)
    data = read_data_list(dir, index, columns)
    labels = []

    if classLabel != '':
        for dF in data:
            ls = [v for v in dF.loc[:, classLabel].values]
            labels += ls
            dF.drop(classLabel, axis=1, inplace=True)
        labels = pd.DataFrame(labels).iloc[:, 0]
        labels, labelKeys = numerify_values(labels)
    else:
        labelKeys = None

    data = join_data_list(data)
    data, xKeys = numerify_index(data)
    data, yKeys = numerify_values(data)
    data = data.reset_index().astype(float)
    labels = labels.astype(float)
    return data, labels, xKeys, yKeys, labelKeys


def process_classification_data(dir, index, column, classLabel):
    if classLabel == '':
        columns = column
    else:
        columns = (column, classLabel)
    data = read_data_list(dir, index, columns)
    labels = []

    if classLabel != '':
        for dF in data:
            ls = [v for v in dF.loc[:, classLabel].values]
            labels += ls
            dF.drop(classLabel, axis=1, inplace=True)
        labels = pd.DataFrame(labels).iloc[:, 0]
        labels, labelKeys = numerify_values(labels)
    else:
        labelKeys = None

    data = join_data_list(data)
    data, xKeys = numerify_index(data)
    data, yKeys = numerify_values(data)
    data = data.reset_index().astype(float)
    labels = labels.astype(float)
    return data, labels, xKeys, yKeys, labelKeys


def process_classification_data(dir, index, column, classLabel):
    if classLabel == '':
        columns = column
    else:
        columns = (column, classLabel)
    data = read_data_list(dir, index, columns)
    labels = []

    if classLabel != '':
        for dF in data:
            ls = [v for v in dF.loc[:, classLabel].values]
            labels += ls
            dF.drop(classLabel, axis=1, inplace=True)
        labels = pd.DataFrame(labels).iloc[:, 0]
        labels, labelKeys = numerify_values(labels)
    else:
        labelKeys = None

    data = join_data_list(data)
    data, xKeys = numerify_index(data)
    data, yKeys = numerify_values(data)
    data = data.reset_index().astype(float)
    labels = labels.astype(float)
    classifiedMatrix = create_classified_matrix(data, labels, labelKeys)
    return data, labels, xKeys, yKeys, labelKeys, classifiedMatrix


def process_classification_data(dir, index, column, classLabel):
    if classLabel == '':
        columns = column
    else:
        columns = (column, classLabel)
    data = read_data_list(dir, index, columns)
    labels = []

    if classLabel != '':
        for dF in data:
            ls = [v for v in dF.loc[:, classLabel].values]
            labels += ls
            dF.drop(classLabel, axis=1, inplace=True)
        labels = pd.DataFrame(labels).iloc[:, 0]
        labels, labelKeys = numerify_values(labels)
    else:
        labelKeys = None

    data = join_data_list(data)
    data, xKeys = numerify_index(data)
    data, yKeys = numerify_values(data)
    data = data.reset_index().astype(float)
    labels = labels.astype(float)
    classifiedMatrix = create_classified_matrix(data, labels, labelKeys)
    return data, labels, xKeys, yKeys, labelKeys, classifiedMatrix


def process_classification_data(dir, index, column, classLabel):
    if classLabel == '':
        columns = column
    else:
        columns = (column, classLabel)
    data = read_data_list(dir, index, columns)
    labels = []
    if classLabel != '':
        for dF in data:
            ls = [v for v in dF.loc[:, classLabel].values]
            labels += ls
            dF.drop(classLabel, axis=1, inplace=True)
        labels = pd.DataFrame(labels).iloc[:, 0]
        labels, labelKeys = numerify_values(labels)
    else:
        labelKeys = None
    data = join_data_list(data)
    data, xKeys = numerify_index(data)
    data, yKeys = numerify_values(data)
    return data, labels, xKeys, yKeys, labelKeys


def process_classification_data(dir, index, column, classLabel):
    if classLabel == '':
        columns = column
    else:
        columns = (column, classLabel)
    data = read_data_list(dir, index, columns)
    labels = []

    if classLabel != '':
        for dF in data:
            ls = [v for v in dF.loc[:, classLabel].values]
            labels += ls
            dF.drop(classLabel, axis=1, inplace=True)
        labels = pd.DataFrame(labels).iloc[:, 0]
        labels, labelKeys = numerify_values(labels)
    else:
        labelKeys = None

    data = join_data_list(data)
    data, xKeys = numerify_index(data)
    data, yKeys = numerify_values(data)
    data = data.reset_index().astype(float)
    labels = labels.astype(float)
    classifiedMatrix = create_classified_matrix(data, labels, labelKeys)
    return data, labels, xKeys, yKeys, labelKeys, classifiedMatrix


def process_data(entries, settings, dtype, filepath, fileName, folderSuffix, files):
    data = []
    for state in files:
        for fld in folderSuffix:
            pth = DATABASE + fld + '\\' + state + fld + '.txt'
            data.append(read_data(pth, dtype))
            data[-1] = remove_nonmatches(settings, data[-1], entries)
            data[-1] = rename_duplicates(data[-1])
            export_results(data[-1], state, fld, filepath, fileName, settings)
    return data


# TODO numerify
def process_regression(dir, index, column, target, mode):
    dataList = read_data_list(dir, index, column, duplicateIndex=True)
    dataMatrix = merge_multidata_list(dataList)
    vectorized = vectorize_data(dataMatrix, mode)
    # data, xKeys = numerify_index(vectorized, series=True)
    # data, yKeys = numerify_values(data)
    xKeys, yKeys = None, None
    # data = data.dropna().astype(float)
    cols = dataList[0].columns
    x = pd.DataFrame(index=vectorized.index, columns=[col for col in cols if col != target])
    y = pd.Series(index=vectorized.index, name=target)
    for i, row in enumerate(vectorized.index):
        c = 0
        for j, col in enumerate(cols):
            if col == target:
                y.iloc[i] = vectorized.iloc[i][j]
            else:
                x.iloc[i, c] = vectorized.iloc[i][j]
                c += 1

    return x, y, xKeys, yKeys


# TODO numerify
def process_regression(dir, index, column, target, mode):
    dataList = read_data_list(dir, index, column, duplicateIndex=True)
    dataMatrix = merge_multidata_list(dataList)
    vectorized = vectorize_data(dataMatrix, mode)
    # data, xKeys = numerify_index(vectorized, series=True)
    # data, yKeys = numerify_values(data)
    xKeys, yKeys = None, None
    # data = data.dropna().astype(float)
    cols = dataList[0].columns
    x = pd.DataFrame(index=vectorized.index, columns=[col for col in cols if col != target])
    y = pd.Series(index=vectorized.index, name=target)
    for i, row in enumerate(vectorized.index):
        c = 0
        for j, col in enumerate(cols):
            if col == target:
                y.iloc[i] = vectorized.iloc[i][j]
            else:
                x.iloc[i, c] = vectorized.iloc[i][j]
                c += 1

    return x, y, xKeys, yKeys


# TODO numerify
def process_regression(dir, index, column, target, mode):
    dataList = read_data_list(dir, index, column, duplicateIndex=True)
    dataMatrix = merge_multidata_list(dataList)
    vectorized = vectorize_data(dataMatrix, mode)
    # data, xKeys = numerify_index(vectorized, series=True)
    # data, yKeys = numerify_values(data)
    xKeys, yKeys = None, None
    # data = data.dropna().astype(float)
    cols = dataList[0].columns
    x = pd.DataFrame(index=vectorized.index, columns=[col for col in cols if col != target])
    y = pd.Series(index=vectorized.index, name=target)
    for i, row in enumerate(vectorized.index):
        c = 0
        for j, col in enumerate(cols):
            if col == target:
                y.iloc[i] = vectorized.iloc[i][j]
            else:
                x.iloc[i, c] = vectorized.iloc[i][j]
                c += 1

    return x, y, xKeys, yKeys


# TODO numerify
def process_regression(dir, index, column, target, mode):
    dataList = read_data_list(dir, index, column, duplicateIndex=True)
    dataMatrix = merge_multidata_list(dataList)
    vectorized = vectorize_data(dataMatrix, mode)
    # data, xKeys = numerify_index(vectorized, series=True)
    # data, yKeys = numerify_values(data)
    xKeys, yKeys = None, None
    # data = data.dropna().astype(float)
    cols = dataList[0].columns
    x = pd.DataFrame(index=vectorized.index, columns=[col for col in cols if col != target])
    y = pd.Series(index=vectorized.index, name=target)
    for i, row in enumerate(vectorized.index):
        c = 0
        for j, col in enumerate(cols):
            if col == target:
                y.iloc[i] = vectorized.iloc[i][j]
            else:
                x.iloc[i, c] = vectorized.iloc[i][j]
                c += 1

    return x, y, xKeys, yKeys


# TODO numerify
def process_regression(dir, index, column, target, mode):
    dataList = read_data_list(dir, index, column, duplicateIndex=True)
    dataMatrix = merge_multidata_list(dataList)
    vectorized = vectorize_data(dataMatrix, mode)
    # data, xKeys = numerify_index(vectorized, series=True)
    # data, yKeys = numerify_values(data)
    xKeys, yKeys = None, None
    # data = data.dropna().astype(float)
    cols = dataList[0].columns
    x = pd.DataFrame(index=vectorized.index, columns=[col for col in cols if col != target])
    y = pd.Series(index=vectorized.index, name=target)
    for i, row in enumerate(vectorized.index):
        c = 0
        for j, col in enumerate(cols):
            if col == target:
                y.iloc[i] = vectorized.iloc[i][j]
            else:
                x.iloc[i, c] = vectorized.iloc[i][j]
                c += 1
    return x, y, xKeys, yKeys


def process_regression_data(dir, index, column, mode):
    dataList = read_data_list(dir, index, column)
    dataMatrix = merge_data_list(dataList)
    vectorized = vectorize_data(dataMatrix, mode)
    data, xKeys = numerify_index(vectorized, series=True)
    data, yKeys = numerify_values(data)
    data = data.dropna().astype(float)
    return data, xKeys, yKeys


# TODO numerify
def process_regression(dir, index, column, target, mode):
    dataList = read_data_list(dir, index, column, duplicateIndex=True)
    dataMatrix = merge_multidata_list(dataList)
    vectorized = vectorize_data(dataMatrix, mode)
    # data, xKeys = numerify_index(vectorized, series=True)
    # data, yKeys = numerify_values(data)
    xKeys, yKeys = None, None
    # data = data.dropna().astype(float)
    cols = dataList[0].columns
    x = pd.DataFrame(index=vectorized.index, columns=[col for col in cols if col != target])
    y = pd.Series(index=vectorized.index, name=target)
    for i, row in enumerate(vectorized.index):
        c = 0
        for j, col in enumerate(cols):
            if col == target:
                y.iloc[i] = vectorized.iloc[i][j]
            else:
                x.iloc[i, c] = vectorized.iloc[i][j]
                c += 1

    return x, y, xKeys, yKeys


# TODO numerify
def process_regression(dir, index, column, target, mode):
    dataList = read_data_list(dir, index, column, duplicateIndex=True)
    dataMatrix = merge_multidata_list(dataList)
    vectorized = vectorize_data(dataMatrix, mode)
    # data, xKeys = numerify_index(vectorized, series=True)
    # data, yKeys = numerify_values(data)
    xKeys, yKeys = None, None
    # data = data.dropna().astype(float)
    cols = dataList[0].columns
    x = pd.DataFrame(index=vectorized.index, columns=[col for col in cols if col != target])
    y = pd.Series(index=vectorized.index, name=target)
    for i, row in enumerate(vectorized.index):
        c = 0
        for j, col in enumerate(cols):
            if col == target:
                y.iloc[i] = vectorized.iloc[i][j]
            else:
                x.iloc[i, c] = vectorized.iloc[i][j]
                c += 1

    return x, y, xKeys, yKeys


# TODO numerify
def process_regression(dir, index, column, target, mode):
    dataList = read_data_list(dir, index, column, duplicateIndex=True)
    dataMatrix = merge_multidata_list(dataList)
    vectorized = vectorize_data(dataMatrix, mode)
    # data, xKeys = numerify_index(vectorized, series=True)
    # data, yKeys = numerify_values(data)
    xKeys, yKeys = None, None
    # data = data.dropna().astype(float)
    cols = dataList[0].columns
    x = pd.DataFrame(index=vectorized.index, columns=[col for col in cols if col != target])
    y = pd.Series(index=vectorized.index, name=target)
    for i, row in enumerate(vectorized.index):
        c = 0
        for j, col in enumerate(cols):
            if col == target:
                y.iloc[i] = vectorized.iloc[i][j]
            else:
                x.iloc[i, c] = vectorized.iloc[i][j]
                c += 1

    return x, y, xKeys, yKeys


# TODO numerify
def process_regression(dir, index, column, target, mode):
    dataList = read_data_list(dir, index, column, duplicateIndex=True)
    dataMatrix = merge_multidata_list(dataList)
    vectorized = vectorize_data(dataMatrix, mode)
    # data, xKeys = numerify_index(vectorized, series=True)
    # data, yKeys = numerify_values(data)
    xKeys, yKeys = None, None
    # data = data.dropna().astype(float)
    cols = dataList[0].columns
    x = pd.DataFrame(index=vectorized.index, columns=[col for col in cols if col != target])
    y = pd.Series(index=vectorized.index, name=target)
    for i, row in enumerate(vectorized.index):
        c = 0
        for j, col in enumerate(cols):
            if col == target:
                y.iloc[i] = vectorized.iloc[i][j]
            else:
                x.iloc[i, c] = vectorized.iloc[i][j]
                c += 1

    return x, y, xKeys, yKeys


# TODO numerify
def process_regression(dir, index, column, target, mode):
    dataList = read_data_list(dir, index, column, duplicateIndex=True)
    dataMatrix = merge_multidata_list(dataList)
    vectorized = vectorize_data(dataMatrix, mode)
    # data, xKeys = numerify_index(vectorized, series=True)
    # data, yKeys = numerify_values(data)
    xKeys, yKeys = None, None
    # data = data.dropna().astype(float)
    cols = dataList[0].columns
    x = pd.DataFrame(index=vectorized.index, columns=[col for col in cols if col != target])
    y = pd.Series(index=vectorized.index, name=target)
    for i, row in enumerate(vectorized.index):
        c = 0
        for j, col in enumerate(cols):
            if col == target:
                y.iloc[i] = vectorized.iloc[i][j]
            else:
                x.iloc[i, c] = vectorized.iloc[i][j]
                c += 1

    return x, y, xKeys, yKeys


def process_regression_data(dir, index, column, mode, matrix=False):
    data = read_data_list(dir, index, column)
    dataMatrix = merge_data_list(data)
    data = vectorize_data(dataMatrix, mode)
    data, xKeys = numerify_index(data, series=True)
    data, yKeys = numerify_values(data)
    data = data.dropna().astype(float)
    if matrix:
        dataMatrix, _ = numerify_index(dataMatrix)
        dataMatrix, _ = numerify_values(dataMatrix)
        return dataMatrix, data, xKeys, yKeys
    else:
        return data, xKeys, yKeys


def process_simulation_data(dataFrame, mode):
    data = vectorize_data(dataFrame, mode)
    data, xKeys = numerify_index(data, series=True)
    data, yKeys = numerify_values(data)
    data = data.dropna().astype(float)
    dataMatrix, xMKeys = numerify_index(dataFrame)
    dataMatrix, yMKeys = numerify_values(dataMatrix)
    return dataMatrix, data, xKeys, yKeys, xMKeys, yMKeys


def query(widget, file, entries, dir, columns, txt, excel):
    if os.stat(os.path.join(dir, file)).st_size != 0:
        try:
            results = search(entries=entries, dir=dir, file=file, dtype='str', columns=columns)
            if not results.empty:
                dh.export_data(results, file, csv=txt, excel=excel)
                widget.success_counter.append(1)
        except Exception as e:
            QMessageBox.critical(widget, "Error", "Couldn't read " + str(file) + ".\n" + str(e), QMessageBox.Ok)


def query(widget, file, entries, dir, columns, txt, excel):
    if os.path.splitext(file)[1] == TXT_EXT and os.stat(os.path.join(dir, file)).st_size != 0:
        try:
            results = search(entries=entries, dir=dir, file=file, dtype='str', columns=columns)
            if not results.empty:
                export_data(results, file, csv=txt, excel=excel)
                widget.success_counter.append(1)
        except Exception as e:
            QMessageBox.critical(widget, "Error", "Couldn't read " + str(file) + ".\n" + str(e), QMessageBox.Ok)


def query_master_loop(  # widget,
    dir, savePath, entries, columns, tol, formats
):
    resultsList = {}
    filesList = []
    cols = [entry for entry in entries if entries[entry] is not '']
    for subdir, dirs, files in os.walk(dir):
        for i, file in enumerate(files, start=1):
            try:
                data = DataReader.read_data(os.path.join(dir, file), columns=columns)
                results = DataCleaner.remove_nonmatches(data, entries, columns=cols, tol=tol)
                if not results.empty:
                    # widget.success_counter.append(1)
                    resultsList[file] = (results)
                    filesList.append(file)
                    """if renameDuplicates:
                        data = rename_duplicates(data)"""
            except Exception as e:
                pass
                """QMessageBox.critical(widget, "Error", "Couldn't read " + str(file) + ".\n" +
                                     str(e), QMessageBox.Ok)"""

    return resultsList, filesList


def query_master_loop(widget, dir, entries, columns, txt, excel):
    resultsList = []
    filesList = []
    for subdir, dirs, files in os.walk(dir):
        for i, file in enumerate(files, start=1):
            try:
                data = dh.read_data(os.path.join(dir, file), dtype='str', columns=columns)
                results = dh.remove_nonmatches(
                    data, entries, [entry for entry in entries if entries[
                        entry] is not '']
                )  # data = rename_duplicates(data)
            except Exception as e:
                QMessageBox.critical(widget, "Error", "Couldn't read " + str(file) + ".\n" + str(e), QMessageBox.Ok)
            if not results.empty:
                resultsList.append(results)
                dh.export_data(results, file, csv=txt, excel=excel)
                widget.success_counter.append(1)
                filesList.append(file)
    return resultsList, filesList


def query_master_loop(widget, dir, entries, columns, txt, excel):
    for subdir, dirs, files in os.walk(dir):
        for i, file in enumerate(files, start=1):
            query(widget, file, entries, subdir, columns, txt=txt, excel=excel)
            widget.p.setValue(int(100 * i / len(files)))


def query_master_loop(widget, dir, savePath, entries, columns, tol, formats):
    resultsList = []
    filesList = []
    for subdir, dirs, files in os.walk(dir):
        for i, file in enumerate(files, start=1):
            try:
                data = dh.read_data(os.path.join(dir, file), dtype='str', columns=columns)
                results = dh.remove_nonmatches(
                    data, entries, [entry for entry in entries if entries[entry] is not ''],
                    tol
                )
                if not results.empty:
                    resultsList.append(results)
                    # dh.export_data(results, os.path.join(savePath, file), formats)
                    widget.success_counter.append(1)
                    filesList.append(file)  # data = rename_duplicates(data)
            except Exception as e:
                QMessageBox.critical(widget, "Error", "Couldn't read " + str(file) + ".\n" + str(e), QMessageBox.Ok)
    return resultsList, filesList


def query_master_loop(widget, dir, savePath, entries, columns, tol, formats):
    resultsList = {}
    filesList = []
    cols = [entry for entry in entries if entries[entry] is not '']
    for subdir, dirs, files in os.walk(dir):
        for i, file in enumerate(files, start=1):
            try:
                data = DataProcessor.read_data(os.path.join(dir, file), columns=columns)
                results = DataProcessor.remove_nonmatches(data, entries, columns=cols, tol=tol)
                if not results.empty:
                    widget.success_counter.append(1)
                    resultsList[file] = (results)
                    filesList.append(file)
                    """if renameDuplicates:
                        data = rename_duplicates(data)"""
            except Exception as e:
                QMessageBox.critical(widget, "Error", "Couldn't read " + str(file) + ".\n" + str(e), QMessageBox.Ok)
    return resultsList, filesList


def query_master_loop(widget, dir, savePath, entries, columns, tol, txt, excel):
    resultsList = []
    filesList = []
    for subdir, dirs, files in os.walk(dir):
        for i, file in enumerate(files, start=1):
            try:
                data = dh.read_data(os.path.join(dir, file), dtype='str', columns=columns)
                results = dh.remove_nonmatches(
                    data, entries, [entry for entry in entries if entries[entry] is not ''],
                    tol
                )
                if not results.empty:
                    resultsList.append(results)
                    dh.export_data(results, os.path.join(savePath, file), csv=txt, excel=excel)
                    widget.success_counter.append(1)
                    filesList.append(file)  # data = rename_duplicates(data)
            except Exception as e:
                QMessageBox.critical(widget, "Error", "Couldn't read " + str(file) + ".\n" + str(e), QMessageBox.Ok)

    return resultsList, filesList


def reduce_expression(expressionList, tol):
    greater = {}
    less = {}

    def inserter(dict, key, op, op1, op2):
        if key not in dict.keys():
            dict[key] = op
        elif dict[key] == op1 and op == op2:
            dict[key] = op

    for expression in expressionList:
        op, number = inequality(expression, tol)
        if op:
            if is_number(number):
                number = float(number)
            if op == operator.gt or op == operator.ge:
                inserter(greater, number, op, operator.gt, operator.ge)
            else:
                inserter(less, number, op, operator.lt, operator.le)
        else:
            ops, numbers = interval(expression, tol)
            if ops != (None, None):
                for op, number in zip(ops, numbers):
                    nums = []
                    if is_number(numbers[0]):
                        nums.append(float(numbers[0]))
                    if is_number(numbers[1]):
                        nums.append(float(numbers[1]))
                    if op == operator.gt or op == operator.ge:
                        inserter(greater, number, op, operator.gt, operator.ge)
                    else:
                        inserter(less, number, op, operator.lt, operator.le)
    if len(greater) > 0:
        significantGreat = (greater[min(greater)], min(greater))
    else:
        significantGreat = (None, None)
    if len(less) > 0:
        significantLeast = (less[max(less)], max(less))
    else:
        significantLeast = (None, None)
    # TODO this if statement is FUCKED UP
    if significantGreat != (None, None) and significantLeast != (None, None):
        if INVERTED_OPERATORS[significantGreat[0]] == '>' or not is_number(significantGreat[1]):
            lowerBound = significantGreat[1]
        else:
            lowerBound = str(int(significantGreat[1]) - 1)
        if INVERTED_OPERATORS[significantLeast[0]] == '<' or not is_number(significantLeast[1]):
            upperBound = significantLeast[1]
        else:
            upperBound = str(int(significantLeast[1]) + 1)
        reduced = str(lowerBound) + '-' + str(upperBound)
        return read_entry(reduced)
    elif significantGreat != (None, None):
        reduced = INVERTED_OPERATORS[significantGreat[0]] + str(significantGreat[1])
        return read_entry(reduced)
    elif significantLeast != (None, None):
        reduced = INVERTED_OPERATORS[significantLeast[0]] + str(significantLeast[1])
        return read_entry(reduced)
    else:
        return expressionList


def reduce_expression(expressionList, tol):
    greater = {}
    less = {}

    def inserter(dict, key, op, op1, op2):
        if key not in dict.keys():
            dict[key] = op
        elif dict[key] == op1 and op == op2:
            dict[key] = op

    for expression in expressionList:
        op, number = inequality(expression, tol)
        if op:
            if is_number(number):
                number = float(number)
            if op == operator.gt or op == operator.ge:
                inserter(greater, number, op, operator.gt, operator.ge)
            else:
                inserter(less, number, op, operator.lt, operator.le)
        else:
            ops, numbers = interval(expression, tol)
            for op, number in zip(ops, numbers):
                nums = []
                if is_number(numbers[0]):
                    nums.append(float(numbers[0]))
                if is_number(numbers[1]):
                    nums.append(float(numbers[1]))
                if op == operator.gt or op == operator.ge:
                    inserter(greater, number, op, operator.gt, operator.ge)
                else:
                    inserter(less, number, op, operator.lt, operator.le)

    if len(greater) > 0:
        significantGreat = (greater[min(greater)], min(greater))
    else:
        significantGreat = (None, None)
    if len(less) > 0:
        significantLeast = (less[max(less)], max(less))
    else:
        significantLeast = (None, None)

    if significantGreat != (None, None) and significantLeast != (None, None):
        if INVERTED_OPERATORS[significantGreat[0]] == '>':
            lowerBound = significantGreat[1]
        else:
            lowerBound = str(int(significantGreat[1]) - 1)

        if INVERTED_OPERATORS[significantLeast[0]] == '<':
            upperBound = significantLeast[1]
        else:
            upperBound = str(int(significantLeast[1]) + 1)

        reduced = str(lowerBound) + '-' + str(upperBound)
        return read_entry(reduced)
    elif significantGreat != (None, None):
        reduced = INVERTED_OPERATORS[significantGreat[0]] + str(significantGreat[1])
        return read_entry(reduced)
    elif significantLeast != (None, None):
        reduced = INVERTED_OPERATORS[significantLeast[0]] + str(significantLeast[1])
        return read_entry(reduced)
    else:
        return expressionList


def read_data(path, dtype=None, columns=None, naValues=None, encoding=None, separator=None):
    """
    Handles the reading of several file types
    path: str, full path to file to read
    dtype: str,
    columns: list of str, columns to read
    naValues: list of str, values to consider null
    """
    kwargs = {'dtype': dtype, 'encoding': encoding, 'columns': columns, 'naValues': naValues}

    funcs = {'csv': _csv, 'excel': _excel, 'hdf5': _hdf5, 'wav': _wav}

    for func in funcs:
        for extension in Extensions.extension[func]:
            try:
                _path = GeneralUtil.add_extension(path, extension)
                result = funcs[func](_path, **kwargs)
                if type(result) == pd.DataFrame:
                    return result
            except Exception as e:
                pass


def read_data(path, dtype=None, columns=None, naValues=None, encoding=None, separator=None):
    """
    Handles the reading of several file types
    path: str, full path to file to read
    dtype: str,
    columns: list of str, columns to read
    naValues: list of str, values to consider null
    """
    kwargs = {'dtype': dtype, 'encoding': encoding, 'columns': columns, 'naValues': naValues}

    funcs = {'csv': _csv, 'excel': _excel, 'hdf5': _hdf5, 'wav': _wav}

    for func in funcs:
        for extension in Extensions.extension[func]:
            try:
                _path = GeneralUtil.add_extension(path, extension)
                result = funcs[func](_path, **kwargs)
                if type(result) == pd.DataFrame:
                    return result
            except Exception as e:
                print(e)


def read_data(path, dtype=None, columns=None, naValues=None):
    """
    Handles the reading of several file types through pandas
    path: str
    dtype: str
    columns: list of str
    naValues: list
    """
    # data = pd.read_hdf(path, os.path.splitext(os.path.basename(path))[0], na_values=NA_VALUES, dtype=dtype, encoding=ENCODING, sep=SEPARATOR)
    try:
        cols = pd.read_csv(path, nrows=0)
        usecols = [c for c in columns if c in cols.columns]
        return pd.read_csv(
            path, usecols=usecols, dtype=dtype, encoding=DataVars.ENCODING, na_values=DataVars.NA_VALUES,
            sep=DataVars.SEPARATOR
        )
    except Exception as e:
        pass
    try:
        return pd.read_csv(path, dtype=dtype)
    except:
        pass
    try:
        return pd.read_excel(path)
    except:
        pass
    try:
        return pd.read_hdf(path)
    except:
        pass
    try:
        return pd.DataFrame(wavfile.read(path)[1])
    except Exception as e:
        print(e)
        pass

        # data.set_index(HEADERS[0], inplace=True)


def read_data(path, dtype=None, columns=None, naValues=None):
    """
    Handles the reading of several file types through pandas
    path: str
    dtype: str
    columns: list of str
    naValues: list
    """
    # data = pd.read_hdf(path, os.path.splitext(os.path.basename(path))[0], na_values=NA_VALUES, dtype=dtype, encoding=ENCODING, sep=SEPARATOR)
    try:
        cols = pd.read_csv(path, nrows=0)
        usecols = [c for c in columns if c in cols.columns]
        return pd.read_csv(
            path, usecols=usecols, dtype=dtype, encoding=Vars.ENCODING, na_values=Vars.NA_VALUES,
            sep=Vars.SEPARATOR
        )
    except Exception as e:
        pass
    try:
        return pd.read_csv(path, dtype=dtype)
    except:
        pass
    try:
        return pd.read_excel(path)
    except:
        pass
    try:
        return pd.read_hdf(path)
    except:
        pass
    try:
        return pd.DataFrame(wavfile.read(path)[1])
    except Exception as e:
        print(e)
        pass

    # data.set_index(HEADERS[0], inplace=True)


def read_data(path, dtype=None, columns=None, naValues=None, encoding=None):
    """
    path: str, full path to file to read
    dtype: str,
    columns: list of str, columns to read
    naValues: list of str, values to consider null
    """
    kwargs = {'dtype': dtype, 'encoding': encoding, 'columns': columns, 'naValues': naValues}
    funcs = {'csv': _csv, 'xlsx': _excel, 'xlsm': _excel, 'hdf5': _hdf5, 'mat': _mat, 'wav': _wav}
    try:
        ext = path.split('.')[-1]
        result = funcs[ext](path, **kwargs)
        if type(result) == pd.DataFrame:
            d = optimize_dataFrame(result)
            return d, get_dtypes(d)
    except Exception as e:
        pass


def read_data(path, dtype='str'):
    """
    :param path:
    :param dtype:
    :return:
    """
    with open(path, newline='') as file: header = next(csv.reader(file))
    data = pd.read_csv(
        path, usecols=[item for item in header if item != FOLDER], na_values=NA_VALUES, dtype=dtype,
        encoding=ENCODING
    )
    data.set_index(ID, inplace=True)
    return data


def read_data_list(dir, index=None, column=None, duplicateIndex=False):
    """
    Read each file in a directory into a pandas DataFrame and collect them into a list.
    dir: str - indicates the path to the directory to read from
    index: str - indicates which column to use as the index for the pandas DataFrames
    column: str or list of str - indicates which column or columns to read into the DataFrame, others will be ignored
    return: list of pandas DataFrames
    """
    if type(column) == tuple:
        column = list(column)
    if index and column and type(column) == str:
        cols = [index, column]
    elif index and column and type(column) == list:
        cols = [index] + column
    elif type(column) == list:
        cols = column
    else:
        cols = [column]
    if cols == [None]:
        cols = None
    data = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if cols == None:
                data.append(read_data(os.path.join(dir, file), dtype='str', naValues=Vars.NA_VALUES))
            else:
                data.append(
                    clean_data(read_data(os.path.join(dir, file), dtype='str', columns=cols, naValues=Vars.NA_VALUES))
                )
    if index:
        for dF in data:
            dF.set_index(index, inplace=True)
            if duplicateIndex:
                dF.insert(0, index, dF.index)
    return data


def read_data_list(dir, index=None, column=None, duplicateIndex=False):
    """
    Read each file in a directory into a pandas DataFrame and collect them into a list.
    dir: str - indicates the path to the directory to read from
    index: str - indicates which column to use as the index for the pandas DataFrames
    column: str or list of str - indicates which column or columns to read into the DataFrame, others will be ignored
    return: list of pandas DataFrames
    """
    if type(column) == tuple:
        column = list(column)
    if index and column and type(column) == str:
        cols = [index, column]
    elif index and column and type(column) == list:
        cols = [index] + column
    elif type(column) == list:
        cols = column
    else:
        cols = [column]
    if cols == [None]:
        cols = None
    data = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if cols == None:
                data.append(read_data(os.path.join(dir, file), dtype='str', naValues=DataVars.NA_VALUES))
            else:
                data.append(
                    clean_data(
                        read_data(os.path.join(dir, file), dtype='str', columns=cols, naValues=DataVars.NA_VALUES)
                    )
                )
    if index:
        for dF in data:
            dF.set_index(index, inplace=True)
            if duplicateIndex:
                dF.insert(0, index, dF.index)
    return data


def read_data_list(dir, index=None, column=None, duplicateIndex=False, naValues=None):
    """
    Read each file in a directory into a pandas DataFrame and collect them into a list.
    dir: str - indicates the path to the directory to read from
    index: str - indicates which column to use as the index for the pandas DataFrames
    column: str or list of str - indicates which column or columns to read into the DataFrame, others will be ignored
    return: list of pandas DataFrames
    """
    if type(column) == tuple:
        column = list(column)
    if index and column and type(column) == str:
        cols = [index, column]
    elif index and column and type(column) == list:
        cols = [index] + column
    elif type(column) == list:
        cols = column
    else:
        cols = [column]
    if cols == [None]:
        cols = None
    data = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if cols == None:
                data.append(read_data(os.path.join(dir, file), dtype='str', naValues=naValues))
            else:
                data.append(
                    clean_data(read_data(os.path.join(dir, file), dtype='str', columns=cols, naValues=naValues))
                )
    if index:
        for dF in data:
            dF.set_index(index, inplace=True)
            if duplicateIndex:
                dF.insert(0, index, dF.index)
    return data


def read_data(path, dtype=None, columns=None, naValues=None, encoding=None):
    """
    path: str, full path to file to read
    dtype: str,
    columns: list of str, columns to read
    naValues: list of str, values to consider null
    """
    kwargs = {'dtype': dtype, 'encoding': encoding, 'columns': columns, 'naValues': naValues}
    funcs = {'csv': _csv, 'excel': _excel, 'hdf5': _hdf5, 'mat': _mat, 'wav': _wav}
    for func in funcs:
        for extension in util.extension[func]:
            try:
                result = funcs[func](path, **kwargs)
                if type(result) == pd.DataFrame:
                    d = optimize_dataFrame(result)
                    return d, get_dtypes(d)
            except Exception as e:
                pass


def read_data(path, dtype='str', columns=None, naValues=None):
    """
    Handles the reading of several file types through pandas
    path: str
    dtype: str
    columns: list of str
    naValues: list
    """
    # data = pd.read_hdf(path, os.path.splitext(os.path.basename(path))[0], na_values=NA_VALUES, dtype=dtype, encoding=ENCODING, sep=SEPARATOR)
    try:
        cols = pd.read_csv(path, nrows=0)
        usecols = [c for c in columns if c in cols.columns]
        data = pd.read_csv(
            path, usecols=usecols, dtype=dtype, encoding=Vars.ENCODING, na_values=Vars.NA_VALUES,
            sep=Vars.SEPARATOR
        )
    except Exception as e:
        print(e)
        data = pd.read_csv(path, dtype=dtype)
    try:
        data = pd.read_excel(path)
    except:
        pass
    try:
        data = pd.read_hdf(path)
    except:
        pass

    # data.set_index(HEADERS[0], inplace=True)
    return data


def read_data(path, dtype='str', columns=None, naValues=None):
    """
    Handles the reading of several file types through pandas
    """
    # data = pd.read_hdf(path, os.path.splitext(os.path.basename(path))[0], na_values=NA_VALUES, dtype=dtype, encoding=ENCODING, sep=SEPARATOR)
    try:
        data = pd.read_csv(
            path, usecols=columns, dtype=dtype, encoding=vars.ENCODING, na_values=NA_VALUES,
            sep=vars.SEPARATOR
        )
    except:
        pass
    try:
        data = pd.read_excel(path)
    except:
        pass
    try:
        data = pd.read_hdf(path)
    except:
        pass

    # data.set_index(HEADERS[0], inplace=True)
    return data


def read_data(path, dtype='str', columns=None, naValues=None):
    """
    Handles the reading of several file types through pandas
    path: str
    dtype: str
    columns: list of str
    naValues: list
    """
    # data = pd.read_hdf(path, os.path.splitext(os.path.basename(path))[0], na_values=NA_VALUES, dtype=dtype, encoding=ENCODING, sep=SEPARATOR)
    try:
        cols = pd.read_csv(path, nrows=0)
        usecols = [c for c in columns if c in cols.columns]
        data = pd.read_csv(
            path, usecols=usecols, dtype=dtype, encoding=vars.ENCODING, na_values=NA_VALUES,
            sep=vars.SEPARATOR
        )
    except Exception as e:
        print(e)
        pass
    try:
        data = pd.read_excel(path)
    except:
        pass
    try:
        data = pd.read_hdf(path)
    except:
        pass
    # data.set_index(HEADERS[0], inplace=True)
    return data


def read_data(path, dtype='str', columns=None):
    """
    Handles the reading of several file types through pandas
    """
    # data = pd.read_hdf(path, os.path.splitext(os.path.basename(path))[0], na_values=NA_VALUES, dtype=dtype, encoding=ENCODING, sep=SEPARATOR)
    try:
        data = pd.read_csv(
            path, usecols=columns, dtype=dtype, encoding=ENCODING, na_values=NA_VALUES,
            sep=SEPARATOR
        )
    except:
        return pd.DataFrame()
    # data.set_index(HEADERS[0], inplace=True)
    return data


def read_data(path, dtype='str'):
    """
    :param path:
    :param dtype:
    :return:
    """
    with open(path, newline='') as file: header = next(csv.reader(file))
    data = pd.read_csv(
        path, usecols=[item for item in header if item != FOLDER], na_values=NA_VALUES, dtype=dtype,
        encoding='latin1'
    )
    data.set_index(ID, inplace=True)
    return data


def read_data(path, dtype='str', columns=None):
    """
    """

    print(path)
    # data = pd.read_hdf(path, os.path.splitext(os.path.basename(path))[0], na_values=NA_VALUES, dtype=dtype, encoding=ENCODING, sep=SEPARATOR)
    data = pd.read_csv(path, dtype=dtype, encoding=ENCODING, na_values=NA_VALUES, sep=SEPARATOR)
    # data.set_index(HEADERS[0], inplace=True)
    return data


def read_data(path, dtype='str', columns=None, naValues=None):
    """
    Handles the reading of several file types through pandas
    path: str
    dtype: str
    columns: list of str
    naValues: list
    """
    # data = pd.read_hdf(path, os.path.splitext(os.path.basename(path))[0], na_values=NA_VALUES, dtype=dtype, encoding=ENCODING, sep=SEPARATOR)
    try:
        data = pd.read_csv(
            path, usecols=columns, dtype=dtype, encoding=vars.ENCODING, na_values=NA_VALUES,
            sep=vars.SEPARATOR
        )
    except:
        pass
    try:
        data = pd.read_excel(path)
    except:
        pass
    try:
        data = pd.read_hdf(path)
    except:
        pass

    # data.set_index(HEADERS[0], inplace=True)
    return data


def read_data_list(dir, index=None, column=None):
    """
    Read each file in a directory into a pandas DataFrame and collect them into a list.
    dir: str - indicates the path to the directory to read from
    index: str - indicates which column to use as the index for the pandas DataFrames
    column: str or list of str - indicates which column or columns to read into the DataFrame, others will be ignored
    return: list of pandas DataFrames
    """
    if type(column) == tuple:
        column = list(column)
    if index and column and type(column) == str:
        cols = [index, column]
    elif index and column and type(column) == list:
        cols = [index] + column
    elif type(column) == list:
        cols = column
    else:
        cols = [column]
    if cols == [None]:
        cols = None
    data = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if cols == None:
                data.append(read_data(os.path.join(dir, file), dtype='str', naValues=NA_VALUES))
            else:
                data.append(
                    clean_data(read_data(os.path.join(dir, file), dtype='str', columns=cols, naValues=NA_VALUES))
                )
    if index:
        for dF in data:
            dF.set_index(index, inplace=True)
    return data


def read_data_list(dir, index=None, column=None):
    """
    Read each file in a directory into a pandas DataFrame and collect them into a list.
    dir: string
    index: string
    column: string
    return: list of pandas DataFrames
    """
    if type(column) == tuple:
        column = list(column)
    if index and column and type(column) == str:
        cols = [index, column]
    elif index and column and type(column) == list:
        cols = [index] + column
    elif type(column) == list:
        cols = column
    else:
        cols = [column]
    if cols == [None]:
        cols = None
    data = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if cols == None:
                data.append(pd.read_csv(os.path.join(dir, file), dtype=str, na_values=NA_VALUES))
            else:
                data.append(
                    clean_data(read_data(os.path.join(dir, file), dtype='str', columns=cols, naValues=NA_VALUES))
                )
    if index:
        for dF in data:
            dF.set_index(index, inplace=True)
    return data


def read_data_list(dir, index=None, column=None, duplicateIndex=False):
    """
    Read each file in a directory into a pandas DataFrame and collect them into a list.
    dir: str - indicates the path to the directory to read from
    index: str - indicates which column to use as the index for the pandas DataFrames
    column: str or list of str - indicates which column or columns to read into the DataFrame, others will be ignored
    return: list of pandas DataFrames
    """
    if type(column) == tuple:
        column = list(column)
    if index and column and type(column) == str:
        cols = [index, column]
    elif index and column and type(column) == list:
        cols = [index] + column
    elif type(column) == list:
        cols = column
    else:
        cols = [column]
    if cols == [None]:
        cols = None
    data = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if cols == None:
                data.append(read_data(os.path.join(dir, file), dtype='str', naValues=NA_VALUES))
            else:
                data.append(
                    clean_data(read_data(os.path.join(dir, file), dtype='str', columns=cols, naValues=NA_VALUES))
                )
    if index:
        for dF in data:
            dF.set_index(index, inplace=True)
            if duplicateIndex:
                dF.insert(0, index, dF.index)
    return data


def read_data_list(dir, index=None, column=None):
    """
    Read each file in a directory into a pandas DataFrame and collect them into a list.
    dir: string
    index: string
    column: string
    return: list of pandas DataFrames
    """
    try:
        if index and column:
            cols = [index, column]
        else:
            cols = [column]
        data = []
        for subdir, dirs, files in os.walk(dir):
            for file in files:
                data.append(
                    clean_data(pd.read_csv(os.path.join(dir, file), dtype=str, usecols=cols, na_values=NA_VALUES))
                )
        if index:
            for dF in data:
                dF.set_index(index, inplace=True)
        return data
    except Exception as e:
        print(e)


def read_entry(entry):
    """
    Converts a string of comma separated values (test: '1, 1, 2') into a Python list (test: ['1', '1', '2'])
    entry: str
    return: list
    """
    entry = list(pd.read_csv(StringIO(entry), dtype='str', keep_default_na=False).columns)
    return entry


def read_entry(entry):
    """
    returns a pandas dataFrame from a comma separated string
    entry: string
    """
    return pd.read_csv(StringIO(entry), dtype='str', keep_default_na=False)


def read_folder(dir, ext):
    """
    Collect all file paths with given extension (ext) inside a folder directory (dir)
    """
    paths = []
    for file in os.listdir(dir):
        if file[-4:].lower() == ext.lower():
            paths.append(file)
    return paths


def read_folder(dir, ext):
    """
    Collect all file paths with given extension (ext) inside a folder directory (dir) into a list
    dir: str - indicates the path to the directory in which to search
    ext: str - indicates the extension to search for, others are ignored
    return: list of str
    """
    paths = []
    for file in os.listdir(dir):
        if file[-len(ext):].lower() == ext.lower():
            paths.append(file)
    return paths


def reduce_expression(expressionList, tol):
    greater = {}
    less = {}

    def inserter(dict, key, op, op1, op2):
        if key not in dict.keys():
            dict[key] = op
        elif dict[key] == op1 and op == op2:
            dict[key] = op

    for expression in expressionList:
        op, number = inequality(expression, tol)
        if op:
            if GeneralUtil.is_number(number):
                number = float(number)
            if op == operator.gt or op == operator.ge:
                inserter(greater, number, op, operator.gt, operator.ge)
            else:
                inserter(less, number, op, operator.lt, operator.le)
        else:
            ops, numbers = interval(expression, tol)
            if ops != (None, None):
                for op, number in zip(ops, numbers):
                    nums = []
                    if GeneralUtil.is_number(numbers[0]):
                        nums.append(float(numbers[0]))
                    if GeneralUtil.is_number(numbers[1]):
                        nums.append(float(numbers[1]))
                    if op == operator.gt or op == operator.ge:
                        inserter(greater, number, op, operator.gt, operator.ge)
                    else:
                        inserter(less, number, op, operator.lt, operator.le)

    if len(greater) > 0:
        significantGreat = (greater[min(greater)], min(greater))
    else:
        significantGreat = (None, None)
    if len(less) > 0:
        significantLeast = (less[max(less)], max(less))
    else:
        significantLeast = (None, None)
    # TODO this if statement is FUCKED UP
    if significantGreat != (None, None) and significantLeast != (None, None):
        if INVERTED_OPERATORS[significantGreat[0]] == '>' or not GeneralUtil.is_number(significantGreat[1]):
            lowerBound = significantGreat[1]
        else:
            lowerBound = str(int(significantGreat[1]) - 1)

        if INVERTED_OPERATORS[significantLeast[0]] == '<' or not GeneralUtil.is_number(significantLeast[1]):
            upperBound = significantLeast[1]
        else:
            upperBound = str(int(significantLeast[1]) + 1)

        reduced = str(lowerBound) + '-' + str(upperBound)
        return read_entry(reduced)
    elif significantGreat != (None, None):
        reduced = INVERTED_OPERATORS[significantGreat[0]] + str(significantGreat[1])
        return read_entry(reduced)
    elif significantLeast != (None, None):
        reduced = INVERTED_OPERATORS[significantLeast[0]] + str(significantLeast[1])
        return read_entry(reduced)
    else:
        return expressionList


def remove_data(dataFrame, column, entry):
    """
    Remove rows in dataFrame which do not contain the values in a list (entry) in the desired column
    dataFrame: pandas DataFrame
    column: str - indicates the column for which to interface7 for the entry
    entry: list of str - indicates which values to interface7 for in the column
    return: pandas DataFrame
    """
    dF = pd.DataFrame(dataFrame)
    matches = pd.DataFrame(dF[dF[column].str.lower().isin(entry)])
    return matches


def remove_data(dataFrame, column, entry):
    """
    remove rows in dataFrame which do not contain the entry value in the desired column
    dataFrame: pandas data frame
    column: string
    entry: string
    """
    dF = dataFrame
    ent = [normalize_string(entry)]
    return pd.DataFrame(dF[dF[column].str.lower().isin(ent)])


def remove_data(dataFrame, column, entry):
    """
    remove rows in dataFrame which do not contain the entry value in the desired column
    """
    dF = dataFrame
    if type(entry) == str:
        entry = [entry]  # homogenizing the type of i
    return pd.DataFrame(dF[dF[column].str.lower().isin(entry)])


def remove_nonmatches(dataFrame, entries, columns):
    """
    entry is a string of comma separated values. Each value in this list is checked independently of the others.
    All matches across each value in entry is accepted. This process is repeated for each column.
    dataFrame: pandas data frame
    entries: string
    columns: list of strings
    """
    dF = dataFrame
    for column in columns:
        dataList = []
        entry = read_entry(entries[column])
        for i in entry:
            i = normalize_string(i)
            dataList = filter_data(dF, entry, dataList, column, i)
            if (str(i) in dF[column].str.lower().values):
                dataList.append(remove_data(dF, column, i))
        dF = join_data_list(dataList)
    return dF


def remove_nonmatches(settings, dataFrame, entries):
    dF = dataFrame
    if not settings[2].get():  # Union
        pass
    else:  # Intersection
        for column in [item for item in entries if item != FOLDER and item != FILE and entries[item].get() != '']:
            collection = []
            entry = pd.read_csv(StringIO(entries[column].get()), sep=',')
            x = inequality(entry, column)
            y = interval(entry, column)
            for i in entry:
                for k in (x, y):
                    for r in k:
                        operator = r[0][0]
                        if len(r[0]) == 1:  # Inequalities
                            temp_df = pd.DataFrame(data=dF[operator(dF[column].astype(float), float(r[1][0]))])
                            collection.append(temp_df)
                        elif len(r[0]) == 2:  # Ranges
                            operator2 = r[0][1]
                            temp_df = pd.DataFrame(
                                data=dF[
                                    operator(
                                        dF[column].astype(float), float(pd.DataFrame(r[1]).columns[0])
                                    ) & operator2(
                                        dF[column].astype(float), float(pd.DataFrame(r[1]).columns[1])
                                    )]
                            )
                            collection.append(temp_df)
                if type(i) == str:  # homogenizing the type of i
                    i = [i]
                temp_df = (pd.DataFrame(dF[dF[column].isin(i)]))
                collection.append(temp_df)
                dF = pd.DataFrame(pd.concat([c for c in collection], join='inner', axis=0))
        return dF


def remove_nonmatches(dataFrame, entries, columns):
    dF = dataFrame
    for column in columns:
        collection = []
        entry = read_entry(entries[column])
        for i in entry:
            i = i.strip().lower()
            collection = filter_data(dF, entry, collection, column, i)
            if (str(i) in dF[column].str.lower().values):
                collection.append(remove_data(dF, column, i))
        dF = join_data(collection)
    return dF


def remove_nonmatches(dataFrame, entries, columns, tol):
    """
    Entry is a string of comma separated values (test: '1,1,2').
    Each value in this list is checked independently of the others.
    All matches across each value in entry is accepted.
    This process is repeated for each column.
    dataFrame: pandas DataFrame
    entries: str - indicates the values for which to interface7
    columns: list of str - indicates the columns in which to perform the interface7, others are ignored
    """
    dF = pd.DataFrame(dataFrame)
    for column in columns:
        dataList = []
        entry = read_entry(entries[column])
        entry = reduce_expression(entry, tol)

        for e in entry:
            e = normalize_string(e)
            filtered = filter_data(dF, column, e, tol)
            if type(filtered) != pd.DataFrame:
                matches = difflib.get_close_matches(e, dF[column].str.lower().values, cutoff=tol)
                cleared = remove_data(dF, column, matches)
                dataList.append(cleared)
            else:
                dataList.append(filtered)
        dF = join_data_list(dataList)
    return dF


def remove_nonmatches(dataFrame, entries, columns, tol):
    """
    Entry is a string of comma separated values (test: '1,1,2').
    Each value in this list is checked independently of the others.
    All matches across each value in entry is accepted. This process is repeated for each column.
    dataFrame: pandas DataFrame
    entries: str - indicates the values for which to interface7
    columns: list of str - indicates the columns in which to perform the interface7, others are ignored
    """
    dF = pd.DataFrame(dataFrame)
    for column in columns:
        dataList = []
        entry = read_entry(entries[column])
        entry = reduce_expression(entry, tol)
        for e in entry:
            e = normalize_string(e)
            filtered = filter_data(dF, column, e, tol)
            if type(filtered) != pd.DataFrame:
                matches = difflib.get_close_matches(str(e), dF[column].dropna().str.lower().values, cutoff=tol)
                cleared = remove_data(dF, column, matches)
                dataList.append(cleared)
            else:
                dataList.append(filtered)
        dF = join_data_list(dataList)
    return dF


def remove_nonmatches(dataFrame, entries, columns=None):
    dF = dataFrame
    if not columns:
        columns = [entry for entry in entries if entry != FOLDER and entries[entry].text() is not '']
    if False == True:  # if not settings[1].text():  # Union
        pass
    else:  # Intersection
        for column in columns:
            collection = []
            entry = pd.read_csv(StringIO(entries[column].text()), sep=',')
            for i in entry:
                if '<' in i or '>' in i or '-' in i:
                    collection = filter(dF, entry, collection, column)
                else:
                    if str(i) not in dF[column]:
                        print("not here")
                        print("i: " + str(i))
                        print("dF: " + dF[column])
                        return
                    else:
                        print("found")
                        if type(i) == str:
                            i = [i]  # homogenizing the type of i
                        collection.append(pd.DataFrame(dF[dF[column].isin(i)]))
            dF = pd.DataFrame(pd.concat([c for c in collection], join='inner', axis=0))
        return dF


def remove_nonmatches(dataFrame, entries, columns=None):
    dF = dataFrame
    if not columns:
        columns = [entry for entry in entries if
                   entry not in (FILE, FOLDER) and entries[entry].text() is not '']
    if False == True:  # if not settings[1].text():  # Union
        pass
    else:  # Intersection
        for column in columns:
            collection = []
            entry = pd.read_csv(StringIO(entries[column].text()), sep=',')
            for i in entry:
                if '<' in i or '>' in i or '-' in i:
                    collection = filter(dF, entry, collection, column)
                else:
                    if not (str(i) in dF[column].values):
                        print("not here")
                    else:
                        print("found")
                        if type(i) == str:
                            i = [i]  # homogenizing the type of i
                        collection.append(pd.DataFrame(dF[dF[column].isin(i)]))
            dF = pd.DataFrame(pd.concat([c for c in collection], join='inner', axis=0))
        return dF


def remove_nonmatches(dataFrame, entries, columns=None):
    dF = dataFrame
    if not columns:
        columns = [entry for entry in entries if entry != FOLDER and entries[entry].text() is not '']
    if False == True:  # if not settings[1].text():  # Union
        pass
    else:  # Intersection
        for column in columns:
            collection = []
            entry = pd.read_csv(StringIO(entries[column].text()), sep=',')
            for i in entry:
                if '<' in i or '>' in i or '-' in i:
                    collection = filter(dF, entry, collection, column)
                else:
                    if type(i) == str:
                        i = [i]  # homogenizing the type of i
                    collection.append(pd.DataFrame(dF[dF[column].isin(i)]))
            dF = pd.DataFrame(pd.concat([c for c in collection], join='inner', axis=0))
        return dF


def remove_duplicates(dataFrame):
    """
    Removes duplicate rows in a pandas DataFrame
    dataFrame: pandas DataFrame
    return: pandas DataFrame
    """
    s = pd.DataFrame([[dataFrame.index[i], dataFrame.iloc[i, 0]] for i in range(len(dataFrame))])
    s.drop_duplicates(inplace=True)
    r = pd.DataFrame([[s.iloc[i, 1]] for i in range(len(s))], index=[s.iloc[i, 0] for i in range(len(s))])
    return r


def rename_duplicates(dataFrame):
    """
    Rename duplicate rows in pandas data frame
    dataFrame: pandas DataFrame
    return: pandas DataFrame
    """
    data = dataFrame
    i = 2
    while True in data.index.duplicated():
        data.index = data.index.where(~data.index.duplicated(), data.index + '_' + str(i))
        i += 1
    return data


def ReduceStates(self, root):
    states = self.listGenerator.getAllStates()
    self.stateCodes = self.listGenerator.getAllStateCodes()
    with open(root + r"\Information\statesList.txt", "w") as statesList:
        with open(root + r"\Information\stateCodesList.txt", "w") as stateCodesList:
            if self.query['State Code'] is not '':  # CHECK IF STATE WAS SPECIFIED
                if not ("," in list(self.query['State Code'])):  # QUERY IS SINGLE STATE
                    for state in self.stateCodes:
                        if self.query['State Code'] == self.stateCodes[state]:
                            statesList.write(state + "\n")
                            stateCodesList.write(self.query['State Code'] + "\n")
                            self.found = True
                            break
                    if not self.found:
                        print(self.query['State Code'] + " is not a valid State Code. Search aborted.")
                        return
                else:  # QUERY IS MORE THAN ONE STATE
                    self.states = re.findall(r".?\w+.?", self.query['State Code'])
                    for stateSearch in self.states:
                        stripState = stateSearch.strip()
                        stripState = stripState.rstrip(",")
                        omega = 0
                        for state in self.stateCodes:
                            if stripState == self.stateCodes[state]:
                                statesList.write(states[omega] + "\n")
                                stateCodesList.write(stripState + "\n")
                                self.found = True
                                break
                            omega += 1
            else:
                for state in self.stateCodes:
                    statesList.write(state + "\n")
                    stateCodesList.write(self.stateCodes[state] + "\n")


def ReduceStates(self, root):
    states = self.listGenerator.getAllStates()
    self.stateCodes = self.listGenerator.getAllStateCodes()
    with open(root + r"\Information\statesList.txt", "w") as statesList:
        with open(root + r"\Information\stateCodesList.txt", "w") as stateCodesList:
            if self.query['State Code'] is not '':  # CHECK IF STATE WAS SPECIFIED
                if not ("," in list(self.query['State Code'])):  # QUERY IS SINGLE STATE
                    for state in self.stateCodes:
                        if self.query['State Code'] == self.stateCodes[state]:
                            statesList.write(state + "\n")
                            stateCodesList.write(self.query['State Code'] + "\n")
                            self.found = True
                            break
                    if not self.found:
                        print(self.query['State Code'] + " is not a valid State Code. Search aborted.")
                        return
                else:  # QUERY IS MORE THAN ONE STATE
                    self.states = re.findall(r".?\w+.?", self.query['State Code'])
                    for stateSearch in self.states:
                        stripState = stateSearch.strip()
                        stripState = stripState.rstrip(",")
                        omega = 0
                        for state in self.stateCodes:
                            if stripState == self.stateCodes[state]:
                                statesList.write(states[omega] + "\n")
                                stateCodesList.write(stripState + "\n")
                                self.found = True
                                break
                            omega += 1
            else:
                for state in self.stateCodes:
                    statesList.write(state + "\n")
                    stateCodesList.write(self.stateCodes[state] + "\n")


def saves(self):
    path = self.util.save_file(fileExt=QRY_FILES)
    try:
        if path[-4:] != QRY_EXT:
            path = path + QRY_EXT
        with open(path, "w") as saveFile:
            for i in range(HEADERS.__len__()):
                saveFile.write(self.entryList[FEATURES[i]].get() + "\n")
    except:
        return


def save_headers(headers):
    with open(HEADERS, "w") as file:
        for header in headers:
            file.write(header + '\n')


def save_file(self, fileExt, index=4):
    path = tkf.asksaveasfilename(filetypes=(fileExt, ALL_FILES))
    self.parent.util.entries[index].delete(0, END)
    self.parent.util.entries[index].insert(0, path)


def search(entries, settings, dtype, filepath, fileName, folderSuffix, files):
    data = []
    for state in files:
        for fld in folderSuffix:
            pth = DATABASE + fld + '\\' + state + fld + TXT_EXT
            data.append(read_data(pth, dtype))
            data[-1] = remove_nonmatches(settings, data[-1], entries)
            data[-1] = rename_duplicates(data[-1])
            export_results(data[-1], state, fld, filepath, fileName, settings)
    return data


def search(entries, dtype, file, dir, columns=None):
    data = read_data(DATABASE + os.path.join(dir, file), dtype)
    if not data.empty:
        data = remove_nonmatches(data, entries, columns)
    if not data.empty:
        data = rename_duplicates(data)
        export_results(data, file)
    return data


def search(entries, dir, file, dtype='str', columns=None):
    data = read_data(os.path.join(dir, file), dtype)
    data = remove_nonmatches(data, entries, columns)
    if data.empty:
        return data
    # data = rename_duplicates(data)
    return data


def search_resume(self, entries, savePath, settings):
    with open(TEMP_SAVE, 'w') as saveFile:
        for i in range(entries.__len__()):
            saveFile.write(entries[FEATURES[i]].text() + "\n")
    search(
        entries=entries, settings=settings, dtype={item: str for item in entries if item != FOLDER},
        filepath=savePath, fileName=name_file(savePath),
        folderSuffix=set_folders([str(i) for i in range(1992, 2017)], entries[FOLDER].text()),
        files=set_files(state_code_state(), entries[FILE].text())
    )


def search_resume(self, settings):
    entryList = self.entryList
    with open(ROOT + "\Database\\temp.txt", "w") as saveFile:
        for i in range(entryList.__len__()):
            saveFile.write(entryList[FEATURES[i]].get() + "\n")
    search(
        entries=entryList, settings=settings, dtype={item: str for item in entryList if item != FOLDER},
        filepath=self.filepath, fileName=name_file(self.filepath),
        folderSuffix=set_folders([str(i) for i in range(1992, 2017)], entryList[FOLDER].get()),
        files=set_files(self.state_code_state(), entryList[FILE].get())
    )


def select_folder(self, index=0):
    path = tkf.askdirectory(parent=self.top, initialdir="/", title='Select Folder')
    self.parent.util.entries[index].delete(0, END)
    self.parent.util.entries[index].insert(0, path)


# TODO this doesn't do anything
def save_query(widget, entries):
    path = save_file(widget, fileExt=QRY_FILES)
    try:
        if path[-4:] != QRY_EXT:
            path = path + QRY_EXT
        with open(path, "w") as saveFile:
            for i in range(HEADERS.__len__()):
                saveFile.write(entries[FEATURES[i]].get() + "\n")
    except:
        return


# TODO this doesn't do anything
def save_query(widget, entries):
    path = save_file(widget, fileExt=QRY_FILES)
    try:
        if path[-4:] != QRY_EXT:
            path = path + QRY_EXT
        with open(path, "w") as saveFile:
            for i in range(dh.HEADERS.__len__()):
                saveFile.write(entries[FEATURES[i]].get() + "\n")
    except:
        return


def set_files(states, entries):
    files = []
    if entries:
        state_codes = pd.read_csv(StringIO(entries), sep=',')
        for state in states:
            if state[:2] in state_codes or state in state_codes:
                files.append(states[state])
    else:
        files = [states[state] for state in states]
    return files


def set_folders(folders, entries):
    fld = folders
    if entries:
        fld = pd.read_csv(StringIO(entries), sep=',')
    return [str(i)[2:] for i in fld]


def setSearchEntryLists(container):
    entryList = {}
    with open(ITEM_NAMES, 'r') as f:
        for line in f:
            entryList[line.strip()] = ttk.Entry(container)
    return entryList


def sort_data(dataFrame):
    if type(dataFrame) == pd.DataFrame:
        sorted = dataFrame.sort_index(axis=0)
        sorted = sorted.sort_index(axis=1)
    elif type(dataFrame) == pd.Series:
        sorted = dataFrame.sort_index()
    return sorted


def sort_data(dataFrame, axis=None):
    """
    Sorts the columns and rows in a pandas DataFrame by numerical or alphabetical order
    dataFrame: pandas DataFrame
    return: pandas DataFrame
    """
    if type(dataFrame) == pd.DataFrame:
        if axis == 0 or axis == None:
            sorted = dataFrame.sort_index(axis=0)
        if axis == 1 or axis == None:
            sorted = sorted.sort_index(axis=1)
    elif type(dataFrame) == pd.Series:
        sorted = dataFrame.sort_index()
    return sorted


def state_code_state():
    states = {}
    with open(STATES, 'r') as states:
        with open(STATE_CODES, 'r') as state_codes:
            for state, state_code in zip(states, state_codes):
                states[state_code.strip()] = state.strip()
    return states


def update_max(level, max):
    """
    If level is greater than max, return level.
    Otherwise, return max.
    """
    if level > max:
        return level
    return max


def validate(widget, columns, checks):
    if not columns:
        QMessageBox.critical(widget, "Empty search", "Enter interface7 criteria.", QMessageBox.Ok)
        return False
    checked = False
    for check in checks:
        if check == True:
            checked = True
    if not checked:
        QMessageBox.critical(widget, "Unchecked Export", "Select an export option.", QMessageBox.Ok)
        return False
    return True


def vectorize_data(dataFrame, mode):
    """
    Converts a matrix of data into a vector. For test, if the matrix contains a row
    with an index labeled as '1' and 2 columns as '0', '1', and '1' containing the values
    'a', 'b', and 'c', respectively, the resulting vector will contain one single column with three rows.
    All three rows will be labeled as '1' with respect to the original row, and each value will be
    'a', 'b', and 'c', with respect to the original values. Values equivalent to None will be dropped.
    If mode is set to 'parallel', the matrix will be transposed before the algorithm is initiated.
    Example:
        0 1 1
    1   a b c
    1   5 1 Nan
    will become:
        0
    1   a
    1   b
    1   c
    1   5
    1   1
    dataFrame: pandas DataFrame
    mode: str - 'series' or 'parallel'
    return: pandas Series
    """
    if len(dataFrame.columns) == 1:
        return pd.Series(dataFrame.iloc[:, 0].as_matrix(), index=dataFrame.index)
    if mode == 'series':
        dF = pd.DataFrame(dataFrame)
    if mode == 'parallel':
        dF = pd.DataFrame(dataFrame.T)
    indices = []
    values = []
    for i, index in enumerate(dF.index):
        for j, column in enumerate(dF.columns):
            if type(dF.iloc[i, j]) != type(np.nan):
                indices.append(index)
                values.append(dF.iloc[i, j])
    prepared = pd.Series(values, index=indices)
    return prepared


###########################################

class AiBase(metaclass=ABCMeta):
    def __init__(self, parent):
        self.parent = parent

    @staticmethod
    def count_matrix(data, matrix, columns=None, hst=False):
        """
        Count data and store the count in matrix (pre-labeled DataFrame objects)
        columns is a list of column labels
        """
        if not hst:
            matrix.fillna(0, inplace=True)
        for i, row in enumerate(data.iterrows()):
            for j, column in enumerate(data.loc[row[0], :]):
                if hst:
                    matrix[data.iloc[i, j]].append(int(columns[j]) - 1991)

                elif columns:  # If columns are specified: row=the value in (i,j), column=column j
                    matrix.loc[data.iloc[i, j], columns[j]] += 1

                else:  # If nothing is specified: transition matrix is assumed
                    if (j < len(data.iloc[i, :]) - 1):
                        try:
                            if (int(data.iloc[i, j + 1]) <= int(data.iloc[i, j])):
                                matrix.loc[data.iloc[i, j], data.iloc[i, j + 1]] += 1
                        except:
                            pass

        return matrix


class DataDude:
    def __init__(self, app):
        try:
            self.app = app
            self.data = pd.DataFrame(
                0, index=np.arange(self.app.dataConfig.buffer),
                columns=['__timestamp__'] + self.app.dataConfig.head.split(
                    self.app.dataConfig.delimiter
                ), )
        except Exception as e:
            statusMsg = 'Data error: {}'.format(str(e))
            self.app.update_status('mainMenu', statusMsg, 'error')

    def __init__(self, buffer, head, delimiter, dataType):
        # app.dataConfig.buffer
        # app.dataConfig.head
        # app.dataConfig.delimiter
        # app.dataConfig.dataType
        self.buffer = buffer
        self.head = head
        self.delimiter = delimiter
        self.dataType = dataType
        self.data = pd.DataFrame(
            0, index=np.arange(self.buffer),
            columns=['__timestamp__'] + self.head.split(self.delimiter), )

    def __init__(self, parent):
        try:
            self.parent = parent
            self.data = pd.DataFrame(
                0, index=np.arange(self.parent.configuration.py.buffer),
                columns=['__timestamp__'] + self.parent.configuration.py.head.split(
                    self.parent.configuration.py.delimiter
                ), )
        except Exception as e:
            alertMsg = 'Data error: {}'.format(str(e))
            self.parent.update_status(alertMsg, 'error')

    def export_data(self, dataFrame, savePath, format, suffix=''):
        """
        dataFrame: pandas DataFrame
        savePath: str
        formats: list of str
        """
        try:
            path = os.path.splitext(savePath)[0] + suffix
            fileName = os.path.basename(os.path.normpath(savePath))
            if format == 'csv':
                dataFrame.to_csv(path_or_buf=path + '.txt', index=False)
            if format == 'xlsx':
                try:  # excel workbook exists
                    writer = pd.ExcelWriter(path + '.xlsx', engine='openpyxl')
                    writer.book = load_workbook(path + '.xlsx')
                    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
                except:  # excel workbook is created if it does not exist
                    writer = pd.ExcelWriter(path + '.xlsx', engine='xlsxwriter')
                dataFrame.to_excel(writer, sheet_name=fileName[-15:-5], index=False, )
                writer.save()
        except Exception as e:
            print('Data export error: ' + str(e))

    def export_data(self, dataFrame, savePath, format, suffix=''):
        """
        dataFrame: pandas DataFrame
        savePath: str
        formats: list of str
        """
        path = os.path.splitext(savePath)[0] + suffix
        fileName = os.path.basename(os.path.normpath(savePath))
        if format == 'csv':
            dataFrame.to_csv(path_or_buf=path + '.txt', index=False)
        if format == 'xlsx':
            try:  # excel workbook exists
                writer = pd.ExcelWriter(path + '.xlsx', engine='openpyxl')
                writer.book = load_workbook(path + '.xlsx')
                writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
            except:  # excel workbook is created if it does not exist
                writer = pd.ExcelWriter(path + '.xlsx', engine='xlsxwriter')
            dataFrame.to_excel(writer, sheet_name=fileName[-15:-5], index=False, )
            writer.save()

    def get(self, var):
        return self.data[var]

    def get(self, var):
        try:
            return self.data[var]
        except Exception as e:
            statusMsg = 'Get data error: {}.'.format(str(e))
            self.app.update_status('mainMenu', statusMsg, 'error')

    def get(self, var):
        try:
            return self.data[var]
        except Exception as e:
            alertMsg = 'Get data error: {}.'.format(e)
            self.parent.update_status(alertMsg, 'error')

    def get_plottable(self):
        results = []
        head = self.app.dataConfig.head.split(self.app.dataConfig.delimiter)
        for _index, _type in enumerate(self.app.dataConfig.dataType.split(self.app.dataConfig.delimiter)):
            if _type in ['d', 'f']:
                results.append(head[_index])
        return results

    def get_plottable(self, *args, **kwargs):
        results = []
        head = self.head.split(self.delimiter)
        for _index, _type in enumerate(self.dataType.split(self.delimiter)):
            if _type in ['d', 'f']:
                results.append(head[_index])
        return results

    def update_data(self, data):
        _data = [self.timestampHMSms] + data.split(self.delimiter)
        _new = pd.DataFrame([_data], columns=['__timestamp__'] + self.head.split(self.delimiter))
        self.data = self.data.append(_new, ignore_index=True)
        buffer = self.buffer
        difference = len(self.data) - buffer
        if difference > 0:
            self.data = self.data[difference:]

    def update_data(self, data):
        try:
            _data = [self.timestampHMSms] + data.split(self.app.dataConfig.delimiter)
            _new = pd.DataFrame(
                [_data], columns=['__timestamp__'] + self.app.dataConfig.head.split(
                    self.app.dataConfig.delimiter
                )
            )
            self.data = self.data.append(_new, ignore_index=True)
            buffer = self.app.dataConfig.buffer
            difference = len(self.data) - buffer
            if difference > 0:
                self.data = self.data[difference:]
        except Exception as e:
            statusMsg = 'Data update error: {}.'.format(e)
            self.app.plotMenu.update(statusMsg, 'error')

    def update_data(self, data):
        try:
            _data = [self.timestampHMSms] + data.split(self.parent.configuration.py.delimiter)
            _new = pd.DataFrame(
                [_data],
                columns=['__timestamp__'] + self.parent.configuration.py.head.split(
                    self.parent.configuration.py.delimiter
                )
            )
            self.data = self.data.append(_new, ignore_index=True)
            buffer = self.parent.configuration.py.buffer
            difference = len(self.data) - buffer
            if difference > 0:
                self.data = self.data[difference:]
        except Exception as e:
            print('Data update error: ' + str(e))

    def update_data_old(self, data):
        try:
            _data = data.split(',')
            _new = pd.DataFrame([_data], columns=self.parent.preferences.head.split(','))
            self.data = self.data.append(_new, ignore_index=True)
            buffer = self.parent.preferences.buffer
            difference = len(self.data) - buffer
            if difference > 0:
                self.data = self.data[difference:]
        except Exception as e:
            print('Data update error: ' + str(e))

    @property
    def time(self):
        return time.time()

    @property
    def time(self, *args, **kwargs):
        return time.time()

    @property
    def date(self):
        return datetime.now().strftime('%m_%d_%Y')

    @property
    def date(self, *args, **kwargs):
        return datetime.now().strftime('%m_%d_%Y')

    @property
    def timestampHMS(self):
        return time.strftime('%H:%M:%S')

    @property
    def timestampHMS(self, *args, **kwargs):
        return time.strftime('%H:%M:%S')

    @property
    def timestampHMSms(self, *args, **kwargs):
        return datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]

    @property
    def timestampHMSms(self):
        return datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]


class DataProcessor:
    def __init__(self, source):
        self.root = os.getcwd()
        self.set_source(source)

    @property
    def scenarioDirs(self):
        return os.listdir(self.source)

    @property
    def source(self):
        return self._source

    def set_source(self, directory):
        self._source = os.path.join(self.root, directory)

    def move_raid(self, destination):
        """
        Utility function to copy raid functions to working folder.
        If the destination does not exist, it will be created inside the
        source folder.
        Iterates over all directories in the source folder.
        For each directory identified, a directory with the same name will
        be created inside the destination folder.
        For each file in a directory, any .raid and .raidz files are identified.
        The .raid files are copied to the corresponding folder inside the
        destination folder. .raidz files are uncompressed and transfered
        to the destination folder.
        :param destination:
        :return:
        """
        dst = os.path.join(self.root, destination)
        self._mkdir(dst)
        for scenarioDir in self.scenarioDirs:
            scenarioFiles = os.listdir(os.path.join(self.source, scenarioDir))
            for file in scenarioFiles:
                split = file.split('.')
                if len(split) > 1:
                    if split[1] == 'raidz':
                        _src = os.path.join(self.source, scenarioDir, file, )
                        _dst = os.path.join(dst, scenarioDir, file, )
                        self._mkdir(os.path.join(dst, scenarioDir))
                        with gzip.open(_src, 'rb') as f_in:
                            with open(_dst.replace('raidz', 'raid'), 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    if split[1] == 'raid':
                        _src = os.path.join(self.source, scenarioDir, file, )
                        _dst = os.path.join(dst, scenarioDir, file, )
                        self._mkdir(os.path.join(dst, scenarioDir))
                        shutil.copyfile(_src, _dst)

    def _mkdir(self, path):
        if os.path.isdir(path):
            return
        os.mkdir(path)

    def make_scenario(self, index):
        for scenarioDir in self.scenarioDirs:
            scenarioFiles = os.listdir(os.path.join(self.source, scenarioDir))
            for file in scenarioFiles:
                fullpath = os.path.join(self.source, scenarioDir, file)
                with open(fullpath, 'r') as f:
                    j = json.load(f)
                break
            break
        return j

    def file_to_json(self, path):
        """
        Loads the data contained in the RAID file specified
        by the path variable using the json library.
        The resulting list is returned.
        Each item in the list is an entry in the database
        contained in the RAID file.
        :param path:
        :return:
        """
        return json.load(path)

    def get_scenario(self, index):
        """
        :param index:
        :return:
        """
        path = os.path.join(self.source, self.scenarioDirs[index])
        path = os.path.join(path, os.listdir(path)[0])
        return scenario.Scenario(path)


class Encryptor:
    def __init__(self, app):
        self.app = app
        self.encryptions = ['None', 'XOR(0)', 'XOR Sequence']

    def __init__(self, app):
        self.app = app
        self.encryptions = ['None', 'XOR(0)', 'XOR Sequence']

    def encrypt(self, msg):
        try:
            e = self.app.encryptionConfig.encryption
            if e == 'XOR(0)':
                return self.encrypt_xor(msg)
            return msg
        except Exception as e:
            self.app.transceiverMenu.update_status(*status.encryptionError, e)

    def encrypt(self, msg):
        try:
            e = self.app.encryptionConfig.encryption
            if e == 'XOR(0)':
                return self.encrypt_xor(msg)
            return None
        except Exception as e:
            self.app.transceiverMenu.update_status(*status.encryptionError, e)

    def encrypt_xor(self, msg):
        try:
            encrypted = ''
            for c in msg:
                encrypted += str(ord(c) ^ 0)
            return encrypted
        except Exception as e:
            self.app.transceiverMenu.update_status(*status.xorEncryptionError, e)

    def encrypt_xor(self, msg):
        try:
            encrypted = ''
            for c in msg:
                encrypted += str(ord(c) ^ 0)
            return encrypted
        except Exception as e:
            self.app.transceiverMenu.update_status(*status.xorError, e)


class Filter:
    operators = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
    indicators = ('--', '-')

    def __init__(self, parent):
        self.parent = parent
        self.operators = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
        self.indicators = ('--', '-')

    def __init__(self, search, item):
        self.search = search
        self.item = item
        self.operators = {'>': operator.gt, '<': operator.lt, '>=': operator.ge, '<=': operator.le}
        self.operator = False

    def __init__(self, query, search, item):
        self.query = query
        self.search = search
        self.item = item
        self.operators = {'>': operator.gt, '<': operator.lt, '>=': operator.ge, '<=': operator.le}
        self.operator = False

    def __del__(self):
        pass

    def range(self, entry, field):
        for value in entry:
            for i in range(len(self.indicators)):
                if self.indicators[i] in value:
                    r = pd.read_csv(StringIO(value), sep=self.indicators[i], engine='python')
                    self.parent.tests.verify_range_entry(r, value, field)
                    if self.indicators[i] == '-':
                        yield ((self.operators['>'], self.operators['<']), r)
                    elif self.indicators[i] == '--':
                        yield ((self.operators['>='], self.operators['<=']), r)
                    break

    def checkInequalityRequest(self, search_list):
        self.search_list = search_list
        for operator in self.operators:
            if (operator in list(self.search_list)):
                if ('=' in list(self.search_list)):
                    return str(operator + '=')
                else:
                    return str(operator)
            else:
                pass

        return False

    def checkRangeRequest(self, search_list):
        self.search_list = search_list

        if (re.findall(r'\w+[-].?\w+', (self.search_list)).__len__() > 0):
            if ('-' in list(self.search_list)):
                counter = 0
                for item in list(self.Search.search[self.item]):
                    if item == '-':
                        counter += 1
                    if counter == 2:
                        return str('--')

                else:
                    return str('-')

            else:
                pass

        return False

    def FilterInequality(self, stripSearch, stripObject):
        stripSearch = stripSearch.lstrip(self.operator)

        if self.operators[self.operator](int(stripObject), int(stripSearch)):
            return True

        else:
            return False

    def FilterMultipleValues(self, filterParameters):
        self.stripObject = filterParameters[0]
        self.parameterNamesWithRespectToNumber = filterParameters[1]
        temporary_search = self.Search.GUI.populator.query[self.parameterNamesWithRespectToNumber[self.item]]
        self.temporaryItemSearchList = re.findall(
            r'[<>]=?\w+,?|\w+-+\w+,?|\w+,?',
            temporary_search
        )  # Interpreting interface7 criteria

        for value in list(self.temporaryItemSearchList):

            if self.item == '1':
                stripValue = value.lstrip("0").strip(",").strip()[:2]
            else:
                stripValue = value.lstrip("0").strip(",").strip()

            self.operator = self.checkInequalityRequest(value)

            if self.operator:

                if self.FilterInequality(stripValue, self.stripObject):
                    return True

                else:
                    pass

            elif self.FilterRange(stripValue, self.stripObject, self.checkRangeRequest(value)):
                return True

            else:

                if stripValue != self.stripObject:
                    pass

                else:
                    return True

        return False

    def FilterRange(self, stripSearch, stripObject, range):
        stripSearch_left = stripSearch[:1]
        stripSearch_right = stripSearch[stripSearch.__len__() - 1:]
        stripObject = stripObject

        try:
            if range == '-':
                operator_list = [operator.lt, operator.gt]
            else:
                operator_list = [operator.le, operator.ge]

            if (operator_list[0](int(stripSearch_left), int(stripObject))) and (
                operator_list[1](int(stripSearch_right), int(stripObject))):
                return True

            else:
                return False
        except:
            pass  # print("ERROR")

    def FilterSingleValue(self, filterParameters):
        self.stripSearch = filterParameters[0]
        self.stripObject = filterParameters[1]
        self.operator = self.checkInequalityRequest(self.Search.search[self.item])

        if self.operator:

            return self.FilterInequality(self.stripSearch, self.stripObject)

        elif self.FilterRange(self.stripSearch, self.stripObject, self.checkRangeRequest(self.stripSearch)):
            return True

        else:

            if (self.stripSearch == self.stripObject):
                return True  # Item matches interface7

            else:
                return False  # Item does not match interface7

    def FilterSingleValue(self, filterParameters):
        self.stripSearch = filterParameters[0]
        self.stripObject = filterParameters[1]
        self.operator = self.checkInequalityRequest(self.search[self.item])

    def FilterMultipleValues(self, filterParameters):
        self.stripObject = filterParameters[0]
        self.parameterNamesWithRespectToNumber = filterParameters[1]
        temporary_search = self.query[self.parameterNamesWithRespectToNumber[self.item]]
        self.temporaryItemSearchList = re.findall(
            r'[<>]=?\w+,?|\w+-+\w+,?|\w+,?',
            temporary_search
        )  # Interpreting interface7 criteria
        for value in list(self.temporaryItemSearchList):

            if self.item == '1':
                stripValue = value.lstrip("0").strip(",").strip()[:2]
            else:
                stripValue = value.lstrip("0").strip(",").strip()

            self.operator = self.checkInequalityRequest(value)

            if self.operator:

                if self.FilterInequality(stripValue, self.stripObject):
                    return True

                else:
                    pass

            elif self.FilterRange(stripValue, self.stripObject, self.checkRangeRequest(value)):
                return True

            else:

                if stripValue != self.stripObject:
                    pass

                else:
                    return True
        return False

    def Run_Filter(self, filterParameters, case):
        self.filterParameters = filterParameters
        self.case = case

        if self.case == 0:
            return self.FilterSingleValue(
                filterParameters
            )  # Filter possible range requests in known single value interface7
        elif self.case == 1:  # In the case of multiple values, a list of all values is iterated and each value in the list is checked for range requests
            return self.FilterMultipleValues(filterParameters)

    def checkRangeRequest(self, search_list):
        self.search_list = search_list

        if (re.findall(r'\w+[-].?\w+', (self.search_list)).__len__() > 0):
            if ('-' in list(self.search_list)):
                counter = 0
                for item in list(self.search[self.item]):
                    if item == '-':
                        counter += 1
                    if counter == 2:
                        return str('--')

                else:
                    return str('-')

            else:
                pass

        return False

    @staticmethod
    def range(entry, field):
        for value in entry:
            for i in range(len(Filter.indicators)):
                if Filter.indicators[i] in value:
                    r = pd.read_csv(StringIO(value), sep=Filter.indicators[i], engine='python')
                    # self.test.verify_range_entry(r, value, field)
                    if Filter.indicators[i] == '-':
                        yield ((Filter.operators['>'], Filter.operators['<']), r)
                    elif Filter.indicators[i] == '--':
                        yield ((Filter.operators['>='], Filter.operators['<=']), r)
                    break


class InitiateSearch:
    def __init__(self, parameterNames, parameterNumbers, query):
        self.finalResults = []  # THE RESULTS OF THE SEARCH WILL BE SAVED HERE FOR EACH YEAR, THE LIST IS CLEARED AFTER IT'S WRITTEN
        self.parameterNames = parameterNames  # LIST OF ALL ITEM NAMES PASSED IN FROM THE INTERFACE
        self.parameterNumbers = parameterNumbers  # LIST OF ALL ITEM NUMBERS PASSED IN FROM THE INTERFACE
        self.query = query  # THIS DICTIONARY STORES ALL THE VALUES IN THE SEARCH
        self.found = True  # THIS BOOLEAN BECOMES FALSE AS SOON AS ONE OF THE SEARCH VALUES DIFFERS FROM A BRIDGE'S ITEM VALUE AND THE BRIDGE IS DISCARDED
        self.initial_year_iteration = 0
        self.years_to_iterate = 25
        self.years, self.states = [], []

        # REDUCE YEARS BY CHECKING WHETHER A YEAR WAS SPECIFIED
        try:
            if (self.query['Year']) is not '':  # CHECK IF YEAR WAS SPECIFIED
                if not ("," in list(self.query['Year'])):
                    if (int(self.query['Year']) >= starting_year) and (
                        int(self.query['Year']) <= current_year):  # DETERMINE WHETHER SPECIFIED YEAR IS VALID
                        self.initial_year_iteration = int(
                            self.query['Year']
                        )  # SET THE STARTING AND ENDING POINTS OF THE SEARCH (YEARS)
                        self.years_to_iterate = self.initial_year_iteration + 1
                        for j in range(self.initial_year_iteration, self.years_to_iterate):
                            years_to_search.append(j)
                    else:
                        print(self.query['Year'] + " is not a valid year. search aborted.")
                        return
                else:
                    self.years = re.findall(r".?\w+.?", self.query['Year'])
                    for year in self.years:
                        stripYear = year.lstrip("0")
                        stripYear = stripYear.strip()
                        years_to_search.append(stripYear.rstrip(","))
            else:
                for j in range(starting_year, current_year + 1):
                    years_to_search.append(j)

        except:
            pass

        # REDUCE STATES BY CHECKING WHETHER A STATE WAS SPECIFIED
        with open(MASTERPATH + '\BridgeDataQuery\Database\Information\\allStates.txt', 'r') as allStates:
            for line in allStates:  # POPULATE A LIST OF ALL STATES
                states.append(line.strip())

        with open(MASTERPATH + '\BridgeDataQuery\Database\Information\\allStateCodes.txt', 'r') as allStateCodes:
            i = 0
            for line in allStateCodes:
                stateCodes[states[i]] = line.strip()  # POPULATE A LIST OF ALL STATE CODES
                i += 1
        with open(MASTERPATH + "\\BridgeDataQuery\Database\Information\statesList.txt", "w") as statesList:
            with open(MASTERPATH + "\\BridgeDataQuery\Database\Information\\stateCodesList.txt", "w") as stateCodesList:
                if self.query['State Code'] is not '':  # CHECK IF STATE WAS SPECIFIED
                    if not ("," in list(self.query['State Code'])):  # QUERY IS SINGLE STATE
                        for state in stateCodes:
                            if self.query['State Code'] == stateCodes[state]:  #
                                statesList.write(state)
                                stateCodesList.write(self.query['State Code'])
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
                            for state in stateCodes:
                                if stripState == stateCodes[state]:
                                    statesList.write(states[omega] + "\n")
                                    stateCodesList.write(stripState + "\n")
                                    self.found = True
                                    break
                                omega += 1

                else:
                    for state in stateCodes:
                        statesList.write(state + "\n")
                        stateCodesList.write(stateCodes[state] + "\n")

        for year in years_to_search:
            first = True
            with open(MASTERPATH + "\\BridgeDataQuery\Database\Information\statesList.txt", "r") as statesList:
                for state in statesList:
                    TextParser(self, str(year), state.strip(), self.parameterNames, self.parameterNumbers, self.query)
                    writeReport(year, MASTERPATH, self.finalResults, first)
                    first = False
                    self.finalResults.clear()

        years_to_search.clear()
        self.years.clear()

        print("report_generator0 has been generated.")

    def setFinalResults(self, bridge):
        self.finalResults.append(bridge)

    def __init__(self, parameterNames, parameterNumbers, query):
        self.finalResults = []  # THE RESULTS OF THE SEARCH WILL BE SAVED HERE FOR EACH YEAR, THE LIST IS CLEARED AFTER IT'S WRITTEN
        self.parameterNames = parameterNames  # LIST OF ALL ITEM NAMES PASSED IN FROM THE INTERFACE
        self.parameterNumbers = parameterNumbers  # LIST OF ALL ITEM NUMBERS PASSED IN FROM THE INTERFACE
        self.query = query  # THIS DICTIONARY STORES ALL THE VALUES IN THE SEARCH
        self.found = True  # THIS BOOLEAN BECOMES FALSE AS SOON AS ONE OF THE SEARCH VALUES DIFFERS FROM A BRIDGE'S ITEM VALUE AND THE BRIDGE IS DISCARDED
        self.initial_year_iteration = 0
        self.years_to_iterate = 25
        self.years, self.states = [], []

        # REDUCE YEARS BY CHECKING WHETHER A YEAR WAS SPECIFIED
        try:
            if (self.query['Year']) is not '':  # CHECK IF YEAR WAS SPECIFIED
                if not ("," in list(self.query['Year'])):
                    if (int(self.query['Year']) >= starting_year) and (
                        int(self.query['Year']) <= current_year):  # DETERMINE WHETHER SPECIFIED YEAR IS VALID
                        self.initial_year_iteration = int(
                            self.query['Year']
                        )  # SET THE STARTING AND ENDING POINTS OF THE SEARCH (YEARS)
                        self.years_to_iterate = self.initial_year_iteration + 1
                        for j in range(self.initial_year_iteration, self.years_to_iterate):
                            years_to_search.append(j)
                    else:
                        print(self.query['Year'] + " is not a valid year. Search aborted.")
                        return
                else:
                    self.years = re.findall(r".?\w+.?", self.query['Year'])
                    for year in self.years:
                        stripYear = year.lstrip("0")
                        stripYear = stripYear.strip()
                        years_to_search.append(stripYear.rstrip(","))
            else:
                for j in range(starting_year, current_year + 1):
                    years_to_search.append(j)

        except:
            pass

        # REDUCE STATES BY CHECKING WHETHER A STATE WAS SPECIFIED
        with open(MASTERPATH + '\BridgeDataQuery\Database\Information\\allStates.txt', 'r') as allStates:
            for line in allStates:  # POPULATE A LIST OF ALL STATES
                states.append(line.strip())

        with open(MASTERPATH + '\BridgeDataQuery\Database\Information\\allStateCodes.txt', 'r') as allStateCodes:
            i = 0
            for line in allStateCodes:
                stateCodes[states[i]] = line.strip()  # POPULATE A LIST OF ALL STATE CODES
                i += 1
        with open(MASTERPATH + "\\BridgeDataQuery\Database\Information\statesList.txt", "w") as statesList:
            with open(MASTERPATH + "\\BridgeDataQuery\Database\Information\\stateCodesList.txt", "w") as stateCodesList:
                if self.query['State Code'] is not '':  # CHECK IF STATE WAS SPECIFIED
                    if not ("," in list(self.query['State Code'])):  # QUERY IS SINGLE STATE
                        for state in stateCodes:
                            if self.query['State Code'] == stateCodes[state]:  #
                                statesList.write(state)
                                stateCodesList.write(self.query['State Code'])
                                self.found = True
                                break

                        if not self.found:
                            print(self.query['State Code'] + " is not a valid State Code. Search aborted.")
                            return

                    else:  # QUERY IS MORE THAN ONE STATE
                        self.states = re.findall(r".?\w+.?", self.query['State Code'])
                        for stateSearch in self.states:
                            stripState = stateSearch.strip()
                            stripState = stripState.rstrip(",")
                            omega = 0
                            for state in stateCodes:
                                if stripState == stateCodes[state]:
                                    statesList.write(states[omega] + "\n")
                                    stateCodesList.write(stripState + "\n")
                                    self.found = True
                                    break
                                omega += 1

                else:
                    for state in stateCodes:
                        statesList.write(state + "\n")
                        stateCodesList.write(stateCodes[state] + "\n")

        for year in years_to_search:
            first = True
            with open(MASTERPATH + "\\BridgeDataQuery\Database\Information\statesList.txt", "r") as statesList:
                for state in statesList:
                    TextParser(self, str(year), state.strip(), self.parameterNames, self.parameterNumbers, self.query)
                    writeReport(year, MASTERPATH, self.finalResults, first)
                    first = False
                    self.finalResults.clear()

        years_to_search.clear()
        self.years.clear()

        print("Report has been generated.")

    def setFinalResults(self, bridge):
        self.finalResults.append(bridge)

    def __init__(
        self, MASTERPATH, parameterNames, parameterNumbers, query, listGenerator, files='multiple',
        filepath='report'
    ):
        self.finalResults = []  # THE RESULTS OF THE SEARCH WILL BE SAVED HERE FOR EACH YEAR, THE LIST IS CLEARED AFTER IT'S WRITTEN
        self.parameterNames = parameterNames  # LIST OF ALL ITEM NAMES PASSED IN FROM THE INTERFACE
        self.parameterNumbers = parameterNumbers  # LIST OF ALL ITEM NUMBERS PASSED IN FROM THE INTERFACE
        self.query = query  # THIS DICTIONARY STORES ALL THE VALUES IN THE SEARCH
        self.found, first = True, True  # THIS BOOLEAN BECOMES FALSE AS SOON AS ONE OF THE SEARCH VALUES DIFFERS FROM A BRIDGE'S ITEM VALUE AND THE BRIDGE IS DISCARDED
        self.initial_year_iteration, self.years_to_iterate = 0, 24
        self.years, self.states, settings = [], [], []

        # CHECKING REPORT PREFERENCES
        with open(
            MASTERPATH + "\BridgeDataQuery\\Utilities14\Temporary_Report_Preferences.txt",
            "r"
        ) as report_preferences:
            for line in report_preferences:
                if line.strip() == 'True':
                    settings.append(True)
                else:
                    settings.append(False)

        dataReducer = NBIData.DataReducer(
            self.query, starting_year, current_year, self.years_to_iterate,
            MASTERPATH, listGenerator
        )

        # REDUCE YEARS AND STATES BY CHECKING WHETHER A YEAR WAS SPECIFIED
        years_to_search = dataReducer.ReduceYears()
        dataReducer.ReduceStates(listGenerator)
        self.list_of_states_to_search = listGenerator.getListOfStatesToSearch()

        for year in years_to_search:
            self.setFinalResults(
                "##########################################" + str(year) + "##########################################"
            )
            if files == 'multiple':
                first = True
            for state in self.list_of_states_to_search:
                RunSearch.RunSearch(
                    MASTERPATH, self, str(year), state.strip(), self.parameterNames,
                    self.parameterNumbers, self.query, listGenerator
                )
                WriteReport.WriteReport(year, MASTERPATH, self.finalResults, first, files, filepath, settings)
                first = False
                self.finalResults.clear()

        years_to_search.clear()
        self.years.clear()

        print("report_generator0 has been generated.")

    def setFinalResults(self, bridge):
        self.finalResults.append(bridge)


class Model:
    def __init__(self, item, files, clean=True, iterations=1, initialState=1, validStates=(0, 1)):
        try:
            self.iS = np.transpose((pd.Series(initialState, index=validStates)))
            self.vS = validStates
            self.data, self.yrs = Model.process_data(
                item=item, dir=files, paths=read_folder(dir=files, ext=TXT_EXT),
                clean=clean
            )
            self.sampler = None
        except:
            pass

    def __init__(self, parent):
        self.parent = parent
        self.iterations = int(parent.parent.util.entries[1].get())
        try:
            self.iS = np.transpose((pd.Series(parent.iSV, index=parent.vS)))
            self.vS = parent.vS
            self.data, self.yrs = Model.process_data(
                item=parent.item.get(), dir=parent.fp_list,
                paths=read_folder(dir=parent.fp_list, ext=TXT_EXT),
                clean=parent.settings[0].get()
            )
        except:
            pass

    def __init__(self, parent):
        self.parent = parent  # parent is Markov Chain Menu
        self.i_sv = parent.i_sv  # Initial state vector
        self.iterations = 24  # The number of iterations that the Markov Chain will run
        self.sig_figs = 30  # The number of significant figures the program will use in calculations
        self.size = len(self.i_sv)  # The size of the initial state vector
        self.item = self.parent.item.get()
        self.tm_object = TransitionMatrix.TransitionMatrix(self, [str(year) for year in range(1992, 2017)])
        # The matrix of probability of transitioning between states (dictionary of dictionaries)
        self.tm = self.tm_object.get_tm()  # TRANSITION MATRIX
        # List of dictionaries
        # Dictionary = year(independent variable), Value = dependent variable
        self.pv_list = [self.i_sv]
        self.insert_pv()
        self.transpose()
        self.monte_carlo()

    def __init__(self, item, files, clean=True, iterations=1, initialState=1, validStates=(0, 1)):
        try:
            self.iS = np.transpose((pd.Series(initialState, index=validStates)))
            self.vS = validStates
            self.data, self.yrs = Model.process_data(
                item=item, dir=files, paths=read_folder(dir=files, ext=TXT_EXT),
                clean=clean
            )
        except:
            pass
        svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1)
        y_rbf = svr_rbf.fit(self.data, self.yrs).predict(self.data)
        plt.scatter(self.data, self.yrs, color='darkorange', label='data')
        plt.plot(self.data, y_rbf, color='navy', lw=2, label='RBF model')
        plt.legend()
        plt.show()

    def __init__(self, parent):
        self.parent = parent
        self.iterations = int(parent.parent.util.entries[1].get())
        try:
            self.iterations = int(parent.parent.util.entries[1].get())
            self.iS = np.transpose((pd.Series(parent.iSV, index=parent.vS)))
            self.vS = parent.vS
            self.data, self.yrs = Model.process_data(
                item=parent.item.get(), dir=parent.fp_list,
                paths=self.read_folder(dir=parent.fp_list, ext='.txt'),
                clean=parent.settings[0].get()
            )
        except:
            pass

    def clean_data(data, columns):
        """
        Clears incomplete data resulting in regular matrix
        data is a pandas DataFrame and columns are the columns in data
        """
        cln = data
        for column in columns:
            cln = (cln[np.isfinite(cln[column].astype(float))])
        cln.columns = columns
        return cln

    def insert_pv(self):
        # This for loop inserts probability vector dictionaries into the vector list.
        for m in range(self.iterations):
            self.new_pv = {}
            self.populate_pv(m)
        self.results = []

    def markov_chain(self, m, state_1):
        # Initially, this for loop will multiply the initial state vector and the transition matrix.
        # Then, it will multiply the result by the transition matrix.
        # This process is repeated until the outputs fill the vector list to an equivalent size.
        for state_2 in (self.i_sv):
            self.new_value += self.pv_list[m][state_2] * self.tm[state_2][state_1]  # MARKOV CHAIN
        self.new_pv[state_1] = (round(self.new_value, self.sig_figs))  # Inserting the result to the probability vector

    def monte_carlo(self):
        self.sub_model = MonteCarlo.MonteCarlo(self, self.legend_labels)  # print(self.probability_vector_list)

    def populate_pv(self, m):
        for state_1 in (self.i_sv):
            self.new_value = 0  # Placeholder initial value for the items to be inserted into the empty dictionary
            self.markov_chain(
                m,
                state_1
            )  # Once the probability vector is filled, it will be inserted to the vector list
        self.pv_list.append(self.new_pv)  # Inserting the probability vector to the vector list

    def raw_frq(self, focus):
        matrix = Model.count_matrix(self.data, pd.DataFrame(index=self.vS, columns=self.yrs), self.yrs)
        return Model.frq(focus, matrix)

    def raw_hst(self, focus):
        if focus == "year":
            return self.data.T.as_matrix().astype(np.float64)
        elif focus == "state":
            return Model.hst(data=self.data, matrix={s: [] for s in self.vS}, columns=self.yrs)

    def transpose(self):
        # TRANSPOSING MATRIX
        self.legend_labels = []
        for j, k in enumerate(self.pv_list[0]):
            self.results.append([])
            self.legend_labels.append(str(k))
            self.results[j] = [self.pv_list[i][k] for i in range(len(self.pv_list))]

    @staticmethod
    def count_matrix(data, matrix, columns=None, hst=False):
        """
        Count data and store the count in matrix (pre-labeled DataFrame objects)
        columns is a list of column labels
        """
        if not hst:
            matrix.fillna(0, inplace=True)
        for i, row in enumerate(data.iterrows()):
            for j, column in enumerate(data.loc[row[0], :]):
                if hst:
                    matrix[data.iloc[i, j]].append(int(columns[j]) - 1991)
                elif columns:  # If columns are specified: row=the value in (i,j), column=column j
                    matrix.loc[data.iloc[i, j], columns[j]] += 1
                else:  # If nothing is specified: transition matrix is assumed
                    if (j < len(data.iloc[i, :]) - 1):
                        try:
                            if (int(data.iloc[i, j + 1]) <= int(data.iloc[i, j])):
                                matrix.loc[data.iloc[i, j], data.iloc[i, j + 1]] += 1
                        except:
                            pass
        return matrix

    @staticmethod
    def frq(focus, matrix):
        """
        Count state transitions in data and store the count in matrix (pre-labeled DataFrame objects)
        """
        matrix = matrix
        if focus == "year":
            frq = matrix.T
            frq = frq[frq.columns[::-1]]
            return normalize_rows(frq).as_matrix()
        elif focus == "state":
            return normalize_rows(matrix.T).T.as_matrix()
        else:
            print("Select focus")

    @staticmethod
    def hst(data, matrix, columns):  # TODO bad method + needs generalizing (only works for freq vs state)
        hst = Model.count_matrix(data=data, matrix=matrix, columns=columns, hst=True)
        hst = pd.DataFrame([hst[m] for m in hst], index=[m for m in hst])
        hst.fillna(0, inplace=True)
        hst.sort_index(ascending=False, inplace=True)
        return hst.as_matrix().astype(np.float64)

    @staticmethod
    def normalize(matrix):
        """
        Normalize matrix by row
        """
        try:
            nrm = matrix.div(matrix.sum(axis=1), axis=0)
            nrm.fillna(0, inplace=True)
            return nrm
        except:
            print("Division by zero")

    @staticmethod
    def process_data(item, dir, paths, clean=False):
        """
        :param item:
        :param dir:
        :param paths:
        :param clean:
        :return:
        """
        data = []
        yrs = []
        for i, file in enumerate(paths):
            data.append(
                pd.read_csv(
                    dir + '/' + file, usecols=[id] + [item], na_values=NA_VALUES,
                    dtype={**{id: str}, **{item: str}}, encoding='latin1'
                )
            )
            yrs.append(re.findall(r"[0-5]{3}(?=.txt)", file)[0])
            data[i].set_index(id, inplace=True)
        data = Model.concat_results(data=data, columns=yrs, index=item)
        if clean:
            data = Model.clean_data(data=data, columns=yrs)
        return data, yrs

    @staticmethod
    def process_data(item, dir, paths, clean=False):
        """
        item is the item of interest (str)
        dir is the directory where the files of data are found (str)
        paths is all the paths of data in the directory (list of str)
        _clean determines whether or not incomplete data is removed
        returns a list of pandas dataFrames and a list of numerics to name the files
        """
        data = []
        yrs = []
        for i, file in enumerate(paths):
            data.append(
                pd.read_csv(
                    dir + '/' + file, usecols=[ID] + [item], na_values=NA_VALUES,
                    dtype={**{ID: str}, **{item: str}}, encoding='latin1'
                )
            )
            yrs.append(re.findall(r"[0-5]{1}(?=.txt)", file)[0])
            data[i].set_index(ID, inplace=True)
        data = concat_data(data=data, columns=yrs, index=item)
        if clean:
            data = clean_data(dataFrame=data, columns=yrs)
        return data, yrs

    @staticmethod
    def process_data(item, dir, paths, clean=False):
        """
        item is the item of interest (str)
        dir is the directory where the files of data are found (str)
        paths is all the paths of data in the directory (list of str)
        clean determines whether or not incomplete data is removed
        returns a list of pandas dataFrames and a list of numerics to name the files
        """
        data = []
        yrs = []
        for i, file in enumerate(paths):
            data.append(
                pd.read_csv(
                    dir + '/' + file, usecols=[ID] + [item], na_values=NA_VALUES,
                    dtype={**{ID: str}, **{item: str}}, encoding=ENCODING
                )
            )
            yrs.append(re.findall(r"[0-9]{2}(?=.txt)", file)[0])
            data[i].set_index(ID, inplace=True)
        data = concat_data(dataFrame=data, columns=yrs, index=item)
        if clean:
            data = clean_data(dataFrame=data, columns=yrs)
        return data, yrs

    @staticmethod
    def process_data(item, dir, paths, clean=False):
        """
        item is the item of interest (str)
        dir is the directory where the files of data are found (str)
        paths is all the paths of data in the directory (list of str)
        _clean determines whether or not incomplete data is removed
        returns a list of pandas dataFrames and a list of numerics to name the files
        """
        data = []
        yrs = []
        for i, file in enumerate(paths):
            data.append(
                pd.read_csv(
                    dir + '/' + file, usecols=[ID] + [item], na_values=NA_VALUES,
                    dtype={**{ID: str}, **{item: str}}, encoding=ENCODING
                )
            )
            yrs.append(re.findall(r"[0-5]{1}(?=.txt)", file)[0])
            data[i].set_index(ID, inplace=True)
        data = concat_data(dataFrame=data, columns=yrs, index=item)
        if clean:
            data = clean_data(dataFrame=data, columns=yrs)
        return data, yrs

    @staticmethod
    def read_folder(dir, ext):
        """
        Collects all file paths with given extension (ext) inside a folder directory (dir)
        """
        paths = []
        for file in os.listdir(dir):
            if file[-4:] == ext:
                paths.append(file)
        return paths


# This object is creating a dictionary and calling the Text Parser to compare the database with the query
class RunSearch:
    def __init__(self, MASTERPATH, parentObject, year, state, parameterNames, parameterNumbers, query, listGenerator):
        self.parentObject, self.year, self.state, self.parameterNames, self.parameterNumbers, self.query = parentObject, year, state, parameterNames, parameterNumbers, query
        self.filename = str(state) + str(year[2:]) + '.txt'
        self.parameterNamesWithRespectToNumber = {}
        self.items, self.temporaryItemSearchList = [], []

        # A dictionary with keys as item numbers and values as item names
        self.parameterNamesWithRespectToNumber = listGenerator.getParameterNamesWithRespectToNumber()

        try:
            TextParser.TextParser(
                MASTERPATH, self, self.year, self.state, self.parameterNames, self.parameterNumbers,
                self.query, self.filename, self.parameterNamesWithRespectToNumber, listGenerator
            )
        except Exception as e:
            print(e)
            # print("File '" + MASTERPATH + '\BridgeDataQuery\Database\Information\Years\\' + self.year + '\\' + self.filename + "' Not Found")
            return

    def __init__(self, parent, year, state, parameterNames, parameterNumbers, query, listGenerator):
        self.parent = GUI
        self.parentObject, self.year, self.state, self.parameterNames, self.parameterNumbers, self.query = parentObject, year, state, parameterNames, parameterNumbers, query
        self.filename = str(state) + str(year[2:]) + '.txt'
        self.parameterNamesWithRespectToNumber = {}
        self.items, self.temporaryItemSearchList = [], []

        # A dictionary with keys as item numbers and values as item names
        self.parameterNamesWithRespectToNumber = listGenerator.getParameterNamesWithRespectToNumber()

        try:
            TextParser.TextParser(
                self.GUI.MASTERPATH, self, self.year, self.state, self.parameterNames,
                self.parameterNumbers, self.query, self.filename,
                self.parameterNamesWithRespectToNumber, listGenerator
            )
        except Exception as e:
            print(e)
            # print("File '" + MASTERPATH + '\BridgeDataQuery\Database\Information\Years\\' + self.year + '\\' + self.filename + "' Not Found")
            return


class Search:
    def __init__(self, parent, settings):
        self.parent = parent
        self.settings = settings
        self.years = [str(i) for i in range(1992, 2017)]
        self.filter = Filter.Filter(parent)
        self.path_constant = self.parent.util.MASTERPATH + '\BridgeDataQuery\Database\Years\\'
        self.states = self.parent.util.lists.state_code_state()
        self.entryList = self.parent.populator.entryList
        self.na_values, self.data = ['N'], []
        self.dtype = {item: str for item in self.entryList if item != 'YEAR'}
        for func in [self.set_years, self.set_states_codes, self.process_the_data]:
            func()

    def __init__(self, parent, settings):
        self.parent = parent
        self.settings = settings
        self.years = [str(i) for i in range(1992, 2017)]
        self.filter = Filter.Filter(parent)
        self.path_constant = self.parent.util.MASTERPATH + '\BridgeDataQuery\Database\Years\\'
        self.states = self.parent.util.lists.state_code_state()
        self.entryList = self.parent.populator.entryList
        self.na_values, self.data = ['N'], []
        self.dtype = {item: str for item in self.entryList if item != 'YEAR'}
        self.fileName = self.name_file(self.parent.filepath)
        # print(self.file_name)
        for func in [self.set_years, self.set_states_codes, self.process_the_data]:
            func()

    def __init__(self, GUI):
        t = datetime.datetime.now().second
        self.GUI = GUI
        self.starting_year, self.current_year = 1992, (datetime.datetime.now().year - 1)
        stateCodes, states, years_to_search = {}, [], []

        self.final_results = []  # THE RESULTS OF THE SEARCH WILL BE SAVED HERE FOR EACH YEAR, THE LIST IS CLEARED AFTER IT'S WRITTEN
        self.found, first = True, True  # THIS BOOLEAN BECOMES FALSE AS SOON AS ONE OF THE SEARCH VALUES DIFFERS FROM A BRIDGE'S ITEM VALUE AND THE BRIDGE IS DISCARDED
        self.initial_year_iteration, self.years_to_iterate = 0, 24
        self.years, self.states, settings = [], [], []

        # CHECKING REPORT PREFERENCES
        with open(
            self.GUI.util.MASTERPATH + "\BridgeDataQuery\\Utilities14\Preferences\Temporary_Report_Preferences.txt",
            "r"
        ) as report_preferences:
            self.settings = [True if line.strip() == 'True' else False for line in report_preferences]

        dataReducer = NBIData.DataReducer(self)

        # REDUCE YEARS AND STATES BY CHECKING WHETHER A YEAR WAS SPECIFIED
        years_to_search = dataReducer.ReduceYears()
        dataReducer.ReduceStates()
        self.list_of_states_to_search = self.GUI.util.list_generator.get_list_of_states_to_search()

        for year in years_to_search:
            self.setFinalResults(
                "##########################################" + str(year) + "##########################################"
            )
            if self.GUI.files == 'multiple':
                first = True
            for state in self.list_of_states_to_search:
                self.run_search(str(year), state.strip())
                WriteReport.WriteReport(self, year, first)
                first = False
                self.final_results.clear()

        years_to_search.clear()
        self.years.clear()

        print("report_generator0 has been generated.")

    def __init__(self, settings=None):
        self.settings = settings

    def __init__(self):
        pass

    def __init__(self):
        pass

    def __init__(self, GUI):
        t = datetime.datetime.now().second
        self.GUI = GUI
        self.starting_year, self.current_year = 1992, (datetime.datetime.now().year - 1)
        stateCodes, states, years_to_search = {}, [], []

        self.final_results = []  # THE RESULTS OF THE SEARCH WILL BE SAVED HERE FOR EACH YEAR, THE LIST IS CLEARED AFTER IT'S WRITTEN
        self.found, first = True, True  # THIS BOOLEAN BECOMES FALSE AS SOON AS ONE OF THE SEARCH VALUES DIFFERS FROM A BRIDGE'S ITEM VALUE AND THE BRIDGE IS DISCARDED
        self.initial_year_iteration, self.years_to_iterate = 0, 24
        self.years, self.states, settings = [], [], []

        # CHECKING REPORT PREFERENCES
        with open(
            self.GUI.util.MASTERPATH + "\BridgeDataQuery\\Utilities14\Preferences\Temporary_Report_Preferences.txt",
            "r"
        ) as report_preferences:
            self.settings = [True if line.strip() == 'True' else False for line in report_preferences]

        dataReducer = NBIData.DataReducer(self)

        # REDUCE YEARS AND STATES BY CHECKING WHETHER A YEAR WAS SPECIFIED
        years_to_search = dataReducer.ReduceYears()
        dataReducer.ReduceStates()
        self.list_of_states_to_search = self.GUI.util.lists.get_list_of_states_to_search()

        for year in years_to_search:
            self.setFinalResults(
                "##########################################" + str(year) + "##########################################"
            )
            if self.GUI.files == 'multiple':
                first = True
            for state in self.list_of_states_to_search:
                self.run_search(str(year), state.strip())
                WriteReport.WriteReport(self, year, first)
                first = False
                self.final_results.clear()

        years_to_search.clear()
        self.years.clear()

        print("report_generator0 has been generated.")

    def __init__(self, parent):
        self.years = [str(i) for i in range(1992, 2017)]
        self.parent = parent
        self.entryList = self.parent.populator.entryList
        self.na_values = ['N']
        self.data = []
        self.dtype = {item: str for item in self.entryList if item != 'YEAR'}
        for func in [self.set_years, self.set_states_codes, self.read_the_csv, self.set_the_index,
                     self.remove_nonmatches, self.rename_duplicates, self.export_results]:
            func()  # print(self.data[0]['COUNTY_CODE_003'])

    def process_the_data(self):
        for state in self.state_paths:
            for y_4, y_2 in zip(self.y_4, self.y_2):
                self.path = self.path_constant + y_4 + '\\' + state + y_2 + '.txt'
                self.read_the_csv(self.path)
                self.set_the_index(self.data[-1])
                self.data[-1] = self.remove_nonmatches(self.data[-1])
                self.rename_duplicates(self.data[-1])
                self.export_results(self.data[-1], state, y_4)

    def read_the_csv(self, path):
        with open(path, newline='') as file:
            header = next(csv.reader(file))
        self.data.append(
            pd.read_csv(
                path, usecols=[item for item in header if item != 'YEAR'], na_values=self.na_values,
                dtype=self.dtype, encoding='latin1'
            )
        )

    @staticmethod
    def set_the_index(dataFrame):
        dataFrame.set_index('STRUCTURE_NUMBER_008', inplace=True)

    def remove_nonmatches(self, dataFrame):
        self.dataFrame = dataFrame

        for column in [item for item in self.entryList if
                       item != 'YEAR' and item != 'STATE_CODE_001' and self.entryList[item].get() != '']:
            collection = []
            entry = pd.read_csv(StringIO(self.entryList[column].get()), sep=',')
            x = self.filter.inequality(entry, column)
            y = self.filter.range(entry, column)

            for i in (x, y):
                for r in i:
                    if len(r[0]) == 1:  # Inequalities
                        temp_df = pd.DataFrame(
                            self.dataFrame[r[0][0](self.dataFrame[column].astype(float), float(r[1][0]))]
                        )
                        collection.append(temp_df)
                    elif len(r[0]) == 2:  # Ranges
                        temp_df = pd.DataFrame(
                            self.dataFrame[r[0][0](
                                self.dataFrame[column].astype(float),
                                float(r[1].columns[0])
                            ) & r[0][1](
                                self.dataFrame[column].astype(float), float(r[1].columns[1])
                            )]
                        )
                        collection.append(temp_df)

            for i in entry:  # Standard Query
                if '<' not in i and '>' not in i and '-' not in i:
                    if type(i) == str:
                        i = list(i)

                    temp_df = (pd.DataFrame(self.dataFrame[self.dataFrame[column].isin(i)]))
                    collection.append(temp_df)

            self.dataFrame = pd.DataFrame(pd.concat([c for c in collection], join='inner', axis=0))

        return self.dataFrame

    def export_results(self, dataFrame, state, y_4):
        if not dataFrame.empty:
            if self.settings[0]:
                dataFrame.to_csv(path_or_buf=self.parent.filepath + '_' + str(state) + str(y_4) + '.txt')

            if self.settings[1]:
                writer = pd.ExcelWriter(self.parent.filepath, engine='xlsxwriter')
                dataFrame.to_excel(writer, sheet_name=str(state) + str(y_4) + '.txt')
                writer.save()

    def set_years(self):
        if self.entryList['YEAR'].get():
            self.years = StringIO(self.entryList['YEAR'].get())
            self.years = pd.read_csv(self.years, sep=',')

        self.y_4 = [str(i) for i in self.years]
        self.y_2 = [str(i)[2:] for i in self.years]

    def set_states_codes(self):
        if self.entryList['STATE_CODE_001'].get():
            self.state_codes = StringIO(self.entryList['STATE_CODE_001'].get())
            self.state_codes = pd.read_csv(self.state_codes, sep=',')

            print(self.state_codes)

        # self.s_3 = [str(i) for i in self.state_codes]  # self.s_2 = [str(i)[:1] for i in self.state_codes]

    def read_the_csv(self):
        # Reading the csv
        for y_4, y_2 in zip(self.y_4, self.y_2):
            self.path = path + y_4 + '\AK' + y_2 + '.txt'
            with open(self.path, newline='') as f:
                header = next(csv.reader(f))

            self.data.append(
                pd.read_csv(
                    self.path, usecols=[item for item in header if item != 'YEAR'], na_values=self.na_values,
                    dtype=self.dtype, encoding='latin1'
                )
            )

    def set_the_index(self):
        # Setting the index to be the structure number
        for i in range(len(self.data)):
            self.data[i].set_index('STRUCTURE_NUMBER_008', inplace=True)

    def remove_nonmatches(self):
        for i in range(len(self.data)):
            for column in [item for item in self.entryList if item != 'YEAR' and self.entryList[item].get() != '']:
                entry = StringIO(self.entryList[column].get())
                entry = pd.read_csv(entry, sep=',')
                self.data[i] = self.data[i][self.data[i][column].isin(entry)]
        """for i in range(len(self.data)):
            for column in [item for item in self.entryList if item!='YEAR' and self.entryList[item].get()!='']:
                self.data[i]=self.data[i][self.data[i][column]==(self.entryList[column].get())]"""

    def rename_duplicates(self):
        # Renaming duplicates
        for i in range(len(self.data)):
            while True in self.data[i].index.duplicated():
                self.data[i].index = self.data[i].index.where(
                    ~self.data[i].index.duplicated(),
                    self.data[i].index + '_dp'
                )

    def export_results(self):
        for i, year in zip(range(len(self.data)), self.years):
            self.data[i].to_csv(path_or_buf=self.parent.filepath + str(year) + '.txt')

    def setFinalResults(self, bridge):
        self.final_results.append(bridge)

    def run_search(self, year, state):
        self.filename = str(state) + str(year[2:]) + '.txt'
        self.parameterNamesWithRespectToNumber = {}
        self.items, self.temporaryItemSearchList = [], []

        # A dictionary with keys as item numbers and values as item names
        self.parameterNamesWithRespectToNumber = self.GUI.util.lists.getParameterNamesWithRespectToNumber()

        """try:"""
        self.text_parser(year, state)
        """except Exception as e:
            print(e)
            #print("File '" + MASTERPATH + '\BridgeDataQuery\Database\Information\Years\\' + self.year + '\\' + self.filename + "' Not Found")
            return"""

    def text_parser(self, year, state):
        self.year, self.state = year, state
        self.search = {}

        with open(
            self.GUI.util.MASTERPATH + '\BridgeDataQuery\Database\Years\\' + self.year + '\\' + self.filename,
            'r'
        ) as temporaryFile:

            line = temporaryFile.readline().strip()
            self.fileStructure = re.findall(r"\d{2}\w?", line)
            # List of items to be searched.
            reader = csv.reader(temporaryFile)

            for items_in_bridge in reader:
                match = self.executing_search(self.GUI.util.lists.getSearchList(), items_in_bridge)

                if match:

                    temporaryBridgeGeneratorObject = GenerateBridge.GenerateBridge(self, items_in_bridge)
                    self.setFinalResults(temporaryBridgeGeneratorObject.get_bridge())

                else:
                    pass

    def executing_search(self, search, items_in_bridge):
        self.search = search
        self.items_in_bridge = items_in_bridge
        match = True

        for i in range(self.fileStructure.__len__()):
            item = self.fileStructure[i].lstrip("0")
            self.Filter_Object = Filter.Filter(self, item)

            stripSearch = self.search[item].lstrip("0").strip()
            stripObject = self.items_in_bridge[i].lstrip("0").strip()

            if self.search[item] == '':  # The item was not specified so it is ignored
                pass

            else:

                if not ("," in list(self.search[item])):  # SEARCH IS A SINGLE VALUE
                    if item == '1':
                        stripSearch = stripSearch[:2]  # States are handled differently
                    match = self.Filter_Object.Run_Filter([stripSearch, stripObject], 0)

                else:  # SEARCH IS MULTIPLE VALUES
                    match = self.Filter_Object.Run_Filter([stripObject, self.parameterNamesWithRespectToNumber], 1)

                if not match:
                    break
        # print(self.interface7)

        return match

    def set_years(self):
        if self.entryList['YEAR'].get():
            self.years = pd.read_csv(StringIO(self.entryList['YEAR'].get()), sep=',')

        self.y_4 = [str(i) for i in self.years]
        self.y_2 = [str(i)[2:] for i in self.years]

    def set_states_codes(self):
        self.state_paths = []

        if self.entryList['STATE_CODE_001'].get():
            self.state_codes = pd.read_csv(StringIO(self.entryList['STATE_CODE_001'].get()), sep=',')
            for state in self.states:
                if state[:2] in self.state_codes or state in self.state_codes:
                    self.state_paths.append(self.states[state])
        else:
            self.state_paths = [self.states[state] for state in self.states]

    def process_the_data(self):
        for state in self.state_paths:
            for y_4, y_2 in zip(self.y_4, self.y_2):
                self.path = self.path_constant + y_4 + '\\' + state + y_2 + '.txt'
                self.read_the_csv(self.path)
                Search.set_the_index(self.data[-1])
                self.data[-1] = self.remove_nonmatches(self.data[-1])
                Search.rename_duplicates(self.data[-1])
                Search.export_results(self.data[-1], state, y_4, self.parent.filepath, self.fileName, self.settings)

    def read_the_csv(self, path):
        with open(path, newline='') as file: header = next(csv.reader(file))
        self.data.append(
            pd.read_csv(
                path, usecols=[item for item in header if item != 'YEAR'], na_values=self.na_values,
                dtype=self.dtype, encoding='latin1'
            )
        )

    @staticmethod
    def set_the_index(dataFrame):
        dataFrame.set_index('STRUCTURE_NUMBER_008', inplace=True)

    def remove_nonmatches(self, dataFrame):
        self.dataFrame = dataFrame

        if self.settings[2]:  # Union
            pass

        elif self.settings[3]:  # Intersection
            for column in [item for item in self.entryList if
                           item != 'YEAR' and item != 'STATE_CODE_001' and self.entryList[item].get() != '']:
                collection = []
                entry = pd.read_csv(StringIO(self.entryList[column].get()), sep=',')
                x = self.filter.inequality(entry, column)
                y = self.filter.range(entry, column)

                for i in entry:
                    for k in (x, y):
                        for r in k:
                            if len(r[0]) == 1:  # Inequalities
                                temp_df = pd.DataFrame(
                                    self.dataFrame[r[0][0](self.dataFrame[column].astype(float), float(r[1][0]))]
                                )
                                collection.append(temp_df)
                            elif len(r[0]) == 2:  # Ranges
                                temp_df = pd.DataFrame(
                                    self.dataFrame[r[0][0](
                                        self.dataFrame[column].astype(float),
                                        float(r[1].columns[0])
                                    ) & r[0][1](
                                        self.dataFrame[column].astype(float), float(r[1].columns[1])
                                    )]
                                )
                                collection.append(temp_df)

                    if type(i) == str:
                        i = [i]

                    temp_df = (pd.DataFrame(self.dataFrame[self.dataFrame[column].isin(i)]))
                    collection.append(temp_df)

                    self.dataFrame = pd.DataFrame(pd.concat([c for c in collection], join='inner', axis=0))

            return self.dataFrame

    @staticmethod
    def rename_duplicates(dataFrame):
        while True in dataFrame.index.duplicated():
            dataFrame.index = dataFrame.index.where(~dataFrame.index.duplicated(), dataFrame.index + '_dp')

    @staticmethod
    def export_results(dataFrame, state, y_4, path, file, settings):
        if not dataFrame.empty:
            if settings[0]:
                dataFrame.to_csv(path_or_buf=path + file + '_' + str(state) + str(y_4) + '.txt')

            if settings[1]:
                try:
                    book = load_workbook(path + file + '.xlsx')
                    writer = pd.ExcelWriter(path + file + '.xlsx', engine='openpyxl')
                    writer.book = book
                    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                    dataFrame.to_excel(writer, sheet_name=str(state) + str(y_4))
                    writer.save()
                except:
                    writer = pd.ExcelWriter(path + file + '.xlsx', engine='xlsxwriter')
                    dataFrame.to_excel(writer, sheet_name=str(state) + str(y_4))
                    writer.save()
        else:
            print("Empty Dataframe.")

    def setFinalResults(self, bridge):
        self.final_results.append(bridge)

    def run_search(self, year, state):
        self.filename = str(state) + str(year[2:]) + '.txt'
        self.parameterNamesWithRespectToNumber = {}
        self.items, self.temporaryItemSearchList = [], []

        # A dictionary with keys as item numbers and values as item names
        self.parameterNamesWithRespectToNumber = self.GUI.util.list_generator.getParameterNamesWithRespectToNumber()

        """try:"""
        self.text_parser(year, state)
        """except Exception as e:
            print(e)
            #print("File '" + MASTERPATH + '\BridgeDataQuery\Database\Information\Years\\' + self.year + '\\' + self.filename + "' Not Found")
            return"""

    def text_parser(self, year, state):
        self.year, self.state = year, state
        self.search = {}

        with open(
            self.GUI.util.MASTERPATH + '\BridgeDataQuery\Database\Information\Years\\' + self.year + '\\' + self.filename,
            'r'
        ) as temporaryFile:

            line = temporaryFile.readline().strip()
            self.fileStructure = re.findall(r"\d{2}\w?", line)
            # List of items to be searched.
            reader = csv.reader(temporaryFile)

            for items_in_bridge in reader:
                match = self.executing_search(self.GUI.util.list_generator.getSearchList(), items_in_bridge)

                if match:

                    temporaryBridgeGeneratorObject = GenerateBridge.GenerateBridge(self, items_in_bridge)
                    self.setFinalResults(temporaryBridgeGeneratorObject.get_bridge())

                else:
                    pass

    def __del__(self):
        pass

    def Search(self, fileStructure, query, search, parameterNamesWithRespectToNumber, items_in_bridge):
        self.fileStructure = fileStructure
        self.query = query
        self.search = search
        self.parameterNamesWithRespectToNumber = parameterNamesWithRespectToNumber
        self.items_in_bridge = items_in_bridge
        match = True

        for i in range(self.fileStructure.__len__()):
            item = self.fileStructure[i].lstrip("0")
            self.Filter_Object = Filter.Filter(self.query, self.search, item)

            stripSearch = self.search[item].lstrip("0").strip()
            stripObject = self.items_in_bridge[i].lstrip("0").strip()

            if self.search[item] == '':  # The item was not specified so it is ignored
                pass

            else:

                if not ("," in list(self.search[item])):  # SEARCH IS A SINGLE VALUE
                    if item == '1':
                        stripSearch = stripSearch[:2]  # States are handled differently
                    match = self.Filter_Object.Run_Filter([stripSearch, stripObject], 0)

                else:  # SEARCH IS MULTIPLE VALUES
                    match = self.Filter_Object.Run_Filter([stripObject, self.parameterNamesWithRespectToNumber], 1)

                if not match:
                    break
        # print(self.search)

        return match

    def Search(self, fileStructure, query, search, parameterNamesWithRespectToNumber, items_in_bridge):
        self.fileStructure = fileStructure
        self.query = query
        self.search = search
        self.parameterNamesWithRespectToNumber = parameterNamesWithRespectToNumber
        self.items_in_bridge = items_in_bridge

        for item in self.fileStructure:
            stripSearch = self.search[item.lstrip("0")].lstrip("0").strip()
            stripObject = self.temporaryBridgeGeneratorObject.getBridgeItems()[item.lstrip("0")].lstrip("0").strip()

            if self.search[item.lstrip("0")] == '':
                pass

            elif item.lstrip("0") != '1':
                if not ("," in list(self.search[item.lstrip("0")])):  # SEARCH IS A SINGLE VALUE
                    if self.search[item.lstrip("0")] == self.temporaryBridgeGeneratorObject.getBridgeItems()[
                        item.lstrip("0")].strip() or stripSearch == stripObject:
                        pass
                    else:
                        return False

                    return True

                else:  # SEARCH IS MULTIPLE VALUES
                    self.temporaryItemSearchList = re.findall(
                        r".?\w+.?", self.query[
                            self.parameterNamesWithRespectToNumber[item.lstrip("0")]]
                    )

                    for value in self.temporaryItemSearchList:
                        stripValue = value.lstrip("0").rstrip(",").strip()

                        if stripValue != stripObject:
                            pass
                        else:
                            return True

                    return False

            elif item.lstrip("0") == '1':
                if not ("," in list(self.search[item.lstrip("0")])):  # SEARCH IS A SINGLE STATE
                    if self.search[item.lstrip("0")][:2] == self.temporaryBridgeGeneratorObject.getBridgeItems()[
                        item.lstrip("0")].strip():
                        pass

                    else:
                        return False

                    return True

                else:  # SEARCH IS MULTIPLE STATES

                    self.temporaryItemSearchList = re.findall(
                        r".?\w+.?", self.query[
                            self.parameterNamesWithRespectToNumber[item.lstrip("0")]]
                    )

                    for value in self.temporaryItemSearchList:
                        stripValue = value.rstrip(",").strip()[:2].lstrip("0")

                        if stripValue != stripObject:
                            pass
                        else:
                            return True

                    return False


class SearchMenu(SubMenu):
    def run(self):
        ReportMenu(self, self.top, "search").populate()

    def populate(self):
        if True:
            pass
        else:
            self.frame = VerticalScrolledFrame(self.top)
        self.frame.grid(row=0, column=0, columnspan=3, sticky="nsew")
        self.entryList = self.setSearchEntryLists(self.frame.interior)
        self.populate_search_field()
        self.util.separator(container=self.top, orient='h', coords=(2, 0), columnspan=3, sticky='we')
        self.util.button(
            container=self.top, text=SM_BT, commands=[self.run, self.clear], cursor=HAND,
            coords=((3, 2), (3, 0)), sticky=['se', 'sw']
        )
        self.util.drop_down_menu(
            parent=self.top, menus=SM_DM, tearoff=[0], labels=SM_DL,
            commands=[[self.load, self.save, self.top.destroy]]
        )

    def populate_search_field(self):
        for i, line in enumerate(self.entryList):
            self.util.raw_label(container=self.frame.interior, text=line, coords=(i + 3, 0), sticky='w')
            self.entryList[FEATURES[i]].grid(row=i + 3, column=2, columnspan=1, sticky='e')
            self.entryList[FEATURES[i]].insert(0, '')

    def clear(self):
        for i in range(self.entryList.__len__()):
            self.entryList[FEATURES[i + 0]].delete(0, self.entryList[FEATURES[i + 0]].get().__len__())

    def save(self):
        path = self.util.save_file(fileExt=QRY_FILES)
        try:
            if path[-4:] != QRY_EXT:
                path = path + QRY_EXT
            with open(path, "w") as saveFile:
                for i in range(self.util.number_of_items):
                    saveFile.write(self.entryList[FEATURES[i]].get() + "\n")
        except:
            return

    def load(self):
        try:
            path = self.util.open_file(fileExt=QRY_FILES)
            with open(path, "r") as loadFile:
                for i in range(self.entryList.__len__()):
                    self.entryList[FEATURES[i]].delete(0, self.entryList[FEATURES[i]].get().__len__())
                for i, line in enumerate(loadFile):
                    if line.strip():
                        self.entryList[FEATURES[i]].insert(i, line.strip())
        except:
            return

    def setSearchEntryLists(self, container):
        entryList = {}
        with open(ITEM_NAMES, 'r') as f:
            for line in f:
                entryList[line.strip()] = self.util.raw_entry(container)
        return entryList


class TextParser(InitiateSearch):
    def __init__(self, parentObject, year, state, parameterNames, parameterNumbers, query):
        self.parentObject, self.year, self.state, self.parameterNames, self.parameterNumbers, self.query = parentObject, year, state, parameterNames, parameterNumbers, query
        self.filename = str(state) + str(year[2:]) + '.txt'
        self.search, self.parameterNamesWithRespectToNumber = {}, {}
        self.items, self.temporaryItemSearchList = [], []
        self.parentObject.setFinalResults(
            self.year + " - " + self.state + "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "
        )

        # A dictionary with keys as item numbers and values as item names
        omega = 0
        for number in self.parameterNumbers:
            self.parameterNamesWithRespectToNumber[parameterNumbers[number]] = self.parameterNames[omega]
            omega += 1

        try:
            with open(
                MASTERPATH + '\BridgeDataQuery\Database\Information\Years\\' + self.year + '\\' + self.filename,
                'r'
            ) as temporaryFile:

                k = 0
                for line in temporaryFile:
                    if k == 0:
                        self.fileStructure = re.findall(r"\d{2}\w?", line)
                    k += 1
                    break

                # This for loop is generating the list of items to be searched.
                for item in parameterNumbers:
                    if parameterNames[item] == '(Reserved)':
                        self.search[parameterNumbers[item]] = ''
                    else:
                        self.search[parameterNumbers[item]] = self.query[parameterNames[item]]

                reader = csv.reader(temporaryFile)

                for row in reader:
                    self.items = (row)
                    temporaryBridgeGeneratorObject = GenerateBridge(self.query, self.fileStructure, self.items)
                    match = True
                    item_number = 0
                    for item in self.fileStructure:
                        stripSearch = self.search[item.lstrip("0")].lstrip("0")
                        stripSearch = stripSearch.strip()
                        stripObject = temporaryBridgeGeneratorObject.getBridgeItems()[item.lstrip("0")].lstrip("0")
                        stripObject = stripObject.strip()
                        if self.search[item.lstrip("0")] == '':
                            pass
                        elif item.lstrip("0") != '1':
                            if not ("," in list(self.search[item.lstrip("0")])):  # SEARCH IS A SINGLE VALUE
                                if self.search[item.lstrip("0")] == temporaryBridgeGeneratorObject.getBridgeItems()[
                                    item.lstrip("0")].strip() or stripSearch == stripObject:
                                    pass
                                else:
                                    match = False
                                    break
                            else:  # SEARCH IS MULTIPLE VALUES
                                match = False
                                self.temporaryItemSearchList = re.findall(
                                    r".?\w+.?", self.query[
                                        self.parameterNamesWithRespectToNumber[item.lstrip("0")]]
                                )

                                for value in self.temporaryItemSearchList:
                                    stripValue = value.lstrip("0")
                                    stripValue = stripValue.rstrip(",")
                                    stripValue = stripValue.strip()
                                    if stripValue != stripObject:
                                        pass
                                    else:
                                        match = True
                                        break

                        elif item.lstrip("0") == '1':
                            if not ("," in list(self.search[item.lstrip("0")])):  # SEARCH IS A SINGLE STATE
                                if self.search[item.lstrip("0")][:2] == temporaryBridgeGeneratorObject.getBridgeItems()[
                                    item.lstrip("0")].strip():
                                    pass
                                else:
                                    match = False
                                    break
                            else:  # SEARCH IS MULTIPLE STATES
                                match = False
                                self.temporaryItemSearchList = re.findall(
                                    r".?\w+.?", self.query[
                                        self.parameterNamesWithRespectToNumber[item.lstrip("0")]]
                                )
                                for value in self.temporaryItemSearchList:
                                    stripValue = value.rstrip(",")
                                    stripValue = stripValue.strip()[:2]
                                    stripValue = stripValue.lstrip("0")
                                    print("Value: " + stripValue)
                                    print("Object: " + stripObject)
                                    if stripValue != stripObject:
                                        print("FAIL")
                                        pass
                                    else:
                                        print("SUCCESS")
                                        match = True
                                        break

                        item_number += 1

                    if match:
                        self.parentObject.setFinalResults(temporaryBridgeGeneratorObject.getBridgeItems())
                    else:
                        pass

                    k += 1
        except:
            print(
                "File '" + MASTERPATH + '\BridgeDataQuery\Database\Information\Years\\' + self.year + '\\' + self.filename + "' Not Found"
            )
            return

    def __init__(self, parentObject, year, state, parameterNames, parameterNumbers, query):
        self.parentObject, self.year, self.state, self.parameterNames, self.parameterNumbers, self.query = parentObject, year, state, parameterNames, parameterNumbers, query
        self.filename = str(state) + str(year[2:]) + '.txt'
        self.search, self.parameterNamesWithRespectToNumber = {}, {}
        self.items, self.temporaryItemSearchList = [], []
        self.parentObject.setFinalResults(
            self.year + " - " + self.state + "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "
        )

        # A dictionary with keys as item numbers and values as item names
        omega = 0
        for number in self.parameterNumbers:
            self.parameterNamesWithRespectToNumber[parameterNumbers[number]] = self.parameterNames[omega]
            omega += 1

        try:
            with open(
                MASTERPATH + '\BridgeDataQuery\Database\Information\Years\\' + self.year + '\\' + self.filename,
                'r'
            ) as temporaryFile:

                k = 0
                for line in temporaryFile:
                    if k == 0:
                        self.fileStructure = re.findall(r"\d{3}\w?", line)
                    k += 1
                    break

                # This for loop is generating the list of items to be searched.
                for item in parameterNumbers:
                    if parameterNames[item] == '(Reserved)':
                        self.search[parameterNumbers[item]] = ''
                    else:
                        self.search[parameterNumbers[item]] = self.query[parameterNames[item]]

                reader = csv.reader(temporaryFile)

                for row in reader:
                    self.items = (row)
                    temporaryBridgeGeneratorObject = GenerateBridge(self.query, self.fileStructure, self.items)
                    match = True
                    item_number = 0
                    for item in self.fileStructure:
                        stripSearch = self.search[item.lstrip("0")].lstrip("0")
                        stripSearch = stripSearch.strip()
                        stripObject = temporaryBridgeGeneratorObject.getBridgeItems()[item.lstrip("0")].lstrip("0")
                        stripObject = stripObject.strip()
                        if self.search[item.lstrip("0")] == '':
                            pass
                        elif item.lstrip("0") != '1':
                            if not ("," in list(self.search[item.lstrip("0")])):  # SEARCH IS A SINGLE VALUE
                                if self.search[item.lstrip("0")] == temporaryBridgeGeneratorObject.getBridgeItems()[
                                    item.lstrip("0")].strip() or stripSearch == stripObject:
                                    pass
                                else:
                                    match = False
                                    break
                            else:  # SEARCH IS MULTIPLE VALUES
                                match = False
                                self.temporaryItemSearchList = re.findall(
                                    r".?\w+.?", self.query[
                                        self.parameterNamesWithRespectToNumber[item.lstrip("0")]]
                                )

                                for value in self.temporaryItemSearchList:
                                    stripValue = value.lstrip("0")
                                    stripValue = stripValue.rstrip(",")
                                    stripValue = stripValue.strip()
                                    if stripValue != stripObject:
                                        pass
                                    else:
                                        match = True
                                        break

                        elif item.lstrip("0") == '1':
                            if not ("," in list(self.search[item.lstrip("0")])):  # SEARCH IS A SINGLE STATE
                                if self.search[item.lstrip("0")][:2] == temporaryBridgeGeneratorObject.getBridgeItems()[
                                    item.lstrip("0")].strip():
                                    pass
                                else:
                                    match = False
                                    break
                            else:  # SEARCH IS MULTIPLE STATES
                                match = False
                                self.temporaryItemSearchList = re.findall(
                                    r".?\w+.?", self.query[
                                        self.parameterNamesWithRespectToNumber[item.lstrip("0")]]
                                )
                                for value in self.temporaryItemSearchList:
                                    stripValue = value.rstrip(",")
                                    stripValue = stripValue.strip()[:2]
                                    stripValue = stripValue.lstrip("0")
                                    print("Value: " + stripValue)
                                    print("Object: " + stripObject)
                                    if stripValue != stripObject:
                                        print("FAIL")
                                        pass
                                    else:
                                        print("SUCCESS")
                                        match = True
                                        break

                        item_number += 1

                    if match:
                        self.parentObject.setFinalResults(temporaryBridgeGeneratorObject.getBridgeItems())
                    else:
                        pass

                    k += 1
        except:
            print(
                "File '" + MASTERPATH + '\BridgeDataQuery\Database\Information\Years\\' + self.year + '\\' + self.filename + "' Not Found"
            )
            return


class TextParser:
    def __init__(
        self, MASTERPATH, parentObject, year, state, parameterNames, parameterNumbers, query, filename,
        parameterNamesWithRespectToNumber, listGenerator
    ):
        self.parentObject, self.year, self.state, self.parameterNames, self.parameterNumbers, self.query, self.filename, self.parameterNamesWithRespectToNumber = parentObject, year, state, parameterNames, parameterNumbers, query, filename, parameterNamesWithRespectToNumber
        self.search = {}

        with open(
            MASTERPATH + '\BridgeDataQuery\Database\Information\Years\\' + self.year + '\\' + self.filename,
            'r'
        ) as temporaryFile:

            line = temporaryFile.readline().strip()
            self.fileStructure = re.findall(r"\d{2}\w?", line)
            # List of items to be searched.
            reader = csv.reader(temporaryFile)

            for items_in_bridge in reader:
                search_object = Search.Search()
                match = search_object.Search(
                    self.fileStructure, self.query, listGenerator.getSearchList(),
                    self.parameterNamesWithRespectToNumber, items_in_bridge
                )

                if match:

                    temporaryBridgeGeneratorObject = GenerateBridge.GenerateBridge(
                        self.query, self.fileStructure,
                        items_in_bridge
                    )
                    self.parentObject.parentObject.setFinalResults(temporaryBridgeGeneratorObject.getBridgeItems())

                else:
                    pass

    def __init__(
        self, MASTERPATH, parentObject, year, state, parameterNames, parameterNumbers, query, filename,
        parameterNamesWithRespectToNumber, listGenerator
    ):
        self.parentObject, self.year, self.state, self.parameterNames, self.parameterNumbers, self.query, self.filename, self.parameterNamesWithRespectToNumber \
            = parentObject, year, state, parameterNames, parameterNumbers, query, filename, parameterNamesWithRespectToNumber
        self.search = {}

        with open(
            MASTERPATH + '\BridgeDataQuery\Database\Information\Years\\' + self.year + '\\' + self.filename,
            'r'
        ) as temporaryFile:

            line = temporaryFile.readline().strip()
            self.fileStructure = re.findall(r"\d{3}\w?", line)
            # List of items to be searched.
            reader = csv.reader(temporaryFile)

            for items_in_bridge in reader:
                search_object = Search.Search()
                match = search_object.Search(
                    self.fileStructure, self.query, listGenerator.getSearchList(),
                    self.parameterNamesWithRespectToNumber, items_in_bridge
                )

                if match:

                    temporaryBridgeGeneratorObject = GenerateBridge.GenerateBridge(
                        self.query, self.fileStructure,
                        items_in_bridge
                    )
                    self.parentObject.parentObject.setFinalResults(temporaryBridgeGeneratorObject.getBridgeItems())

                else:
                    pass


# Generates a new, unique, empty list of values. The sum of these values is 1.
class VectorGenerator:
    def __init__(self):
        pass

    def get_New_List_Vector(self):
        self.new_probability_vector = []  # When the object is instantiated, a new, unique, empty list is created
        return self.new_probability_vector  # Allows external access to an object's unique list

    def get_New_Dict_Vector(self):
        self.new_probability_vector = {}  # When the object is instantiated, a new, unique, empty list is created
        return self.new_probability_vector  # Allows external access to an object's unique list


class WriteReport:
    def __init__(self, year, MASTERPATH, results, first, files, filepath, settings):
        self.year, self.results, self.first, self.files, self.filepath = year, results, first, files, filepath
        if first:
            mode = "w"
        else:
            mode = "a"

        csv_export = settings[0]
        machine_export = settings[1]
        single_file = settings[2]
        multiple_files = settings[3]

        # CSV EXPORT
        if csv_export:
            csv_version_bridge = []

            with open(self.filepath + "-csv_version.txt", mode) as report:
                for bridge in self.results:  # Iterating through the bridge dictionaries in the results list
                    if '#########' not in bridge:
                        for value in bridge:  # Iterating through the values in the bridge dictionary
                            csv_version_bridge.append(bridge[value])

                        for item in range(csv_version_bridge.__len__()):
                            report.write((str(csv_version_bridge[item])) + ",")

                        report.write("\n")
                        csv_version_bridge[:] = []

                    else:  # Adding the indicator of a new state/year
                        bridge = re.findall(r'\w+', str(bridge))  # Converting the indicator into a list of components
                        csv_version_bridge.append(
                            str(bridge[0]).strip("#")
                        )  # Adding the result to the the list of csv results

        # MACHINE LEARNING INPUT FILE EXPORT
        if machine_export:
            if multiple_files:
                with open(self.filepath + "-" + str(self.year)[2:] + ".txt", mode) as report:
                    for bridge in self.results:
                        report.write(str(bridge) + "\n")

            if single_file:
                with open(self.filepath + ".txt", mode) as report:
                    for bridge in self.results:
                        report.write(str(bridge) + "\n")

    def __init__(self, Search, year, first, settings, csvExport, machineExport, singleFile, multipleFiles):
        self.Search = Search
        self.year, self.first = year, first
        if first:
            mode = "w"
        else:
            mode = "a"
        # csvExport = settings[0]
        # machineExport = settings[1]
        # singleFile = settings[1]
        # multipleFiles = settings[2]
        # CSV EXPORT
        if csvExport:
            csvVersionBridge = []
            with open(self.Search.GUI.filepath + "-csv_version.txt", mode) as report:
                for bridge in self.Search.final_results:  # Iterating through the bridge dictionaries in the results list
                    if '#########' not in bridge:
                        csvVersionBridge = [bridge[value] for value in bridge]
                        for item in range(csvVersionBridge.__len__()):
                            report.write((str(csvVersionBridge[item])) + ",")
                        report.write("\n")
                        csvVersionBridge[:] = []
                    else:  # Adding the indicator of a new state/year
                        bridge = re.findall(r'\w+', str(bridge))  # Converting the indicator into a list of components
                        csvVersionBridge.append(
                            str(bridge[0]).strip("#")
                        )  # Adding the result to the the list of csv results
        # MACHINE LEARNING INPUT FILE EXPORT
        if machineExport:
            if multipleFiles:
                with open(self.Search.GUI.filepath + "-" + str(self.year)[2:] + ".txt", mode) as report:
                    for bridge in self.Search.final_results:
                        report.write(str(bridge) + "\n")
            if singleFile:
                with open(self.Search.GUI.filepath + ".txt", mode) as report:
                    for bridge in self.Search.final_results:
                        report.write(str(bridge) + "\n")

    def __init__(self, year, MASTERPATH, results, first):

        current_time = ''  # str(datetime.datetime.now())[17:19]
        self.year = year
        self.results = results
        self.first = first
        if first:
            with open(MASTERPATH + "\\BridgeDataQuery\Reports\\report" + str(self.year)[2:] + ".txt", "w") as report:
                for bridge in self.results:
                    report.write(str(bridge) + "\n")
        else:
            with open(MASTERPATH + "\\BridgeDataQuery\Reports\\report" + str(self.year)[2:] + ".txt", "a") as report:
                for bridge in self.results:
                    report.write(str(bridge) + "\n")


df1 = pd.DataFrame(
    (1 / 18000) * np.array(np.exp(np.linspace(0, 10, 100))),
    # 20*np.array(np.sin(np.linspace(0, 1 * np.pi, 100)))+np.array([np.random.randint(0,10) for i in range(0, 100)]),
    # 3*np.array([i for i in range(100)])+20
    # np.array([[(x**1)/(x-1)] for x in range(1, 102)])
    # +1/dev * np.random.normal(0, 1, size=(100, 1)),
    columns=['var'], index=[i for i in range(1, 101)]
)

df3 = pd.DataFrame(  # np.array(np.log(np.linspace(1, 100, 100))),
    20 * np.array(np.sin(np.linspace(0, 2 * np.pi, 100))) + 6 * np.array(
        [np.random.randint(0, 10) for i in range(0, 100)]
    ),  # 3*np.array([i for i in range(100)])+20
    # np.array([[x**1] for x in range(0, 100)])
    # +dev * np.random.normal(0, 1, size=(100, 1)),
    # 1 * np.random.normal(0, .01, size=(100, 1)) +
    # 1 * np.random.normal(1000, .01, size=(100, 1)),

    columns=['var3'], index=[i for i in range(1, 101)]
)

df = df1  # pd.concat([df1, df2, df3], axis=1)

df2 = pd.DataFrame(  # np.array(np.log(np.linspace(1, 100, 100))),
    20 * np.array(np.sin(np.linspace(0, 2 * np.pi, 100))) + 6 * np.array(
        [np.random.randint(0, 10) for i in range(0, 100)]
    ),  # 3*np.array([i for i in range(100)])+20
    # np.array([[x**1] for x in range(0, 100)])
    # +dev * np.random.normal(0, 1, size=(100, 1)),
    columns=['var2'], index=[i for i in range(1, 101)]
)

plot_violin(df2['var2'])
plot_box(df2['var2'])
plot_violin(df2['var2'])
plot_box(df2['var2'])
plot_scatter(df2['var2'])
plot_histogram(df2['var2'])
plot_scatter(df2['var2'])
plot_histogram(df2['var2'])
