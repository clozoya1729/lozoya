import os
import uuid

import pandas as pd
import sympy as sym
import unicodedata

import lozoya.text

extension = {
    'csv': ['csv', 'txt'], 'excel': ['xls', 'xlsx', 'xlsm', 'xltx', 'xltm'], 'hdf5': ['hdf5'], 'mat': ['mat'],
    'wav': ['wav'], 'txt': ['txt']
}
SEPARATORS = (',', '\t', '|', '/', '\\', '.', ';', ':', '-', ' ')


def smart(string):
    string = string.replace(' ', '')
    string = string.replace('\t', '')
    string = string.replace('[', '')
    string = string.replace(']', '')
    string = string.replace('"', '')
    string = string.replace("'", '')
    string = string.replace('\n', '')
    return string


def is_number(string):
    """
    Determines whether a value is numerical or not.
    E.g., if '7.12' is passed, True will be returned.
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


def generate_job_id():
    unique = False
    while not unique:
        jobID = str(uuid.uuid4())
        unique = check_id_uniqueness(jobID)
    return jobID


def check_id_uniqueness(jobRoot, jobID):
    subDirs = [x[0] for i, x in enumerate(os.walk(jobRoot)) if i > 0]
    if jobID in subDirs:
        return False
    return True


def ik(field, n):
    new = ''
    j = 0
    for i, a in enumerate(lozoya.text.str_to_list(field)):
        _ = a.strip(',')
        if (i != n - 1):
            if j == 0:
                new += _
                j = 1
            else:
                new += ',{}'.format(_)
    return new


def ic(field, n):
    old = ''
    new = ''
    j = 0
    for i, a in enumerate(lozoya.text.str_to_list(field)):
        _ = a.strip(',')
        if (i != n - 1):
            if j == 0:
                old += _
                if n != 1:
                    new += _
                else:
                    new += str(int(_) - 1)
                j = 1
            else:
                old += ',{}'.format(_)
                if i > n - 1:
                    new += ',{}'.format(int(_) - 1)
                else:
                    new += ',{}'.format(_)
    return old, new


def enum_dict(keys, value):
    """
    Creates a dictionary containing identical values for each key.
    keys: list
    value: anything
    return: dictionary
    """
    return {i: value for i in range(keys)}


def make_symbol(arg, symSim=False):
    '''
    args: list or dictionary to search
    i: int, position to search in args
    return: value in args[i] if it exists, else None
    '''
    try:
        if symSim:
            return sym.N(arg)
        else:
            if arg - int(arg) == 0:
                return int(arg)
            return arg
    except:
        pass
    try:
        return sym.Symbol(arg)
    except:
        return None


def clean_file_contents(contents):
    return str(contents).replace('\n', '').replace('\r', '')


def read_csv(path):
    return pd.read_csv(path)
