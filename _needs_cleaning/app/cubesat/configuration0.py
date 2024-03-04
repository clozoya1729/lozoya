import json
import os

import pandas as pd

name = 'IDA ML'
root = r'E:\github2\test\ida'


class MLDude:
    def __init__(self):
        self.basepath = os.path.join(os.getcwd(), '10hz')
        self.dirs = os.listdir(self.basepath)
        self.files = []
        self.columnNames = []
        self.directory()
        self.file()

    def directory(self):
        for dir in self.dirs:
            fullpath = os.path.join(self.basepath, dir)
            files = os.listdir(fullpath)
            for file in files:
                r = file.split('.')
                if len(r) > 1:
                    if r[1] == 'csv':
                        filepath = os.path.join(fullpath, file)
                        self.files.append(filepath)
        print(self.files)
        n = 0

    def file(self):
        for i, file in enumerate(self.files):
            df = pd.read_csv(file, nrows=2)
            col = ' Height (AGL)'
            if col in df.columns:
                try:
                    df = pd.read_csv(file)
                    pNan = df[col].isna().sum() / len(df)
                    if pNan < 0.1:
                        df[col].plot()
                        break
                    n += 1
                except Exception as e:
                    print(e)


class Scenario:
    def __init__(self, fullpath):
        '''
        Scenario object contains the name of a scenario and the path to the corresponding RAID file.
        The path is passed through the fullpath argument.
        The RAID file is read and stored in the _json variable.
        The _json variable is a list.
        Each  item in the list is an entry in the database contained in the RAID file.
        An entry may be a ROUTE or ATTACK_PLAN type and contains several variables.
        :param fullpath: A string of text containing the full path to a RAID file.
        '''
        self.fullpath = fullpath
        self.name = os.path.basename(os.path.normpath(fullpath))
        self.json = self._set_json()

    def _set_json(self):
        '''
        Load the data into the _json variable
        :return:
        '''
        with open(self.fullpath, 'r') as f:
            _j = json.load(f)
        return _j


def get_object_by_key(key, value, scenario):
    '''
    Return an object from a scenario if the key and value match the provided key and value.
    :param key:
    :param value:
    :param scenario:
    :return:
    '''
    for object in scenario.json:
        if key in object:
            if object[key] == value:
                return object
