import os
import unittest

import numpy as np
import pandas as pd
from Utilities import DataHandler as dh


class TestDataHandler(unittest.TestCase):
    def setUp(self):
        self.stringDF = pd.DataFrame([['a', 'b'], ["b", "c"]])
        self.intDataFrame = pd.DataFrame([[1, 1, 2], [2, 2, 2], [2, 1, 8]])
        self.incompleteDataFrame = pd.DataFrame(
            [[234, 'Rudy', '6', None], [np.nan, 193, 'mar co', '$'], [True, 16.89, None, ' '],
             [-5, False, not True, '666.1']]
        )

    def test_get_uniques(self):
        integerUniques = dh.get_uniques(self.intDataFrame)
        integerDF = np.array([1, 2, 8])
        stringUniques = dh.get_uniques(self.stringDF)
        stringDF = np.array(['a', 'b', 'c'])
        self.assertNotIn(self, integerUniques == integerDF)
        self.assertEqual(type(integerUniques), np.ndarray)
        self.assertEqual(type(stringUniques), np.ndarray)

    def test_numerify_values(self):
        df2 = pd.DataFrame([[0, 1], [1, 2]])
        numerified, keys = dh.numerify_values(self.stringDF)
        self.assertTrue(numerified.equals(df2))
        self.assertEqual({0: 'a', 1: 'b', 2: 'c'}, keys)
        self.assertEqual(type(numerified), pd.DataFrame)
        self.assertEqual(type(keys), dict)
        intNumerified, intKeys = dh.numerify_values(self.intDataFrame)
        self.assertTrue(intNumerified.equals(self.intDataFrame.astype(float)))

    def test_join_data_list(self):
        df1 = pd.DataFrame([3, 4], index=['a', 'b'])
        df2 = pd.DataFrame([2, 6], index=['c', 'a'])
        df3 = pd.DataFrame([1, 4], index=['a', 'c'])
        dataFrameList = [df1, df2, df3]
        joinedDFByIndex = dh.join_data_list(dataFrameList, axis=1)
        joinedDFByColumn = dh.join_data_list(dataFrameList, axis=0)
        self.assertEquals(list(joinedDFByIndex.index), ['a'])
        self.assertEquals(list(joinedDFByColumn.index), ['a', 'b', 'c', 'a', 'a', 'c'])
        self.assertEqual(type(joinedDFByIndex), pd.DataFrame)
        self.assertEqual(type(joinedDFByColumn), pd.DataFrame)

    def test_line_reader(self):
        path = os.path.join(os.getcwd(), 'web_libraries/test.txt')
        result = dh.line_reader(path)
        self.assertEqual(type(result), list)
        self.assertEqual(result[0], 'lorem epsum af')
        self.assertEqual(result[1], '@#$#$%#!@?><,.')
        self.assertEquals(result[2], '123532653870')
        self.assertEqual(len(result), 3)

    def test_normalize_axis(self):
        DF2 = pd.DataFrame([[1 / 4.0, 1 / 4.0, 2 / 4.0], [2 / 6.0, 2 / 6.0, 2 / 6.0], [2 / 11.0, 1 / 11.0, 8 / 11.0]])
        DF3 = pd.DataFrame([[1 / 5.0, 1 / 4.0, 2 / 12.0], [2 / 5.0, 2 / 4.0, 2 / 12.0], [2 / 5.0, 1 / 4.0, 8 / 12.0]])
        self.assertTrue(dh.normalize_axis(self.intDataFrame, axis=0).equals(DF2))
        self.assertTrue(dh.normalize_axis(self.intDataFrame, axis=1).equals(DF3))
        self.assertEqual(type(DF2), pd.DataFrame)
        self.assertEqual(type(DF3), pd.DataFrame)

    def test_remove_data(self):
        testDataFrame = [234, 'Rudy', '6', None]
        self.assertTrue()

    def test_read_entry(self):
        testString = "5, Rudy, 1.6, True, Christian, 54, Jethro, Ricardo, Keren"
        self.assertTrue()

    def test_join_data(self):
        DF = pd.DataFrame([])


"""
Author: Christian Lozoya, 2017
"""

import re

import lozoya.gui.popup


class Tests():
    def __init__(self, parent):
        self.parent = parent

    def test_range_entry(self, entries, value, field):
        for i, entry in enumerate(entries):
            if entry == 'Unnamed: ' + str(i):
                lozoya.gui.popup.PopupError(
                    self.parent, "Invalid Input",
                    "Invalid input:\n" + str(value) + "\nin field:\n" + str(field)
                ).populate()
                break

    def test_inequality_entry(self, entry, value, field):
        if re.findall(r'(?<==)<|(?<==)>', value):
            lozoya.gui.popup.PopupError(
                self.parent, "Invalid Input",
                "Invalid input:\n" + str(value) + "\nin field:\n" + str(
                    field
                ) + "\nInequality should be <,<=, >, or >="
            ).populate()

    def test_entry(self, entry):
        if len(entry) != 0:
            return True
        return False

    def test_checks(self, settings):
        for s in settings:
            if s == True:
                return True
        return False


import re

import lozoya.gui.popup


class Tests:
    def __init__(self, parent):
        self.parent = parent

    def verify_range_entry(self, entries, value, field):
        for i, entry in enumerate(entries):
            if entry == 'Unnamed: ' + str(i):
                print(value)
                lozoya.gui_api.popup.PopupError(
                    self.parent, "Invalid Input",
                    "Invalid input:\n" + str(value) + "\nin field:\n" + str(field)
                ).populate()
                break

    def verify_inequality_entry(self, entry, value, field):
        if re.findall(r'(?<==)<|(?<==)>', value):
            lozoya.gui_api.popup.PopupError(
                self.parent, "Invalid Input",
                "Invalid input:\n" + str(value) + "\nin field:\n" + str(
                    field
                ) + "\nInequality should be <,<=, >, or >="
            ).populate()

    def verify_filepath_entry(self, entry):
        if len(entry) == 0:
            lozoya.gui_api.popup.PopupError(self.parent, "Missing Input", "Enter a valid path")
            return False
        return True

    def verify_settings_checks(self, settings):
        for s in settings:
            if s == True:
                return True
        lozoya.gui_api.popup.PopupError(self.parent, "Missing Selection", "Select")
        return False
