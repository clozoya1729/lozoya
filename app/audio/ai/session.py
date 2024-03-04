import os
import re

import base
import util
from task import Task


class Session(base.PathObject):
    """
    A Session represents a ProTools file.
    """

    def __init__(self, path):
        base.PathObject.__init__(self, path)
        self._transformations = None

    @property
    def tasks(self):
        """
        Returns all Task directories inside the
        Bounced Files directory of the Session directory
        inside the Project.
        """
        path = os.path.join(self.path, 'Bounced Files')
        tasks = util.get_subdirs(path)
        return [Task(task)
                for task in tasks]

    @property
    def transformations(self):
        """
        Returns a dictionary of all transformations
        per signal. (key, value) pairs are in the form
        of (signal name, transformations list).
        The list of transformations is a list of str
        rather than a list of Transformation objects.
        """
        if not self._transformations:
            try:
                tracks = self.read_session_text()
            except:
                tracks = self.read_session_tasks()
            self._transformations = tracks
        return self._transformations

    @property
    def transformationsDict(self):
        return {key: iter(self.transformations[key])
                for key in self.transformations}

    @property
    def taskNames(self):
        return [task.__name__ for task in self.tasks]

    def read_session_text(self):
        """
        Reads a ProTools Session Text file
        with the same name as the Project
        and returns a dictionary of each plugin
        used in each track of the ProTools session.
        This dictionary has (key, value) pairs
        in the form of (signal name, transformation list)
        where the transformation list is the list of
        plugins found within the ProTools Session Text file.
        """
        signals = {}
        _signals = []
        path = os.path.join(self.path,
                            '{}.txt'.format(self.__name__))
        with open(path) as f:
            for line in f:
                if 'TRACK NAME:' in line:
                    signalName = line[11:].strip()
                    signals[signalName] = []
                    _signals.append(signalName)
                if 'PLUG-INS:' in line:
                    tabbedTransformations = line[9:].strip()
                    _transformations = [part
                                        for part in tabbedTransformations.split('\t')
                                        if part]
                    transformations = [util.remove_parenthesis(p)
                                       for p in _transformations]
                    signals[_signals[-1]] = transformations
        return signals

    def read_session_tasks(self):
        """
        Creates a sorted list of the name of each Task
        folder inside the session. These are interpreted
        as transformations. A dictionary with (key, value)
        pairs of (signal name, transformation list) is returned,
        where the "transformation list" is the list of task
        folders in which the signal is found.
        This is used as a fallback in case
        a session text file with the same name as the
        Project is not found.
        """
        signals = {}
        for task in self.tasks:
            for file in task.bouncedFiles:
                signalName = util.strip_extension(file.__name__, '.wav')
                if signalName not in signals:
                    signals[signalName] = []
                t = re.sub('^\d+', '', task.__name__)
                t = util.standardize_string(t)
                signals[signalName].append(t)
        return signals
