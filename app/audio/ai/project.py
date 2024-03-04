import os
from collections import OrderedDict


class Project:
    def __init__(self, rootDir):
        self.rootDir = rootDir
        self.name = os.path.basename(os.path.normpath(rootDir))

        # PROJECT DIRECTORY
        self.mixDir = self._root('Mix')
        self.masterDir = self._root('Master')
        self.projectDir = self._root('Project')
        self.tracksDir = self._root('Tracks')

        # MIX DIRECTORY
        self.alignDir = os.path.join(self.mixDir, 'Align')
        self.automationDir = os.path.join(self.mixDir, 'Automation')
        self.panningDir = os.path.join(self.mixDir, 'Panning')
        self.pluginsDir = os.path.join(self.mixDir, 'Plugins')
        self.trimDir = os.path.join(self.mixDir, 'Trim')
        self.taskSequencesDir = os.path.join(self.mixDir, 'TaskSequences')

        # PLUGINS DIRECTORY
        self.modelSequencesDir = os.path.join(self.pluginsDir, 'ModelSequences')
        self.pluginSequencesDir = os.path.join(self.pluginsDir, 'PluginSequences')

        self.tracks = [track[:-4] for track in os.listdir(self.tracksDir)]
        self._make_sequences()
        self.modelSequences = self.sequences(self.modelSequencesDir)
        self.pluginSequences = self.sequences(self.pluginSequencesDir)
        self.taskSequences = self.sequences(self.taskSequencesDir)
        self.paths = self.paths()
        self.structure = self.structure()

    def _root(self, dir):
        return os.path.join(self.rootDir, dir)

    def __repr__(self):
        return 'Project {}'.format(self.name)

    def _make_sequences(self):
        def remove_parenthesis(string):
            s = ''
            for char in string:
                if char == '(':
                    s = s.strip()
                    break
                s += char
            return s

        tracks = {}
        _tracks = []
        with open(os.path.join(self.projectDir, '{}.txt'.format(self.name))) as f:
            for line in f:
                if 'SAMPLE RATE:' in line:
                    self.samplingRate = float(line[12:].strip())
                if 'BIT DEPTH:' in line:
                    self.bitDepth = line[10:].strip()
                if 'TRACK NAME:' in line:
                    _ = line[11:].strip()
                    trackName = remove_parenthesis(_)
                    tracks[trackName] = []
                    _tracks.append(trackName)
                if 'PLUG-INS:' in line:
                    _ = line[9:].strip()
                    _plugins = [part for part in _.split('\t') if part]
                    plugins = [remove_parenthesis(p) for p in _plugins]
                    tracks[_tracks[-1]] = plugins
        for track in tracks:
            try:
                filePath = os.path.join(self.pluginSequencesDir, '{}.txt'.format(track))
                with open(filePath, 'w') as f:
                    for plugin in tracks[track]:
                        f.write('{}\n'.format(plugin))
            except Exception as e:
                print(e)

    def print_structure(self, smallPaths=False):
        print('Project: {}'.format(self.name))
        for track in self.structure:
            print('\tTrack: {0}'.format(track))
            postPlugins = False
            for o, task in enumerate(self.taskSequences[track], start=1):
                print('\t\tTask {0}:'.format(o))
                if postPlugins:
                    p = o - 1
                else:
                    p = o
                if task == 'Plugins':
                    nonPlugins = 0
                    for i, model in enumerate(self.structure[track]):
                        if model['Model'] not in ['Align', 'Automation', 'Panning', 'Plugins', 'Trim']:
                            print('\t\t\tPlugin {}:'.format(i + 1 - nonPlugins))
                            for metaData in model:
                                if metaData == 'Plugin':
                                    print('\t\t\t\t{0}: {1}'.format(metaData, model[metaData]))
                                elif metaData == 'Model':
                                    print('\t\t\t\t{0}: {1}'.format(metaData, model[metaData]))
                                else:
                                    if smallPaths:
                                        components = [part for part in model[metaData].split(os.path.sep) if part]

                                        _1 = os.path.join(
                                            '{0}{1}...'.format(
                                                components[0], os.path.sep,
                                                components[-3]
                                            )
                                        )
                                        _2 = os.path.join(_1, components[-2])
                                        path = os.path.join(_2, components[-1])
                                    else:
                                        path = model[metaData]
                                    print('\t\t\t\t{0}: {1}'.format(metaData, path))
                        else:
                            nonPlugins += 1
                    postPlugins = True
                else:
                    print(
                        '\t\t\tModel: {}'.format(
                            task
                        )
                    )  # print('\t\t\tInput: {}'.format(self.structure[track][p - 1]['Input']))  # print('\t\t\tOutput: {}'.format(self.structure[track][p - 1]['Output']))

    def sequences(self, directory):
        sequences = {}
        for track in self.tracks:
            file = '{}.txt'.format(track)
            sequences[track] = []
            with open(os.path.join(directory, file)) as f:
                for line in f:
                    sequences[track].append(line.strip())
        return sequences

    def paths(self):
        '''
        tracks:
        :return:
        '''
        paths = {}
        for track in self.taskSequences:
            fileName = '{}.wav'.format(track)
            paths[track] = [os.path.join(self.tracksDir, fileName)]
            for i, task in enumerate(self.taskSequences[track]):
                if task == 'Plugins':
                    for k, plugin in enumerate(self.modelSequences[track], start=1):
                        pluginPath = os.path.join(self.pluginsDir, 'Plugin{}'.format(k))
                        paths[track].append(os.path.join(pluginPath, fileName))
                else:
                    taskPath = os.path.join(self.mixDir, task)
                    paths[track].append(os.path.join(taskPath, fileName))
        return paths

    def structure(self):
        structure = {}
        for track in self.tracks:
            structure[track] = []
            enu = [i + 1 for i in range(len(self.pluginSequences[track]))]
            postPlugins = False
            for i, task in enumerate(self.taskSequences[track], start=1):
                if postPlugins:
                    m = i + len(self.pluginSequences[track]) - 1
                else:
                    m = i
                inputPath = self.paths[track][m - 1]
                outputPath = self.paths[track][m]
                structure[track].append(
                    OrderedDict([('Task', task), ('Model', task), ('Input', inputPath), ('Output', outputPath)])
                )

                if task == 'Plugins':
                    for k, plugin, model in zip(enu, self.pluginSequences[track], self.modelSequences[track]):
                        output = self.paths[track][k + (m - 1)]
                        structure[track].append(
                            OrderedDict(
                                [('Plugin', plugin), ('Model', model), ('Input', self.paths[track][k + (m - 1) - 1]),
                                 ('Output', output)]
                            )
                        )
                    postPlugins = True

        return structure


def get_all_projects(directory):
    return [Project(os.path.join(directory, projectDir)) for projectDir in os.listdir(directory)]


"""trainingDataDir = r'Z:\Family\LoPar Technologies\LoParDatasets\AudioProducerTrainingData'
projects = get_all_projects(trainingDataDir)
for project in projects:
    print(project.print_structure())"""

"""
This is how to create a Project object
dataDir = r'Z:\Family\LoPar Technologies\LoParAudioProduction\Template'
Project1 = Project(dataDir)

dataDir2 = r'Z:\Family\LoPar Technologies\LoParAudioProduction\Template2'
Project2 = Project(dataDir2)
print(Project1.modelSequences['T1'])
This is how to navigate the Project
for track in Project1.structure:
    for model in Project1.structure[track]:
        print(model['Model'])
        print(model['Input'])
        print(model['Output'])
"""

"""object_oriented Project:
    def __init__(self, masterDir):
        self.masterDir = masterDir
        self.name = os.path.basename(os.path.normpath(masterDir))
        self.projectDir = os.path.join(masterDir,
                                       'Project')

        self.panningDir = os.path.join(masterDir,
                                       'Panning')
        self.pluginSequencesDir = os.path.join(masterDir,
                                        'PluginSequences')
        self.producerSequencesDir = os.path.join(masterDir,
                                                 'ProducerSequences')
        self.modelSequencesDir = os.path.join(masterDir,
                                               'ModelSequences')
        self.tracksDir = os.path.join(masterDir,
                                      'Tracks')
        self.trimDir = os.path.join(masterDir,
                                    'Trim')
        self._make_sequences()
        self.tracks = self.get_tracks(self.pluginSequencesDir)
        self.pluginSequences, self.modelSequences = self.get_sequences(self.tracks)
        self.paths = self.get_paths(self.modelSequences)
        self.structure = self.get_structure()

    def __repr__(self):
        return 'Project {}'.format(self.name)

    def _make_sequences(self):
        def remove_parenthesis(string):
            s = ''
            for char in string:
                if char == '(':
                    s = s.strip()
                    break
                s += char
            return s

        tracks = {}
        _tracks = []
        with open(os.path.join(self.projectDir, '{}.txt'.format(self.name))) as f:
            for line in f:
                if 'SAMPLE RATE:' in line:
                    self.samplingRate = float(line[12:].strip())
                if 'BIT DEPTH:' in line:
                    self.bitDepth = line[10:].strip()
                if 'TRACK NAME:' in line:
                    _ = line[11:].strip()
                    trackName = remove_parenthesis(_)
                    tracks[trackName] = []
                    _tracks.append(trackName)
                if 'PLUG-INS:' in line:
                    _ = line[5:].strip()
                    _plugins = [part
                                for part in _.split('\t')
                                if part]
                    plugins = [remove_parenthesis(p)
                               for p in _plugins]
                    tracks[_tracks[-1]] = plugins
        for track in tracks:
            filePath = os.path.join(self.pluginSequencesDir, '{}.txt'.format(track))
            with open(filePath, 'w') as f:
                for plugin in tracks[track]:
                    f.write('{}\n'.format(plugin))

    def print_structure(self, smallPaths=True):
        print('Project: {}'.format(self.name))
        for track in self.structure:
            print('\tTrack: {0}'.format(track))
            for i, model in enumerate(self.structure[track]):
                print('\t\tPlugin {}:'.format(i + 1))
                for metaData in model:
                    if metaData == 'Plugin':
                        print('\t\t\t{0}: {1}'.format(metaData, model[metaData]))
                    elif metaData == 'Model':
                        print('\t\t\t{0}: {1}'.format(metaData, model[metaData]))
                    else:
                        if smallPaths:
                            components = [part
                                          for part in model[metaData].split(os.path.sep)
                                          if part]

                            _1 = os.path.join(
                                os.path.join('{0}{1}...'.format(components[0],
                                                                os.path.sep),
                                             components[-2])
                            )
                            _2 = os.path.join(_1, components[-1])
                            path = os.path.join(_2, components[-1])
                        else:
                            path = model[metaData]
                        print('\t\t\t{0}: {1}'.format(metaData, path))
        return ''

    def get_tracks(self, dir):
        return [track[:-3]
                for track in
                os.listdir(dir)]

    def get_sequences(self, tracks):
        plugins = {}
        models = {}
        for track in tracks:
            file = '{}.txt'.format(track)
            plugins[track] = []
            models[track] = []
            with open(os.path.join(self.pluginSequencesDir, file)) as f:
                for line in f:
                    plugins[track].append(line.strip())
            with open(os.path.join(self.modelSequencesDir, file)) as f:
                for line in f:
                    models[track].append(line.strip())
        return plugins, models

    def get_paths(self, tracks):
        d = {}
        for track in tracks:
            fileName = '{}.wav'.format(track)
            d[track] = [os.path.join(self.tracksDir, fileName)]
            for i, plugin in enumerate(tracks[track]):
                pluginPath = os.path.join(self.masterDir, 'Plugin{}'.format(i+1))
                d[track].append(os.path.join(pluginPath, fileName))
        return d

    def get_structure(self):
        project = {}
        for track in self.tracks:
            project[track] = []
            enu = [i+1 for i in range(len(self.pluginSequences[track]))]
            for i, plugin, model in zip(enu,
                                        self.pluginSequences[track],
                                        self.modelSequences[track]):
                output = self.paths[track][i]
                project[track].append(
                    OrderedDict([
                        ('Plugin',
                         plugin),
                        ('Model',
                         model),
                        ('Input',
                         self.paths[track][i - 1]),
                        ('Output',
                         output)]))
        return project


def get_all_projects(directory):
    return [Project(os.path.join(directory, projectDir))
            for projectDir in
            os.listdir(directory)]"""

import os

import wavio

from app.audio.ai import base
import util
from metadata_processor import Metadata
from session import Session
from app.audio.ai.transformation import TransformationSequence
from vf_signal import SignalSequence, Signal


class Project(base.PathObject):
    """
    Project contains any arbitrary number
    of Session directories and one singular
    Signals directory.
    """

    def __init__(self, path, signalsDir, reference):
        base.PathObject.__init__(self, path)
        self.signalsDir = signalsDir
        self.reference = reference

    @property
    def sessions(self):
        """
        Returns a list of all Sessions in the Project.
        """
        sessions = util.get_subdirs(self.path, exclude=['Tracks'])
        return [Session(session) for session in sessions]

    @property
    def signals(self):
        """
        Returns the files in signalsDir
        in the Project directory. These are the
        original files uploaded by the client.
        """
        signals = util.get_files(self.signalsDir)
        return [Signal(signal) for signal in signals]

    @property
    def signalSequences(self):
        """
        Returns a list of SignalSequence.
        Each SignalSequence corresponds
        to a TransformationSequence.
        :return:
        """
        return [SignalSequence(self, signal) for signal in self.signals]

    @property
    def transformationSequences(self):
        """
        Returns a list of TransformationSequence.
        Each TransformationSequence corresponds
        to a SignalSequence.
        :return:
        """
        return [TransformationSequence(self, signal) for signal in self.signals]

    @property
    def maxLength(self):
        """
        Returns the length in samples of
        the longest Signal in the Project.
        """
        max = 0
        final = sorted([util.path_end(path) for path in util.get_subdirs(self.path)])[-2]
        finalPath = os.path.join(self.path, final, 'Bounced Files')
        fenal = sorted([util.path_end(path) for path in util.get_subdirs(finalPath)])[-1]
        fp = os.path.join(self.path, final, 'Bounced Files', fenal)
        signals = util.get_files(fp)
        signalsList = [Signal(signal) for signal in signals]
        for signal in self.signals:
            data = wavio.read(signal.path).data
            if len(data) > max:
                max = len(data)
        return max

    @property
    @util.attempt_property
    def samplingRate(self):
        """
        Returns the sampling rate of the project.
        """
        for session in self.sessions:
            with open(os.path.join(session.path, '{}.txt'.format(session.__name__))) as f:
                for line in f:
                    if 'SAMPLE RATE:' in line:
                        return float(line[12:].strip())

    @property
    @util.attempt_property
    def bitDepth(self):
        """
        Returns the bit depth of the project.
        """
        for session in self.sessions:
            with open(os.path.join(session.path, '{}.txt'.format(session.__name__))) as f:
                for line in f:
                    if 'BIT DEPTH:' in line:
                        return line[10:].strip()

    def training_metadata(self, signalSequence, transformationSequence):
        trainingDataMetadata = []
        for io, transformation in zip(signalSequence.io, transformationSequence.transformations):
            metadata = Metadata(
                io,
                transformation,
                os.path.join(
                    os.path.dirname(io[0]),
                    self.reference
                )
            )
            trainingDataMetadata.append(metadata)
        return trainingDataMetadata

    @property
    def signalSequencesDict(self):
        signalSequences = self.signalSequences
        return {s.__name__: iter(signalSequences[i])
                for i, s in enumerate(signalSequences)}


def get_task_name(t):
    _taskName = project_path_collector.path_end(os.path.dirname(t))
    if _taskName == 'Tracks':
        return _taskName
    taskName = _taskName[1:].strip()
    return taskName


def penultimate(p):
    someDict = {}
    for project in p:
        print(project)
        for tracks in p[project]:
            print('\t{}'.format(tracks))
            for i, t in enumerate(p[project][tracks]):

                if i > 0:
                    # print('\t\t{}'.format(t))
                    a = p[project][tracks][i - 1]
                    b = p[project][tracks][i]
                    k = (a, b)
                    k0 = (get_task_name(a), get_task_name(b))
                    print('\t\t{}'.format(k0))
                    taskName = get_task_name(t)
                    if taskName not in someDict:
                        someDict[taskName] = []
                    else:
                        someDict[taskName].append(k)
    return someDict


def final(someDict):
    for task in someDict:
        print(task)
        for t in someDict[task]:
            print('\t{}'.format(t))
        print('\n')


"""
skip Tracks folder.
in the loop, take the current folder
and pair it with the previous folder
in a tuple as follows:
(previous, current)
"""
"""
the name of the current folder will be used as the key
to a dictionary:
someDict[currentFolder] = []
the list inside someDict[currentFolder] is going to contain
the tuples. Secretly "currentFolder" is actually the name
of the AI we are training. e.g. EQ, Compressor, etc
"""

path = r'Z:\Family\LoParTechnologies\PythonServer\python\Services\Audio\Gee, I dunno\Projects'
p = ProjectPathCollector.get_errthang(path)
x = penultimate(p)
# final(x)

import os

import project_path_collector


def get_task_name(t):
    _taskName = Util.path_end(os.path.dirname(t))
    if _taskName == 'Tracks':
        return _taskName
    taskName = _taskName[1:].strip()
    return taskName


def penultimate(p):
    someDict = {}
    for project in p:
        print(project)
        for tracks in p[project]:
            print('\t{}'.format(tracks))
            for i, t in enumerate(p[project][tracks]):

                if i > 0:
                    # print('\t\t{}'.format(t))
                    a = p[project][tracks][i - 1]
                    b = p[project][tracks][i]
                    k = (a, b)
                    k0 = (get_task_name(a), get_task_name(b))
                    print('\t\t{}'.format(k0))
                    taskName = get_task_name(t)
                    if taskName not in someDict:
                        someDict[taskName] = []
                    someDict[taskName].append(k)
    return someDict


def final(someDict):
    for task in someDict:
        print(task)
        for t in someDict[task]:
            print('\t{}'.format(t))
        print('\n')


"""
skip Tracks folder.
in the loop, take the current folder
and pair it with the previous folder
in a tuple as follows:
(previous, current)
"""
"""
the name of the current folder will be used as the key
to a dictionary:
someDict[currentFolder] = []
the list inside someDict[currentFolder] is going to contain
the tuples. Secretly "currentFolder" is actually the name
of the AI we are training. e.g. EQ, Compressor, etc
"""

# path = r'Z:\Family\LoParTechnologies\PythonServer\python\Services2\Audio\Gee, I dunno\Projects'
path = r'C:\Users\christian.lozoya\Downloads\over9000\Crawler\Projects'
p = ProjectPathCollector.get_errthang(path)
x = penultimate(p)  # final(x)
