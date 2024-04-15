import os

import util

from app.audio.ai.other import metadata_processor as mp

masterPath = r'Projects'
projectTitle = 'Skelpolu - Anomalous Weeping'
projectPath = util.get_subdirs(masterPath)[0]
assert (projectPath == 'Projects{}{}'.format(os.sep, projectTitle))
print('Project Path: {}'.format(projectPath))

project = mp.Project(path=projectPath, signalsDir=os.path.join(projectPath, 'Tracks'), reference='Master 1.wav')
print('Project Object: {}'.format(project))
print('Project Object Path: {}'.format(project.path))
assert (project.path == projectPath)
assert (project.signalsDir == os.path.join(projectPath, 'Tracks'))
assert (project.reference == 'Master 1.wav')

print(project.sessions)
assert (type(project.sessions) == type([]))
session = project.sessions[0]
print(session.path)
assert (session.path == os.path.join(project.path, '1 {}'.format(projectTitle)))

transformations = session.transformationsDict
assert (sorted(list(transformations.keys())) == ['Master 1', 'kick', 'snare'])

tasks = session.tasks
print(tasks)
taskNames = session.taskNames
print(taskNames)

assert (taskNames == ['1 GainStaged', '1 Panning', '2 Eq'])


def test_homogenize_length():
    pass


def test_interleave():
    pass


def test_pad_data():
    pass


def test_temporal_padding():
    pass
