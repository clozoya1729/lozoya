from collections import OrderedDict


def mingo(r, sessions):
    for session in sessions:
        tasks = explore_tasks(session)
        for task in tasks:
            files = get_files(task)
            for f in files:
                r.append(f)
    return r


def _collect(path):
    r = []
    tracks = get_tracks(path)
    for track in tracks:
        r.append(track)
    sessions = get_sessions(path)
    return mingo(r, sessions)


def collect_project_files(path):
    pathList = get_project_paths(path)
    projectFiles = []
    for path in pathList:
        projectFiles.append((util.path_end(path), []))
        r = _collect(path)
        for re in r:
            projectFiles[-1][1].append(re)
    return OrderedDict(projectFiles)


def crawl_letter(path):
    projects = util.get_subdirs(path)
    return [os.path.join(path, project) for project in projects]


def explore_tasks(session):
    path = os.path.join(session, 'Bounced Files')
    tasks = util.get_subdirs(path)
    return [os.path.join(path, task) for task in tasks]


def get_project_paths(path):
    p = []
    subdirs = util.get_subdirs(path)
    for subdir in subdirs:
        subpath = os.path.join(path, subdir)
        projectPaths = crawl_letter(subpath)
        p.extend(projectPaths)
    return p


def get_sessions(path):
    sessions = util.get_subdirs(path)
    return [os.path.join(path, session) for session in sessions if session != 'Tracks']


def get_subdirs(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def get_files(task):
    files = next(os.walk(task))[2]
    return [os.path.join(task, f) for f in files]


def get_tracks(path):
    tracksDir = os.path.join(path, 'Tracks')
    return get_files(tracksDir)


def path_end(path):
    return os.path.basename(os.path.normpath(path))


def get_errthang(path):
    projects = collect_project_files(path)
    p = {project: dingo(projects[project]) for project in projects}
    return p


def dingo(project):
    tracks = {}
    for path in project:
        track = util.path_end(path)
        if track not in tracks:
            tracks[track] = []
        tracks[track].append(path)
    return tracks


import os


def vf_cleanup():
    count = 0
    nSteps = 10
    majenta = [str(i) for i in range(nSteps)]
    path = ''
    level1 = next(os.walk(path))[1]
    level2 = []
    for d in level1:
        if d in majenta:
            level2.extend(next(os.walk(os.path.join(path, d)))[1])

    for d in level1:
        if d in level2:
            print(d)
            os.rmdir(d)
            count += 1

    print('{} folders deleted'.format(count))


# The following are Utilities14 print functions
# to facilitate interpretability

def print_signals(project):
    print('  Signals: {}'.format(project.signals))


def print_sessions(project):
    for session in project.sessions:
        print('  {}'.format(session))
        for task in session.tasks:
            print('    {}'.format(task))
            print('      Bounced Files: {}'.format(task.bouncedFiles))


def print_signal_sequences(project):
    print('Signal Sequences:')
    for sequence in project.signalSequences:
        print('  {}:'.format(sequence.signal.__name__))
        for signal in sequence.signals:
            print('    {}'.format(signal))


def print_transformation_sequences(project):
    print('Transformation Sequences:')
    for sequence in project.transformationSequences:
        if sequence.transformations:
            print('  {}:'.format(sequence))
            for transformation in sequence.transformations:
                print('    {}'.format(transformation))


def print_project(project):
    print('-----BEGIN PROJECT-----')
    print(project)
    print(project.bitDepth)
    print(project.samplingRate)
    print_signals(project)
    print_sessions(project)
    print_signal_sequences(project)
    print_transformation_sequences(project)
    print('------END PROJECT------\n')
