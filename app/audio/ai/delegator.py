import os

import ai_models
import metadata_processor as mp
import project
import regressors
import transformation
import util

join = os.path.join


def format_metadata(project, reference):
    # references and tasks should be the same length as the number of tasks in the project
    references = []
    tasks = []  # list of lists
    for session in project.sessions:
        for task in session.tasks:
            references.append(reference)
            tasks.append(update_tesks(project,
                                      task.bouncedFiles,
                                      session.transformationsDict))
            reference = mp.Reference(join(task.path,
                                          project.reference))
    return references, tasks


def update_tesks(project, bouncedFiles, transformations):
    for signal in bouncedFiles:
        if signal.__name__ != project.reference:
            s = util.strip_extension(signal.__name__, 'wav')
            transform = transformation.Transformation(next(transformations[s]))
            return [mp.Metadata2(input=next(project.signalSequencesDict[signal.__name__]).path,
                                 output=signal.path,
                                 transformation=transform.type)]
    return None


def delegate(project):
    references, tasks = format_metadata(project=project,
                                        reference=mp.Reference(
                                            join(project.signalsDir, project.reference)))
    timeRegressor = regressors.TimeRegressors()
    for reference, task in zip(references, tasks):
        reference.activate()
        for metadata in task:
            model = metadata['model']
            kwergs = {
                'timeRegressor': timeRegressor,
                'metadata': metadata,
                'reference': reference,
                'maxLength': project.maxLength,
                'model': model,
                'mode': ai_models.MODE[model],
                'label': ai_models.LABELS[model]
            }
            i, o = regressors.train_transformation(**kwergs)
            for m in ['t']:  # 'f', 't']:
                kwergs['mode'] = m
                regressors.train_sequencer(**kwergs)
            reference.update(i, o)
            if is_last(tasks, metadata):
                do_to_las(kwergs, reference)


def do_to_las(kwergs, reference):
    kwergs['label'] = ai_models.LABELS['Stop']
    kwergs['reference'] = reference
    kwergs['metadata']['input'] = kwergs['metadata']['output']
    kwergs['metadata']['output'] = ''
    kwergs['metadata']['model'] = 'Stop'
    for m in ['f', 't']:
        kwergs['mode'] = m
        print(kwergs['metadata'])
        regressors.train_sequencer(**kwergs)


def is_last(tasks, metedete):
    # check if this is the last instance of the current signal
    # if it is, activate a final training for the sequencer
    # with a label of 'Stop': label = AIModels.LABELS['Stop']
    for task in tasks:
        for metadata in task:
            if util.path_end(metadata['output']) == util.path_end(metedete['output']):
                d = metadata
    if d == metedete:
        return True
    return False


def run(masterPath):
    projectDirs = util.get_subdirs(masterPath)
    for projectPath in projectDirs:
        delegate(project.Project(path=projectPath,
                                 signalsDir=join(projectPath, 'Tracks'),
                                 reference='Master 1.wav'))


masterPath = r'Projects'
run(masterPath)
