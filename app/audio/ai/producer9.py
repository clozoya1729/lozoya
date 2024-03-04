import os

from utils.util import path_end, strip_ext


def get_next_dir(currDir):
    nextDir = int(path_end(currDir)) + 1
    if len(str(nextDir)) == 1:
        return str('0{}'.format(nextDir))


def update_transformations(transDir, transformation, filePath):
    fileName = '{}.txt'.format(strip_ext(path_end(filePath)))
    path = os.path.join(transDir, fileName)
    if os.path.isfile(path):
        mode = 'a'
    else:
        mode = 'w'
    with open(path, mode) as o:
        o.write(transformation + '\n')
