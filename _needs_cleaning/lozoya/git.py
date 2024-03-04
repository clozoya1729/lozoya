import os
import shutil
import importlib

import pygit2


def run(dirName, project, moduleName, f, *args, **kwargs):
    username = 'lopartechnologies'
    accessToken = 'a1f29182d2f7f0495e29e8d06a9b0f8b212e2117'
    url = 'https://{}:x-oauth-basic@github.com/{}/{}.git'.format(accessToken, username, project)
    try:
        if os.path.isdir(dirName):
            shutil.rmtree(dirName)
        os.mkdir(dirName)
        try:
            pygit2.clone_repository(url, dirName)
            module = importlib.import_module('{}.{}'.format(dirName, moduleName))
            getattr(module, f)(*args, **kwargs)
        except Exception as e:
            print(e)
        shutil.rmtree(dirName)
    except Exception as e:
        print(e)
