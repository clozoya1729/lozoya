from collections import OrderedDict

'''def inverted(self):
    files = []
    for p in self:
        if 'Tracks' in p:
            files.append([(p, f) for f in self[p]])
    for f in files:
        print(f)
    return files'''

''''''


class Crawler(OrderedDict):
    def __repr__(self):
        r = ''
        for key in self:
            r += 'Project: {}\n'.format(key)
            for p in self[key]:
                r += '\t{0}: {1}'.format(p, self[key][p])
                r += '\n'
            r += '\n'
        return r

    def get_projects(self, path):
        """
        Returns a list of
        paths to all projects in
        the Projects directory as
        an OrderedDict with (key, value)
        pairs of (project name, project path).
        """
        projects = []
        for directory, subdirectories, files in os.walk(path):
            base = os.path.basename(os.path.normpath(directory))
            if '-' in base and base not in ['A-L', 'M-Z']:
                projects.append((base, directory))
        return OrderedDict(projects)

    def get_files(self, path, extensions):
        """
        path: str, path to a project folder
        """
        for directory, subdirectories, files in os.walk(path):
            if files:
                matches = self.find_ext(files, extensions)
                if matches:
                    projectName = self.get_project_name(directory)
                    self[projectName] = OrderedDict([('path', directory), ('files', matches)])

    def find_ext(self, files, extFind):
        matches = []
        for f in files:
            for ext in extFind:
                if ext in f:
                    matches.append(f)
        return matches

    def get_project_name(self, directory):
        """
        Project name is identified by
        moving up the folder hierarchy
        until a folder containing
        a hyphen is found, as in:
        Artist Name - Song Title
        Therefore, no folder should have a hyphen,
        except for the main project folder and the folders
        'A-L' and 'M-Z'.
        """
        base = ''
        d = directory
        while '-' not in base:
            base = os.path.basename(os.path.normpath(d))
            d = os.path.dirname(d)
        return base


path = r'Z:\Family\LoParTechnologies\PythonServer\data\AudioTest'
extensions = ['.wav']

c = Crawler()
projects = c.get_projects(path)
for project in projects:
    c.get_files(projects[project], extensions)
print(c)


# inverted = c.inverted()


def blerg(path, suffix):
    for directory, subdirectories, files in os.walk(path):
        if files:
            for file in files:
                f = os.path.join(directory, file)
                if rString in f:
                    newF = '{0}.wav'.format(f[:-(4 + len(suffix))])
                    os.rename(src=f, dst=newF)


path = r'Z:\Family\ProTools\A\Abeltones Big Band - Corine Corine\1\Bounced Files'
blerg(path, suffix='-St')

from collections import OrderedDict

'''def inverted(self):
    files = []
    for p in self:
        if 'Tracks' in p:
            files.append([(p, f) for f in self[p]])
    for f in files:
        print(f)
    return files'''

''''''


class Crawler(OrderedDict):
    def __repr__(self):
        r = ''
        for key in self:
            r += 'Project: {}\n'.format(key)
            for p in self[key]:
                r += '\t{0}: {1}'.format(p, self[key][p])
                r += '\n'
            r += '\n'
        return r


    def get_files(self, path, extensions):
        """
        path: str, path to a project folder
        """
        for directory, subdirectories, files in os.walk(path):
            if files:
                print(directory)
                print(files)
                print('\n')
                matches = self.find_ext(files, extensions)
                if matches:
                    projectName = self.get_project_name(directory)
                    self[projectName] = OrderedDict([('path', directory), ('files', matches)])

    def find_ext(self, files, extFind):
        matches = []
        for f in files:
            for ext in extFind:
                if ext in f:
                    matches.append(f)
        return matches

    def get_project_name(self, directory):
        """
        Project name is identified by
        moving up the folder hierarchy
        until a folder containing
        a hyphen is found, as in:
        Artist Name - Song Title
        Therefore, no folder should have a hyphen,
        except for the main project folder and the folders
        'A-L' and 'M-Z'.
        """
        base = ''
        d = directory
        while '-' not in base:
            base = os.path.basename(os.path.normpath(d))
            d = os.path.dirname(d)
        return base


path = r'C:\Users\christian.lozoya\Downloads\Crawler\Projects'
extensions = ['.txt']

c = Crawler()
projects = c.get_projects(path)
for project in projects:
    c.get_files(projects[project], extensions)
print(c)
# inverted = c.inverted()

import collections
import os


def path_end(path):
    return os.path.basename(os.path.normpath(path))


def crawl(pathList):
    projects = []
    for path in pathList:
        projects.append((path_end(path), []))
        sessions = ProjectPathCollector.get_sessions(path)
        for session in sessions:
            projects[-1][1].append((path_end(session), []))
            tasks = ProjectPathCollector.explore_tasks(session)
            for task in tasks:
                pass  # print(task)

    projectsDict = collections.OrderedDict(projects)
    for key in projectsDict:
        print('Project: {}'.format(key))
        for key2 in projectsDict[key]:
            print('\tSession: {}'.format(key2))


path = r'C:\Users\christian.lozoya\Downloads\over9000\Crawler\Projects'
projectPaths = ProjectPathCollector.get_project_paths(path)
crawl(projectPaths)
