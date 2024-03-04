import ast
import hashlib
import inspect
import os
import subprocess
import uuid
from collections import defaultdict

import lozoya.text


def brute_change_duplicate_filenames(path):
    repeat = True
    while repeat:
        repeat = False
        allPaths = []
        for root, directories, files in os.walk(path):
            for file in files:
                allPaths.append(os.path.join(root, file))
        for pathA in allPaths:
            for pathB in allPaths:
                try:
                    filename = os.path.split(pathB)[-1]
                    if (pathA != pathB) and (filename == os.path.split(pathA)[-1]):
                        print('Renaming:\n - %s\n - %s' % (pathA, pathB))
                        filenameroot, extension = filename.split('.')
                        directoryB = os.path.split(pathB)[0]
                        newFilenameB = '%s%s.%s' % (filenameroot, uuid.uuid1(), extension)
                        newPath = os.path.join(directoryB, newFilenameB)
                        os.rename(pathB, newPath)
                        repeat = True
                        break
                except Exception as e:
                    pass


def check_for_duplicates(path, ignoredSubstrings=[], ignoreEmptyFiles=True):
    """
    based on https://stackoverflow.com/a/36113168/300783
    """
    filesBySize = get_files_by_size(path, ignoredSubstrings)
    filesBySmallHash = get_files_by_small_hash(filesBySize)
    # For all files with the hash on the first 1024 bytes, get their hash on the full
    # file - collisions will be duplicates
    filesByFullHash = dict()
    for files in filesBySmallHash.values():
        if len(files) < 2:
            # the hash of the first 1k bytes is unique -> skip this file
            continue
        for filename in files:
            if (os.path.getsize(filename) > 2) or (ignoreEmptyFiles == False):
                try:
                    fullHash = get_hash(filename, first_chunk_only=False)
                except OSError as e:
                    print(e)
                    # the file access might've changed till the exec point got here
                    continue
                if fullHash in filesByFullHash:
                    duplicate = filesByFullHash[fullHash]
                    if not lozoya.text.match_substring(filename, ignoredSubstrings):
                        return (filename, duplicate)
                else:
                    filesByFullHash[fullHash] = filename
    return False


def chunk_reader(fobj, chunk_size=1024):
    """ Generator that reads a file in chunks of bytes """
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            return
        yield chunk


def clear_directories(directory):
    for folder in directory:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)


def cleanup_filename(filename):
    newFilename = ' '.join(filename.split('-'))
    newFilename = ' '.join(newFilename.split())
    newFilename = newFilename.lower()
    newFilename = newFilename.replace(' ', '_')
    newFilename = newFilename.replace('-', '_')
    return newFilename


def find_first_duplicates(path, ignoredSubstrings=[], show_in_explorer=False, ignoreEmptyFiles=False):
    print("Searching for duplicates...")
    result = check_for_duplicates(path, ignoredSubstrings, ignoreEmptyFiles=ignoreEmptyFiles)
    if result:
        filename, duplicate = result
        # write_to_log('%s\n%s\n-\n' % (filename, duplicate), logpath)
        print("Duplicate found:\n - %s\n - %s" % (duplicate, filename))
        if show_in_explorer:
            show_duplicates_in_explorer(filename, duplicate)
    else:
        print("No duplicates found.")


def get_files_by_size(path, ignoredSubstrings):
    filesBySize = defaultdict(list)
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            fullPath = os.path.join(dirpath, filename)
            if not lozoya.text.match_substring(fullPath, ignoredSubstrings):
                try:
                    # if the target is a symlink (soft one), this will
                    # dereference it - change the value to the actual target file
                    fullPath = os.path.realpath(fullPath)
                    fileSize = os.path.getsize(fullPath)
                except OSError as e:
                    # not accessible (permissions, etc) - pass on
                    print(e)
                    continue
                filesBySize[fileSize].append(fullPath)
    return filesBySize


def get_files_by_small_hash(filesBySize):
    # For all files with the same file size, get their hash on the first 1024 bytes
    filesBySmallHash = defaultdict(list)
    for file_size, files in filesBySize.items():
        if len(files) < 2:
            continue  # this file size is unique, no need to spend cpu cycles on it
        for filename in files:
            try:
                small_hash = get_hash(filename, first_chunk_only=True)
            except OSError:
                # the file access might've changed till the exec point got here
                continue
            filesBySmallHash[(file_size, small_hash)].append(filename)
    return filesBySmallHash


def get_hash(filename, first_chunk_only=False, hash_algo=hashlib.sha1):
    hashobj = hash_algo()
    with open(filename, "rb") as f:
        if first_chunk_only:
            hashobj.update(f.read(1024))
        else:
            for chunk in chunk_reader(f):
                hashobj.update(chunk)
    return hashobj.digest()


def remove_paths(paths):
    for path in paths:
        os.remove(path)


def get_empty_files(path, ignoredSubstrings=[]):
    files = get_files_in_directory(path)
    return [x for x in files if (is_file_empty(x) and not is_path_ignored(x, ignoredSubstrings))]


def get_files(path):
    subpaths = get_subpaths(path)
    return list(filter(lambda subpath: os.path.isfile(subpath), subpaths))


def get_files2(directory):
    _files = []
    for dir, subdir, files in os.walk(directory):
        for file in files:
            _files.append(os.path.join(dir, file))
    return _files


def get_folders(path):
    subpaths = get_subpaths(path)
    return list(filter(lambda path: os.path.isdir(path), subpaths))


def get_empty_folders(path, ignoredSubstrings=[]):
    folders = get_folders(path)
    return list(filter(lambda path: is_folder_empty(path), folders))


def get_duplicate_files(path, ignoredSubstrings=[]):
    return get_files_in_directory(path)


def get_subpaths(path, ignoredSubstrings=[]):
    return get_files_in_directory(path)


def get_files_in_directory(path):
    paths = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in os.walk(path) for f in filenames]
    return [x for x in paths if os.path.isfile(x)]


def is_file_empty(path, emptySize=0):
    return os.path.getsize(path) <= emptySize


def is_folder_empty(path, emptySize=0):
    return os.path.getsize(path) <= emptySize


def is_path_ignored(path, ignoredSubstrings=[]):
    return lozoya.text.match_substring(path, ignoredSubstrings)


def remove_all_empty_directories(path, ignoredSubstrings=[]):
    print('Removing all empty directories...')
    run = True
    while (run):
        for root, directories, files in os.walk(path):
            if (not lozoya.text.match_substring(root, ignoredSubstrings)):
                run = remove_empty_directory(root)
                if run:
                    break
    print('Finished removing all empty directories.')


def remove_all_duplicates(path, ignoredSubstrings=[], preferredPath=None):
    print("Removing all duplicates...")
    logpath = '{}/duplicates.log'.format(path)
    result = check_for_duplicates(path, ignoredSubstrings)
    if result:
        while result:
            try:
                filename, duplicate = result
                if preferredPath:
                    if preferredPath in filename:
                        os.remove(duplicate)
                        write_to_log('+ %s\nx %s\n-' % (filename, duplicate), logpath)
                    else:
                        os.remove(filename)
                        write_to_log('+ %s\nx %s\n-' % (duplicate, filename), logpath)
                else:
                    os.remove(duplicate)
                    write_to_log('+ %s\nx %s\n-' % (filename, duplicate), logpath)
                result = check_for_duplicates(path, ignoredSubstrings)
            except Exception as e:
                print(e)
                print(filename, duplicate)
                break
    else:
        print("No duplicates found.")
    print("Finished removing all duplicates.")


def remove_by_substring(path, substringList):
    print("Removing all paths containing {}...".format(substringList))
    logpath = '{}/removed_by_substring.log'.format(path)
    files = get_files_in_directory(path)
    files = [x for x in files if lozoya.text.match_substring(x, substringList)]
    for file in files:
        write_to_log('x %s' % file, logpath)
        os.remove(file)


def remove_empty_directory(directory):
    try:
        os.rmdir(directory)
        print("Removed: {}".format(directory))
        return True
    except OSError as e:
        return False


def remove_all_empty_files(path, ignoredSubstrings):
    print("Removing all empty files...")
    logpath = '{}/empty_files.log'.format(path)
    files = get_empty_files(path, ignoredSubstrings)
    for file in files:
        write_to_log('x %s' % file, logpath)
        os.remove(file)


def show_in_explorer(path):
    subprocess.Popen(r'explorer /select,"{}"'.format(path))


def show_duplicates_in_explorer(fileA, fileB):
    show_in_explorer(fileA)
    if os.path.split(fileA)[0] != os.path.split(fileB)[0]:
        show_in_explorer(fileB)


def file_count(path):
    files = get_files(path)
    return len(files)


def folder_count(path):
    folders = get_folders(path)
    return len(folders)


def file_size_sum(paths):
    size = 0
    for path in paths:
        size += os.path.getsize(path)
    return size


def sort_files_by_type(path):
    print('Sorting files by type...')
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            try:
                oldPath = os.path.join(dirpath, filename)
                extension = oldPath.split('.')[-1]
                directory = os.path.join(path, extension)
                newPath = os.path.join(directory, filename)
                if not (os.path.isdir(directory)):
                    os.makedirs(directory)
                os.rename(oldPath, newPath)
            except Exception as e:
                pass


def update_filename(root, filename):
    newFilename = cleanup_filename(filename)
    oldPath = os.path.join(root, filename)
    newPath = os.path.join(root, newFilename)
    os.rename(oldPath, newPath)


def update_directories(root, directories):
    for directory in directories:
        newName = cleanup_filename(directory)
        oldPath = os.path.join(root, directory)
        newPath = os.path.join(root, newName)
        os.rename(oldPath, newPath)


def write_to_log(msg, logpath):
    print(msg)
    with open(logpath, 'a') as file:
        file.write('{}\n'.format(msg))


def readable_size(size):
    if size < 1024:
        unit = 'B'
    elif size < 1000000:
        size = size / 1000
        unit = 'kB'
    elif size < 1000000000:
        size = size / 1000000
        unit = 'MB'
    else:
        size = size / 1000000000
        unit = 'GB'

    return '{} {}'.format(size, unit)


class TSConfigurationFile:
    def __init__(self, path, delimiter='='):
        self._path = path
        self._delimiter = delimiter
        self.load_from_file()

    def __setattr__(self, attribute, value):
        try:
            super().__setattr__(attribute, ast.literal_eval(value))
        except Exception as e:
            super().__setattr__(attribute, value)

    @property
    def configuration(self, *args, **kwargs):
        return {attribute: getattr(self, attribute) for attribute in vars(self)}

    def load_from_file(self, *args, **kwargs):
        self.__setattr__ = self.update
        if os.path.isfile(self._path) and not is_file_empty(self._path):
            with open(self._path, 'r') as preferenceFile:
                for line in preferenceFile:
                    attribute, value = line.strip('\n').split(self._delimiter)
                    self.__setattr__(attribute, value)
            return True
        self.save_to_file(*args, **kwargs)
        return False

    def save_to_file(self, *args, **kwargs):
        with open(self._path, 'w') as preferenceFile:
            configuration = self.configuration
            for attribute in sorted(configuration):
                if attribute[0] != '_':
                    preferenceFile.write('{}{}{}\n'.format(attribute, self._delimiter, configuration[attribute]))

    def update(self, attribute, value):
        setattr(self, attribute, value)
        self.save_to_file()


def make_configuration_file_from_class(template: object, path: str):
    class TSConfigurationFileTemplate(template, TSConfigurationFile):
        def __init__(self, path):
            template.__init__(self)
            TSConfigurationFile.__init__(self, path)
            self.__name__ = template.__name__

    return TSConfigurationFileTemplate(path)


def make_configuration_files_from_class(templates: str, root: str):
    configuratorList = []
    for template in templates:
        path = f'{root}/{template.__name__}.config'
        configurator = make_configuration_file_from_class(template=template, path=path)
        configuratorList.append(configurator)
    return configuratorList


def make_configuration_file_from_function(template, path: str):
    class TSConfigurationFileTemplate(TSConfigurationFile):
        def __init__(self, path):
            self.__name__ = template.__name__
            for key in inspect.signature(template).parameters:
                parameter = inspect.signature(template).parameters[key]
                if (parameter.default and type(parameter.default) != type(inspect._empty)):
                    self.__dict__[key] = parameter.default
            TSConfigurationFile.__init__(self, path)

    return TSConfigurationFileTemplate(path)


def make_configuration_files_from_function(templates, root: str):
    configuratorList = []
    for template in templates:
        path = f'{root}/{template.__name__}.config'
        configurator = make_configuration_file_from_function(template=template, path=path)
        configuratorList.append(configurator)
    return configuratorList
