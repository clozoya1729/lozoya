from lozoya.file import remove_all_empty_directories, remove_all_empty_files, remove_by_substring, get_empty_files

ignoredSubstrings = [
    '.idea',
    '.git',
    'LICENSE',
    'python_virtual_environment',
    'unreal',
    '__init__.py',
]

removeSubstrings = [
    '.drawio.bkp',
]

searchedExtensions = [
    '.csv',
    '.doc',
    '.drawio'
    '.jpeg',
    '.jpg',
    '.pdf',
    '.png',
    '.ppt',
    '.tif',
    '.txt',
    '.wmv',
    '.xls',
]


def run(path):
    print("Running on {}".format(path))
    # sort_files_by_type(path)
    # brute_change_duplicate_filenames(path)
    remove_by_substring(path, ['__pycache__'])
    remove_all_empty_files(path, ignoredSubstrings)
    # remove_all_duplicates(path, ignoredSubstrings, preferredPath='')
    remove_all_empty_directories(path, ignoredSubstrings)
    print("Finished")


run(path=r'E:\github')
