import util


class PathObject:
    """
    A PathObject contains a path string
    and Utilities14 functions to simplify
    interpretability and testing.
    """

    def __init__(self, path):
        self.path = path

    def __repr__(self):
        """
        When an instance of this object_oriented is
        printed, this function is triggered.
        Instead of printing the object, this
        function will override the default
        printing behavior. The result is that
        the object will be printed as follows:
        Object Type: Object Name
        E.g. if the object is a Signal object
        linked to a path to a file called
        ExampleFile.wav, printing the Signal
        object will result in the following:
        Signal: ExampleFile.wav
        """
        return '{0}: {1}'.format(self.__class__.__name__,
                                 util.path_end(self.path))

    @property
    def __name__(self):
        """
        Returns the last part of the path.
        E.g. if the path is 'dir1/dir2/file.ext'
        this function will return 'file.ext'.
        """
        return util.path_end(self.path)
