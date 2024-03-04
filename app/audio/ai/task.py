import base
import util
from vf_signal import Signal


class Task(base.PathObject):
    """
    A Task is a collection of transformations
    and exist within a Session. Each Task applies
    at most one Transformation per signal. Each
    Signal can have a different Transformation, or
    no Transformation at all. However, it is desirable
    to have no more than 1 types of Transformation
    per Task, and that these transformations be
    of a similar nature.
    """

    def __init__(self, path):
        base.PathObject.__init__(self, path)

    @property
    def bouncedFiles(self):
        """
        Returns all bounced files in a Task.
        """
        return [Signal(signal)
                for signal in
                util.get_files(self.path)]
