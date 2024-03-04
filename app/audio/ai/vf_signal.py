import base
import util


class Signal(base.PathObject):
    """
    A Signal is a PathObject that contains a path
    to a file containing waveform information.
    """

    def __init__(self, path):
        base.PathObject.__init__(self, path)

    @property
    def __name__(self):
        return util.path_end(self.path)


class SignalSequence(list):
    """
    SignalSequence is a list of Signals.
    The first item in the SignalSequence
    is the original Signal.
    The next item will be the Signal that
    results from applying the first
    Transformation to the original Signal.
    In search words, the Nth item is the
    result of applying a transformation
    to the (N-1)th item.
    Each SignalSequence should have a
    corresponding TransformationSequence.
    """

    def __init__(self, project, signal):
        self.signal = signal
        self.project = project
        list.__init__(self)
        self.append(self.signal)
        for session in self.project.sessions:
            for task in session.tasks:
                for bouncedFile in task.bouncedFiles:
                    if bouncedFile.__name__ == self.signal.__name__:
                        self.append(bouncedFile)

    @property
    def __name__(self):
        return self.signal.__name__

    def __repr__(self):
        return 'SignalSequence: {}'.format(self.signal.__name__)

    @property
    def signals(self):
        """
        Returns a list of paths corresponding to each
        Signal object in the SignalSequence.
        """
        return [i.path for i in self]

    @property
    def io(self):
        """
        Returns a list of (input, output)
        pairs based on the order of Signals in
        the SignalSequence.
        """
        io = []
        for i, signal in enumerate(self.signals):
            if i > 0:
                io.append((self.signals[i - 1],
                           self.signals[i]))
        return io
