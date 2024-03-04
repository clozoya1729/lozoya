import ai
import util


class Transformation:
    """
    A Transformation is any alteration to a Signal.
    E.g. compression, reverb, gain staging.
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return 'Transformation: {}'.format(self.type)

    @property
    def type(self):
        """
        Uses the transformationTypes dictionary
        to match the Transformation to a general
        transformation and returns the general
        transformation.
        """
        for transformationType in ai.TRANSFORMATIONS:
            if util.standardize_string(self.name) in ai.TRANSFORMATIONS[transformationType]:
                return transformationType


class TransformationSequence(list):
    """
    A TransformationSequence is a list
    of Transformations.
    Each Transformation corresponds to
    a Signal. Therefore, every
    TransformationSequence should have
    a corresponding SignalSequence.
    """

    def __init__(self, project, signal):
        self.signal = signal
        self.project = project
        list.__init__(self)
        for session in self.project.sessions:
            for t in session.transformations:
                if t == util.strip_extension(self.signal.__name__, '.wav'):
                    self.extend(session.transformations[t])

    def __repr__(self):
        return 'TransformationSequence: {}'.format(self.signal.__name__)

    @property
    def transformations(self):
        """
        Returns the Transformation objects
        contained in the TransformationSequence.
        """
        return [Transformation(i) for i in self]
