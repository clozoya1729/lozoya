import uuid

import django.db.models
import numpy as np
import pandas as pd

import app.data.app_django.tensorstone.models as general_models


class TSModelMathBase(general_models.TSModelAppSingleInput):
    result = np.ndarray((0,))

    def __init__(self, *args, **kwargs):
        super(TSModelMathBase, self).__init__(*args, **kwargs)
        self.lamb = lambda x: x
        self.objectType = 'mathfunction'

    @property
    def x(self):
        return np.fromiter(self.get_input()['out'], dtype=float)

    def compute(self):
        try:
            if self.defaultColumn in self.get_columns():
                self.output = self.get_input().apply(self.lamb)
            else:
                _ = self.get_input()
                _[self.inputColumns] = _[self.input].apply(self.lamb)
                self.output = _
            self.set_output()
        except:
            pass

    def compute(self):
        try:
            collies = self.get_input_columns()
            _ = self.get_input().astype(float)
            if self.defaultColumn in collies:
                self.output = _.apply(self.lamb)
            else:
                _[collies] = _[collies].apply(self.lamb)
                self.output = _
            self.set_output()
        except Exception as e:
            print('MATH MODEL ERROR: ' + str(e))

    def compute(self):
        print(self.result)
        self._output = {'out': self.result.tolist(), }
        self.set_output()

    def get_col(self):
        return self.inputColumns.split(',')

    def refresh_input(self):
        self.compute()
        super(TSModelMathBase, self).save()

    def save(self, *args, **kwargs):
        try:
            self.compute()
        except Exception as e:
            print(e)
        super(TSModelMathBase, self).save()

    class Meta:
        verbose_name = 'Math/Base'
        verbose_name_plural = 'Math/Base'


class TSModelMathBaseFunction(general_models.TSModelAppSingleInput):
    result = np.ndarray((0,))

    def __init__(self, *args, **kwargs):
        super(TSModelMathBase, self).__init__(*args, **kwargs)
        self.lamb = lambda x: x
        self.objectType = 'mathfunction'

    def get_col(self):
        return self.inputcolumns.split(',')

    def refresh_input(self):
        self.compute()
        super(TSModelMathBase, self).save()

    def compute(self):
        try:
            collies = self.get_input_columns()
            df = self.get_input().astype(float)
            if self.defaultColumn in collies:
                df = df.apply(self.lamb)
            else:
                df[collies] = df[collies].apply(self.lamb)
            self.set_output(df)
        except Exception as e:
            print('MATH MODEL ERROR: ' + str(e))

    def save(self, *args, **kwargs):
        self.compute()
        super(TSModelMathBase, self).save()

    class Meta:
        verbose_name = 'Math/Base'
        verbose_name_plural = 'Math/Base'


class TSModelMathBaseGenerator(general_models.TSModelObject):
    rows = models.IntegerField(default=10, )
    columns = models.IntegerField(default=1, )
    result = np.ndarray((0,))

    def __init__(self, *args, **kwargs):
        super(TSModelMathBaseGenerator, self).__init__(*args, **kwargs)
        self.objectType = 'generator'
        self.compute()

    def compute(self):
        index = ['col{}'.format(i) for i in range(self.columns)]
        data = [self._compute() for i in range(self.columns)]
        df = pd.DataFrame(data=data, index=index).T
        self.set_output(df)

    def _compute(self):
        pass

    def save(self, *args, **kwargs):
        self.compute()
        super(TSModelMathBaseGenerator, self).save()

    class Meta:
        verbose_name = 'Generator/Base'
        verbose_name_plural = 'Generator/Base'


class TSModelMathBaseGenerator(general_models.TSModelObject):
    a = models.FloatField(default=0, )
    b = models.FloatField(default=1, )
    steps = models.IntegerField(default=10, )
    result = np.ndarray((0,))

    def __init__(self, *args, **kwargs):
        super(TSModelMathBaseGenerator, self).__init__(*args, **kwargs)
        self.objectType = 'generator'
        self.compute()

    def compute(self):
        self._output = {'out': self.result.tolist(), }
        self.set_output()

    def save(self, *args, **kwargs):
        self.compute()
        super(TSModelMathBaseGenerator, self).save()

    class Meta:
        verbose_name = 'Base'
        verbose_name_plural = 'Base'


class TSModelMathBaseGenerator(general_models.TSModelObject):
    a = models.FloatField(default=0, )
    b = models.FloatField(default=1, )
    rows = models.IntegerField(default=10, )
    columns = models.IntegerField(default=1, )
    result = np.ndarray((0,))

    def __init__(self, *args, **kwargs):
        super(TSModelMathBaseGenerator, self).__init__(*args, **kwargs)
        self.objectType = 'generator'
        self.compute()

    def compute(self):
        try:
            index = ['col{}'.format(i) for i in range(self.columns)]
            data = [self._compute() for i in range(self.columns)]
            self.output = pd.DataFrame(data=data, index=index).T
            self.set_output()
        except Exception as e:
            print('GENERATOR MODEL ERROR! ' + str(e))

    def _compute(self):
        pass

    def save(self, *args, **kwargs):
        self.compute()
        super(TSModelMathBaseGenerator, self).save()

    class Meta:
        verbose_name = 'Generator/Base'
        verbose_name_plural = 'Generator/Base'


class TSModelMathDerivative(TSModelMathBase):
    pass


class TSModelMathDistFit():
    pass


class TSModelMathIntegral(TSModelMathBase):
    pass


class TSModelMathLinear(TSModelMathBase):
    m = models.FloatField(default=1, verbose_name='Slope', )
    b = models.FloatField(default=0, verbose_name='Intercept', )

    def compute(self):
        print('cuchi')
        self.result = self.m * self.x + self.b
        super(self.__class__, self).compute()

    class Meta:
        verbose_name = 'Linear'
        verbose_name_plural = 'Linear'


class TSModelMathRandomNormal(TSModelMathGeneratorBase):
    mean = models.FloatField(default=0, )
    standardDeviation = models.FloatField(default=1, )

    def _compute(self):
        return np.random.normal(loc=self.mean, scale=self.standardDeviation, size=self.rows, )

    class Meta:
        verbose_name = 'Generator/RandomNormal'
        verbose_name_plural = 'Generator/RandomNormal'


class TSModelMathRandomNormal(TSModelMathBaseGenerator):
    def __init__(self, *args, **kwargs):
        self._meta.get_field('a').verbose_name = 'Mean'
        self._meta.get_field('b').verbose_name = 'Standard Deviation'
        super(TSModelMathRandomNormal, self).__init__(*args, **kwargs)

    def _compute(self):
        return np.random.normal(loc=self.a, scale=self.b, size=self.rows, )

    def compute(self):
        self.result = np.random.normal(loc=self.a, scale=self.b, size=self.steps, )
        super(TSModelMathRandomNormal, self).compute()


class TSModelMathRandomUniform(TSModelMathBaseGenerator):
    def __init__(self, *args, **kwargs):
        self._meta.get_field('a').verbose_name = 'Minimum'
        self._meta.get_field('b').verbose_name = 'Maximum'
        super(TSModelMathRandomUniform, self).__init__(*args, **kwargs)

    def _compute(self):
        return np.random.uniform(low=self.a, high=self.b, size=self.rows, )


class TSModelMathRandomUniform(TSModelMathGeneratorBase):
    minimum = models.FloatField(default=0, )
    maximum = models.FloatField(default=1, )

    def _compute(self):
        return np.random.uniform(low=self.minimum, high=self.maximum, size=self.rows, )


class TSModelMathRandomUniform(TSModelMathBaseGenerator):
    def __init__(self, *args, **kwargs):
        self._meta.get_field('a').verbose_name = 'Minimum'
        self._meta.get_field('b').verbose_name = 'Maximum'
        super(TSModelMathRandomUniform, self).__init__(*args, **kwargs)

    def _compute(self):
        return np.random.uniform(low=self.a, high=self.b, size=self.rows, )

    def compute(self):
        self.result = np.random.uniform(low=self.a, high=self.b, size=self.steps, )
        super(TSModelMathRandomUniform, self).compute()

    class Meta:
        verbose_name = 'Generator/RandomUniform'
        verbose_name_plural = 'Generator/RandomUniform'


class TSModelMathRange(TSModelMathBaseGenerator):
    def __init__(self, *args, **kwargs):
        self._meta.get_field('a').verbose_name = 'Minimum'
        self._meta.get_field('b').verbose_name = 'Maximum'
        super(TSModelMathRange, self).__init__(*args, **kwargs)

    def _compute(self):
        return np.linspace(self.a, self.b, self.rows)

    def compute(self):
        self.result = np.linspace(self.a, self.b, self.steps)
        super(TSModelMathRange, self).compute()

    class Meta:
        verbose_name = 'Generator/Range'
        verbose_name_plural = 'Generator/Range'


class TSModelMathRange(TSModelMathGeneratorBase):
    minimum = models.FloatField(default=0, )
    maximum = models.FloatField(default=1, )

    def _compute(self):
        return np.linspace(start=self.minimum, stop=self.maximum, num=self.rows, )


class TSModelMathSine(TSModelMathBase):
    a = models.FloatField(default=1, verbose_name='Amplitude', )
    f = models.FloatField(default=1, verbose_name='Frequency', )
    h = models.FloatField(default=0, verbose_name='Horizontal Offset')
    v = models.FloatField(default=0, verbose_name='Vertical Offset')

    def __init__(self, *args, **kwargs):
        super(TSModelMathSine, self).__init__(*args, **kwargs)
        self.lamb = lambda x: self.a * np.sin(self.f * x + self.h) + self.v

    def compute(self):
        super(TSModelMathSine, self).compute()

    def compute(self):
        self.result = self.a * np.sin(self.f * self.x + self.h) + self.v
        super(self.__class__, self).compute()

    class Meta:
        verbose_name = 'Math/Sine'
        verbose_name_plural = 'Math/Sine'


class TSModelMathSine(TSModelMathBaseFunction):
    amplitude = models.FloatField(default=1, )
    frequency = models.FloatField(default=1, )
    horizontalOffset = models.FloatField(default=0, )
    verticalOffset = models.FloatField(default=0, )

    def __init__(self, *args, **kwargs):
        super(TSModelMathSine, self).__init__(*args, **kwargs)
        a, f = self.amplitude, self.frequency
        h, v = self.horizontalOffset, self.verticalOffset
        self.lamb = lambda x: a * np.sin(f * x + h) + v

    def compute(self):
        super(TSModelMathSine, self).compute()

    class Meta:
        verbose_name = 'Math/Sine'
        verbose_name_plural = 'Math/Sine'


class TSModelMathSlopIntercept(TSModelMathBase):
    m = models.FloatField(default=1, verbose_name='Slope', )
    b = models.FloatField(default=0, verbose_name='Intercept', )

    def __init__(self, *args, **kwargs):
        super(TSModelMathSlopIntercept, self).__init__(*args, **kwargs)
        self.lamb = lambda x: self.m * x + self.b

    def compute(self):
        super(TSModelMathSlopIntercept, self).compute()

    class Meta:
        verbose_name = 'Math/Linear'
        verbose_name_plural = 'Math/Linear'


class TSModelMathSlopIntercept(TSModelMathBaseFunction):
    slope = models.FloatField(default=1, verbose_name='Slope', )
    intercept = models.FloatField(default=0, verbose_name='Intercept', )

    def __init__(self, *args, **kwargs):
        super(TSModelMathSlopIntercept, self).__init__(*args, **kwargs)
        m, b = self.slope, self.intercept
        self.lamb = lambda x: m * x + b

    def compute(self):
        super(TSModelMathSlopIntercept, self).compute()

    class Meta:
        verbose_name = 'Math/Linear'
        verbose_name_plural = 'Math/Linear'


class TSModelStatsAnalysis(django.db.models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tracks = models.TextField()

    def __str__(self):
        return str(self.id)

    def __getitem__(self, item):
        return getattr(self, item)

    class Meta:
        verbose_name_plural = 'StatsAnalaysis'


class TSModelMathTranspose(TSModelMathBase):
    pass


objectTypes = {
    'Range':           TSModelMathRange,
    'Random Normal':   TSModelMathRandomNormal,
    'Random Uniform':  TSModelMathRandomUniform,
    'Slope-Intercept': TSModelMathSlopIntercept,
    'Sine':            TSModelMathSine,
}
propertyDict = {
    'Amplitude':         'a',
    'Frequency':         'f',
    'HorizontalOffset':  'h',
    'Intercept':         'b',
    'Minimum':           'a',
    'Maximum':           'b',
    'Mean':              'a',
    'Slope':             'm',
    'StandardDeviation': 'b',
    'VerticalOffset':    'v',
}
