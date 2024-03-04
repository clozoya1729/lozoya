import django.db.models
import pandas as pd

import app.data.app_django.tensorstone.models as general_models
import views

scatterDimensionChoices = ((2, 2), (3, 3), (4, 4),)


class TSPlotBase(general_models.TSModelAppSingleInput):
    plot = None

    def compute(self):
        pass

    def refresh_input(self):
        self.compute()
        super(TSPlotBase, self).refresh_input()

    def make_plot(self):
        pass

    def compress_plot(self):
        # TODO
        return self.plot

    def uncompress_plot(self):
        # TODO
        return self.plot

    class Meta:
        verbose_name = 'Plot/Base'
        verbose_name_plural = 'Plot/Base'


class TSPlotHistogram(TSPlotBase):
    def make_plot(self):
        pass


class TSPlotScatter(TSPlotBase):
    dimensions = django.db.models.IntegerField(choices=scatterDimensionChoices, default=2, )
    x = django.db.models.CharField(blank=True, max_length=100, null=True, default='col0', )
    y = django.db.models.CharField(blank=True, max_length=100, null=True, default='col0', )

    def compute(self):
        print('computing plot')
        data = self.get_input()
        plot = views.make_the_scatter_plot_2d(x=data[self.x], y=data[self.y], )
        text_file = open("Output.txt", "w")
        text_file.write(plot)
        text_file.close()
        # self.compress_plot()
        self.output = plot
        self.set_output()

    def compute(self):
        data = self.get_input()
        if isinstance(data, pd.DataFrame):
            plot = views.make_the_scatter_plot_2d(x=data[self.x], y=data[self.y], )
            self.set_output(plot)

    def make_axis_choices(self):
        data = self.get_input()
        choices = []
        for key in data:
            choices.append((key, key))
        return choices

    def refresh_input(self):
        # TODO
        self.set_axis_choices()
        super(TSPlotScatter, self).refresh_input()

    def save(self, *args, **kwargs):
        try:
            self.refresh_input()
            self.compute()
        except Exception as e:
            print('PLOT MODEL ERROR: ' + str(e))
        super(TSPlotScatter, self).save()

    def set_axis_choices(self):
        choices = self.make_axis_choices()
        _choices = [choices[i][0] for i in range(len(choices))]
        for i, axis in enumerate(['x', 'y']):
            self._meta.get_field(axis).choices = choices
            if not getattr(self, axis) or getattr(self, axis) not in _choices:
                setattr(self, axis, choices[i][0])
        super(TSPlotScatter, self).save()

    class Meta:
        verbose_name = 'Plot/Scatter'
        verbose_name_plural = 'Plot/Scatter'


propertyDict = {

}
objectTypes = {
    'Histogram': TSPlotHistogram,
    'Scatter2D': TSPlotScatter,
}
