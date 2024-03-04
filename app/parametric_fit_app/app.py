import numpy as np

import configuration
import forms
import lozoya.data
import lozoya.gui
import lozoya.math


class App(lozoya.gui.TSApp):
    def __init__(self, name, root):
        lozoya.gui.TSApp.__init__(self, name, root)
        self.formData = forms.TSFormData(
            callback0=self.set_file,
            callback1=self.set_data,
        )
        self.formFit = forms.TSFormFit(
            callback=self.fit,
        )
        self.plotData = lozoya.gui.TSPlot()
        self.plotResiduals = lozoya.gui.TSPlot()
        self.table = lozoya.gui.TSTable(
            name='table',
            cellText=['a', 'b'],
            rowLabels=['c', 'd'],
            rowColors=['red', 'green'],
            colLabels=['e', 'f'],
        )
        fields = [
            self.formData,
            self.formFit,
            self.plotData,
            self.plotResiduals,
            self.table,
        ]
        self.form = lozoya.gui.TSForm(
            fields=fields
        )
        self.window.setCentralWidget(self.form)

    def set_file(self):
        self.formData.update_file()
        self.data = lozoya.data.read_csv(self.formData.filePath)
        self.formData.update_xy_choices(options=self.data.columns)
        self.plotData.clear()
        self.plotResiduals.clear()
        self.table.clear()

    def set_data(self):
        x, y = self.formData.get_selected_xy()
        self.x = self.data[x]
        self.y = self.data[y]
        self.plotData.plot(self.x, self.y, xlabel=x, ylabel=y)
        self.plotResiduals.clear()

    def fit(self):
        models = lozoya.math.regressionModels
        self.fitCurve, self.fitCurveArgs = lozoya.math.rank_parametric_fits(models=models, x=self.x, y=self.y)
        self.yFit = self.fitCurve(self.x, *self.fitCurveArgs)
        self.residuals = self.y - self.yFit
        self.plotData.plot(x=self.x, y=self.yFit, clear=False)
        self.plotResiduals.plot(x=self.x, y=np.zeros(len(self.x)), clear=False)
        self.plotResiduals.plot(x=self.x, y=self.residuals, clear=False)


App(
    name=configuration.name,
    root=configuration.root,
).exec()
