import os

import app.data.app_django.tensorstone.configuration as ts_general_configuration
import configuration
import forms
import lozoya.file


def get_replacement_dict(n, plot, axSelect, dimSelect):
    return {
        '%n%':           str(n),
        '%plot%':        str(plot),
        '%axes%':        str(axSelect),
        '%dimensions%':  str(dimSelect),
        '%axID%':        configuration.axArea.format(n),
        '%dimArea%':     configuration.dimArea.format(n),
        '%plotID%':      configuration.plot.format(n),
        '%accordionID%': configuration.accordion.format(n),
    }


def get_plot_template(n, columns, plot):
    dimSelect = forms.TSFormDimensionSelect(n)
    axSelect = forms.TSFormAxesSelect(n, 2, columns)
    plotTemplatePath = os.path.join(
        ts_general_configuration.templatesPath,
        ts_general_configuration.analysisPlot
    )
    with open(plotTemplatePath, 'r') as f:
        t = lozoya.file_api.clean_file_contents(f.read())
    r = get_replacement_dict(n, plot, axSelect, dimSelect)
    for key in r:
        t = t.replace(key, r[key])
    return t
