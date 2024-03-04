from collections import OrderedDict

import lozoya.data as data_api
import regressors


def get_metadata(model, x, y, reference, length, last):
    return OrderedDict(
        [('model', model), ('input', data_api.read(x, length=length)),
         ('output', data_api.read(y, length, length=length)), ('maxLength', length),
         ('reference', reference), ('last', last)]
    )


regressor = regressors.Regressor()
run()
