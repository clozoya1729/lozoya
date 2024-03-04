import numpy as np

funcFams = {
    'Exponential': lambda x, a, b: a * np.exp(b * x),
    'Logarithmic': lambda x, a, b: a * np.log(x) + b,
}
