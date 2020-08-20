import numpy as np

def safe_positive(x):
    if np.isnan(x):
        return np.nan
    else:
        return int(x >= 0)