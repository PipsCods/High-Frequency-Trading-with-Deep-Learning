"""
PACKAGES
"""
import numpy as np
import pandas as pd

"""
Financial tools
"""
def sharpe_ratio(x, annualization: bool = True):
  if annualization:
    return np.round(np.sqrt(12) * x.mean(0) / x.std(0), 2)
  else:
      return np.round(x.mean(0) / x.std(0), 2)