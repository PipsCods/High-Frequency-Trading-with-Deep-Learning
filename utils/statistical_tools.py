"""
PACKAGES
"""
import numpy as np
import pandas as pd

"""
Statistics tools
"""
def normalize(data: np.ndarray,
              ready_normalization: dict = None,
              use_std: bool = False)->tuple:
  """
  Normalizes the data using either standard scaling (z-score) or min-max scaling.

  This function normalizes the input DataFrame using either standard
  scaling or min-max scaling. Standard scaling subtracts the mean and
  divides by the standard deviation of each column, while min-max scaling
  scales the values to a range between 0 and 1.

  The function first checks if the specified normalization method is valid.
  Then, it applies the chosen normalization method to each column of the
  input DataFrame using the respective formula.
  """

  if ready_normalization is None:
      data_std = data.std(0)
      data_mean = data.mean(0)
      if use_std:
        data = (data - data_mean) / data_std # this is z-scoring of the data
      else:
        data_max = np.max(data, axis=0)
        data_min = np.min(data, axis=0)
  else:
      data_std = ready_normalization['std']
      data_mean = ready_normalization['mean']
      if use_std:
        data = (data - data_mean) / data_std # this is z-scoring of the data
      else:
        data_max = ready_normalization['max']
        data_min = ready_normalization['min']
  if not use_std:
    data = data - data_min
    data = data/(data_max - data_min)
    data = data - 0.5
  normalization = {'std': data_std,
                   'mean': data_mean,
                    'max': data_max,
                    'min': data_min}
  return data, normalization

