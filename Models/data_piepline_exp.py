#Copyright (c) 2020 Hanwei Wu
def time_slicing(t_series, lag, forca_lag):
   history = [t_series[i:i+lag] for i in range(len(t_series)-lag-forca_lag+1)]
   forecast = t_series[lag+forca_lag-1:]
   return history, forecast
def data_standlization(dataset):
  """standlize a tensor dataset

  Args:
  dataset -- tensors of shape (n, T)
  
  Returns:
  dataset -- tensors of shape (n, T), contains normalized values. 
  """
  means = dataset.mean()
  stds = dataset.std()
  dataset = (dataset - means) / stds
  return dataset

def standlization(dataset):
  means = dataset.mean()
  stds = dataset.std()
  dataset = (dataset - means) / stds
  return dataset, means, stds


def destandlization(dataset, means, stds):
  dataset = dataset*stds
  dataset += means
  return dataset