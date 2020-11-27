
#de https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
from pandas import DataFrame
from pandas import concat
from sklearn.model_selection import train_test_split
import numpy as np

from mates import module, comps2module

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def dict2array(dd, use_module=False) :
  '''
  Given a dictionary with keys X, Y, Z return a (len(dd), 3) shaped array
  print(f"dict2array::useMdoule {use_module}")
  '''
  _X = dd['X']
  _Y = dd['Y']
  _Z = dd['Z']
  _l = _X.size
  dimensions = (3, 1)[use_module]
  _acc = np.empty((_l, dimensions))
  for i,x,y,z in zip(range(_l),_X, _Y, _Z):
    _acc[i] = ([x,y,z], [module(x,y,z)])[use_module]

  return np.array(_acc)


def arr2Sequences(arr, start = 0, window_len = 150, stride=50):
  '''
  Given a (Any, 3) array return a matrix with sape
  (len(array)/stride, 150, 3) which is the result of windowing initial array
  with given stride
  '''
  sub_windows = np.array([np.arange(s, s+window_len) for s in range(start, arr.shape[0] - window_len -1, stride)])
  #print(f"Shape: {arr.shape}, size: {arr.size}, sub_windows: {sub_windows.shape}")
  return arr[sub_windows]


def getSequences(data, window_len = 150, stride=50, stateless = True, use_module = False):
  '''
  Given an array of data dictionaries.
  convert to:
   * if stateless: window all lists and concatenate results in one array
   * else: window all lists without merge
  '''
  _h = [ dict2array(d, use_module = use_module) for d in data ]

  seqs = [ arr2Sequences(a, window_len=window_len, stride=stride) for a in _h]
  if stateless == True :
    seqs_linear = [entry for subseq in seqs for entry in subseq]
    return seqs_linear
  else:
    return seqs


def getTrainData(data, X_len=100, Y_len=50, stride=-1, stateless = True, remove_from_x = True, use_module = False, test_split = 0.2):
  """
  data es un vector de vectroes sesión
  """
  stride = (Y_len, stride)[stride >= 0]
  x = getSequences(data, window_len=(X_len + Y_len), stride=stride, use_module = use_module)
  # reshape
  y = [subseq[-Y_len:] for subseq in x ]
  if remove_from_x == True:
    x = [subseq[:-Y_len] for subseq in x]
  x,y = np.array(x).reshape(len(x), x[0].shape[0], x[0].shape[1]), np.array(y).reshape(len(y), y[0].shape[0], y[0].shape[1])
  return train_test_split(x, y, test_size = test_split)




def splitOverlap(array,size,overlap):
  result = []
  while True:
      if len(array) <= size:
          result.append(array)
          return result
      else:
          result.append(array[:size])
          array = array[size-overlap:]
  return result #aunque nunca se llegue


#@title #### convertSession(data)
def convertSession(data):
  session = data['sessionData']

  acc_x = np.array(session['accelerationX'])
  acc_y = np.array(session['accelerationY'])
  acc_z = np.array(session['accelerationZ'])
  acc_module = comps2module(acc_x, acc_y, acc_z)

  # print(session.keys())


  processed_session = {
      "uid" : data['uid'],
      "activity": ('Unknown', session['activity'])[session['activity']=='unknownActivity'],
      "samplingRate": session.get('rate', 0) or data.get('rate', 0),
      "trigger" : session.get('triggerMethod', '') or data.get('triggerMethod', ''),
      "triggerValue" : session.get('triggerAccel', 0) or data.get('triggerAcceleration', 0),
      "sensorResolution": data.get('sensorResolution', 0.00119641), # por omision: valores del Fossil Gen 3
      "sensorMaxRange": data.get('sensorMaxRange', 384.887), # por omision: valores Fossil Gen 3
      "data": {
        "X": acc_x,
        "Y": acc_y,
        "Z" : acc_z,
      },
      #"acceleration_x": acc_x,
      #"acceleration_y": acc_y,
      #"acceleration_z" : acc_z,
      #"acceleration_x_corr" : autocorr(acc_x),
      #"acceleration_y_corr" : autocorr(acc_y),
      #"acceleration_z_corr" : autocorr(acc_z),
      #"acceleration_module": acc_module,
      #"acceleration_module_corr": autocorr(acc_module),
      "corr_x": freq_from_autocorr(acc_x),
      "corr_y": freq_from_autocorr(acc_y),
      "corr_z": freq_from_autocorr(acc_z),
      "corr_mod": freq_from_autocorr(acc_module)
  }

  return processed_session
