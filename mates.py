import pandas as pd
import numpy as np
import math

'''
rmse of 2 series
'''
def rmse(y, pred):
  d = np.linalg.norm(y - pred)
  return d / np.sqrt(len(pred))

def module(x, y, z):
  return math.sqrt(x*x+y*y+z*z)

def comps2module(X, Y, Z):
  result = []
  for x,y,z in zip(X,Y,Z):
    result.append(module(x,y,z))
  return np.array(result)

def autocorr(x):
  result = np.correlate(x, x, mode='full')
  return np.array(result[len(result)//2:])

def corr(x,y):
  result = np.correlate(x, y, mode='full')
  return np.array(result[len(result)//2:])

def freq_from_autocorr(x):
    """Estimate frequency using autocorrelation

    Pros: Best method for finding the true fundamental of any repeating wave,
    even with strong harmonics or completely missing fundamental

    Cons: Not as accurate, currently has trouble with finding the true peak

    """
    corr = autocorr(x)


    # Find the first low point
    d = np.diff(corr)
    start = np.where(d>0)[0]

    if len(start)>0 :
      return np.argmax(corr[start[0]:]) + start[0]
    return 0

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    # Also could zero-pad before doing circular autocorrelation.
    #peak = np.argmax(corr[start:])
    #px, py = parabolic(corr, peak)

    #print("minimum at {}, peak at {}".format(start, peak + start))


    #return fs / px
    #return peak + start
