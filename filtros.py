import numpy as np
# Filtro Paso Bajo IIR de primer orden
# alpha es a la vez el factor de atenuación y de frecuencia de corte
# la fórmula suele ser yi = (1-alpha)yi-1 + alpha*xi
# con alpha = At/(At + RC) (At es el periodo de muestreo: 1/fmuestreo)
# RC = 1/(2*PI*fcorte)
def lpf(signal, fm=50, fc=1):
    T = 1/fm
    RC = 1/(2*np.pi*fc)
    alpha = T/(T+RC)
    print(f"alpha: {alpha}, T: {T}, RC: {RC}")
    filtered = []
    y = 0
    for x in signal:
      y = (1-alpha)*y + alpha*x
      filtered.append(y)

    return filtered

# Este tipo de filtros también son como un
# enventanado con ventana EWMA
# https://en.wikipedia.org/wiki/EWMA_chart


# plot freqResponse of filter:
from scipy import signal
import matplotlib.pyplot as plt

def plotFreqResponse(A, B, fm):
  w,h = signal.freqz(B,A)
  fn = 0.5*fm # freq nyquist

  plt.plot(fn*w/np.pi, np.abs(h))

  plt.plot(1, 0.5*np.sqrt(2), 'ko')
  plt.axvline(1, color='k')

  plt.xlim(0, fn)
  plt.title('LPF Freq Response')
  plt.xlabel('Freq(Hz)')
  plt.show()



#plotLPF(fc=1)
