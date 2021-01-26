import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np

from gdrive import savePlot

sns.set()
matplotlib.rcParams['figure.figsize'] = [17, 12]

LS = {
    'modulo':'k-',
    'X':'r--',
    'Y':'g--',
    'Z':'b--'
}

def plot_compare_many(series, names, xLabel='x', yLabel='y', xScale='linear', yScale='linear', indexes=None, cols=1, rows=1, saveFile=None):
  fig, axes = plt.subplots(rows, cols)
  color = iter(plt.cm.rainbow(np.linspace(0, 1, len(names))))
  legend = [];

  print(axes)
  print("yScale {}".format(yScale))
  #axes.label_outer()

  print(axes)
  print(isinstance(axes, (np.ndarray, np.generic)))

  #axesList = ( [axes], axes.flatten() ) [isinstance(axes, (np.ndarray, np.generic) ) ]

  axesList = axes.flatten() if isinstance(axes, (np.ndarray, np.generic)) else [axes]
  print(axesList)
  graph_idx = 0

  for serie, name in zip(series, names):
    graph_color = next(color)

    axe = axesList[graph_idx % len(axesList)]
    graph_idx+=1

    axe.set_xscale(xScale)
    axe.set_yscale(yScale)

    axe.set_xlabel(xLabel)
    axe.set_ylabel(yLabel)
    axe.set_title(name)


    if indexes is None :
      axe.plot(serie, color=graph_color, label=name)
    else:
      axe.plot(indexes, serie, color=graph_color, label=name)
      legend.append(name)

    if saveFile is not None:
      savePlot(saveFile, plt, (8, 6))
    else:
      plt.show()
  '''
    if indexes is None :
      plt.plot(serie, color=graph_color)
    else:
      plt.plot(indexes, serie, color=graph_color)

    plt.xscale=xScale
    plt.yscale=yScale

    legend.append(name)

  plt.ylabel(yLabel)
  plt.xlabel(xLabel)
  plt.legend(legend)
  plt.show()
  '''

def plot_separate_many(series, names, xLabel='x', yLabel='y', xScale='linear', yScale='linear', indexes=None, separate=True, saveFile=None):
  fig, axes = plt.subplots(len(series), sharex=True)
  color = iter(plt.cm.rainbow(np.linspace(0, 1, len(names))))
  legend = [];

  print("yScale {}".format(yScale))

  for serie, name, axe in zip(series, names, axes):
    graph_color = next(color)

    if indexes is None :
      axe.plot(serie, color=graph_color)
    else:
      axe.plot(indexes, serie, color=graph_color)
    legend.append(name)

    axe.set_xscale(xScale)
    axe.set_yscale(yScale)

    axe.set(xlabel=xLabel, ylabel=yLabel)
    axe.set_title(name)
    axe.label_outer()

  #for axe in axes.flat:
  #  axe.label_outer()
  if saveFile is not None:
    savePlot(saveFile, plt, (8, 6))
  else:
    plt.show()


def df_plot(df, useIndex=False, separate=False, applyFunc=None, xLabel='x', yLabel='y', xScale='linear', yScale='linear', indexes=None):
  series=[]
  names=[]
  idxName=df.index.name
  for col in df.columns.values:
    if col != idxName :
      names.append(col)
      p_col = df[col].to_numpy()

      if applyFunc != None :
        series.append(applyFunc(p_col))
      else :
        series.append(p_col)
  if useIndex :
    indexes = df[idxName]

  if separate == True :
    plot_separate_many(series, names, xLabel, yLabel, xScale, yScale, indexes)
  else :
    plot_compare_many(series, names, xLabel, yLabel, xScale, yScale, indexes)


def plotSeries(df):
  #df.modulo.plot(linewidth=2.0, label="Modulo")
  df[['modulo','X', 'Y', 'Z']].plot(subplots=True, style=LS)
  plt.xlabel("Tiempo(segundos)")
  plt.ylabel("Aceleración")
  plt.show()

def plotAutocor(df):
  plt.psd(df.modulo, linewidth=2.0, label="Modulo", linestyle=LS['modulo'])
  plt.psd(df.X, label="X", linestyle=LS['X'])
  plt.psd(df.Y, label="Y", linestyle=LS['Y'])
  plt.psd(df.Z, label="Z", linestyle=LS['Z'])
  plt.show()

def plotFFT(df, length, zoom=2, samplingRate=50):
  df_plot(df, applyFunc=lambda x : np.fft.rfft(x,norm="ortho")[0:length//zoom],
          xLabel='Freq (Hz)', yLabel='Power', yScale='log',
          indexes=np.fft.rfftfreq(length, d=1./samplingRate)[0:length//zoom])
