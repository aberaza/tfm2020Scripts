from tensorflow.keras.models import save_model, load_model
import os
import json

DRIVE_MOUNT_PATH = '/content/drive/My Drive/pfmdata2020'
GDRIVE_ACCESS = True

class GDriveproxy():
  def __init__(self, mountPath):
    self.path = mountPath

  def readFile(self, filename):
    filepath = f"{self.mount}/filename"


def setDrivePath(path):
  DRIVE_MOUNT_PATH = path
def setDriveAccess(access):
  GDRIVE_ACCESS = access==True


'''
    HISTORY
'''

def saveHistory(history, filename):
  fullpath = F'{DRIVE_MOUNT_PATH}/{filename}.json'
  print(f'Saving history {fullpath}')
  with open(fullpath, 'w') as f:
    json.dump(history, f)
  return None

def loadHistory(filename):
  fullpath = F'{DRIVE_MOUNT_PATH}/{filename}.json'
  h = None
  if GDRIVE_ACCESS and os.path.isfile(fullpath) :
    with open(fullpath, 'r') as f:
      h = json.loads(f.read())
  return h

'''
    TF MODELS
'''

def saveModel(model, filename, history=None, include_optimizer=True):
  if not GDRIVE_ACCESS:
    print("No access to GDRIVE, cannot save")
    return None
  fullpath = F'{DRIVE_MOUNT_PATH}/{filename}'
  print(f'Saving {fullpath}...')
  save_model(model, fullpath, overwrite = True)

  if history is not None:
    saveHistory(history, filename)
  return None

def readModel(filename):
  fullpath = F'{DRIVE_MOUNT_PATH}/{filename}'

  if GDRIVE_ACCESS and os.path.exists(fullpath) :
    print(f"Cargando modelo {fullpath}...")
    return load_model(fullpath), loadHistory(filename)
  else :
    return None, None

import pathlib

def saveTFLiteModel(model, filename):
  tflite_dir = pathlib.Path(F'{DRIVE_MOUNT_PATH}/tflite/')
  tflite_models_dir.mkdir(exist_ok=True, parents=True)
  model_file = tflite_dir/F"{filename}.tflite"
  model_file.write_bytes(model)
  return None

def getTFLiteInterpeter(filename=None, model=None):
  interpreter = None
  if filename is not None:
    model_file = F"{DRIVE_MOUNT_PATH}/tflite/{filename}"
    if os.path.exists(model_file):
      interpreter = tf.lite.Interpreter(model_path=str(model_file))
  elif model is not None:
    interpreter = tf.lite.Interpreter(model_content=model)

  return interpreter

def savePlot(name, fig=None, figsize=(6,4)):
  pathname = f"{PLOTS_PATH}/{name}"
  print(f"Save plot to {pathname}")

  import matplotlib.pyplot as plt
  params = {"pgf.texsystem": "pdflatex",
    "figure.figsize": figsize,
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
         ]
  }
  plt.rcParams.update(params)

  # alternativa bbox_inches=extent1

  ((fig,plt)[fig is None]).savefig(pathname, format="pgf", bbox_inches='tight', dpi=72)


'''
    PARTIALS?
'''
def savePartialModel(model, filename, epoch=0, history=None) :
  if not GDRIVE_ACCESS:
    print("No access to GDRIVE, cannot save")
    return None

  fullpath = F"{DRIVE_MOUNT_PATH}/PARTIAL-{filename}"
  model.save_weights(fullpath, overwrite = True)
  if history is not None:
    saveHistory(history, f"PARTIAL-{filename}")
  return None

def readPartialModel(filename, model) :
  fullpath = F"{DRIVE_MOUNT_PATH}/PARTIAL-{filename}"
  histpath = f"PARTIAL-{filename}"
  print(f"loading: {fullpath}  \n and {histpath}")
  if GDRIVE_ACCESS and os.path.exists(f"{fullpath}.index"):
    print(f"Cargando {fullpath}")
    return model.load_weights(fullpath), loadHistory(histpath)
  else:
    print(f"{fullpath} no existe, empezando con modelo nuevo")
    return model, None
