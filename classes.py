import tensorflow as tf
import numpy as np

from train import model_to_tflite
from gdrive import saveModel, saveTFLiteModel
from mates import rmse


class DetectorInterface:
    def train(self, trainData):
      """Train using many sessions"""
      pass

    def trainSession(self, session):
        """Train parameters of the model"""
        pass

    def evaluate(self, session):
        """Evaluate session"""
        pass

class ModelInterface():
  def __init__(self, name, model=None):
    self.model = model
    self.name = name

  def predict(self, data):
    pass

  def evaluate(self, x, y):
    _y = self.predict(x)
    return rmse(y, _y)

  def getModel(self):
    return self.model

  def save(self):
    pass

class BasicModel(ModelInterface):
  prefix=''
  def __init__(self, name, model=None, history=None):

    if model is None:
      global cargar_o_crear_modelo
      model, trained, history =  cargar_o_crear_modelo(f"{self.prefix}-{name}")

    super(BasicModel, self).__init__(name, model)
    self.history=history
    self.trained = history is not None

  def predict(self, data):
    return self.model.predict(np.expand_dims(data, axis=0), batch_size=1)

  def save(self):
    saveModel(self.model, f"{self.prefix}-{self.name}", self.history)

OPTIM_NAMES = ('', 'FLOAT16', 'DYNAMIC', 'INT')

class LiteModel(ModelInterface):
  prefix='LITE'
  def __init__(self, name, model=None, keras_model=None, optim=0):
    if keras_model is not None:
      model = model_to_tflite(keras_model, optim=optim)
    if model is not None:
      self.interpreter = tf.lite.Interpreter(model_content=model)
    else:
      self.interpreter = tf.lite.Interpreter(model_path=f"{prefix}-{name}-{OPTIM_NAMES[optim]}")
    self.interpreter.allocate_tensors()
    super(LiteModel, self).__init__(name, model)

  def predict(self, data):
    tf_data_in = tf.cast(np.expand_dims(data, axis=0), tf.float32)
    #interpreter = tf.lite.Interpreter(model_content=self.)
    #interpreter.allocate_tensors()
    # Get input and output tensors
    in_details = self.interpreter.get_input_details()
    out_details = self.interpreter.get_output_details()
    self.interpreter.set_tensor(in_details[0]['index'], tf_data_in) #np.array(x_val[0], dtype=np.float32))

    self.interpreter.invoke()
    output_data = self.interpreter.get_tensor(out_details[0]['index'])
    return output_data

  def save(self):
    saveTFLiteModel(self.model, f"{self.prefix}-{name}-{OPTIM_NAMES[optim]}")

class TrainModel(BasicModel):
  prefix = "TRAIN"
  def __init__(self, name, cell, nb_units, input_shape, output_dims, output_steps, nb_layers, bidirectional, stateful, lr=0.001, encoderLength=None):
    self.cell = cell
    self.nb_units = nb_units
    self.input_shape = input_shape
    self.output_dims = output_dims
    self.output_steps = output_steps
    self.nb_layers = nb_layers
    self.bidirectional = bidirectional
    self.stateful = stateful
    self.lr = lr
    self.encoderLength = encoderLength

    global cargar_o_crear_modelo
    _model, self.isTrained, _history = cargar_o_crear_modelo(f"{self.prefix}-{name}", nb_units, input_shape, output_dims, output_steps, nb_layers, stateful, bidirectional, cell, True, lr, encoderLength)
    info(f"Model is trained {self.isTrained}")
    if self.isTrained is False:
      weights, _history = readPartialModel(name, _model)
      info(f"Load PARTIAL model and history {_history}")
    if _history is not None:
      info(f"Trained for {_history['epoch']} epochs")
    super(TrainModel, self).__init__(name, _model, _history)

  def train(self, x, y, epochs=10, initialEpoch=None, validation_data=None, patience=2, statsEvery=20):
    lr_step = 0
    if self.history is not None:
      lr_step = self.history['lr_steps']

    cb = [ LRCallback(RNN_LR, RNN_MINIMUM_LR, step = lr_step)]
    self.model, self.history, weights = batchTrain(self.model, x, y, self.name,
                epochs, validation_data=validation_data,
                history=self.history, initialEpoch=initialEpoch, callbacks=cb)
    self.isTrained = True
    self.save()
    return BasicModel(self.name, self._getTrainedKerasModel(weights), self.history)

  def getTrainedModel(self):
    return BasicModel(self.name, self._getTrainedKerasModel(), self.history)

  def _getTrainedKerasModel(self, weights = None, history=None):
    if weights is None:
      weights = self.model.get_weights()
    if history is None:
      history = self.history

    _model,_t,_h = cargar_o_crear_modelo("FAKE", self.nb_units, self.input_shape,
                self.output_dims, self.output_steps, self.nb_layers, self.stateful,
                self.bidirectional, self.cell, False, self.lr, self.encoderLength,
                forceNew=True)
    _model.set_weights(weights)
    return _model


