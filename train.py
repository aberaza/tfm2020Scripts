import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, CallbackList
from gdrive import saveHistory, loadHistory, saveModel, savePartialModel, readModel, readPartialModel
from gdrive import savePlot, saveTFLiteModel


def createHistory(keys, history=None):
    if history is None:
      history = dict()
      history['last_step'] = 0
      history['lr_steps'] = 0
      history['epoch'] = 0
    for key in keys:
      history[key] = []
    return history

def populateHistory(history, newItems):
  #add new items
  for key, value in newItems.items():
    if key not in history:
      print("add key ", key)
      history[key] = []
    history[key].append(value)
  history['last_step'] += 1
  return history

# Define some callbacks
class ResetStatesCallback(Callback):
  def __init__(self, maxLen):
    self.counter = 0
    self.maxLen = maxLen

  def on_batch_begin(self, batch, logs={}):
    if self.counter % self.maxLen == 0:
      self.model.reset_states()
      self.counter = 0 #prevent overflow
    self.counter +=1

class InfoCallback(Callback):
  epoch = -1

  def __init__(self, every=20, batch_size = 0):
    super(InfoCallback, self).__init__()
    self.every=every
    self.batch_size = batch_size

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch = epoch

  def on_epoch_end(self, epoch, logs=None):
    if logs is not None :
      _evalLogs = {(key,value) for key,value in logs.items() if key.startswith("eval")}
      print(f"\tEVAL :: {[f'{key}={value};' for key, value in _evalLogs]}")

  def on_train_batch_end(self, batch, logs=None):
    if logs is not None and batch%self.every == 0:
      print(f"\rEPOCH {self.epoch} :: Batch {batch}/{self.batch_size} :: {[(key,value) for key,value in logs.items()]}", end='')

class SaveCallback(Callback):
  loss = None
  weights = None
  wait = 0
  lr = None

  def __init__(self, name, patience, history=None, restoreBestWeights=True):
    super(SaveCallback,self).__init__()
    self.name = name
    self.patience = patience
    self.history = history
    self.restoreBestWeights = restoreBestWeights

  def on_train_begin(self, logs=None):
    #self.lr = float(self.model.optimizer.lr.numpy())
    self.lr = float(K.get_value(self.model.optimizer.lr))

  def on_epoch_begin(self, epoch, logs=None):
    elr = float(K.get_value(self.model.optimizer.lr))#float(self.model.optimizer.lr.numpy())
    self.history['lr'].append(elr)
    if elr != self.lr:
      self.lr = elr
      steps = (self.history['lr_steps'], 0)[self.history['lr_steps'] is None]
      print("steps", steps)
      self.history['lr_steps'] = steps + 1

  def on_epoch_end(self, epoch, logs=None):
    eval_loss = logs['eval-loss']
    self.history = populateHistory(self.history, logs)
    self.history['epoch'] = epoch

    if self.loss is None or eval_loss <= self.loss:
      self.loss = eval_loss
      savePartialModel(self.model, self.name, epoch, self.history)
      self.weights = self.model.get_weights()
      self.wait = 0
    else:
      self.wait+=1
      if self.wait >= self.patience :
        print(f"\rpatience superseeded: stopping...", end='')
        self.model.stop_training = True
        print(f"\t restore best weights...")
        self.model.set_weights(self.weights)


    def on_train_end(self, logs=None):
      # save only best weights
      if self.restoreBestWeights and self.wait > 0 :
        self.model.set_weights(self.weights)

class LRCallback(Callback):
  best_loss = None
  last_loss = None

  def __init__(self, lrstart, lrfinal, step=0, lr_history=[], start_epoch=None, end_epoch = None):
    super(LRCallback, self).__init__()
    self.step = step
    self.lr_history = lr_history
    self.lrstart = lrstart
    self.lrfinal = lrfinal
    self.start_epoch = start_epoch
    self.end_epoch = end_epoch


  def on_train_begin(self, logs=None):
    lr = updateLR(self.model, initialLr=self.lrstart, objectiveLr=self.lrfinal, step=self.step)
    print(f"LR {lr}")

  def on_epoch_end(self, epoch, logs=None):
    if self.start_epoch is not None and epoch < self.start_epoch:
      return
    if self.end_epoch is not None and epoch > self.end_epoch:
      return

    eval_loss = logs['eval-loss']
    if self.last_loss is None or eval_loss <= self.last_loss:
      self.last_loss = eval_loss
    else:
      self.step += 1
      lr = updateLR(self.model, initialLr=self.lrstart, objectiveLr=self.lrfinal, step=self.step)
      print(f"LR update = {lr}")


def updateLR(model, newLR=None, initialLr=0.001, objectiveLr=0.00000001, step=0):
  currentLR = float(K.get_value(model.optimizer.lr)) #model.optimizer.lr.numpy()
  MaxSteps = 20
  # modelo exponencial sobre 20 steps
  '''
    newLR = lr ** (step/MaxSteps)
  '''
  newLR = currentLR
  # psuedo polynomial decay
  if step <= MaxSteps and step > 0:
    newLR = (initialLr - objectiveLr)*(1-step/MaxSteps)**3 + objectiveLr
    print(f"updateLR ({step}) = {newLR}")
    #model.optimizer.lr = tf.Variable(newLR, dtype=tf.float32)
    K.set_value(model.optimizer.lr, K.get_value(newLR))
  return newLR


def batchTrain(model, x, y, name, epochs, validation_split=0.15, validation_data=None, history=None, initialEpoch=None, patience=5, statsEvery=20, callbacks=[], restoreBestWeights=True):
    best_loss, best_weights = None, None
    wait_step = 0
    #initial epoch overrides history epoch?
    initialEpoch = (initialEpoch,0)[initialEpoch is None]
    if history is None:
      metric_labels = model.metrics_names
      history = createHistory(metric_labels + ['lr'])
    else:
      initialEpoch = history['epoch'] + 1
      #updateLR(model, history['lr_steps'])


    cbl = CallbackList([
      InfoCallback(statsEvery,x.shape[0]),
      SaveCallback(name, patience, history, restoreBestWeights)
      ]+callbacks, model=model)

    if x.ndim == 3:
      x=np.expand_dims(x, axis=1)
      y=np.expand_dims(y, axis=1)
    if validation_data is not None:
      x_test, y_test = validation_data
      if x_test.ndim == 3:
        x_test = np.expand_dims(x_test, axis=1)
        y_test = np.expand_dims(y_test, axis=1)
    else:
      x, x_test, y, y_test = train_test_split(x, y, test_size=validation_split)

    cbl.on_train_begin()

    for epoch in range(initialEpoch, epochs):
      cbl.on_epoch_begin(epoch)
      train_metrics = None
      eval_metrics = None
      for _x, _y, batch in zip(x, y, range(x.shape[0])):
        cbl.on_train_batch_begin(batch)
        train_metrics = model.train_on_batch(_x,_y, reset_metrics = False, return_dict = True)
        cbl.on_train_batch_end(batch, train_metrics)
        '''
        if batch % statsEvery == 0:
          print(f"\rEPOCH {epoch} :: Batch {batch}/{x.shape[0]} :: {[(key,value) for key,value in train_metrics.items()]}", end='')
        '''
      for _x, _y in zip(x_test, y_test):
        eval_metrics = model.test_on_batch(_x, _y, reset_metrics = False, return_dict = True)
      '''
      info(f"\tEVAL {batch} :: {[(f'eval-{key}',value) for key, value in eval_metrics.items()]}")
      '''
      for key,value in  eval_metrics.items():
        train_metrics[f"eval-{key}"] = value

      cbl.on_epoch_end(epoch, train_metrics)
      '''
      history = populateHistory(history, train_metrics)
      history['epoch'] = epoch
      history['lr'].append(float(model.optimizer.lr.numpy()))
      eval_loss = train_metrics['eval-loss']
      '''

      ### Borrar o usar lo mismo sin .metric si no funciona
      model.reset_metrics()

      '''
      # Guardar los mejores weights
      if best_loss is None or eval_loss <= best_loss:
        best_loss = eval_loss
        savePartialModel(model, name, epoch, history)
        best_weights = model.get_weights()
        wait_step = 0
      else:
        wait_step+=1
        if wait_step >= patience :
          info(f"patience superseeded... stopping")
          break
        lr_steps = len(set(history['lr']))
        updateLR(model, step=lr_steps)
        history['lr_steps']=lr_steps
      '''
    cbl.on_train_end()

    return (model, history, best_weights)


'''
OPTIM:
None -> Conversión a TFLite sin ningún cambio
1-> Float16 quantization.Float16 quantization reduces the model size by quantizing the model’s weight parameters to float16 bit floating-point numbers for a minimal impact on accuracy and latency. This quantization technique significantly reduces the model size by half.
Let’s add float16 quantization of weights while convert model into TensorFlow Lite. First set the optimizations flag to default optimizations that quantize all fixed parameters such as weights. Then specify float16 is the supported type on the target platform:
2-> Dynamic range quantization:The post-training dynamic range quantization converting model weights to 8-bit precision during model conversation from TensorFlow graphdefs to TensorFlow Lite format. Dynamic range quantization enables 4x reduction in the model size.
The model’s activation outputs are always stored in floating-point. In dynamic range quantization, the weight parameters are quantized post-training and activation are quantized dynamically at inference.
3-> nteger quantization. Microcontroller devices, Edge TPU performs an integer-based operation. So above generated TFLite model won’t compatible with integer-only hardware. To execute the TensorFlow model on integer-only hardware, we need to quantize all model parameters, input and output tensor to an integer.
'''

QUANT_NONE    = None
QUANT_FLOAT16 = 1
QUANT_DYNAMIC = 2
QUANT_INT8    = 3
def model_to_tflite(model, optim=None, dataset=None):
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  if optim is not None:
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
  if optim == 1: # caso OPTIM_FLOAT_W_FLOAT_A y no dataset
    converter.target_spec.supported_types = [tf.float16]
  elif optim == 2: # caso optim = 0 y no dataset
    None
  elif optim == 3:
    rep_data = tf.data.Dataset.from_tensor_slices(tf.cast(dataset, tf.float32)).batch(1)
    def rep_dataset_gen():
      for input in rep_data.take(200):
        yield [input]
    converter.representative_dataset = rep_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32
  elif optim == 4:
    rep_data = tf.data.Dataset.from_tensor_slices(tf.cast(dataset, tf.float32)).batch(1)
    def rep_dataset_gen():
      for input in rep_data.take(200):
        yield [input]
    converter.representative_dataset = rep_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]

  tflite_model = converter.convert()
  # Save to disk
  #open("cifar_quant_8bit.tflite", "wb").write(tflite_model)
  return tflite_model
  