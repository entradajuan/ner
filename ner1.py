import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
tf.__version__

print(tf.__version__)
print(tfds.__version__)

tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    print(e)

!wget https://gmb.let.rug.nl/releases/gmb-2.2.0.zip
!unzip gmb-2.2.0.zip


