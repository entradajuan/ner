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

import os
ruta = './gmb-2.2.0/data/'
fnames = []
for root, dirs, files in os.walk(ruta):
  for name in files:
    if name.endswith(".tags"):
            fnames.append(os.path.join(root, name))

print(len(fnames))
print(fnames[0])

!mkdir ner

import csv
import collections
 
ner_tags = collections.Counter()
iob_tags = collections.Counter()


total_sentences = 0
outfiles = []
for idx, file in enumerate(fnames):
  print(idx, '  --  ', file)
  with open(file, 'rb') as content:
    data = content.read().decode('utf-8').strip()
