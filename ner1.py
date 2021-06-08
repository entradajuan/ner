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

!rm -r ner
!mkdir ner

import csv
import collections
 
ner_tags = collections.Counter()
iob_tags = collections.Counter()

def strip_ner_subcat(tag):
  return tag.split('-')[0]

def iob (ner_list):
  iob_tokens = []
  for idx, token in enumerate(ner_list):
    iob_tags[token] += 1
    if (idx == 0):
      iob_tokens.append("B-" + token)
    elif (ner_list[idx-1] == token):
      iob_tokens.append("I-" + token)
    else:
      iob_tokens.append("B-" + token)

  return iob_tokens

total_sentences = 0
outfiles = []
for idx, file in enumerate(fnames):
  with open(file, 'rb') as content:
    data = content.read().decode('utf-8').strip()
    sentences = data.split("\n\n")
    print(idx, '  --  ', file,'  --  ', len(sentences))
    total_sentences += len(sentences)

    with open("./ner/"+str(idx)+"-"+os.path.basename(file), 'w') as outfile:
      outfiles.append("./ner/"+str(idx)+"-"+os.path.basename(file))
      writer = csv.writer(outfile)
      
      for sentence in sentences:
        tokens = sentence.split('\n')
        words, pos, ner = [], [], []
        
        for tok in tokens:
          t = tok.split('\t')
          words.append(t[0])
          pos.append(t[1])
          ner_tags[t[3]] += 1
          ner.append(strip_ner_subcat(t[3]))

        writer.writerow([" ".join(words),  " ".join(iob(ner)), " ".join(pos)])
      
      