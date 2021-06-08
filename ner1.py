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
    if (token != 'O'):
      if (idx == 0):
        iob_tokens.append("B-" + token)
        iob_tags["B-" + token] += 1
      elif (ner_list[idx-1] == token):
        iob_tokens.append("I-" + token)
        iob_tags["I-" + token] += 1
      else:
        iob_tokens.append("B-" + token)
        iob_tags["B-" + token] += 1
    else:
      iob_tokens.append(token)
      iob_tags[token] += 1  

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
      
print(total_sentences)

print(ner_tags)
print(len(ner_tags))

print(iob_tags)
print(len(iob_tags))

print(*iob_tags)
print(*iob_tags.items())

import matplotlib.pyplot as plt
labels , values = zip(*iob_tags.items())

print(labels)
indexes = np.arange(len(labels))

plt.bar(indexes, values)
plt.xticks(indexes, labels, rotation='vertical')
plt.margins(0.01)
plt.subplots_adjust(bottom=0.15)
plt.show()

import glob
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

files = glob.glob("./ner/*.tags")

dfs = [pd.read_csv(f, header=None, names=['text', 'label', 'pos']) for f in files] 
df = pd.concat(dfs, ignore_index= True )
print(df.head())
print(df.isna().sum())
print(df.describe())
print(df.info())

text_tok = Tokenizer(filters= '[\\^\t\n]', lower=False, split=' ', oov_token='<OOV>')
pos_tok = Tokenizer(filters= '\t\n', lower=False, split=' ', oov_token='<OOV>')
ner_tok = Tokenizer(filters= '\t\n', lower=False, split=' ', oov_token='<OOV>')

text_tok.fit_on_texts(df['text'])
pos_tok.fit_on_texts(df['pos'])
ner_tok.fit_on_texts(df['label'])

text_config = text_tok.get_config()
ner_config = ner_tok.get_config()

print(text_config['document_count'])
print(ner_config)

text_vocab = eval(text_config['index_word'])
ner_vocab = eval(ner_config['index_word'])

print(len(text_vocab))
print(len(ner_vocab))

x_tok = text_tok.texts_to_sequences(df['text'])
y_tok = ner_tok.texts_to_sequences(df['label'])

print(type(y_tok))
print(x_tok[0])
print(y_tok[0])

print(text_tok.sequences_to_texts([x_tok[0]]), df['text'][0])
print(ner_tok.sequences_to_texts([y_tok[0]]), df['label'][0])

from tensorflow.keras.preprocessing import sequence

max_len = 100
x_pad = sequence.pad_sequences(x_tok, padding='post', maxlen=max_len)
y_pad = sequence.pad_sequences(y_tok, padding='post', maxlen=max_len)

print(type(x_pad))
print(x_pad.shape)
print(y_pad.shape)

print(text_tok.sequences_to_texts([x_pad[0]]))
print(ner_tok.sequences_to_texts([y_pad[0]]))

num_classes =len(ner_vocab) + 1
print(num_classes)

Y = tf.keras.utils.to_categorical(y_pad, num_classes=num_classes)
print(type(Y))
print(Y.shape)
print(Y[0][0])
print(y_pad[0][0])

vocab_size = len(text_vocab) + 1
embedding_dim = 64
rnn_units = 100
BATCH_SIZE = 90

from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense

dropout = 0.2
def build_model(vocab_size, embedding_dim, rnn_units, dropout, batch_size, classes):
  model = tf.keras.Sequential([Embedding(vocab_size, embedding_dim, mask_zero=True, batch_input_shape=[BATCH_SIZE, None] ),
                               Bidirectional(LSTM(units=rnn_units, return_sequences=True, dropout=dropout, kernel_initializer=tf.keras.initializers.he_normal() )),
                               TimeDistributed(Dense(rnn_units, activation= 'relu')),
                               Dense(num_classes, activation='softmax')
                               ])
  return model

model = build_model(vocab_size, embedding_dim, rnn_units, dropout, BATCH_SIZE, num_classes)
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X = x_pad
total_sentences = Y.shape[0]
test_size = round(total_sentences / BATCH_SIZE * 0.2)
print(total_sentences)
print(total_sentences / BATCH_SIZE * 0.2)
print(test_size)

X_train = X[BATCH_SIZE*test_size:]
Y_train = Y[BATCH_SIZE*test_size:]

X_test = X[0:BATCH_SIZE*test_size]
Y_test = Y[0:BATCH_SIZE*test_size]

print(X_train.shape)
print(X_test.shape)
print(X_train.shape[0] + X_test.shape[0])

model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=15)

model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)

y_predict = model.predict(X_test, batch_size=BATCH_SIZE)
print(type(y_predict))

print(y_predict[0])

print(text_tok.sequences_to_texts([X_test[0]]))
print(text_tok.sequences_to_texts([x_pad[0]]))
print(ner_tok.sequences_to_texts([y_pad[0]]))

y_pred = tf.argmax(y_predict, -1)
print(type(y_pred))
print(y_pred)

y_pnp = y_pred.numpy()
ner_tok.sequences_to_texts([y_pnp[0]])

