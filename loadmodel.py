import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

# LOAD MODEL
model2 = tf.keras.models.load_model('path_to_my_model.h5')
print(model2.summary())

# LOAD TOKENIZER
with open('tokenizer.pickle', 'rb') as handle:
    text_tok = pickle.load(handle)

text = ['Rachel Maddow Strikes Multi-Year Deal With MSNBC']

max_len = 100
sentence = text_tok.texts_to_sequences(text)
sentence = sequence.pad_sequences(sentence, padding='post', maxlen=max_len)


y_predict = model2.predict(sentence)
y_pred = tf.argmax(y_predict, -1)
print(type(y_predict))

