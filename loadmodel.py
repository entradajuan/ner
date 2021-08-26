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

with open('ner_tokenizer.pickle', 'rb') as handle:
    ner_tok = pickle.load(handle)

text = ['Juani Lopez goes tomorrow to Bosnia']

max_len = 100
sentence = text_tok.texts_to_sequences(text)
sentence = sequence.pad_sequences(sentence, padding='post', maxlen=max_len)


y_predict = model2.predict(sentence)
y_pred = tf.argmax(y_predict, -1)
print(type(y_predict))
print(y_pred)
y_pnp = y_pred.numpy()
print(text)
ner_tok.sequences_to_texts([y_pnp[0]])

