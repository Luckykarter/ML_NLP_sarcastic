from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import io


class PreProcess:
    def __init__(self,
                 sentences,
                 labels,
                 vocab_size=10000,
                 embedding_dim=16,
                 max_length=100,
                 training_size=20000,
                 trunc_type='post',
                 padding_type='post',
                 oov_token='<OOV>',
                 train_epochs=30,
                 model_file='keras_model.h5'
                 ):
        # self.labels = np.array(labels)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.trunc_type = trunc_type
        self.padding_type = padding_type
        self.model_file = model_file

        # initialize tokenizer OOV - out of vocabulary
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
        # fit once during initialization - tokenize sentences (split into words)
        self.tokenizer.fit_on_texts(sentences[0:training_size])
        self.train_data = self.get_sequences(sentences[0:training_size])
        self.test_data = self.get_sequences(sentences[training_size:])

        self.train_labels = np.array(labels[0:training_size])
        self.test_labels = np.array(labels[training_size:])

        self.reverse_word_index = dict([(value, key) for
                                        (key, value) in self.tokenizer.word_index.items()])

        if os.path.exists(model_file):
            self.model = tf.keras.models.load_model(model_file)
            self.history = None
        else:
            self.model = self.get_keras_model()
            self.train_model(train_epochs)

    # return always padded sequences
    def get_sequences(self, sentences):
        # convert letters into numbers
        seq = self.tokenizer.texts_to_sequences(sentences)
        # pad sequences - i.e. make them to be the same length
        padded = pad_sequences(seq, maxlen=self.max_length, padding=self.padding_type,
                               truncating=self.trunc_type)
        return np.array(padded)

    def get_keras_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size,
                                      self.embedding_dim,
                                      input_length=self.max_length),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def train_model(self, num_epochs):
        self.history = self.model.fit(self.train_data, self.train_labels, epochs=num_epochs,
                                      validation_data=(self.test_data, self.test_labels),
                                      verbose=1)
        tf.keras.models.save_model(self.model_file)

    def plot_graphs(self):
        if not self.history:
            print('No history for saved model')
            return
        for par in ('accuracy', 'loss'):
            plt.plot(self.history.history[par])
            plt.plot(self.history.history['val_' + par])
            plt.xlabel('Epochs')
            plt.ylabel(par)
            plt.legend([par, 'val_' + par])
            plt.show()

    def decode_sentence(self, text):
        return ' '.join([self.reverse_word_index.get(i, '?') for i in text])

    def save_vectors(self):
        e = self.model.layers[0]
        weights = e.get_weights()[0]

        out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
        out_m = io.open('meta.tsv', 'w', encoding='utf-8')
        for word_num in range(1, self.vocab_size):
            word = self.reverse_word_index[word_num]
            embeddings = weights[word_num]
            out_m.write(word + '\n')
            out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')
        out_v.close()
        out_m.close()

    def predict(self, sentence):
        print(self.model.predict(self.get_sequences(sentence)))
