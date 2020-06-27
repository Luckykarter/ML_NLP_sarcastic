from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import pickle


class PreProcess:

    def __init__(self,
                 sentences,
                 labels,
                 vocab_size=10000,
                 embedding_dim=16,
                 max_length=100,
                 training_size=1,
                 trunc_type='post',
                 padding_type='post',
                 oov_token='<OOV>',
                 train_epochs=30,
                 model_type='embed'
                 ):
        MODEL_TYPES = ('embed', 'lstm', 'conv', 'gru')
        # self.labels = np.array(labels)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.trunc_type = trunc_type
        self.padding_type = padding_type
        self.model_type = model_type.lower()
        assert self.model_type in MODEL_TYPES, 'Unknown type of model. Possible types:\n{}'.format(
            MODEL_TYPES
        )
        self.model_file = model_type + '_model.h5'
        self.history_file = model_type + '_history'

        # part of sentences that used for training
        training_size = int(training_size * len(sentences))

        # initialize tokenizer OOV - out of vocabulary
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
        # fit once during initialization - tokenize sentences (split into words)
        self.tokenizer.fit_on_texts(sentences[0:training_size])
        self.train_data = self.get_sequences(sentences[0:training_size])
        self.test_data = self.get_sequences(sentences[training_size:])

        print('{} training sentences\n{} validation sentences'.format(
            len(self.train_data), len(self.test_data)
        ))

        self.train_labels = np.array(labels[0:training_size])
        self.test_labels = np.array(labels[training_size:])

        self.reverse_word_index = dict([(value, key) for
                                        (key, value) in self.tokenizer.word_index.items()])

        if os.path.exists(self.model_file):
            print('Use pre-saved model: ', os.path.abspath(self.model_file))
            self.model = tf.keras.models.load_model(self.model_file)
            self.history = None
            if os.path.exists(self.history_file):
                self.history = pickle.load(open(self.history_file, 'rb'))
        else:
            self.get_keras_model()
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

        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size,
                                      self.embedding_dim,
                                      input_length=self.max_length)
        ])

        if self.model_type == 'embed':
            self.model.add(tf.keras.layers.GlobalAveragePooling1D())
        elif self.model_type == 'gru':
            self.model.add(tf.keras.layers.GRU(32))
        elif self.model_type == 'lstm':
            self.model.add(tf.keras.layers.LSTM(32))
        elif self.model_type == 'conv':
            self.model.add(tf.keras.layers.Conv1D(128, 5, activation=tf.nn.relu))
            self.model.add(tf.keras.layers.GlobalAveragePooling1D())

        self.model.add(tf.keras.layers.Dense(24, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train_model(self, num_epochs):
        history = self.model.fit(self.train_data, self.train_labels, epochs=num_epochs,
                                 validation_data=(self.test_data, self.test_labels),
                                 verbose=1)

        self.history = history.history
        self.model.save(self.model_file)
        with open(self.history_file, 'wb') as file:
            pickle.dump(self.history, file)

    def plot_graphs(self):
        if not self.history:
            print('No history for saved model')
            return
        for par in ('accuracy', 'loss'):
            plt.plot(self.history[par])
            plt.plot(self.history['val_' + par])
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
        return self.model.predict(self.get_sequences(sentence))
