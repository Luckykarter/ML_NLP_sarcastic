from resource.getdata import get_datastore
from preprocess import PreProcess


sentences = []
labels = []
urls = []

datastore = get_datastore()
# split JSON into lists
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

p = PreProcess(sentences, labels,
               vocab_size=10000,
               embedding_dim=16,
               max_length=100,
               trunc_type='post',
               padding_type='post',
               oov_token='<OOV>',
               training_size=20000,
               train_epochs=30)

p.plot_graphs()
p.save_vectors()

my_sentence = ["granny starting to fear spiders in the garden might be real",
               "game of thrones season finale showing this sunday night"]

p.predict(my_sentence)