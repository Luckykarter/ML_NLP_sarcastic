from resource.getdata import get_datastore
from preprocess import PreProcess

"""
This is an example of binary neural network that
recognizes sentiment of the sentence - if the sentence
is sarcastic or not. Input data is in format of two arrays
that contain sentences and labels.
Sentences then can be split between training and validation
data using parameter training_size of PreProcess class
"""

sentences = []
labels = []

datastore = get_datastore()
# split JSON into lists
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

p = PreProcess(sentences, labels,
               vocab_size=10000,
               embedding_dim=16,
               max_length=100,
               trunc_type='post',
               padding_type='post',
               oov_token='<OOV>',
               training_size=0.5,
               train_epochs=30)

p.plot_graphs()
p.save_vectors()

my_sentences = ["granny starting to fear spiders in the garden might be real",
                "mom starting to fear son's web series closest thing she will have to grandchild",
                "boehner just wants wife to listen, not come up with alternative debt-reduction ideas",
                "game of thrones season finale showing this sunday night",
                "Light travels faster than sound. This is why some people appear bright until they speak.",
                "Hello world",
                "You look good when your eyes are closed, but you look the best when my eyes closed.",
                "jango is a high-level Python Web framework that encourages rapid development and clean design."]

predictions = p.predict(my_sentences)

for i in range(len(my_sentences)):
    if predictions[i][0] > 0.5:
        label = 'Sarcastic:\t'
    else:
        label = 'Normal:\t'
    print(label, my_sentences[i])
