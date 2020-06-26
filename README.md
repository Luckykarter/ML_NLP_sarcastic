# ML_NLP_sarcastic
Natural language processing of sarcastic dataset from Kaggle
Neural network for sarcasm detection

resource/getdata.py downloads Kaggle sarcastic dataset from here:
https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home

main.py creates object of class PreProcess which includes the following operations:
- splits given data to training data and test data
- tokenize and preprocess training/test data
- creates keras model 
- trains keras model and evaluates it on test data
- saves model as 'keras_model.h5'
- saves given text data as vectors 'vecs.tsv' and meta 'meta.tsv' which can be visualized using:
http://projector.tensorflow.org/

