import re
import pymorphy2
import pandas as pd
from nltk import word_tokenize
from sklearn.utils import shuffle
from gensim.models import FastText


def read_data(path):
    return pd.read_csv(path, error_bad_lines=False,
                       warn_bad_lines=False, header=None, sep=';')


def get_vector(sentence, FT_model):
    sentence = re.findall('[а-яё]+',  str(word_tokenize(sentence.lower())))
    vector = []
    morph = pymorphy2.MorphAnalyzer()

    for word in sentence:
        word = morph.parse(word)[0].normal_form
        try:
            if vector:
                vector += FT_model[word]
                continue

            vector = FT_model[word]

        except(Exception):
            pass

    return vector


def get_data():
    positive = read_data('positive.csv')
    negative = read_data('negative.csv')

    x = []
    y = []

    FT_model = FastText.load(r'models\fasttext.model')

    for tweet in positive[:][3]:
        x.append(get_vector(tweet, FT_model))
        y.append(0)

    for tweet in negative[:][3]:
        x.append(get_vector(tweet, FT_model))
        y.append(1)

    x, y = shuffle(x, y)

    return x, y
