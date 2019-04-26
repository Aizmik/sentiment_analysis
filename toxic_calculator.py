from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize


def word_feats(words):
    return dict([(word, True) for word in words])


positive_words = []
for word in open('positive_words_ru.txt', 'r', encoding='UTF-8'):
    positive_words.append(word[:len(word) - 1])

negative_words = [] 
for word in open('negative_words_ru.txt', 'r', encoding='UTF-8'):
    negative_words.append(word)


positive_features = [(word_feats(pos), 'pos') for pos in positive_words]
negative_features = [(word_feats(neg), 'neg') for neg in negative_words]

train_set = negative_features + positive_features
classifier = NaiveBayesClassifier.train(train_set)

while(True):
    neg = 0
    pos = 0
    sentence = input()
    sentence = sentence.lower()
    words = word_tokenize(sentence)
    for word in words:
        category = classifier.classify(word_feats(word))
        if category == 'neg':
            neg = neg + 1
        if category == 'pos':
            pos = pos + 1

    print('Positive: ' + str(float(pos)/len(words)))
    print('Negative: ' + str(float(neg)/len(words)))
