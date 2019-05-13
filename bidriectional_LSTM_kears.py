import numpy as np
from data import get_data
from data import get_vector
from gensim.models import FastText
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

x, y = get_data()
val_size = 20
input_length = x.shape[0] - val_size

partial_x_train = x[:input_length]
partial_y_train = y[:input_length]
y_val = y[input_length:]
x_val = x[input_length:]


model = Sequential()
model.add(Embedding(9, 128, input_length=9))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          batch_size=512,
          epochs=40,
          validation_data=[x_val, y_val],
          verbose=1)


FT_model = FastText.load(r'models\fasttext.model')
while(True):
    sent = input()
    print(model.predict(np.array([get_vector(sent, FT_model)])))
