import numpy as np
import tensorflow as tf
from data import get_data
from data import get_vector
from tensorflow import keras
from gensim.models import FastText
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

x, y = get_data()
val_size = 20
input_length = x.shape[0] - val_size

partial_x_train = x[:input_length]
partial_y_train = y[:input_length]
y_val = y[input_length:]
x_val = x[input_length:]


model = keras.models.Sequential()
model.add(Dropout(rate=0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=40,
          batch_size=512,
          validation_data=(x_val, y_val),
          verbose=1)


FT_model = FastText.load(r'models\fasttext.model')
while(True):
    sent = input()
    print(model.predict(np.array([get_vector(sent, FT_model)])))
