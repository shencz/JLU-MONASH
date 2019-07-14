# coding=utf-8
from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data()
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Max train sequence length: {}'.format(
    np.max(list(map(len, x_train)))))
print('Max test sequence length: {}'.format(
    np.max(list(map(len, x_test)))))

word2id = imdb.get_word_index()
max_features = len(word2id)

x_all = x_train.tolist() + x_test.tolist()
y_all = y_train.tolist() + y_test.tolist()

x = []
y = []
for num, item in enumerate(x_all):
    if len(item) <= 400:
        x.append(item)
        y.append(y_all[num])
maxlen = max(len(i) for i in x)
print(maxlen)


embedding_dims = 200
batch_size = 32
epochs = 5

x = sequence.pad_sequences(x, maxlen=maxlen, padding='post')
y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(type(x_train), type(x_test), type(y_train), type(y_test))
# save training and test data for predicting testing data and other training methods
np.savetxt('x_train_ft.txt', x_train)
np.savetxt('y_train_ft.txt', y_train)
np.savetxt('x_test_ft.txt', x_test)
np.savetxt('y_test_ft.txt', y_test)

print('Build model...')
model = Sequential()

model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))

model.add(GlobalAveragePooling1D())

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

model.summary()
score = model.evaluate(x_test, y_test, batch_size=32)
print(score)
model_json = model.to_json()
with open("fasttext_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("fasttext_model.hdf5")
