# coding=utf-8
from __future__ import print_function
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences

ngram_range = 3
embedding_dims = 128
batch_size = 32
epochs = 7


print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(index_from=3)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(
    np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(
    np.mean(list(map(len, x_test)), dtype=int)))


x_all = x_train.tolist() + x_test.tolist()
y_all = y_train.tolist() + y_test.tolist()

x = []
y = []
for num, item in enumerate(x_all):
    if len(item) <= 400:
        x.append(item)
        y.append(y_all[num])

word2id = imdb.get_word_index()
word2id = {word:index+2 for word, index in word2id.items()}
word2id.update({"<PAD/>":0, "<STA/>":1, "<UNK/>":2})
max_features = len(word2id)
# print(max_features)
id2word = dict([(v, k)for (k, v)in word2id.items()])



print('Adding {}-gram features'.format(ngram_range))
# Create set of unique n-gram from the training set.
ngram_set = set()
# for input_list in x:
#     for i in range(2, ngram_range + 1):
#         set_of_ngram = create_ngram_set(input_list, ngram_value=i)
#         ngram_set.update(set_of_ngram)
for input_list in x:
    set_of_ngram = create_ngram_set(input_list, ngram_value=3)
    ngram_set.update(set_of_ngram)
    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
start_index = max_features + 1
token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
f = open('token_indice3.txt','w')
f.write(str(token_indice))
f.close()
# print(token_indice)
indice_token = {token_indice[k]: k for k in token_indice}
print(type(token_indice))
print(len(token_indice))
print(type(indice_token))
print(len(indice_token))
    # max_features is the highest integer that could be found in the dataset.
max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
x_train = add_ngram(x, token_indice, ngram_range)
    # x_test = add_ngram(x_test, token_indice, ngram_range)
print('Average train sequence length: {}'.format(
    np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(
    np.mean(list(map(len, x_test)), dtype=int)))
#
#
maxlen = max(len(x) for x in x_train)
#
print(maxlen)
#print(x_train[:10])

x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(x_train, y, test_size=0.1, random_state=42)
print(type(x_train), type(x_test), type(y_train), type(y_test))
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
np.savetxt('x_train_nft3.txt', x_train)
np.savetxt('y_train_nft3.txt', y_train)
np.savetxt('x_test_nft3.txt', x_test)
np.savetxt('y_test_nft3.txt', y_test)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))

# we add a GlobalAveragePooling1D, which will average the embeddings
# of all words in the document
model.add(GlobalAveragePooling1D())

# We project onto a single unit output layer, and squash it with a sigmoid:
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
with open("ft_trgram.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("ft_trgram.hdf5")
