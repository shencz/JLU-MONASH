# coding=utf-8
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.datasets import imdb
from sklearn.model_selection import train_test_split


# set parameters:  设定参数
# max_features = 5000  # 最大特征数（词汇表大小）
sequence_length = 400# 序列最大长度
batch_size = 16      # 每批数据量大小
embedding_dim = 128  # 词嵌入维度
num_filters = 512    # 1维卷积核个数
filter_sizes = [3,4,5]    # 卷积核长度
nb_epoch = 10        # 迭代次数
drop = 0.5

# 载入 imdb 数据
print('Loading data...')
(X_train, Y_train), (X_test, y_test) = imdb.load_data()
word2id = imdb.get_word_index()
word2id = {word:index+2 for word, index in word2id.items()}
word2id.update({"<PAD/>":0, "<STA/>":1, "<UNK/>":2})
max_features = len(word2id)
print(max_features)
id2word = dict([(v, k)for (k, v)in word2id.items()])
max_features = len(word2id)
print("max_feature:", max_features)
X_all = X_train.tolist() + X_test.tolist()
Y_all = Y_train.tolist() + y_test.tolist()

print(type(X_all))
print(Y_all[:10])
X_sample = []
Y_sample = []
for num, sentence in enumerate(X_all):
    if len(sentence) <= 400:
        X_sample.append(sentence)
        if Y_all[num] == 0:
            Y_sample.append([1, 0])
        elif Y_all[num] == 1:
            Y_sample.append([0, 1])

print(Y_sample[:10])
X_sample_padded = sequence.pad_sequences(X_sample, maxlen=400, padding="post")
Y_sample = np.array(Y_sample)
x_train,x_test, y_train,  y_test = train_test_split(X_sample_padded, Y_sample, test_size=0.1, random_state=42)
print(type(x_train), type(x_test), type(y_train), type(y_test))
# x_train = np.asarray(x_train)
# x_test = np.asarray(x_test)

# save training and test data for predicting testing data and other training methods
np.savetxt('x_train_cnn.txt', x_train)
np.savetxt('y_train_cnn.txt', y_train)
np.savetxt('x_test_cnn.txt', x_test)
np.savetxt('y_test_cnn.txt', y_test)

print(x_train.shape)
print(y_train.shape)



# 构建模型
print('Build model...')
model = Sequential()

print("Creating Model...")

inputs = Input(shape=(sequence_length,), dtype='int32')

embedding = Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=sequence_length)(inputs)

reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal',
                activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal',
                activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal',
                activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

flatten = Flatten()(concatenated_tensor)

dropout = Dropout(drop)(flatten)

output = Dense(units=2, activation='softmax')(dropout)


# create the model
model = Model(inputs=inputs, outputs=output)

# checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)  # optimizer

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

# train the model by fitting the data into the model
print("Training Model")
model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1,
          validation_data=(x_test, y_test))

# display the structure of model
model.summary()

# save the model
print("Save model to disk")
model_json = model.to_json()
with open("CNN_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("CNN_model.hdf5")
