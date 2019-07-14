from keras.models import model_from_json
from keras.models import Model

import keras.preprocessing.text as txt
import numpy as np
from collections import defaultdict
import json
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing import sequence
from keras.datasets import imdb
import re
import os

def normalize(word_prob):
    """
    This function normalizes probability of words for clearer visualization
    :param word_prob: list of word probability
    :return: normalized list of word probability
    """
    word_prob = [(e - min(word_prob)) / (max(word_prob) - min(word_prob)) for e in word_prob]
    return word_prob
def softmax(x):
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)

        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))

        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)

    assert x.shape == orig_shape
    return x

word2id = imdb.get_word_index()
word2id = {word:index+2 for word, index in word2id.items()}
word2id.update({"<PAD/>":0, "<STA/>":1, "<UNK/>":2})
max_features = len(word2id)
# print(max_features)
id2word = dict([(v, k)for (k, v)in word2id.items()])

print('Load model')
# load json and create model
json_file = open('fasttext_model.json', 'r')
print("read json")
loaded_model_text = json_file.read()
print("load model from json")
loaded_model = model_from_json(loaded_model_text)
json_file.close()
# load weights into new model
print("load weights into new model")
loaded_model.load_weights('fasttext_model.hdf5')

y = np.loadtxt('y_test_ft.txt')
x = np.loadtxt('x_test_ft.txt')
t = 106

maxlen = 400
# print(x[t],y[t])
text = []
sentence_len = 1
for i in x[t]:
    if i == 0:
        text.append("<PAD/>")
    elif i == 1:
        text.append("<STA/>")
    elif i == 2:
        text.append("<UNK/>")
    else:
        text.append(id2word[i-1])
        sentence_len += 1
# print(text)
# print(sentence_len)

ans = loaded_model.predict_classes(np.array([x[t], ]))
print(ans)
layer = loaded_model.get_layer('dense_1')
weight_Dense_1 , bias_Dense_1= layer.get_weights()

# print("weight_Dense_1:", weight_Dense_1.shape)
# print(type(weight_Dense_1))

embedding_layer_model = Model(inputs=loaded_model.input, outputs=loaded_model.get_layer('embedding_1').output)
embedding_layer_output = embedding_layer_model.predict(np.array([x[t],]))
target_dim = embedding_layer_output
# print("target_dim:", target_dim.shape)
# print(type(target_dim))
# print(target_dim[0])
gcs = []
for num, item in enumerate(target_dim[0]):
    if num < sentence_len:
        gcs.append(np.dot(item,weight_Dense_1).tolist()[0])
# print(gcs)
# print("gcs", np.array(gcs))
prob = normalize(gcs)
# prob = softmax(np.array(gcs))
# print("soft:",prob)
#
# print("normalize:", prob)
if ans == [[1]]:
    pos_gcs = []
    print("this is a positive sample")
    with open("visualizationft.html", "w") as html_file:
        for index, items in enumerate(prob):
            html_file.write(
                '<font style="background: rgba(255, 0, 0, %f)">%s</font>\n' % (items * items,text[index]))
            print(items,text[index])
            pos_gcs.append(items*items)

    print ('Visualization file have been saved in local directory')
    arr_mean = np.mean(pos_gcs)
    print("mean:", arr_mean)
    arr_var = np.var(pos_gcs)
    print("var:", arr_var)
    arr_std = np.std(pos_gcs)
    print("std:", arr_std)


if ans ==[[0]]:
    neg_gcs = []
    print("this is a negative sample")
    with open("visualizationft.html", "w") as html_file:
        for index, items in enumerate(prob):
            html_file.write(
                '<font style="background: rgba(255, 0, 0, %f)">%s</font>\n' % ((1-items) * (1-items),text[index]))
            print(1-items,text[index])
            neg_gcs.append((1-items)*(1-items))

    print ('Visualization file have been saved in local directory')
    arr_mean = np.mean(neg_gcs)
    print("mean:", arr_mean)
    arr_var = np.var(neg_gcs)
    print("var:", arr_var)
    arr_std = np.std(neg_gcs)
    print("std:", arr_std)
print("sigmoid:",loaded_model.predict(np.array([x[t], ])))
# print(sentence_len)
print("predict:",ans)
print("label:", y[t])
