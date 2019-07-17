from keras.models import model_from_json
from keras.models import Model

import numpy as np
from keras.datasets import imdb
from collections import defaultdict

def normalize(word_prob):
    """
    This function normalizes probability of words for clearer visualization
    :param word_prob: list of word probability
    :return: normalized list of word probability
    """
    word_prob = [(e - min(word_prob)) / (max(word_prob) - min(word_prob)) for e in word_prob]
    return word_prob

print('Load model')
# load json and create model
json_file = open('ft_bigram.json', 'r')
print("read json")
loaded_model_text = json_file.read()
print("load model from json")
loaded_model = model_from_json(loaded_model_text)
json_file.close()
# load weights into new model
print("load weights into new model")
loaded_model.load_weights('ft_bigram.hdf5')



word2id = imdb.get_word_index()
word2id = {word:index+2 for word, index in word2id.items()}
word2id.update({"<PAD/>":0, "<STA/>":1, "<UNK/>":2})
max_features = len(word2id)
print(max_features)
id2word = dict([(v, k)for (k, v)in word2id.items()])


f = open('token_indice.txt','r')
a = f.read()
token_indice = eval(a)
f.close()
# print(type(token_indice))
# print(len(token_indice))
# print([i for i in token_indice.keys()][:5])
# print([i for i in token_indice.values()][:5])
indice_token = {token_indice[k]: k for k in token_indice}
x = np.loadtxt('x_test_nft.txt')
y = np.loadtxt('y_test_nft.txt')

maxlen = 799
t = 108
# print(x[t])
text = []
sentence_len = 1
pad_len = 0
for i in x[t]:
    if i == 0:
        text.append("<PAD/>")
        pad_len += 1
    elif i == 1:
        text.append("<STA/>")
    elif i == 2:
        text.append("<UNK/>")
        sentence_len +=1
    elif i < 88588:
        text.append(id2word[i-1])
        sentence_len += 1
    else:
        text.append(indice_token[i])

print(text)
ans = loaded_model.predict_classes(np.array([x[t], ]))
print(loaded_model.predict(np.array([x[t], ])))
# print(sentence_len)
print(ans)
print(y[t])


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
    gcs.append(np.dot(item,weight_Dense_1).tolist()[0])
# print(gcs)
# print(len(gcs))
#
# print(pad_len)
prob = []
for i in range(len(gcs)):
    if i > sentence_len-1 and i < 799 - pad_len:
        # print(text[i])
        # print(gcs[i])
        for j in range(len(x[t])):
            if text[i][0] == x[t][j] and text[i][1] == x[t][j+1]:
                # print(x[t][j], x[t][j+1])
                # print(gcs[j], gcs[j+1])
                # print(j)
                gcs[j] = gcs[j] + gcs[i]
                gcs[j+1] = gcs[j+1] + gcs[i]

prob =[i/3 for i in gcs[:sentence_len]]
prob[0] = prob[0] * 3 / 2
prob[-1] = prob[-1] * 3 / 2
# for i in range(len(gcs)):
#     if i > sentence_len-1 and i < 799 - pad_len:
#
#         if text[i][0] == 1:
#             text[i] = (id2word[text[i][0]], id2word[text[i][1]-1])
#         else:
#             text[i] = (id2word[text[i][0]-1], id2word[text[i][1]-1])
#         # print(text[i])
#         # print(gcs[i])



prob = normalize(prob)
# prob = softmax(np.array(gcs))
# print("soft:",prob)
if ans == [[1]]:
    pos_gcs = []
    print("this is a positive sample")
    with open("visualizationftn.html", "w") as html_file:
        for index, items in enumerate(prob):

            html_file.write(
                '<font style="background: rgba(255, 0, 0, %f)">%s</font>\n' % (items * items,text[index]))
            print(items,text[index])
            pos_gcs.append(items * items)
    print ('Visualization file have been saved in local directory')
    arr_mean = np.mean(pos_gcs)
    print("mean:", arr_mean)
    arr_var = np.var(pos_gcs)
    print("var:", arr_var)
    arr_std = np.std(pos_gcs)
    print("std:", arr_std)


if ans == [[0]]:
    neg_gcs = []
    print("this is a negative sample")
    with open("visualizationftn.html", "w") as html_file:
        for index, items in enumerate(prob):
            html_file.write(
                '<font style="background: rgba(255, 0, 0, %f)">%s</font>\n' % ((1-items) * (1-items),text[index]))
            print(1-items,text[index])
            neg_gcs.append((1 - items) * (1 - items))
    print ('Visualization file have been saved in local directory')
    arr_mean = np.mean(neg_gcs)
    print("mean:", arr_mean)
    arr_var = np.var(neg_gcs)
    print("var:", arr_var)
    arr_std = np.std(neg_gcs)
    print("std:", arr_std)

# print(sentence_len)
# print(ans)
# print(y[t])
# arr_mean = np.mean(prob[:sentence_len])
# print(arr_mean)
# arr_var = np.var(prob[:sentence_len])
# print(arr_var)
# arr_std = np.std(prob[:sentence_len])
# print(arr_std)
print("sigmoid:",loaded_model.predict(np.array([x[t], ])))
# print(sentence_len)
print("predict:",ans)
print("label:", y[t])