# coding=utf-8
from __future__ import print_function


from keras.models import Model
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np

def normalize(word_prob):
    """
    This function normalizes probability of words for clearer visualization
    :param word_prob: list of word probability
    :return: normalized list of word probability
    """
    word_prob = [(e - min(word_prob)) / (max(word_prob) - min(word_prob)) for e in word_prob]
    return word_prob


print('Loading testing data')
y_test = np.loadtxt('y_test_ft.txt')
x_test = np.loadtxt('x_test_ft.txt')

print('Loading model')
json_file = open('fasttext_model.json')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into model
loaded_model.load_weights('fasttext_model.hdf5')
loaded_model.summary()

maxlen = 400

# y = [loaded_model.predict_classes(np.array([x_test[i], ])).tolist()[0] for i in range(len(x_test))]
# y_pred = []
# for i in range(len(y)):
#     y_pred.append(y[i][0])
# y_pred = np.array(y_pred)

mean_r = []
mean_w = []
var_r = []
var_w = []
std_r = []
std_w = []
sigmoid_p_p = []
sigmoid_p_n = []
sigmoid_n_p = []
sigmoid_n_n = []


for t in range(len(x_test)):

    sentence_len = 0
    pad_len = 0
    for i in x_test[t]:
        if i == 0:
            pad_len += 1

        elif i == 1:
            sentence_len +=1

        elif i == 2:
            sentence_len += 1

        else:
            sentence_len += 1



    layer = loaded_model.get_layer('dense_1')
    weight_Dense_1, bias_Dense_1 = layer.get_weights()
    embedding_layer_model = Model(inputs=loaded_model.input, outputs=loaded_model.get_layer('embedding_1').output)
    embedding_layer_output = embedding_layer_model.predict(np.array([x_test[t], ]))
    target_dim = embedding_layer_output
    gcs = []
    for num, item in enumerate(target_dim[0]):
        gcs.append(np.dot(item, weight_Dense_1).tolist()[0])
    # prob = gcs[:sentence_len]
    prob = normalize(gcs)
    ans = loaded_model.predict_classes(np.array([x_test[t], ]))
    if ans == [[1]]:
        if y_test[t] == 1:
            pos_gcs = [i*i for i in prob]
            arr_mean = np.mean(pos_gcs)
            arr_var = np.var(pos_gcs)
            arr_std = np.std(pos_gcs)
            mean_r.append(arr_mean)
            var_r.append(arr_var)
            std_r.append(arr_std)
            sigmoid_p_p.append(loaded_model.predict(np.array([x_test[t], ]))[0][0])
        elif y_test[t] == 0:
            neg_gcs = [i * i for i in prob]
            arr_mean = np.mean(neg_gcs)
            arr_var = np.var(neg_gcs)
            arr_std = np.std(neg_gcs)
            mean_w.append(arr_mean)
            var_w.append(arr_var)
            std_w.append(arr_std)
            sigmoid_p_n.append(loaded_model.predict(np.array([x_test[t], ]))[0][0])
    elif ans == [[0]]:
        if y_test[t] == 0:
            pos_gcs = [(1 - i) * (1 - i) for i in prob]
            arr_mean = np.mean(pos_gcs)
            arr_var = np.var(pos_gcs)
            arr_std = np.std(pos_gcs)
            mean_r.append(arr_mean)
            var_r.append(arr_var)
            std_r.append(arr_std)
            sigmoid_n_n.append(loaded_model.predict(np.array([x_test[t], ]))[0][0])
        elif y_test[t] == 1:
            pos_gcs = [(1 - i) * (1 - i) for i in prob]
            arr_mean = np.mean(pos_gcs)
            arr_var = np.var(pos_gcs)
            arr_std = np.std(pos_gcs)
            mean_w.append(arr_mean)
            var_w.append(arr_var)
            std_w.append(arr_std)
            sigmoid_n_p.append(loaded_model.predict(np.array([x_test[t], ]))[0][0])
print(mean_r)
print(mean_w)
print(len(mean_r))
print(len(mean_w))

plt.figure()
plt.hist(mean_r,bins=20, facecolor="blue", edgecolor="black", alpha = 0.7)
plt.title("mean_right")
plt.savefig('./umean_right.jpg')
plt.figure()
plt.hist(mean_w,bins=20, facecolor="blue", edgecolor="black", alpha = 0.7)
plt.title("mean_wrong")
plt.savefig('./umean_wrong.jpg')
plt.figure()
plt.hist(var_r, bins=20, facecolor="blue", edgecolor="black", alpha = 0.7)
plt.title("var_right")
plt.savefig('./uvar_right.jpg')
plt.figure()
plt.hist(var_w, bins=20, facecolor="blue", edgecolor="black", alpha = 0.7)
plt.title("var_wrong")
plt.savefig('./uvar_wrong.jpg')
plt.figure()
plt.hist(std_r, bins=20, facecolor="blue", edgecolor="black", alpha = 0.7)
plt.title("std_right")
plt.savefig('./ustd_right.jpg')
plt.figure()
plt.hist(std_w, bins=20, facecolor="blue", edgecolor="black", alpha = 0.7)
plt.title("std_wrong")
plt.savefig('./ustd_wrong.jpg')
plt.figure()
plt.hist(sigmoid_p_p, bins=20, facecolor="blue", edgecolor="black", alpha = 0.7)
plt.title("sigmoid_p_p")
plt.savefig('./usigmoid_p_p.jpg')
plt.figure()
plt.hist(sigmoid_n_p, bins=20, facecolor="blue", edgecolor="black", alpha = 0.7)
plt.title("sigmoid_n_p")
plt.savefig('./usigmoid_n_p.jpg')
plt.figure()
plt.hist(sigmoid_n_n, bins=20, facecolor="blue", edgecolor="black", alpha = 0.7)
plt.title("sigmoid_n_n")
plt.savefig('./usigmoid_n_n.jpg')
plt.figure()
plt.hist(sigmoid_p_n, bins=20, facecolor="blue", edgecolor="black", alpha = 0.7)
plt.title("sigmoid_p_n")
plt.savefig('./usigmoid_p_n.jpg')
plt.show()












