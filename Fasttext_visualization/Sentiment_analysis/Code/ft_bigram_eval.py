# coding=utf-8
from __future__ import print_function

from keras.datasets import imdb
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import sequence
from numpy import set_printoptions
np.set_printoptions(threshold=np.inf)


def predict():
    """
    This function loads the corresponding model for specified edit type
    and calculate precision, recall and F1-score for each model
    :param pattern: edit type (string)
    :return: None
    """

    # print('Loading data...')
    # (x_train, y_train), (x_test, y_test) = imdb.load_data()
    # print(len(x_train), 'train sequences')
    # print(len(x_test), 'test sequences')
    # print('Average train sequence length: {}'.format(
    #     np.max(list(map(len, x_train)))))
    # print('Average test sequence length: {}'.format(
    #     np.max(list(map(len, x_test)))))
    #
    # # print(x[0])
    # x_all = x_train.tolist() + x_test.tolist()
    # y_all = y_train.tolist() + y_test.tolist()
    #
    # x = []
    # y = []
    # for num, item in enumerate(x_all):
    #     if len(item) <= 400:
    #         x.append(item)
    #         y.append(y_all[num])
    # maxlen = max(len(i) for i in x)
    # x_test = sequence.pad_sequences(x, maxlen=maxlen, padding='post')
    # y_test = np.array(y)
    # load testing data
    print('Loading testing data')
    y_test = np.loadtxt('y_test_nft.txt')
    x_test = np.loadtxt('x_test_nft.txt')
    # print('Pad sequences (samples x time)')
    # X_train = sequence.pad_sequences(x_train, maxlen=400, padding='post')
    # X_test = sequence.pad_sequences(x_test, maxlen=400, padding='post')
    # print('X_train shape:', X_train.shape)
    # print('X_test shape:', X_test.shape)
    # print(type(X_test))

    # load model structure
    print ('Loading model')
    json_file = open('ft_bigram.json')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into model
    loaded_model.load_weights('ft_bigram.hdf5')
    loaded_model.summary()

    y = [loaded_model.predict_classes(np.array([x_test[i], ])).tolist()[0] for i in range(len(x_test))]

    y_pred = []
    for i in range(len(y)):
        y_pred.append(y[i][0])
    y_pred = np.array(y_pred)
    # print(y_pred)
    # print(y_test)


    num = 0
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            num = num + 1
    # print(num)
    # print(len(y_pred))

    # calculate accuracy
    def acc(y_true, y_pred):
       return np.equal(y_true,y_pred).mean()

    # calculate precision
    def precision(y_true, y_pred):
        tp = 0
        p = 0
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                p+=1
                if y_true[i] == 1:
                    tp += 1

        return (float(tp)/p)

    # calculate recall
    def recall(y_true, y_pred):
        tp = 0
        t = 0
        for i in range(len(y_pred)):
            if y_true[i] == 1:
                t += 1
                if y_pred[i] == 1:
                    tp += 1

        return (float(tp)/t)

    def f1(preci, rec):
        return (2 * preci * rec / (preci + rec))

    # save predicted y and gound truth for further observation
    #np.savetxt(pattern + 'y_pred.txt',np.array(y_pred))
    #np.savetxt(pattern + 'y_true.txt',y)

    # print precision and recall, F1-score for testing data
    preci = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    print(" test accuracy: " + str(acc(y_test, y_pred)))
    print(" test precision: " + str(preci))
    print(" test recall: " + str(rec))
    print(" test F1-score: " + str(f1(preci,rec)))


if __name__ == "__main__":
    predict()