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

    
    print('Loading testing data')
    y_test = np.loadtxt('y_test_nft.txt')
    x_test = np.loadtxt('x_test_nft.txt')
   
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
