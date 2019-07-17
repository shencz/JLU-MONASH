from keras.models import model_from_json
import numpy as np
def predict():
    """
    This function loads the corresponding model for specified edit type
    and calculate precision, recall and F1-score for each model
    :param pattern: edit type (string)
    :return: None
    """
    #load testing data
    print('Loading testing data')
    y = np.loadtxt('y_test_cnn.txt')
    x = np.loadtxt('x_test_cnn.txt')

    # load model structure
    print ('Loading model')
    json_file = open('CNN_model.json')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into model
    loaded_model.load_weights('CNN_model.hdf5')
    y_pred = [np.argmax(loaded_model.predict(np.array([x[i], ]))) for i in range(len(x))]

    # calculate accuracy
    def acc(y_true, y_pred):
       return np.equal(np.argmax(y_true, axis=-1),y_pred).mean()

    # calculate precision
    def precision(y_true, y_pred):
        tp = 0
        p = 0
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                p+=1
                if np.argmax(y_true, axis=-1)[i] == 1:
                    tp += 1

        return (float(tp)/p)

    # calculate recall
    def recall(y_true, y_pred):
        tp = 0
        t = 0
        for i in range(len(y_pred)):
            if np.argmax(y_true, axis=-1)[i] == 1:
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
    preci = precision(y, y_pred)
    rec = recall(y, y_pred)
    print(" test accuracy: " + str(acc(y, y_pred)))
    print(" test precision: " + str(preci))
    print(" test recall: " + str(rec))
    print(" test F1-score: " + str(f1(preci,rec)))


if __name__ == "__main__":
    predict()