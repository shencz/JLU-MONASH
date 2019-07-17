from keras.models import model_from_json
import numpy as np


print('Load model')
# load json and create model
json_file = open('CNN_model.json', 'r')
print("read json")
loaded_model_text = json_file.read()
print("load model from json")
loaded_model = model_from_json(loaded_model_text)
json_file.close()
# load weights into new model
print("load weights into new model")
loaded_model.load_weights('CNN_model.hdf5')
loaded_model.summary()


print('Loading testing data')
y_test = np.loadtxt('y_test_cnn.txt')
x_test = np.loadtxt('x_test_cnn.txt')

maxlen = 400
num = 0
wa_cnn = []
for t in range(len(x_test)):
    ans = loaded_model.predict(np.array([x_test[t], ]))
    if np.argmax(ans) != np.argmax(y_test[t]):
        print(t, np.argmax(ans), np.argmax(y_test[t]))
        wa_cnn.append([t,np.argmax(ans),np.argmax(y_test[t])])
        num = num + 1

print(num)
print(t)
np.savetxt("WA_CNN.txt",wa_cnn)