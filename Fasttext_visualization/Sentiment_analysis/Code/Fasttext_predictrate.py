from keras.models import model_from_json
import numpy as np


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
loaded_model.summary()


print('Loading testing data')
y_test = np.loadtxt('y_test_ft.txt')
x_test = np.loadtxt('x_test_ft.txt')

maxlen = 400
num = 0
wa_ft = []
for t in range(len(x_test)):
    ans = loaded_model.predict_classes(np.array([x_test[t], ]))
    if ans[0][0] != y_test[t]:
        print(t, ans, y_test[t])
        wa_ft.append([t,ans[0][0], y_test[t]])
        num = num + 1

print(num)
print(t)
np.savetxt("WA_FT.txt",wa_ft)