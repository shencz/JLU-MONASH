from keras.models import model_from_json
import numpy as np

#
# print('Load model')
# # load json and create model
# json_file = open('ft_bigram.json', 'r')
# print("read json")
# loaded_model_text = json_file.read()
# print("load model from json")
# loaded_model = model_from_json(loaded_model_text)
# json_file.close()
# # load weights into new model
# print("load weights into new model")
# loaded_model.load_weights('ft_bigram.hdf5')
# loaded_model.summary()
#
#
# print('Loading testing data')
# y_test = np.loadtxt('y_test_nft.txt')
# x_test = np.loadtxt('x_test_nft.txt')
#
# maxlen = 799
# num = 0
# wa_nft = []
# for t in range(len(x_test)):
#     ans = loaded_model.predict_classes(np.array([x_test[t], ]))
#     if ans[0][0] != y_test[t]:
#         print(t, ans, y_test[t])
#         wa_nft.append([t, ans[0][0], y_test[t]])
#         num = num + 1
#
# print(num)
# print(t)
# np.savetxt("WA_NFT.txt",wa_nft)

print('Load model')
# load json and create model
json_file = open('ft_trgram.json', 'r')
print("read json")
loaded_model_text = json_file.read()
print("load model from json")
loaded_model = model_from_json(loaded_model_text)
json_file.close()
# load weights into new model
print("load weights into new model")
loaded_model.load_weights('ft_trgram.hdf5')
loaded_model.summary()


print('Loading testing data')
y_test = np.loadtxt('y_test_nft3.txt')
x_test = np.loadtxt('x_test_nft3.txt')


num = 0
wa_nft3 = []
for t in range(len(x_test)):
    ans = loaded_model.predict_classes(np.array([x_test[t], ]))
    if ans[0][0] != y_test[t]:
        print(t, ans, y_test[t])
        wa_nft3.append([t, ans[0][0], y_test[t]])
        num = num + 1

print(num)
print(t)
np.savetxt("WA_NFT3.txt",wa_nft3)