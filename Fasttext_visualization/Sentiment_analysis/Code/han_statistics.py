#!/usr/bin/python
"""
Example of attention coefficients visualization
Uses saved model, so it should be executed after train.py
"""
from rnn_train import *

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,MODEL_PATH)
    num = 0
    wa_han = []
    for t in range(len(X_test)):
        x_batch_test, y_batch_test = X_test[t:t + 1], y_test[t:t + 1]
        seq_len_test = np.array([list(x).index(0) + 1 for x in x_batch_test])
        sigmoid_p = sess.run(sigmoid_val, feed_dict={batch_ph: x_batch_test, target_ph: y_batch_test,
                                                     seq_len_ph: seq_len_test, keep_prob_ph: 1.0})
        if y_test[t:t+1][0] != round(sigmoid_p, 0):
            num = num + 1
        # Represent the sample by words rather than indices
            print(t,round(sigmoid_p, 0),y_test[t:t + 1],sigmoid_p)
            wa_han.append([t,round(sigmoid_p, 0),y_test[t:t + 1],sigmoid_p])
    print(num,t)
    np.savetxt("WA_HAN.txt",wa_han)







