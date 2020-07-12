from keras.layers import Dense, LSTM
# from keras.layers import Input, Dropout, Activation
# from keras.layers import Convolution2D, MaxPooling2D, Reshape
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
# from matplotlib import pyplot as plt
# import random
import argparse
from utils.loadData import read_data
from utils.evaluation import calMetrix
import time

pt = lambda s:print(type(s),s)

nb_epoch = 100
# number_of_batch = 100
batch_size = 72
output_dim = 1

def classifyRes(arr):
    arr = arr.reshape(arr.size,)
    for i in range(arr.size):
        arr[i] = 1 if arr[i]>0.5 else 0
    return arr

def model_fit(train_X, train_y, test_X, test_y):
    # define model
    model = Sequential()
    # model.add(TimeDistributed(cnn))
    model.add(TimeDistributed(Convolution1D(128, 4, border_mode='same'), input_shape=train_X.shape[1:]))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128, return_sequences=True, name="lstm_layer0"))
    model.add(LSTM(128, return_sequences=False, name="lstm_layer1"))
    # model.add(LSTM(100, return_sequences=True, name="lstm_layer2"))
    model.add(Dense(output_dim, activation='sigmoid'))
    # model.add(GlobalAveragePooling1D(name="global_avg"))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # %%
    model.fit(train_X, train_y,batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, validation_data=(test_X, test_y))

    test_y = test_y.reshape(test_y.size, 1)
    predict_y = model.predict(test_X)
    predict_y = predict_y.reshape(predict_y.size, 1)
    predict_y = classifyRes(predict_y)
    calMetrix(__file__, predict_y, test_y)
    t0 = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    model.save('../model/CNN_LSTM_model_%s.h5'%t0)

    # plt.plot(y_predict, 'r',label='forecast')
    # plt.plot(y_test, 'b',label='actual')
    # plt.legend()
    # plt.grid()
    # plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="train data path")
    parser.add_argument("--xpath", help="Xdata Path", default='', type=str,  required=True)
    parser.add_argument("--ypath", help="ydata Path", default='', type=str, required=True)
    parser.add_argument("--group", help="group", default='', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    for train_X, train_y, test_X, test_y in read_data(args.xpath, args.ypath, args.group):
        model_fit(train_X, train_y, test_X, test_y)
    # python3 CNNLSTM_args.py --xpath X1_10days.npy --ypath y1_10days.npy --group SPL
