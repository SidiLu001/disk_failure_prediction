# from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import argparse
from utils.loadData import read_data
from utils.evaluation import calMetrix
import time

def classifyRes(arr):
    arr = arr.reshape(arr.size,)
    for i in range(arr.size):
        arr[i] = 1 if arr[i]>0.5 else 0
    return arr


def model_fit(train_X, train_y, test_X, test_y):
    # train_X, train_y, test_X, test_y = read_data()
    outdim = train_y.shape[1]
    validation_number = int(0.9 * train_X.shape[0])
    validation_X = train_X[validation_number:]
    validation_y = train_y[validation_number:]
    train_X = train_X[:validation_number]
    train_y = train_y[:validation_number]

    model = Sequential()
    model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, dropout=0.25))
    # print(model.layers)
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    # model.add(Dense(outdim))
    model.add(Dense(outdim, activation='sigmoid'))
    # model.add(Dense(outdim, activation='softmax'))
    # sigmoid and softmax are activation functions used by the neural network output layer
    # for binary discrimination and multi-class discrimination
    # binary cross-entropy and categorical cross-entropy are corresponding loss functions

    # model.add(Activation('linear'))

    # model.compile(loss='mse', optimizer='rmsprop')
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  # loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )

    # fit network
    history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(validation_X, validation_y),
                        verbose=0,
                        shuffle=True)

    # summarize performance of the model
    # scores = model.evaluate(train_X, train_y, verbose=0)
    # print(model.metrics_names)
    # print("model loss: %.2f%%" % (scores*100))

    # plot history
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='validation')
    # plt.legend()
    # plt.grid()
    # plt.savefig('loss.png')
    # plt.clf()
    np.savetxt('loss.csv', history.history['loss'])
    np.savetxt('val_loss.csv', history.history['val_loss'])
    t0 = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    model.save('../model/LSTM_model_%s.h5'%t0)

    # make a prediction
    y_pre = model.predict(test_X)
    # yhat = yhat.reshape(train_y.shape[1])
    print(y_pre.shape)
    y_pre = classifyRes(y_pre)
    calMetrix(__file__, y_pre, test_y)

    # inv_yhat = yhat
    # inv_test_y = test_y
    np.savetxt('forecast.csv', y_pre)
    np.savetxt('actual.csv', test_y)

    # for k in range(inv_yhat.shape[1]):
    #     plt.plot(inv_yhat[:, k], label='forecast')
    #     plt.plot(inv_test_y[:, k], label='actual')
    #     plt.legend()
    #     plt.grid()
    #     plt.savefig( 'var_'+str(k)+'.png')
    #     plt.clf()
    #
    # plt.close()

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

