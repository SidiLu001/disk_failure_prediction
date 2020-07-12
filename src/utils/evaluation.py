from sklearn import metrics
import numpy as np
from utils.util import getBasename

def calMetrix(fileName, y_pre, test_y):
    print('******************* %s********************'%getBasename(fileName))
    print('confusion_matrix:')
    confm = metrics.confusion_matrix(y_pre, test_y)
    print(confm)
    print('accuracy_score: ')
    print(metrics.accuracy_score(y_pre, test_y))
    print('precision_score: ')
    print(metrics.precision_score(y_pre, test_y))
    print('recall_score: ')
    print(metrics.recall_score(y_pre, test_y, average='binary'))
    print('f1_score: ')
    print(metrics.f1_score(y_pre, test_y))
    TN, FP, FN, TP = confm[0,0], confm[0,1], confm[1,0], confm[1,1]
    hh = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    MCC = (TP * TN - FP * FN) / np.sqrt(hh)
    print('MCC: ', MCC)