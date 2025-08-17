import math

import numpy as np
from sklearn.metrics import f1_score, auc, roc_auc_score, confusion_matrix, accuracy_score, roc_curve
from getData import GetDataSet
import torch


def metrics_compute(preds_sum, test_label):
    a = preds_sum.cpu()
    b = test_label.cpu()
    # 计算混淆矩阵，并显示
    cm = confusion_matrix(b, a)  # 默认正类是 1
    expected = b.numpy()
    predicted = a.numpy()

    accuracy = accuracy_score(expected, predicted)
    F1_score = f1_score(expected, predicted)

    TP = np.sum(np.multiply(expected, predicted))
    FP = np.sum(np.logical_and(np.equal(expected, 0), np.equal(predicted, 1)))
    FN = np.sum(np.logical_and(np.equal(expected, 1), np.equal(predicted, 0)))
    TN = np.sum(np.logical_and(np.equal(expected, 0), np.equal(predicted, 0)))

    G_mean = math.sqrt((TP / (TP + FN)) * (TN / (TN + FP)))
    AUC = roc_auc_score(expected, predicted)

    #print('F1-score:{}, G-mean:{}, AUC:{}'.format(F1_score, G_mean, AUC))

    return accuracy, F1_score, G_mean, AUC