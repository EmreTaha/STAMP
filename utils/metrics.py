from sklearn import metrics
import numpy as np
import scipy.stats

def roc_aucs(labels, preds, average=None, multi_class="raise"):
    return metrics.roc_auc_score(labels, preds, average=average, multi_class=multi_class)

def balanced_accs(labels, preds):
    return metrics.balanced_accuracy_score(labels, preds)

def pr_aucs(labels, preds, average=None, multi_class=None):
    return metrics.average_precision_score(labels, preds, average=average)

def f1_score(labels, preds, average=None):
    preds = np.around(preds)
    if average:
        return metrics.f1_score(labels, preds, average=average)
    else:
        return metrics.f1_score(labels, preds, average=average)[1]