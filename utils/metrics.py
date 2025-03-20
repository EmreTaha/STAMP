from sklearn import metrics
import numpy as np
import scipy.stats
#TODO ust sinif yaz bunlari onun icine ayri obje olarak koy

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

def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    # Taken from https://github.com/yandexdataschool/roc_comparison/blob/master/compare_auc_delong_xu.py
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-1*ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

def delong_auc_var_conf(ground_truth, predictions, alpha_conf=0.95):

    auc, auc_cov = delong_roc_variance(
        ground_truth,
        predictions)

    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha_conf) / 2)

    ci = scipy.stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    ci[ci > 1] = 1

    return auc, auc_cov, ci

def stats_calculator(path, metrics=['Test roc-auc','Test pr-auc', 'Test balanced accuracy'], folds=4):
    """
    Calculates mean and std of metrics from logs of a cross-validation experiment. The structure should be path/fold_i/logs.txt
    Args:
        path: path to the folder containing the logs
        metrics: list of metrics to calculate mean and std
        folds: number of folds
    """
    results_dict = {key: [] for key in metrics}
    for i in range(folds):
        logs = open(path + '/fold_'+str(i)+'/logs.txt', 'r').readlines()[-1]
        for j in metrics:
            assert j in logs, 'Metric ' + j+ ' not found in logs'
            results_dict[j].append(float(logs.split(j+': ')[1].split('.. ')[0]))
    for key in results_dict.keys():
        temp = np.array(results_dict[key])
        results_dict[key] = key+" Mean: "+str(np.mean(temp))+" Std: "+str(np.std(temp))
    with open(path + '/results.txt', 'w') as f:
        for key in results_dict.keys():
            f.write(results_dict[key]+'\n')