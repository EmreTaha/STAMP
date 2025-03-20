import numpy as np

from . import roc_aucs, pr_aucs, f1_score, balanced_accs, delong_auc_var_conf

def eval_binary(preds, labels, save_dir, outputs):
    test_auc = roc_aucs(labels, preds)
    test_prauc =pr_aucs(labels, preds)
    test_f1 = f1_score(labels, np.around(preds))
    test_baccs = balanced_accs(labels, np.around(preds))
    _, test_delong_auc_cov, test_delong_ci = delong_auc_var_conf(np.array(labels), np.array(preds))

    print(  f"Test roc-auc: {test_auc:.3f}.. "
            f"Test delong roc-auc-cov: {test_delong_auc_cov:.3f}.."
            f"Test delong roc-auc-ci: {['{:.3f}'.format(x) for x in test_delong_ci]}.. "
            f"Test pr-auc: {test_prauc:.3f}.. "
            f"Test F1: {test_f1:.3f}.. "
            f"Test balanced accuracy: {test_baccs:.3f}.. "
            f"Test accuracy: {correct / total:.3f}")
        
    outputs.append((f"Test roc-auc: {test_auc:.3f}.. "
            f"Test delong roc-auc-cov: {test_delong_auc_cov:.3f}.."
            f"Test delong roc-auc-ci: {['{:.3f}'.format(x) for x in test_delong_ci]}.. "
            f"Test pr-auc: {test_prauc:.3f}.. "
            f"Test balanced accuracy: {test_baccs:.3f}.. "
            f"Test F1: {test_f1:.3f}.. "
            f"Test accuracy: {correct / total:.3f}"))

    test_prec, test_recall, _ = precision_recall_curve(labels, test_preds)
    # plot the precision-recall curves
    labels = np.array([i.item() for i in labels])
    no_skill = len(labels[labels==1]) / len(labels)
    print("No skill pr-auc: ", no_skill)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(test_recall, test_prec, marker='.', label='Model')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(save_dir+'/test_pruac.png')
