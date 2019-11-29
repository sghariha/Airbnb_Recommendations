import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

def accuracy(gtruth, predictions):
    return (gtruth == predictions).mean()

def ber(tn, fp, fn, tp):
    return 1.0 - 0.5 *(tp/(tp+fn) + tn / (tn+fp))

def fbeta(precision, recall, beta):
    return (1 + beta ** 2) * precision * recall / (beta**2 * precision + recall)

def unravel_confusion_matrix(gtruth, predictions, manual=False):
    if manual:
        TP_ = np.logical_and(predictions, gtruth)
        FP_ = np.logical_and(predictions, np.logical_not(gtruth))
        TN_ = np.logical_and(np.logical_not(predictions), np.logical_not(gtruth))
        FN_ = np.logical_and(np.logical_not(predictions), gtruth)

        TP = sum(TP_)
        FP = sum(FP_)
        TN = sum(TN_)
        FN = sum(FN_)
    else:
        TN, FP, FN, TP = confusion_matrix(gtruth, predictions).ravel()
    return TN, FP, FN, TP

def precision_recall(gtruth, predictions, manual=False):
    if manual:
        # precision / recall
        retrieved = sum(predictions)
        relevant = sum(gtruth)
        intersection = sum([y and p for y, p in zip(gtruth, predictions)])

        precision = intersection / retrieved
        recall = intersection / relevant
    else:
        precision, recall, _, _ = precision_recall_fscore_support(gtruth, predictions)
    return precision, recall

def evaluate_lr(gtruth, predictions, verbose=True, normalize=True, beta=0):
    metrics = {}
    metrics['acc'] = accuracy(gtruth, predictions)
    # metrics['cm'] = confusion_matrix(gtruth, predictions)
    # if normalize:
    #     np.set_printoptions(precision=3, suppress=True)
    #     metrics['cm'] = metrics['cm'].astype('float') / metrics['cm'].sum(axis=1)[:, np.newaxis]
    # tn, fp, fn, tp = unravel_confusion_matrix(gtruth, predictions)
    # metrics['ber'] = ber(tn, fp, fn, tp)
    # if beta:
    #     metrics['precision'], metrics['recall'] = precision_recall(gtruth, predictions, manual=True)
    #     metrics['fbeta'] = fbeta(metrics['precision'], metrics['recall'], beta)
    # if verbose:
    #     print('\nLogistic Regression Evaluation Results')
    #     print('{:40}'.format('-'*40))
    #     print('ACC:{:4.2f}'.format(metrics['acc']))
    #     print('BER:{:4.2f}'.format(metrics['ber']))
    #     print('POSITIVE LABELS:{:5.2f}'.format(tp/(tp+fn)))
    #     print('POSITIVE PREDICTIONS:{:5.2f}'.format(tp/(tp+fp)))
    #     print("CONFUSION MATRIX:")
    #     print(metrics['cm'])
    return metrics
