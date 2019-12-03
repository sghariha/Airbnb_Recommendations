"""Evaluate the model"""

# Standard dist
import os
import pickle

# Third party imports
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

# Project level imports
from data.d_utils import read_in_dataset

# Module level constants
DATA_DIR = '../airbnb-recruiting-new-user-bookings'
# Select dataset type
# DATASET_TYPE = 'processed'
# DATASET_TYPE = 'feat_eng'
DATASET_TYPE = 'merged_sessions'
CSV_FNAMES = {
    'train-processed': os.path.join(DATA_DIR, 'train_users_2-processed.csv'),
    'test-processed': os.path.join(DATA_DIR, 'test_users-processed.csv'),
    'train-feat_eng': os.path.join(DATA_DIR, 'train_users_2-feature_eng.csv'),
    'test-feat_eng': os.path.join(DATA_DIR, 'test_users-feature_eng.csv'),
    'train-merged_sessions': os.path.join(DATA_DIR, 'train_users-merged_sessions.csv'),
    'test-merged_sessions': os.path.join(DATA_DIR, 'test_users-merged_sessions.csv')
}

def main():
    # load the model from disk
    filename = '../finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    # Set up dataset
    class AirBnB(): pass
    airbnb = AirBnB()
    airbnb.X_test = read_in_dataset(CSV_FNAMES['test-{}'.format(DATASET_TYPE)], verbose=True)
    airbnb.test_id = airbnb.X_test.pop('id')
    airbnb.X_test.pop('country_destination')

    # Run model on test set
    airbnb.y_pred_test = loaded_model.predict_proba(airbnb.X_test)

    # Write test predictions to submission file
    # Taking the 5 classes with highest probabilities
    ids = []  # list of ids
    cts = []  # list of countries
    for i in range(len(airbnb.test_id)):
        idx = airbnb.test_id[i]
        ids += [idx] * 5
        cts += airbnb.le.inverse_transform(np.argsort(airbnb.y_pred_test[i])[::-1])[:5].tolist()
    print('complete')
    # Generate submission
    sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
    sub.to_csv('sub.csv', index=False)

"""Evaluation Metric Functions"""
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

def evaluate_model(gtruth, predictions, verbose=True, normalize=True, beta=0):
    """Compute all relevant evaluation metrics for a given model"""
    metrics = {}
    metrics['acc'] = accuracy(gtruth, predictions)
    metrics['cm'] = confusion_matrix(gtruth, predictions)
    if normalize:
        np.set_printoptions(precision=3, suppress=True)
        metrics['cm'] = metrics['cm'].astype('float') / metrics['cm'].sum(axis=1)[:, np.newaxis]
    tn, fp, fn, tp = unravel_confusion_matrix(gtruth, predictions)
    metrics['ber'] = ber(tn, fp, fn, tp)
    if beta:
        metrics['precision'], metrics['recall'] = precision_recall(gtruth, predictions, manual=True)
        metrics['fbeta'] = fbeta(metrics['precision'], metrics['recall'], beta)
    if verbose:
        print('\nModel Evaluation Results')
        print('{:40}'.format('-'*40))
        print('ACC:{:4.2f}'.format(metrics['acc']))
        print('BER:{:4.2f}'.format(metrics['ber']))
        print('POSITIVE LABELS:{:5.2f}'.format(tp/(tp+fn)))
        print('POSITIVE PREDICTIONS:{:5.2f}'.format(tp/(tp+fp)))
        print("CONFUSION MATRIX:")
        print(metrics['cm'])
    return metrics

def plot_feature_importances(importances, feature_decoder, top_k=10):
    """Plot feature importances for XGB/Random Forest model"""
    import matplotlib.pyplot as plt
    import numpy as np
    sorted_idx = np.argsort(importances)[::-1]
    if top_k:
        sorted_idx = sorted_idx[:top_k]
    importances = importances[sorted_idx]
    decoded_feature_names = [feature_decoder[idx] for idx in sorted_idx]
    plt.barh(range(len(importances)), importances[::-1])
    plt.yticks(range(len(importances)), decoded_feature_names[::-1])
    plt.xlabel('Importance values')
    plt.ylabel('Features')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('figs_feature_importance.png', bbox_inches = "tight")
    plt.show()

if __name__ == '__main__':
    main()
