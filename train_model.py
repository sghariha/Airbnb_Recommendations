"""Train and evaluate AirBnB Recommender System

#TODO write evaluation metric (NDCG), confusion matrix plots
#TODO conduct cross validation experiment
#TODO include pearson correlation
#TODO some kind of clustering

"""

# Standard dist imports
import os
import datetime

# Third party imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

# Project level imports
from data.d_utils import read_in_dataset, experiment_features
from model.eval_model import evaluate_lr

# Module level constants
DATA_DIR = './airbnb-recruiting-new-user-bookings'
CSV_FNAMES = {
    'train-processed': os.path.join(DATA_DIR, 'train_users_2-processed.csv'),
    'test-processed': os.path.join(DATA_DIR, 'test_users-processed.csv'),
    'train-feat_eng': os.path.join(DATA_DIR, 'train_users_2-feature_eng.csv'),
    'test-feat_eng': os.path.join(DATA_DIR, 'test_users-feature_eng.csv'),
    'train-merged_sessions': os.path.join(DATA_DIR, 'train_users-merged_sessions.csv'),
    'test-merged_sessions': os.path.join(DATA_DIR, 'test_users-merged_sessions.csv')
}
# Select dataset type
# DATASET_TYPE = 'processed'
# DATASET_TYPE = 'feat_eng'
DATASET_TYPE = 'merged_sessions'

# Model Default
XGB_MODEL = True

# EXPERIMENTAL_FEATURES
STATS_flag = False
RATIOS_flag = False
CASTED_flag = True

# Read in training set and encode labels
start_timer = datetime.datetime.now()
class AirBnB(): pass
airbnb = AirBnB()
airbnb.X = read_in_dataset(CSV_FNAMES['train-{}'.format(DATASET_TYPE)], verbose=True)
# Select here which experimeental features to run
# default is the baseline
airbnb.X = experiment_features(data=airbnb.X,
                               stats=STATS_flag,
                               ratios=RATIOS_flag,
                               casted=CASTED_flag, verbose=True)
airbnb.train_labels = airbnb.X.pop('country_destination')
airbnb.le = LabelEncoder()
airbnb.le.fit(airbnb.train_labels)
airbnb.target_labels = airbnb.le.classes_
airbnb.y = airbnb.le.transform(airbnb.train_labels)

# Partition and split datasets
SEED = 42
PARTITION = 0.10
airbnb.X_train, airbnb.X_val, \
airbnb.y_train, airbnb.y_val = train_test_split(airbnb.X, airbnb.y, test_size=PARTITION, shuffle=True, random_state=SEED)
# Read in test set
airbnb.X_test = read_in_dataset(CSV_FNAMES['test-{}'.format(DATASET_TYPE)], keep_id=True, verbose=True)
airbnb.X_test = experiment_features(data=airbnb.X_test,
                                    stats=STATS_flag,
                                    ratios=RATIOS_flag,
                                    casted=CASTED_flag, verbose=False)
airbnb.test_id = airbnb.X_test.pop('id')
airbnb.X_test.pop('country_destination')
print('Dataset sizes: TRAIN: {:5} | VAL: {:5} | TEST {:5}'.format(
    airbnb.X_train.shape[0], airbnb.X_val.shape[0], airbnb.X_test.shape[0]))

# Release memory usage
del airbnb.X
del airbnb.y

#Compute class weights
class_weight_list = compute_class_weight('balanced',
                                         np.unique(np.ravel(airbnb.y_train,order='C')),
                                         np.ravel(airbnb.y_train,order='C'))
class_weight = dict(zip(np.unique(airbnb.y_train), class_weight_list))
print(class_weight)


print('Training classifier')
if XGB_MODEL:
    # === XGB Classifier (tuned) ===#
    model = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25, class_weight=class_weight,
                        objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=SEED, verbosity=2)
    model.fit(airbnb.X_train,airbnb.y_train)
    airbnb.y_pred_train = model.predict(airbnb.X_train)
    airbnb.y_pred_val = model.predict(airbnb.X_val)
    airbnb.y_pred_test = model.predict_proba(airbnb.X_test)
else:
    # === Logistic Regression ===#
    model = LogisticRegression(solver='lbfgs', multi_class='auto', verbose=1)
    model.fit(airbnb.X_train, airbnb.y_train)
    airbnb.y_pred_train = model.predict(airbnb.X_train)
    airbnb.y_pred_val = model.predict(airbnb.X_val)
    airbnb.y_pred_test = model.predict_proba(airbnb.X_test)

# Evaluate classifiers
print('Training set')
print(evaluate_lr(airbnb.y_train, airbnb.y_pred_train))
print()
print('Validation set')
print(evaluate_lr(airbnb.y_val, airbnb.y_pred_val))
print()

# Write test predictions to submission file
#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(airbnb.test_id)):
    idx = airbnb.test_id[i]
    ids += [idx] * 5
    cts += airbnb.le.inverse_transform(np.argsort(airbnb.y_pred_test[i])[::-1])[:5].tolist()
print('complete')
#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)

now  = datetime.datetime.now()
duration = now - start_timer
print('Completion @ {} | elapsed time: {}'.format(now, duration/datetime.timedelta(minutes=1)))
