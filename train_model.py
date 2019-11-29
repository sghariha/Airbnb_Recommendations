"""Train and evaluate AirBnB Recommender System"""

# Standard dist imports
import os

# Third party imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost.sklearn import XGBClassifier

# Project level imports
from data.d_utils import read_in_dataset
from model.eval_model import evaluate_lr

# Module level constants
DATA_DIR = './airbnb-recruiting-new-user-bookings'
CSV_FNAMES = {
    'age_bucket': os.path.join(DATA_DIR, 'age_gender_bkts.csv'),
    'countries': os.path.join(DATA_DIR, 'countries.csv'),
    'sessions': os.path.join(DATA_DIR, 'sessions.csv'),
    'test': os.path.join(DATA_DIR, 'test_users.csv'),
    'train-processed': os.path.join(DATA_DIR, 'train_users_2-processed.csv'),
    'test-processed': os.path.join(DATA_DIR, 'test_users-processed.csv')
}

# Read in training set and encode labels
class AirBnB(): pass
airbnb = AirBnB()
airbnb.X = read_in_dataset(CSV_FNAMES['train-processed'], verbose=True)
airbnb.train_labels = airbnb.X.pop('country_destination')
airbnb.le = LabelEncoder()
airbnb.le.fit(airbnb.train_labels)
airbnb.target_labels = airbnb.le.classes_
airbnb.y = airbnb.le.transform(airbnb.train_labels)

# Partition and split datasets
# test set read in
SEED = 42
PARTITION = 0.10
airbnb.X_train, airbnb.X_val, \
airbnb.y_train, airbnb.y_val = train_test_split(airbnb.X, airbnb.y, test_size=PARTITION, shuffle=True, random_state=SEED)
airbnb.X_test = read_in_dataset(CSV_FNAMES['test-processed'], verbose=True)
airbnb.test_id = airbnb.X_test.pop('id')
airbnb.X_test.pop('country_destination')
print('Dataset sizes: TRAIN: {:5} | VAL: {:5} | TEST {:5}'.format(airbnb.X_train.shape[0], airbnb.X_val.shape[0], airbnb.X_test.shape[0]))

#Compute class weights
class_weight_list = compute_class_weight('balanced',
                                         np.unique(np.ravel(airbnb.y_train,order='C')),
                                         np.ravel(airbnb.y_train,order='C'))
class_weight = dict(zip(np.unique(airbnb.y_train), class_weight_list))
print(class_weight)

#=== Logistic Regression ===#
print('Initializing classifier')
model = LogisticRegression(solver='lbfgs', multi_class='auto')
model.fit(airbnb.X_train, airbnb.y_train)
airbnb.y_pred_train = model.predict(airbnb.X_train)
airbnb.y_pred_val = model.predict(airbnb.X_val)
airbnb.y_pred_test = model.predict_proba(airbnb.X_test)

#=== XGB Classifier (tuned) ===#
# model = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25, class_weight=class_weight,
#                     objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=SEED)
# model.fit(airbnb.X_train,airbnb.y_train)
# airbnb.y_pred_train = model.predict(airbnb.X_train)
# airbnb.y_pred_val = model.predict(airbnb.X_val)
# airbnb.y_pred_test = model.predict_proba(airbnb.X_test)

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
