import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data.d_utils import read_in_dataset
from model.eval_model import evaluate_lr

DATA_DIR = './airbnb-recruiting-new-user-bookings'
CSV_FNAMES = {
    'age_bucket': os.path.join(DATA_DIR, 'age_gender_bkts.csv'),
    'countries': os.path.join(DATA_DIR, 'countries.csv'),
    'sessions': os.path.join(DATA_DIR, 'sessions.csv'),
    'test': os.path.join(DATA_DIR, 'test_users.csv'),
    'train': os.path.join(DATA_DIR, 'train_users_2-processed.csv'),
    'baseline': os.path.join(DATA_DIR, 'baseline.csv')
}

class AirBnB(): pass
airbnb = AirBnB()
airbnb.X = read_in_dataset(CSV_FNAMES['train'], verbose=True)
airbnb.y = airbnb.X.pop('country_destination')

SEED = 42
PARTITION = 0.10
airbnb.X_train, airbnb.X_val, \
airbnb.y_train, airbnb.y_val = train_test_split(airbnb.X, airbnb.y, test_size=PARTITION, shuffle=True, random_state=SEED)
print('Dataset sizes: TRAIN: {:5} | VAL: {:5}'.format(airbnb.X_train.shape[0], airbnb.X_val.shape[0]))

model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=100000)
model.fit(airbnb.X_train, airbnb.y_train)
airbnb.y_pred_train = model.predict(airbnb.X_train)
airbnb.y_pred_val = model.predict(airbnb.X_val)

print('Training set')
print(evaluate_lr(airbnb.y_train, airbnb.y_pred_train))
print()
print('Validation set')
print(evaluate_lr(airbnb.y_val, airbnb.y_pred_val))
print()
