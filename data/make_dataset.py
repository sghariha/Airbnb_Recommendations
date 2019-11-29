import os

import pandas as pd

from data.d_utils import read_in_dataset

PRE = 'processed'
DATA_DIR = '../airbnb-recruiting-new-user-bookings'
CSV_FNAMES = {
    'age_bucket': os.path.join(DATA_DIR, 'age_gender_bkts.csv'),
    'countries': os.path.join(DATA_DIR, 'countries.csv'),
    'sessions': os.path.join(DATA_DIR, 'sessions.csv'),
    'test': os.path.join(DATA_DIR, 'test_users.csv'),
    'train': os.path.join(DATA_DIR, 'train_users_2.csv')
}

categorical_features = ['gender', 'signup_method', 'signup_flow',
                     'language', 'affiliate_channel', 'affiliate_provider',
                      'signup_app', 'first_device_type', 'first_browser']

BASELINE_FEATURES = ['id', 'date_account_created', 'timestamp_first_active',
                     'date_first_booking', 'age', 'country_destination'] + categorical_features

class BaselineDataset():

    def __init__(self, data):
        self.df = self.process(data.copy())

    def process(self, data, verbose=False):
        df = data.copy()
        df['date_account_created'] = pd.to_datetime(df['date_account_created'], infer_datetime_format=True)
        # df['year'] = getattr(df['date_account_created'].dt, 'year')
        df['month'] = getattr(df['date_account_created'].dt, 'month')
        df['week'] = getattr(df['date_account_created'].dt, 'week')
        # df = pd.get_dummies(df, columns=['year', 'month', 'week'])
        df = df[df['age'].between(left=13, right=95)]
        df['age'] = df['age'].fillna(df['age'].describe()['50%'])
        # for i in categorical_features:
        #     df[i] = df[i].astype('category')
        #     df[i] = df[i].cat.codes
        df = pd.get_dummies(df, columns=categorical_features)
        df['country_destination'] = df['country_destination'].astype('category')
        df['country_destination'] = df['country_destination'].cat.codes
        df = df.drop(['id', 'date_first_booking', 'date_account_created',
                      'timestamp_first_active', 'first_affiliate_tracked'], axis=1)
        return df

    def save(self, csv_fname):
        self.df.to_csv(csv_fname, index=False)
        print('Dataset saved to {}'.format(csv_fname))

def make_dataset(csv_fname, do_baseline=True, save=True):
    # create baseline dataset then train your model
    df = read_in_dataset(csv_fname, verbose=True)
    # Create baseline dataset
    if do_baseline:
        baseline = BaselineDataset(data=df)
        output_fname = csv_fname.split('.csv')[0] + '-{}.csv'.format(PRE)
        if save:
            baseline.save(output_fname)

if __name__ == '__main__':
    make_dataset(CSV_FNAMES['train'])
