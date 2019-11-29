import os

import numpy as np
import pandas as pd

from d_utils import read_in_dataset

PRE = 'processed'
DATA_DIR = '../airbnb-recruiting-new-user-bookings'
CSV_FNAMES = {
    'age_bucket': os.path.join(DATA_DIR, 'age_gender_bkts.csv'),
    'countries': os.path.join(DATA_DIR, 'countries.csv'),
    'sessions': os.path.join(DATA_DIR, 'sessions.csv'),
    'test': os.path.join(DATA_DIR, 'test_users.csv'),
    'train': os.path.join(DATA_DIR, 'train_users_2.csv'),
    'train-processed': os.path.join(DATA_DIR, 'train_users_2-processed.csv'),
    'test-processed': os.path.join(DATA_DIR, 'test_users-processed.csv')
}

class BaselineDataset():
    def __init__(self, data, mode='train'):
        self.mode = mode
        self.df = self.process(data.copy())
        print('Processed')

    def process(self, data, verbose=True):
        df = data.copy()
        dropped_features = ['date_first_booking']
        if self.mode == 'test':
            dropped_features.pop(dropped_features.index('id'))
        df = df.drop(dropped_features, axis=1)
        df = df.fillna(-1)
        #Feature Engineering
        # date_account_created (process the month, year, and day)
        dac = np.vstack(df.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
        df['dac_year'] = dac[:, 0]
        df['dac_month'] = dac[:, 1]
        df['dac_day'] = dac[:, 2]
        df = df.drop(['date_account_created'], axis=1)

        # timestamp_first_active (process the month, year, and day)
        tfa = np.vstack(df.timestamp_first_active.astype(str).apply(
            lambda x: list(map(int, [x[:4], x[4:6], x[6:8], x[8:10], x[10:12], x[12:14]]))).values)
        df['tfa_year'] = tfa[:, 0]
        df['tfa_month'] = tfa[:, 1]
        df['tfa_day'] = tfa[:, 2]
        df = df.drop(['timestamp_first_active'], axis=1)

        # Age - set any outlier values to -1 (median/avg apparently made it worse)
        av = df.age.values
        df['age'] = np.where(np.logical_or(av < 14, av > 100), -1, av)

        # One-hot-encoding features
        ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
                     'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
        for f in ohe_feats:
            df_dummy = pd.get_dummies(df[f], prefix=f)
            df = df.drop([f], axis=1)
            df = pd.concat((df, df_dummy), axis=1)
        return df

    def split(self, data, index_train):
        df = data.copy()
        X_train = df.iloc[:index_train]
        X_test = df.iloc[index_train:]
        # drop id from training set
        X_train = X_train.drop('id', axis=1)
        return X_train, X_test

    # def save(self, csv_fname):
    #     self.df.to_csv(csv_fname, index=False)
    #     print('Dataset saved to {}'.format(csv_fname))

def make_dataset(csv_fname, do_baseline=True, save=True):
    # create baseline dataset then train your model
    df = read_in_dataset(csv_fname, verbose=True)
    # Create baseline dataset
    if do_baseline:
        if 'test' in csv_fname:
            mode = 'test'
        baseline = BaselineDataset(data=df, mode=mode)
        output_fname = csv_fname.split('.csv')[0] + '-{}.csv'.format(PRE)
        if save:
            baseline.save(output_fname)

if __name__ == '__main__':
    # make_dataset(CSV_FNAMES['train'])
    # read_in_dataset(CSV_FNAMES['train-processed'], verbose=True)

    # make_dataset(CSV_FNAMES['test'])
    # read_in_dataset(CSV_FNAMES['test-processed'], verbose=True)
    df_train = read_in_dataset(CSV_FNAMES['train'])
    df_test = read_in_dataset(CSV_FNAMES['test'])
    idx_train = df_train.shape[0]
    df = pd.concat((df_train, df_test), axis=0, ignore_index=True, sort=True)
    baseline = BaselineDataset(data=df)
    X_train, X_test = baseline.split(data=baseline.df, index_train=idx_train)

    output_fname = CSV_FNAMES['train'].split('.csv')[0] + '-{}.csv'.format(PRE)
    X_train.to_csv(output_fname, index=False)
    output_fname = CSV_FNAMES['test'].split('.csv')[0] + '-{}.csv'.format(PRE)
    X_test.to_csv(output_fname, index=False)
    print('Complete')
