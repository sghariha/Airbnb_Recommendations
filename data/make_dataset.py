import os

import numpy as np
import pandas as pd

from d_utils import read_in_dataset

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

# features
'''
statistical_features = [i for i in df.columns if i.endswith('_elapsed')] + ['n_actions_per_user', 'n_distinct_action_detail', 'n_distinct_action_types',
'n_distinct_actions',
'n_distinct_device_types']
ratios = ['ratio_distinct_actions',
'ratio_distinct_actions_types',
'ratio_distinct_action_details',
'ratio_distinct_devices']
casted_features = [i for i in df.columns if i.endswith('_ratio')]
'''

class BaselineDataset():
    def __init__(self, data, drop_raw=True):
        self.data = self.process(data.copy(), drop_raw=drop_raw)
        print('Processed')

    def process(self, data, drop_raw=True, verbose=True):
        df = data.copy()
        dropped_features = ['date_first_booking']
        df = df.drop(dropped_features, axis=1)
        df = df.fillna(-1)
        #Feature Engineering
        # date_account_created (process the month, year, and day)
        dac = np.vstack(df.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
        df['dac_year'] = dac[:, 0]
        df['dac_month'] = dac[:, 1]
        df['dac_day'] = dac[:, 2]

        # timestamp_first_active (process the month, year, and day)
        tfa = np.vstack(df.timestamp_first_active.astype(str).apply(
            lambda x: list(map(int, [x[:4], x[4:6], x[6:8], x[8:10], x[10:12], x[12:14]]))).values)
        df['tfa_year'] = tfa[:, 0]
        df['tfa_month'] = tfa[:, 1]
        df['tfa_day'] = tfa[:, 2]

        # Age - set any outlier values to -1 (median/avg apparently made it worse)
        av = df.age.values
        df['age'] = np.where(np.logical_or(av < 14, av > 100), -1, av)

        # One-hot-encoding features
        ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
                     'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
        df = self.one_hot_encode_features(df, ohe_feats, drop_raw)

        # drop all of the raw columns after processing them
        self.dropped_raw_features = ohe_feats + ['date_account_created', 'timestamp_first_active']
        if drop_raw:
            df = df.drop(self.dropped_raw_features, axis=1)
        return df

    def one_hot_encode_features(self, data, ohe_features, drop_raw=False):
        df = data.copy()
        for f in ohe_features:
            df_dummy = pd.get_dummies(df[f], prefix=f)
            if drop_raw:
                df = df.drop([f], axis=1)
            df = pd.concat((df, df_dummy), axis=1)
        return df

    def split(self, data, index_train):
        df = data.copy()
        X_train = df.iloc[:index_train]
        X_test = df.iloc[index_train:]
        # drop id from training set
        X_train = X_train.drop('id', axis=1)
        print('SUCCESS: Dataset split')
        return X_train, X_test

class AirBnBDataset(BaselineDataset):
    def __init__(self, data, process_data=False):
        if process_data:
            data = self.process(data.copy())
            print('SUCCESS: Processed')

        if isinstance(data, BaselineDataset):
            self.dropped_raw_features = data.dropped_raw_features
            data = data.data

        self.data = self.feature_engineer(data)
        print('SUCCESS: Features engineered')

    def feature_engineer(self, data):
        df = data.copy()

        #=== Feature engineering ===#
        # get seasons
        time_col = 'timestamp_first_active'
        datefmt = '%Y%m%d%H%M%S'
        seasons = ['winter', 'winter', 'spring',
                   'spring', 'spring', 'summer', 'summer',
                   'summer', 'fall', 'fall', 'fall', 'winter']
        month_to_season = dict(zip(range(1, 13), seasons))
        if not np.issubdtype(df[time_col].dtype, np.datetime64):
            df[time_col] = pd.to_datetime(df[time_col], format=datefmt)
        df['tfa_seasons'] = df[time_col].dt.month.map(month_to_season)

        ohe_features = ['tfa_seasons']
        df = self.one_hot_encode_features(df, ohe_features, drop_raw=True)

        df = df.drop(self.dropped_raw_features, axis=1)
        return df

if __name__ == '__main__':
    PRE = 'processed'
    do_baseline = False

    df_train = read_in_dataset(CSV_FNAMES['train'])
    df_test = read_in_dataset(CSV_FNAMES['test'])
    idx_train = df_train.shape[0]
    df = pd.concat((df_train, df_test), axis=0, ignore_index=True, sort=True)

    drop_raw = True if do_baseline else False
    dataset = BaselineDataset(data=df, drop_raw=drop_raw)

    if not do_baseline:
        PRE = 'feature_eng'
        dataset = AirBnBDataset(dataset)
    X_train, X_test = dataset.split(data=dataset.data, index_train=idx_train)

    output_fname = CSV_FNAMES['train'].split('.csv')[0] + '-{}.csv'.format(PRE)
    X_train.to_csv(output_fname, index=False)
    output_fname = CSV_FNAMES['test'].split('.csv')[0] + '-{}.csv'.format(PRE)
    X_test.to_csv(output_fname, index=False)
    print('Complete')
