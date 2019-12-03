"""Make Airbnb Recommendation dataset

This script features creating datasets in multiple fashions:
- raw baseline features <--- cleaned up data prior to feature engineering
- feature engineered datasets

One can choose between the two by setting the module level constants
```
    do_baseline = False
    do_merged_sessions = True
```

"""
# Standard Dist imports
import os

# Third party imports
import numpy as np
import pandas as pd

# Project level imports
from d_utils import read_in_dataset
from merge_baseline_sessions import mergeBaselineAndSessionFeatures

# Module level constants
DATA_DIR = '../airbnb-recruiting-new-user-bookings'
CSV_FNAMES = {
    'test': os.path.join(DATA_DIR, 'test_users.csv'),
    'train': os.path.join(DATA_DIR, 'train_users_2.csv'),
    'train-processed': os.path.join(DATA_DIR, 'train_users_2-processed.csv'),
    'test-processed': os.path.join(DATA_DIR, 'test_users-processed.csv'),
    'train-feature_eng': os.path.join(DATA_DIR, 'train_users_2-feature_eng.csv'),
    'test-feature_eng': os.path.join(DATA_DIR, 'test_users-feature_eng.csv'),
    'sessions-eng': os.path.join('../sessions-data/sessions-engineered.csv'),
    'train-merged_sessions': os.path.join(DATA_DIR, 'train_users-merged_sessions.csv'),
    'test-merged_sessions': os.path.join(DATA_DIR, 'test_users-merged_sessions.csv')
}

class BaselineDataset():
    """Creates our baseline features from the airbnb dataset

    The BaselineDataset is set up to establish our baseline performance, by
    only using cleaned up features from the airbnb dataset.
    """
    def __init__(self, data, drop_raw=True):
        """Initializes BaselineDataset

        One can access the dataset by acessing its `data` attribute after
        initialization.

        :param data (pd.DataFrame): Dataset to be processed
        :param drop_raw (bool): Flag for dropping the raw features. Default set to True.
        """
        self.data = self.process(data.copy(), drop_raw=drop_raw)
        print('Processed')

    def process(self, data, drop_raw=True, verbose=True):
        """ Processes the dataset

        :param data (pd.DataFrame): Dataset to be processed
        :param drop_raw (bool): Flag for dropping the raw features. Default set to True.
        :param verbose (bool): Flag for verbosity. Default set to True
        :return:
            pd.DataFrame: Processed dataset
        """
        df = data.copy()
        dropped_features = ['date_first_booking']
        df = df.drop(dropped_features, axis=1)
        df = df.fillna(-1)
        # Feature Engineering
        # date_account_created (process the month, year, and day)
        dac = np.vstack(
            df.date_account_created.astype(str)
            .apply(lambda x: list(map(int, x.split("-"))))
            .values
        )
        df["dac_year"] = dac[:, 0]
        df["dac_month"] = dac[:, 1]
        df["dac_day"] = dac[:, 2]

        # timestamp_first_active (process the month, year, and day)
        tfa = np.vstack(
            df.timestamp_first_active.astype(str)
            .apply(
                lambda x: list(
                    map(int, [x[:4], x[4:6], x[6:8], x[8:10], x[10:12], x[12:14]])
                )
            )
            .values
        )
        df["tfa_year"] = tfa[:, 0]
        df["tfa_month"] = tfa[:, 1]
        df["tfa_day"] = tfa[:, 2]

        # Age - set any outlier values to -1 (median/avg apparently made it worse)
        av = df.age.values
        df['age'] = np.where(np.logical_or(av < 14, av > 100), -1, av)

        # One-hot-encoding features
        ohe_feats = [
            "gender",
            "signup_method",
            "signup_flow",
            "language",
            "affiliate_channel",
            "affiliate_provider",
            "first_affiliate_tracked",
            "signup_app",
            "first_device_type",
            "first_browser",
        ]
        df = self.one_hot_encode_features(df, ohe_feats, drop_raw)

        # drop all of the raw columns after processing them
        self.dropped_raw_features = ohe_feats + ['date_account_created', 'timestamp_first_active']
        if drop_raw:
            df = df.drop(self.dropped_raw_features, axis=1)
        return df

    def one_hot_encode_features(self, data, ohe_features, drop_raw=False):
        """One hot encoded features

        :param data (pd.DataFrame): Dataset
        :param ohe_features (list): List of features to be one hot encoded
        :param drop_raw (bool): Flag for dropping the raw features. Default set to True.
        :return:
            pd.DataFrame: One hot encoded features within a Dataframe
        """
        df = data.copy()
        for f in ohe_features:
            df_dummy = pd.get_dummies(df[f], prefix=f)
            if drop_raw:
                df = df.drop([f], axis=1)
            df = pd.concat((df, df_dummy), axis=1)
        return df

    def split(self, data, index_train):
        """ Splits up the dataset into training and testing

        Use case is to combine the training and test set, so we can process
        both datasets in a uniform fashion. This function is to help split them back into
        its original dataset after the preprocessing.

        :param data (pd.DataFrame): Dataset to be split
        :param index_train (int): Training index
        :return:
            pd.DataFrame: Training set
            pd.DataFrame: Test set
        """
        df = data.copy()
        X_train = df.iloc[:index_train]
        X_test = df.iloc[index_train:]
        # drop id from training set
        print('SUCCESS: Dataset split')
        return X_train, X_test

class AirBnBDataset(BaselineDataset):
    """The Feature Engineered AirBnB Dataset

    This class is to help differentiate from our baseline as the main dataset
    that is used to train our recommender system.
    """
    def __init__(self, data, process_data=False):
        """ Initializes AirBnBDataset

        Processes the dataset if it has not been processed yet in the Baseline

        :param data (pd.DataFrame): Dataset to be feature engineered/processed
        :param process_data (bool): Flag for processing the dataset
        """
        # Process the dataset if set to True
        if process_data:
            data = self.process(data.copy())
            print('SUCCESS: Processed')

        # Pass in any planned raw features from the Baseline Dataset
        # that need to be dropped after feature engineering them
        if isinstance(data, BaselineDataset):
            self.dropped_raw_features = data.dropped_raw_features
            data = data.data

        # Feature engineer the dataset
        self.data = self.feature_engineer(data)
        print('SUCCESS: Features engineered')

    def feature_engineer(self, data):
        """ Create newly feature engineered dataset

        Used during initialization of the AirBnB Dataset.
        Seasonal feature engineering is added here.

        :param data (pd.DataFrame): Dataset to be feature engineered
        :return:
            pd.DataFrame: New dataset
        """
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
    do_baseline = False
    do_merged_sessions = True

    # Merge our baseline and sessions dataset if set to true
    # the sessions data is lengthy in generation time, so it lives in its own script
    # aside from our dataset making script. This is to help combine them.
    if do_merged_sessions:
        PRE = 'feature_eng'
        train_input = CSV_FNAMES['train-{}'.format(PRE)]
        test_input = CSV_FNAMES['test-{}'.format(PRE)]
        sessions_input = CSV_FNAMES['sessions-eng']
        print('Merging training set')
        mergeBaselineAndSessionFeatures(train_input, sessions_input, CSV_FNAMES['train-merged_sessions'])
        print('Merging test set')
        mergeBaselineAndSessionFeatures(test_input, sessions_input, CSV_FNAMES['test-merged_sessions'])

    else:
        # Load up our datasets from the csv files
        PRE = 'processed'
        df_train = read_in_dataset(CSV_FNAMES['train'])
        df_test = read_in_dataset(CSV_FNAMES['test'])
        idx_train = df_train.shape[0]
        df = pd.concat((df_train, df_test), axis=0, ignore_index=True, sort=True)

        # Process the dataset into our Baseline
        drop_raw = True if do_baseline else False
        dataset = BaselineDataset(data=df, drop_raw=drop_raw)

        # Feature engineer the dataset if we don't want to settle
        # for the BaselineDataset
        if not do_baseline:
            PRE = 'feature_eng'
            dataset = AirBnBDataset(dataset)

        # split up the dataset back into its training and test given the original indexing
        X_train, X_test = dataset.split(data=dataset.data, index_train=idx_train)

        #=== Save dataset ===#
        output_fname = CSV_FNAMES['train'].split('.csv')[0] + '-{}.csv'.format(PRE)
        X_train.to_csv(output_fname, index=False)
        output_fname = CSV_FNAMES['test'].split('.csv')[0] + '-{}.csv'.format(PRE)
        X_test.to_csv(output_fname, index=False)
        print('Complete')
