import os
import pandas as pd

def read_in_dataset(csv_fname, raw=False, verbose=False):
    """ Read in one of the Salary datasets
    Args:
        csv_fname (str): Abs path of the dataset (e.g. test_features.csv, train_features.csv)
        raw (bool): Flag for raw or processed data. Default is raw
        verbose (bool): Print out verbosity
    Returns:
        pd.DataFrame: dataset
    """
    df = pd.read_csv(csv_fname)
    if verbose:
        print('\n{0:*^80}'.format(' Reading in the dataset '))
        print("\nit has {0} rows and {1} columns".format(*df.shape))
        print('\n{0:*^80}\n'.format(' It has the following columns '))
        print(df.info())
        print('\n{0:*^80}\n'.format(' The first 5 rows look like this '))
        print(df.head())
    return df
