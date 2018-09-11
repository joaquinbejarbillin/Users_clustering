
import pandas as pd
from sqlalchemy import create_engine

import matplotlib.pyplot as plt
#from pandas.tools.plotting import scatter_matrix
from pandas.plotting import scatter_matrix

import collections
import numpy as np

import dill

host = 'prod-pentaho.cxfaihg8elfv.eu-west-1.rds.amazonaws.com'
db = 'billin_prod'
user = 'billin'
password = 'ThisIsTheRiverOfTheNight'






class DB():

    def __init__(self, user=None, password=None, host=None, db=None):

        __host = 'prod-pentaho.cxfaihg8elfv.eu-west-1.rds.amazonaws.com'
        __db = 'billin_prod'
        __user = 'billin'
        __password = 'ThisIsTheRiverOfTheNight'

        if user is None:
            self.user = __user
        else:
            self.user = user

        if password is None:
            self.password = __password
        else:
            self.password = password

        if host is None:
            self.host = __host
        else:
            self.host = host

        if db is None:
            self.db = __db
        else:
            self.db = db

        self.engine = create_engine(
            'postgresql://{}:{}@{}:5432/{}'.format(self.user, self.password, self.host, self.db))

    def gettable(self, table):
        # contacts = pd.read_sql_query('select * from "contacts"',con=engine)
        # users = pd.read_sql_query('select * from "users"',con=engine)
        # sessions = pd.read_sql_query('select * from "sessions"',con=engine)
        # premiums = pd.read_sql_query('select * from "premiums"',con=engine)
        # gocardlesses = pd.read_sql_query('select * from "gocardlesses"',con=engine)
        # campaigns = pd.read_sql_query('select * from "campaigns"',con=engine)
        # campaign_details = pd.read_sql_query('select * from "campaign-details"',con=engine)
        # businessesUsers = pd.read_sql_query('select * from "businessesUsers"',con=engine)
        # businesses = pd.read_sql_query('select * from "businesses"',con=engine)
        # businessConstacts = pd.read_sql_query('select * from "businessContacts"',con=engine)
        # bankAccounts = pd.read_sql_query('select * from "bankAccounts"',con=engine)
        # addresses = pd.read_sql_query('select * from "addresses"',con=engine)

        return pd.read_sql_query('select * from "{}"'.format(table), con=self.engine)

    def filter(self, table, filter):

        return pd.read_sql_query('select * from "{}" where {}'.format(table,filter), con=self.engine)



def ScatterMatrix(df):

    _, ax = plt.subplots(figsize=(20, 20))

    _ = scatter_matrix(df.dropna(axis=0,how='any'), alpha=0.4, diagonal='kde', ax=ax)


def rulePCA(X,  n=2):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n)
    try:
        datapca = pca.fit_transform(X)
    except:
        print(X)
        raise

    return datapca, pca


def moving_average(data, window_size):
    """ Computes moving average using discrete linear convolution of two one dimensional sequences.
    Args:
    -----
            data (pandas.Series): independent variable
            window_size (int): rolling window size

    Returns:
    --------
            ndarray of linear convolution

    References:
    ------------
    [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.
    [2] API Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html

    """
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')


def explain_anomalies(y, window_size, sigma=1.0):
    """ Helps in exploring the anamolies using stationary standard deviation
    Args:
    -----
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma (int): value for standard deviation

    Returns:
    --------
        a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
        containing information about the points indentified as anomalies

    """
    avg = moving_average(y, window_size).tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    std = np.std(residual)
    return {'standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i) for
                                                       index, y_i, avg_i in izip(count(), y, avg)
                                                       if
                                                       (y_i > avg_i + (sigma * std)) | (y_i < avg_i - (sigma * std))])}


def explain_anomalies_rolling_std(y, window_size, sigma=1.0):
    """ Helps in exploring the anamolies using rolling standard deviation
    Args:
    -----
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma (int): value for standard deviation

    Returns:
    --------
        a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
        containing information about the points indentified as anomalies
    """
    avg = moving_average(y, window_size)
    avg_list = avg.tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    testing_std = pd.rolling_std(residual, window_size)
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = testing_std_as_df.replace(np.nan,
                                            testing_std_as_df.ix[window_size - 1]).round(3).iloc[:, 0].tolist()
    std = np.std(residual)
    return {'stationary standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i)
                                                       for index, y_i, avg_i, rs_i in izip(count(),
                                                                                           y, avg_list, rolling_std)
                                                       if (y_i > avg_i + (sigma * rs_i)) | (
                                                                   y_i < avg_i - (sigma * rs_i))])}


# This function is repsonsible for displaying how the function performs on the given dataset.
def plot_results(x, y, window_size, sigma_value=1,
                 text_xlabel="X Axis", text_ylabel="Y Axis", applying_rolling_std=False):
    """ Helps in generating the plot and flagging the anamolies.
        Supports both moving and stationary standard deviation. Use the 'applying_rolling_std' to switch
        between the two.
    Args:
    -----
        x (pandas.Series): dependent variable
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma_value (int): value for standard deviation
        text_xlabel (str): label for annotating the X Axis
        text_ylabel (str): label for annotatin the Y Axis
        applying_rolling_std (boolean): True/False for using rolling vs stationary standard deviation
    """
    plt.figure(figsize=(15, 8))
    plt.plot(x, y, "k.")
    y_av = moving_average(y, window_size)
    plt.plot(x, y_av, color='green')
    # plt.xlim(0, 1000)
    plt.xlabel(text_xlabel)
    plt.ylabel(text_ylabel)

    # Query for the anomalies and plot the same
    events = {}
    if applying_rolling_std:
        events = explain_anomalies_rolling_std(y, window_size=window_size, sigma=sigma_value)
    else:
        events = explain_anomalies(y, window_size=window_size, sigma=sigma_value)

    x_anomaly = np.fromiter(events['anomalies_dict'].keys(), dtype=int, count=len(events['anomalies_dict']))
    y_anomaly = np.fromiter(events['anomalies_dict'].values(), dtype=float, count=len(events['anomalies_dict']))
    # plt.plot(x_anomaly, y_anomaly, "r*", markersize=12)

    # add grid and lines and enable the plot
    plt.grid(True)
    plt.show()

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        dill.dump(obj, output)

def load_object(filename):
    with open(filename, 'rb') as file:
        obj = dill.load(file)
    return obj