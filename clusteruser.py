from __future__ import division
import pandas as pd
import os
# Import libraries necessary for this analysis
from itertools import  count

#import renders as rs
from library import *


import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import pandas as pd

from random import randint


izip = zip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



#host = 'prod-pentaho.cxfaihg8elfv.eu-west-1.rds.amazonaws.com'
host = '10.0.5.209'
db = 'billin_prod'
user = 'billin'
password = 'ThisIsTheRiverOfTheNight'

db = DB(host=host, db=db, user=user, password=password)

df = db.gettable('users')
df.set_index('id',inplace=True)
columns = df.describe().columns.tolist()

print('Saving DF')
save_object(df, './df.pkl')

temp = df.drop(labels=['password', 'email', 'created_at', 'updated_at', 'locked', 'vip', 'third_party_sharing_permission', 'name', 'lastname'], axis=1)
temp['last_seen'] = temp.last_seen.map(lambda c: int(c.timestamp()) if c is not pd.NaT else 0 )
temp['deleted_at'] = temp.deleted_at.map(lambda c: int(c.timestamp()) if c is not pd.NaT else 0 )

yesterday = int(pd.datetime.today().timestamp()) - 86400
lastweek = int(pd.datetime.today().timestamp()) - (86400 * 7)
lastmonth = int(pd.datetime.today().timestamp()) - (86400 * 30)
threemonths = int(pd.datetime.today().timestamp()) - (86400 * 90)
sixmonths = int(pd.datetime.today().timestamp()) - (86400 * 180)

temp['last_seen_yesterday'] = temp.last_seen.map(lambda c: 1 if (c - yesterday) >= 0 else 0)
temp['last_seen_lastweek'] = temp.last_seen.map(lambda c: 1 if (c - lastweek) >= 0 else 0)
temp['last_seen_lastmonth'] = temp.last_seen.map(lambda c: 1 if (c - lastmonth) >= 0 else 0)
temp['last_seen_threemonths'] = temp.last_seen.map(lambda c: 1 if (c - threemonths) >= 0 else 0)
temp['last_seen_sixmonths'] = temp.last_seen.map(lambda c: 1 if (c - sixmonths) >= 0 else 0)

temp['deleted_at_yesterday'] = temp.deleted_at.map(lambda c: 1 if (c - yesterday) >= 0 else 0)
temp['deleted_at_lastweek'] = temp.deleted_at.map(lambda c: 1 if (c - lastweek) >= 0 else 0)
temp['deleted_at_lastmonth'] = temp.deleted_at.map(lambda c: 1 if (c - lastmonth) >= 0 else 0)
temp['deleted_at_threemonths'] = temp.deleted_at.map(lambda c: 1 if (c - threemonths) >= 0 else 0)
temp['deleted_at_sixmonths'] = temp.deleted_at.map(lambda c: 1 if (c - sixmonths) >= 0 else 0)
temp.drop(columns=['last_seen', 'deleted_at'], inplace=True)

temp.activated = temp.activated.astype(int)
temp.email_invoice_received = temp.email_invoice_received.astype(int)
temp.email_business_role_change = temp.email_business_role_change.astype(int)
temp.email_business_user_added = temp.email_business_user_added.astype(int)
temp.email_chat_new_messages = temp.email_chat_new_messages.astype(int)
temp.email_contact_new_request = temp.email_contact_new_request.astype(int)
temp.email_invoice_due_dates = temp.email_invoice_due_dates.astype(int)
temp.email_invoice_status_changed = temp.email_invoice_status_changed.astype(int)

temp.services_communication_permission = temp.services_communication_permission.astype(int)

temp.avatar = temp.avatar.map(lambda a: 0 if a == 'default.png' else a)
temp.avatar = temp.avatar.map(lambda a: 1 if a is None else a)
temp.avatar = temp.avatar.map(lambda a: 2 if a != 0 and a != 1 else a)

temp = pd.concat([temp, pd.get_dummies(temp.signup_from, prefix='signup_from_')], axis=1)
temp.drop(columns='signup_from', inplace=True)

temp = pd.concat([temp, pd.get_dummies(temp.sourced, prefix='sourced_')], axis=1)
temp.drop(columns='sourced', inplace=True)

signup_by = pd.get_dummies(temp.signup_by, dummy_na=True)
signup_by.columns = ['signup_by_A', 'signup_by_B', 'signup_by_NaN']
temp = pd.concat([temp, signup_by], axis=1)
temp.drop(columns='signup_by', inplace=True)

expo_token = pd.get_dummies(temp.expo_token, dummy_na=True)
expo_token.columns = ['expo_token_A', 'expo_token_B', 'expo_token_C','expo_token_D','expo_token_E','expo_token_NaN']
temp = pd.concat([temp, expo_token], axis=1)
temp.drop(columns='expo_token', inplace=True)

temp.referrer = temp.referrer.map(lambda r: 0 if r is None else 1)

indexes = temp.index

print('creating PCA')
X2,pca  = rulePCA(temp, n=2)
#dforig = df[columns].copy()
df2pca = pd.DataFrame(X2,columns=['d1','d2'])

X3,pca  = rulePCA(temp, n=3)
#dforig = df[columns].copy()
df3pca = pd.DataFrame(X3,columns=['d1','d2', 'd3'])

rng = np.random.RandomState(42)

# Generate train data
X_train = df2pca[['d1', 'd2']].as_matrix()
X_test = df2pca.sample(n=1000).as_matrix()
X_outliers = df2pca.sample(n=100).as_matrix()

print('IsolationForest')
# fit the model
clf = IsolationForest(n_estimators=50, max_samples=50, random_state=rng, n_jobs=48, contamination=0.07)
clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

__space_size = 300
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), __space_size),            np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), __space_size))

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

df = pd.DataFrame(X_train)
df['output'] = y_pred_train

inliers = df.loc[df['output'] == 1].drop(columns='output').as_matrix()
outliers = df.loc[df['output'] == -1].drop(columns='output').as_matrix()

rng = np.random.RandomState(42)

# Generate train data
X_train = df3pca.as_matrix()
X_test = df3pca.sample(n=1000).as_matrix()
X_outliers = df3pca.sample(n=100).as_matrix()

# fit the model
clf = IsolationForest(n_estimators=50, max_samples=50, random_state=rng, n_jobs=48, contamination=0.07)
clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

print('Outliers')

df = pd.DataFrame(X_train)

df['outlier'] = y_pred_train
df.columns = ['d1', 'd2', 'd3','outlier']
df.index=indexes

inliers = df.loc[df['outlier'] == 1].drop(columns='outlier').as_matrix()
outliers = df.loc[df['outlier'] == -1].drop(columns='outlier').as_matrix()

print('Creating Clustering')
#sampledf3pca = df3pca.sample(n=1700)
sampledf3pca = df3pca.copy()
#db = DBSCAN(eps=0.3, min_samples=500, algorithm='kd_tree', n_jobs=-1).fit(sampledf3pca)
db = SpectralClustering(n_clusters=3, eigen_solver='arpack', affinity="nearest_neighbors", n_jobs=-1).fit(sampledf3pca)
#db = KMeans(n_clusters=3, precompute_distances=True, n_jobs=-1).fit(sampledf3pca)


print('Saving Clusters')
save_object(db, './Cluster.pkl')

labels = db.labels_
sampledf3pca['Cluster'] = pd.Series(labels, index=sampledf3pca.index)
Cluster0 = sampledf3pca.loc[sampledf3pca['Cluster'] == 0].drop(columns='Cluster').as_matrix()
Cluster1 = sampledf3pca.loc[sampledf3pca['Cluster'] == 1].drop(columns='Cluster').as_matrix()
Cluster2 = sampledf3pca.loc[sampledf3pca['Cluster'] == 2].drop(columns='Cluster').as_matrix()

save_object(sampledf3pca, './sampledf3pca.pkl')
save_object(temp, './temp.pkl')


