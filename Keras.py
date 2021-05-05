from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns

import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
##
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pprint import pprint
##
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV



def get_important_features(transformed_features, components_, columns):
    """
    This function will return the most "important"
    features so we can determine which have the most
    effect on multi-dimensional scaling
    """
    num_columns = len(columns)

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    # Sort each column by it's length. These are your *original*
    # columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    return important_features

np.set_printoptions(precision=12, suppress=True, linewidth=150)
pd.options.display.float_format = '{:.6f}'.format
sns.set()
tf.__version__

databank = pd.read_csv('data.csv',low_memory=False, index_col=0)
databank.columns = databank.columns.str.lower()
databank.columns = databank.columns.str.rsplit('(', n=1).str.get(0)
databank.columns = databank.columns.str.replace(" ", "_")
databank.columns = databank.columns.str.replace("\\.", "")
databank.columns = databank.columns.str.replace("-", "_")
databank.columns = databank.columns.str.rstrip('_')
##
databank = databank.drop(['phase'], axis=1)
databank['viscosity'] = pd.to_numeric(databank['viscosity'],errors = 'coerce')
databank['therm_cond'] = pd.to_numeric(databank['therm_cond'],errors = 'coerce')

databank.isnull().sum().sum()

databank = databank.dropna()
databank.info()
databank.head()

X = databank.drop(labels='fluid', axis=1)

corr = X.corr()
corr

features_corr = ~(corr.mask(np.eye(len(corr), dtype=bool)).abs() > 0.95).any() # 0.95 / 0.99
features_corr

X_good = corr.loc[features_corr, features_corr]
lst_variable_corr = X_good.columns.values.tolist()
X_corr = X[np.intersect1d(X.columns, lst_variable_corr)]

df_corr = X[X_corr.columns]
df_corr

pca = PCA(n_components=12, svd_solver='full')
pca.fit(X)


# In[ ]:


T = pca.transform(X)
T.shape
pca.explained_variance_ratio_

components = pd.DataFrame(pca.components_, columns = X.columns, index=[1,2,3,4,5,6,7,8,9,10,11,12])
components

pca_result = get_important_features(T, pca.components_, X.columns.values)
pca_result = pd.DataFrame(pca_result,columns=['PCA_Value','Variable'])
threshold = 3
pca_result = pca_result[pca_result["PCA_Value"] >= 3]
pca_result

X_pca = pca_result['Variable']
df_pca = X[X_pca]

df_pca

y = databank.fluid
X = df_pca # df_corr

model = Sequential()
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
history = model.fit(X, y, epochs=10, batch_size=10, validation_split=0.2)



