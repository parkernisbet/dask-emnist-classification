# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
# ---

# %% {"active": "py"}
# !/usr/bin/env python

# %% [md]
# # Dask EMNIST Classification

# %% [md]
'''This notebook explores handwritten character classification using 
Dask-parallelized gradient boosted decision trees (XGBoost). The dataset was 
sourced from [Kaggle](https://www.kaggle.com/vaibhao/handwritten-characters) 
and is a semi-subset of the more well known 
[Extended MNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) 
(EMNIST) database. It includes just north of 850,000 handwritten digits, spread 
across 39 unique characters: all 26 English alphabet letters (A - Z), 9 real 
numbers (1 - 9), and 4 special characters (@, #, $, &). Note that this 
dataset's author merged the two categories 'O' (letter) and '0' (number) to 
reduce misclassifiations. The images have already been divided into train and 
validation folders, each containing subdirectories for all of the above 
mentioned 39 characters. In contrast to our prior work classifying MNIST 
numerical digits, this database can be viewed as a multi-faceted data volume 
expansion: 

    - a 12 fold increase in datapoints
    - a 3.6 fold increase in classes
    - a 1.3 fold increase in image size'''

# %% [md]
# ### Table of Contents:
# 1. [Data Loading / Cleaning](#s1)
# 2. [Exploratory Data Analysis](#s2)
# 3. [Feature Space Reduction](#s3)
# 4. [Classification / Evaluation](#s4)

# %% [md]
# ### Data Loading / Cleaning <a class="anchor" id="s1"></a>

# %%
# consolidated module imports
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import seaborn as sns
import time
from dask import array as da, distributed
from dask_ml.decomposition import IncrementalPCA
from dask_ml.naive_bayes import GaussianNB
from dask_ml.preprocessing import StandardScaler
from dask_ml.xgboost import XGBClassifier
from joblib import delayed, Parallel, parallel_backend
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix
from subprocess import check_call
from zipfile import ZipFile

# %%

# %% [md]
'''The below cell skip re-downloading the .zip file if said zip file and the 
train / validation folders are present in the current directory. Unzipping the 
dataset may take a while depending upon computer specs, it is expanding from 
1.7GB to a little over 3.3GB.'''

# %%
# downloading and unzipping kaggle images
paths = ['Train', 'Validation', 'handwritten-characters.zip']
checks = [os.path.exists(path) for path in paths]
if set(checks) != {True}:
    cmd = 'kaggle datasets download -d vaibhao/handwritten-characters'
    check_call(cmd, shell = True)
    with ZipFile('handwritten-characters.zip', 'r') as z:
        z.extractall()
    try:
        check_call('rm -r dataset', shell = True)
    except:
        pass

# %% [md]
'''The 'to_array' function pads all images that do not match a size of (32, 32) 
with a 2px border.'''

# %%
# functions to load images
def to_array(full):
    '''
    Reads in an image and returns a, padded if necessary, flattened array.

        Arguments:
            full (string): full string path to image

        Returns:
            arr (array): flattened 1d array representing image
    '''

    img = Image.open(full)
    if img.size != (32, 32):
        img = ImageOps.expand(img, border = 2)
    return np.array(img).ravel()

def load_from_path(path):
    '''
    Loads images from directory into dask array.

        Arguments:
            path (string): path to directory to be indexed

        Returns:
            images (array): n x d dask array of flattened images
            labels (array): n x 1 array of labels

    '''

    path = path + '/' if path[-1] != '/' else path
    children = os.listdir(path)
    imgs = []
    labs = []
    for dir in children:
        files = os.listdir(path + dir)
        imgs.extend(Parallel(n_jobs = -1)(delayed(to_array)(path + dir + '/' \
            + f) for f in files))
        labs.extend([dir]*len(files))
    images = np.vstack(imgs)
    labels = np.array(labs)
    return (da.from_array(images, chunks = (15625, 1024)), labels)

# %%
# timing loading of train data
t1 = time.time()
X_train, y_train = load_from_path('Train')
t2 = time.time()
print(f'Execution time: {t2 - t1}')
print(f'Images loaded: {X_train.shape[0]}')

# %%
# same for validation data
t1 = time.time()
X_val, y_val = load_from_path('Validation')
t2 = time.time()
print(f'Execution time: {t2 - t1}')
print(f'Images loaded: {X_val.shape[0]}')

# %% [md]
# ### Exploratory Data Analysis <a class="anchor" id="s2"></a>

# %%
# creating dask client
client = distributed.client._get_global_client() or \
    distributed.Client(processes = False)
print(client)

# %%
# visualizing training and test class breakdowns
classes, counts = [], []
for end, full in zip(['train', 'val'], ['Train', 'Validation']):
    exec(f'classes, counts = np.unique(y_{end}, return_counts = True)')
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    ax.bar(classes, counts)
    ax.set_title(f'{full} Class Size Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    plt.show()

# %% [md]
'''In both the training and validation dataframes there look to be rather 
serious class imbalances, centered mostly on numbers 1 - 9 and excluding 7. 
Sklearn's SVC classifier has a built in class_weight parameter to combat this, 
though depending on compute times we may seek to manually prune data points 
from majority classes.'''

# %%
# dataset describe
ind = [random.randint(0, 1023) for x in range(12)]
ind.sort()
print(pd.DataFrame(X_train[:, np.r_[ind]].compute(), \
    columns = [f'feature {str(x)}' for x in ind]).describe())

# %% [md]
'''Although the above print is only for 12 random dataframe columns, it is 
still rather representative of most other columns. The dataframe is a "mostly" 
sparse matrix with values "mostly" ranging from 0 to 255, though there are some 
columns that run contrary to this.'''

# %%
# sparsity sanity check
zeros = (X_train == 0).compute().sum()
total = X_train.size
print(f'X_train sparseness: {round(100*zeros/total, 2)}%')
zeros = (X_val == 0).compute().sum()
total = X_val.size
print(f'X_val sparseness: {round(100*zeros/total, 2)}%')

# %%
# plotting mean character plots
indices = lambda x: np.where(y_train == x)
mean_row = lambda x: np.mean(x, axis = 0)
means = Parallel(n_jobs = -1)(delayed(mean_row)(
    X_train[indices(cl)].compute()) for cl in classes)
fig, ax = plt.subplots(8, 5)
fig.set_size_inches(12, 20)
means.append(np.array([0]*1024))
for num, arr in enumerate(means):
    plt.subplot(8, 5, num + 1)
    s = sns.heatmap(arr.reshape((32, 32)), cmap = 'binary_r', cbar = False, \
        xticklabels = [], yticklabels = [])
plt.show()

# %% [md]
# ### Feature Space Reduction <a class="anchor" id="s3"></a>

# %% [md]
'''At the moment, '''

# %%
# scaling arrays
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_val = ss.transform(X_val)

# %%
# out of memory pca decomposition
ipca = IncrementalPCA(n_components = 1024, batch_size = 15625)
t1 = time.time()
with parallel_backend('dask'):
    ipca.fit(X_train)
t2 = time.time()
print(f'Fit time: {t2 - t1}')
ratios = ipca.explained_variance_ratio_

# %%
# function to find optimal component count
def find_n(ratios, tol):
    '''
    Finds minimum number of components required to achieve passed tolerance.

        Arguments:
            ratios (array): svd variance ratios
            tol (float): minimum accumulative explained variance

        Returns:
            n (int): minimum number of components for tolerance
    '''

    low = 0
    high = len(ratios)
    while True:
        ind = (low + high)//2
        if ratios[:ind].sum() > tol:
            if high == ind:
                break
            high = ind
        else:
            if low == ind:
                break
            low = ind
    return high

# %%
# finding optimal n component number
ns = []
for i, tol in enumerate([.95, .99]):
    ns.append(find_n(ratios, tol))
    print(f'Optimal n, tol = {tol}: {ns[i]}')

# %%
# refitting pca model for .95
ipca = IncrementalPCA(n_components = ns[0], batch_size = 15625)
t1 = time.time()
with parallel_backend('dask'):
    ipca.fit(X_train)
t2 = time.time()
print(f'Fit time: {t2 - t1}')

# %%
# transforming dask arrays
X_train = ipca.transform(X_train).rechunk('auto')
X_val = ipca.transform(X_val).rechunk('auto')

# %%
# saving data arrays to pickles
pickle.dump(X_train, open('X_train.pkl', 'wb'))
pickle.dump(y_train, open('y_train.pkl', 'wb'))
pickle.dump(X_val, open('X_val.pkl', 'wb'))
pickle.dump(y_val, open('y_val.pkl', 'wb'))

# %% [md]
# ### Classification / Evaluation <a class="anchor" id="s4"></a>

# %% [md]
'''This section is more of a three-for-one, as it includes construction of the 
underlying XGBoost classifier, incremental hyperparameter tuning to lock in an 
optimal configuration, and then finally scoring using the validation set.'''

# %%
# encoding labels as numbers
maps = {char: num for num, char in enumerate(classes)}
to_num = lambda x: maps[x]
y_train = da.from_array(np.array(list(map(to_num, y_train.flatten()))), \
    chunks = (31250))
y_val = da.from_array(np.array(list(map(to_num, y_val.flatten()))), \
    chunks = (31250))

# %%
# training xgboost classifier
xgb = XGBClassifier(booster = 'gbtree', objective = 'multi:softprob', \
    max_delta_step = 1, eval_metric = 'auc', seed = 42, max_depth = 6, \
    min_child_weight = 2, subsample = .5, num_parallel_trees = 12, \
    use_label_encoder = False, verbosity = 2, colsample_bytree = .5, \
    n_estimators = 200)
# params = {'max_depth': list(range(4, 16)), \
#     'min_child_weight': list(range(1, 20, 2)), \
#     'subsample': np.linspace(.5, 1.0, 10)}
# search = HyperbandSearchCV(xgb, params, max_iter = 30, aggressiveness = 4)
t1 = time.time()
# search.fit(X_train, y_train)
xgb.fit(X_train, y_train, classes = np.unique(y_train))
t2 = time.time()
print(t2 - t1)
print(xgb.best_params_)
pickle.dump(xgb, open('xgb.pkl', 'wb'))

# %%
y_train = pickle.load(open('y_train.pkl'))
y_val = pickle.load(open('y_val.pkl'))

# %%
# naive bayes classifier
gnb = GaussianNB()
t1 = time.time()
gnb.fit(X_train, y_train)
t2 = time.time()

# %%
y_pred = gnb.predict(X_val).compute()

# %%
# random forest classifier
from sklearn.ensemble import RandomForestClassifier

t1 = time.time()
rfc = RandomForestClassifier(n_estimators = 10, max_depth = 12, n_jobs = -1)
rfc.fit(X_train, y_train)
t2 = time.time()
print(t2 - t1, rfc.score(X_val, y_val))

# %%
# closing dask client
client.close()
