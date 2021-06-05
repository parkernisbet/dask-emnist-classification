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
mentioned 39 characters.

Our work won't include the entire image set, but rather only a subset. The 
full dataset suffers from severe class imbalance, so we will be limiting the 
loading of images to keep all classes equivalent.'''

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
from dask_ml.model_selection import train_test_split
from dask_ml.naive_bayes import GaussianNB
from dask_ml.preprocessing import StandardScaler
from hyperopt import fmin, hp, STATUS_OK, tpe
from joblib import delayed, Parallel, parallel_backend
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, f1_score
from subprocess import check_call
from zipfile import ZipFile

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

# %%
# checking class imbalance
counter = lambda x: len(os.listdir(x))
dirs = sorted(os.listdir('Train'))
train_counts = {}
val_counts = {}
for dir in dirs:
    train_counts[dir] = len(os.listdir(f'Train/{dir}'))
    val_counts[dir] = len(os.listdir(f'Validation/{dir}'))

# %%
# visualizing dirstributions
for end, full in zip(['train', 'val'], ['Training', 'Validation']):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    exec(f'ax.bar({end}_counts.keys(), {end}_counts.values())')
    ax.set_title(f'{full} Class Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    plt.show()

# %% [md]
'''In both the training and validation directories there look to be rather 
serious class imbalances, centered mostly on numbers 1 - 9 (excluding 7). 
Some sklearn modules do have the ability to counteract this with a built in 
'class_weight' parameter (by inverse weighting majority classes during 
training), though to be safe we only going to load in enough images so that all 
classes remain equal.'''

# %% [md]
'''The 'to_array' function below pads all images that do not match a size of 
(32, 32) with a 2px border.'''

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

    limit = min(train_counts.values()) if path == 'Train' else \
        min(val_counts.values())
    path = path + '/' if path[-1] != '/' else path
    children = os.listdir(path)
    imgs = []
    labs = []
    for dir in children:
        files = os.listdir(path + dir)
        files = random.sample(files, limit)
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
'''In total we will be working with 170820 images, quite a bit less than the 
original dataset. Outside of this project, something to explore here would be 
to try to generate more images from the original set, albeit with minor 
transformations to give the illusion of uniqueness (shifts, skews, scales, 
etc.). This would allow for us to take advantage of a larger portion of the 
initial image set, though this could lead to overfitting in the long run.'''

# %%
# train test split
X_train, X_test, y_train, y_test = train_test_split(X_train.compute(), \
    y_train, shuffle = True, test_size = .15)
X_train = da.from_array(X_train, chunks = (15625, 1024))
X_test = da.from_array(X_test, chunks = (15625, 1024))

# %% [md]
# ### Exploratory Data Analysis <a class="anchor" id="s2"></a>

# %%
# creating dask client
client = distributed.client._get_global_client() or \
    distributed.Client(n_workers = 2, processes = False)
print(client)

# %%
# dataset describe
ind = [random.randint(0, 1023) for x in range(12)]
ind.sort()
print(pd.DataFrame(X_train[:, np.r_[ind]].compute(), \
    columns = [f'feature {str(x)}' for x in ind]).describe())

# %% [md]
'''Although the above print is only for 12 random array columns, it is 
still rather representative of most other columns. The array is a "mostly" 
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
classes = train_counts.keys()
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
'''Note that some letters are a combination of both lower and upper case 
characters. This should make for some added complexity when trying to predict 
from our constructed models.'''

# %% [md]
# ### Feature Space Reduction <a class="anchor" id="s3"></a>

# %% [md]
'''Normally there are quite a few methods we could use from sklearn to reduce 
our total features, though not all of these are easily parallelizable 
with Dask. Lucky for us, dask_ml includes a pre-built version of sklearn's 
'IncrementalPCA' module that plays nice with Dask's backend. The only caveat is 
that the dask_ml implementation doesn't completely scale the input data (only a 
mean centering), so we will be scaling it first.'''

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

# %% [md]
'''PCA allows for a more than 50% reduction in feature count, while only 
incurring a 5% loss in explained variance. For datasets orders of magnitude 
larger than ours, this would have critical impacts on training times and 
storage size.'''

# %%
# refitting pca model for .95
ipca = IncrementalPCA(n_components = ns[0], batch_size = 15625)
t1 = time.time()
with parallel_backend('dask'):
    ipca.fit(X_train)
t2 = time.time()
print(f'Fit time: {t2 - t1}')

# %%
# reducing dask arrays
X_train = ipca.transform(X_train).rechunk('auto')
X_test = ipca.transform(X_test).rechunk('auto')
X_val = ipca.transform(X_val).rechunk('auto')

# %%
# saving data arrays to pickles
pickle.dump(X_train, open('X_train.pkl', 'wb'))
pickle.dump(y_train, open('y_train.pkl', 'wb'))
pickle.dump(X_test, open('X_test.pkl', 'wb'))
pickle.dump(y_test, open('y_test.pkl', 'wb'))
pickle.dump(X_val, open('X_val.pkl', 'wb'))
pickle.dump(y_val, open('y_val.pkl', 'wb'))

# %% [md]
# ### Classification / Evaluation <a class="anchor" id="s4"></a>

# %% [md]
'''This section is more of a three-for-one, as it includes classifier 
construction, hyperparameter tuning to lock in an optimal configuration, and 
test set scoring.'''

# %%
# loading variables from disk
for var in ['X_train', 'y_train', 'X_val', 'y_val']:
    exec(f'{var} = pickle.load(open("{var}.pkl", "rb"))')

# %%
# model accuracies
accs = {}
times = {}

# %%

# %%
# naive bayes classifier
gnb = GaussianNB()
t1 = time.time()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test).compute()
accs['gnb'] = (f1_score(y_pred, y_test, average = 'macro', zero_division = 0), \
    accuracy_score(y_pred, y_test))
t2 = time.time()
times['gnb'] = t2 - t1
print(classification_report(y_pred, y_test, zero_division = 0))

# %%
'''The accuracy for the above two models are low, but this can be attributed to 
limited samples from each class. The 
'''

# %%
import lightgbm
from dask.diagnostics import ProgressBar

dlgb = lightgbm.DaskLGBMClassifier(max_depth = 8, tree_learner = 'data', \
    n_estimators = 50)

with ProgressBar():
    dlgb.fit(X_train, da.from_array(y_train, chunks = 31250))

# %%
print(classification_report(y_val, y_pred))

# %%
# closing dask client
client.close()
