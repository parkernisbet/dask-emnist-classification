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
Dask-parallelized support vector machines. The dataset was sourced from 
[Kaggle](https://www.kaggle.com/vaibhao/handwritten-characters) and is a 
semi-subset of the more well known [Extended MNIST]
(https://www.nist.gov/itl/products-and-services/emnist-dataset) 
(EMNIST) database. It includes just north of 850,000 handwritten digits, spread 
across 39 unique characters: all 26 capitalized English alphabet letters 
(A - Z), 9 real numbers (1 - 9), and 4 special characters (@, #, $, &). Note 
that this dataset's author merged the two categories 'O' (letter) and '0' 
(number) to reduce misclassifiations. The images have already been divided into 
train and validation folders, each containing subdirectories for all of the 
above mentioned 39 characters. In contrast to our prior work classifying MNIST 
numerical digits, this database can be viewed as a multi-faceted data volume 
explosion: 

    - a 12 fold increase in datapoints
    - a 3.6 fold increase in classes
    - a 1.3 fold increase in image size'''

# %% [md]
# ### TOC:
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
import seaborn as sns
import time
from dask import array as da, dataframe as ddf, distributed
from joblib import delayed, Parallel
from dask_ml.decomposition import TruncatedSVD
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
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
    Loads images from directory into numpy array.

        Arguments:
            path (string): path to directory to be indexed

        Returns:
            images (array): n x d array of flattened images
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
print(pd.DataFrame(X_train[:, :4].compute()).describe())

# %% [md]
'''Although the above print is only for the first four columns of the 
dataframe, it is rather representative of most columns. The dataframe is a 
sparse matrix with values mostly ranging from 0 to 255, some columns have max 
values lower than this.'''

# %% [md]
# plotting mean character plots
# indices = lambda x: np.where(y_train == x)
# mean_row = lambda x: np.mean(x, axis = 0)
# means = Parallel(n_jobs = -1)(delayed(mean_row)(
#     X_train[indices(cl)].compute()) for cl in classes)
# fig, ax = plt.subplots(8, 5)
# fig.set_size_inches(12, 20)
# means.append(np.array([0]*1024))
# for num, arr in enumerate(means):
#     plt.subplot(8, 5, num + 1)
#     s = sns.heatmap(arr.reshape((32, 32)), cmap = 'binary_r', cbar = False, \
#         xticklabels = [], yticklabels = [])
# plt.show()

# %% [md]
# ### Feature Space Reduction <class="anchor" id="s3"></a>

# %%
client.close()

# %%
