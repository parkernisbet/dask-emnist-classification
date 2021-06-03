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
'''kjkhkjdshfhjkjhlfkjahselkjfhaeslkjfhldskjfhlskdjhflkdsjhflkjdshflksjdhflksjdhflkjsdhflkjsdh
lfkjhsdkjfhlsdkjhflksjdhflksjdhflskjdhflskdjhflksjdhflkjshdflkjsdhflkjsdhflkjsdhfklsjdhflksdjh'''

# %%
# consolidated module imports
import numpy as np
import os
import pickle
import time
from joblib import Parallel, delayed
from PIL import Image, ImageOps
from subprocess import check_call
from zipfile import ZipFile

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
# functions to load images
def to_array(full):
    '''
    Reads in an image from the provided full path and returns a flattened array.
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
        imgs.extend(Parallel(n_jobs = -1)(delayed(to_array)(path + dir + '/' + f) for f in files))
        labs.extend([dir]*len(files))
    images = np.vstack(imgs)
    labels = np.array(labs)
    return (images, labels)

# %%
# timing loading of train data
t1 = time.time()
X_train, y_train = load_from_path('Train')
t2 = time.time()
print(f'Execution time: {t2 - t1}')
print(f'Images loaded: {X_train.shape[0]}')

# %%
# ditto for validation data
t1 = time.time()
X_val, y_val = load_from_path('Validation')
t2 = time.time()
print(f'Execution time: {t2 - t1}')
print(f'Images loaded: {X_val.shape[0]}')

# %%

