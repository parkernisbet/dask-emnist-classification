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

# %%
# consolidated module imports
import numpy as np
import os
import pickle
import time
from PIL import Image
from subprocess import check_call
from zipfile import ZipFile

# %%
# downloading and unzipping kaggle images
paths = ['Train', 'Validation', 'handwritten-characters.zip']
checks = [os.path.exists(path) for path in paths]
if set(checks) == {True}:
    cmd = 'kaggle datasets download -d vaibhao/handwritten-characters'
    check_call(cmd, shell = True)
    with ZipFile('handwritten-characters.zip', 'r') as z:
        z.extractall()
    try:
        check_call('rm -r dataset', shell = True)
    except:
        pass

# %%
# function to load images
def load_from_path(path):
    '''
    Loads images from directory into numpy array.

        Arguments:
            path (string): path to directory to be indexed

        Returns:
            images (array): n x d array of flattened images
            labels (array): n x 1 array of labels

    '''

    global errors

    errors = []
    path = path + '/' if path[-1] != '/' else path
    children = os.listdir(path)
    imgs = []
    for dir in children:
        for img in os.listdir(path + dir):
            tmp = np.array(Image.open(path + dir + '/' + img))
            if tmp.size == 1024:
                imgs.append(tmp.ravel())
            else:
                errors.append(f'{dir}/{img}')
        labs = [dir]*len(imgs)
        print(dir)
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
print(f'Error count: {len(errors)}')
pickle.dump(errors, open('train_load_errors.pkl', 'wb'))

# %%
# ditto for validation data

t1 = time.time()
X_val, y_val = load_from_path('Validation')
t2 = time.time()
print(f'Execution time: {t2 - t1}')
print(f'Images loaded: {X_val.shape[0]}')
print(f'Error count: {len(errors)}')
pickle.dump(errors, open('test_load_errors.pkl', 'wb'))

# %%
''' Adelle, the errors are from images not sized 32 x 32. Full lists of non-conformers can be read from the two pickle files. '''
