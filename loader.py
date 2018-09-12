from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import h5py
import logging
import numpy as np
import pandas as pd
import pickle

_PATH = "./data/"

def save_to_pickle(test_set=None):
	X_train, X_test, y_train, y_test = load_data(test_set)
	with open("data/X_train.pkl", "wb") as file:
    		pickle.dump(X_train, file)
	with open("data/X_test.pkl", "wb") as file:
    		pickle.dump(X_test, file)
	with open("data/y_train.pkl", "wb") as file:
    		pickle.dump(y_train, file)
	with open("data/y_test.pkl", "wb") as file:
    		pickle.dump(y_train, file)

def load_from_pickle():
	X_train = pickle.load("data/X_train.pkl")
	X_test = pickle.load("data/X_test.pkl")
	y_train = pickle.load("data/y_train.pkl")
	y_test = pickle.load("data/y_test.pkl")
	return X_train, X_test, y_train, y_test

def save_to_hdf5(X, y, filepath="./data/processed/data.hdf5"):
	hfile = h5py.File(filepath, "w", libver='latest')
	hfile.create_dataset("X", data=X)
	hfile.create_dataset("y", data=y)
	hfile.close()

def load_from_hdf5(filepath="./data/processed/data.hdf5", test_set=None):
	hfile = h5py.File(filepath, "r", libver='latest')
	if test_set:
		return hfile['X'][:test_set], hfile['y'][:test_set]
	return hfile['X'][:], hfile['y'][:]

def load_from_csv(filepath="./data/imfdb_total.csv", test_set=None):
	observations = pd.read_csv(filepath, index_col=0)

	# Get test subset
	if test_set:
		df_unhappy = observations[observations['emotion'] != 'HAPPINESS'].sample(test_set)
		df_happy = observations[observations['emotion'] == 'HAPPINESS'].sample(test_set)
		observations = pd.concat([df_unhappy, df_happy])

	# Target
	y = np.where(observations.emotion == 'HAPPINESS', 1, 0)

	# Features (photos from paths)
	X = get_input(observations['path'].values)
	return X, y

def get_input(img_paths):
    X = np.empty((0, 224, 224, 3))
    pbar = tqdm(total=len(img_paths), dynamic_ncols=True)
    for img_path in img_paths:
        img = imread(img_path)
        img = resize(img, (224, 224), preserve_range=True).astype(np.float32)
        img = img/img.max()*255
        img.astype(int)
        img = np.swapaxes(img, 0, 1)
        img = np.expand_dims(img, axis=0)
        X = np.concatenate((X, img), axis=0)
        pbar.update(1)
    pbar.close()
    return X
