from datetime import datetime
from keras.applications import vgg16, resnet50, inception_v3
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense
from keras.models import load_model, Model
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
from tensorflow.python.client import device_lib

import loader
import logging
import pandas as pd
import numpy as np
import os

_data_path = "./data/{}"
_data_processed_path = "./data/processed/{}.hdf5"
_model_path = "./models/initial/{}.h5"

def _get_data(model, hdf5, test_set, preprocess=True):
	if hdf5:
		logging.info("Loading data " + _data_processed_path.format(hdf5))
		X, y = loader.load_from_hdf5(_data_processed_path.format(hdf5), test_set)
	else:
		logging.info("Loading labeled images: ./data/imfdb_total.csv")
		X, y = loader.load_data("./data/imfdb_total.csv", test_set)
	if not preprocess:
		return train_test_split(X, y, test_size=0.33)
	if "vgg16" in model:
		X = vgg16.preprocess_input(X)
	elif "rn50" in model:
		X = resnet50.preprocess_input(X)
	elif "iv3" in model:
		X = inception_v3.preprocess_input(X)
	else:
		X = vgg16.preprocess_input(X)
	return train_test_split(X, y, test_size=0.33)

def _get_model(model):
	default_message = "Model not specified"
	if model:
		try:
			logging.info("Loading model from " + _model_path.format(model))
			my_model = load_model(_model_path.format(model))
			return my_model
		except Exception as err:
			logging.error(err)
			default_message = "Cannot load model"
	# If valid model file not provided, create default model
	logging.info(default_message + ", creating default VGG16")
	base_model = VGG16(weights='imagenet', include_top=False)
	input = Input(shape=(224,224,3), name='image_input')
	for layer in base_model.layers:
    		layer.trainable=False

	output_model = base_model(input)
	x = Flatten(name='flatten')(output_model)
	x = Dense(1, activation='sigmoid', name='logistic')(x)  # 2 = binary

	my_model = Model(input=input, output=x)
	my_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return my_model

def decode_predictions(predictions, multiclass=False):
	if multiclass:
		return [np.argmax(p) for p in predictions]  ## ?
	return [round(p[0]) for p in predictions]

def save_model(model, tag):
	path = "./models/trained/"
	now = datetime.now().strftime("%Y_%m_%d_%H%M")
	if tag:
		path += tag + "_" + now + ".h5"
	else:
		path += now + ".h5"
	model.save(path)

def log_metrics(my_model, X, y_true):
	y_pred = decode_predictions(my_model.predict(X))
	logging.info("Accuracy score:")
	logging.info(accuracy_score(y_true, y_pred))
	logging.info("Recall score:")
	logging.info(recall_score(y_true, y_pred))
	logging.info("Precision score:")
	logging.info(precision_score(y_true, y_pred))
	logging.info("ROC AUC:")
	logging.info(roc_auc_score(y_true, y_pred))

def run(model=None, hdf5=None, test_set=None, epochs=4, gpus=1, backend="tensorflow", preprocess=True):
	# TODO: load params from config file
	logging.basicConfig()
	logging.getLogger().setLevel(logging.INFO)

	# Set backend
	if backend == "tensorflow":
		os.environ['KERAS_BACKEND'] = 'tensorflow'
	elif backend == "theano":
		os.environ['KERAS_BACKEND'] = 'theano'

	# Check for gpu
	local_devices = device_lib.list_local_devices()
	if gpus == "max":
		gpus = len(local_devices)
	elif gpus > len(local_devices):
		logging.info(local_devices)
		logging.info("Not enough GPUs detected")
		return
	
	# Features + target
	X_train, X_test, y_train, y_test = _get_data(model, hdf5, test_set, preprocess)

	# Initialize model
	my_model = _get_model(model)
	my_model.summary()

	# Train
	logging.info("Train model")
	my_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

	# Metrics
	logging.info("Calculating metrics (TRAIN)...")
	log_metrics(my_model, X_train[:500], y_train[:500])
	logging.info("Calculating metrics (TEST)...")
	log_metrics(my_model, X_test, y_test)

	return my_model

