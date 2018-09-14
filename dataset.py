import emotion_codes
import extracter2
import h5py
import logging
from glob import glob
from sklearn.model_selection import train_test_split

# X[][][] = [R, G, B] = a pixel
# X[][] = [pixel] * 224 = a row
# X[] = [pixel] * 224 * 224 = the image
# X = set of images

class Dataset:
	"""70% train, 30% test"""
	train = {}
	test = {}

	def __init__(self, dir="./data/processed/default", encoding="multiclass", class_size=None):
		X = []
		label = []
		for file in glob(dir+"/*.h5"):
			with h5py.File(file, "r") as hfile:
				X_ = hfile['X'][:class_size]
				file_ = hfile['file'][:class_size]
				label_ = hfile['label'][:class_size]
				if len(X_) < class_size:
					logging.warning("Class size {} but only {} rows".format(class_size, len(X_)))
				X.extend(list(zip(file_, X_)))
				label.extend(label_)
		X_train, X_test, l_train, l_test = train_test_split(X, label, test_size=0.3)
		self.train['X'] = [x[1] for x in X_train]
		self.train['file'] = [x[0] for x in X_train]
		self.train['y'] = emotion_codes.get_emotion_code(emotion=l_train)
		self.test['X'] = [x[1] for x in X_test]
		self.test['file'] = [x[0] for x in X_test]
		self.test['y'] = emotion_codes.get_emotion_code(emotion=l_test)

	def X_train(self):
		return self.train['X']

	def X_test(self):
		return self.test['X']

	def y_train(self):
		return self.train['y']

	def y_test(self):
		return self.test['y']

	def class_count(self):
		print("Train set")
		print(self.y_train().groupby(by=lambda x: x).count())
		print()
		print("Test set")
		print(self.y_test().groupby(by=lambda x: x).count())

	def augment_test(self, seed=None):
		# TODO: generate images from seeds and add to test set
		self.augment_seed = seed
		pass

