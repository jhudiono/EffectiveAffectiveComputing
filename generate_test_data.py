# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

from glob import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import csv
import pandas as pd
import re

def get_samples(n=3):
	df = pd.read_csv("data/imfdb_total.csv", index_col=0)
	df = df.dropna()
	emotions = ['NEUTRAL', 'ANGER', 'DISGUST', 'FEAR', 'HAPPINESS', 'SADNESS', 'SURPRISE']
	seed_df = []
	for emotion in emotions:
		seed_df.append(df[df['emotion'] == emotion].sample(n))
	return pd.concat(seed_df)

def generate_image(row, datagen, n=5):
	img_path = row['path']
	tag = row['emotion'] + "_" + row['gender'][:2] + row['position'][:2]
	save_dir = 'data/generated'
	save_format = 'jpeg'
	seed_img = load_img(img_path)
	x = img_to_array(seed_img)  # shape (3, 150, 150)
	x = x.reshape((1,) + x.shape)  # shape (1, 3, 150, 150)
	i = 0
	for batch in datagen.flow(x, batch_size=1, save_to_dir='data/generated', save_prefix=tag, save_format=save_format):
		i += 1
		if i > n:
			break

def generate_metadata():
	p = re.compile('data\/generated\/([A-Z]*)_([A-Z]{2})([A-Z]{2})')
	files = glob('data/generated/*')
	rows = []
	for f in files:
		m = p.match(f)
		# gender, emotion, position, path
		rows.append([m.group(2), m.group(1), m.group(3), f])
	with open("data/generated.csv", "w") as output:
		writer = csv.writer(output, delimiter='\t', lineterminator='\n')
		writer.writerows(rows)

def run(n=5):
	datagen = ImageDataGenerator(
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')

	df = get_samples()
	for index, row in df.iterrows():
		try:
			generate_image(row, datagen)
		except Exception as err:
			logging.error(err)
	generate_metadata()

	
	
