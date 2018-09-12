from collections import defaultdict
from glob import glob
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
import logging
import traceback

def get_annotation_files():
    def get_shape(p):
        try:
            return imread(p).shape
        except Exception as err:
            return (err, None, None)
    
    movie_dirs = glob("data/IMFDB_final/*/*")
    
    txt_missing = []
    total_data = []
    pbar = tqdm(total=len(movie_dirs))
    for movie_dir in movie_dirs:
        try:
            metadata = glob(movie_dir+"/*.txt")[0]
            df = pd.read_csv(metadata, sep="\s+", header=None)#[[2, 7, 9, 10, 11, 13, 15]] 
            #df.columns = ['file', 'movie', 'name', 'gender', 'emotion', 'light', 'pose']
            df['path'] = movie_dir + "/images/" + df[2] # 2
            df[['height', 'width', 'channels']] = df['path'].apply(get_shape).apply(pd.Series)
            total_data.append(df)
        except IndexError:
            txt_missing.append(movie_dir)
        except Exception as err:
            print("Skipping", movie_dir, err)
        pbar.update(1)
    print("Could not find .txt file:", len(txt_missing), txt_missing)
    return pd.concat(total_data)#[['height', 'width', 'channels', 7, 9, 10, 11, 13, 15, 'path']]#[['emotion', 'gender', 'pose', 'path']]

def extract_data(filepath="./data/imfdb_total.csv", batch_size=1000, test_set=None, compression='gzip', save_filepath="data/processed/data.hdf5", classes=['HAPPINESS'], debug=False):
	logging.basicConfig()
	logging.getLogger().setLevel(logging.INFO)

	observations = pd.read_csv(filepath, index_col=0)

	# Get test subset
	if test_set:
		batch_size = min(batch_size, test_set)
		dfs = []
		for emotion in classes:
			df_emotion = observations[observations['emotion'] == emotion]
			length = min(test_set, len(df_emotion)-1)
			if length != test_set:
				logging.info(emotion + " class smaller than specified size, max # rows " + str(length))
			dfs.append(df_emotion.sample(length))
		observations = pd.concat(dfs)

	# Initialize h5py file and create datasets
	hfile = h5py.File(save_filepath, "w", libver="latest")
	hfile.create_dataset("X", maxshape=(None, 224, 224, 3), compression=compression,
		chunks=True, shape=(len(observations), 224, 224, 3))
	hfile.create_dataset("y", maxshape=(None,), compression=compression,
		chunks=True, shape=(len(observations),))

	# Write data to h5py file
	X, y = _df_to_hdf5(hfile, observations, batch_size, 0, debug=debug)
	hfile.close()

	return X, y

def append_data(filepath="./data/imfdb_total.csv", start_index=0, end_index=None, batch_size=1000, save_filepath="data/processed/data.hdf5", classes=['HAPPINESS'], debug=False):
	logging.basicConfig()
	logging.getLogger().setLevel(logging.INFO)

	# Get subset of data within range
	observations = pd.read_csv(filepath, index_col=0)
	observations = observations[~observations['path'].isna()].iloc[start_index:end_index]

	# Get h5py file handle
	hfile = h5py.File(save_filepath, "r+", libver="latest")

	# Write data to h5py file
	X, y = _df_to_hdf5(hfile, observations, batch_size, start_index, debug=debug)
	hfile.close()

	return X, y

def get_emotion_code(emotion=None):
	codes = {
		"NEUTRAL": 0,
		"ANGER": 1,
		"DISGUST": 2,
		"FEAR": 3,
		"HAPPINESS": 4,
		"SADNESS": 5,
		"SURPRISE": 6
	}
	codes = defaultdict(int, codes)  # lambda: -1
	if isinstance(emotion, list):
		return [codes[e.upper()] for e in emotion]
	elif emotion:
		return codes[emotion.upper()]
	return list(codes.items())

def _df_to_hdf5(hfile, df, batch_size, start_index, debug=False):
	X = np.empty((0, 224, 224, 3))
	y = np.empty((0,))
	pbar = tqdm(desc="Total rows processed", total=len(df))
	for k, chunk in df.groupby(np.arange(len(df))//batch_size):
		logging.info("Preprocessing batch " + str(k))
		start_index = k*batch_size
		end_index = start_index + len(chunk)
		try:
			X_chunk, y_chunk = _process_chunk(chunk, k, pbar, debug)
			#y_chunk = chunk.emotion.apply(lambda e: get_emotion_code(e))
			logging.info("Writing X dataset " + str(start_index) + " -> " + str(end_index))
			hfile['X'][start_index:(start_index+X_chunk.shape[0])] = X_chunk
			logging.info("Writing y dataset " + str(start_index) + " -> " + str(end_index))
			hfile['y'][start_index:(start_index+y_chunk.shape[0])] = y_chunk
			logging.info("Batch " + str(k) + " complete")
			X = np.concatenate((X, X_chunk), axis=0)
			y = np.concatenate((y, y_chunk), axis=0)
	except Exception as err:
		logging.info("Error on batch " + str(start_index) + " -> " + str(end_index))
		logging.error(err)
		logging.error(traceback.format_exc())
	pbar.close()
	return X, y

def _process_chunk(chunk, k, pbar=None, debug=False):
    img_paths = chunk['path'].values
    targets = chunk['emotion'].values
    close_pbar = False
    if not pbar:
        pbar = tqdm(desc="Resize, adjust dimensions", total=len(img_paths), dynamic_ncols=True)
        close_pbar = True

    if debug:
        debug_file = open("data/processed/debug/"+str(k)+".txt", "w")

    skipped = 0
    X = np.empty((0, 224, 224, 3))
    y = np.empty((0,))
    pbar = tqdm(total=len(img_paths))
    for img_path, target in zip(img_paths, target):
        try:
            y_val = get_emotion_code(target)
            img = _process_img(img_path)
            if debug:
                debug_file.write(target + " " + str(y_val) + " " + img_path + "\n")
            X = np.concatenate((X, img), axis=0)
            y = np.concatenate((y, [y_val]), axis=0)
        except Exception as err:
            logging.error(err)
            logging.info("Skipping image " + img_path + ", " + target)
            logging.error(traceback.format_exc())
            skipped += 1
            debug_file.write("ERROR " + img_path + ", " + target + "\n")
        pbar.update(1)
    if close_pbar:
        pbar.close()
    if debug:
        debug_file.close()
    if skipped > 0:
        logging.warning("Skipped " + str(skipped) + " images")
    return X, y

def _process_img(img_path):
    img = imread(img_path)  # shape (h, w, c)
    img = resize(img, (224, 224), preserve_range=True).astype(np.float32)  # convert to shape (224, 224, 3)
    img = img/img.max()*255
    img.astype(int)
    img = np.expand_dims(img, axis=0)  # convert to shape (1, 224, 224, 3)
    return img
