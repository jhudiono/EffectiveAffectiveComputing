from glob import glob
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import csv
import h5py
import numpy as np
import pandas as pd
import logging
import traceback

# img_data[[10, 11, 15, 'path']].to_csv("data/imfdb_total.csv", header=["gender", "emotion", "position", "path"])

def get_annotation_files():
    def get_shape(p):
        try:
            return imread(p).shape
        except Exception as err:
            return (err, None, None)
    
    movie_dirs = glob("data/IMFDB_final/*/*")
    
    txt_missing = []
    total_data = []
    pbar = tqdm(total=len(movie_dirs), dynamic_ncols=True)
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
    logging.error("Could not find .txt file:" + str(len(txt_missing)) + str(txt_missing))
    return pd.concat(total_data)#[['height', 'width', 'channels', 7, 9, 10, 11, 13, 15, 'path']]#[['emotion', 'gender', 'pose', 'path']]

def extract_data(filepath="./data/imfdb_total.csv", batch_size=1000, test_set=None, compression='gzip', save_filepath="data/processed/data.hdf5", debug=False):
	logging.basicConfig()
	logging.getLogger().setLevel(logging.INFO)

	observations = pd.read_csv(filepath, index_col=0)
	observations = observations[~observations['path'].isna()]

	# Get test subset
	if test_set:
		df_unhappy = observations[observations['emotion'] != 'HAPPINESS'].sample(test_set)
		df_happy = observations[observations['emotion'] == 'HAPPINESS'].sample(test_set)
		observations = pd.concat([df_unhappy, df_happy])

	# Initialize h5py file and create datasets
	hfile = h5py.File(save_filepath, "w", libver="latest")
	hfile.create_dataset("X", maxshape=(None, 224, 224, 3), compression=compression,
		shape=(len(observations), 224, 224, 3))
	hfile.create_dataset("y", maxshape=(None,), compression=compression,
		shape=(len(observations),))

	# Write data to h5py file
	X, y = _df_to_hdf5(hfile, observations, batch_size, 0, debug=debug)
	hfile.close()

	return X, y

def append_data(filepath="./data/imfdb_total.csv", start_index=0, end_index=None, batch_size=1000, save_filepath="data/processed/data.hdf5", debug=False):
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

def _df_to_hdf5(hfile, df, batch_size, start_index, debug=False):
	X = np.empty((0, 224, 224, 3))
	y = np.empty((0,))
	pbar = tqdm(desc="Total rows processed", total=len(df), dynamic_ncols=True)
	start_index = 0
	for k, chunk in df.groupby(np.arange(len(df))//batch_size):
		logging.info("Preprocessing batch " + str(k))
		end_index = start_index + len(chunk)
		try:
			X_chunk, y_chunk = _process_chunk(chunk, k, pbar, debug=debug)
			logging.debug("X shape " + str(X_chunk.shape) + ", y shape " + str(y_chunk.shape))
			if X_chunk.shape[0] != y_chunk.shape[0]:
				logging.error("Number of features and targets don't match, skipping batch " + str(k))
				continue
			end_index = start_index + len(X_chunk)
			#y_chunk = np.where(chunk.emotion == 'HAPPINESS', 1, 0)
			logging.info("Writing X dataset " + str(start_index) + " -> " + str(end_index))
			hfile['X'][start_index:end_index] = X_chunk
			logging.info("Writing y dataset " + str(start_index) + " -> " + str(end_index))
			hfile['y'][start_index:end_index] = y_chunk
			logging.info("Batch " + str(k) + " complete")
			start_index = end_index
			X = np.concatenate((X, X_chunk), axis=0)
			y = np.concatenate((y, y_chunk), axis=0)
		except Exception as err:
			logging.info("Error on batch " + str(start_index) + " -> " + str(end_index))
			logging.error(err)
			logging.error(traceback.format_exc())
		#pbar.update(batch_size)
	pbar.close()
	return X, y

# Returns X in shape (<rows>, 224, 224, 3), y in shape (<rows>,)
def _process_chunk(chunk, k, pbar=None, debug=False):
    img_paths = chunk['path'].values
    targets = chunk['emotion'].values

    if debug:
        debug_list = []

    skipped = 0
    X = np.empty((0, 224, 224, 3))
    y = np.empty((0,))
    close_pbar = False
    if not pbar:
        pbar = tqdm(desc="Resize, adjust dimensions", total=len(img_paths), dynamic_ncols=True)
        close_pbar = True
    for img_path, target in zip(img_paths, targets):
        try:
            y_val = 0
            if target == 'HAPPINESS':
                y_val = 1
            img = imread(img_path)  # shape (h, w, c)
            img = resize(img, (224, 224), preserve_range=True).astype(np.float32)  # convert to shape (224, 224, 3)
            img = img/img.max()*255
            img.astype(int)
            #img = np.swapaxes(img, 0, 1)
            img = np.expand_dims(img, axis=0)  # convert to shape (1, 224, 224, 3)
            if debug:
                debug_list.append([str(target), str(y_val), str(img_path)])
            X = np.concatenate((X, img), axis=0)
            y = np.concatenate((y, [y_val]), axis=0)
        except Exception as err:
            logging.error(err)
            logging.info("Skipping image " + str(img_path) + ", " + target)
            logging.error(traceback.format_exc())
            skipped += 1
            if debug:
                debug_list.append([str(target), "ERROR", str(img_path)])
        pbar.update(1)
    if close_pbar:
        pbar.close()
    if debug and len(debug_list) > 0:
        with open("data/processed/debug/"+str(k)+".txt", "w") as output:
            writer = csv.write(output, lineterminator='\n')
            writer.writerows(debug_list)
    if skipped > 0:
        logging.warning("Skipped " + str(skipped) + " images")
    return X, y


