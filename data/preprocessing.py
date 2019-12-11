import os
from scipy import misc
import numpy as np
from PIL import Image, ImageFile
import random
import json
import pickle
import cv2
import h5py
import tensorflow as tf


ImageFile.LOAD_TRUNCATED_IMAGES = True


# ----------------------------- IMAGE PREPROCESSING --------------------------------------#
def resize_image(image, image_dimension):
	""" Input: Image Path
		Output: Image
		Resizes image to height of 96px, while maintaining aspect ratio
	"""
	base_height = image_dimension
	img = image
	height_percent = (base_height/float(img.size[1]))
	wsize = int((float(img.size[0])*float(height_percent)))
	# print("Width", wsize)
	img = img.resize((wsize, base_height),Image.ANTIALIAS )

	return img

def generate_crop(img, image_dimension, num_vals):
	""" Input: Image object, the width and height of our image, number of cropped images
		Output: A list of mnormalized numpy arrays normalized between 0 and 1.
		Randomly generates 15 cropped images.
	"""
	cropped_images = []
	width = len(np.array(img)[1])

	if width > image_dimension + num_vals:
		bounds = random.sample(range(0, width-image_dimension), num_vals)
		for i in range(num_vals):
			new_img = img.crop((bounds[i], 0, bounds[i] + image_dimension, image_dimension))
			new_img = np.array(new_img) / 255.0

			cropped_images.append(new_img)
	return cropped_images

def generate_crop_samples(root_dir):
	""" Input: root_dir, directory of desired images
		Output: none
		Creates some cropping samples that can be viewed.
	"""
	count = 0

	for subdir in os.listdir(root_dir): 
		subdir_path = root_dir + "/" + subdir
		font_name = subdir

		for file in os.listdir(subdir_path):
			if count == 3:
				break
			image_path = subdir_path + "/" + file
			image = alter_image(image_path)
			image = resize_image(image, 96)
			cropped_images = generate_crop(image, 96, 10)
			count+=1

	crop_count = 0
	for crop in cropped_images:
		for row in range(len(crop)):
			for col in range(len(crop[0])):
				crop[row][col] = int(crop[row][col] * 255)

		crop = np.array(crop, dtype=np.uint8)
		crop = Image.fromarray(crop)
		crop = crop.convert('L')
		crop.save("./crops/"+ str(crop_count) + ".png", format='PNG')
		crop_count += 1

def alter_image(image_path):
	""" Input: Image path
		Output: Altered image object
		Function to apply all of the filters (noise, blur, perspective rotation & translation) to a single image.
	"""

	img = Image.open(image_path)
	img = img.convert("L") #convert image to grey scale
	img = np.array(img)

	# noise
	row, col = img.shape
	gauss = np.random.normal(0, 3, (row, col))
	gauss = gauss.reshape(row, col)
	noised_image = img + gauss

	# blur
	blurred_image = cv2.GaussianBlur(noised_image, ksize = (3, 3), sigmaX = random.uniform(2.5, 3.5))

	# perspective transform and translation
	rotatation_angle = [-4, -2, 0, 2, 4]
	translate_x = [-5, -3, 0, 3, 5]
	translate_y = [-5, -3, 0, 3, 5]
	angle = random.choice(rotatation_angle)
	angle = random.choice(rotatation_angle)
	angle = random.choice(rotatation_angle)
	tx = random.choice(translate_x)
	ty = random.choice(translate_y)

	rows, cols = img.shape
	M_translate = np.float32([[1,0,tx],[0,1,ty]])
	M_rotate = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)

	affined_image = cv2.warpAffine(blurred_image, M_translate, (cols, rows))
	affined_image = cv2.warpAffine(affined_image, M_rotate, (cols, rows))

	# shading
	affined_image = np.array(affined_image) * random.uniform(0.2, 1.5)
	affined_image = np.clip(affined_image, 0, 255).astype(np.uint8)
	final_image = Image.fromarray(affined_image)

	return final_image




# ----------------------------- SPLITTING SYNTHETIC DATA --------------------------------------#
def create_hdf5(root_dir):
	""" Input: Root directory (string)
		Output: Creates hdf5 files to use for our model.

		Processes synthetic data by segmenting them into the following
		1) Synthetic train inputs for autoencoder - 10%
		2) Train input & labels for DeepFont model - 80%
		3) Test input & labels for DeepFont Model - 10%
	"""
	test_inputs = []
	test_labels = []

	with open('150_fonts.json') as json_file:
		font_subset = json.load(json_file)

	total_folder_count  = 0
	for subdir in os.listdir(root_dir): # goes through all font folders

		if subdir in font_subset:
			subdir_path = root_dir + "/" + subdir
			font_name = subdir

			# here, we have to split up our files into the three pixels
			file_count = 0
			for file in os.listdir(subdir_path): # goes through all sample images
				image_path = subdir_path + "/" + file
				image = alter_image(image_path)

				image = resize_image(image, 96)
				cropped_images = generate_crop(image, 96, 10)

				if file_count < 100:
				  for c in cropped_images:
					  scae_inputs.append(c)
				if file_count < 200 and file_count >= 100:
					for c in cropped_images:
						test_inputs.append(c)
						test_labels.append(font_subset[font_name])
				else:
					for c in cropped_images:
						train_inputs.append(c)
						train_labels.append(font_subset[font_name])

				file_count += 1

		if total_folder_count % 100 == 0:
			print(total_folder_count, "folders done")
		total_folder_count += 1


	scae_inputs = np.array(scae_inputs)
	train_inputs = np.array(train_inputs)
	train_labels = np.array(train_labels)
	test_inputs = np.array(test_inputs)
	test_labels = np.array(test_labels)

	shuffle_and_save(train_inputs, "train_inputs", train_labels, "train_labels", 5)
	shuffle_and_save(test_inputs, "test_inputs", test_labels, "test_labels", 10)
	shuffle_and_save_autoencoder(scae_inputs, "synthetic_scae_inputs")

	print("Finished preprocessing...")




# ----------------------------- AUTOENCODER SPECIFIC FUNCS --------------------------------------#
def process_unlabeled_real(root_dir):
	""" Input: Root directory (string) - Train inputs for Autoencoder

		Preprocess the unlabeled real data.
	"""
	scae_inputs = []

	print("Starting processing of unlabeled real...")
	count = 0

	for f in os.scandir(root_dir):
		if count % 13 == 0 and (f.name.endswith(".jpeg") or f.name.endswith(".jpg") or f.name.endswith(".png")):

			image_path = f.path

			image = alter_image(image_path)
			image = resize_image(image, 96)

			if count % 13000 == 0:
				count_str = str(count)
				print( "Images preprocessed: ", count)

			cropped_images = generate_crop(image, 96, 10)

			for c in cropped_images:
				scae_inputs.append(c)
		count += 1

	print("Number of images in file: ", len(scae_inputs))
	with h5py.File('scae_real_inputs_fixed.hdf5', 'w') as f:
		 f.create_dataset('scae',data=scae_inputs)


def get_data_for_autoencoder():
	""" Input: Root directory (string)
		Output: Creates hdf5 files to use for our model.

		Processes synthetic data by segmenting them into the following
		1) Synthetic train inputs for autoencoder - 10%
		2) Train input & labels for DeepFont model - 80%
		3) Test input & labels for DeepFont Model - 10%
	"""
	with h5py.File('synthetic_scae_inputs.hdf5', 'r') as hf:
		scae_synthetic_inputs = hf['synthetic_scae_inputs'][:]
	with h5py.File('scae_real_inputs_fixed.hdf5', 'r') as hf:
		scae_real_inputs = hf['scae_real_inputs'][:]
	return scae_real_inputs, scae_synthetic_inputs




# ----------------------------- DEEPFONT SPECIFIC FUNCS --------------------------------------#
def get_train_df():
	""" Input: None
		Output: None

		Opens the train inputs and train labels and returns them.
	"""
	with h5py.File('shuffled_train_labels.hdf5', 'r') as hf:
		train_labels = hf['shuffled_train_labels'][:]

	print("shuffled train labels finished")

	with h5py.File('shuffled_train_inputs.hdf5', 'r') as hf:
		train_inputs = hf['shuffled_train_inputs'][:]

	print("shuffled train inputs finished")
	return train_inputs, train_labels

def get_test_df():
	""" Input: None
		Output: None

		Opens the test inputs and test labels and returns them.
	"""
	with h5py.File('combined_test_labels.hdf5', 'r') as hf:
		test_labels = hf['combined_test_labels'][:]

	with h5py.File('combined_test_inputs.hdf5', 'r') as hf:
		test_inputs = hf['combined_test_inputs'][:]

	return test_inputs, test_labels




# ----------------------------- SHUFFLING FUNCTIONS --------------------------------------#
def train_shuffle():
	""" Input: None
		Output: None

		Shuffles the training set in groups of 10.
	"""
	print("Shuffling...")

	shuffle_size = 10

	with h5py.File('train_inputs.hdf5', 'r') as hf:
		train_inputs = hf['train_inputs'][:]

	print("train labels finished")

	with h5py.File('train_labels.hdf5', 'r') as hf:
		train_labels = hf['train_labels'][:]

	temp = list(range(len(train_inputs)//shuffle_size)) # list with all the indices of test_inputs divided by ten?
	random.shuffle(temp) #
	train_inputs_copy = train_inputs[:]
	train_labels_copy = train_labels[:]

	for i, j in enumerate(temp):
		if not i == j:
			train_inputs_copy[i*shuffle_size:(i+1)*shuffle_size] = train_inputs[j*shuffle_size:(j+1)*shuffle_size]
			train_labels_copy[i*shuffle_size:(i+1)*shuffle_size] = train_labels[j*shuffle_size:(j+1)*shuffle_size]

	train_inputs_copy = np.array(train_inputs_copy)
	train_labels_copy = np.array(train_labels_copy)

	with h5py.File('shuffled_train_inputs.hdf5', 'w') as f:
		f.create_dataset("shuffled_train_inputs",data=train_inputs_copy)

	with h5py.File('shuffled_train_labels.hdf5', 'w') as f:
		f.create_dataset("shuffled_train_labels",data=train_labels_copy)

	print("done shuffling!")

def shuffle_and_save(inputs, inputs_file_name, labels, labels_file_name, shuffle_size):
	""" Input: inputs, desired inputs file name, labels, desired labels file name, group number to shuffle by
		Output: None

		Shuffles the inputs and labels by a shuffle_size and then dumps them into hdf5 files.
	"""
	test_inputs = inputs
	test_labels = labels

	temp = list(range(len(test_inputs)//shuffle_size)) # list with all the indices of test_inputs divided by ten?
	random.shuffle(temp) #
	test_inputs_copy = test_inputs[:]
	test_labels_copy = test_labels[:]

	for i, j in enumerate(temp):
		if not i == j:
			test_inputs_copy[i*shuffle_size:(i+1)*shuffle_size] = test_inputs[j*shuffle_size:(j+1)*shuffle_size]
			test_labels_copy[i*shuffle_size:(i+1)*shuffle_size] = test_labels[j*shuffle_size:(j+1)*shuffle_size]

	test_inputs_copy = np.array(test_inputs_copy)
	test_labels_copy = np.array(test_labels_copy)

	with h5py.File(inputs_file_name + '.hdf5', 'w') as f:
		f.create_dataset(inputs_file_name,data=test_inputs_copy)

	with h5py.File(labels_file_name + '.hdf5', 'w') as f:
		f.create_dataset(labels_file_name,data=test_labels_copy)

def shuffle_and_save_autoencoder(inputs, inputs_file_name):
	""" Input: synthetic inputs, filename to save them under
		Output: none

		Shuffles and saves the synthetic inputs for the autoencoder.
	"""
	random.shuffle(inputs)

	with h5py.File(inputs_file_name + '.hdf5', 'w') as f:
		 f.create_dataset(inputs_file_name,data=inputs)




# ----------------------------- TESTING DATA FOR DEEPFONT --------------------------------------#
def get_real_test(root_dir):
	""" Input: directory of real test data
		Output: None

		Preprocesses the real test data and returns real test inputs, real test labels,
	"""
	with open('150_fonts.json') as json_file:
		font_subset = json.load(json_file)

	real_test_inputs = []
	real_test_labels = []

	total_folder_count  = 0
	for subdir in os.listdir(root_dir):
		if subdir in font_subset:
			subdir_path = root_dir + "/" + subdir
			font_name = subdir

			file_count = 0
			for file in os.listdir(subdir_path): # goes through all sample images
				image_path = subdir_path + "/" + file
				image = alter_image(image_path)
				image = resize_image(image, 96)
				cropped_images = generate_crop(image, 96, 10)

				for c in cropped_images:
					real_test_inputs.append(c)
					real_test_labels.append(font_subset[font_name])

				file_count += 1

		if total_folder_count % 100 == 0:
			print(total_folder_count, "folders done")
		total_folder_count += 1

	return real_test_inputs, real_test_labels

def combine_real_synthetic_test():
	""" Input: None
		Output: None

		Combines the real testing data with the synthetic testing data, shuffles, then saves them as hdf5 files.
	"""
	real_inputs, real_labels = get_real_test("./VFR_real_test")
	print("finished processing real inputs & labels")

	with h5py.File('test_labels.hdf5', 'r') as hf:
		synth_labels = hf['test_labels'][:]

	with h5py.File('test_inputs.hdf5', 'r') as hf:
		synth_inputs = hf['test_inputs'][:]

	combined_inputs = np.concatenate((synth_inputs, real_inputs), axis=0)
	combined_labels = np.concatenate((synth_labels, real_labels), axis=0)
	shuffle_and_save(combined_inputs, "combined_test_inputs", combined_labels, "combined_test_labels", 10)
	print("finished shuffling")

def check_labels_and_inputs():
	""" Input: None
		Output: None

		Function to check labels and inputs.
	"""
	with h5py.File('combined_test_labels.hdf5', 'r') as hf:
		combined_labels = hf['combined_test_labels'][:]

	with h5py.File('combined_test_inputs.hdf5', 'r') as hf:
		combined_inputs = hf['combined_test_inputs'][:]

	print("CHECKING... COMBINED LABELS", combined_labels[0:100])
	print("CHECKING... COMBINED INPUTS", combined_inputs[0:100])




# ----------------------------- DICTIONARY FUNCTIONS --------------------------------------#
def create_font_dictionary():
	""" Input: none
		Output: none
		Creates a font dictionary based on 150 selected fonts.
		dict key is fontname, dict val is index
	"""
	path = "./150_fonts.txt"

	f = open(path, 'r')
	content = f.read().split()
	dict = {}
	count = 0
	for line in content:
		dict[line] = count
		count += 1
	with open('150_fonts.json', 'w') as fp:
		json.dump(dict, fp,  indent=4)


def reversed_dict():
	""" Input: none
		Output: none

		Function to create a dictionary with only 150 fonts.
		Dict key is index, dict value is fontname.
	"""
	path = "./150_fonts.txt"

	f = open(path, 'r')
	content = f.read().split()
	dict = {}
	count = 0
	for line in content:
		print(line)
		dict[str(count)] = line
		count += 1
	with open('150_fonts_backwards.json', 'w') as fp:
		json.dump(dict, fp,  indent=4)


def create_total_font_dictionary():
	""" Input: none
		Output: none
		Creates a font dictionary based on the entire font library used by the authors of DeepFont.
		dict key is fontname, dict val is index
	"""
	path = "./fontlist.txt"

	f = open(path, 'r')
	content = f.read().split()
	dict = {}
	count = 1
	for line in content:
		dict[line] = count
		count += 1
	with open('font_dict.json', 'w') as fp:
		json.dump(dict, fp,  indent=4)

def create_total_font_dictionary_backwards():
	""" Input: none
		Output: none
		Creates a font dictionary based on the entire font library used by the authors of DeepFont.
		dict key is index, dict val is fontname
	"""
	path = "./fontlist.txt"

	f = open(path, 'r')
	content = f.read().split()
	dict = {}
	count = 1
	for line in content:
		dict[count] = line
		count += 1
	with open('backwards_font_dict.json', 'w') as fp:
		json.dump(dict, fp,  indent=4)


def get_font_dict():
	""" Input: none
		Output: font dict
		Opens font dict and returns it.
	"""
	with open('font_dict.json') as json_file:
		font_dict = json.load(json_file)
	return font_dict

# ----------------------------- MAIN ----------------------------------#
def main():
	print ("We used main to run our preprocess functions. :]")


if __name__ == "__main__":
	main()
