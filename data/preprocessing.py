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

def resize_image(image, image_dimension):
	""" Input: Image Path
		Output: Image
		Resizes image to height of 96px. Maintains aspect ratio
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
		Randomly generates 15 cropped images
	"""
	cropped_images = []
	width = len(np.array(img)[1])
	# 111 is 96 + 15; we need at least 15 random crops possible, thus the width must be greater than 111
	# in the condition when width < 111, we shoould find a way to edit the image rather than omitting it
	if width > image_dimension + num_vals:
		bounds = random.sample(range(0, width-image_dimension), num_vals)
		for i in range(num_vals):
			new_img = img.crop((bounds[i], 0, bounds[i] + image_dimension, image_dimension))
			new_img = new_img.convert("L")
			new_img.save("./crops/crop" + str(i) + ".png", format='PNG')

			new_img = np.array(new_img) / 255.0

			# for row in new_img:
			#     for num in row:
			#         if num < 0:
			#             print(num)
			cropped_images.append(new_img)
	return cropped_images

def alter_image(image_path):
	""" Function to apply all of the filters to a single image.
		Input: Image path
		Output: Altered image object
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

	# perspective transform
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
	# perhaps we can pick the top right corner pixel color and fill the bg
	# but if its synthetic, then we can just make samples of White Text on Brack background

	# shading
	affined_image = np.array(affined_image) * random.uniform(0.2, 1.5)
	affined_image = np.clip(affined_image, 0, 255).astype(np.uint8)
	final_image = Image.fromarray(affined_image)


	return final_image
	# final_image = final_image.convert("L")
	# final_image.save("test1.png", format='PNG')

def create_font_dictionary():
	path = "./150_fonts.txt"

	f = open(path, 'r')
	content = f.read().split()
	dict = {}
	count = 0
	for line in content:
		print(line)
		dict[line] = count
		count += 1
	with open('150_fonts.json', 'w') as fp:
		json.dump(dict, fp,  indent=4)

	#     #     pickle.dump(cropped_images, output)

def create_total_font_dictionary():
	path = "./fontlist.txt"

	f = open(path, 'r')
	content = f.read().split()
	dict = {}
	count = 1
	for line in content:
		print(line)
		dict[line] = count
		count += 1
	with open('font_dict.json', 'w') as fp:
		json.dump(dict, fp,  indent=4)

	#     #     pickle.dump(cropped_images, output)


# from index:name
def create_total_font_dictionary_backwards():
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

	#     #     pickle.dump(cropped_images, output)

def get_font_dict():
	with open('font_dict.json') as json_file:
		font_dict = json.load(json_file)
	return font_dict

def process_unlabeled_real(root_dir):
	""" Input: Root directory (string)
		1) Train inputs for SCAEd
	"""
	scae_inputs = []

	print("Starting processing of unlabeled real...")
	count = 0

	# files = [f.path for f in os.scandir(root_dir) if f.name.endswith(".jpg") or f.name.endswith(".png")]
	for f in os.scandir(root_dir):

		if count % 13 == 0 and (f.name.endswith(".jpeg") or f.name.endswith(".jpg") or f.name.endswith(".png")):

			image_path = f.path

			image = alter_image(image_path)
			image = resize_image(image, 96)

			if count % 13000 == 0:
				count_str = str(count)
				# image.save("./imgs/" + count_str + ".png", "PNG")
				print( "Images preprocessed: ", count)

			cropped_images = generate_crop(image, 96, 10)

			for c in cropped_images:
				scae_inputs.append(c)
		count += 1

	print("Number of images in file: ", len(scae_inputs))
	with h5py.File('scae_real_inputs_fixed.hdf5', 'w') as f:
		 f.create_dataset('scae',data=scae_inputs)


def generate_crop_samples(root_dir):
	count = 0

	for subdir in os.listdir(root_dir): # goes through all font folders

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

def create_hdf5(root_dir):
	""" Input: Root directory (string)
		Output: Creates 5 pickle files to use for our model.
		1) Train inputs for SCAE
		2) Train input & labels for DeepFont model
		3) Test input & labels for DeepFont Model
	"""
	scae_inputs = []
	train_inputs = []
	train_labels = []
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
				if file_count >= 200:
					break
				image_path = subdir_path + "/" + file
				image = alter_image(image_path)

				image = resize_image(image, 96)
				# print(np.array(image))
				cropped_images = generate_crop(image, 96, 10)

				# print(cropped_images)

				if file_count < 100:
					for c in cropped_images:
						scae_inputs.append(c)

				elif file_count < 200:
					for c in cropped_images:
						test_inputs.append(c)
						test_labels.append(font_subset[font_name])
				# else:
				#     for c in cropped_images:
				#         train_inputs.append(c)
				#         train_labels.append(font_subset[font_name])

				file_count += 1

		if total_folder_count % 100 == 0:
			print(total_folder_count, "folders done")
		total_folder_count += 1


	scae_inputs = np.array(scae_inputs)
	# train_inputs = np.array(train_inputs)
	# train_labels = np.array(train_labels)
	test_inputs = np.array(test_inputs)
	test_labels = np.array(test_labels)

	# shuffle_and_save(train_inputs, "train_inputs", train_labels, "train_labels", 5)
	shuffle_and_save(test_inputs, "test_inputs", test_labels, "test_labels", 10)
	shuffle_and_save_scae(scae_inputs, "synthetic_scae_inputs")


	# with h5py.File('scae_synthetic_inputs.hdf5', 'w') as f:
	#      f.create_dataset('scae_synthetic_inputs',data=scae_inputs)

	# with h5py.File('train_inputs.hdf5', 'w') as f:
	#     f.create_dataset('train_inputs',data=train_inputs)

	# with h5py.File('train_labels.hdf5', 'w') as f:
	#     f.create_dataset('train_labels',data=train_labels)

	# with h5py.File('test_inputs.hdf5', 'w') as f:
	#     f.create_dataset('test_inputs',data=test_inputs)

	# with h5py.File('test_labels.hdf5', 'w') as f:
	#     f.create_dataset('test_labels',data=test_labels)

	print("Finished preprocessing...")

def shuffle_and_save(inputs, inputs_file_name, labels, labels_file_name, shuffle_size):

	test_inputs = inputs
	test_labels = labels

	temp = list(range(len(test_inputs)//shuffle_size)) # list with all the indices of test_inputs divided by ten?
	random.shuffle(temp) #
	test_inputs_copy = test_inputs[:]
	test_labels_copy = test_labels[:]

	for i, j in enumerate(temp):
		if not i == j:
			test_inputs_copy[i*shuffle_size],test_inputs_copy[(i*shuffle_size)+1] = test_inputs[j*shuffle_size],test_inputs[(j*shuffle_size)+1]
			test_labels_copy[i*shuffle_size],test_labels_copy[(i*shuffle_size)+1] = test_labels[j*shuffle_size],test_labels[(j*shuffle_size)+1]

	with h5py.File(inputs_file_name + '.hdf5', 'w') as f:
		f.create_dataset(inputs_file_name,data=test_inputs_copy)

	with h5py.File(labels_file_name + '.hdf5', 'w') as f:
		f.create_dataset(labels_file_name,data=test_labels_copy)


def shuffle_and_save_scae(inputs, inputs_file_name):
	random.shuffle(inputs)

	with h5py.File(inputs_file_name + '.hdf5', 'w') as f:
		 f.create_dataset(inputs_file_name,data=inputs)


def get_data_for_scae():
	with h5py.File('synthetic_scae_inputs.hdf5', 'r') as hf:
		scae_synthetic_inputs = hf['synthetic_scae_inputs'][:]
	return scae_synthetic_inputs

def get_train():
	with h5py.File('shuffled_train_labels.hdf5', 'r') as hf:
		train_labels = hf['shuffled_train_labels'][:]

	print("shuffled train labels finished")

	with h5py.File('shuffled_train_inputs.hdf5', 'r') as hf:
		train_inputs = hf['shuffled_train_inputs'][:]

	print("shuffled train inputs finished")

	# train_inputs, train_labels = shuffle_data_for_train(train_inputs, train_labels)
	return train_inputs, train_labels



def big_shuffler():

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
			train_inputs_copy[i*shuffle_size],train_inputs_copy[(i*shuffle_size)+1] = train_inputs[j*shuffle_size],train_inputs[(j*shuffle_size)+1]
			train_labels_copy[i*shuffle_size],train_labels_copy[(i*shuffle_size)+1] = train_labels[j*shuffle_size],train_labels[(j*shuffle_size)+1]

	with h5py.File('shuffled_train_inputs.hdf5', 'w') as f:
		f.create_dataset(shuffled_train_inputs,data=train_inputs_copy)

	with h5py.File('shuffled_train_labels.hdf5', 'w') as f:
		f.create_dataset(shuffled_train_labels,data=train_labels_copy)



# def get_small_inputs():
#     with h5py.File('../data/test_inputs.hdf5', 'r') as hf:
#         test_inputs = hf['test_inputs'][:]


#     with h5py.File('../data/test_labels.hdf5', 'r') as hf:
#         test_labels = hf['test_labels'][:]


#     # train_inputs, train_labels = shuffle_data_for_train(train_inputs, train_labels)
#     return test_inputs, test_labels





def get_real_test(root_dir):

	with open('150_fonts.json') as json_file:
		font_subset = json.load(json_file)

	real_test_inputs = []
	real_test_labels = []

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

				for c in cropped_images:
					real_test_inputs.append(c)
					real_test_labels.append(font_subset[font_name])

				file_count += 1

		if total_folder_count % 100 == 0:
			print(total_folder_count, "folders done")
		total_folder_count += 1

	print("total files:", file_count)
	real_test_inputs = np.array(real_test_inputs)
	real_test_labels = np.array(real_test_labels)

	return real_test_inputs, real_test_labels


def combine_real_synth_for_scae():
	with h5py.File('synthetic_scae_inputs.hdf5', 'r') as synth, h5py.File('scae_real_inputs_fixed.hdf5', 'r') as real:
		synth_data = synth['synthetic_scae_inputs'][:]
		real_data = real['scae'][:]

	all_data = np.concatenate((synth_data, real_data), axis=0)

	random.shuffle(all_data)
	return all_data

def get_test():


	# test_inputs, test_labels = shuffle_data_for_test(test_inputs, test_labels)
	return test_inputs, test_labels

def combine_real_synthetic_test():
	real_inputs, real_labels = get_real_test("./VFR_real_test")

	print("finished processing real inputs & labels")

	with h5py.File('shuffled_test_labels.hdf5', 'r') as hf:
		synth_labels = hf['shuffled_test_labels'][:]

	with h5py.File('shuffled_test_inputs.hdf5', 'r') as hf:
		synth_inputs = hf['shuffled_test_inputs'][:]

	print("finished uploading synthetic")

	combined_inputs = np.concatenate((synth_inputs, real_inputs), axis=0)
	combined_labels = np.concatenate((synth_labels, real_labels), axis=0)

	shuffle_and_save(combined_inputs, "combined_test_inputs", combined_labels, "combined_test_inputs")

	print("finished shufflin")

def main():
	# generate_crop_samples("./real_test_sample")
	combine_real_synthetic_test()


if __name__ == "__main__":
	main()
