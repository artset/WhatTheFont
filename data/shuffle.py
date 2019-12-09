import os
import numpy as np
import random
import json
import cv2
import h5py
import tensorflow as tf

# this shuffle_train will break... too big.
def shuffle_train():
	
	with h5py.File('train_inputs.hdf5', 'r') as hf:
		train_inputs = hf['train_inputs'][:]

	with h5py.File('train_labels.hdf5', 'r') as hf:
		train_labels = hf['train_labels'][:]

	indices = tf.range(len(train_inputs))
	tf.random.shuffle(indices)
	tf.gather(train_inputs, indices)
	tf.gather(train_labels, indices)

	with h5py.File('shuffled_train_inputs.hdf5', 'w') as f:
		 f.create_dataset('shuffled_train_inputs',data=train_inputs)

	with h5py.File('shuffled_train_labels.hdf5', 'w') as f:
		f.create_dataset('shuffled_train_labels',data=train_labels)

def shuffle_train_split():
	# this doesn't work and it's UTTERLY SAD

	with h5py.File('train_inputs.hdf5', 'r') as hf:
		train_inputs = hf['train_inputs'][:]

	with h5py.File('train_labels.hdf5', 'r') as hf:
		train_labels = hf['train_labels'][:]

	batch_count = 100
	batch_size = len(train_inputs) // batch_count
	print("batch size", batch_size)

	final_input = []
	final_label = []

	for i in range(batch_count):
		batch_input = train_inputs[i * batch_size : (i+1) * batch_size]
		batch_label = train_labels[i * batch_size: (i+1) * batch_size]

		indices = tf.range(len(batch_input))
		tf.random.shuffle(indices)
		tf.gather(batch_input, indices)
		tf.gather(batch_label, indices)

		# saved_input = 'shuffled_train_inputs_' + str(i+1) + '.hdf5'
		# saved_label = 'shuffled_train_labels_' + str(i+1) + '.hdf5'

		# with h5py.File(saved_input, 'w') as f:
		# 	f.create_dataset('shuffled_train_inputs_' + str(i+1),data=train_inputs)

		# with h5py.File(saved_label, 'w') as f:
		# 	f.create_dataset('shuffled_train_labels_' + str(i+1),data=train_labels)

		final_input = np.vstack((final_input, batch_input))
		final_label += np.vstack((final_label, batch_label))


		print('***', i,"batches***")
	
	with h5py.File('shuffled_train_inputs.hdf5', 'w') as f:
		 f.create_dataset('shuffled_train_inputs',data=final_input)

	with h5py.File('shuffled_train_labels.hdf5', 'w') as f:
		f.create_dataset('shuffled_train_labels',data=final_label)
	
	print('*** saved train inputs & labels ***')


    
def shuffle_data_for_test():

	print("start...")

	with h5py.File('test_inputs.hdf5', 'r') as hf:
		test_inputs = hf['test_inputs'][:]

	with h5py.File('test_labels.hdf5', 'r') as hf:
		test_labels = hf['test_labels'][:]
	
	print("opened files..")
	temp = list(range(len(test_inputs)//10)) # list with all the indices of test_inputs divided by ten?
	random.shuffle(temp) #
	test_inputs_copy = test_inputs[:]
	test_labels_copy = test_labels[:]

	for i, j in enumerate(temp):
		if not i == j:
			test_inputs_copy[i*10],test_inputs_copy[(i*10)+1] = test_inputs[j*10],test_inputs[(j*10)+1]
			test_labels_copy[i*10],test_labels_copy[(i*10)+1] = test_labels[j*10],test_labels[(j*10)+1]

	print("creating files...")
	with h5py.File('shuffled_test_inputs.hdf5', 'w') as f:
		f.create_dataset('shuffled_test_inputs',data=test_inputs_copy)

	with h5py.File('shuffled_test_labels.hdf5', 'w') as f:
		f.create_dataset('shuffled_test_labels',data=test_labels_copy)
	print("done.")


def shuffle_data_for_train():

	with h5py.File('train_inputs.hdf5', 'r') as hf:
		test_inputs = hf['train_inputs'][:]

	with h5py.File('train_labels.hdf5', 'r') as hf:
		test_labels = hf['train_labels'][:]
	
	shuffle_size = 5
	temp = list(range(len(test_inputs)//shuffle_size)) # list with all the indices of test_inputs divided by ten?
	random.shuffle(temp) #
	test_inputs_copy = test_inputs[:]
	test_labels_copy = test_labels[:]

	for i, j in enumerate(temp):
		if not i == j:
			test_inputs_copy[i*shuffle_size],test_inputs_copy[(i*shuffle_size)+1] = test_inputs[j*shuffle_size],test_inputs[(j*shuffle_size)+1]
			test_labels_copy[i*shuffle_size],test_labels_copy[(i*shuffle_size)+1] = test_labels[j*shuffle_size],test_labels[(j*shuffle_size)+1]

	with h5py.File('shuffled_train_inputs.hdf5', 'w') as f:
		f.create_dataset('shuffled_train_inputs',data=test_inputs_copy)

	with h5py.File('shuffled_train_labels.hdf5', 'w') as f:
		f.create_dataset('shuffled_train_labels',data=test_labels_copy)


def main():
	shuffle_data_for_train()

if __name__ == "__main__":
	main()