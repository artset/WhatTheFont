import os
import numpy as np
import random
import json
import cv2
import h5py
import tensorflow as tf


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



def main():
	# create_hdf5('./syn_train')
	# create_font_dictionary()
	# create_total_font_dictionary()
	shuffle_train()


if __name__ == "__main__":
	main()