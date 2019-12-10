import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
import tensorflow_hub as hub
from collections import Counter
import numpy as np

import sys
sys.path.append('../data')

from imageio import imwrite
import os
import argparse
from preprocessing import *

# this time, katherine is here T_TTTT 


# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)




parser = argparse.ArgumentParser(description='DCGAN')

parser.add_argument('--img-dir', type=str, default='./data/celebA',
					help='Data where training images live')

parser.add_argument('--out-dir', type=str, default='./output',
					help='Data where sampled output images will be written')

parser.add_argument('--mode', type=str, default='train',
					help='Can be "train" or "test"')

parser.add_argument('--restore-checkpoint', action='store_true',
					help='Use this flag if you want to resuming training from a previously-saved checkpoint')

parser.add_argument('--z-dim', type=int, default=100,
					help='Dimensionality of the latent space')

parser.add_argument('--batch-size', type=int, default=128,
					help='Sizes of image batches fed through the network')

parser.add_argument('--num-data-threads', type=int, default=2,
					help='Number of threads to use when loading & pre-processing training images')

parser.add_argument('--num-epochs', type=int, default=10,
					help='Number of passes through the training data to make before stopping')

parser.add_argument('--learn-rate', type=float, default=0.0002,
					help='Learning rate for Adam optimizer')

parser.add_argument('--beta1', type=float, default=0.5,
					help='"beta1" parameter for Adam optimizer')

parser.add_argument('--num-gen-updates', type=int, default=2,
					help='Number of generator updates per discriminator update')

parser.add_argument('--log-every', type=int, default=7,
					help='Print losses after every [this many] training iterations')

parser.add_argument('--save-every', type=int, default=500,
					help='Save the state of the network after every [this many] training iterations')

parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
					help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')

args = parser.parse_args()



class DeepFont(tf.keras.Model): #is this how to convert to sequential?
	def __init__(self):
		"""
		The model for the generator network is defined here.
		"""
		super(DeepFont, self).__init__()
		# TODO: Define the model, loss, and optimizer
		self.batch_size = 128
		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.layers.Reshape((96, 96, 1)))
		self.model.add(tf.keras.layers.Conv2D(trainable=False, filters=64, strides=(2,2), kernel_size=(3,3), padding='same', name='conv_layer1', input_shape=(96, 96,1))) #, input_shape=(args.batch_size,)
		self.model.add(tf.keras.layers.BatchNormalization())
		self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))

		self.model.add(tf.keras.layers.Conv2D(trainable=False, filters=128, strides=(1,1), kernel_size=(3,3), padding='same', name='conv_layer2'))
		self.model.add(tf.keras.layers.BatchNormalization())
		self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))

		self.model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same'))
		self.model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same'))
		self.model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same'))

		self.model.add(tf.keras.layers.Flatten())
		self.model.add(tf.keras.layers.Dense(512, activation='relu'))
		self.model.add(tf.keras.layers.Dense(512, activation='relu'))
		self.model.add(tf.keras.layers.Dense(150))

		self.reshape_test = tf.keras.layers.Reshape((self.batch_size, 10, 150))

		self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)

	def call(self, inputs):
		print(inputs)
		return self.model(inputs)

	def loss_function(self, probs, labels):
		loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits = True)
		print("probs:")
		print(probs)
		print("labels:")
		print(labels)
		return loss

	def total_accuracy(self, logits, labels):
		"""  given a batch of images [(batch_size * cropped_img) x num_classes]
			 labels = 
			 and the outputted logits, computes accuracy
		"""
		print("----------total accuracy ----------")

		acc = 0 

		print("input to reshape", logits)
		sums = self.reshape_test(logits) # batch_size x cropped_img x num_classes

		sums = np.sum(predictions, axis = 1) # sums the columns of the logits

		probabilities = tf.nn.softmax(sums) # batchsize x num_classes

		top_five = np.argsort(probabilities, axis = 1)[:][-5:]

		for i in range in (len(labels)):
			if labels[i] in top_five[i]:
				acc += 1

		return acc / float(len(labels))



def train(model, train_inputs, train_labels):
	# num_batches = int(len(train_inputs)/model.batch_size)
	num_batches = len(train_inputs)//model.batch_size
	for i in range(num_batches):
		with tf.GradientTape() as tape:
			temp_inputs = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
			temp_train_labels = train_labels[i*model.batch_size:(i+1)*model.batch_size]

			predictions = model.call(temp_inputs)
			loss = model.loss_function(predictions, temp_train_labels)
			if i % 100 == 0:
				print("---Batch", i, " Loss: ", loss)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):

	# 4 batches with one image in each batch_inputs
	num_batches = len(test_inputs) // (model.batch_size * 10)
	cropped_images = 10

	acc = 0


	for i in range(num_batches): # hardcode 15 because each i is an image
		# print("-------------batch", i, "-------------")
		batch_inputs = test_inputs[i * model.batch_size: (i+1) * model.batch_size ]
		batch_labels = test_labels[i * model.batch_size : (i+1) * model.batch_size]

		predictions = model.call(batch_inputs) # prediction for a single image
		acc += model.total_accuracy(predictions, batch_labels)
		# print("summed accuracy", acc)
	return acc / float(num_batches)


## --------------------------------------------------------------------------------------

def main():

	model = DeepFont()
	model.load_weights('weights_leaky_relu.h5', by_name=True)

	# For saving/loading models
	checkpoint_dir = './checkpoints_df'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(model = model)
	manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
	# Ensure the output directory exists
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	if args.restore_checkpoint or args.mode == 'test':
		# restores the lates checkpoint using from the manager
		print("Running test mode...")
		checkpoint.restore(manager.latest_checkpoint)

	try:
		# Specify an invalid GPU device
		with tf.device('/device:' + args.device):
			# train_inputs, train_labels, test_inputs, test_labels = get_data()
			if args.mode == 'train':
				train_inputs, train_labels = get_train()

				for epoch in range(0, args.num_epochs):
					print('========================== EPOCH %d  ==========================' % epoch)
					train(model, train_inputs, train_labels)
					# Save at the end of the epoch, too
					print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
					manager.save()
			if args.mode == 'test':
				test_inputs, test_labels = get_test()
				test_labels = relabel_labels(test_labels)
				print("--test accuracy--", test(model, test_inputs, test_labels))
	except RuntimeError as e:
		print(e)

if __name__ == '__main__':
   main()