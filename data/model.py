import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
import tensorflow_hub as hub

import numpy as np

from imageio import imwrite
import os
import argparse
from preprocessing import *

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
		self.batch_size = 100
		self.model = tf.keras.Sequential()
		# self.model.add(tf.keras.layers.Reshape((100, 105, 105,)))
		self.model.add(tf.keras.layers.Conv2D(filters=64, strides=(2,2), kernel_size=(3,3), padding='same', name='conv_layer1', input_shape=(105, 105,1))) #, input_shape=(args.batch_size,)
		self.model.add(tf.keras.layers.BatchNormalization())
		self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))

		self.model.add(tf.keras.layers.Conv2D(filters=128, strides=(1,1), kernel_size=(3,3), padding='same', name='conv_layer2'))
		self.model.add(tf.keras.layers.BatchNormalization())
		self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))

		self.model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same'))
		self.model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same'))
		self.model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same'))

		self.model.add(tf.keras.layers.Flatten())
		self.model.add(tf.keras.layers.Dense(4096))
		self.model.add(tf.keras.layers.Dense(4096))
		self.model.add(tf.keras.layers.Dense(2383, activation='softmax'))

		self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)

	@tf.function
	def call(self, inputs):
		"""
		Executes the generator model on the random noise vectors.

		:param inputs: a batch of random noise vectors, shape=[batch_size, z_dim]

		:return: prescaled generated images, shape=[batch_size, height, width, channel]
		"""
		# TODO: Call the forward pass
		return self.model(inputs)

	@tf.function
	def loss_function(self, probs, labels):
		"""
		Outputs the loss given the discriminator output on the generated images.

		:param disc_fake_output: the discrimator output on the generated images, shape=[batch_size,1]

		:return: loss, the cross entropy loss, scalar
		"""
		# TODO: Calculate the loss
		loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
		return loss

def train(model, train_inputs, train_labels):
	"""
	Train the model for one epoch. Save a checkpoint every 500 or so batches.

	:param generator: generator model
	:param discriminator: discriminator model
	:param dataset_ierator: iterator over dataset, see preprocess.py for more information
	:param manager: the manager that handles saving checkpoints by calling save()

	:return: The average FID score over the epoch
	"""
	# num_batches = int(len(train_inputs)/model.batch_size)
	num_batches = len(train_inputs)//model.batch_size
	for i in range(num_batches):
		with tf.GradientTape() as tape:
			temp_inputs = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
			temp_train_labels = train_labels[i*model.batch_size:(i+1)*model.batch_size]

			predictions = model.call(temp_inputs)
			loss = model.loss_function(predictions, temp_train_labels)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Test the model by generating some samples.
def test(model, test_inputs, test_labels):
	"""
	Test the model.

	:param generator: generator model

	:return: None
	"""
	loss = 0
	num_batches = 2
	for i in range(num_batches):
		temp_inputs = test_inputs[i*2:(i+1)*2]
		temp_test_labels = test_labels[i*2:(i+1)*2]

		predictions = model.call(temp_inputs)
		loss += model.loss_function(predictions, temp_test_labels)
	loss = loss/2
	return loss

## --------------------------------------------------------------------------------------

def main():
	# Initialize generator and discriminator models
	print("poop")
	test_dict, test_images, test_labels = get_test()
	dict, images, font_labels = get_train()
	images = np.array(images)

	model = DeepFont()
	model.load_weights('./weights/weights.h5', by_name=True)

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
		checkpoint.restore(manager.latest_checkpoint)

	try:
		# Specify an invalid GPU device
		with tf.device('/device:' + args.device):
			if args.mode == 'train':
				for epoch in range(0, args.num_epochs):
					print('========================== EPOCH %d  ==========================' % epoch)
					train(model, images, font_labels)
					# Save at the end of the epoch, too
					print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
					manager.save()
			if args.mode == 'test':
				test(model, test_images, test_labels)
	except RuntimeError as e:
		print(e)

if __name__ == '__main__':
   main()
