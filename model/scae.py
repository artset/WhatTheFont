import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
import tensorflow_hub as hub

import sys
sys.path.append('../data')

from scae_preprocessing import *
import numpy as np

from imageio import imwrite
import os
import argparse
import sys
import random

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)

## --------------------------------------------------------------------------------------

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

parser.add_argument('--num-epochs', type=int, default=5,
                    help='Number of passes through the training data to make before stopping')

parser.add_argument('--learn-rate', type=float, default=0.001,
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



class SCAE(tf.keras.Model):
    def __init__(self):
        super(SCAE, self).__init__()

        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.batch_size = 128
        self.epoch = 1
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

        # Conv2D(64, (3, 3), activation='relu', padding='same')
        self.stride_size = 2
        self.reshape = tf.keras.layers.Reshape((96, 96, 1))
        self.conv_layer1 = Conv2D(input_shape=(96, 96,1), filters=64, strides=self.stride_size, kernel_size=(3,3), activation=self.leaky_relu, padding='same', name='conv_layer1', dtype=tf.float32)
        self.conv_layer2 = Conv2D(filters=128, strides=self.stride_size, kernel_size=(3,3), activation=self.leaky_relu, padding='same', name='conv_layer2', dtype=tf.float32)
        self.deconv_layer1 = Conv2DTranspose(filters=64, strides=self.stride_size, kernel_size=(3,3), activation=self.leaky_relu, padding='same', name='deconv_layer1', dtype=tf.float32)
        self.deconv_layer2 = Conv2DTranspose(filters=1, strides=self.stride_size, kernel_size=(3,3), activation=self.leaky_relu, padding='same', name='deconv_layer2', dtype=tf.float32)


    def call(self, inputs):
        inputs = self.reshape(inputs)

        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("inputs.shape", inputs.shape)
        c1 = self.conv_layer1(inputs)
        # print("c1 shape", c1.shape)
#         c1 = tf.nn.max_pool(c1, [1, 2, 2 ,1], 2, self.padding) #? idk
        c2 = self.conv_layer2(c1)
        # print("c2 shape", c2.shape)

        d1 = self.deconv_layer1(c2)
        # print("d1 shape", d1.shape)
        # paper says unpool, we say not now
        d2 = self.deconv_layer2(d1)
        return d2

    def loss(self, original, decoded):
        mse = tf.keras.losses.MeanSquaredError()
        decoded = tf.squeeze(decoded)
        return mse(original, decoded)
        # original = self.reshape(original)
        # return tf.reduce_sum((original-decoded)**2) / original.shape[0]
        # original = self.reshape(original)
        # difference_sq = (original - decoded)**2
        # return tf.reduce_sum(difference_sq)

def train(model, real_images, fake_images):
    iterations = (len(real_images) + len(fake_images)) // model.batch_size
    fake_batch = len(fake_images) // iterations
    real_batch = len(real_images) // iterations

    total_loss = 0

    for i in range(iterations):
        # image_inputs = images[i * model.batch_size : (i+1) * model.batch_size]
        real_inputs = real_images[i * real_batch : (i+1) * real_batch]
        fake_inputs = fake_images[i * fake_batch : (i+1) * fake_batch]

        inputs = np.concatenate((real_inputs, fake_inputs), axis=0)
        random.shuffle(inputs)

        with tf.GradientTape() as tape:
            res = model(inputs)
            loss = model.loss(inputs, res)
        total_loss += loss

        if (i % 100 == 0):
            print("loss", loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    total_loss = total_loss / float(iterations)

    print("AVERAGE LOSS THIS EPOCH", total_loss)
    # TODO: This isn't really a test, it's moreso a 'show the outputs of our decoder for poster'
    # 1) Call the encoder, and decoder outputs
    # 2) Save encoder + decoder on some samples.

    pass

def test(model, real_images, synthetic_images):
    # TODO: This isn't really a test, it's moreso a 'show the outputs of our decoder for poster'
    # 1) Call the encoder, and decoder outputs
    # 2) Save encoder + decoder on some samples.
    real_inputs = real_images[:40]
    # print(real_inputs[0].shape)
    # print(np.dtype(real_inputs))
    # print(real_inputs.shape)
    # fake_inputs = fake_images[:5]

    # inputs = np.concatenate((real_inputs, fake_inputs), axis=0)
    # print(real_inputs)

    babby = 0
    for real in real_inputs:
        for row in range(len(real)):
            for col in range(len(real[0])):
                real[row][col] = int(real[row][col] * 255)
        # print(real)
        # print(np.dtype(real))
        real = np.array(real, dtype=np.uint8)
        real = Image.fromarray(real)
        real = real.convert('L')
        real.save("./scae_in/"+ str(babby) + ".png", format='PNG')
        babby += 1
    # real = [int(num *255) for row in inputs for num in row]
    # real = [Image.fromarray(np.array(img)) for img in real]
    # real = [img.convert('L') for img in real]
    # babby = 0
    # for img in real:
    #     img.save("./scae_in/"+ str(babby) + ".png", format='PNG')
    #     babby += 1

    res = model.call(real_inputs)
    res = np.array(res)
    res = np.squeeze(res)
    print(np.shape(res))
    # res = np.reshape(res, (96, 96))
    # print(res[3])
    # print(res[4])
    # print(res[4].shape)


    # for real in res:
    #     real = np.squeeze(real)
    #     print (real.shape)
        # real *= 255.0/real.max()

    # print(res)
    count = 0
    for real in res:
        # print ("------------------------------------")
        # print(real.shape)
        print("----------------------")
        print("ORIGINAL ARRAY")
        print (real)
        print("----------------------")
        real /= np.max(real)/255.0
        real = np.clip(real, 0, 255)
        print("NORMED ARRAY")
        print(real)
        print("----------------------")
        for row in range(len(real)):
            for col in range(len(real[0])):
                real[row][col] = int(real[row][col])
        print("BEFORE CONVERT TO IMG")
        print(real)
        print("----------------------")
        real = np.array(real, dtype=np.uint8)
        real = Image.fromarray(real)
        real = real.convert('L')
        real.save("./scae_out/"+ str(count) + ".png", format='PNG')
        count += 1
    # res = [int(num * 255) for row in res for num in row]
    # res = [Image.fromarray(np.array(img)) for img in res]
    # res = [img.convert('L') for img in res]
    # count = 0
    # for img in res:
    #     img.save("./scae_out/"+ str(count) + ".png", format='PNG')
    #     count += 1

def main():

    # images = get_data('scae_real_inputs.h5')
    # images = np.array(images)

    # Initialize generator and discriminator models
    scae = SCAE()

    # For saving/loading models
    checkpoint_dir = './checkpoints_ae'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(scae = scae)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    # Ensure the output directory exists
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.restore_checkpoint or args.mode == 'test':
        # restores the latest checkpoint using from the manager
        checkpoint.restore(manager.latest_checkpoint)

    try:
        # Specify an invalid GPU device
        with tf.device('/device:' + args.device):
            if args.mode == 'train':
                # real_images, fake_images = combine_real_synth_for_scae()
                real_images, fake_images = combine_real_synth_for_scae()
                for epoch in range(0, args.num_epochs):
                    print('========================== EPOCH %d  ==========================' % epoch)
                    # train(scae, real_images, fake_images)
                    train(scae, real_images, fake_images)
                    print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
                    manager.save()
                    scae.save_weights('./weights/weights_leaky_relu.h5')
            if args.mode == 'test':
                real_images, fake_images = combine_real_synth_for_scae()
                test(scae, real_images, fake_images)
    except RuntimeError as e:
        print(e)

if __name__== "__main__":
    main()
