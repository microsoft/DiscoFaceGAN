# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# A tensorflow implementation of face parsing network 
# proposed by Lin et al. 19,
# https://arxiv.org/abs/1906.01342.
#--------------------------------------------------------------
import tensorflow as tf 
from scipy.io import loadmat,savemat
import os
import numpy as np
from training.parser_utils import *
from training.resnet_block import *

def fpn(c1,c2,c3,c4,data_format = 'channels_first'):
	with tf.variable_scope('c4'):
		h = tf.shape(c4)[2]
		w = tf.shape(c4)[3]
		f4 = conv2d_fixed_padding(c4,256, 1, 1, data_format,use_bias = True)
		f4 = tf.transpose(f4,perm=[0,2,3,1])
		f4 = tf.image.resize_images(f4,[2*h,2*w],align_corners = True)
		f4 = tf.transpose(f4,perm=[0,3,1,2])


	with tf.variable_scope('c3'):
		h = tf.shape(c3)[2]
		w = tf.shape(c3)[3]
		f3 = conv2d_fixed_padding(c3,256, 1, 1, data_format,use_bias = True)
		f3 += f4
		f3 = tf.transpose(f3,perm=[0,2,3,1])
		f3 = tf.image.resize_images(f3,[2*h,2*w],align_corners = True)
		f3 = tf.transpose(f3,perm=[0,3,1,2])

	with tf.variable_scope('c2'):
		h = tf.shape(c2)[2]
		w = tf.shape(c2)[3]
		f2 = conv2d_fixed_padding(c2,256, 1, 1, data_format,use_bias = True)
		f2 += f3
		f2 = tf.transpose(f2,perm=[0,2,3,1])
		f2 = tf.image.resize_images(f2,[2*h,2*w],align_corners = True)
		f2 = tf.transpose(f2,perm=[0,3,1,2])

	with tf.variable_scope('c1'):
		h = tf.shape(c1)[2]
		w = tf.shape(c1)[3]
		f1 = conv2d_fixed_padding(c1,256, 1, 1, data_format,use_bias = True)
		f1 += f2

	with tf.variable_scope('convlast'):
		x = conv2d_fixed_padding(f1,256, 3, 1, data_format,use_bias = True)


	return x

def MaskNet(x,is_training = False,data_format = 'channels_first'):
	with tf.variable_scope('neck'):
		x = conv2d_fixed_padding(x,256, 3, 1, data_format,use_bias = True)
		x = batch_norm_relu(x, is_training, data_format)
		x = conv2d_fixed_padding(x,256, 3, 1, data_format,use_bias = True)
		x = batch_norm_relu(x, is_training, data_format)

	with tf.variable_scope('convlast'):
		x = conv2d_fixed_padding(x,3, 1, 1, data_format,use_bias = True)
	x = tf.nn.softmax(x,axis = 1)
	x = tf.transpose(x,perm=[0,2,3,1])
	x = tf.image.resize_images(x,[512,512],align_corners = True)
	x = tf.transpose(x,perm=[0,3,1,2])

	return x



def FaceParser(inputs, data_format = 'channels_first',is_training = False):
	with tf.variable_scope('resnet',reuse = tf.AUTO_REUSE):
		with tf.variable_scope('block0'):
			inputs = conv2d_fixed_padding(
				inputs=inputs, filters=64, kernel_size=7,
				strides=2, data_format=data_format)

			inputs = batch_norm_relu(inputs, is_training, data_format)

			inputs = tf.layers.max_pooling2d(
				inputs=inputs, pool_size=3,
				strides=2, padding='SAME',
				data_format=data_format)

		with tf.variable_scope('block1'):
			inputs = building_block(inputs, 64, is_training, None, 1, data_format)
			c1 = inputs = building_block(inputs, 64, is_training, None, 1, data_format)

		with tf.variable_scope('block2'):

			c2 = inputs = block_layer(inputs, filters = 128, blocks = 2, strides = 2, training = is_training,
                data_format = data_format)

		with tf.variable_scope('block3'):

			c3 = inputs = block_layer(inputs, filters = 256, blocks = 2, strides = 2, training = is_training,
                data_format = data_format)

		with tf.variable_scope('block4'):

			c4 = inputs = block_layer(inputs, filters = 512, blocks = 2, strides = 2, training = is_training,
                data_format = data_format)

	with tf.variable_scope('fpn',reuse = tf.AUTO_REUSE):

		x = fpn(c1,c2,c3,c4)

	with tf.variable_scope('MaskNet',reuse = tf.AUTO_REUSE):
		x = MaskNet(x)

	return x

# Get hair segmentation from input image
def Parsing(inputs,lm):
	lm = tf.stack([lm[:,:,0],256 - lm[:,:,1]],axis = 2)
	lm5p = transfer_68to5(lm)
	lm5p = tf.stop_gradient(lm5p)

	warp_inputs,tinv = preprocess_image_seg(inputs,lm5p)
	warp_inputs = normalize_image(warp_inputs)
	warp_inputs = tf.transpose(warp_inputs,perm=[0,3,1,2])

	with tf.variable_scope('FaceParser'):
		outputs = FaceParser(warp_inputs)     

	outputs = tf.transpose(outputs,[0,2,3,1])
	ori_image = reverse_warp_and_distort(outputs,tinv)
	ori_image = tf.transpose(ori_image,perm=[0,2,1,3])   # rotate hair segmentation
	return ori_image 