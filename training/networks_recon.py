# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf 
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim

# 3D face reconstruction network using resnet_v1_50 by Deng et al. 19,
# https://github.com/microsoft/Deep3DFaceReconstruction
#-----------------------------------------------------------------------------------------------

def R_Net(inputs,is_training=True,reuse = None):
	#input: [Batchsize,H,W,C], 0-255, BGR image
	inputs = tf.cast(inputs,tf.float32)
	# standard ResNet50 backbone (without the last classfication FC layer)
	with slim.arg_scope(resnet_v1.resnet_arg_scope()):
		net,end_points = resnet_v1.resnet_v1_50(inputs,is_training = is_training ,reuse = reuse)

	# Modified FC layer with 257 channels for reconstruction coefficients
	net_id = slim.conv2d(net, 80, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-id')
	net_ex = slim.conv2d(net, 64, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-ex')
	net_tex = slim.conv2d(net, 80, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-tex')
	net_angles = slim.conv2d(net, 3, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-angles')
	net_gamma = slim.conv2d(net, 27, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-gamma')
	net_t_xy = slim.conv2d(net, 2, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-XY')
	net_t_z = slim.conv2d(net, 1, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-Z')


	net_id = tf.squeeze(net_id, [1,2], name='fc-id/squeezed')
	net_ex = tf.squeeze(net_ex, [1,2], name='fc-ex/squeezed')
	net_tex = tf.squeeze(net_tex, [1,2],name='fc-tex/squeezed')
	net_angles = tf.squeeze(net_angles,[1,2], name='fc-angles/squeezed')
	net_gamma = tf.squeeze(net_gamma,[1,2], name='fc-gamma/squeezed')
	net_t_xy = tf.squeeze(net_t_xy,[1,2], name='fc-XY/squeezed')
	net_t_z = tf.squeeze(net_t_z,[1,2], name='fc-Z/squeezed')

	net_ = tf.concat([net_id,net_tex,net_ex,net_angles,net_gamma,net_t_xy,net_t_z], axis = 1)

	return net_