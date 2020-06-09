# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Face recognition network proposed by Schroff et al. 15,
# https://arxiv.org/abs/1503.03832,
# https://github.com/davidsandberg/facenet

import tensorflow as tf 
from training.inception_resnet_v1 import inception_resnet_v1
slim = tf.contrib.slim


def Perceptual_Net(input_imgs):
    #input_imgs: [Batchsize,H,W,C], 0-255, BGR image
    #meanface: a mean face RGB image for normalization

    input_imgs = tf.cast(input_imgs,tf.float32)
    input_imgs = tf.clip_by_value(input_imgs,0,255)
    input_imgs = (input_imgs - 127.5)/128.0

    #standard face-net backbone
    batch_norm_params = {
    'decay': 0.995,
    'epsilon': 0.001,
    'updates_collections': None}


    with slim.arg_scope([slim.conv2d, slim.fully_connected],weights_initializer=slim.initializers.xavier_initializer(), 
        weights_regularizer=slim.l2_regularizer(0.0),
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):
        feature_128,_ = inception_resnet_v1(input_imgs, bottleneck_layer_size=128, is_training=False, reuse=tf.AUTO_REUSE)

    #output the last FC layer feature(before classification) as identity feature
    return feature_128