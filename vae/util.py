# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# https://github.com/daib13/TwoStageVAE
import tensorflow as tf 
from tensorflow.contrib import layers 
import math 
import numpy as np 
from tensorflow.python.training.moving_averages import assign_moving_average


def spectral_norm(input_):
    """Performs Spectral Normalization on a weight tensor."""
    if len(input_.shape) < 2:
        raise ValueError("Spectral norm can only be applied to multi-dimensional tensors")

    # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
    # to (C_out, C_in * KH * KW). But Sonnet's and Compare_gan's Conv2D kernel
    # weight shape is (KH, KW, C_in, C_out), so it should be reshaped to
    # (KH * KW * C_in, C_out), and similarly for other layers that put output
    # channels as last dimension.
    # n.b. this means that w here is equivalent to w.T in the paper.
    w = tf.reshape(input_, [-1, input_.get_shape().as_list()[-1]])

    # Persisted approximation of first left singular vector of matrix `w`.

    u_var = tf.get_variable(
        input_.name.replace(":", "") + "/u_var",
        shape=(w.shape[0], 1),
        dtype=w.dtype,
        initializer=tf.random_normal_initializer(),
        trainable=False)
    u = u_var

    # Use power iteration method to approximate spectral norm.
    # The authors suggest that "one round of power iteration was sufficient in the
    # actual experiment to achieve satisfactory performance". According to
    # observation, the spectral norm become very accurate after ~20 steps.

    power_iteration_rounds = 1
    for _ in range(power_iteration_rounds):
        # `v` approximates the first right singular vector of matrix `w`.
        v = tf.nn.l2_normalize(tf.matmul(tf.transpose(w), u), dim=None, epsilon=1e-12)
        u = tf.nn.l2_normalize(tf.matmul(w, v), dim=None, epsilon=1e-12)

    # Update persisted approximation.
    with tf.control_dependencies([tf.assign(u_var, u, name="update_u")]):
        u = tf.identity(u)

    # The authors of SN-GAN chose to stop gradient propagating through u and v.
    # In johnme@'s experiments it wasn't clear that this helps, but it doesn't
    # seem to hinder either so it's kept in order to be a faithful implementation.
    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)

    # Largest singular value of `w`.
    norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
    norm_value.shape.assert_is_fully_defined()
    norm_value.shape.assert_is_compatible_with([1, 1])

    w_normalized = w / norm_value

    # Unflatten normalized weights to match the unnormalized tensor.
    w_tensor_normalized = tf.reshape(w_normalized, input_.shape)
    return w_tensor_normalized


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name="conv2d",
           initializer=tf.truncated_normal_initializer, use_sn=False):
  with tf.variable_scope(name):
    w = tf.get_variable(
        "w", [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=initializer(stddev=stddev))
    if use_sn:
      conv = tf.nn.conv2d(input_, spectral_norm(w), strides=[1, d_h, d_w, 1], padding="SAME")
    else:
      conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding="SAME")
    biases = tf.get_variable(
        "biases", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, use_sn=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if use_sn:
            return tf.matmul(input_, spectral_norm(matrix)) + bias
        else:
            return tf.matmul(input_, matrix) + bias


def lrelu(input_, leak=0.2, name="lrelu"):
      return tf.maximum(input_, leak * input_, name=name)


def batch_norm(x, is_training, scope, eps=1e-5, decay=0.999, affine=True):
    def mean_var_with_update(moving_mean, moving_variance):
        if len(x.get_shape().as_list()) == 4:
            statistics_axis = [0, 1, 2]
        else:
            statistics_axis = [0]
        mean, variance = tf.nn.moments(x, statistics_axis, name='moments')
        with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay), assign_moving_average(moving_variance, variance, decay)]):
            return tf.identity(mean), tf.identity(variance)

    with tf.name_scope(scope):
        with tf.variable_scope(scope + '_w'):
            params_shape = x.get_shape().as_list()[-1:]
            moving_mean = tf.get_variable('mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
            moving_variance = tf.get_variable('variance', params_shape, initializer=tf.ones_initializer, trainable=False)

            mean, variance = tf.cond(is_training, lambda: mean_var_with_update(moving_mean, moving_variance), lambda: (moving_mean, moving_variance))
            if affine:
                beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
                gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
                return tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            else:
                return tf.nn.batch_normalization(x, mean, variance, None, None, eps)


def deconv2d(input_, output_shape, k_h, k_w, d_h, d_w, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable("biases", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())


def downsample(x, out_dim, kernel_size, name):
    with tf.variable_scope(name):
        input_shape = x.get_shape().as_list()
        assert(len(input_shape) == 4)
        return tf.layers.conv2d(x, out_dim, kernel_size, 2, 'same')


def upsample(x, out_dim, kernel_size, name):
    with tf.variable_scope(name):
        input_shape = x.get_shape().as_list()
        assert(len(input_shape) == 4)
        return tf.layers.conv2d_transpose(x, out_dim, kernel_size, 2, 'same')


def res_block(x, out_dim, is_training, name, depth=2, kernel_size=3):
    with tf.variable_scope(name):
        y = x
        for i in range(depth):
            y = tf.nn.relu(batch_norm(y, is_training, 'bn'+str(i)))
            y = tf.layers.conv2d(y, out_dim, kernel_size, padding='same', name='layer'+str(i))
        s = tf.layers.conv2d(x, out_dim, kernel_size, padding='same', name='shortcut')
        return y + s 


def res_fc_block(x, out_dim, name, depth=2):
    with tf.variable_scope(name):
        y = x 
        for i in range(depth):
            y = tf.layers.dense(tf.nn.relu(y), out_dim, name='layer'+str(i))
        s = tf.layers.dense(x, out_dim, name='shortcut')
        return y + s 


def scale_block(x, out_dim, is_training, name, block_per_scale=1, depth_per_block=2, kernel_size=3):
    with tf.variable_scope(name):
        y = x 
        for i in range(block_per_scale):
            y = res_block(y, out_dim, is_training, 'block'+str(i), depth_per_block, kernel_size)
        return y 


def scale_fc_block(x, out_dim, name, block_per_scale=1, depth_per_block=2):
    with tf.variable_scope(name):
        y = x 
        for i in range(block_per_scale):
            y = res_fc_block(y, out_dim, 'block'+str(i), depth_per_block)
        return y 