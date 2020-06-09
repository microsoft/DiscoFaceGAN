# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# https://github.com/daib13/TwoStageVAE
import tensorflow as tf 
import math 
import numpy as np 
from tensorflow.python.training.moving_averages import assign_moving_average

class TwoStageVaeModel(object):
    def __init__(self, x, latent_dim=128,ch_dim = 512,ch_depth = 3, cross_entropy_loss=False):
        self.x = x
        self.batch_size = x.get_shape().as_list()[0]
        self.latent_dim = latent_dim
        self.ch_dim = ch_dim 
        self.ch_depth = ch_depth
        self.cross_entropy_loss = cross_entropy_loss

        self.is_training = tf.placeholder(tf.bool, [], 'is_training')

        self.__build_network()
        self.__build_loss()
        self.__build_summary()
        self.__build_optimizer()

    def __build_network(self):
        with tf.variable_scope('stage1'):
            self.build_encoder1()
            self.build_decoder1()

    def __build_loss(self):
        HALF_LOG_TWO_PI = 0.91893

        self.kl_loss1 = tf.reduce_sum(tf.square(self.mu_z) + tf.square(self.sd_z) - 2 * self.logsd_z - 1) / 2.0 / float(self.batch_size)
        if not self.cross_entropy_loss:
            self.gen_loss1 = tf.reduce_sum(tf.square((self.x - self.x_hat) / self.gamma_x) / 2.0 + self.loggamma_x + HALF_LOG_TWO_PI) / float(self.batch_size)
        else:
            self.gen_loss1 = -tf.reduce_sum(self.x * tf.log(tf.maximum(self.x_hat, 1e-8)) + (1-self.x) * tf.log(tf.maximum(1-self.x_hat, 1e-8))) / float(self.batch_size)
        self.loss1 = self.kl_loss1 + self.gen_loss1 


    def __build_summary(self):
        with tf.name_scope('stage1_summary'):
            self.summary1 = []
            self.summary1.append(tf.summary.scalar('kl_loss', self.kl_loss1))
            self.summary1.append(tf.summary.scalar('gen_loss', self.gen_loss1))
            self.summary1.append(tf.summary.scalar('loss', self.loss1))
            self.summary1.append(tf.summary.scalar('gamma', self.gamma_x))
            self.summary1 = tf.summary.merge(self.summary1)

    def __build_optimizer(self):
        all_variables = tf.global_variables()
        variables1 = [var for var in all_variables if 'stage1' in var.name]
        self.lr = tf.placeholder(tf.float32, [], 'lr')
        self.global_step = tf.get_variable('global_step', [], tf.int32, tf.zeros_initializer(), trainable=False)
        self.opt1 = tf.train.AdamOptimizer(self.lr).minimize(self.loss1, self.global_step, var_list=variables1)

    def step(self, data_worker, lr, sess, writer=None, write_iteration=600):
        input_batch = data_worker.fetch_train_batch()
        loss, summary, _ = sess.run([self.loss1, self.summary1, self.opt1], feed_dict={self.x: input_batch, self.lr: lr, self.is_training: True})

        global_step = self.global_step.eval(sess)
        if global_step % write_iteration == 0 and writer is not None:
            writer.add_summary(summary, global_step)
        return loss 

    def generate(self, sess, num_sample):
        num_iter = math.ceil(float(num_sample) / float(self.batch_size))
        gen_samples = []
        for i in range(num_iter):
            z = np.random.normal(0, 1, [self.batch_size, self.latent_dim])
            # x = f_1(z)
            x = sess.run(self.x_hat, feed_dict={self.z: z, self.is_training: False})
            gen_samples.append(x)
        gen_samples = np.concatenate(gen_samples, 0)
        return gen_samples[0:num_sample]


class MLP(TwoStageVaeModel):
    def __init__(self, x, latent_dim=128,ch_dim = 512, ch_depth = 3, cross_entropy_loss=False):
        super(MLP, self).__init__(x, latent_dim, ch_dim, ch_depth, cross_entropy_loss)

    def build_encoder1(self):
        with tf.variable_scope('encoder'):
            y = self.x 
            for i in range(self.ch_depth):
                y = tf.layers.dense(y, self.ch_dim, tf.nn.relu, name='fc'+str(i))

            self.mu_z = tf.layers.dense(y, self.latent_dim)
            self.logsd_z = tf.layers.dense(y, self.latent_dim)
            self.sd_z = tf.exp(self.logsd_z)
            self.z = self.mu_z + tf.random_normal([self.batch_size, self.latent_dim]) * self.sd_z 

    def build_decoder1(self):
        with tf.variable_scope('decoder'):
            y = self.z 
            self.final_side_length = self.x.get_shape().as_list()[1]
            for i in range(self.ch_depth):
                y = tf.layers.dense(y, self.ch_dim, tf.nn.relu, name='fc'+str(i))

            self.x_hat = tf.layers.dense(y, self.final_side_length, name='x_hat')
            self.loggamma_x = tf.get_variable('loggamma_x', [], tf.float32, tf.zeros_initializer())
            self.gamma_x = tf.exp(self.loggamma_x)