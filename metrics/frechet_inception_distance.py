# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Frechet Inception Distance (FID)."""

import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc
from training.networks_stylegan import CoeffDecoder
from training.training_loop import z_to_lambda_mapping

#----------------------------------------------------------------------------
# Modified by Deng et al.
def restore_weights_and_initialize():
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()

    # add batch normalization params into trainable variables 
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list +=bn_moving_vars

    var_id_list = [v for v in var_list if 'id' in v.name and 'stage1' in v.name]
    var_exp_list = [v for v in var_list if 'exp' in v.name and 'stage1' in v.name]
    var_gamma_list = [v for v in var_list if 'gamma' in v.name and 'stage1' in v.name]
    var_rot_list = [v for v in var_list if 'rot' in v.name and 'stage1' in v.name]

    saver_id = tf.train.Saver(var_list = var_id_list,max_to_keep = 100)
    saver_exp = tf.train.Saver(var_list = var_exp_list,max_to_keep = 100)
    saver_gamma = tf.train.Saver(var_list = var_gamma_list,max_to_keep = 100)
    saver_rot = tf.train.Saver(var_list = var_rot_list,max_to_keep = 100)
    
    saver_id.restore(tf.get_default_session(),'./vae/weights/id/stage1_epoch_395.ckpt')
    saver_exp.restore(tf.get_default_session(),'./vae/weights/exp/stage1_epoch_395.ckpt')
    saver_gamma.restore(tf.get_default_session(),'./vae/weights/gamma/stage1_epoch_395.ckpt')
    saver_rot.restore(tf.get_default_session(),'./vae/weights/rot/stage1_epoch_395.ckpt')
#----------------------------------------------------------------------------

class FID(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        inception = misc.load_pkl('https://drive.google.com/uc?id=1MzTY44rLToO5APn8TZmfR7_ENSe5aZUn') # inception_v3_features.pkl
        activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)

        # Calculate statistics for reals.
        cache_file = self._get_cache_file_for_reals(num_images=self.num_images)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if os.path.isfile(cache_file):
            mu_real, sigma_real = misc.load_pkl(cache_file)
        else:
            for idx, images in enumerate(self._iterate_reals(minibatch_size=minibatch_size)):
                begin = idx * minibatch_size
                end = min(begin + minibatch_size, self.num_images)
                activations[begin:end] = inception.run(images[:end-begin], num_gpus=num_gpus, assume_frozen=True)
                if end == self.num_images:
                    break
            mu_real = np.mean(activations, axis=0)
            sigma_real = np.cov(activations, rowvar=False)
            misc.save_pkl((mu_real, sigma_real), cache_file)

        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                inception_clone = inception.clone()

                #----------------------------------------------------------------------------
                # Modified by Deng et al.
                latents = tf.random_normal([self.minibatch_per_gpu,128+32+16+3])
                INPUTcoeff = z_to_lambda_mapping(latents)

                if Gs_clone.input_shape[1] == 254:
                    INPUTcoeff_w_noise = INPUTcoeff
                else:
                    noise_coeff = tf.random_normal([self.minibatch_per_gpu,Gs_clone.input_shape[1]-254])
                    INPUTcoeff_w_noise = tf.concat([INPUTcoeff,noise_coeff], axis = 1)
                images = Gs_clone.get_output_for(INPUTcoeff_w_noise, None, is_validation=True, randomize_noise=True)
                images = tflib.convert_images_to_uint8(images)
                result_expr.append(inception_clone.get_output_for(images))

                restore_weights_and_initialize()
                #----------------------------------------------------------------------------

        # Calculate statistics for fakes.
        for begin in range(0, self.num_images, minibatch_size):
            end = min(begin + minibatch_size, self.num_images)
            activations[begin:end] = np.concatenate(tflib.run(result_expr), axis=0)[:end-begin]
        mu_fake = np.mean(activations, axis=0)
        sigma_fake = np.cov(activations, rowvar=False)

        # Calculate FID.
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        self._report_result(np.real(dist))

#----------------------------------------------------------------------------
