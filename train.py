# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Main entry point for training StyleGAN networks."""

import copy
import dnnlib
from dnnlib import EasyDict
import argparse
import config
from metrics import metric_base

#----------------------------------------------------------------------------
# Official training configs for StyleGAN, targeted mainly for FFHQ.

if 1:
    desc          = 'sgan'     
    train = EasyDict()                                                            # Description string included in result subdir name.
    G             = EasyDict(func_name='training.networks_stylegan.G_style')               # Options for generator network.
    D             = EasyDict(func_name='training.networks_stylegan.D_basic')               # Options for discriminator network.
    G_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # Options for generator optimizer.
    D_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # Options for discriminator optimizer.
    G_loss        = EasyDict(func_name='training.loss.G_logistic_nonsaturating')           # Options for generator loss.
    D_loss        = EasyDict(func_name='training.loss.D_logistic_simplegp', r1_gamma=10.0) # Options for discriminator loss.
    dataset       = EasyDict()                                                             # Options for load_dataset().
    sched         = EasyDict()                                                             # Options for TrainingSchedule.
    grid          = EasyDict(size='1080p', layout='random')                                   # Options for setup_snapshot_image_grid().
    metrics       = [metric_base.fid50k]                                                   # Options for MetricGroup.
    submit_config = dnnlib.SubmitConfig()                                                  # Options for dnnlib.submit_run().
    tf_config     = {'rnd.np_random_seed': 1000}                                           # Options for tflib.init_tf().

    # Dataset.
    desc += '-ffhq256';  dataset = EasyDict(tfrecord_dir='ffhq_align', resolution=256); train.mirror_augment = True

    # Number of GPUs.
    #desc += '-1gpu'; submit_config.num_gpus = 1; sched.minibatch_base = 4; sched.minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}
    #desc += '-2gpu'; submit_config.num_gpus = 2; sched.minibatch_base = 8; sched.minibatch_dict = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}
    desc += '-4gpu'; submit_config.num_gpus = 4; sched.minibatch_base = 16; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
    # desc += '-8gpu'; submit_config.num_gpus = 8; sched.minibatch_base = 32; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}

    # Default options.
    train.total_kimg = 25000
    sched.lod_initial_resolution = 8
    sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)

#----------------------------------------------------------------------------
# Main entry point for training.
# Calls the function indicated by 'train' using the selected options.


#----------------------------------------------------------------------------
# Modified by Deng et al.
def parse_args():
    desc = "Tensorflow implementation of DisentangledFaceGAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--w_id', type=float, default=3, help='weight for identity perceptual loss')
    parser.add_argument('--w_lm', type=float, default=500, help='weight for landmark loss')
    parser.add_argument('--w_gamma', type=float, default=10, help='weight for lighting loss')
    parser.add_argument('--w_skin', type=float, default=20, help='weight for face region loss')
    parser.add_argument('--w_exp_warp', type=float, default=10, help='weight for expression change loss')
    parser.add_argument('--w_gamma_change', type=float, default=10, help='weight for lighting change loss')
    parser.add_argument('--noise_dim', type=int, default=32, help='dimension of the additional noise factor')
    parser.add_argument('--stage', type=int, default=1, help='training stage. 1 = imitative losses only; 2 = imitative losses and contrastive losses')
    parser.add_argument('--run_id', type=int, default=0, help='run ID or network pkl to resume training from')
    parser.add_argument('--snapshot', type=int, default=0, help='snapshot index to resume training from')
    parser.add_argument('--kimg', type=float, default=0, help='assumed training progress at certain number of images')

    return parser.parse_args()
#----------------------------------------------------------------------------


def main():

    #------------------------------------------------------------------------
    # Modified by Deng et al.
    args = parse_args()
    if args is None:
      exit()


    weight_args = EasyDict()
    weight_args.update(w_id=args.w_id,w_lm=args.w_lm,w_gamma=args.w_gamma,w_skin=args.w_skin,
        w_exp_warp=args.w_exp_warp,w_gamma_change=args.w_gamma_change)

    train.update(run_func_name='training.training_loop.training_loop')
    kwargs = EasyDict(train)

    # stage 1: training with only imitative losses with 15000k images.
    if args.stage == 1:
        train_stage = EasyDict(func_name='training.training_utils.training_stage1')
        kwargs.update(total_kimg=15000)

    # stage 2: training with imitative losses and contrastive losses.
    else:
        train_stage = EasyDict(func_name='training.training_utils.training_stage2')
        kwargs.update(resume_run_id=args.run_id,resume_snapshot=args.snapshot,resume_kimg=args.kimg)
        kwargs.update(total_kimg=25000)
        weight_args.update(w_lm=100)

    kwargs.update(train_stage_args=train_stage)
    kwargs.update(weight_args = weight_args,noise_dim = args.noise_dim)
    #------------------------------------------------------------------------

    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.update(dataset_args=dataset, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(submit_config)
    kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
    kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
