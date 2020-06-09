# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# https://github.com/daib13/TwoStageVAE
import argparse 
import os 
from two_stage_vae_model import *
import numpy as np 
import tensorflow as tf 
import math 
import time 
from data_loader import *


def main():
    tf.reset_default_graph()
    exp_folder = os.path.join(args.output_path, args.factor)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    model_path = os.path.join(exp_folder, 'from_scratch')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # train VAE for different factors
    if args.factor == 'id':
        coeff_dim = 160
        latent_dim = 128
        ch_dim = 512
        ch_depth = 3
    elif args.factor == 'exp':
        coeff_dim = 64
        latent_dim = 32
        ch_dim = 256
        ch_depth = 3
    elif args.factor == 'gamma':
        coeff_dim = 27
        latent_dim = 16
        ch_dim = 128
        ch_depth = 3
    else:
        coeff_dim = 3
        latent_dim = 3
        ch_dim = 32
        ch_depth = 3

    # input
    input_x = tf.placeholder(tf.float32, [args.batch_size, coeff_dim], 'x')
    data_worker = DataFetchWorker(factor=args.factor,path=args.datapath, batch_size=args.batch_size, shuffle=True)
    num_sample = data_worker.total_subj_train
    print(num_sample)

    # model
    with tf.variable_scope(args.factor):
        model = MLP(input_x, latent_dim, ch_dim, ch_depth, args.cross_entropy_loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(exp_folder, sess.graph)
    saver = tf.train.Saver()



    # train model
    iteration_per_epoch = np.ceil(num_sample / args.batch_size).astype(np.int32)
    if not args.val:
        # first stage
        data_worker.run()

        for epoch in range(args.epochs):
            lr = args.lr if args.lr_epochs <= 0 else args.lr * math.pow(args.lr_fac, math.floor(float(epoch) / float(args.lr_epochs)))
            epoch_loss = 0
            for j in range(iteration_per_epoch):
                loss = model.step(data_worker, lr, sess, writer, args.write_iteration)
                epoch_loss += loss 
            epoch_loss /= iteration_per_epoch

            print('Date: {date}\t'
                  'Epoch: [Stage 1][{0}/{1}]\t'
                  'Loss: {2:.4f}.'.format(epoch, args.epochs, epoch_loss, date=time.strftime('%Y-%m-%d %H:%M:%S')))

            if epoch%5 == 0:
                saver.save(sess, os.path.join(model_path, 'stage1_'+'epoch_%d.ckpt'%epoch))

        data_worker.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-folder', type=str, default='.')
    parser.add_argument('--output-path', type=str, default='./weights')
    parser.add_argument('--factor', type=str, default='rot')

    parser.add_argument('--datapath', type=str, default='../FFHQ_data/coeff')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--write-iteration', type=int, default=600)

    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr-epochs', type=int, default=150)
    parser.add_argument('--lr-fac', type=float, default=0.5)

    parser.add_argument('--cross-entropy-loss', default=False, action='store_true')    
    parser.add_argument('--val', default=False, action='store_true')

    args = parser.parse_args()
    print(args)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    main()