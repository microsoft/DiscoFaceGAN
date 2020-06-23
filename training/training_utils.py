# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Imitative-contrastive training scheme
import os
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from training import misc
from training.loss import *
from training.loss_control import *
from training.networks_stylegan import CoeffDecoder

def process_reals(x, lod, mirror_augment, drange_data, drange_net):
    with tf.name_scope('ProcessReals'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            x = misc.adjust_dynamic_range(x, drange_data, drange_net)
        if mirror_augment:
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
        with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
            s = tf.shape(x)
            y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
            y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
            y = tf.tile(y, [1, 1, 1, 2, 1, 2])
            y = tf.reshape(y, [-1, s[1], s[2], s[3]])
            x = tflib.lerp(x, y, lod - tf.floor(lod))
        with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
            s = tf.shape(x)
            factor = tf.cast(2 ** tf.floor(lod), tf.int32)
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

def restore_weights_and_initialize(train_stage_args):
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

    resnet_vars = [v for v in var_list if 'resnet_v1_50' in v.name]
    res_fc = [v for v in var_list if 'fc-id' in v.name or 'fc-ex' in v.name or 'fc-tex' in v.name or 'fc-angles' in v.name or 'fc-gamma' in v.name or 'fc-XY' in v.name or 'fc-Z' in v.name]
    resnet_vars += res_fc

    facerec_vars = [v for v in var_list if 'InceptionResnetV1' in v.name]
    

    saver_resnet = tf.train.Saver(var_list = resnet_vars)
    saver_facerec = tf.train.Saver(var_list = facerec_vars)
    saver_id = tf.train.Saver(var_list = var_id_list,max_to_keep = 100)
    saver_exp = tf.train.Saver(var_list = var_exp_list,max_to_keep = 100)
    saver_gamma = tf.train.Saver(var_list = var_gamma_list,max_to_keep = 100)
    saver_rot = tf.train.Saver(var_list = var_rot_list,max_to_keep = 100)
    

    saver_resnet.restore(tf.get_default_session(),os.path.join('./training/pretrained_weights/recon_net','FaceReconModel'))
    saver_facerec.restore(tf.get_default_session(),'./training/pretrained_weights/id_net/model-20170512-110547.ckpt-250000')
    saver_id.restore(tf.get_default_session(),'./vae/weights/id/stage1_epoch_395.ckpt')
    saver_exp.restore(tf.get_default_session(),'./vae/weights/exp/stage1_epoch_395.ckpt')
    saver_gamma.restore(tf.get_default_session(),'./vae/weights/gamma/stage1_epoch_395.ckpt')
    saver_rot.restore(tf.get_default_session(),'./vae/weights/rot/stage1_epoch_395.ckpt')

    if train_stage_args.func_name == 'training.training_utils.training_stage2':
        parser_vars = [v for v in var_list if 'FaceParser' in v.name]
        saver_parser = tf.train.Saver(var_list = parser_vars)
        saver_parser.restore(tf.get_default_session(),os.path.join('./training/pretrained_weights/parsing_net','faceparser_public'))


#----------------------------------------------------------------------------
# stage 1: train with imitative losses 
def training_stage1(
    FaceRender,
    noise_dim,
    weight_args,
    G_gpu,
    D_gpu,
    G_opt,
    D_opt,
    training_set, 
    G_loss_args,
    D_loss_args,
    lod_assign_ops,
    reals,
    labels,
    minibatch_split,
    resolution,
    drange_net,
    lod_in):
    
    print('Stage1: Imitative learning...\n')
    G_loss,D_loss = imitative_learning(FaceRender,noise_dim,weight_args,G_gpu,D_gpu,G_opt,D_opt,training_set, G_loss_args,\
        D_loss_args,lod_assign_ops,reals,labels,minibatch_split,resolution,drange_net,lod_in)

    return G_loss,D_loss

# stage 2: train with imitative losses and contrastive losses
def training_stage2(
    FaceRender,
    noise_dim,
    weight_args,
    G_gpu,
    D_gpu,
    G_opt,
    D_opt,
    training_set, 
    G_loss_args,
    D_loss_args,
    lod_assign_ops,
    reals,
    labels,
    minibatch_split,
    resolution,
    drange_net,
    lod_in):


    print('Stage2: Imitative learning and contrastive learning...\n')
    G_loss1,D_loss1 = contrastive_learning(FaceRender,noise_dim,weight_args,G_gpu,D_gpu,G_opt,D_opt,training_set, G_loss_args,\
        D_loss_args,lod_assign_ops,reals,labels,minibatch_split,resolution,drange_net,lod_in)

    G_loss2,D_loss2 = imitative_learning(FaceRender,noise_dim,weight_args,G_gpu,D_gpu,G_opt,D_opt,training_set, G_loss_args,\
        D_loss_args,lod_assign_ops,reals,labels,minibatch_split,resolution,drange_net,lod_in)


    G_loss = G_loss1 + G_loss2
    D_loss = D_loss1 + D_loss2

    return G_loss,D_loss


# Mapping z sampled from normal distribution to lambda space variables with physical meanings
def z_to_lambda_mapping(latents):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        with tf.variable_scope('id'):
            IDcoeff = CoeffDecoder(z = latents[:,:128],coeff_length = 160,ch_dim = 512, ch_depth = 3)
        with tf.variable_scope('exp'):
            EXPcoeff = CoeffDecoder(z = latents[:,128:128+32],coeff_length = 64,ch_dim = 256, ch_depth = 3)
        with tf.variable_scope('gamma'):
            GAMMAcoeff = CoeffDecoder(z = latents[:,128+32:128+32+16],coeff_length = 27,ch_dim = 128, ch_depth = 3)
        with tf.variable_scope('rot'):
            Rotcoeff = CoeffDecoder(z = latents[:,128+32+16:128+32+16+3],coeff_length = 3,ch_dim = 32, ch_depth = 3)

        INPUTcoeff = tf.concat([IDcoeff,EXPcoeff,Rotcoeff,GAMMAcoeff], axis = 1)

        return INPUTcoeff


def imitative_learning(
    FaceRender,
    noise_dim,
    weight_args,
    G_gpu,
    D_gpu,
    G_opt,
    D_opt,
    training_set, 
    G_loss_args,
    D_loss_args,
    lod_assign_ops,
    reals,
    labels,
    minibatch_split,
    resolution,
    drange_net,
    lod_in):

    latents = tf.random_normal([minibatch_split,128+32+16+3])
    INPUTcoeff = z_to_lambda_mapping(latents)

    noise_coeff = tf.random_normal([minibatch_split,noise_dim])

    INPUTcoeff_w_noise = tf.concat([INPUTcoeff,noise_coeff], axis = 1)
    INPUTcoeff_w_t = tf.concat([INPUTcoeff,tf.zeros([minibatch_split,3])], axis = 1)

    with tf.name_scope('FaceRender'):
        render_img,render_mask,render_landmark,_ = FaceRender.Reconstruction_Block(INPUTcoeff_w_t,resolution,minibatch_split,progressive=True)
        render_img = tf.transpose(render_img,perm=[0,3,1,2])
        render_mask = tf.transpose(render_mask,perm=[0,3,1,2])
        render_img = process_reals(render_img, lod_in, False, training_set.dynamic_range, drange_net)
        render_mask = process_reals(render_mask, lod_in, False, drange_net, drange_net)

        render_mask = tf.squeeze(render_mask,axis = 1)


    with tf.name_scope('G_loss'), tf.control_dependencies(lod_assign_ops):
        G_loss,fake_images = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, latents = INPUTcoeff_w_noise, opt=G_opt, training_set=training_set, minibatch_size=minibatch_split, **G_loss_args)
        l1_loss = L1_loss(render_img,fake_images,render_mask)
        skin_color_loss = Skin_color_loss(fake_images,render_img,render_mask)
        lm_loss, gamma_loss = Reconstruction_loss(fake_images,render_landmark,INPUTcoeff,FaceRender)
        id_loss = ID_loss(render_img,fake_images,render_mask)

        add_loss = tf.cond(resolution<=32, lambda:l1_loss*20., 
            lambda:lm_loss*weight_args.w_lm + gamma_loss*weight_args.w_gamma + id_loss*weight_args.w_id + skin_color_loss*weight_args.w_skin)

        G_loss += add_loss    
    with tf.name_scope('D_loss'), tf.control_dependencies(lod_assign_ops):
        D_loss = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, latents = INPUTcoeff_w_noise, opt=D_opt, training_set=training_set, minibatch_size=minibatch_split, reals=reals, labels=labels, **D_loss_args)

        return G_loss,D_loss


def contrastive_learning(
    FaceRender,
    noise_dim,
    weight_args,
    G_gpu,
    D_gpu,
    G_opt,
    D_opt,
    training_set, 
    G_loss_args,
    D_loss_args,
    lod_assign_ops,
    reals,
    labels,
    minibatch_split,
    resolution,
    drange_net,
    lod_in):

    # expression change pair
    latents_id = tf.tile(tf.random_normal([1,128]),[2,1])
    latents_exp = tf.random_normal([2,32])
    latents_gamma = tf.tile(tf.random_normal([1,16]),[2,1])
    latents_rot = tf.tile(tf.random_normal([1,3]),[2,1])
    latents_exp_pair = tf.concat([latents_id,latents_exp,latents_gamma,latents_rot], axis = 1)

    # lighting change pair
    latents_id = tf.tile(tf.random_normal([1,128]),[2,1])
    latents_exp = tf.tile(tf.random_normal([1,32]),[2,1])
    latents_gamma = tf.random_normal([2,16])
    latents_rot = tf.tile(tf.random_normal([1,3]),[2,1])
    latents_gamma_pair = tf.concat([latents_id,latents_exp,latents_gamma,latents_rot], axis = 1)

    latents = tf.concat([latents_exp_pair,latents_gamma_pair],axis = 0)
    INPUTcoeff = z_to_lambda_mapping(latents)

    noise_coeff = tf.random_normal([1,noise_dim])
    noise_coeff1 = tf.tile(noise_coeff,[2,1])
    noise_coeff = tf.random_normal([1,noise_dim])
    noise_coeff2 = tf.tile(noise_coeff,[2,1])
    noise_coeff_ = tf.concat([noise_coeff1,noise_coeff2],axis = 0)

    INPUTcoeff_w_noise = tf.concat([INPUTcoeff,noise_coeff_], axis = 1)
    INPUTcoeff_w_t = tf.concat([INPUTcoeff,tf.zeros([4,3])], axis = 1)

    with tf.name_scope('FaceRender'):
        render_img,render_mask,render_landmark,render_shape = FaceRender.Reconstruction_Block(INPUTcoeff_w_t,res=256,batchsize=4,progressive=False)
        render_img = tf.transpose(render_img,perm=[0,3,1,2])
        render_mask = tf.transpose(render_mask,perm=[0,3,1,2])
        render_img = process_reals(render_img, lod_in, False, training_set.dynamic_range, drange_net)
        render_mask = process_reals(render_mask, lod_in, False, drange_net, drange_net)
        render_mask = tf.squeeze(render_mask,axis = 1)

        shape1 = tf.expand_dims(render_shape[0],0)
        shape2 = tf.expand_dims(render_shape[1],0)
        mask1 = tf.expand_dims(render_mask[0],0)
        mask2 = tf.expand_dims(render_mask[1],0)

    with tf.name_scope('G_loss'), tf.control_dependencies(lod_assign_ops):
        G_loss,fake_images = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, latents = INPUTcoeff_w_noise, opt=G_opt, training_set=training_set, minibatch_size=minibatch_split, **G_loss_args)

        fake1 = tf.expand_dims(fake_images[0],0)
        fake1 = tf.transpose(fake1,perm=[0,2,3,1])
        fake2 = tf.expand_dims(fake_images[1],0)
        fake2 = tf.transpose(fake2,perm=[0,2,3,1])
        exp_warp_loss = Exp_warp_loss(fake1,fake2,shape1,shape2,mask1,mask2,FaceRender)

        fake3 = tf.expand_dims(fake_images[2],0)
        fake3 = tf.transpose(fake3,perm=[0,2,3,1])
        fake4 = tf.expand_dims(fake_images[3],0)
        fake4 = tf.transpose(fake4,perm=[0,2,3,1])        
        gamma_change_loss = Gamma_change_loss(fake3,fake4,FaceRender)

        add_loss = weight_args.w_exp_warp*exp_warp_loss + weight_args.w_gamma_change*gamma_change_loss
        G_loss += add_loss

    with tf.name_scope('D_loss'), tf.control_dependencies(lod_assign_ops):
        D_loss = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, latents = INPUTcoeff_w_noise, opt=D_opt, training_set=training_set, minibatch_size=minibatch_split, reals=reals, labels=labels, **D_loss_args)

        return G_loss, D_loss


