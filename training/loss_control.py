# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Losses for imitative and contrastive learning
import tensorflow as tf
from dnnlib.tflib.autosummary import autosummary
from training.networks_recon import R_Net
from training.networks_id import Perceptual_Net
from training.networks_parser import Parsing
import numpy as np

#---------------------------------------------------------------------------

def gaussian_kernel(size=5,sigma=2):
    x_points = np.arange(-(size-1)//2,(size-1)//2+1,1)
    y_points = x_points[::-1]
    xs,ys = np.meshgrid(x_points,y_points)
    kernel = np.exp(-(xs**2+ys**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    kernel = kernel/kernel.sum()
    kernel = tf.constant(kernel,dtype=tf.float32)

    return kernel

def gaussian_blur(image,size=5,sigma=2):
    kernel = gaussian_kernel(size=size,sigma=sigma)
    kernel = tf.tile(tf.reshape(kernel,[tf.shape(kernel)[0],tf.shape(kernel)[1],1,1]),[1,1,3,1])
    blur_image = tf.nn.depthwise_conv2d(image,kernel,strides=[1,1,1,1],padding='SAME',data_format='NHWC')

    return blur_image

#----------------------------------------------------------------------------
# Imitative losses

# L1 loss between rendered image and fake image
def L1_loss(render_img,fake_images,render_mask):
    l1_loss = tf.reduce_sum(tf.sqrt(tf.reduce_sum((render_img - fake_images)**2, axis = 1) + 1e-8 )*render_mask)/tf.reduce_sum(render_mask)
    l1_loss = autosummary('Loss/l1_loss', l1_loss)
    return l1_loss

# landmark loss and lighting loss between rendered image and fake image
def Reconstruction_loss(fake_image,landmark_label,coeff_label,FaceRender):
    landmark_label = landmark_label*224./256.

    fake_image = (fake_image+1)*127.5
    fake_image = tf.clip_by_value(fake_image,0,255)
    fake_image = tf.transpose(fake_image,perm=[0,2,3,1])
    fake_image = tf.reverse(fake_image,[3]) #RGBtoBGR
    fake_image = tf.image.resize_images(fake_image,size=[224, 224], method=tf.image.ResizeMethod.BILINEAR)

    # input to R_Net should have a shape of [batchsize,224,224,3], color range from 0-255 in BGR order.
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        coeff = R_Net(fake_image,is_training=False, reuse=tf.AUTO_REUSE)
    landmark_p = FaceRender.Get_landmark(coeff) #224*224

    landmark_weight = tf.ones([1,68])
    landmark_weight = tf.reshape(landmark_weight,[1,68,1])
    lm_loss =  tf.reduce_mean(tf.square((landmark_p-landmark_label)/224)*landmark_weight)


    fake_gamma = coeff[:,227:254]
    render_gamma = coeff_label[:,227:254]

    gamma_loss = tf.reduce_mean(tf.abs(fake_gamma - render_gamma)) 

    lm_loss = autosummary('Loss/lm_loss', lm_loss)
    gamma_loss = autosummary('Loss/gamma_loss', gamma_loss)


    return lm_loss,gamma_loss

# identity similarity loss between rendered image and fake image
def ID_loss(render_image,fake_image,render_mask):

    render_image = (render_image+1)*127.5
    render_image = tf.clip_by_value(render_image,0,255)
    render_image = tf.transpose(render_image,perm=[0,2,3,1])
    render_image = tf.image.resize_images(render_image,size=[160,160], method=tf.image.ResizeMethod.BILINEAR)
    fake_image = (fake_image+1)*127.5
    fake_image = tf.clip_by_value(fake_image,0,255)
    fake_image = tf.transpose(fake_image,perm=[0,2,3,1])
    fake_image = fake_image*tf.expand_dims(render_mask,3)
    fake_image = tf.image.resize_images(fake_image,size=[160,160], method=tf.image.ResizeMethod.BILINEAR)

    render_image = tf.reshape(render_image,[-1,160,160,3])

    # input to face recognition network should have a shape of [batchsize,160,160,3], color range from 0-255 in RGB order.
    id_fake = Perceptual_Net(fake_image)
    id_render = Perceptual_Net(render_image)

    id_fake = tf.nn.l2_normalize(id_fake, dim = 1)
    id_render = tf.nn.l2_normalize(id_render, dim = 1)
    # cosine similarity
    sim = tf.reduce_sum(id_fake*id_render,1)
    loss = tf.reduce_mean(tf.maximum(0.3,1.0 - sim))   # need clip! IMPORTANT

    loss = autosummary('Loss/id_loss', loss)

    return loss

# average skin color loss between rendered image and fake image
def Skin_color_loss(fake,render,mask):
    mask = tf.expand_dims(mask,1)
    mean_fake = tf.reduce_sum(fake*mask,[2,3])/tf.reduce_sum(mask,[2,3])
    mean_render = tf.reduce_sum(render*mask,[2,3])/tf.reduce_sum(mask,[2,3])

    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum((mean_fake - mean_render)**2, axis = 1) + 1e-8 ))
    loss = autosummary('Loss/skin_loss', loss)

    return loss

#----------------------------------------------------------------------------
# Contrastive losses

# loss for expression change
def Exp_warp_loss(fake1,fake2,r_shape1,r_shape2,mask1,mask2,FaceRender):

    pos1_2d = FaceRender.Projection_block(r_shape1)
    pos2_2d = FaceRender.Projection_block(r_shape2)
    pos_diff = pos1_2d - pos2_2d
    pos_diff = tf.stack([-pos_diff[:,:,1],pos_diff[:,:,0]],axis = 2)
    pos_diff = tf.concat([pos_diff,tf.zeros([tf.shape(pos_diff)[0],tf.shape(pos_diff)[1],1])], axis = 2)
    flow_1to2,_ = FaceRender.Render_block(r_shape2,tf.zeros_like(r_shape2),pos_diff,FaceRender.facemodel,256,1)
    flow_1to2 = flow_1to2[:,:,:,:2]
    fake_1to2 = tf.contrib.image.dense_image_warp(fake1,-flow_1to2) # IMPORTANT!
    loss_mask = tf.cast((mask1 - mask2) <= 0, tf.float32)
    fake2 = gaussian_blur(fake2,size=5,sigma=2)
    fake_1to2 = gaussian_blur(fake_1to2,size=5,sigma=2)

    loss = tf.reduce_sum(tf.sqrt(tf.reduce_sum((fake2 - fake_1to2)**2,axis = 3) + 1e-8)*loss_mask)/tf.reduce_sum(loss_mask)
    loss = autosummary('Loss/Exp_warp_loss', loss)

    return loss

# loss for lighting change
def Gamma_change_loss(fake1,fake2,FaceRender):
    fake_image = tf.concat([fake1,fake2],axis = 0)
    fake_image = (fake_image+1)*127.5
    fake_image = tf.clip_by_value(fake_image,0,255)
    fake_image = tf.reverse(fake_image,[3]) #RGBtoBGR
    fake_image = tf.image.resize_images(fake_image,size=[224, 224], method=tf.image.ResizeMethod.BILINEAR)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        coeff = R_Net(fake_image,is_training=False, reuse=tf.AUTO_REUSE)
    landmark_p = FaceRender.Get_landmark(coeff)
    landmark_p = landmark_p*256/224.

    lm1 = tf.expand_dims(landmark_p[0],0)
    lm2 = tf.expand_dims(landmark_p[1],0)
    hair_region_loss = Hair_region_loss(fake1,fake2,lm1,lm2)
    id_consistent_loss = ID_consistent_loss(fake1,fake2)
    lm_consistent_loss = Lm_consistent_loss(lm1,lm2)

    loss = hair_region_loss + 2*id_consistent_loss + 1000*lm_consistent_loss
    return loss

# hair region consistency between fake image pair
def Hair_region_loss(fake1,fake2,lm1,lm2):
    fake1 = (fake1+1)*127.5
    fake1 = tf.clip_by_value(fake1,0,255)
    fake2 = (fake2+1)*127.5
    fake2 = tf.clip_by_value(fake2,0,255)

    # input to face parser should have a shape of [batchsize,256,256,3], color range from 0-255 in RGB order.
    seg_mask1 = Parsing(fake1,lm1)
    seg_mask2 = Parsing(fake2,lm2)

    hair_mask1 = seg_mask1[:,:,:,2]
    hair_mask2 = seg_mask2[:,:,:,2]

    loss = tf.reduce_mean((hair_mask1-hair_mask2)**2)
    loss = autosummary('Loss/Hair_region_loss', loss)

    return loss


# identity consistency between fake image pair
def ID_consistent_loss(fake1,fake2):
    fake1 = (fake1+1)*127.5
    fake1 = tf.clip_by_value(fake1,0,255)
    fake1 = tf.image.resize_images(fake1,size=[160,160], method=tf.image.ResizeMethod.BILINEAR)

    fake2 = (fake2+1)*127.5
    fake2 = tf.clip_by_value(fake2,0,255)  
    fake2 = tf.image.resize_images(fake2,size=[160,160], method=tf.image.ResizeMethod.BILINEAR)

    id_fake1 = Perceptual_Net(fake1)
    id_fake2 = Perceptual_Net(fake2)

    id_fake1 = tf.nn.l2_normalize(id_fake1, dim = 1)
    id_fake2 = tf.nn.l2_normalize(id_fake2, dim = 1)
    # cosine similarity
    sim = tf.reduce_sum(id_fake1*id_fake2,1)
    loss = tf.reduce_mean(1.0 - sim)
    loss = autosummary('Loss/ID_consistent_loss', loss)

    return loss

# landmark consistency between fake image pair
def Lm_consistent_loss(landmark_p1,landmark_p2):
    loss = tf.reduce_mean(tf.square((landmark_p1-landmark_p2)/224))
    loss = autosummary('Loss/Lm_consistent_loss', loss)

    return loss