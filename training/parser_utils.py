# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Face parsing network proposed by Lin et al. 19,
# https://arxiv.org/abs/1906.01342,
# transfered to tensorflow version.
import tensorflow as tf 
from scipy.io import loadmat
import cv2
import os
import numpy as np

def transfer_68to5(points):
	# print(points)
	p1 = tf.reduce_mean(points[:,36:42,:],1)
	p2 = tf.reduce_mean(points[:,42:48,:],1)
	p3 = points[:,30,:]
	p4 = points[:,48,:]
	p5 = points[:,54,:]

	p = tf.stack([p1,p2,p3,p4,p5],axis = 1)

	return p

def standard_face_pts_512():
	pts = tf.constant([
		196.0, 226.0,
		316.0, 226.0,
		256.0, 286.0,
		220.0, 360.4,
		292.0, 360.4])

	pts = tf.reshape(pts,[5,2])

	return pts

def normalize_image(image):
    _mean = tf.constant([[[[0.485, 0.456, 0.406]]]])    # rgb
    _std = tf.constant([[[[0.229, 0.224, 0.225]]]])
    return (image / 255.0 - _mean) / _std

def affine_transform(points,std_points,batchsize):

	# batchsize = points.shape[0]
	p_num = points.shape[1]
	x = std_points[:,:,0]
	y = std_points[:,:,1]

	u = points[:,:,0]
	v = points[:,:,1]

	X1 = tf.stack([x,y,tf.ones([batchsize,p_num]),tf.zeros([batchsize,p_num])],axis = 2)
	X2 = tf.stack([y,-x,tf.zeros([batchsize,p_num]),tf.ones([batchsize,p_num])],axis = 2)
	X = tf.concat([X1,X2],axis = 1)

	U = tf.expand_dims(tf.concat([u,v],axis = 1),2)

	r = tf.squeeze(tf.matrix_solve_ls(X,U),[2])
	sc = r[:,0]
	ss = r[:,1]
	tx = r[:,2]
	ty = r[:,3]

	transform = tf.stack([sc,ss,tx,-ss,sc,ty,tf.zeros([batchsize]),tf.zeros([batchsize])],axis = 1)
	t = tf.stack([sc,-ss,tf.zeros([batchsize]),ss,sc,tf.zeros([batchsize]),tx,ty,tf.ones([batchsize])],axis = 1)
	t = tf.reshape(t,[-1,3,3])
	t = t + tf.reshape(tf.eye(3),[-1,3,3])*1e-5
	tinv = tf.matrix_inverse(t)

	return t,tinv

# similarity transformation for images
def similarity_transform(points,batchsize):

	std_points = standard_face_pts_512()
	std_points = tf.tile(tf.expand_dims(tf.reshape(std_points,[5,2]),0),[batchsize,1,1])

	t,tinv = affine_transform(points,std_points,batchsize)

	return t,tinv

def warp_and_distort(image,transform_matrix_inv,batchsize):
	yy = loadmat('./training/pretrained_weights/parsing_net/yy.mat')['grid']
	xx = loadmat('./training/pretrained_weights/parsing_net/xx.mat')['grid']
	yy = tf.constant(yy)
	xx = tf.constant(xx)
	yy = tf.tile(tf.expand_dims(yy,0),[batchsize,1,1])
	xx = tf.tile(tf.expand_dims(xx,0),[batchsize,1,1])

	yy = tf.reshape(yy,[-1,512*512])
	xx = tf.reshape(xx,[-1,512*512])
	xxyy_one = tf.stack([xx,yy,tf.ones_like(xx)], axis = 1) #batchx3x(h*w)
	transform_matrix_inv = tf.transpose(transform_matrix_inv,perm=[0,2,1])
	xxyy_one = tf.matmul(transform_matrix_inv,xxyy_one)

	xx = tf.reshape(xxyy_one[:,0,:]/xxyy_one[:,2,:], [-1,512,512])
	yy = tf.reshape(xxyy_one[:,1,:]/xxyy_one[:,2,:], [-1,512,512])	


	warp_image = tf.contrib.resampler.resampler(image,tf.stack([xx,yy],axis = 3))

	return warp_image

def preprocess_image_seg(image,lm5p):
	batchsize = 1
	t,tinv = similarity_transform(lm5p,batchsize)
	warp_image = warp_and_distort(image,t,batchsize)
	return warp_image,tinv

def _meshgrid(h,w):
	yy, xx = tf.meshgrid(np.arange(0,h,dtype = np.float32),np.arange(0,w, dtype = np.float32))
	return yy,xx

def _safe_arctanh(x):
	x = tf.clip_by_value(x,-0.999,0.999)
	x = tf.math.atanh(x)
	return x

def _distort(yy,xx,h,w,src_h,src_w,rescale = 1.0,distort_lambda = 1.0):

	def _non_linear(a):
		nl_part1 = tf.cast(a > (1.0 - distort_lambda),tf.float32)
		nl_part2 = tf.cast(a < (-1.0 + distort_lambda),tf.float32)
		nl_part3 = tf.cast(a == (-1.0 + distort_lambda),tf.float32)

		a_part1 = _safe_arctanh((a - 1.0 + distort_lambda)/distort_lambda)*distort_lambda + 1.0 - distort_lambda
		a_part2 = _safe_arctanh((a + 1.0 - distort_lambda)/distort_lambda)*distort_lambda - 1.0 + distort_lambda

		a = a_part1*nl_part1 + a_part2*nl_part2 + a*nl_part3
		return a

	yy = (yy / (h/2.0) - 1.0)*rescale
	yy = (_non_linear(yy) + 1.0) * src_h / 2.0
	xx = (xx / (w/2.0) -1.0)*rescale
	xx = (_non_linear(xx) + 1.0) * src_w / 2.0

	return yy,xx

def _undistort(yy,xx,h,w,src_h,src_w,rescale = 1.0,distort_lambda = 1.0):

	def _non_linear(a):
		nl_part1 = tf.cast(a > (1.0 - distort_lambda),tf.float32)
		nl_part2 = tf.cast(a < (-1.0 + distort_lambda),tf.float32)
		nl_part3 = tf.cast(a == (-1.0 + distort_lambda),tf.float32)

		a_part1 = tf.math.tanh((a - 1.0 + distort_lambda)/distort_lambda)*distort_lambda + 1.0 - distort_lambda
		a_part2 = tf.math.tanh((a + 1.0 - distort_lambda)/distort_lambda)*distort_lambda - 1.0 + distort_lambda

		a = a_part1*nl_part1 + a_part2*nl_part2 + a*nl_part3
		return a

	yy = _non_linear(yy / (h/2.0) -1.0)
	yy = (yy / rescale + 1.0) *src_h / 2.0
	xx = _non_linear(xx / (w/2.0) -1.0)
	xx = (xx / rescale + 1.0) *src_w / 2.0

	return yy,xx

def reverse_warp_and_distort(image,transform_matrix):
	batchsize = 1

	yy,xx = _meshgrid(256,256)


	yy = tf.tile(tf.expand_dims(yy,0),[batchsize,1,1])
	xx = tf.tile(tf.expand_dims(xx,0),[batchsize,1,1])
	yy = tf.reshape(yy,[-1,256*256])
	xx = tf.reshape(xx,[-1,256*256])

	xxyy_one = tf.stack([xx,yy,tf.ones_like(xx)], axis = 1) #batchx3x(h*w)
	transform_matrix = tf.transpose(transform_matrix,perm=[0,2,1])
	xxyy_one = tf.matmul(transform_matrix,xxyy_one)

	xx = tf.reshape(xxyy_one[:,0,:]/xxyy_one[:,2,:], [-1,256,256])
	yy = tf.reshape(xxyy_one[:,1,:]/xxyy_one[:,2,:], [-1,256,256])

	yy, xx = _undistort(yy,xx,512,512,512,512)

	warp_image = tf.contrib.resampler.resampler(image,tf.stack([xx,yy],axis = 3))

	return warp_image