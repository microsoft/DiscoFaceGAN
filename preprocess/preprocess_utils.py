# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np 
import os
from scipy.io import loadmat,savemat
from PIL import Image,ImageOps
from array import array
import cv2


# Load expression basis provided by Guo et al.,
# https://github.com/Juyong/3DFace.
def LoadExpBasis():
	n_vertex = 53215
	Expbin = open('./renderer/BFM face model/Exp_Pca.bin','rb')
	exp_dim = array('i')
	exp_dim.fromfile(Expbin,1)
	expMU = array('f')
	expPC = array('f')
	expMU.fromfile(Expbin,3*n_vertex)
	expPC.fromfile(Expbin,3*exp_dim[0]*n_vertex)

	expPC = np.array(expPC)
	expPC = np.reshape(expPC,[exp_dim[0],-1])
	expPC = np.transpose(expPC)

	expEV = np.loadtxt('./renderer/BFM face model/std_exp.txt')

	return expPC,expEV

# Load BFM09 face model and transfer it to our face model
def transferBFM09():
	original_BFM = loadmat('./renderer/BFM face model/01_MorphableModel.mat')
	shapePC = original_BFM['shapePC'] # shape basis
	shapeEV = original_BFM['shapeEV'] # corresponding eigen value
	shapeMU = original_BFM['shapeMU'] # mean face
	texPC = original_BFM['texPC'] # texture basis
	texEV = original_BFM['texEV'] # eigen value
	texMU = original_BFM['texMU'] # mean texture

	expPC,expEV = LoadExpBasis() # expression basis and eigen value

	idBase = shapePC*np.reshape(shapeEV,[-1,199])
	idBase = idBase/1e5 # unify the scale to decimeter
	idBase = idBase[:,:80] # use only first 80 basis

	exBase = expPC*np.reshape(expEV,[-1,79])
	exBase = exBase/1e5 # unify the scale to decimeter
	exBase = exBase[:,:64] # use only first 64 basis

	texBase = texPC*np.reshape(texEV,[-1,199])
	texBase = texBase[:,:80] # use only first 80 basis

	# Our face model is cropped along face landmarks which contains only 35709 vertex.
	# original BFM09 contains 53490 vertex, and expression basis provided by Guo et al. contains 53215 vertex.
	# thus we select corresponding vertex to get our face model.

	index_exp = loadmat('./renderer/BFM face model/BFM_front_idx.mat')
	index_exp = index_exp['idx'].astype(np.int32) - 1 #starts from 0 (to 53215)

	index_shape = loadmat('./renderer/BFM face model/BFM_exp_idx.mat')
	index_shape = index_shape['trimIndex'].astype(np.int32) - 1 #starts from 0 (to 53490)
	index_shape = index_shape[index_exp]


	idBase = np.reshape(idBase,[-1,3,80])
	idBase = idBase[index_shape,:,:]
	idBase = np.reshape(idBase,[-1,80])

	texBase = np.reshape(texBase,[-1,3,80])
	texBase = texBase[index_shape,:,:]
	texBase = np.reshape(texBase,[-1,80])

	exBase = np.reshape(exBase,[-1,3,64])
	exBase = exBase[index_exp,:,:]
	exBase = np.reshape(exBase,[-1,64])

	meanshape = np.reshape(shapeMU,[-1,3])/1e5
	meanshape = meanshape[index_shape,:]
	meanshape = np.reshape(meanshape,[1,-1])

	meantex = np.reshape(texMU,[-1,3])
	meantex = meantex[index_shape,:]
	meantex = np.reshape(meantex,[1,-1])

	# region used for image rendering, and 68 landmarks index etc.
	gan_tl = loadmat('./renderer/BFM face model/gan_tl.mat')
	gan_tl = gan_tl['f']

	gan_mask = loadmat('./renderer/BFM face model/gan_mask.mat')
	gan_mask = gan_mask['idx']

	other_info = loadmat('./renderer/BFM face model/facemodel_info.mat')
	keypoints = other_info['keypoints']
	point_buf = other_info['point_buf']
	tri = other_info['tri']

	# save our face model
	savemat('./renderer/BFM face model/BFM_model_front_gan.mat',{'meanshape':meanshape,'meantex':meantex,'idBase':idBase,'exBase':exBase,'texBase':texBase,\
		'tri':tri,'point_buf':point_buf,'keypoints':keypoints,'gan_mask':gan_mask,'gan_tl':gan_tl})

#calculating least sqaures problem
def POS(xp,x):
	npts = xp.shape[1]

	A = np.zeros([2*npts,8])

	A[0:2*npts-1:2,0:3] = x.transpose()
	A[0:2*npts-1:2,3] = 1

	A[1:2*npts:2,4:7] = x.transpose()
	A[1:2*npts:2,7] = 1;

	b = np.reshape(xp.transpose(),[2*npts,1])

	k,_,_,_ = np.linalg.lstsq(A,b)

	R1 = k[0:3]
	R2 = k[4:7]
	sTx = k[3]
	sTy = k[7]
	s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
	t = np.stack([sTx,sTy],axis = 0)

	return t,s

# align image for 3D face reconstruction
def process_img(img,lm,t,s,target_size = 512.):
	w0,h0 = img.size
	w = (w0/s*102).astype(np.int32)
	h = (h0/s*102).astype(np.int32)
	img = img.resize((w,h),resample = Image.BICUBIC)

	left = (w/2 - target_size/2 + float((t[0] - w0/2)*102/s)).astype(np.int32)
	right = left + target_size
	up = (h/2 - target_size/2 + float((h0/2 - t[1])*102/s)).astype(np.int32)
	below = up + target_size

	img = img.crop((left,up,right,below))
	lm = np.stack([lm[:,0] - t[0] + w0/2,lm[:,1] - t[1] + h0/2],axis = 1)/s*102
	lm = lm - np.reshape(np.array([(w/2 - target_size/2),(h/2-target_size/2)]),[1,2])

	return img,lm

def Preprocess(img,lm,lm3D,target_size = 512.):

	w0,h0 = img.size

	# change from image plane coordinates to 3D sapce coordinates(X-Y plane)
	lm = np.stack([lm[:,0],h0 - 1 - lm[:,1]], axis = 1)

	# calculate translation and scale factors using 5 facial landmarks and standard landmarks
	t,s = POS(lm.transpose(),lm3D.transpose()) 
	s = s*224./target_size

	# processing the image
	img_new,lm_new = process_img(img,lm,t,s,target_size = target_size)
	lm_new = np.stack([lm_new[:,0],target_size - lm_new[:,1]], axis = 1)
	trans_params = np.array([w0,h0,102.0/s,t[0],t[1]])

	return img_new,lm_new,trans_params


def load_lm3d():

	Lm3D = loadmat('preprocess/similarity_Lm3D_all.mat')
	Lm3D = Lm3D['lm']

	# calculate 5 facial landmarks using 68 landmarks
	lm_idx = np.array([31,37,40,43,46,49,55]) - 1
	Lm3D = np.stack([Lm3D[lm_idx[0],:],np.mean(Lm3D[lm_idx[[1,2]],:],0),np.mean(Lm3D[lm_idx[[3,4]],:],0),Lm3D[lm_idx[5],:],Lm3D[lm_idx[6],:]], axis = 0)
	Lm3D = Lm3D[[1,2,0,3,4],:]

	return Lm3D

# load input images and corresponding 5 landmarks
def load_img(img_path,lm_path):

	image = Image.open(img_path)
	lm = np.loadtxt(lm_path)

	return image,lm


# Crop and rescale face region for GAN training
def crop_n_rescale_face_region(image,coeff):
	tx = coeff[0,254]
	ty = coeff[0,255]
	tz = coeff[0,256]
	f = 1015.*512/224
	cam_pos = 10.
	scale = 1.22*224/512

	# cancel translation and rescale face size
	M = np.float32([[1,0,-f*tx/(cam_pos - tz)],[0,1,f*ty/(cam_pos - tz)]])
	(rows, cols) = image.shape[:2]
	img_shift = cv2.warpAffine(image,M,(cols,rows))

	# crop image to 256*256
	scale_ = scale*(cam_pos - tz)/cam_pos
	w = int(cols*scale_)
	h = int(rows*scale_)
	res = cv2.resize(img_shift,(w,h))
	res = Image.fromarray(res.astype(np.uint8),'RGB')
	res = ImageOps.expand(res,border=10,fill = 'black')
	res = res.crop((round(w/2)-128+10,round(h/2)-128+10,round(w/2)+128+10,round(h/2)+128+10))
	res = np.array(res)
	res = res.astype(np.uint8)

	return res
