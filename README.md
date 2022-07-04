## *DiscoFaceGAN*: *Dis*entangled and *Co*ntrollable Face Image Generation via 3D Imitative-Contrastive Learning ##

<p align="center"> 
<img src="/images/teaser.gif">
</p>


This is a tensorflow implementation of the following paper:

**Disentangled and Controllable Face Image Generation via 3D Imitative-Contrastive Learning**, CVPR 2020. (**_Oral_**)

Yu Deng, Jiaolong Yang, Dong Chen, Fang Wen, and Xin Tong

Paper: [https://arxiv.org/abs/2004.11660](https://arxiv.org/abs/2004.11660v2)

Abstract: _We propose **DiscoFaceGAN**, an approach for face image generation of virtual people with **DIS**entangled, precisely-**CO**ntrollable latent representations for identity of non-existing people, expression, pose, and illumination. We embed 3D priors into adversarial learning and train the network to imitate the image formation of an analytic 3D face deformation and rendering process. To deal with the generation freedom induced by the domain gap between real and rendered faces, we further introduce contrastive learning to promote disentanglement by comparing pairs of generated images. Experiments show that through our imitative-contrastive learning, the factor variations are very well disentangled and the properties of a generated face can be precisely controlled. We also analyze the learned latent space and present several meaningful properties supporting factor disentanglement. Our method can also be used to embed real images into the disentangled latent space. We hope our method could provide new understandings of the relationship between physical properties and deep image synthesis._

## Features

### ● Factor disentanglement
When generating face images, we can freely change the four factors including identity, expression, lighting, and pose. The factor variations are highly disentangled: changing one factor does not affect others.

<p align="center"> 
<img src="/images/disentangled.png">
</p>


### ● Reference based generation
We achieve reference-based generation where we extract expression, pose and lighting from a given image and generate new identities with similar properties.

<p align="center"> 
<img src="/images/reference.png" width="600">
</p>

### ● Real image pose manipulation
We can use our method to embed a real image into the disentangled latent space and edit it, such as pose manipulation.
<p align="center"> 
<img src="/images/pose.png" width="700">
</p>

### ● Real image lighting editing
We can edit the lighting of a real image.
<p align="center"> 
<img src="/images/light.png" width="600">
</p>

### ● Real image expression transfer
We can also achieve expression transfer of real images.
<p align="center"> 
<img src="/images/expression.png" width="600">
</p>

##
The training code of our model are mainly borrowed from [StyleGAN](https://github.com/NVlabs/stylegan), although our method can be applied to any form of GANs.

## Testing requirements
- Python 3.6. We recommend Anaconda3 with numpy 1.14.3 or newer.
- Tensorflow 1.12 with GPU support.
- CUDA toolkit 9.0 or newer, cuDNN 7.3.1 or newer.

## Testing with pre-trained network
1. Clone the repository:

```
git clone https://github.com/microsoft/DiscoFaceGAN.git
cd DiscoFaceGAN
```
2. Generate images using pre-trained network:

```
# Generate face images with random variations of expression, lighting, and pose
python generate_images.py

# Generate face images with random variations of expression
python generate_images.py --factor 1

# Generate face images with random variations of lighting
python generate_images.py --factor 2

# Generate face images with random variations of pose
python generate_images.py --factor 3
```

## Training requirements

- Only Linux is supported.
- Python 3.6. We recommend Anaconda3 with numpy 1.14.3 or newer.
- Tensorflow 1.12 with GPU support.
- CUDA toolkit 9.0 or newer, cuDNN 7.3.1 or newer.
- One or more high-end NVIDIA GPUs. We recommend using at least 4 Tesla P100 GPUs for training.
- [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model). 
- [Expression Basis](https://github.com/Juyong/3DFace) provided by Guo et al.. The original BFM09 model does not handle expression variations so extra expression basis are needed. 
- [tf_mesh_renderer](https://github.com/google/tf_mesh_renderer) provided by Genova et al.. We use the tool to render synthetic face images during training. We recommend using its [older version](https://github.com/google/tf_mesh_renderer/tree/ba27ea1798f6ee8d03ddbc52f42ab4241f9328bb) because we find its latest version unstable during our training process.
- [Facenet](https://github.com/davidsandberg/facenet) provided by 
Sandberg et al. 
In our paper, we use a network to exrtact deep face features. This network model cannot be publicly released. As an alternative, we recommend using the Facenet model. We use the version [20170512-110547](https://github.com/davidsandberg/facenet/blob/529c3b0b5fc8da4e0f48d2818906120f2e5687e6/README.md) trained on MS-Celeb-1M. Training process has been tested with this model to ensure similar results.
- [3D face reconstruction network](https://github.com/microsoft/Deep3DFaceReconstruction). We use the network to extract identity, expression, lighting, and pose coefficients.
- [Face parsing network](https://arxiv.org/abs/1906.01342) provided by Lin et al.. We use the network to obtain hair segmentation masks during training.


## Training preparation

1. Download the Basel Face Model. Due to the license agreement of Basel Face Model, you have to submit an application on its [home page](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads). After getting the access to BFM data, download "01_MorphableModel.mat" and put it in "./renderer/BFM face model".
2. Download the [Expression Basis](https://github.com/Juyong/3DFace). You can find a link named "CoarseData" in the first row of Introduction part in their repository. Download and unzip the Coarse_Dataset.zip. Put "Exp_Pca.bin" in "./renderer/BFM face model".
3. Install tf_mesh_renderer. For convenience, we provide a [pre-compiled file](https://drive.google.com/file/d/1VlPXvrvH_HYUf9ePlkvL2E93a1aK4eUg/view?usp=sharing) of the library under tensorflow 1.12. Download the file and put it in "./renderer".
4. Download the [pre-trained weights](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit) of Facenet provided by Sandberg et al., unzip it and put all files in "./training/pretrained_weights/id_net".
5. Download the [pre-trained weights](https://drive.google.com/file/d/176LCdUDxAj7T2awQ5knPMPawq5Q2RUWM/view?usp=sharing) of 3D face reconstruction network, unzip it and put all files in "./training/pretrained_weights/recon_net".
6. Download the [pre-trained weights](https://drive.google.com/file/d/1YkvI_B-cPNo1NhTjiEk8O8FVnVpIypNd/view?usp=sharing) of face parser provided by Lin et al., unzip it and put all files in "./training/pretrained_weights/parsing_net".

## Data pre-processing
1. Download [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset). Detect 5 facial landmarks for all images. We recommend using [dlib](http://dlib.net/) or [MTCNN](https://github.com/ipazc/mtcnn). Save all images in <raw_image_path> and corresponding landmarks in <raw_lm_path>. Note that an image and its detected landmark file should have same name.
2. Align images and extract coefficients for VAE and GAN training:

```
python preprocess_data.py --image_path=<raw_image_path> --lm_path=<raw_lm_path> --save_path=<save_path_for_processed_data>
```
3. Convert the aligned images to multi-resolution TFRecords similar as in [StyleGAN](https://github.com/NVlabs/stylegan):

```
python dataset_tool.py create_from_images ./datasets/ffhq_align <save_path_for_processed_data>/img
```

## Training networks
1. We provide pre-trained VAEs for factors of identity, expression, lighting, and pose. To train new models from scratch, run:

```
cd vae

# train VAE for identity coefficients
python demo.py --datapath <save_path_for_processed_data>/coeff --factor id

# train VAE for expression coefficients
python demo.py --datapath <save_path_for_processed_data>/coeff --factor exp

# train VAE for lighting coefficients
python demo.py --datapath <save_path_for_processed_data>/coeff --factor gamma

# train VAE for pose coefficients
python demo.py --datapath <save_path_for_processed_data>/coeff --factor rot
```
2. Train the Stylegan generator with imitative-contrastive learning scheme:

```
# Stage 1 with only imitative losses, training with 15000k images
python train.py 

# Stage 2 with both imitative losses and contrastive losses, training with another 5000k images
python train.py --stage 2 --run_id <stage1_model_id> --snapshot <stage1_model_snapshot> --kimg <stage1_model_snapshot> 
# For example
python train.py --stage 2 --run_id 0 --snapshot 14926 --kimg 14926
```

After training, the network can be used similarly as the provided pre-trained model:
```
# Generate face images with specific model
python generate_images.py --model <your_model_path.pkl>
```

We have trained the model using a configuration of 4 Tesla P100 GPUs. It takes 6d 15h for stage 1 and 5d 8h for stage 2.

## Contact
If you have any questions, please contact Yu Deng (dengyu2008@hotmail.com) and Jiaolong Yang (jiaoyan@microsoft.com)

## License

Copyright &copy; Microsoft Corporation.

Licensed under the MIT license.

## Citation

Please cite the following paper if this model helps your research:

    @inproceedings{deng2020disentangled,
		title={Disentangled and Controllable Face Image Generation via 3D Imitative-Contrastive Learning},
    	author={Yu Deng and Jiaolong Yang and Dong Chen and Fang Wen and Xin Tong},
	    booktitle={IEEE Computer Vision and Pattern Recognition},
	    year={2020}
	}

##
The real face images on this page are from the public FFHQ dataset released under Creative Commons BY-NC-SA 4.0 license. Detailed information can be found on its [website](https://github.com/NVlabs/ffhq-dataset).
