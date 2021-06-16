# -*- coding:utf-8 -*-
import argparse
import numpy as np
from pathlib import Path
import cv2

# Make sure that caffe is on the python path:
# this file is expected to be in {caffe_root}/examples  
caffe_root = '/home/gaokaijun/caffe_v11/'
import os  
os.chdir(caffe_root)  
import sys  
sys.path.insert(0, 'python') 

import caffe  
caffe.set_device(0)  
caffe.set_mode_gpu()  

model_deploy = '/data_1/noise2noise-master/by_caffe/SRResNet-BN_deploy.prototxt'  
model_weights = '/data_1/noise2noise-master/by_caffe/snap/0515_iter_40276.caffemodel'  

net = caffe.Net(model_deploy, model_weights, caffe.TEST)
print("load caffe model done") 

def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)

def gaussian_noise(img):
    min_stddev = 0
    max_stddev = 50
    noise_img = img.astype(np.float)
    stddev = np.random.uniform(min_stddev, max_stddev)
    noise = np.random.randn(*img.shape) * stddev
    noise_img += noise
    noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
    return noise_img
#return gaussian_noise

def main():
    image_dir = "/data_1/noise2noise-master/chejian_img/test_in.txt"
    fid = open(image_dir, 'r')
    lines = fid.readlines()
    fid.close()
    for line in lines:
        line = line.strip('\n')
        #print line
        image = cv2.imread(line)

        img_width = 128 
        img_height = 128
        h, w, _ = image.shape
        i = np.random.randint(h - img_height + 1)
        j = np.random.randint(w - img_width + 1)
        image = image[i:i + img_height, j:j + img_width]

        noise_image = gaussian_noise(image)

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]  
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  
        transformer.set_transpose('data', (2, 0, 1))  
        transformer.set_mean('data', np.array([0,0,0])) # mean pixel  
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]  
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB 

        #Run the net and examine the top_k results  
        net.blobs['data'].data[...] = transformer.preprocess('data', noise_image / 255.0)
        # Forward pass. 
        out = net.forward()
        denoised_image = net.blobs['conv3'].data
        channel_swap = (0, 2, 3, 1)
  
         
        denoised_image = net.forward()['conv3']
        #h1, w1, c1, b1= denoised_image.shape
        #print("h1: ", h1, " w1: ", w1, " c1: ", c1, " b1: ", b1)

        out_image = out_image.transpose(channel_swap)

        #denoised_image = denoised_image[1:3]
        denoised_image = np.swapaxes(denoised_image, 1, 3)
        denoised_image = np.swapaxes(denoised_image, 1, 2)
        #h1, w1, c1, b1= denoised_image.shape
        #print("h1: ", h1, " w1: ", w1, " c1: ", c1, " b1: ", b1)

        out_image = np.zeros((img_height, img_width * 3, 3), dtype=np.uint8)
        out_image[:, :img_width] = image
        out_image[:, img_width:img_width * 2] = noise_image
        #c=np.array(denoised_image[0,:,:,:], dtype=np.uint8)
        out_image[:, img_width * 2:] = denoised_image[0,:,:,:]

        print(denoised_image[0,:,:,:])

        cv2.imshow("result", out_image)
        cv2.waitKey()


if __name__ == '__main__':
    main()
