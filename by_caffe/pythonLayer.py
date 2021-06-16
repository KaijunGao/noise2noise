#coding:utf-8
#-*- encoding=utf-8 -*-

import sys
import cv2
import caffe
import numpy as np
import random 
import gc

# Make sure that caffe is on the python path:  
#caffe_root = '/home/gaokaijun/caffe_v11/'  # this file is expected to be in {caffe_root}/examples  
#import os  
#os.chdir(caffe_root)  
#import sys  
#sys.path.insert(0, 'python') 

def gaussian_noise(img):
    min_stddev = 0
    max_stddev = 50
    noise_img = img.astype(np.float)
    stddev = np.random.uniform(min_stddev, max_stddev)
    noise = np.random.randn(*img.shape) * stddev
    noise_img += noise
    noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
    return noise_img
return gaussian_noise

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()

################################################################################
#########################Data Layer By Python###################################
################################################################################
class Data_Layer_train(caffe.Layer):
    def setup(self, bottom, top):
        self.batch_size = 32
        img_width = 128
        img_height = 128
        data_src_file = '/data_1/noise2noise-master/chejian_img/test_in.txt'

        self.batch_loader = BatchLoader(data_src_file, img_width, img_height)
        #top[0].reshape(self.batch_size, 3, img_width, img_height)
        #top[1].reshape(self.batch_size, 3, img_width, img_height)
        top[0].reshape(self.batch_size, 3, img_height, img_width)
        top[1].reshape(self.batch_size, 3, img_height, img_width)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            src, dst = self.batch_loader.load_next_image()
            top[0].data[itt, ...] = src
            top[1].data[itt, ...] = dst

    def backward(self, top, propagate_down, bottom):
        pass

class BatchLoader(object):
    def __init__(self, data_src_file, img_width, img_height):
        self.mean = 0 #128
        self.im_shape_w = img_width
        self.im_shape_h = img_height
        self.data_src_list = []
        self.data_dst_list = []

        print("Start Reading img_src & img_dst into Memory...")
        fid = open(data_src_file, 'r')
        lines = fid.readlines()
        fid.close()
        cur_=0
        sum_=len(lines)
        for line in lines:
            view_bar(cur_, sum_)
            cur_+=1
            line = line.strip('\n')
            #print line
            img_src = cv2.imread(line)
            h,w,ch = img_src.shape
            #if h!=self.im_shape_h or w!=self.im_shape_w:
                #img_src = cv2.resize(img_src, (int(self.im_shape_w), int(self.im_shape_h)))

            img_src_roi = img_src
            img_dst_roi = img_src
            if h >= im_shape_h and w >= im_shape_w:
                h, w, _ = img_src.shape
                i = np.random.randint(h - im_shape_h + 1)
                j = np.random.randint(w - im_shape_w + 1)
                img_src_roi = img_src[i:i + im_shape_h, j:j + im_shape_w]

                #
                img_dst_roi = gaussian_noise(img_src_roi)

                cv2.imshow("source_resize", img_src_roi)
                cv2.imshow("target_resize", img_dst_roi)
                cv2.waitKey()
            else:
                #
                img_dst_roi = gaussian_noise(img_src_roi)

                cv2.imshow("source", img_src_roi)
                cv2.imshow("target", img_dst_roi)
                cv2.waitKey()


            #img_src_roi = (img_src_roi - 127.5)*0.007843137
            img_src_roi = np.swapaxes(img_src, 0, 2)
            img_src_roi = np.swapaxes(img_src, 1, 2)

            #img_dst_roi = (img_dst_roi - 127.5)*0.007843137
            img_dst_roi = np.swapaxes(img_dst, 0, 2)
            img_dst_roi = np.swapaxes(img_dst, 1, 2)

            self.data_src_list.append(img_src_roi)
            self.data_dst_list.append(img_dst_roi)

            del img_src_roi
            del img_dst_roi
            gc.collect()
        self.cur = 0
        print "\n",str(len(self.data_src_list))," img have been read into Memory..."

    def load_next_image(self): 
        if self.cur == len(self.data_src_list):
            self.cur = 0
            random.shuffle(self.data_src_list)
        #cur_data = self.label_list[self.label_cur]  # Get the image index and label
        data_src = self.data_src_list[self.cur]
        data_dst = self.data_dst_list[self.cur]

        self.cur += 1
        return data_src, data_dst
