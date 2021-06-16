#/usr/bin/python3
#-*- encoding=utf-8 -*-
import codecs 
import random
import os
import pygame
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

start,end = (0x4E00, 0x9FA5)  #汉字编码的范围
'''
with codecs.open("chinese.txt", "wb", encoding="utf-8") as f:
    for codepoint in range(int(start),int(end)):
        x = random.randint(1, 5)
        if x == 1:
            f.write(unichr(codepoint))  #写出汉字
'''

chinese_dir = 'chinese' 
if not os.path.exists(chinese_dir): 
    os.mkdir(chinese_dir) 
    
pygame.init() 
start,end = (0x4E00, 0x9FA5) # 汉字编码范围 

for codepoint in range(int(start), int(end)): 
    x = random.randint(1, 7)
    if x == 1:
        word = unichr(codepoint) 

        #font = pygame.font.Font("simkai.ttf", size) 
        # 当前目录下要有微软雅黑的字体文件msyh.ttc,或者去c:\Windows\Fonts目录下找.64是生成汉字的字体大小 
        #color = (random.randint(150, 255), random.randint(0, 90), random.randint(0, 90))
        #rtext = font.render(word, True, color) 


        frame = cv2.imread('/data_1/noise2noise-master/chejian_img/0202_test/5N1CL0MN9FC560760_1_1.jpg') 
        frame_cv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        frame_pil = Image.fromarray(frame_cv2) #转为PIL的图片格式 

        draw = ImageDraw.Draw(frame_pil) 
        size = random.randint(24, 64)


        rand_font = random.randint(0, 5)
        list_font = ["msyh.ttf", "msyhbd.ttf", "simfang.ttf", "simhei.ttf", "simkai.ttf", "simsun.ttc"]
        font="font/" + list_font[rand_font]
        font = ImageFont.truetype(font, size, encoding="utf-8") 
        # 第一个参数为字体，中文黑体 # 第二个为字体大小 

        #str = word
        str='检验专用章'
        if not isinstance(str, unicode): 
            str = str.decode('utf8') 

        color = (random.randint(150, 255), random.randint(0, 90), random.randint(0, 90))
        draw.text((100,20), str, color, font=font)

        frame_cv2 = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR) 
        cv2.imshow("img", frame_cv2) 
        cv2.waitKey(0)

        #pygame.image.save(rtext, os.path.join(chinese_dir, word + ".png"))
