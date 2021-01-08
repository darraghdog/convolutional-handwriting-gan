#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:24:25 2021

@author: dhanley
"""
import os
from PIL import Image
import cv2
import glob
from scipy import ndimage
import numpy as np
import pandas as pd
import xmltodict
import html
from tqdm import tqdm
import torch, torchvision
import keras
import random

def remove_space(img, threshold = 40, kernel_size = 2, keepprop = 6, out_height = 54):
    # Filter horizontal
    imgfilt = ndimage.minimum_filter(img[6:-6].max(2), size=kernel_size)
    # Image.fromarray(imgfilt)
    rows = np.max(imgfilt, 0) > threshold
    from_,to_ = np.where(rows)[0][[0, -1]]
    keeprand = np.random.choice(range(from_, to_), (to_-from_)//keepprop, replace=False)
    rows[keeprand] = True
    imgout = img[:,rows]
    # Rescale to imgin height
    scale = out_height/imgout.shape[0] 
    imgout = cv2.resize(imgout, (int(round(scale*imgout.shape[1])), out_height))
    return imgout

'''
def remove_border(img, border = 6):
    imgout = np.zeros((54, img.shape[1] - border * 2, 3), dtype=np.uint8)
    imgcliph = min(img.shape[0] - border * 2, 54)
    imgout[:imgcliph] = img[border:-border,border:-border]
    return imgout
'''

INPATH='/Users/dhanley/Documents/scrgan/Datasets/dread'
adf = pd.read_csv(f'{INPATH}/cck25k.csv', names = ['id', 'numberfull', 'number'], dtype={'numberfull': str})
imgnms = [i.split('/')[-1][:-6] for i in glob.glob(f'{INPATH}/cck25kclean/*')]
adf = adf[adf.id.isin(imgnms)]
adf = adf.dropna()

adfnd = pd.read_csv(f'{INPATH}/cck25kcleannodash.csv', dtype={'number': str})

imgls = [cv2.imread(f'{INPATH}/cck25kclean/{f}_1.png')[:,:,::-1] for t,f in adf.id.sample(5).iteritems()]
img = imgls[0]
Image.fromarray(img)


imgls = [cv2.imread(f'{INPATH}/cck25kclean/{f}_1.png') for t,f in adf.id.sample(10).iteritems()]
grid=54
bigimg = np.ones((len(imgls) * grid, max([i.shape[1] for i in imgls]), 3), dtype=np.uint8) * 255
for t, img in enumerate(imgls):
    h,w,c = img.shape
    bigimg[(t*grid):(t*grid)+img.shape[0], :w] = 255 - img
Image.fromarray(bigimg)

imgls = [cv2.imread(f'{INPATH}/cck25kclean/{f}_1.png')[:,:,::-1] for t,f in adf.id.sample(55).iteritems()]
imgls = [remove_space(i, threshold = 40, keepprop = 1+ random.randrange(20)) for i in imgls]
# pd.Series(i.shape[1] for i in imgls).hist(bins = 100)
imgls = [255-i for i in imgls if (i.shape[1]>250) &  (i.shape[1]<500) ]
bigimg = np.ones((len(imgls) * 54, 501, 3), dtype=np.uint8) * 255
bigimg[:,:,2] = 255
for t, img in enumerate(imgls):
    h,w,c = img.shape
    bigimg[t*h:(t+1)*h, :w] = img
Image.fromarray(bigimg)


imgls = [cv2.imread(f'{INPATH}/cck25kcleannodash/{f}_2.png')[:,:,::-1] for t,f in adfnd.id.sample(50).iteritems()]
# pd.Series(i.shape[1] for i in imgls).hist(bins = 100)
bigimg = np.ones((len(imgls) * 54, max([i.shape[1] for i in imgls]), 3), dtype=np.uint8) * 255
bigimg[:,:,2] = 255
for t, img in enumerate(imgls):
    h,w,c = img.shape
    bigimg[t*h:(t+1)*h, :w] = 255 - img
Image.fromarray(bigimg)


'''
IAM
'''


root_dir = f'{INPATH}/../IAM'
labels_name = 'original'
images_name = 'wordImages'
mode = ['tr']
words = True
remove_punc = True
resize = 'charResize'

images_dir = os.path.join(root_dir, images_name)
labels_dir = os.path.join( root_dir, labels_name)
full_ann_files = []
im_dirs = []
line_ann_dirs = []
image_path_list, label_list = [], []
for mod in mode:
    part_file = os.path.join(root_dir, 'original_partition', mod + '.lst')
    with open(part_file)as fp:
        for line in fp:
            name = line.split('-')
            if int(name[-1][:-1]) == 0:
                anno_file = os.path.join(labels_dir, '-'.join(name[:2]) + '.xml')
                full_ann_files.append(anno_file)
                im_dir = os.path.join(images_dir, name[0], '-'.join(name[:2]))
                im_dirs.append(im_dir)

lables_to_skip = ['.', '', ',', '"', "'", '(', ')', ':', ';', '!']
for i, anno_file in enumerate(full_ann_files):
    with open(anno_file) as f:
        try:
            line = f.read()
            annotation_content = xmltodict.parse(line)
            lines = annotation_content['form']['handwritten-part']['line']
            if words:
                lines_list = []
                for j in range(len(lines)):
                    lines_list.extend(lines[j]['word'])
                lines = lines_list
        except:
            print('line is not decodable')
        for line in lines:
            try:
                label = html.unescape(line['@text'])
            except:
                continue
            if remove_punc and label in lables_to_skip:
                continue
            id = line['@id']
            imagePath = os.path.join(im_dirs[i], id + '.png')
            image_path_list.append(imagePath)
            label_list.append(label)
nSamples = len(image_path_list)

imgH = 32 
imgH = 32           # height of the resized image
init_gap = 0        # insert a gap before the beginning of the text with this number of pixels
charmaxW = 17       # The maximum character width
charminW = 16       # The minimum character width
h_gap = 0           # Insert a gap below and above the text
discard_wide = True # Discard images which have a character width 3 times larger than the maximum allowed character size (instead of resizing them) - this helps discard outlier images
discard_narr = True # Discard images which have a character width 3 times smaller than the minimum allowed charcter size.

import random
imgls = []
#for i in tqdm(range(nSamples)):
for i in tqdm(random.sample(range(nSamples), 50)):
    imagePath = image_path_list[i]
    label = label_list[i]
    
    if not os.path.exists(imagePath):
        print('%s does not exist' % imagePath)
        continue
    try:
        im = Image.open(imagePath)
    except:
        continue
    
    if resize in ['charResize', 'keepRatio']:
        width, height = im.size
        new_height = imgH - (h_gap * 2)
        len_word = len(label)
        width = int(width * imgH / height)
        new_width = width
        if resize=='charResize':
            if (width/len_word > (charmaxW-1)) or (width/len_word < charminW) :
                if discard_wide and width/len_word > 3*((charmaxW-1)):
                    print('%s has a width larger than max image width' % imagePath)
                    continue
                if discard_narr and (width / len_word) < (charminW/3):
                    print('%s has a width smaller than min image width' % imagePath)
                    continue
                else:
                    new_width = len_word * random.randrange(charminW, charmaxW)

        # reshape the image to the new dimensions
        im = im.resize((new_width, new_height))
        # append with 256 to add left, upper and lower white edges
        init_w = int(random.normalvariate(init_gap, init_gap / 2))
        new_im = Image.new("RGB", (new_width+init_gap, imgH), color=(256,256,256))
        new_im.paste(im, (abs(init_w), h_gap))
        im = new_im
    imgls.append(np.array(im))
        
bigimg = np.ones((32 * len(imgls), max([i.shape[1] for i in imgls]), 3), dtype=np.uint8)*255
for t, tmpimg in enumerate(imgls):
    bigimg[(t)*32:(1+t)*32,:tmpimg.shape[1]] = tmpimg
Image.fromarray(bigimg)