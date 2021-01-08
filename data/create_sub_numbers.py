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
import hashlib

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

def split_on_dash(img, label, threshold = 40, kernel_size = 2):
    # Filter horizontal
    imgfilt = ndimage.minimum_filter(img[6:-6].max(2), size=kernel_size)
    # Image.fromarray(imgfilt)
    rows = np.where(np.max(imgfilt, 0) > threshold)[0]
    diff = rows[1:] - rows[:-1]
    
    try:
        nbigcuts = len(np.where(diff>=15)[0])
        minbigcut = min(diff[diff>=15]) 
        nmdlcuts = len(np.where((diff<minbigcut) & (diff>(minbigcut-12)))[0])
        if not ((nbigcuts==4) & (nmdlcuts==0)):
            return None
    except:
        return None
    cuts = ((rows[np.where(diff>15)[0] + 1] + rows[np.where(diff>15)[0]])/ 2).astype(np.int16)
    if (cuts[0] < 80) or (img.shape[1] - cuts[-1] < 80) : 
        return None
    imgl = [img[:, :cuts[0]], img[:,cuts[1]:cuts[2]], img[:,cuts[3]:]]
    labl = label.split('-')
    if random.choice([True, False]):
        for p1,p2 in [(1,0), (2,0), (2,1)]: 
            imgl.append(remove_space(np.concatenate((imgl[p1], imgl[p2]), 1), keepprop = random.randrange(3, 20) ))
            labl.append(labl[p1]+ labl[p2])
        outls = [(i,l) for i,l in zip(imgl[3:], labl[3:])]
    else:
        outls = [(remove_space(i, keepprop = random.randrange(3, 20) ),l) for i,l in zip(imgl, labl)]

    return outls

def hashname(v):
    hash_object = hashlib.sha1(bytes(str(v), encoding='utf-8'))
    hex_dig = hash_object.hexdigest()[:24]
    return hex_dig

INPATH='/Users/dhanley/Documents/scrgan/Datasets/dread'
OUTPATH = f'{INPATH}/cck25kcleannodash/'
anno_file = f'{INPATH}/cck25k.csv'
adf = pd.read_csv(anno_file, names = ['id', 'numberfull', 'number'])
imgnms = [i.split('/')[-1][:-6] for i in glob.glob(f'{INPATH}/cck25kclean/*')]
adf = adf[adf.id.isin(imgnms)]
adf = adf.dropna()


imgls = [(cv2.imread(f'{INPATH}/cck25kclean/{row.id}_1.png'), row.numberfull) for t,row in adf.iterrows()]
imgls = [split_on_dash(*i) for i in imgls ]
imgls = [i for i in imgls if i is not None]
imgls = [i for l in imgls for i in l] # Flatten the list

if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)

adfnodash = []
for t, (img, label) in enumerate(imgls):
    id_ = hashname(t)
    fname = f'{OUTPATH}/{id_}_2.png'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(fname, img)
    adfnodash.append([id_, label])
adfout = pd.DataFrame(adfnodash, columns = ['id', 'number'])
adfout.to_csv(f'{INPATH}/cck25kcleannodash.csv', index = False)
