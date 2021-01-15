# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import os
import platform
import sys
PATH = '/Users/dhanley/Documents/scrgan/' \
    if platform.system() == 'Darwin' else '/mount/scrgan'
os.chdir(PATH)
sys.path.append(PATH)
import lmdb
import cv2
import numpy as np
import argparse
import shutil
import sys
from PIL import Image
import glob
import random
import io
import xmltodict
import html
from sklearn.decomposition import PCA
import math
from tqdm import tqdm
from itertools import compress
import pandas as pd
from collections import Counter
from scipy import ndimage
from collections import defaultdict
import keras

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def replace_specials(s):
    substitutes = {"‘": "'", 
                   "`": "'", 
                   "…": "...", 
                   "=":":", 
                   "”":'"', 
                   "“":'"', 
                   "~":"-", 
                   "–":"-", 
                   "’":"'", 
                   "[":"(", 
                   "]":")" }
    for k, v in substitutes.items():
        s = s.replace(k, v)
    return s

def remove_space(img, threshold = 40, kernel_size = 3, keepprop = 6, vborder = 3, resize = False):
    # Filter horizontal
    imgfilt = ndimage.minimum_filter(img[6:-6].max(2), size=kernel_size)
    rows = np.max(imgfilt, 0) > threshold
    from_,to_ = np.where(rows)[0][[0, -1]]
    keeprand = np.random.choice(range(from_, to_), (to_-from_)//keepprop, replace=False)
    rows[keeprand] = True
    imgout = img[:,rows]
    # Filter vertical
    h,w,_ = imgout.shape
    imgfilt = ndimage.minimum_filter(imgout[:, vborder:-vborder].max(2), size=kernel_size)
    cols = np.where(np.max(imgfilt, 1) > threshold) [0]
    from_ = 0 if cols[0] < 2 else  cols[0]+vborder
    to_ = h if cols[-2]==h-2*vborder-1 else cols[-1]
    imgout = imgout[from_:to_]
    # rows = np.where(np.max(imgfilt, 0) > threshold)[0]
    if resize:
        # Rescale to imgin height
        scale = h/imgout.shape[0] 
        imgout = cv2.resize(imgout, (int(round(scale*imgout.shape[1])), h))
    return imgout

def remove_spacev1(img, threshold = 40, kernel_size = 2, keepprop = 6, out_height = 54):
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

def remove_border(img, border = 6):
    imgout = np.zeros((54, img.shape[1] - border * 2, 3), dtype=np.uint8)
    imgcliph = min(img.shape[0] - border * 2, 54)
    imgout[:imgcliph] = img[border:-border,border:-border]
    return imgout

def filter_mnist(img):
    rfrom, rto = np.where(img.min(0)<255)[0][[0,-1]]
    cfrom, cto = np.where(img.min(1)<255)[0][[0,-1]]
    h,w = img.shape[:2]
    rfrom, cfrom = max(rfrom-1,0), max(cfrom-1,0)
    rto, cto = max(rto+1,w), min(cto+1,h)
    return img[cfrom:cto, rfrom:rto]

#def remove_border(img, border = 6):
#    return img[border:-border,border:-border]

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k, v)


def find_rot_angle(idx_letters):
    idx_letters = np.array(idx_letters).transpose()
    pca = PCA(n_components=2)
    pca.fit(idx_letters)
    comp = pca.components_
    angle = math.atan(comp[0][0]/comp[0][1])
    return math.degrees(angle)

def read_data_from_folder(folder_path):
    image_path_list = []
    label_list = []
    pics = os.listdir(folder_path)
    pics.sort(key=lambda i: len(i))
    for pic in pics:
        image_path_list.append(folder_path + '/' + pic)
        label_list.append(pic.split('_')[0])
    return image_path_list, label_list


def read_data_from_file(file_path):
    image_path_list = []
    label_list = []
    f = open(file_path)
    while True:
        line1 = f.readline()
        line2 = f.readline()
        if not line1 or not line2:
            break
        line1 = line1.replace('\r', '').replace('\n', '')
        line2 = line2.replace('\r', '').replace('\n', '')
        image_path_list.append(line1)
        label_list.append(line2)

    return image_path_list, label_list


def show_demo(demo_number, image_path_list, label_list):
    print('\nShow some demo to prevent creating wrong lmdb data')
    print('The first line is the path to image and the second line is the image label')
    for i in range(demo_number):
        print('image: %s\nlabel: %s\n' % (image_path_list[i], label_list[i]))

def create_img_label_list(top_dir,dataset, mode, words, author_number, remove_punc):
    root_dir = os.path.join(top_dir, dataset)
    output_dir = root_dir + (dataset=='IAM')*('/words'*words + '/lines'*(not words)) 
    image_path_list, label_list = [], []
    author_id = 'None'
    if dataset=='CVL':
        root_dir = os.path.join(root_dir, 'cvl-database-1-1')
        if words:
            images_name = 'words'
        else:
            images_name = 'lines'
        if mode == 'tr' or mode == 'val':
            mode_dir = ['trainset']
        elif mode == 'te':
            mode_dir = ['testset']
        elif mode == 'all':
            mode_dir = ['testset', 'trainset']
        idx = 1
        for mod in mode_dir:
            images_dir = os.path.join(root_dir, mod, images_name)
            for path, subdirs, files in os.walk(images_dir):
                for name in files:
                    if (mode == 'tr' and idx >= 10000) or (
                            mode == 'val' and idx < 10000) or mode == 'te' or mode == 'all' or mode == 'tr_3te':
                        if os.path.splitext(name)[0].split('-')[1] == '6':
                            continue
                        label = os.path.splitext(name)[0].split('-')[-1]
                        if 'ä' in label or 'ü' in label or label=='':
                            continue
                        imagePath = os.path.join(path, name)
                        label_list.append(label)
                        image_path_list.append(imagePath)
                    idx += 1

    elif dataset=='IAM':
        labels_name = 'original'
        if mode=='all':
            mode = ['te', 'va1', 'va2', 'tr']
        elif mode=='valtest':
            mode=['te', 'va1', 'va2']
        else:
            mode = [mode]
        if words:
            images_name = 'wordImages'
        else:
            images_name = 'lineImages'
        images_dir = os.path.join(root_dir, images_name)
        labels_dir = os.path.join(root_dir, labels_name)
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

        if author_number >= 0:
            full_ann_files = [full_ann_files[author_number]]
            im_dirs = [im_dirs[author_number]]
            author_id = im_dirs[0].split('/')[-1]

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
                    
    elif dataset=='dread':
        anno_file = f'{root_dir}/cck25k.csv'
        adf = pd.read_csv(anno_file, names = ['id', 'numberfull', 'number'], dtype= str)
        image_path_list, label_list = [], []
        for t,row in adf.dropna().iterrows():
            image_path = f'{root_dir}/cck25k/{row.id}_1.png'
            label = row.numberfull
            image_path_list.append(image_path)
            label_list.append(label)
            
    elif dataset=='dreadclean':
        root_dir = root_dir.replace('dreadclean', 'dread')
        anno_file = f'{root_dir}/cck25k.csv'
        adf = pd.read_csv(anno_file, names = ['id', 'numberfull', 'number'], dtype=str)
        imgnms = [i.split('/')[-1][:-6] for i in glob.glob(f'{root_dir}/cck25kclean/*')]
        adf = adf[adf.id.isin(imgnms)]
        image_path_list, label_list = [], []
        for t,row in adf.dropna().iterrows():
            image_path = f'{root_dir}/cck25kclean/{row.id}_1.png'
            label = row.numberfull
            image_path_list.append(image_path)
            label_list.append(label)
            
    elif dataset=='dreadcleannodash':
        root_dir = root_dir.replace('dreadcleannodash', 'dread')
        anno_file = f'{root_dir}/cck25kcleannodash.csv'
        adf = pd.read_csv(anno_file, dtype=str)
        imgnms = [i.split('/')[-1][:-6] for i in glob.glob(f'{root_dir}/cck25kcleannodash/*')]
        adf = adf[adf.id.isin(imgnms)]
        image_path_list, label_list = [], []
        for t,row in adf.dropna().iterrows():
            image_path = f'{root_dir}/cck25kcleannodash/{row.id}_2.png'
            label = row.number
            image_path_list.append(image_path)
            label_list.append(label)
    
    elif dataset=='Appen':
        anno_file = f'{root_dir}/appen_allcaps.csv'
        adf = pd.read_csv(anno_file, dtype=str)
        adf['FILENAME'] = adf['FILENAME'].apply(lambda x: f'{root_dir}/appen_allcaps/{x}')
        for t,row in adf.dropna().iterrows():
            image_path_list.append(row.FILENAME)
            label_list.append(row.IDENTITY)
            
    elif dataset=='finetune':
        ftfiles = glob.glob(f'{root_dir}/batch*.csv')
        ftdf = pd.concat([pd.read_csv(f, error_bad_lines=False, \
                        names = ['fname', 'label'])\
                          .assign(batch = f.split('/')[-1][:6]) \
                        for t,f in enumerate(ftfiles)], 0)

        for t, row in ftdf.iterrows():
            fname = f'{root_dir}/finetune/{row.batch}/{row.fname}'
            image_path_list.append(fname)
            label_list.append(replace_specials(row.label))
        

    return image_path_list, label_list, output_dir, author_id

def createDataset(image_path_list, label_list, outputPath, mode, author_id, remove_punc, \
                  resize, imgH, init_gap, h_gap, charminW, charmaxW, discard_wide, discard_narr, labeled):
    assert (len(image_path_list) == len(label_list))
    nSamples = len(image_path_list)

    outputPath = outputPath + (resize=='charResize') * ('/h%schar%sto%s/'%(imgH, charminW, charmaxW)) + (resize=='keepRatio') * ('/h%s/'%(imgH)) \
                 + (resize=='noResize') * ('/noResize/') + (author_id!='None') * ('single_authors/'+author_id+'/' ) \
                 + mode + (resize!='noResize') * (('_initGap%s'%(init_gap)) * (init_gap>0) + ('_hGap%s'%(h_gap)) * (h_gap>0) \
                 + '_NoDiscard_wide' * (not discard_wide) + '_NoDiscard_wide' * (not discard_narr))+'_unlabeld' * (not labeled) +\
                 (('IAM' in outputPath) and remove_punc) *'_removePunc'
    print(outputPath)
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
        os.makedirs(outputPath)
    else:
        os.makedirs(outputPath)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    labelctr = defaultdict(int)
    for i in tqdm(range(nSamples)):

        imagePath = image_path_list[i]
        label = label_list[i]
        #if 'finetune' in imagePath: break
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        try:
            im = Image.open(imagePath)
        except:
            continue
        
        if 'dread' in imagePath:
            immat = np.array(im.convert('RGB'))
            if 'clean' not in imagePath:
                immat = remove_border(immat)
            try:
                if 'cleannodash' not in imagePath:
                    immat = remove_spacev1(immat, threshold = 40, keepprop = 1+ random.randrange(20))
                immat = 255 - immat
                immat = cv2.cvtColor(immat, cv2.COLOR_BGR2GRAY)
            except:
                continue
            
            if not ((immat.shape[1]>150) &  (immat.shape[1]<500)):
                print('%s has a width larger outside the 150 to 500 threshold for numbers'% imagePath)
                continue
            im = Image.fromarray(immat)

        if resize in ['charResize', 'keepRatio']:
            width, height = im.size
            new_height = imgH - (h_gap * 2)
            len_word = len(label)
            width = int(width * imgH / height)
            new_width = width
            if (resize=='charResize') & ('dread' not in imagePath) & ('Appen' not in imagePath):
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
        
        imgByteArr = io.BytesIO()
        im.save(imgByteArr, format='tiff')
        wordBin = imgByteArr.getvalue()
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = wordBin
        if labeled:
            cache[labelKey] = label
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
        
        splpath = imagePath.split('/')
        labelctr[splpath[1] + ('clean' in splpath[2])* (splpath[2]) ] += 1
        
    if mnistsamp>0:
        (x_train, y_train), _ = keras.datasets.mnist.load_data()
        ntotal = len(x_train)
        x_train = 255-x_train
        y_train = y_train.astype(str)
        mnist_images, mnist_labels = [], []
        for i in range(mnistsamp):
            samp = np.random.choice(ntotal, 1+np.random.choice(6))
            img = np.concatenate(list(x_train[samp.tolist()]) ,1  )
            img = Image.fromarray(img).resize((32*len(samp), 32))
            label = ''.join(y_train[samp.tolist()].tolist())
            mnist_images.append(img)
            mnist_labels.append(label)
        print(f'Add {len(mnist_images)} mnist images.')
        for im, label in zip(mnist_images, mnist_labels):
            imgByteArr = io.BytesIO()
            im.save(imgByteArr, format='tiff')
            wordBin = imgByteArr.getvalue()
            imageKey = 'image-%09d' % cnt
            labelKey = 'label-%09d' % cnt
            cache[imageKey] = wordBin
            if labeled:
                cache[labelKey] = label
            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, nSamples))
            cnt += 1
            labelctr['mnist'] += 1
        
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)
    print(f'Processed count of each class : {[f"{k}: {v}" for k,v in labelctr.items()]}')

def createDict(label_list, top_dir, dataset, mode, words, remove_punc):
    lex_name = dataset+'_' + mode + (dataset in ['IAM','RIMES'])*('_words' * words) + (dataset=='IAM') * ('_removePunc' * remove_punc)
    all_words = '|||'.join(label_list).split('|||')
    unique_words = []
    words = []
    for x in tqdm(all_words):
        if x!='' and x!=' ':
            words.append(x)
            if x not in unique_words:
                unique_words.append(x)
    print(len(words))
    print(len(unique_words))
    with open(os.path.join(top_dir, 'Lexicon', lex_name+'_stratified.txt'), "w") as file:
        file.write("\n".join(unique_words))
    file.close()
    with open(os.path.join(top_dir, 'Lexicon', lex_name + '_NOTstratified.txt'), "w") as file:
        file.write("\n".join(words))
    file.close()


def printAlphabet(label_list):
    # get all unique alphabets - ignoring alphabet longer than one char
    all_chars = ''.join(label_list)
    unique_chars = []
    for x in all_chars:
        if x not in unique_chars and len(x) == 1:
            unique_chars.append(x)

    # for unique_char in unique_chars:
    print(''.join(unique_chars))

if __name__ == '__main__':
    create_Dict = True # create a dictionary of the generated dataset
    iamdataset = 'IAM'     #CVL/IAM/RIMES/gw
    mode = 'tr'        # tr/te/val/va1/va2/all
    labeled = True
    top_dir = 'Datasets'
    # parameter relevant for IAM/RIMES:
    words = False        # use words images, otherwise use lines
    #parameters relevant for IAM:
    author_number = -1  # use only images of a specific writer. If the value is -1, use all writers, otherwise use the index of this specific writer
    remove_punc = True  # remove images which include only one punctuation mark from the list ['.', '', ',', '"', "'", '(', ')', ':', ';', '!']

    resize = 'charResize'  # charResize|keepRatio|noResize - type of resize,
                        # char - resize so that each character's width will be in a specific range (inside this range the width will be chosen randomly),
                        # keepRatio - resize to a specific image height while keeping the height-width aspect-ratio the same.
                        # noResize - do not resize the image
    imgH = 32           # height of the resized image
    init_gap = 0        # insert a gap before the beginning of the text with this number of pixels
    charmaxW = 17       # The maximum character width
    charminW = 16       # The minimum character width
    h_gap = 0           # Insert a gap below and above the text
    discard_wide = True # Discard images which have a character width 3 times larger than the maximum allowed character size (instead of resizing them) - this helps discard outlier images
    discard_narr = True # Discard images which have a character width 3 times smaller than the minimum allowed charcter size.
    mnistsamp = 0   # Number of mnist samples to add
    appensamp = 3000
    numsamp = 5000
    '''
    image_path_list, label_list, outputPath, author_id = \
                create_img_label_list(top_dir,iamdataset, mode, words, author_number, remove_punc)
    image_path_list_dread, label_list_dread, _       , _         = \
                create_img_label_list(top_dir,'dreadclean', mode, words, author_number, remove_punc)
    image_path_list_dreadnum, label_list_dreadnum, _       , _         = \
                create_img_label_list(top_dir,'dreadcleannodash', mode, words, author_number, remove_punc)
    image_path_list_appen, label_list_appen, _       , _         = \
                create_img_label_list(top_dir,'Appen', mode, words, author_number, remove_punc)
    image_path_list_finetune, label_list_finetune, _       , _         = \
                create_img_label_list(top_dir,'finetune', mode, words, author_number, remove_punc)
    image_path_list_appen, label_list_appen = image_path_list_appen[:appensamp], label_list_appen[:appensamp]
    image_path_list_dread, label_list_dread = image_path_list_dread[:numsamp], label_list_dread[:numsamp]
    image_path_list_dreadnum, label_list_dreadnum = image_path_list_dreadnum[:numsamp], label_list_dreadnum[:numsamp]
                
    image_path_list += image_path_list_dread + image_path_list_dreadnum + image_path_list_appen + image_path_list_finetune#
    label_list += label_list_dread + label_list_dreadnum + label_list_appen  + label_list_finetune#
    '''
    image_path_list, label_list, _       , _         = \
                create_img_label_list(top_dir,'finetune', mode, words, author_number, remove_punc)
    
    ctrdf = pd.DataFrame(Counter(list(''.join(label_list))).most_common(), columns = ['token', 'count'])
    ctrdf = ctrdf.sort_values('token').reset_index(drop= True)
    print(ctrdf)
    # in a previous version we also cut the white edges of the image to keep a tight rectangle around the word but it
    # seems in all the datasets we use this is already the case so I removed it. If there are problems maybe we should add this back.
    outputPath = 'lines_finetuneonly'
    createDataset(image_path_list, label_list, outputPath, mode, author_id, remove_punc, resize, imgH, init_gap, h_gap, charminW, charmaxW, discard_wide, discard_narr, labeled)
    if create_Dict:
        createDict(label_list, top_dir, iamdataset, mode, words, remove_punc)
    printAlphabet(label_list)
    
    # Add to english words
    lexpath = os.path.join(top_dir, 'Lexicon')
    pd.DataFrame({'words': label_list})\
            .astype(str).to_csv(f'{lexpath}/english_lines_ftuneonly.txt', index = False, sep = '|')
    
    
    