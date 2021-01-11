import numpy as np 
import pandas as pd
import os
import string
import cv2
from unidecode import unidecode
import dill
from IPython.display import Image as IPythonImage, display
from tqdm import tqdm
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
from multiprocessing import Pool
import collections
from nltk.metrics import edit_distance
import glob
from scipy import ndimage

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Look at the labels
INPATH = 'Datasets/Appen'
train_df = pd.read_csv(f"{INPATH}/written_name_train_v2.csv")
train_df.head()
# Display a few randomly chosen images along with their labels
PATH = f"{INPATH}/train_v2/train"

def process_image(fname):
    file_path = f'Datasets/Appen/train_v2/train/{fname}'
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return text

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def remove_space(img, threshold = 40, kernel_size = 2, keepprop = 6, out_height = 54, buffer = 2):
    # Filter horizontal
    imgfilt = ndimage.minimum_filter(img.max(2), size=kernel_size)
    # Image.fromarray(imgfilt)
    rows = np.max(imgfilt, 0) > threshold
    from_,to_ = np.where(rows)[0][[0, -1]]
    from_ = max(0, from_ - buffer)
    to_ = min(img.shape[1], to_ + buffer)
    keeprand = np.random.choice(range(from_, to_), (to_-from_)//keepprop, replace=False)
    rows[keeprand] = True
    imgout = img[:,rows]
    # Rescale to imgin height
    scale = out_height/imgout.shape[0] 
    imgout = cv2.resize(imgout, (int(round(scale*imgout.shape[1])), out_height))
    return imgout

pool = Pool(8)
myls = train_df.FILENAME.tolist()

N = 100
pbar = tqdm(chunks(myls, N), total =len(myls)//N)
outls = []
for chunk in pbar:
    outls += pool.map(process_image,  chunk)
    pbar.set_description("Processed %s" % len(outls))
train_df['ocr'] = outls

train_df.to_feather(f"{INPATH}/written_name_train_proc.feather")
train_df = pd.read_feather(f"{INPATH}/written_name_train_proc.feather")
train_df['ocr'] = train_df['ocr'].str.replace(r"\n", "")
train_df['ocr'] = train_df['ocr'].str.replace(r"\x0c", "")

train_df['hammdist'] = train_df.fillna('').apply(lambda x: edit_distance(x["IDENTITY"], x["ocr"]), axis=1)
train_df = train_df.query('hammdist < 2').reset_index(drop = True)


'''
Visualise the strings
'''

imgs = [cv2.imread(f"{PATH}/{fname}") for fname in train_df.FILENAME.tolist()[250:1000]]
bigimg = np.zeros((1000,400, 3), dtype = np.uint8) 
bigimg[:,:,2] = 255
vctr = 0
for t,img in enumerate(imgs):
    img = 255 - remove_space(255-img, keepprop = 1, threshold = 30, buffer = 4)
    h,w,c = img.shape
    if (vctr + h) > bigimg.shape[0] :
        break
    bigimg[vctr:h+vctr, :w] = img
    vctr += h
Image.fromarray(bigimg)


ACDIR = f"{INPATH}/appen_allcaps"
os.mkdir(ACDIR)

for t,fname in tqdm(enumerate(validation_df.FILENAME)):
    try:
        img = cv2.imread(f"{PATH}/{fname}")
        img = 255 - remove_space(255-img.copy(), keepprop = 1, threshold = 10)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{ACDIR}/{fname}", img )
    except:
        continue

vfiles = [f.split('/')[-1] for f in glob.glob(f"{ACDIR}/*")]
validation_df[ validation_df.FILENAME.isin( vfiles)][['FILENAME', 'IDENTITY']].to_csv(f"{INPATH}/appen_allcaps.csv", index = False)




