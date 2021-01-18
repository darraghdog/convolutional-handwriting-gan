import os
import platform
import sys
PATH = '/Users/dhanley/Documents/scrgan/' \
    if platform.system() == 'Darwin' else '/mount/scrgan'
os.chdir(PATH)
sys.path.append(PATH)
from PIL import Image
import torch
import random
import cv2
from tqdm import tqdm
import numpy as np
from options.test_options import TestOptions
from models.BigGAN_networks import Generator
import data.alphabets as alphabets
from util.util import prepare_z_y, Distribution
import random
import albumentations as A
from datasets import list_datasets, load_dataset, list_metrics, load_metric
import re, string
import pandas as pd

def get_word(word):
    encoded = [char_to_int[char] for char in word]
    words = torch.zeros((1, len(encoded), 80), dtype=torch.int32)
    for i, code in enumerate(encoded):
        words[0, i, code] = 1
    return words

def generate_image(word = 'meet', seed = None, device = device):
    if seed is None:
        seed = np.random.randint(0, 10e4)
    words = get_word(word).to(device)
    z, _ = prepare_z_y(1, 128, 80, device=device, seed=seed)
    with torch.no_grad():
        res = model.forward(z=z, y=words)
    res = res.detach().numpy()[0, 0] * 255
    im = np.array(Image.fromarray(res).convert('RGB'))
    return im

def fill_space(w):
    return np.ones((32, w, 3), dtype = np.uint8)*255

def load_model():
    opt = TestOptions().parse()  # get test options
    opt.n_classes = 80
    gen = Generator(**vars(opt))
    load_dir = 'checkpoints/demov17_IAMlinesftunecharH32W16rmPunct_lex_english_lines_CapitalizeLex_GANres16_bs32/'
    state_dict = torch.load(f'{load_dir}/15_net_G.pth')
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    gen.load_state_dict(state_dict)
    gen.eval()
    return gen

def corpus_to_samples(corpus, minlen = 36):
    yset = []
    l = []
    for i in corpus.split():
        l.append(i)
        if len(' '.join(l)) > minlen:
            yset.append( ' '.join(l) )
            l = []
    return yset

random.seed(0)
NSAMP = 20
OUTDIR = 'Datasets/generative'
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
if not os.path.exists(f'{OUTDIR}/images'):
    os.makedirs(f'{OUTDIR}/images')
    
# Load models
model = load_model().to(device)
model.eval()
char_to_int = dict((s,t) for t,s in enumerate(alphabets.alphabetEnglish))

# Generate corpus

pattern = re.compile('[\W_]+')
squad_dataset = load_dataset('squad')
# generate corpus
corpus = [re.sub(r'[^A-Za-z]+', '',  squad_dataset['train'][i]['context']) for i in range(10000)]
corpus = ' '.join(list(set(corpus)))
#corpus = ' '.join(random.sample(corpus.split(), len(corpus.split())))

# Get letter distirbution
vc = pd.Series(list(corpus.replace(' ', '').lower())).value_counts()
vcchars = vc.index
vcwts = vc.values / vc.sum()
# Numbers
nums = np.arange(10).tolist() + ['-']
def gennums(f,t):
    return ''.join(np.random.choice(nums*2,size=random.randint(f,t),replace=False).astype(str).tolist())


yset = []
for ii in range(NSAMP):
    ls = []
    for i in range(10): ls.append(''.join( np.random.choice(vcchars,size=random.randint(2,8),replace=False, p=vcwts).tolist()  ))
    yset.append(' '.join(ls)[:40])
    
hashids = [("%32x" % random.getrandbits(128))[-16:] for i in range(NSAMP)]
ysetlower = pd.DataFrame({'image': [f'images/{i}_1.jpg' for i in hashids], 
                          'label' : [i.lower() for i in yset]})
ysetupper = pd.DataFrame({'image': [f'images/{i}_2.jpg' for i in hashids], 
                          'label' : [i.upper() for i in yset]})
ysetcamel = pd.DataFrame({'image': [f'images/{i}_3.jpg' for i in hashids], 
                          'label' : [i.title() for i in yset]})
ysetnums = pd.DataFrame({'image': [f'images/{i}_4.jpg' for i in hashids], 
                          'label' : [gennums(4,12) for i in yset]})
ysetlower.to_csv(f'{OUTDIR}/labels_lower.csv', index = False)
ysetupper.to_csv(f'{OUTDIR}/labels_upper.csv', index = False)
ysetcamel.to_csv(f'{OUTDIR}/labels_camel.csv', index = False)
ysetnums.to_csv(f'{OUTDIR}/labels_nums.csv', index = False)

for ysetdf in [ysetlower, ysetupper, ysetcamel, ysetnums]:
    for t, row in tqdm(ysetdf.iterrows()):
        img = generate_image('i', t) # from some reason we need to run it twice to pick up the new seed 
        img = generate_image(row.label, t)
        fname = f'{OUTDIR}/{row.image}'
        cv2.imwrite(fname, img)



