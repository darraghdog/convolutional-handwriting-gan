from PIL import Image
import torch
import numpy as np
from options.test_options import TestOptions
from models.BigGAN_networks import Generator
import data.alphabets as alphabets
from util.util import prepare_z_y


def load_model():
    opt = TestOptions().parse()  # get test options
    opt.n_classes = 80
    gen = Generator(**vars(opt))

    state_dict = torch.load('./checkpoints/140_net_G.pth')
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    gen.load_state_dict(state_dict)
    gen.eval()
    return gen

model = load_model()

char_to_int = dict((s,t) for t,s in enumerate(alphabets.alphabetEnglish))

def get_word(word):
    encoded = [char_to_int[char] for char in word]
    words = torch.zeros((1, len(encoded), 80), dtype=torch.int32)
    for i, code in enumerate(encoded):
        words[0, i, code] = 1
    return words

def generate_image(word, seed = None):
    if seed is None:
        seed = np.random.randint(0, 10e4)
    words = get_word(word)
    z, _ = prepare_z_y(1, 128, 80, device='cpu', seed=seed)
    res = model.forward(z=z, y=words)
    res = res.detach().numpy()[0, 0] * 255
    im = np.array(Image.fromarray(res).convert('RGB'))
    return im

def fill_space(w):
    return np.ones((32, w, 3), dtype = np.uint8)*255

# SHIBUYA SOLASTA 14F, 1-21-1 Dogenzaka, Shibuya, Tokyo, 150-0043 Japan
img = generate_image('Here we go, Tatsuya')
Image.fromarray(img)

SEED = 10001
addr = 'SHIBUYA SOLASTA 14F, 1-21-1 Dogenzaka, Shibuya, Tokyo, 150-0043 Japan'
out = [[np.concatenate((generate_image(ii, seed = SEED), fill_space(10)), 1) for ii in i.split()] for i in addr.split(', ')]
out = [np.concatenate(w, 1) for w in out   ]
max_w = max([i.shape[1] for i in out])
out = [ np.concatenate((i, fill_space( max_w - i.shape[1])),1) for i in out]
Image.fromarray(np.concatenate(out, 0))