import argparse
import numpy as np
import math
from pathlib import Path
import os
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from fonts import Fonts


DATA_DIR = 'data'
Path(DATA_DIR).mkdir(exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=62, help="number of classes for dataset")
parser.add_argument("--n_domains", type=int, default=100, help="number of fonts for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--print_interval", type=int, default=400, help="interval between printing")
parser.add_argument("--save_dir", type=str, default='baseline', help="interval between printing")
parser.add_argument("--eval_only", action='store_true', help="only eval")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay param")
parser.add_argument("--num_examples", type=int, default=2500, help="num labeled examples")
opt = parser.parse_args()
print(opt)

IMAGE_DIR = 'images'
if opt.weight_decay > 0:
    IMAGE_DIR = f'images_decay{opt.weight_decay}'
if opt.num_examples != 2500:
    IMAGE_DIR += f'_numexamples_{opt.num_examples}'
os.makedirs(IMAGE_DIR, exist_ok=True)
model_root = Path('models')
model_root.mkdir(exist_ok=True)
suffix = ''
if opt.weight_decay != 0:
    suffix += f'decay_{opt.weight_decay}'

if opt.num_examples != 2500:
    suffix += f'numexamples_{opt.num_examples}'

save_dir = model_root / (opt.save_dir + suffix)
save_dir.mkdir(exist_ok=True)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        self.domain_emb = nn.Embedding(opt.n_domains, opt.latent_dim)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 256),
            *block(256, 512),
            *block(512, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, domain, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), self.domain_emb(domain)), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


# Loss functions
mse_loss = torch.nn.MSELoss()
eval_mse_loss = torch.nn.MSELoss(reduction='sum')

# Initialize generator and discriminator
generator = Generator()

if cuda:
    generator.cuda()
    mse_loss.cuda()

# Configure data loader
dataloader = torch.utils.data.DataLoader(
    Fonts(
        DATA_DIR,
        split='train',
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
        num_examples=opt.num_examples,
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

val_dataset = Fonts(
        DATA_DIR,
        split='val',
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
        num_examples=opt.num_examples,
    )
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

test_dataset = Fonts(
        DATA_DIR,
        split='test',
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
        num_examples=opt.num_examples,
    )
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done, split='train', pi=False):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    # pick first couple of fonts
    fonts = Variable(LongTensor(np.asarray([i for i in range(n_row) for _ in range(n_row)])))
    gen_imgs = generator(fonts, labels)
    if pi:
        gen_imgs = pi_model(gen_imgs)
        save_image(gen_imgs.data, f"{IMAGE_DIR}/{split}_{batches_done}_pi.png", nrow=n_row, normalize=True)
    else:
        save_image(gen_imgs.data, f"{IMAGE_DIR}/{split}_{batches_done}.png", nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

if not opt.eval_only:
    best_val_loss = None
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels, domains) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))
            domains = Variable(domains.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(domains, labels)

            g_loss = mse_loss(gen_imgs, real_imgs)

            l2_loss = torch.tensor(0.)
            l2_loss = l2_loss.cuda()
            for name, param in generator.named_parameters():
                if 'bias' not in name:
                    l2_loss = l2_loss + torch.sum(torch.pow(param, 2))
            g_loss += opt.weight_decay * l2_loss

            g_loss.backward()
            optimizer_G.step()

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done, split='train')
            if batches_done % opt.print_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader),g_loss.item())
                )
        val_loss = 0
        for i, (imgs, labels, domains) in enumerate(val_dataloader):
            batch_size = imgs.shape[0]

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))
            domains = Variable(domains.type(LongTensor))

            # Generate a batch of images
            gen_imgs = generator(domains, labels)
            val_loss += eval_mse_loss(gen_imgs, real_imgs).detach().cpu().numpy()
            if i % opt.sample_interval == 0:
                sample_image(n_row=10, batches_done=i, split='test')
        val_loss = val_loss / len(val_dataset) / opt.img_size / opt.img_size
        print(f"VAL_LOSS: {val_loss}, BEST: {best_val_loss}")

        if best_val_loss is None or val_loss < best_val_loss:
            # save
            checkpoint_fpath = str(save_dir / 'checkpoint.pt')
            state = {'state_dict': generator.state_dict()}
            torch.save(state, checkpoint_fpath)
            best_val_loss = val_loss

checkpoint_fpath = str(save_dir / 'checkpoint.pt')
checkpoint = torch.load(checkpoint_fpath)
print(f"Loading {checkpoint_fpath}")
generator.load_state_dict(checkpoint['state_dict'])

from unet import GeneratorUNet
pi_model = GeneratorUNet(in_channels=1, out_channels=1)
if cuda:
    pi_model.cuda()
checkpoint_fpath = 'models/composed_diff_dropout0.1/pi_checkpoint.pt'
checkpoint = torch.load(checkpoint_fpath)
print(f"Loading {checkpoint_fpath}")
pi_model.load_state_dict(checkpoint['state_dict'])

pi_model.eval()
generator.eval()
loss = 0
pi_loss = 0
print('EVALUATE')
for i, (imgs, labels, domains) in tqdm(enumerate(test_dataloader)):
    batch_size = imgs.shape[0]

    # Configure input
    real_imgs = Variable(imgs.type(FloatTensor))
    labels = Variable(labels.type(LongTensor))
    domains = Variable(domains.type(LongTensor))

    # Generate a batch of images
    gen_imgs = generator(domains, labels)
    loss += eval_mse_loss(gen_imgs, real_imgs).detach().cpu().numpy()
    gen_imgs_pi = pi_model(gen_imgs)
    pi_loss += eval_mse_loss(gen_imgs_pi, real_imgs).detach().cpu().numpy()
sample_image(n_row=10, batches_done=0, split='test', pi=False)
sample_image(n_row=10, batches_done=0, split='test', pi=True)
print('TEST LOSS: ', loss / len(test_dataset) / opt.img_size / opt.img_size)
print('TEST LOSS with PI: ', pi_loss / len(test_dataset) / opt.img_size / opt.img_size)
