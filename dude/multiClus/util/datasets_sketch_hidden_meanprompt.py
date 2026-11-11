import argparse
import itertools
import math
import os
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import nn
import PIL
from PIL import Image
from torchvision import transforms

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif')


def is_image_file(filename):
    # return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    return filename.endswith(IMG_EXTENSIONS)

def load_filenames(dataset):
    basedir = '../dataset'
    if dataset == 'sketch':
        data_dir = 'data/sketch'
        imgdir = os.path.join(basedir, 'sketchsub')
        with open(os.path.join(data_dir, 'subset.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'sketchimg':
        data_dir = 'data/sketch'
        imgdir = os.path.join(basedir, 'sketchimgsub')
        with open(os.path.join(data_dir, 'sketchimgsub.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'domainnet_sketch':
        data_dir = 'data/domainnet'
        imgdir = os.path.join(basedir, 'domainnet_sub')
        with open(os.path.join(data_dir, 'sketch_sub.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'domainnet_clipart':
        data_dir = 'data/domainnet'
        imgdir = os.path.join(basedir, 'domainnet_sub')
        with open(os.path.join(data_dir, 'clipart_sub.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'domainnet_infograph':
        data_dir = 'data/domainnet'
        imgdir = os.path.join(basedir, 'domainnet_sub')
        with open(os.path.join(data_dir, 'infograph_sub.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'domainnet_real':
        data_dir = 'data/domainnet'
        imgdir = os.path.join(basedir, 'domainnet_sub')
        with open(os.path.join(data_dir, 'real_sub.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'domainnet_quickdraw':
        data_dir = 'data/domainnet'
        imgdir = os.path.join(basedir, 'domainnet_sub')
        with open(os.path.join(data_dir, 'quickdraw_sub.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'domainnet_sketchclipart':
        data_dir = 'data/domainnet'
        imgdir = os.path.join(basedir, 'domainnet_sub')
        with open(os.path.join(data_dir, 'sketchclipart_catsub.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'domainnet_sketchclipart_general':
        data_dir = 'data/domainnet'
        imgdir = 'plots_aug_noblack'
        with open(os.path.join(data_dir, 'sketchclipart_general_noblack.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'domainnet_sketchinfograph':
        data_dir = 'data/domainnet'
        imgdir = os.path.join(basedir, 'domainnet_sub')
        with open(os.path.join(data_dir, 'sketchinfograph_catsub.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'domainnet_infographreal':
        data_dir = 'data/domainnet'
        imgdir = os.path.join(basedir, 'domainnet_sub')
        with open(os.path.join(data_dir, 'infographreal_catsub.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'domainnet_paintingclipart':
        data_dir = 'data/domainnet'
        imgdir = os.path.join(basedir, 'domainnet_sub')
        with open(os.path.join(data_dir, 'paintingclipart_catsub.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'domainnet_paintingquickdraw':
        data_dir = 'data/domainnet'
        imgdir = os.path.join(basedir, 'domainnet_sub')
        with open(os.path.join(data_dir, 'paintingquickdraw_catsub.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'domainnet_quickdrawreal':
        data_dir = 'data/domainnet'
        imgdir = os.path.join(basedir, 'domainnet_sub')
        with open(os.path.join(data_dir, 'quickdrawreal_catsub.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'officehome_artreal':
        data_dir = 'data/officehome'
        imgdir = os.path.join(basedir, 'OfficeHomeDataset_10072016')
        with open(os.path.join(data_dir, 'artreal.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'officehome_artproduct':
        data_dir = 'data/officehome'
        imgdir = os.path.join(basedir, 'OfficeHomeDataset_10072016')
        with open(os.path.join(data_dir, 'artproduct.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'officehome_clipartreal':
        data_dir = 'data/officehome'
        imgdir = os.path.join(basedir, 'OfficeHomeDataset_10072016')
        with open(os.path.join(data_dir, 'clipartreal.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'officehome_productreal':
        data_dir = 'data/officehome'
        imgdir = os.path.join(basedir, 'OfficeHomeDataset_10072016')
        with open(os.path.join(data_dir, 'productreal.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'officehome_productclipart':
        data_dir = 'data/officehome'
        imgdir = os.path.join(basedir, 'OfficeHomeDataset_10072016')
        with open(os.path.join(data_dir, 'productclipart.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'officehome_artclipart':
        data_dir = 'data/officehome'
        imgdir = os.path.join(basedir, 'OfficeHomeDataset_10072016')
        with open(os.path.join(data_dir, 'artclipart.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'officehome_art':
        data_dir = 'data/officehome'
        imgdir = os.path.join(basedir, 'OfficeHomeDataset_10072016')
        with open(os.path.join(data_dir, 'art.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'officehome_clipart':
        data_dir = 'data/officehome'
        imgdir = os.path.join(basedir, 'OfficeHomeDataset_10072016')
        with open(os.path.join(data_dir, 'clipart.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'officehome_product':
        data_dir = 'data/officehome'
        imgdir = os.path.join(basedir, 'OfficeHomeDataset_10072016')
        with open(os.path.join(data_dir, 'product.txt'), 'r') as f:
            data = f.readlines()
    elif dataset == 'officehome_real':
        data_dir = 'data/officehome'
        imgdir = os.path.join(basedir, 'OfficeHomeDataset_10072016')
        with open(os.path.join(data_dir, 'real.txt'), 'r') as f:
            data = f.readlines()
    img_paths = [os.path.join(imgdir, _.split()[0]) for _ in data]
    labels = [int(_.split()[-1]) for _ in data]
    return img_paths, labels

class TextualInversionDataset_test(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        interpolation="bicubic",
        set="train",
        random_crop=False,
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.random_crop=random_crop
        self.image_paths, self.labels = load_filenames(data_root)
        self.num_images = len(self.image_paths)
        self._length = self.num_images
        file_path = f"{self.data_root}meanprompt_share_features.npy"
        if os.path.exists(file_path):
            self.hidden = np.load(file_path)
        else:
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        if set == "train":
            self._length = self.num_images

        self.interpolation = {
            "linear": PIL.Image.Resampling.BILINEAR,
            "bilinear": PIL.Image.Resampling.BILINEAR,
            "bicubic": PIL.Image.Resampling.BICUBIC,
            "lanczos": PIL.Image.Resampling.LANCZOS,
            "nearest": PIL.Image.Resampling.NEAREST,
        }[interpolation]
        local_f = open('configs/text_all.json')
        self.templates = json.load(local_f)
        
            
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        file = self.image_paths[i % self.num_images]
        label = self.labels[i % self.num_images]
        image = Image.open(file)
        # filename = file.split('.jpg')[0].split('/')[-1]
        filename = os.path.splitext(os.path.basename(file))[0]
        hidden_feature = self.hidden[i % self.num_images]
        example["filename"] = filename
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if self.data_root == 'sketchimg':
            text = "a photo of a <R>" if  '_p' in filename else "a sketch of a <R>"
        elif 'domainnet' in self.data_root and 'general' not in self.data_root:
            text = "a photo of a <R>" 
        elif 'officehome' in self.data_root:
            text = "a photo of a <R>" 

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)
        if self.random_crop:
            #print('set random crop and flip')
            image = random_crop_arr(image, self.size)
            if random.random() < 0.5:
                image = image[:, ::-1]
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        example["label"] = int(label)
        example["hid_feature"] = torch.from_numpy(hidden_feature)
        example["index"] = i
        return example

class TextualInversionDataset_gethid(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        interpolation="bicubic",
        set="train",
        random_crop=False,
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.random_crop=random_crop
        self.image_paths, self.labels = load_filenames(data_root)
        self.num_images = len(self.image_paths)
        self._length = self.num_images
        if set == "train":
            self._length = self.num_images

        self.interpolation = {
            "linear": PIL.Image.Resampling.BILINEAR,
            "bilinear": PIL.Image.Resampling.BILINEAR,
            "bicubic": PIL.Image.Resampling.BICUBIC,
            "lanczos": PIL.Image.Resampling.LANCZOS,
            "nearest": PIL.Image.Resampling.NEAREST,
        }[interpolation]
        local_f = open('configs/text_all.json')
        self.templates = json.load(local_f)
        
            
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        file = self.image_paths[i % self.num_images]
        # print(file)
        label = self.labels[i % self.num_images]
        image = Image.open(file)
        # filename = file.split('.jpg')[0].split('/')[-1]
        filename = os.path.splitext(os.path.basename(file))[0]
        example["filename"] = filename
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if self.data_root == 'sketchimg':
            text = "a photo of a <R>" if  '_p' in filename else "a sketch of a <R>"
        elif 'domainnet' in self.data_root and 'general' not in self.data_root:
            text = "a photo of a <R>" 
        elif 'officehome' in self.data_root:
            text = "a photo of a <R>" 
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)
        if self.random_crop:
            #print('set random crop and flip')
            image = random_crop_arr(image, self.size)
            if random.random() < 0.5:
                image = image[:, ::-1]
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        example["label"] = int(label)
        example["index"] = i
        return example
    
def create_dataloader(train_dataset, train_batch_size=16):
        return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, drop_last=False)

def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]



import torch
from torchvision import transforms

# Define your custom Cutout and Augment classes (these would be based on your implementation)
class Cutout:
    def __init__(self, n_holes, length, random=True):
        self.n_holes = n_holes
        self.length = length
        self.random = random

    def __call__(self, img):
        # Apply cutout logic here
        return img

class Augment:
    def __init__(self, num_augs):
        self.num_augs = num_augs

    def __call__(self, img):
        # Apply augmentation logic here
        return img

# Define the create_dataloader function
def create_dataloader_aug(train_dataset, train_batch_size=16, p=None):
    if p is None:
        p = {
            'augmentation_kwargs': {
                'crop_size': 32,
                'normalize': {
                    'mean': [0.4914, 0.4822, 0.4465],
                    'std': [0.2023, 0.1994, 0.2010]
                },
                'num_strong_augs': 4,
                'cutout_kwargs': {
                    'n_holes': 1,
                    'length': 16,
                    'random': True
                }
            },
            'transformation_kwargs': {
                'crop_size': 32,
                'normalize': {
                    'mean': [0.4914, 0.4822, 0.4465],
                    'std': [0.2023, 0.1994, 0.2010]
                }
            }
        }

    # Set the transformations using augmentation and transformation kwargs
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
        # Augment(p['augmentation_kwargs']['num_strong_augs']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=p['augmentation_kwargs']['normalize']['mean'], 
            std=p['augmentation_kwargs']['normalize']['std']
        ),
        # Cutout(
        #     n_holes=p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
        #     length=p['augmentation_kwargs']['cutout_kwargs']['length'],
        #     random=p['augmentation_kwargs']['cutout_kwargs']['random']
        # )
    ])
    
    # Apply the transform to the dataset
    train_dataset.transform = transform

    return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
