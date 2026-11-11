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
    dataset_config = {
        'sketch': ('data/sketch', 'sketchsub', 'subset.txt'),
        'sketchimg': ('data/sketch', 'sketchimgsub', 'sketchimgsub.txt'),
        'domainnet_sketch': ('data/domainnet', 'domainnet_sub', 'sketch_sub.txt'),
        'domainnet_clipart': ('data/domainnet', 'domainnet_sub', 'clipart_sub.txt'),
        'domainnet_infograph': ('data/domainnet', 'domainnet_sub', 'infograph_sub.txt'),
        'domainnet_real': ('data/domainnet', 'domainnet_sub', 'real_sub.txt'),
        'domainnet_quickdraw': ('data/domainnet', 'domainnet_sub', 'quickdraw_sub.txt'),
        'domainnet_sketchclipart': ('data/domainnet', 'domainnet_sub', 'sketchclipart_catsub.txt'),
        'domainnet_sketchclipart_general': ('data/domainnet', 'plots_aug_noblack', 'sketchclipart_general_noblack.txt'),
        'domainnet_sketchinfograph': ('data/domainnet', 'domainnet_sub', 'sketchinfograph_catsub.txt'),
        'domainnet_infographreal': ('data/domainnet', 'domainnet_sub', 'infographreal_catsub.txt'),
        'domainnet_paintingclipart': ('data/domainnet', 'domainnet_sub', 'paintingclipart_catsub.txt'),
        'domainnet_paintingquickdraw': ('data/domainnet', 'domainnet_sub', 'paintingquickdraw_catsub.txt'),
        'domainnet_quickdrawreal': ('data/domainnet', 'domainnet_sub', 'quickdrawreal_catsub.txt'),
        'officehome_artreal': ('data/officehome', 'OfficeHomeDataset_10072016', 'artreal.txt'),
        'officehome_artproduct': ('data/officehome', 'OfficeHomeDataset_10072016', 'artproduct.txt'),
        'officehome_clipartreal': ('data/officehome', 'OfficeHomeDataset_10072016', 'clipartreal.txt'),
        'officehome_productreal': ('data/officehome', 'OfficeHomeDataset_10072016', 'productreal.txt'),
        'officehome_productclipart': ('data/officehome', 'OfficeHomeDataset_10072016', 'productclipart.txt'),
        'officehome_artclipart': ('data/officehome', 'OfficeHomeDataset_10072016', 'artclipart.txt'),
        'officehome_art': ('data/officehome', 'OfficeHomeDataset_10072016', 'art.txt'),
        'officehome_clipart': ('data/officehome', 'OfficeHomeDataset_10072016', 'clipart.txt'),
        'officehome_product': ('data/officehome', 'OfficeHomeDataset_10072016', 'product.txt'),
        'officehome_real': ('data/officehome', 'OfficeHomeDataset_10072016', 'real.txt'),
        'pacs_art': ('data/pacs', 'PACS', 'art.txt'),
        'pacs_cartoon': ('data/pacs', 'PACS', 'cartoon.txt'),
        'pacs_photo': ('data/pacs', 'PACS', 'photo.txt'),
        'pacs_sketch': ('data/pacs', 'PACS', 'sketch.txt'),
        'pacs_artcartoon': ('data/pacs', 'PACS', 'artcartoon.txt'),
        'pacs_artphoto': ('data/pacs', 'PACS', 'artphoto.txt'),
        'pacs_artsketch': ('data/pacs', 'PACS', 'artsketch.txt'),
        'pacs_cartoonphoto': ('data/pacs', 'PACS', 'cartoonphoto.txt'),
        'pacs_cartoonsketch': ('data/pacs', 'PACS', 'cartoonsketch.txt'),
        'pacs_photosketch': ('data/pacs', 'PACS', 'photosketch.txt'),
    }

    if dataset in dataset_config:
        data_dir, imgsubdir, txtfile = dataset_config[dataset]
        imgdir = imgsubdir if os.path.isabs(imgsubdir) else os.path.join(basedir, imgsubdir)
        with open(os.path.join(data_dir, txtfile), 'r') as f:
            data = f.readlines()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

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
        file_path = f"{self.data_root}_resnet_features.npy"
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
            # print(self.data_root)
            if  'sketch' in filename: text = "a sketch of a <R>" 
            elif  'clipart' in filename: text = "a clipart of a <R>" 
            elif  'infograph' in filename: text = "a infograph of a <R>" 
            elif  'real' in filename: text = "a photo of a <R>" 
            elif  'painting' in filename: text = "a painting of a <R>"
            elif 'quickdraw' in file: text = "A minimalistic, abstract line drawing of a <R>."
        elif 'domainnet' in self.data_root and 'general' in self.data_root:
            text = "a photo of a <R>"
        elif 'officehome' in self.data_root:
            if  'Art' in file: text = "an painting of a <R>" 
            elif  'Clipart' in file: text = "a clipart of a <R>" 
            elif 'Real' in file: text = "a photo of a <R>" 
            elif  'Product' in file: text = "a product photo of a <R>" 
        elif 'pacs' in self.data_root:
            if  'art_painting' in file: text = "an painting of a <R>" 
            elif  'cartoon' in file: text = "a cartoon of a <R>" 
            elif 'photo' in file: text = "a photo of a <R>" 
            elif  'sketch' in file: text = "a sketch of a <R>" 

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
