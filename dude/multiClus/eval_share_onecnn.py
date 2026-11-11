import argparse
import itertools
import math
import os
import random
import json
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances
import torch
import faiss
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from util.eval_utils import get_stat,get_match,get_acc,get_nmi
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from util.datasets_sketch_hidden_test import TextualInversionDataset_test, create_dataloader
from util.utils import freeze_params, importance_sampling_fn
from util.eval_utils import cal_kmeans_loop, eval_clusterhead
import time
import datetime
from train_sketch_each import train_one_epoch
from typing import List, Optional
from einops import rearrange
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from models_cnn import CNN
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告

def parse_args():
    parser = argparse.ArgumentParser(
        description="training script")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='runwayml/stable-diffusion-v1-5',
        required=False,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        required=True,
        help=
        "The folder that contains the exemplar images (and coarse descriptions) of the specific relation."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=
        ("The resolution for input images, all the images in the train/validation dataset will be resized to this"
         " resolution"),
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=
        "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=409,
        help=
        "Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-05,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."),
    )
    parser.add_argument(
        "--import_relate_timestep",
        action='store_true',
        default=False,
        help="Relation-Focal Importance Sampling",
    )
    parser.add_argument(
        "--do_not_apply_masked_loss",
        action="store_false",
        help="Use masked loss instead of standard epsilon prediciton loss",
        dest="apply_masked_loss"
    )
    parser.add_argument('--prec-nums', default='1,5,15', type=str,
                    help='the evaluation metric')
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        required=False,
        help=
        "The directory where the trained model stored.",
    )
    args = parser.parse_args()
    # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if env_local_rank != -1 and env_local_rank != args.local_rank:
    #     args.local_rank = env_local_rank
    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")
    return args

def kmeans(X, cluster_num):
    print("Perform K-means clustering...")
    d = X.shape[1]
    X = X.astype(np.float32)
    kmeans = faiss.Kmeans(d, cluster_num, gpu=True, spherical=True, niter=300, nredo=10)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    I = I.reshape(-1)
    print("K-means clustering done.")
    return I

def retrieval_precision_cal(features_A, targets_A, features_B, targets_B, preck=(1, 5, 15)):

    dists = cosine_similarity(features_A, features_B)

    res_A = []
    res_B = []
    for domain_id in ['A', 'B']:
        if domain_id == 'A':
            query_targets = targets_A
            gallery_targets = targets_B

            all_dists = dists

            res = res_A
        else:
            query_targets = targets_B
            gallery_targets = targets_A

            all_dists = dists.transpose()
            res = res_B

        sorted_indices = np.argsort(-all_dists, axis=1)
        print(sorted_indices)
        sorted_cates = gallery_targets[sorted_indices.flatten()].reshape(sorted_indices.shape)
        correct = (sorted_cates == np.tile(query_targets[:, np.newaxis], sorted_cates.shape[1]))

        for k in preck:
            total_num = 0
            positive_num = 0
            for index in range(all_dists.shape[0]):

                temp_total = min(k, (gallery_targets == query_targets[index]).sum())
                pred = correct[index, :temp_total]

                total_num += temp_total
                positive_num += pred.sum()
            res.append(positive_num / total_num * 100.0)

    return res_A, res_B

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    # Load models
    placeholder_token = '<R>'
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer") 
    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    
    text_encoder.resize_token_embeddings(len(tokenizer))
    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)
    

    if 'domainnet_sketchclipart' in args.train_data_dir and 'general' not in args.train_data_dir:
        targets_k=7 
        train_data_dir_dm1 = 'domainnet_clipart'
        train_data_dir_dm2 = 'domainnet_sketch'
        labels_test_dm1 = np.load(f'{train_data_dir_dm1}_share_labels.npy')
        labels_test_dm2 = np.load(f'{train_data_dir_dm2}_share_labels.npy')
    elif 'domainnet_sketchinfograph' in args.train_data_dir and 'general' not in args.train_data_dir:
        targets_k=7 
        train_data_dir_dm1 = 'domainnet_sketch'
        train_data_dir_dm2 = 'domainnet_infograph'
        labels_test_dm1 = np.load(f'{train_data_dir_dm1}_share_labels.npy')
        labels_test_dm2 = np.load(f'{train_data_dir_dm2}_share_labels.npy')
    elif 'domainnet_infographreal' in args.train_data_dir and 'general' not in args.train_data_dir:
        targets_k=7 
        train_data_dir_dm1 = 'domainnet_infograph'
        train_data_dir_dm2 = 'domainnet_real'
        labels_test_dm1 = np.load(f'{train_data_dir_dm1}_share_labels.npy')
        labels_test_dm2 = np.load(f'{train_data_dir_dm2}_share_labels.npy')
    elif 'domainnet_quickdrawreal' in args.train_data_dir and 'general' not in args.train_data_dir:
        targets_k=7 
        train_data_dir_dm1 = 'domainnet_quickdraw'
        train_data_dir_dm2 = 'domainnet_real'
        labels_test_dm1 = np.load(f'{train_data_dir_dm1}_share_labels.npy')
        labels_test_dm2 = np.load(f'{train_data_dir_dm2}_share_labels.npy')
    elif 'domainnet_paintingclipart' in args.train_data_dir and 'general' not in args.train_data_dir:
        targets_k=7 
        train_data_dir_dm1 = 'domainnet_painting'
        train_data_dir_dm2 = 'domainnet_clipart'
        labels_test_dm1 = np.load(f'{train_data_dir_dm1}_share_labels.npy')
        labels_test_dm2 = np.load(f'{train_data_dir_dm2}_share_labels.npy')
    elif 'domainnet_paintingquickdraw' in args.train_data_dir and 'general' not in args.train_data_dir:
        targets_k=7 
        train_data_dir_dm1 = 'domainnet_painting'
        train_data_dir_dm2 = 'domainnet_quickdraw'
        labels_test_dm1 = np.load(f'{train_data_dir_dm1}_share_labels.npy')
        labels_test_dm2 = np.load(f'{train_data_dir_dm2}_share_labels.npy')
    
    elif 'domainnet' in args.train_data_dir and 'general' in args.train_data_dir:
        targets_k=7 
        # labels_test_all = np.load(f'domainnet_sketchclipart_general_share_labels.npy')
        train_data_dir_dm1 = 'domainnet_clipart'
        train_data_dir_dm2 = 'domainnet_sketch'
        labels_test_dm1 = np.load(f'{train_data_dir_dm1}_general_share_labels.npy')
        labels_test_dm2 = np.load(f'{train_data_dir_dm2}_general_share_labels.npy')
    else:
        raise NotImplementedError
    test_dataset_all = TextualInversionDataset_test(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=512,
        set="train",
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        # distributed_type='MULTI_GPU' 
        # distribute_type: Accelerate.DistributedType.MULTI_GPU
        # fp16=True,
        # cpu=True,
    )
    test_dataloader_all = create_dataloader(test_dataset_all, args.train_batch_size)
    model = CNN() 
    model.eval()
    model = torch.nn.DataParallel(model)
    epoch = args.max_train_steps
    save_mlp = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pth")
    checkpoint = torch.load(save_mlp)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.module
    text_encoder, test_dataloader_all, model = accelerator.prepare(
        text_encoder , test_dataloader_all, model
    )
    features = []
    for step, batch in enumerate(test_dataloader_all):
        with torch.no_grad():  # Disable gradient computation
            feature = model(batch["hid_feature"]) 
            features.append(feature)
    features = torch.cat(features, dim=0)
    features = features.detach().cpu().numpy()
    features_dm1 = features[:len(labels_test_dm1)]
    features_dm2 = features[len(labels_test_dm1):]
    prec_nums = args.prec_nums.split(',')
    preck = [int(prec_nums[0]), int(prec_nums[1]), int(prec_nums[2])]
    res_1, res_2 = retrieval_precision_cal(features_dm1,
                                                   labels_test_dm1,
                                                   features_dm2,
                                                   labels_test_dm2,
                                                   preck=preck)

    print("Domain {}->{}: P@{}: {}; P@{}: {}; P@{}: {} \n".format(
                train_data_dir_dm1, train_data_dir_dm2,
                int(prec_nums[0]), res_1[0],
                int(prec_nums[1]), res_1[1],
                int(prec_nums[2]), res_1[2]))
    
    res_1, res_2 = retrieval_precision_cal(features_dm2,
                                                   labels_test_dm2,
                                                   features_dm1,
                                                   labels_test_dm1,
                                                   preck=preck)

    print("Domain {}->{}: P@{}: {}; P@{}: {}; P@{}: {} \n".format(
                train_data_dir_dm2, train_data_dir_dm1,
                int(prec_nums[0]), res_1[0],
                int(prec_nums[1]), res_1[1],
                int(prec_nums[2]), res_1[2]))
        
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
    # mainloop(args)
    
         