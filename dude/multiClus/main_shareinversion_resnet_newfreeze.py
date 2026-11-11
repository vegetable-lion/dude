import argparse
import itertools
import math
import os
import random
import json
import numpy as np
import torch
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
from util.datasets_resnet_hidden_freeze import TextualInversionDataset_test, create_dataloader
from util.utils import freeze_params, importance_sampling_fn
from util.eval_utils import cal_kmeans_loop, eval_clusterhead
import time
import datetime
from train_cnn import train_one_epoch
from models_mlp import MLP
from typing import List, Optional
from einops import rearrange
from sklearn.mixture import GaussianMixture
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
        default=5000,
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
    
    model = MLP() 
    gmm = GaussianMixture(n_components=2)
    test_features = np.load(f'{args.train_data_dir}_resnet_features.npy'
            )
    labels_test = np.load(f'{args.train_data_dir}_resnet_labels.npy')
    train_dataset = TextualInversionDataset_test(
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
    train_dataloader = create_dataloader(train_dataset, args.train_batch_size)

    if args.scale_lr:
        learning_rate = (
            args.learning_rate *  args.train_batch_size * accelerator.num_processes
        )
    print("base lr: %.2e" % args.learning_rate )
    print("actual lr: %.2e" % learning_rate)
    optimizer = torch.optim.AdamW(
        model.parameters(),  # only optimize the embeddings
        lr=learning_rate,
    )
    vae, unet, noise_scheduler,text_encoder, optimizer, train_dataloader, model = accelerator.prepare(
        vae, unet, noise_scheduler, text_encoder, optimizer,
        train_dataloader, model
    )
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    uncond_tokens = [""]
    print(uncond_tokens)
    uncond_input = tokenizer(
        uncond_tokens,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    print(uncond_input.shape)
    uncond_embeds = text_encoder(uncond_input)[0].to(dtype=weight_dtype)
    #shared_noise = torch.randn((1, 4, 64, 64), device=accelerator.device)
    # Move vae and unet to device
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    model.to(accelerator.device, dtype=weight_dtype)
    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    unet.eval()
    processors = unet.attn_processors
    num_train_epochs = args.max_train_steps
    
    if args.import_relate_timestep==False:
        list_of_candidates = None
        prob_dist = None
    else:
        list_of_candidates = [
                x for x in range(1000)
        ]
        prob_dist = [
            importance_sampling_fn(x,
                                1000,
                                0.5)
            for x in list_of_candidates
        ]
        prob_sum = 0
        # normalize the prob_list so that sum of prob is 1
        for i in prob_dist:
            prob_sum += i
        prob_dist = [x / prob_sum for x in prob_dist]
        
    # Train!
    print(f"Start training for {num_train_epochs} epochs")
    start_time = time.time()
    global_step = 0
    loads = 0
    for epoch in range(num_train_epochs):
        print(f'=================================')
        print(f'epoch [{epoch}/{num_train_epochs}]')
        model.train()
        text_encoder.train()
        if args.model_dir!= None:
            if loads==0:
                checkpoint = torch.load(args.model_dir)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch_load = checkpoint['epoch']
                print('load the model and optimizer')
                print(f'load epoch = {epoch_load}')
                loads=1
            else:
                train_one_epoch(
                    vae, unet, noise_scheduler, text_encoder, placeholder_token_id, processors, 
                    model, gmm, train_dataloader, accelerator, uncond_embeds, uncond_input,
                    global_step, optimizer, weight_dtype, 
                    args=args, list_of_candidates=list_of_candidates, prob_dist=prob_dist)
            
                # eval_clusterhead(test_features, labels_test, model, args=args)
        if args.model_dir== None:
            train_one_epoch(
                    vae, unet, noise_scheduler, text_encoder, placeholder_token_id, processors, 
                    model, gmm, train_dataloader, accelerator, uncond_embeds, uncond_input,
                    global_step, optimizer, weight_dtype, 
                    args=args, list_of_candidates=list_of_candidates, prob_dist=prob_dist)
            
        # save_mlp = os.path.join(args.output_dir, f"cnn_model_epoch{epoch}.pth")
        # torch.save(model.state_dict(), save_mlp)  
        # if accelerator.is_main_process:
        #     model_save = accelerator.unwrap_model(model)
        #     torch.save(model_save.state_dict(), save_mlp) 
        # print(f'save cnn_model_epoch{epoch}.pth')
        if accelerator.is_main_process:    
            if args.model_dir!= None:
                epoch_save = epoch+epoch_load
            else:
                epoch_save = epoch
            if (epoch_save+1)%10==0:
                save_mlp = os.path.join(args.output_dir, f"checkpoint_epoch{epoch_save}.pth")
                print(f'save checkpoint{epoch_save}.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch_save
                }, save_mlp)
            
        accelerator.wait_for_everyone()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
       
if __name__ == '__main__':
    args = parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
    
         