import abc
import torch
from typing import List, Optional
from einops import rearrange
import torch.nn.functional as F
import os
import numpy as np
from torchvision import transforms
from PIL import Image

class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # if attn.shape[1] <= 32**2:
        self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        
def register_attention_control(unet, controller):
    attn_procs = {}
    cross_att_count = 0
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue
        cross_att_count += 1
        attn_procs[name] = P2PCrossAttnProcessor(
            controller=controller, place_in_unet=place_in_unet
        )

    unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count
    return unet, controller

class P2PCrossAttnProcessor:
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    
def aggregate_attention(
    controller, res: int, from_where: List[str], is_cross: bool, select: int
):
    out = []
    attention_maps = get_average_attention(controller)
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(
                    1, -1, res, res, item.shape[-1]
                )[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out

def get_average_attention(controller):
    average_attention = {
        key: [item / controller.cur_step for item in controller.attention_store[key]]
        for key in controller.attention_store
    }
    #print(average_attention.keys())
    return average_attention

def get_self_attention(args, accelerator, controller, gmm, eot_attn_mask, pil, filename, global_step):
    pil = pil * 0.5 + 0.5
    out = []
    rw = 0.0
    attention_maps = get_average_attention(controller)
    for location in ("down", "mid", "up"):
        for item in attention_maps[f"{location}_self"]:
            res = int(item.shape[-1] ** 0.5)
            self_maps = item.reshape(
                1, -1, item.shape[-2], item.shape[-1]
            )
            self_maps = self_maps.sum(1) / self_maps.shape[1]
            self_maps = self_maps.reshape(1, -1, res, res)
            upsampled_self_maps = F.interpolate(
                input=self_maps, size=(64, 64), mode="bilinear"
            )
            upsampled_self_maps = upsampled_self_maps.reshape(
                1, res, res, -1
            )

            upsampled_self_maps = torch.repeat_interleave(
                upsampled_self_maps, repeats=int(64 / res), dim=1
            )
            upsampled_self_maps = torch.repeat_interleave(
                upsampled_self_maps, repeats=int(64 / res), dim=2
            )

            upsampled_self_maps = upsampled_self_maps.reshape(
                1, 64**2, 64**2
            )

            r_weight = float(res / 64)

            out.append(upsampled_self_maps * r_weight)
            rw += r_weight

    out = torch.stack(out, dim=0)
    out = out.sum(0) / rw

    feat_map = out.reshape(64**2, 64**2)
    # cluster_attention_draw(
    #     args, accelerator, gmm, feat_map, eot_attn_mask, pil, filename, global_step
    # )
    binary_mask = cluster_attention(
        gmm, eot_attn_mask
    )
    return binary_mask

def cluster_attention_draw(
    args, accelerator, gmm, feat_map, eot_attn_mask, pil, filename, global_step
):  
    eot_attn_mask = F.interpolate(
        input=eot_attn_mask[None, None], size=(64, 64), mode="bilinear"
    )
    eot_attn_mask = eot_attn_mask.reshape(64, 64)

    eot_attn_mask = (eot_attn_mask - eot_attn_mask.min()) / (
        eot_attn_mask.max() - eot_attn_mask.min()
    )

    transform = transforms.ToPILImage()
    mask = transform(eot_attn_mask)
    mask.save(
        os.path.join(
            args.output_dir,
            "{}_eot_attention.png".format(filename),
        )
    )
    #print(eot_attn_mask.shape)
    gmm.fit(rearrange(eot_attn_mask.to("cpu"), "h w -> (h w) 1"))
    threshold = np.mean(gmm.means_)
    binary_img = eot_attn_mask > threshold
    #print(binary_img.shape)
    #print(binary_img)
    binary_img = binary_img.cuda()
    assert len(binary_img.size()) == 2
    binary_img = rearrange(binary_img, "h w -> 1 1 h w").to(torch.float32)
    weights = torch.full((1, 1, 3, 3), 1/9)
    binary_img[...,1:-1,1:-1] = torch.nn.functional.conv2d(binary_img, weights.to(binary_img.device), bias=torch.zeros(1, device=binary_img.device))
    binary_img[binary_img >= 0.5] = 1
    binary_img[binary_img < 0.5] = 0
    #print(binary_img)
    binary_img = transform(binary_img.squeeze())
    binary_img.save(
        os.path.join(
            args.output_dir,
            "{}_binary_gmm.png".format(filename),
        )
    )

    pil = transform(pil)
    # pil.save(
    #     os.path.join(args.output_dir, "attention/{}-step/image.png".format(global_step))
    # )
    image_masked = vis_masked_image(pil, binary_img)

    image_masked.save(
        os.path.join(
            args.output_dir,
            "{}_final_masked.png".format(filename),
            )
    )
    
def cluster_attention(
    gmm, eot_attn_mask
):  
    eot_attn_mask = F.interpolate(
        input=eot_attn_mask[None, None], size=(64, 64), mode="bilinear"
    )
    eot_attn_mask = eot_attn_mask.reshape(64, 64)
    eot_attn_mask = (eot_attn_mask - eot_attn_mask.min()) / (
        eot_attn_mask.max() - eot_attn_mask.min()
    )
    gmm.fit(rearrange(eot_attn_mask.to("cpu"), "h w -> (h w) 1"))
    threshold = np.mean(gmm.means_)
    binary_img = eot_attn_mask > threshold
    binary_img = binary_img.cuda()
    assert len(binary_img.size()) == 2
    binary_img = rearrange(binary_img, "h w -> 1 1 h w").to(torch.float32)
    weights = torch.full((1, 1, 3, 3), 1/9)
    binary_img[...,1:-1,1:-1] = torch.nn.functional.conv2d(binary_img, weights.to(binary_img.device), bias=torch.zeros(1, device=binary_img.device))
    binary_img[binary_img >= 0.5] = 1
    binary_img[binary_img < 0.5] = 0
    #print(binary_img.shape)
    return binary_img
    

    
       
def vis_masked_image(image, mask):
    tsfm = transforms.Resize([256, 256])
    image_np = np.array(tsfm(image))

    tsfm_mask = transforms.Resize(
        [256, 256], interpolation=transforms.InterpolationMode.NEAREST
    )
    mask_np = np.array(tsfm_mask(mask)).reshape(256, 256, 1) / 255

    image_masked = image_np * mask_np.astype(np.uint8)

    image_masked = Image.fromarray(image_masked)

    return image_masked