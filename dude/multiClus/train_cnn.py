import math
import sys
import torch
import torch.nn.functional as F
import os
import numpy as np
from util.mask import AttentionStore, register_attention_control, aggregate_attention, get_self_attention

def train_one_epoch(
    vae, unet, noise_scheduler, text_encoder, placeholder_token_id, processors, 
    model, gmm, train_dataloader, accelerator, uncond_embeds, uncond_input,
    global_step, optimizer, weight_dtype, 
    args=None,
    list_of_candidates=None,
    prob_dist=None,
):
    model.train()
    text_encoder.train()
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            processors_clone = processors.copy()
            latents = (
                vae.encode(batch["pixel_values"].to(dtype=weight_dtype))
                .latent_dist.sample()
                .detach()
            )
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            #noise_un = shared_noise.repeat(latents.shape[0], 1, 1, 1)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            # uncond_embeds_batch = uncond_embeds.repeat(latents.shape[0], 1, 1)
            # # timesteps = torch.randint(500, 900, (bsz,), device=latents.device).long()
            # timesteps_un = torch.full((bsz,), 150, device=latents.device).long()
            # # Add noise to the latents according to the noise magnitude at each timestep
            # noisy_latents_un = noise_scheduler.add_noise(
            #     latents, noise_un, timesteps_un
            # )
            
            # _, hidden = unet.forward2(
            #         noisy_latents_un, timesteps_un, uncond_embeds_batch
            #     )
            # hidden = hidden.sample
            hidden = batch["hid_feature"]
            # print(hidden.shape)
            # print(torch.norm(hidden))

            noise = torch.randn(latents.shape).to(latents.device)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            if args.import_relate_timestep == False:
                timesteps = torch.randint(
                    500, 900, (bsz,), device=latents.device
                ).long()
            # ======================================================================================================
            else:
                timesteps = np.random.choice(
                    list_of_candidates, size=bsz, replace=True, p=prob_dist
                )
                timesteps = torch.tensor(timesteps).cuda().long()
            # ======================================================================================================

            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            indices = []
            for i in range(batch["input_ids"].size(0)):
                indices.append(
                    torch.nonzero(
                        batch["input_ids"][i] == placeholder_token_id, as_tuple=False
                    ).cpu()
                )
            indices = [t.item() for t in indices]
            # print(indices)

            # 检查每个样本是否找到了 placeholder_token_id
            if accelerator.num_processes > 1:
                token_embeds = text_encoder.module.get_input_embeddings()
            else:
                token_embeds = text_encoder.get_input_embeddings()
            token_embeds_values = token_embeds(batch["input_ids"]).clone()

            # 使用 mlp 计算
            cnn_outputs = model(hidden)
                
            # print(cnn_outputs[0][:20])
            # print(torch.norm(cnn_outputs[0]))
            # print(torch.norm(token_embeds_values[0][0]))
            # 创建一个和 token_embeds_values 大小相同的零张量
            updates = torch.zeros_like(token_embeds_values)
            zero_vector = torch.zeros(768)
            for i, idx in enumerate(indices):
                token_embeds_values[i, idx, :] = zero_vector
                updates[i, idx, :] = cnn_outputs[i]

            # 将更新应用到 token_embeds_values
            token_embeds_values = token_embeds_values + updates
            # print(token_embeds_values[0][:20])
            if accelerator.num_processes > 1:
                text_inputs = text_encoder.module.forward2(
                            token_embeds_values,
                            batch["input_ids"])[0].to(dtype=weight_dtype)
            else:
                text_inputs = text_encoder.forward2(
                            token_embeds_values,
                            batch["input_ids"])[0].to(dtype=weight_dtype)
            encoder_hidden_states = text_inputs
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
            accelerator.backward(loss)
            # loss.backward()
            print("loss ", loss)
            if global_step % 500 == 0:
                # save_mlp = os.path.join(args.output_dir, f"cnn_model_{global_step}.pth")
                # torch.save(model.state_dict(), save_mlp)
                print(f"global_step: {global_step}")
            optimizer.step()
            optimizer.zero_grad()
        if accelerator.sync_gradients:
            global_step += 1
    return global_step
