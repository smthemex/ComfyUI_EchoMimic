import torch
import numpy as np

def get_alpha(alphas_cumprod, timestep):
    timestep_lt_zero_mask = torch.lt(timestep, 0).to(alphas_cumprod.dtype)
    normal_alpha = alphas_cumprod[torch.clip(timestep, 0)]
    one_alpha = torch.ones_like(normal_alpha).to(normal_alpha.dtype).to(normal_alpha.dtype) 
    return normal_alpha * (1 - timestep_lt_zero_mask) + one_alpha * timestep_lt_zero_mask

def get_timestep_list_wrt_margin_and_nsteps(timestep, margin, total_steps):
    time_dtype = timestep.dtype
    time_device = timestep.device
    timestep_list = timestep.cpu().numpy().reshape(-1).tolist()
    if type(margin) is int:
        margin_list = [margin for _ in range(len(timestep_list))]
    else:
        assert margin.dtype == time_dtype
        assert margin.device == time_device
        margin_list = margin.cpu().numpy().reshape(-1).tolist()

    result_list = []
    for curr_t, margin_t in zip(timestep_list, margin_list):
        next_t = min(1000, max(-1, curr_t - margin_t))
        curr_to_next_steps = [round(i) for i in np.linspace(curr_t, next_t, total_steps + 1)]
        result_list.append(curr_to_next_steps)

    timestep_list = [
        torch.tensor([result_list[i][j] for i in range(len(result_list))], dtype=time_dtype, device=time_device)
        for j in range(len(result_list[0]))
    ]

    return timestep_list 

def scheduler_pred_onestep(
        model, noisy_images, scheduler, timestep, timestep_prev,
        audio_cond_fea, face_musk_fea, guidance_scale,
    ):
    # print("face mush feat shape {}".format(torch.cat([torch.zeros_like(face_musk_fea), face_musk_fea], dim=0).shape))
    noisy_pred_uncond, noisy_pred_text = model(
        torch.cat([noisy_images, noisy_images], dim=0), 
        torch.cat([timestep, timestep], dim=0), 
        encoder_hidden_states=None,
        audio_cond_fea = torch.cat([torch.zeros_like(audio_cond_fea), audio_cond_fea], dim=0),
        face_musk_fea=torch.cat([torch.zeros_like(face_musk_fea), face_musk_fea], dim=0),
        return_dict=False,
        )[0].chunk(2)
    # noisy_pred_text = model(noisy_images, timestep, audio_cond_fea=audio_cond_fea, face_musk_fea = face_musk_fea, encoder_hidden_states=None,).sample
    # noisy_pred_uncond = model(noisy_images, timestep, audio_cond_fea=torch.zeros_like(audio_cond_fea), face_musk_fea = face_musk_fea, encoder_hidden_states=None,).sample

    #noisy_pred_uncond = model(noisy_images, timestep, uncond_encoder_hidden_states, return_dict=False, **unet_kwargs)[0]
    noisy_pred = noisy_pred_uncond + guidance_scale * (noisy_pred_text - noisy_pred_uncond)
    #embed()

    #print(noisy_images.std(), noisy_residual.std())
    prev_sample, pred_original_sample = scheduler.pred_prev(
        noisy_images, noisy_pred, timestep, timestep_prev
    )

    return prev_sample, pred_original_sample

def scheduler_pred_multisteps(
        npred_model, noisy_images, scheduler, timestep_list,
        audio_cond_fea, face_musk_fea,
        guidance_scale, 
    ):
    prev_sample = noisy_images
    origin = noisy_images

    #assert encoder_hidden_states.shape[0] == noisy_images.shape[0] * 2, (encoder_hidden_states.shape, noisy_images.shape)
    for step_idx, (timestep_home, timestep_end) in enumerate(zip(timestep_list[:-1], timestep_list[1:])):
        assert timestep_home.dtype is torch.int64
        assert timestep_end.dtype is torch.int64
        #timestep_home_gt_end_mask = torch.gt(timestep_home, timestep_end).view(-1, 1, 1, 1).to(prev_sample.dtype)
        timestep_home_ne_end_mask = torch.ne(timestep_home, timestep_end).view(-1, 1, 1, 1, 1).to(prev_sample.dtype)
        prev_sample_curr, origin_curr = scheduler_pred_onestep(
            npred_model, prev_sample, scheduler, torch.clip(timestep_home, 0, 999), timestep_end,
            audio_cond_fea=audio_cond_fea, 
            face_musk_fea=face_musk_fea,
            guidance_scale=guidance_scale,
        )
        prev_sample = prev_sample_curr * timestep_home_ne_end_mask + prev_sample * (1 - timestep_home_ne_end_mask)
        origin = origin_curr * timestep_home_ne_end_mask + origin * (1 - timestep_home_ne_end_mask)

    return prev_sample, origin

def psuedo_velocity_wrt_noisy_and_timestep(noisy_images, noisy_images_pre, alphas_cumprod, timestep, timestep_prev):
    alpha_prod_t = get_alpha(alphas_cumprod, timestep).view(-1, 1, 1, 1, 1).detach()
    beta_prod_t = 1 - alpha_prod_t
    alpha_prod_t_prev = get_alpha(alphas_cumprod, timestep_prev).view(-1, 1, 1, 1, 1).detach()
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    a_s = (alpha_prod_t_prev ** (0.5)).to(noisy_images.dtype)
    a_t = (alpha_prod_t ** (0.5)).to(noisy_images.dtype)
    b_s = (beta_prod_t_prev ** (0.5)).to(noisy_images.dtype)
    b_t = (beta_prod_t ** (0.5)).to(noisy_images.dtype)

    psuedo_velocity = (noisy_images_pre - (
        a_s * a_t + b_s * b_t
    ) * noisy_images) / (
        b_s * a_t -  a_s * b_t
    )

    return psuedo_velocity

def origin_by_velocity_and_sample(velocity, noisy_images, alphas_cumprod, timestep):
    alpha_prod_t = get_alpha(alphas_cumprod, timestep).view(-1, 1, 1, 1, 1).detach()
    beta_prod_t = 1 - alpha_prod_t
    a_t = (alpha_prod_t ** (0.5)).to(noisy_images.dtype)
    b_t = (beta_prod_t ** (0.5)).to(noisy_images.dtype)

    pred_original_sample = a_t * noisy_images - b_t * velocity
    return pred_original_sample
    