from abc import ABC, abstractmethod
from diffusers import DDPMScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler
import torch
from diffusers.utils.torch_utils import randn_tensor
from IPython import embed
import numpy as np

class MySchedulers(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def pred_prev(self,
        noisy_images, noisy_residual, timestep, timestep_prev
    ):
        raise NotImplementedError("not implemented")

def get_alpha(alphas_cumprod, timestep):
    timestep_lt_zero_mask = torch.lt(timestep, 0).to(alphas_cumprod.dtype)
    timestep_gt_999_mask = torch.gt(timestep, 999).to(alphas_cumprod.dtype)
    normal_alpha = alphas_cumprod[torch.clip(timestep, 0, 999)]
    one_alpha = torch.ones_like(normal_alpha).to(normal_alpha.dtype).to(normal_alpha.dtype) 
    zero_alpha = torch.zeros_like(normal_alpha).to(normal_alpha.dtype).to(normal_alpha.dtype) 
    return (normal_alpha * (1 - timestep_lt_zero_mask) + one_alpha * timestep_lt_zero_mask) * (1 - timestep_gt_999_mask) + zero_alpha * timestep_gt_999_mask

class MyDDIM(MySchedulers):
    def __init__(self, ddpm_or_ddim_scheduler, normnoise=False) -> None:
        super(MyDDIM, self).__init__()
        assert isinstance(ddpm_or_ddim_scheduler, DDPMScheduler) or isinstance(ddpm_or_ddim_scheduler, DDIMScheduler)
        self.alphas_cumprod = ddpm_or_ddim_scheduler.alphas_cumprod
        self.normnoise = normnoise
        self.prediction_type = ddpm_or_ddim_scheduler.config.prediction_type

    def pred_prev(self,
        noisy_images, model_output, timestep, timestep_prev
    ):
        torch_dtype = model_output.dtype

        #noisy_images = noisy_images.to(torch.float32)
        #noisy_residual = noisy_residual.to(torch.float32)

        #print(noisy_residual.std())
        
        #print(noisy_residual.std())

        alphas_cumprod = self.alphas_cumprod.to(noisy_images.device) #.to(noisy_images.dtype)
        alpha_prod_t = get_alpha(alphas_cumprod, timestep).view(-1, 1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        if self.prediction_type == "epsilon":
            if self.normnoise:
                model_output = model_output / (torch.std(model_output, dim=(1,2,3), keepdim=True) + 0.0001)

            pred_original_sample = (noisy_images - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (noisy_images - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * noisy_images - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * noisy_images

        alpha_prod_t_prev = get_alpha(alphas_cumprod, timestep_prev).view(-1, 1, 1, 1, 1)
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (alpha_prod_t_prev ** (0.5)) * pred_original_sample + beta_prod_t_prev ** (0.5) * pred_epsilon

        return prev_sample.to(torch_dtype), pred_original_sample.to(torch_dtype)

    def pred_prev_train(self,
        noisy_images, noisy_residual, timestep, timestep_prev
    ):
        torch_dtype = noisy_residual.dtype

        #noisy_images = noisy_images.to(torch.float32)
        #noisy_residual = noisy_residual.to(torch.float32)

        print(noisy_residual.std())
        if self.normnoise:
            noisy_residual = noisy_residual / (torch.std(noisy_residual, dim=(1,2,3), keepdim=True) + 0.0001)

        print(noisy_residual.std())

        alphas_cumprod = self.alphas_cumprod.to(noisy_images.device) #.to(noisy_images.dtype)
        alpha_prod_t = get_alpha(alphas_cumprod, timestep).view(-1, 1, 1, 1, 1).to(torch_dtype).detach()
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (noisy_images - beta_prod_t ** (0.5) * noisy_residual) / alpha_prod_t ** (0.5)
        pred_epsilon = noisy_residual

        alpha_prod_t_prev = get_alpha(alphas_cumprod, timestep_prev).view(-1, 1, 1, 1, 1).to(torch_dtype).detach()
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = beta_prod_t_prev ** (0.5) * pred_epsilon
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (alpha_prod_t_prev ** (0.5)) * pred_original_sample + pred_sample_direction

        #return prev_sample.to(torch_dtype), pred_original_sample.to(torch_dtype)
        #embed()
        assert prev_sample.dtype == torch_dtype
        assert pred_original_sample.dtype ==  torch_dtype

        return prev_sample, pred_original_sample
    
    
    def add_more_noise(self, noisy_latents, noise, timestep, timestep_more):
        alphas_cumprod = self.alphas_cumprod.to(noisy_latents.device)
        alpha_prod_t = get_alpha(alphas_cumprod, timestep).view(-1, 1, 1, 1, 1)
        alpha_prod_t_more = get_alpha(alphas_cumprod, timestep_more).view(-1, 1, 1, 1, 1)

        sqrt_alpha_prod = alpha_prod_t ** (0.5)
        sqrt_one_minus_alpha_prod = (1 - alpha_prod_t) ** (0.5)
        sqrt_alpha_prod_more = alpha_prod_t_more ** (0.5)
        sqrt_one_minus_alpha_prod_more = (1 - alpha_prod_t_more) ** (0.5)

        #noisy_latents = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise_p
        #noisy_latents * (sqrt_alpha_prod_more / sqrt_alpha_prod) 
        #    = sqrt_alpha_prod_more * original_samples + sqrt_one_minus_alpha_prod * noise_p * (sqrt_alpha_prod_more / sqrt_alpha_prod) 

        noise_coe = ((sqrt_one_minus_alpha_prod_more ** 2) - (sqrt_one_minus_alpha_prod * sqrt_alpha_prod_more / sqrt_alpha_prod) ** 2) ** (0.5)

        return noisy_latents * (sqrt_alpha_prod_more / sqrt_alpha_prod) + noise_coe * noise
    

def get_sigma(sigmas, timestep):
    timestep_lt_zero_mask = torch.lt(timestep, 0).to(sigmas.dtype)
    normal_sigma = sigmas[torch.clip(timestep, 0)]
    zero_sigma = torch.zeros_like(normal_sigma).to(normal_sigma.dtype).to(normal_sigma.device) 
    return normal_sigma * (1 - timestep_lt_zero_mask) + zero_sigma * timestep_lt_zero_mask

class MyEulerA(MySchedulers):
    def __init__(self, eulera_scheduler, normnoise=False) -> None:
        super(MyEulerA, self).__init__()
        assert isinstance(eulera_scheduler, EulerAncestralDiscreteScheduler)
        self.sigmas = ((1 - eulera_scheduler.alphas_cumprod) / eulera_scheduler.alphas_cumprod) ** 0.5
        assert len(self.sigmas) == 1000
        self.generator = None
        self.normnoise = normnoise

    def pred_prev(self,
        noisy_images, noisy_residual, timestep, timestep_prev
    ):
        torch_dtype = noisy_residual.dtype
        noisy_images = noisy_images.to(torch.float32)
        noisy_residual = noisy_residual.to(torch.float32)

        if self.normnoise:
            noisy_residual = noisy_residual / (torch.std(noisy_residual, dim=(1,2,3), keepdim=True) + 0.0001)

        sigmas = self.sigmas.to(noisy_images.device)

        sigma_from = get_sigma(sigmas, timestep).view(-1, 1, 1, 1, 1)
        sigma_to = get_sigma(sigmas, timestep_prev).view(-1, 1, 1, 1, 1)
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
        
        sample = noisy_images * ((sigma_from**2 + 1) ** 0.5)

        pred_original_sample = sample - sigma_from * noisy_residual
        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_from

        dt = sigma_down - sigma_from

        prev_sample = sample + derivative * dt

        device = noisy_residual.device
        noise = randn_tensor(noisy_residual.shape, dtype=torch_dtype, device=device, generator=self.generator).to(noisy_residual.dtype)

        #embed()
        prev_sample = prev_sample + noise * sigma_up
        #print(sigma_up, ((sigma_to**2 + 1) ** 0.5))

        
        prev_sample = prev_sample / ((sigma_to**2 + 1) ** 0.5)

        #embed()

        return prev_sample.to(torch_dtype), pred_original_sample.to(torch_dtype)

    
if __name__ == "__main__":
    a = MyDDIM()
    
    
        
        