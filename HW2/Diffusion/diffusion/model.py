import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import (
    cosine_beta_schedule,
    default,
    extract,
    unnormalize_to_zero_to_one,
)
from einops import rearrange, reduce
import numpy as np

class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        timesteps = 1000,
        sampling_timesteps = None,
        ddim_sampling_eta = 1.,
    ):
        super(DiffusionModel, self).__init__()

        self.model = model
        self.channels = self.model.channels
        self.device = torch.cuda.current_device()
        # self.device = "cpu"

        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1. - self.betas
        ##################################################################
        # TODO 3.1: Compute the cumulative products for current and
        # previous timesteps.
        ##################################################################
        self.alphas_cumprod = torch.cumprod(alphas, dim = 0)
        # self.alphas_cumprod_prev =  torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1.]).cuda(), self.alphas_cumprod[0:-1]))
        # self.alphas_cumprod_prev = torch.cat((torch.tensor([1.]).to("cpu"), self.alphas_cumprod[0:-1]))

        ##################################################################
        # TODO 3.1: Pre-compute values needed for forward process.
        ##################################################################
        # This is the coefficient of x_t when predicting x_0
        self.x_0_pred_coef_1 = 1/torch.sqrt(self.alphas_cumprod)
        # This is the coefficient of pred_noise when predicting x_0
        self.x_0_pred_coef_2 = -torch.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)

        ##################################################################
        # TODO 3.1: Compute the coefficients for the mean.
        ##################################################################
        # This is coefficient of x_0 in the DDPM section
        self.posterior_mean_coef1 = (torch.sqrt(self.alphas_cumprod_prev) * self.betas) / (1 - self.alphas_cumprod)
        # This is coefficient of x_t in the DDPM section
        self.posterior_mean_coef2 = (torch.sqrt(alphas) * (1 - self.alphas_cumprod_prev)) / (1 - self.alphas_cumprod)

        ##################################################################
        # TODO 3.1: Compute posterior variance.
        ##################################################################
        # Calculations for posterior q(x_{t-1} | x_t, x_0) in DDPM
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min =1e-20))

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    def get_posterior_parameters(self, x_0, x_t, t):
        # Compute the posterior mean and variance for x_{t-1}
        # using the coefficients, x_t, and x_0.
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_t, t):
        ##################################################################
        # TODO 3.1: Given a noised image x_t, predict x_0 and the additive
        # noise to predict the additive noise, use the denoising model.
        # Hint: You can use extract function from utils.py. See
        # get_posterior_parameters() for usage examples.
        ##################################################################
        pred_noise = self.model(x_t, t)
        x_0 = extract(self.x_0_pred_coef_1, t, x_t.shape) * x_t + extract(self.x_0_pred_coef_2, t, pred_noise.shape) * pred_noise
        x_0 = torch.clamp(x_0, -1, 1)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        return (pred_noise, x_0)

    @torch.no_grad()
    def predict_denoised_at_prev_timestep(self, x, t: int):
        ##################################################################
        # TODO 3.1: Given x at timestep t, predict the denoised image at
        # x_{t-1}, and return the predicted starting image.
        # noise to predict the additive noise, use the denoising model.
        # Hint: To do this, you will need a predicted x_0. You should've
        # already implemented a function to give you x_0 above!
        ##################################################################
        pred_noise, x_0 = self.model_predictions(x, t)

        posterior_mean, posterior_variance, posterior_log_variance_clipped = self.get_posterior_parameters(x_0, x, t)

        if torch.all(t > 0):
            z = torch.randn_like(posterior_mean)
        else:
            z = 0

        pred_img = posterior_mean +  z * torch.sqrt(posterior_variance)


        # x_0 = None
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        return pred_img, x_0

    @torch.no_grad()
    def sample_ddpm(self, shape, z):
        img = z
        for t in tqdm(range(self.num_timesteps-1, 0, -1)):
            batched_times = torch.full((img.shape[0],), t, device=self.device, dtype=torch.long)
            img, _ = self.predict_denoised_at_prev_timestep(img, batched_times)
        img = unnormalize_to_zero_to_one(img)
        return img

    def sample_times(self, total_timesteps, sampling_timesteps):
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        return list(reversed(times.int().tolist()))

    def get_time_pairs(self, times):
        return list(zip(times[:-1], times[1:]))

    def ddim_step(self, batch, device, tau_i, tau_isub1, img, model_predictions, alphas_cumprod, eta):
        ##################################################################
        # TODO 3.2: Compute the output image for a single step of the DDIM
        # sampling process.
        ##################################################################
        # tau_i = tau_i.repeat(batch).cuda().type(torch.long)
        # print("Alpha cumprod shape: ", alphas_cumprod.shape)

        tau_i = torch.full((batch,), tau_i, device=device,dtype = torch.long)
        tau_isub1 = torch.full((batch,), tau_isub1, device=device,dtype = torch.long)
        # tau_isub1[-1] = 0


        # print("Tau I: ", tau_i.shape)
        # print("Tau I Prev: ", tau_isub1.shape)

        # tau_isub1 = tau_isub1.repeat(batch).cuda().type(torch.long)

        # Step 1: Predict x_0 and the additive noise for tau_i
        eps_t, x_0 = model_predictions(img, tau_i)

        if torch.all(tau_i > 0 ):
            if torch.all(tau_isub1>0):
                z = torch.randn_like(img)
            else:
                return img, x_0


        # Step 2: Extract \alpha_{\tau_{i - 1}} and \alpha_{\tau_{i}}

        # pass
        # print("Alpha cumprod shape: ", alphas_cumprod.shape)
        alpha_tau = extract(alphas_cumprod, tau_i, img.shape)
        # print("Alpha cumprod shape: ", alphas_cumprod.shape)


        alpha_tau_prev = extract(alphas_cumprod, tau_isub1, img.shape)
        # print("Alpha tau prev shape: ", alpha_tau_prev.shape)
        # print("Alpha tau shape: ", alpha_tau.shape)


        # Step 3: Compute \sigma_{\tau_{i}}
        # pass
        beta_tau  = ((1 - alpha_tau_prev) / (1 - alpha_tau)) * extract(self.betas, tau_isub1, img.shape)
        # print("Beta tau shape: ", beta_tau.shape)

        
        sigma_tau = eta * beta_tau
        # Step 4: Compute the coefficient of \epsilon_{\tau_{i}}
        # pass
        coeff_eps = torch.sqrt(1 - alpha_tau_prev - sigma_tau)

        # Step 5: Sample from q(x_{\tau_{i - 1}} | x_{\tau_t}, x_0)
        # HINT: Use the reparameterization trick
        mean_tau = torch.sqrt(alpha_tau_prev) * x_0 + coeff_eps * eps_t

        # if torch.all(tau_i > 0):
        #     z = torch.randn_like(img)
        # else:
        #     return img, x_0

        # if torch.all(tau_i > 0):
        #     z = torch.randn_like(img)
        # else:
        #     z = 0

        # z = 1
        
        img = mean_tau + z * sigma_tau**0.5
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

        return img, x_0

    def sample_ddim(self, shape, z):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = self.sample_times(total_timesteps, sampling_timesteps)
        time_pairs = self.get_time_pairs(times)

        img = z
        for tau_i, tau_isub1 in tqdm(time_pairs, desc='sampling loop time step'):
            img, _ = self.ddim_step(batch, device, tau_i, tau_isub1, img, self.model_predictions, self.alphas_cumprod, eta)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = torch.randn(shape, device = self.betas.device)
        return sample_fn(shape, z)

    @torch.no_grad()
    def sample_given_z(self, z, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = z.reshape(shape)
        return sample_fn(shape, z)
