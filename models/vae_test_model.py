from typing import *
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import numpy as np
# %matplotlib nbagg
# %matplotlib inline
import pandas as pd

import math 
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution
from torch.distributions import Bernoulli
from torch.distributions import Normal
from collections import defaultdict


class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()
        
    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()
        
    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()
        
    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        #return self.Normal(self.mu, self.sigma * self.sample_epsilon())  # <- your code
        #z = torch.normal(self.mu, (self.sigma * self.sample_epsilon()))
        z = self.mu + self.sigma * self.sample_epsilon()
        return z

    def log_prob(self, z:Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        #return np.log(self.rsample()).sum() # <- your code
        log_scale = torch.log(self.sigma)
        return -((z - self.mu)**2 / (2*self.sigma**2)) - log_scale - math.log(math.sqrt(2*math.pi))
        #return self.log_prob(z)

class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, input_shape:torch.Size, latent_features:int) -> None:
        super(VariationalAutoencoder, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)

        

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,
                                out_channels=64,
                                kernel_size=8,
                                stride=2,
                                padding=3),
            nn.ReLU(),
            
            #nn.Conv2d(in_channels=64,
            #                    out_channels=32,
            #                    kernel_size=2,
            #                    stride=2,
            #                    padding=0),
            #nn.ReLU(),
            
            nn.Conv2d(in_channels=64,
                                out_channels=3,
                                kernel_size=6,
                                stride=2,
                                padding=3),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=2,
                               stride=1,
                               padding=0)
        )

        
        
        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3,
                                out_channels=32,
                                kernel_size=6,
                                stride=2,
                                padding=2),
            nn.ReLU(),
            
            #nn.ConvTranspose2d(in_channels=32,
            #                    out_channels=64,
            #                    kernel_size=2,
            #                    stride=2,
            #                    padding=0),
            #nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=32,
                                out_channels=3,
                                kernel_size=8,
                                stride=2,
                                padding=3,output_padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=2,
                               stride=1,
                               padding=0)
        )

        # Prior for SR
        self.prior_nn = nn.Sequential(
            nn.Conv2d(in_channels=3,
                                out_channels=64,
                                kernel_size=9,
                                stride=1,
                                padding=4),
            nn.ReLU(),
            
            #nn.Conv2d(in_channels=64,
            #                    out_channels=32,
            #                    kernel_size=1,
            #                    stride=1,
            #                    padding=0),
            #nn.ReLU(),
            
            nn.Conv2d(in_channels=64,
                                out_channels=3,
                                kernel_size=5,
                                stride=1,
                                padding=2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        )
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.encoder(x)

        mu, log_sigma =  h_x.chunk(2, dim=1)
        
        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def prior_sr(self, y:Tensor) -> Distribution:
        h_y = self.prior_nn(y)
        mu, log_sigma = h_y.chunk(2, dim=1)
        log_sigma2 = torch.zeros(torch.Size(log_sigma.shape))
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        log_sigma2 = log_sigma2.to(device)
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma2)

    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        px_logits = self.decoder(z)
        px_logits = px_logits.view(-1, *self.input_shape) # reshape the output to input_shape number of columns (rows are unspecified)
        return Bernoulli(logits=px_logits)

    def observation_model_normal(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        h_z = self.decoder(z)

        mu, log_sigma = h_z.chunk(2, dim =1)

        mu = mu.view(-1, *self.input_shape)
        log_sigma = log_sigma.view(-1, *self.input_shape)

        #sampled = sampled.view(-1,*self.input_shape)
        
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
        

    def forward(self, x, y) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""

        # flatten the input
        #x = x.view(x.size(0), -1)
        #y = y.view(y.size(0), -1)

        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        #pz = self.prior(batch_size=x.size(0))

        # p(z|y)
        pz= self.prior_sr(y)
        zy = pz.rsample()

        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model_normal(zy+z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, y):
        """sample z~p(z) and return p(x|z)"""
        
        #y = y.view(y.size(0), -1)

        # define the prior p(z)
        pz = self.prior_sr(y)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model_normal(z)
        
        return {'px': px, 'pz': pz, 'z': z}



# latent_features = 2
# vae = VariationalAutoencoder(images[0].shape, latent_features)
# print(vae)

def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class VariationalInference(nn.Module):
    def __init__(self, beta:float=0.95):
        super().__init__()
        self.beta = beta
        
    def forward(self, model:nn.Module, x:Tensor, y:Tensor) -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = model(y,x)
        
        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]
        
        # evaluate log probabilities
        log_px = reduce(px.log_prob(y))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))
        
        # compute the ELBO with and without the beta parameter: 
        # `L^\beta = E_q [ log p(x|z) - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl = log_qz - log_pz

        # elbo = torch.mean(log_px) - kl # <- your code here
        # beta_elbo = torch.mean(log_px) - self.beta* kl # <- your code here
        
        elbo = log_px - kl # <- your code here
        beta_elbo = log_px - self.beta* kl # <- your code here
        
        # loss
        loss = -beta_elbo.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}
            
        return loss, diagnostics, outputs


def test_vae_train_step(model, device, train_loader, optimizer, loss_func):
  model.train()
  training_epoch_data = defaultdict(list)
  for data, target in train_loader:
    optimizer.zero_grad()
    data = data.to(device)
    target = target.to(device)

    loss, diagnostics, outputs = loss_func(model, data, target)

    loss.backward()
    optimizer.step()

    for k, v in diagnostics.items():
      training_data[k] += [v.mean().item()]
  for k, v in training_epoch_data.items():
    training_data[k] += [np.mean(training_epoch_data[k])]

  return {"training elbo": training_data['elbo'][-1],
          "training kl": training_data['kl'][-1]}

def test_vae_test_step(model, device, test_loader, loss_func):
  model.eval()
  training_epoch_data = defaultdict(list)
  with torch.no_grad():
    for data,target in test_loader:
      data = data.to(device)
      target = target.to(device)

      loss, diagnostics, outputs = lossfunc(model, data, target)

      for k, v in diagnostics.items():
        training_data[k] += [v.mean().item()]
    for k, v in training_epoch_data.items():
      training_data[k] += [np.mean(training_epoch_data[k])]

  return {"training elbo": training_data['elbo'][-1],
          "training kl": training_data['kl'][-1]}
