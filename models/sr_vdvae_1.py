import torch
from torch import nn
from torch.nn import functional as F
from vdvae.vae_helpers import HModule, get_1x1, get_3x3, DmolNet, draw_gaussian_diag_samples, gaussian_analytical_kl
from collections import defaultdict
import numpy as np
import itertools

from vdvae.train import *
from vdvae.hps import *


class SRVAE(HModule):
	def build(self):
        a=1	##################################################################################

	def build_model():
		self.H1, self.logprint1 = set_up_hyperparams()
		self.H1.image_size = 256
		self.H1.image_channels = 3
		self.vae, self.ema_vae = load_vaes(self.H1, self.logprint1)

	def build_parcial_model():
		self.H2, self.logprint2 = set_up_hyperparams()
		self.H2.image_size = 64
		self.H2.image_channels = 3
		n_batch = self.H2.n_batch
		self.H2.update(i64)
		self.H2.n_batch = n_batch
		self.vae_sr, self.ema_vae_sr = load_vaes(self.H2, self.logprint2)

	def load_saved_models(model_path, model_path_ema, model_path_sr, model_path_ema_sr):
		
		model_state_dict_save = {k:v for k,v in torch.load(model_path, map_location="cuda").items()}
		model_state_dict = self.vae.state_dict()
		model_state_dict.update(model_state_dict_save)
		self.vae.load_state_dict(model_state_dict)

		model_state_dict_save = {k:v for k,v in torch.load(model_path_ema, map_location="cuda").items()}
		model_state_dict = self.ema_vae.state_dict()
		model_state_dict.update(model_state_dict_save)
		self.ema_vae.load_state_dict(model_state_dict)


		model_state_dict_save = {k:v for k,v in torch.load(model_path_sr, map_location="cuda").items()}
		model_state_dict = self.vae_sr.state_dict()
		model_state_dict.update(model_state_dict_save)
		self.vae_sr.load_state_dict(model_state_dict)

		model_state_dict_save = {k:v for k,v in torch.load(model_path_ema_sr, map_location="cuda").items()}
		model_state_dict = self.ema_vae_sr.state_dict()
		model_state_dict.update(model_state_dict_save)
		self.ema_vae_sr.load_state_dict(model_state_dict)


    def forward(self, x, x_target):
    	a = 1 ##################################################################################
        
    def forward_sr_sample(self, x, n_batch):
    	activations_sr = self.ema_vae_sr.forward_sr_activations(x)
    	output = self.ema_vae.forward_sr_sample(n_batch, activations_sr)
    	return output

