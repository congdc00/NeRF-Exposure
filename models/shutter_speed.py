# Render color

import torch
import torch.nn as nn

import models
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp
from systems.utils import update_module_step


@models.register('volume-brightness')
class VolumeBrightness(nn.Module):
    def __init__(self, config):
        super(VolumeBrightness, self).__init__()
        self.config = config
        self.n_ori_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 1

        encoding = get_encoding(self.n_ori_dims, self.config.dir_encoding_config)

        self.n_input_dims = encoding.n_output_dims #+ self.config.input_feature_dim #16 +16
        self.encoding = encoding
        self.network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)   
    
    def forward(self,is_freeze, origins, *args):
        """
        Args:
            origins torch.Size([97790, 3])
        Result:
            brightness: torch.Size([97790, 1])
        """
        origins = (origins + 1.) / 2. # (-1, 1) => (0, 1)
        origins_embd = self.encoding(origins.view(-1, self.n_ori_dims)) # origins_embd torch.Size([97790, 16])
        
        network_inp = torch.cat([origins_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1) #([97790, 32])

        #freeze
        print(f"frezee {self.network.named_parameters()}")
        for param in self.network.parameters():
            param.requires_grad = False
            print(f"param {param.requires_grad}")

        brightness = self.network(network_inp).view(*origins.shape[:-1], self.n_output_dims).float() #*features.shape[:-1] => [97790,]

        # Dung cho neus
        if 'brightness_activation' in self.config:
            brightness = get_activation(self.config.brightness_activation)(brightness)
        return brightness

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self, out):
        return {}

