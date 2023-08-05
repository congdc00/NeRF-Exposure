import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss
from skimage.metrics import structural_similarity as ssim
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug
import os
import models
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR
from loguru import logger
import numpy as np
@systems.register('nerf-system')
class NeRFSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    save_PSNR = {}
    save_std_PSNR = {}
    save_std_SSIM = {}
    save_SSIM = {}
    save_std_pe = {}
    save_pe = {}

    def prepare(self):
        self.criterions = {
            'psnr': PSNR(),
            'ssim': ssim
        }
        self.train_num_samples = self.config.model.train_num_rays * self.config.model.num_samples_per_ray
        self.train_num_rays = self.config.model.train_num_rays

    def forward(self, batch):
        return self.model(batch['rays'])
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        if stage in ['train']:
            
            c2w = self.dataset.all_c2w[index] # Lấy thông tin file transform
            
            # Khởi tạo meshgrid
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )

            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]

            
            rays_o, rays_d = get_rays(directions, c2w) # Khởi tạo tia
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank) # Khởi tạo nhãn
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
        else:
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)
        
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])        
        
        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask
        })
    
    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
        
        loss_rgb = F.smooth_l1_loss(out['comp_rgb'][out['rays_valid'][...,0]], batch['rgb'][out['rays_valid'][...,0]])
        self.log('train/loss_rgb', loss_rgb)
        loss += loss_rgb * self.C(self.config.system.loss.lambda_rgb)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss, but still slows down training by ~30%
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))
        
        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        
        try:
            print(f"\n batch  rgb: {batch['rgb'].shape} ")
            out = self(batch) 
        except:
            print(f"\n batch  rgb: {batch['rgb'].shape} ")
            self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
                {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            ]) 
            return {
                'psnr': 0.0,
                'ssim': 0.0,
                'index': batch['index']
            }
        W, H = self.dataset.img_wh
        image_origin = batch['rgb'] 
        image_predict = out['comp_rgb']
        psnr = self.criterions['psnr'](out['comp_rgb'].to(batch['rgb']), batch['rgb'])

        image_array1 = image_predict.view(H, W, 3).cpu().numpy()
        image_array2 = image_origin.view(H, W, 3).cpu().numpy()
        ssim = self.criterions['ssim'](image_array1, image_array2,multichannel=True, full=True)
        # mask_object = batch['fg_mask'].view(-1, 1)
        # rgb_non_bg= (batch['rgb']*mask_object)
        # psnr_object = self.criterions['psnr'](out['comp_rgb'].to(batch['rgb'])*mask_object, rgb_non_bg)
        
        # mask_bg = torch.ones_like(mask_object) - mask_object
        # background_rgb = (batch['rgb']*mask_bg)
        # psnr_background = self.criterions['psnr'](out['comp_rgb'].to(batch['rgb'])*mask_bg, background_rgb)
        # print(f"\n -------- psnr object {psnr_object} and psnr background {psnr_background}")
        

        
        if batch_idx == 0:
            self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
                {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            ])  
            torch.save(out['theta'], "theta.pt")
            torch.save(out['positions'], "positions.pt")
        return {
            'psnr': psnr,
            'ssim': ssim,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set_psnr = {}
            out_set_ssim = {}
            check_ssim = {}
            num_imgs = 0
            num_all_imgs = 0
            for step_out in out:
                num_all_imgs += 1
                if int(step_out['index'].item()) == 0:
                    print(f"\n\nr_{step_out['index'].item()}.png with psnr {step_out['psnr'].item()}")
                # DP
                if step_out['index'].ndim == 1:
                    if int(step_out['psnr']) != 0.0:
                        out_set_psnr[step_out['index'].item()] = {'psnr': step_out['psnr']}
                        out_set_ssim[step_out['index'].item()] = {'ssim': torch.tensor(step_out['ssim'])}
                        num_imgs += 1
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        if int(step_out['psnr'][oi]) != 0.0:
                            out_set_psnr[index[0].item()] = {'psnr': step_out['psnr'][oi]}
                            out_set_ssim[index[0].item()] = {'ssim': torch.tensor(step_out['ssim'][oi])}
                            check_ssim[f"r_{step_out['index'].item()}.png"] = {torch.tensor(step_out['ssim'][oi])}
                            num_imgs += 1
            
            if num_imgs == 0:
                logger.error(f"Validation False")
                psnr = 0
                ssim_score = 0
                psnr_standard = 0
                ssim_standard = 0
            else: 

                list_psnr = torch.stack([o['psnr'] for o in out_set_psnr.values()])
                psnr = torch.mean(list_psnr) 
                psnr_standard= torch.std(list_psnr) 

                list_ssim = torch.stack([o['ssim'] for o in out_set_ssim.values()])
                ssim_score = torch.mean(list_ssim) 
                ssim_standard= torch.std(list_ssim) 

                log_text = f"Validation on {num_imgs}/{num_all_imgs} images  -- SSIM {ssim_score} -- std PSNR: {psnr_standard} -- std SSIM: {ssim_standard}"
                for key, value in check_ssim.items():
                     print(f"Name dataset: {key} \t SSIM: {value}")
                if num_imgs<num_all_imgs:
                    logger.warning(log_text)
                else:
                    logger.info(log_text)

            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True, sync_dist=True)      
    def test_step(self, batch, batch_idx):  
        # try:
        #     out = self(batch) 
        # except:
        #     logger.warning(f"Validation Failed")
        #     return {
        #         'psnr': 0.0,
        #         # 'ssim': ssim,
        #         'index': batch['index']}
        # psnr = self.criterions['psnr'](out['comp_rgb'].to(batch['rgb']), batch['rgb'])
        # W, H = self.dataset.img_wh
        # if batch_idx == 0:
        #     self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
        #         {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #         {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #         {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}}
        #     ])
        # return {
        #     'psnr': psnr,
        #     'index': batch['index']
        # }   
        pass   
    
    def test_epoch_end(self, out):
        pass
        # out = self.all_gather(out)
        # if self.trainer.is_global_zero:
        #     out_set = {}
        #     for step_out in out:
        #         # DP
        #         if step_out['index'].ndim == 1:
        #             out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
        #         # DDP
        #         else:
        #             for oi, index in enumerate(step_out['index']):
        #                 out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
        #     psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
        #     self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    

        #     self.save_img_sequence(
        #         f"it{self.global_step}-test",
        #         f"it{self.global_step}-test",
        #         '(\d+)\.png',
        #         save_format='mp4',
        #         fps=30
        #     )
            
        #     self.export()

    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )    
