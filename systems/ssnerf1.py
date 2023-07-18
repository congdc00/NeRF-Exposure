import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss
import imageio
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug
import os
from loguru import logger
import models
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, SSIM
from tabulate import tabulate

@systems.register('ssnerf1-system')
class SSNeRF1System(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
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

            if self.dataset.directions.ndim == 3: # (H, W, 3) -> [800,800,3]
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3) 
                directions = self.dataset.directions[index, y, x]

            rays_o, rays_d = get_rays(directions, c2w) # Khởi tạo tia [8192,3], [8192,3]
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank) # Khởi tạo nhãn [8192, 3]
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
        
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1) #[8192, 6]

        # Chọn background
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
        '''
        args:
            - batch_idx: index của từng batch
        '''
        out = self(batch) #['comp_rgb', 'opacity', 'depth', 'rays_valid', 'num_samples', 'weights', 'points', 'intervals', 'ray_indices']
        loss = 0.
        """true
        {'index': tensor([0], device='cuda:0'), 
        'rays': tensor([[ 0.2047,  0.7417,  3.9570,  0.1800, -0.5525, -0.8138],
        [ 0.2047,  0.7417,  3.9570,  0.1793, -0.5525, -0.8140],
        [ 0.2047,  0.7417,  3.9570,  0.1785, -0.5524, -0.8142],
        ...,
        [ 0.2047,  0.7417,  3.9570, -0.2691,  0.2242, -0.9366],
        [ 0.2047,  0.7417,  3.9570, -0.2698,  0.2244, -0.9364],
        [ 0.2047,  0.7417,  3.9570, -0.2705,  0.2245, -0.9362]],
       device='cuda:0'), 'rgb': tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        ...,
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], device='cuda:0'), 
        'fg_mask': tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0')}
        """
        """{false
            batch['rays'] torch.Size([8192, 6]) 	 batch['rgb'] torch.Size([8192, 3]) 	 batch['fg_mask'] torch.Size([8192])"""
        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
        
        
        loss_rgb = F.smooth_l1_loss(out['comp_rgb'][out['rays_valid'][...,0]], batch['rgb'][out['rays_valid'][...,0]])
        
        ex_predict = out['bright_ness']
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ex_template = torch.ones(out['bright_ness'].shape).to(device)
        
        ex_delta_matrix = torch.pow(ex_predict - ex_template, 2)

        ex_delta = torch.mean(ex_delta_matrix)
        # print(f"loss_rgb {loss_rgb} & ex {ex_delta}")
        k = 0.001
        total_loss = loss_rgb + k*ex_delta
        # print(f"total_loss {total_loss}")

        self.log('train/loss_rgb', total_loss)
        loss += total_loss * self.C(self.config.system.loss.lambda_rgb)
        # print(f"loss_rgb 2 {loss_rgb}")
        # distortion loss proposed in MipNeRF360
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)

        losses_model_reg = self.model.regularizations(out)
        # print(f"losses_model_reg {losses_model_reg}")

        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)

            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))
        
        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        logger.info(f"validation_step")
    
        try:
            out = self(batch) 
        except:
            return {
                'psnr': 0.0,
                # 'ssim': ssim,
                'index': batch['index']
            }
           
        image_origin = batch['rgb'] 
        image_predict = out['comp_rgb']
        color_predict = out["real_rgb"]
        density_predict = out['depth']
        expore_sure_predict = out['bright_ness'][0].item()

        psnr = self.criterions['psnr'](color_predict.to(image_origin), image_origin)

        W, H = self.dataset.img_wh
        torch.save(out['theta'], "theta.pt")
        torch.save(out['positions'], "positions.pt")

        # Save difference brightness 
        file_path = f"./log_bright_ness.txt"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
        else:
            content = ""
        with open(file_path, 'w',newline="\n") as file:
            content += str(expore_sure_predict) +"\n"
            file.write(str(content))

        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': image_predict.view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': color_predict.view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': density_predict.view(H, W), 'kwargs': {}}
        ])

        return {
            'psnr': psnr,
            # 'ssim': ssim,
            'index': batch['index']
        }
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set_psnr = {}
            # out_set_ssim = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set_psnr[step_out['index'].item()] = {'psnr': step_out['psnr']}
                    # out_set_ssim[step_out['index'].item()] = {'ssim': step_out['ssim']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set_psnr[index[0].item()] = {'psnr': step_out['psnr'][oi]}
                        # out_set_ssim[index[0].item()] = {'ssim': step_out['ssim'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set_psnr.values()]))
            # ssim = torch.mean(torch.stack([o['ssim'] for o in out_set_ssim.values()]))

            file_path = "./log_psnr.txt"
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    content = file.read()
            else:
                content = ""
            with open(file_path, 'w') as file:
                content = content + str(round(psnr.tolist(),2)) + "\n"
                file.write(content)

            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)         
            # self.log('val/ssim', ssim, prog_bar=True, rank_zero_only=True)         

    def test_step(self, batch, batch_idx):  
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=30
            )
            
            self.export()

    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )    
