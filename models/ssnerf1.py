import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from models.utils import chunk_batch
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching, render_weight_from_density, accumulate_along_rays


@models.register('ssnerf1')
class SSNeRF1Model(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry) # density
        self.texture = models.make(self.config.texture.name, self.config.texture) # radiant
        self.shutter_speed = models.make(self.config.shutter_speed.name, self.config.shutter_speed) # shutter_speed

        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))

        if self.config.learned_background:
            self.occupancy_grid_res = 256
            self.near_plane, self.far_plane = 0.2, 1e4
            self.cone_angle = 10**(math.log10(self.far_plane) / self.config.num_samples_per_ray) - 1. # approximate
            self.render_step_size = 0.01 # render_step_size = max(distance_to_camera * self.cone_angle, self.render_step_size)
            self.contraction_type = ContractionType.UN_BOUNDED_SPHERE
        else:
            self.occupancy_grid_res = 128
            self.near_plane, self.far_plane = None, None
            self.cone_angle = 0.0
            self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
            self.contraction_type = ContractionType.AABB

        self.geometry.contraction_type = self.contraction_type

        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=self.occupancy_grid_res,
                contraction_type=self.contraction_type
            )
        self.randomized = self.config.randomized
        self.background_color = None
    
    def update_step(self, epoch, global_step):
        
        # Lan truyen nguoc
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)
        update_module_step(self.shutter_speed, epoch, global_step)

        def occ_eval_fn(x):
            density, _ = self.geometry(x)
            return density[...,None] * self.render_step_size
        
        if self.training and self.config.grid_prune:
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn)

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def forward_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3) -> [8192, 3], [8192, 3]

        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, _ = self.geometry(positions)
            return density[...,None]

        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=None if self.config.learned_background else self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                sigma_fn=sigma_fn,
                near_plane=self.near_plane, far_plane=self.far_plane,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=self.cone_angle, 
                alpha_thre=0.0
            )   
            
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints  # positions [N_rays, 3]  == R_sample
        intervals = t_ends - t_starts

        # Step 1: Predict colour point
        # Forward
        density, cor_feature = self.geometry(positions) # Dự đoán mật độ thể tích => density [N_rays];cor_feature [N_rays, 16]16 là số chiều được mã hoá ra
        rgb = self.texture(cor_feature, t_dirs) # Dự đoán ra màu sắc
        bright_ness = self.shutter_speed(t_origins)

        # network_inp torch.Size([97790, 32])
        # density torch.Size([97790])
        # cor_feature torch.Size([97790, 16])
        # rgb torch.Size([97790, 3])
        # dir_feature torch.Size([97790, 16])
        # bright_ness torch.Size([97790, 1])

        # Step 2: Rendering 
        # print(f"shape {bright_ness.shape}")

        # fake_brightness = torch.ones_like(bright_ness)
        # print(f"fake_brightness {fake_brightness.shape}")
        # new_rgb = rgb*fake_brightness

        new_rgb = rgb*bright_ness

        # Trọng số
        weights = render_weight_from_density(t_starts, t_ends, density[...,None], ray_indices=ray_indices, n_rays=n_rays) #([Num_points, 1])
        
        # Độ mờ
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        
        # Màu sắc dự đoán ra
        real_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays) 
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=new_rgb, n_rays=n_rays) #([Num_points, 1])
        # depth
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)    

        #Độ sáng
        comp_rgb = comp_rgb + self.background_color * (1.0 - opacity) 
        real_rgb = real_rgb + self.background_color * (1.0 - opacity) 
        # print(f"-------------{bright_ness[12]}----------")
        
        # Export 
        out = {
            'comp_rgb': comp_rgb,
            # 'bright_ness':bright_ness,
            "real_rgb": real_rgb,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device),
            'theta': weights,
            'positions':positions
        }
        
        if self.training:
            out.update({
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': intervals.view(-1),
                'ray_indices': ray_indices.view(-1)
            })
        
        return out

    def forward(self, rays):
        if self.training:
            out = self.forward_(rays)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, True, rays)
            
        return {**out,}

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        losses.update(self.shutter_speed.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank))
            viewdirs = torch.zeros(feature.shape[0], 3).to(feature)
            viewdirs[...,2] = -1. # set the viewing directions to be -z (looking down)
            rgb = self.texture(feature, viewdirs).clamp(0,1)
            mesh['v_rgb'] = rgb.cpu()
        return mesh
