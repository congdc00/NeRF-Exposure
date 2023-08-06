import os
import json
import math
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF
import imageio
import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank

from loguru import logger

class BlenderDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True

        with open(os.path.join(self.config.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)

        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = 800, 800

        if 'img_wh' in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.config:
            w, h = W // self.config.img_downscale, H // self.config.img_downscale
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")
        
        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)

        self.near, self.far = self.config.near_plane, self.config.far_plane

        self.focal = 0.5 * w / math.tan(0.5 * meta['camera_angle_x']) # scaled focal length

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.w, self.h, self.focal, self.focal, self.w//2, self.h//2).to(self.rank) # (h, w, 3)           

        self.all_c2w, self.all_images, self.all_fg_masks, self.all_factor = [], [], [], []

        for i, frame in enumerate(meta['frames']):
            c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
            self.all_c2w.append(c2w)

            img_path = os.path.join(self.config.root_dir, f"{frame['file_path']}.png")
            
            ## Dang exr
            # img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH  | cv2.IMREAD_UNCHANGED)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = Image.fromarray(color_coverted)

            # Dang png 1
            # img = imageio.imread(img_path)

            ## Dang png 2
            
            img = Image.open(img_path)
            try:
                # print(f"Thay đổi độ sáng ảnh từ 1 -> {frame['factor']}")
                exposure_factor = float(frame['factor'])
                img.convert("F")
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(exposure_factor)
                exposure_factor = torch.Tensor([exposure_factor])
                
                
            except:
                k = float(1.0)
                exposure_factor = torch.Tensor([k])
            finally:
                self.all_factor.append(exposure_factor)
                

            img = img.resize(self.img_wh, Image.BICUBIC)
            
            img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
            self.all_fg_masks.append(img[..., -1]) # (h, w)
            self.all_images.append(img[...,:3])
        
        self.all_c2w, self.all_images, self.all_fg_masks, self.all_factor = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank), \
            torch.stack(self.all_factor, dim=0).float().to(self.rank)

class BlenderDataset(Dataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class BlenderIterableDataset(IterableDataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('blender')
class BlenderDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = BlenderIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = BlenderDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = BlenderDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = BlenderDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
