import numpy as np 
import os 
import torch 
import cv2
import json, hashlib
import torch.distributed as dist
import math

from .transform2D_utils import crop_image_depth_and_intrinsic_by_pp, resize_image_depth_and_intrinsic

def string_to_filename(s: str) -> str:
    # Use SHA-256 and truncate for filename
    h = hashlib.sha256(s.encode()).hexdigest()
    return h[:16]  # 16 hex chars = 64 bits, usually enough to avoid collisions

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, name,
                 root, img_size=None, aspect_ratio=None, 
                 sampled_file=None, sample_extracted=False,
                 example_num=None, load_depths=True, use_hash=True,
                 **kwargs):
        super().__init__()
        self.root = root
        self.name = name
        self.img_size, self.aspect_ratio = img_size, aspect_ratio
        self.load_depths = load_depths
        self.use_hash = use_hash
        # if img_size is not None and aspect_ratio is not None:
        #     self.target_shape = np.array([round(img_size * aspect_ratio), img_size])
        # else:
        #     self.target_shape = None  # Keep original size
        self.sample_extracted = sample_extracted
        if sample_extracted == False:
            assert sampled_file is not None, "sampled_file should be provided if sample_extracted is False"
            self.sampled_list = json.load(open(sampled_file, 'r')) #[[scene_name,[img_name1,...]]]
        else:
            self.sampled_list = []
            for scene_name in os.listdir(self.root):
                img_list = sorted(os.listdir(os.path.join(self.root, scene_name, 'images')))
                self.sampled_list.append([scene_name, img_list])
        if example_num is not None and example_num < len(self.sampled_list):
            selected_indices = np.linspace(0, len(self.sampled_list)-1, example_num).astype(int).tolist()
            self.sampled_list = [self.sampled_list[i] for i in selected_indices]
        
    def __len__(self):
        return len(self.sampled_list)

    def read_img_pose(self, scene_name, img_name):
        raise NotImplementedError

    def read_scene_pose(self, scene_name):
        raise NotImplementedError
    
    def read_img_rgb(self, scene_name, img_name):
        raise NotImplementedError

    def read_img_depth(self, scene_name, img_name):
        raise NotImplementedError

    def preprocess_2D_simple(self, target_shape, image, depth=None, K=None, intr_convention='opencv'):
        image = image.copy() #(H,W,3) uint8
        K = K.copy()
        image, depth, K = resize_image_depth_and_intrinsic(image, depth, K, target_shape=target_shape, intr_convention=intr_convention, safe_bound=0)
        return image, depth, K
    def preprocess_2D_vggt(self, target_shape, image, depth=None, K=None, intr_convention='opencv'):
        #TODO support rotation. This also involves extrinsics.
        image = image.copy() #(H,W,3) uint8
        K = K.copy()

        # 1. Crop images so that principal point is at the center. (!target_shape=image.shape)
        image, depth, K = crop_image_depth_and_intrinsic_by_pp(image, depth, K, target_shape=image.shape, strict=False, intr_convention=intr_convention)
        
        # 2. Resize images if needed 
        safe_bound = 4 # target_shape = target_shape + safe_bound, resize_scale = max(target_shape + safe_bound) / original_size)
        image, depth, K = resize_image_depth_and_intrinsic(image, depth, K, target_shape=target_shape, intr_convention=intr_convention, safe_bound=safe_bound)
        # The output image may have width < target_width or height < target_height. Up to this step, we don't apply any zero-padding 
        # In fact, the output image at this step should keep center-pp 

        # 3. Crop or pad to the exact target_shape
        # Zero-pad 
        image, depth, K = crop_image_depth_and_intrinsic_by_pp(image, depth, K, target_shape=target_shape, strict=True, intr_convention=intr_convention)
        return image, depth, K


    def get_all_seqnames(self):
        seqnames = []
        for scene_name, img_names in self.sampled_list:
            if self.sample_extracted:
                seqnames.append(scene_name)
            else:
                if self.use_hash:
                    to_hash = f'{scene_name}_{"-".join([imn.split(".")[0] for imn in img_names])}'
                    seqnames.append(f'{self.name}_{string_to_filename(to_hash)}')
                else:
                    seqnames.append(f'{scene_name}')
        return seqnames

    def __getitem__(self, idx):
        scene_name, img_names = self.sampled_list[idx]
        batch = {"img_names": img_names, "scene_name":self.name, "dataset_name":self.name, 'intr_convention': "opencv" }
        if self.sample_extracted:
            batch["seq_name"] = f'{scene_name}'
        else:
            if self.use_hash:
                to_hash = f'{scene_name}_{"-".join([imn.split(".")[0] for imn in img_names])}'
                batch["seq_name"] = f'{self.name}_{string_to_filename(to_hash)}'
            else:
                batch["seq_name"] = f'{scene_name}'

        batch.update({"images":[], "extrinsics":[], "intrinsics":[]}) # depths also encode the valid masks
        if self.load_depths:
            batch['depths'] = []
        for iii, img_name in enumerate(img_names):
            image = self.read_img_rgb(scene_name, img_name)
            if self.load_depths:
                depth = self.read_img_depth(scene_name, img_name)
            else:
                depth = None
            scene_pose = self.read_scene_pose(scene_name)
            K = scene_pose[img_name]['K']
            batch["extrinsics"].append(scene_pose[img_name]['w2c']) 

            height, width = image.shape[:2]

            if 'height' in scene_pose[img_name]:
                assert height==scene_pose[img_name]['height'] and width==scene_pose[img_name]['width'], f"Image size mismatch {height} {width} {scene_pose[img_name]['height']} {scene_pose[img_name]['width']}"
            
            if scene_pose[img_name]['intr_convention'] == 'opencv':
                center_pp = (K[0,2] == width/2-0.5) and (K[1,2] == height/2-0.5)
            else:
                raise NotImplementedError(f"Intrinsics convention {scene_pose[img_name]['intr_convention']} not implemented")

            if self.img_size is not None:
                if self.aspect_ratio is not None:
                    target_shape = np.array([round(self.img_size * self.aspect_ratio), self.img_size])
                else:
                    # preserve the aspect ratio (Then every image should have the same shapes ..)
                    target_shape = np.array([round(height*self.img_size/width/14)*14, self.img_size])
                if self.sample_extracted and width==target_shape[1] and height==target_shape[0]:
                    processed_image, processsed_depth, processed_K = self.preprocess_2D_simple(image=image, depth=depth, K=K, target_shape=target_shape, intr_convention="opencv")
                else:
                    processed_image, processsed_depth, processed_K = self.preprocess_2D_vggt(image=image, depth=depth, K=K, target_shape=target_shape, intr_convention="opencv")
                batch["images"].append(processed_image)
                if self.load_depths:
                    batch["depths"].append(processsed_depth)
                batch["intrinsics"].append(processed_K)
            else:
                batch["images"].append(image)
                if self.load_depths:
                    batch["depths"].append(depth)
                batch["intrinsics"].append(K)

        # Convert list of arrays to tensors
        batch["images"] = torch.from_numpy(np.stack(batch["images"])).float() / 255.0  #(N, H, W, 3) float32 in [0,1]
        if self.load_depths:
            batch["depths"] = torch.from_numpy(np.stack(batch["depths"]))  #(N,H,W)
            batch["point_masks"] = (batch["depths"] > 1e-5)  #(N,H,W) bool
        batch["extrinsics"] = torch.from_numpy(np.stack(batch["extrinsics"]).astype(np.float32))  #(N,4,4)
        batch["intrinsics"] = torch.from_numpy(np.stack(batch["intrinsics"]).astype(np.float32))  #(N,3,3)
        return batch


