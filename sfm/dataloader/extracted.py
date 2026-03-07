from .base_dataset import BaseDataset
import os
import numpy as np 
import json, cv2
from utils.geometry import closed_form_inverse_se3

class ExtractedDataset(BaseDataset):
    def __init__(self,
        **kwargs,
        ):
        kwargs['sample_extracted'] = True
        super().__init__(**kwargs)
    
    def read_scene_pose(self, scene_name):
        extrinsics = np.load(os.path.join(self.root,  scene_name, 'extrinsics.npy'))
        intrinsics = np.load(os.path.join(self.root,  scene_name, 'intrinsics.npy'))
        # The dataset has linear camera and use opencv convention
        imgname2pose = {}
        for i in range(extrinsics.shape[0]):
            imgname = f"{i:06d}.jpg"
            w2c = extrinsics[i]
            c2w = closed_form_inverse_se3(w2c)
            imgname2pose[imgname] = {'c2w':c2w, 'w2c':w2c}
            width = int((intrinsics[i,0,2]+0.5)*2)
            height = int((intrinsics[i,1,2]+0.5)*2)
            imgname2pose[imgname].update({'K': intrinsics[i], 'width': width, 'height': height, 'intr_convention':'opencv'})
        return imgname2pose
    
    def read_img_pose(self, scene_name, img_name):
        return self.read_scene_pose(scene_name)[img_name]
    
    def read_img_rgb(self, scene_name, img_name):
        image = cv2.imread(os.path.join(self.root,  scene_name, f'images/{img_name}'))
        if image is None:
            raise FileNotFoundError(f"Image {img_name} not found in {scene_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        return image

    def read_img_depth(self, scene_name, img_name):
        depth = np.load(os.path.join(self.root,  scene_name, f'depths/{img_name.replace(".jpg", ".npy")}'))
        return depth
