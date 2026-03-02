import torch 
from ggpt.dataloader.base_dataset import BaseDataset

class DemoDataset(BaseDataset):
    def __init__(self, name, 
        ff_data=None, geo_data=None,
        **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.ff_data = ff_data
        self.geo_data = geo_data
        self.mode = 'demo' 
    
    def load_scene(self, idx):
        assert idx == 0
        scene = {
            'dataset_name': 'demo',
            'scene_name': self.name,
            'ff_pts': self.ff_data['points'], # (N,H,W,3)
            'ff_conf': self.ff_data['points_conf'], # (N,H,W)
            'geo_pts': self.geo_data['points'], # (N,H,W,3)
            'geo_msks': self.geo_data['point_masks'], # (N,H,W)
            'images': self.ff_data['images_ff'], # (N,H,W,3)
        }
        return scene 

    def __len__(self):
        return 1
    


