
import numpy as np
import os, random, cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 
import torch.nn as nn
from torch.utils.data import  DataLoader

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict 

from ggpt.dataloader import build_val_dataloader
from utils.common import set_seed, move_to_device, init_DDP
from ggpt.model.base import BasePredictor

from utils.logger import EvalLogger
from utils.points import aggregate_chunks, align_eval_points
from utils.geometry import project_point_map_to_depth_map_torch#, estimate_depths_from_points
import sys 
from evaluation import eval_points, eval_multiview_depths, rmse
from utils.io import save_xyzrgb_to_ply

from tqdm import tqdm
import time



@hydra.main(version_base=None, config_path="../configs",config_name="benchmark_ggpt")
def main(cfg):
    set_seed(cfg.common_config.seed)
    if cfg.common_config.ddp:
        rank, device_id = init_DDP()
        device = 'cuda'
    else:
        raise NotImplementedError("DDP is not supported for GGPT yet!")
        device = 'cuda'
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger = EvalLogger(output_dir)
    model = instantiate(cfg.ggptmodel_config).eval()
    ckpt = torch.load(cfg.common_config.load_ckpt, map_location='cpu')
    ckpt = {k.replace('module.',''):v for k,v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], find_unused_parameters=False)

    val_dataloader = build_val_dataloader(cfg)
    unnormalize_func = val_dataloader.dataset.unnormalize_pts
    for batch in tqdm(val_dataloader, total=len(val_dataloader)):
        assert len(batch)==1, "Eval dataset should have only one scene!"
        scene_chunks, scene = batch[0]
        
        chunks_batch = [[chunk] for chunk in scene_chunks] # Add the batch dimension (batch-size=1, each chunk is a single batch)
        to_collect =  {'ff_pts':[], 'ff_pts_conf':[]}
        t0 = time.time()
        for chunk_batch in chunks_batch:
            chunk_batch = move_to_device(chunk_batch, device)
            with torch.no_grad():
                out = model(chunk_batch)
            to_collect['ff_pts'].append(unnormalize_func(chunk_batch[0], out['ff_pts_out']))
            to_collect['ff_pts_conf'].append(out['ff_pts_conf_out'])
        if len(scene_chunks)==0:
            print(f"{scene['dataset_name']}-{scene['scene_name']}: No valid chunks found!")
            pred_pts = scene['ff_pts'].clone().to(device)
            pred_confs = scene['ff_conf'].clone().to(device)
            pred_mask = torch.ones_like(scene['ff_conf'], dtype=torch.bool).to(device)
        else:
            ff_pts_all = torch.cat(to_collect['ff_pts'], dim=0) # (num_chunks, num_view, H, W, 3)
            ff_pts_conf_all = torch.cat(to_collect['ff_pts_conf'], dim=0) # (num_chunks, num_view, H, W)
            msks_in_scene = torch.stack([chunk['msks_in_scene'] for chunk in scene_chunks], dim=0).to(device) # (num_chunks, num_view, H, W)
            pred_pts, pred_confs, pred_mask = aggregate_chunks(ff_pts_all, ff_pts_conf_all, msks_in_scene, scene)
        t1 = time.time()
        runtime = t1 - t0
        logger.write({'runtime': runtime}, prefix='runtime', dataset_key=scene['dataset_name'], seq_key=scene['scene_name'])
        eval_mask, gt_pts = scene['gt_msks'].to(device), scene['gt_pts_metric'].to(device)
        
        max_error = 0.03

        if scene['gt_extrinsics'] is not None:
            gt_depths = project_point_map_to_depth_map_torch(gt_pts, scene['gt_extrinsics'].to(device), scene['gt_intrinsics'].to(device))
        else:
            gt_depths = estimate_depths_from_points(gt_pts, scene['gt_intrinsics'].to(device), conf=None, refine_intrinsic=False)
        metric_init, aligned_pts_init, sim3d_mat = eval_points(gt_points=gt_pts, pred_points=scene['ff_pts_original'].to(device), eval_mask=eval_mask, umeyama=True, umeyama_max_error=max_error)
        
        metric_pred, aligned_pts_pred, sim3d_mat = eval_points(gt_points=gt_pts, pred_points=pred_pts, eval_mask=eval_mask, umeyama=True, umeyama_max_error=max_error)
            
        logger.write(metric_init, prefix='init', dataset_key=scene['dataset_name'], seq_key=scene['scene_name'])
        logger.write(metric_pred, prefix='pred', dataset_key=scene['dataset_name'], seq_key=scene['scene_name'])
        logger.save(False)

        if cfg.common_config.save_vis:
            save_dir = os.path.join(output_dir, 'save', scene['dataset_name'], scene['scene_name'])
            os.makedirs(save_dir, exist_ok=True)
            rgb = scene['images'][eval_mask.cpu()]
            from utils.io import save_images_as_grid
            save_images_as_grid(scene['images'], os.path.join(save_dir, 'rgb.png'), num_per_row=len(scene['images']))   
            save_xyzrgb_to_ply(gt_pts[eval_mask], scene['images'][eval_mask.cpu()], os.path.join(save_dir, 'gt_points.ply'))
            save_xyzrgb_to_ply(aligned_pts_init[eval_mask], scene['images'][eval_mask.cpu()],os.path.join(save_dir, 'init_aligned_points.ply'))
            save_xyzrgb_to_ply(aligned_pts_pred[eval_mask], scene['images'][eval_mask.cpu()], os.path.join(save_dir, 'pred_aligned_points.ply'))

            eval_geo_mask = scene['geo_msks'].to(device) & eval_mask
            metric_geo, aligned_pts_geo = align_eval_points(B=gt_pts, A=scene['geo_pts'].to(device), mask=eval_geo_mask, max_error=max_error)
            save_xyzrgb_to_ply(aligned_pts_geo[scene['geo_msks'].to(device)], scene['images'][scene['geo_msks']], os.path.join(save_dir, 'geo_aligned_points.ply'))

    logger.save(True)






if __name__ == "__main__":
    main()