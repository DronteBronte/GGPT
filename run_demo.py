import pycolmap # Fix for spconv collision
import os, random, cv2
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from feedforward import FeedForward_Model, preprocess
from matching import init_match_models

from sfm.sfm_func import run_sfm


from utils.basic import set_seed, Print
from utils.io import save_xyzrgb_to_ply

import hydra
from glob import glob
import PIL
from omegaconf import OmegaConf
from hydra.utils import instantiate
from sfm.run_benchmark_sfm import move_to_device
from tqdm import tqdm
INTMAX = 1e6

def filter_points(points, confs=None, masks=None, max_pts_num=None, conf_quantile_thresh=0.2):
    """
    point: (..., 3)
    confs: (...)
    masks: (...)
    max_pts_num: int
    conf_quantile_thresh: float
    """
    if masks is None:
        masks = torch.ones_like(confs, dtype=torch.bool)
    if confs is not None:
        #To prevent confs[masks] has too many elements (INTMAX)
        if confs[masks].numel() > INTMAX:
            stride = int(confs[masks].numel() / INTMAX)*5
            conf_thresh = torch.quantile(confs[masks][::stride], conf_quantile_thresh)
        else:
            conf_thresh = torch.quantile(confs[masks], conf_quantile_thresh)
        masks = masks & (confs >= conf_thresh)

    if max_pts_num is not None:
        if masks.sum() > max_pts_num:
            masks_ = masks.view(-1)
            indices = torch.nonzero(masks_, as_tuple=False)
            indices = indices[torch.randperm(indices.shape[0])[:max_pts_num]]
            masks_ = torch.zeros_like(masks_, dtype=torch.bool)
            masks_[indices] = True
            masks = masks_.view_as(masks)
    return masks


@hydra.main(version_base=None, config_path="configs",config_name="demo")
def main(cfg):
    set_seed(cfg.common_config.seed)
    device = 'cuda:0'
    """
    Prepare Models (FeedForward and Matchers)
    """
    ff_model = FeedForward_Model(cfg.feedforward_config).to(device)
    Print(f"Initialized FeedForward model: {cfg.feedforward_config.model}")
    ff_model.eval()
    """
    Feedforward prediction
    """    
    images = []
    for ii, image_path in enumerate(sorted(glob(os.path.join(cfg.image_dir, "*")))):
        if image_path.lower().endswith(".jpg") or image_path.lower().endswith(".png") or image_path.lower().endswith(".jpeg"):
            img = cv2.imread(image_path)[:,:,::-1]
            if ii==0:
                h, w = img.shape[:2]
            elif h!=img.shape[0] or w!=img.shape[1]:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR) #So that all images have the same size
            images.append(img)
    images = torch.from_numpy(np.stack(images, axis=0)).float()/255.0
    # reshape it to H/W divisible by 14
    H, W = images.shape[1], images.shape[2]
    if W > cfg.match_config.max_width:
        H = int(round(H*cfg.match_config.max_width/W))
        W = cfg.match_config.max_width
    newH, newW = int(round(H/14)*14), int(round(W/14)*14)
    images = torch.nn.functional.interpolate(images.permute(0,3,1,2), size=(newH, newW), mode='bilinear', align_corners=False).permute(0,2,3,1)

    images = images.to(device)  
    # Feed-forward inference as initialization
    with torch.no_grad():
        ff_outputs = ff_model(images, preprocessed=False)
    Print("FeedForward inference done.")

    if cfg.common_config.reduce_memory:
        del ff_model
        torch.cuda.empty_cache()

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    """
    SfM 
    Dense Matching + Sparse BA + Direct linear Triangulation
    """
    match_models = init_match_models(cfg.match_config.models, device=device)
    Print(f"Initialized Matching models: {list(match_models.keys())}")

    sfm_outputs = run_sfm(images, ff_outputs, match_models, cfg)

    if not sfm_outputs['points_success']:
        Print("SfM fails. The input is a hard case for geometry reconstruction :)")
        exit()

    if cfg.common_config.reduce_memory:
        del match_models 
    else:
        del ff_model, match_models 
    torch.cuda.empty_cache()

    ggpt_model = instantiate(cfg.ggptmodel_config).eval()
    ckpt = torch.load(cfg.common_config.ggpt_ckpt, map_location='cpu')
    ckpt = {k.replace('module.',''):v for k,v in ckpt.items()}
    ggpt_model.load_state_dict(ckpt, strict=True)
    ggpt_model = ggpt_model.to(device)
    print(f"Loaded GGPT model from {cfg.common_config.ggpt_ckpt}")

    from ggpt.dataloader.demo_dataset import DemoDataset
    from utils.points import aggregate_chunks
    import time
    
    demo_dataset = DemoDataset(name='demo', ff_data=ff_outputs, geo_data=sfm_outputs)
    scene_chunks, scene = demo_dataset[0]
    chunks_batch = [[chunk] for chunk in scene_chunks] # Add the batch dimension (batch-size=1, each chunk is a single batch)
    to_collect =  {'ff_pts':[], 'ff_pts_conf':[]}
    t0 = time.time()
    for chunk_batch in tqdm(chunks_batch, desc="GGPT inference"):
        chunk_batch = move_to_device(chunk_batch, device)
        with torch.no_grad():
            out = ggpt_model(chunk_batch)
        to_collect['ff_pts'].append(demo_dataset.unnormalize_pts(chunk_batch[0], out['ff_pts_out']))
        to_collect['ff_pts_conf'].append(out['ff_pts_conf_out'])
        
    ff_pts_all = torch.cat(to_collect['ff_pts'], dim=0) # (num_chunks, num_view, H, W, 3)
    ff_pts_conf_all = torch.cat(to_collect['ff_pts_conf'], dim=0) # (num_chunks, num_view, H, W)
    msks_in_scene = torch.stack([chunk['msks_in_scene'] for chunk in scene_chunks], dim=0).to(device) # (num_chunks, num_view, H, W)
    pred_pts, pred_confs, pred_mask = aggregate_chunks(ff_pts_all, ff_pts_conf_all, msks_in_scene, scene)
    
    t1 = time.time()
    Print(f"GGPT inference done in {t1 - t0:.2f}s.")

    if cfg.common_config.save_vis:
        sfm_masks = filter_points(sfm_outputs['points'], None, sfm_outputs['point_masks'], cfg.common_config.max_pts_num, cfg.common_config.conf_quantile_thresh)
        save_xyzrgb_to_ply(points=sfm_outputs['points'][sfm_masks], rgb=ff_outputs['images_ff'][sfm_masks], filename=os.path.join(output_dir, 'sfm_dlt_points.ply'))
        Print(f"Saved SfM DLT points to {os.path.join(output_dir, 'sfm_dlt_points.ply')}")

        pred_mask = filter_points(pred_pts, pred_confs, None, cfg.common_config.max_pts_num, cfg.common_config.conf_quantile_thresh)
        save_xyzrgb_to_ply(points=ff_outputs['points'][pred_mask], rgb=ff_outputs['images_ff'][pred_mask], filename=os.path.join(output_dir, 'ff_points.ply'))
        Print(f"Saved feedforward points to {os.path.join(output_dir, 'ff_points.ply')}")
        save_xyzrgb_to_ply(points=pred_pts[pred_mask], rgb=scene['images'][pred_mask].to(device),filename=os.path.join(output_dir, 'ggpt_points.ply'))
        Print(f"Saved predicted points to {os.path.join(output_dir, 'ggpt_points.ply')}")
    return



if __name__ == "__main__":
    main()