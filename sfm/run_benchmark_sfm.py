import pycolmap # Fix for spconv collision
import os, random, cv2
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feedforward import FeedForward_Model
from matching import init_match_models
from sfm.dataloader import get_valComposedDataLoader

from sfm.sfm_func import run_sfm
from evaluation import eval_points, eval_cameras, EvalLogger, eval_multiview_depths


from utils.basic import set_seed, Print
from utils.io import save_xyzrgb_to_ply, save_images_as_grid
from utils.geometry import unproject_depth_map_to_point_map_torch, project_point_map_to_depth_map_torch
import time
import hydra
from glob import glob
import PIL
def move_to_device(batch, device):
    if type(batch) is dict:
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif type(batch) is list:
        return [move_to_device(v, device) for v in batch]
    elif type(batch) is torch.Tensor:
        return batch.to(device)
    else:
        return batch


def prepare_batch(batch, output_width=518):
    """
    Resize images, depth, K to FeedForward input size
    Unproject depth maps to point maps
    """
    from feedforward import preprocess
    batch_ffres = []
    for batch_ in batch:
        batch_ffres_ = {key:value for key,value in batch_.items() if type(value) is str}
        in_h, in_w = batch_['images'][0].shape[:2]
        batch_ffres_['images'] = preprocess(batch_['images'], output_width=output_width).to(batch_['images'][0].device)
        ff_h, ff_w = batch_ffres_['images'].shape[1:3]
        if 'depths' in batch_:
            if batch_['depths'].shape[1] == ff_h and batch_['depths'].shape[2] == ff_w:
                batch_ffres_['depths'] = batch_['depths']
            else:
                batch_ffres_['depths'] = torch.nn.functional.interpolate(batch_['depths'].unsqueeze(1), size=(ff_h, ff_w), mode='nearest-exact').squeeze(1)
        
        if 'point_masks' in batch_:
            if batch_['point_masks'].shape[1] == ff_h and batch_['point_masks'].shape[2] == ff_w:
                batch_ffres_['point_masks'] = batch_['point_masks']
            else:
                batch_ffres_['point_masks'] = torch.nn.functional.interpolate(batch_['point_masks'].unsqueeze(1).float(), size=(ff_h, ff_w), mode='nearest-exact').squeeze(1).bool()
        if 'intrinsics' in batch_:
            Ks = batch_['intrinsics'].clone()
            Ks[:,0,0] *= ff_w / batch_['images'][0].shape[1]
            Ks[:,1,1] *= ff_h / batch_['images'][0].shape[0]
            Ks[:,0,2] = (ff_w - 1) / 2.0
            Ks[:,1,2] = (ff_h - 1) / 2.0
            batch_ffres_['intrinsics'] = Ks
        if 'extrinsics' in batch_:
            batch_ffres_['extrinsics'] = batch_['extrinsics']
        if 'depths' in batch_ffres_:
            batch_ffres_['points'] = unproject_depth_map_to_point_map_torch(depth_map=batch_ffres_['depths'], extrinsics_cam=batch_['extrinsics'], intrinsics_cam=batch_ffres_['intrinsics'])
        elif 'points' in batch_:
            if in_h == ff_h and in_w == ff_w:
                batch_ffres_['points'] = batch_['points']
            else:
                batch_ffres_['points'] = torch.nn.functional.interpolate(batch_['points'].permute(0,3,1,2).float(), size=(ff_h, ff_w), mode='nearest-exact').permute(0,2,3,1).bool() 
            batch_ffres_['depths'] = project_point_map_to_depth_map_torch(point_map=batch_ffres_['points'], extrinsics_cam=batch_['extrinsics'], intrinsics_cam=batch_ffres_['intrinsics'])
            batch_ffres_['depths'][batch_ffres_['point_masks'] == False] = 0
        batch_ffres.append(batch_ffres_)
    return batch_ffres

def init_DDP(cfg=None):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f'cuda:{local_rank}'
    print(f"Start running basic DDP example on rank {rank}, device {device}")
    
    return rank, local_rank, device

@hydra.main(version_base=None, config_path="../configs",config_name="benchmark_sfm")
def main(cfg):
    set_seed(cfg.common_config.seed)
    if cfg.common_config.ddp:
        rank, local_rank, device = init_DDP()
    else:
        device = 'cuda'
    """
    Prepare Models (FeedForward and Matchers)
    """
    ff_model = FeedForward_Model(cfg.feedforward_config).to(device)
    Print(f"Initialized FeedForward model: {cfg.feedforward_config.model}")
    ff_model.eval()
    match_models = init_match_models(cfg.match_config.models, device=device)
    Print(f"Initialized matching models: {list(match_models.keys())}")

    """
    Load evaluation sets (with GT)
    """
    val_dataloader = get_valComposedDataLoader(cfg)
    eval_logger = EvalLogger(dirname=os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'eval_logs'))
    for batch_cpu in val_dataloader:
        batch_rawres = move_to_device(batch_cpu, device)
        batch_ffres = prepare_batch(batch_rawres, output_width=518)
        batch_ffres = batch_ffres[0]  #Assume batch size =1 for evaluation
        output_dir = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'save', batch_ffres['scene_name'], batch_ffres['seq_name'])
        os.makedirs(output_dir, exist_ok=True)
        if cfg.common_config.save_outputs:
            # Save GT (used for evaluation in GGPT)
            gt_outputs = {
                'points': batch_ffres['points'], 'point_masks': batch_ffres['point_masks'],
                'intrinsics': batch_ffres['intrinsics'],
                'extrinsics': batch_ffres['extrinsics'],
                'images': batch_ffres['images'],
            }
            torch.save(gt_outputs,  os.path.join(output_dir, 'gt.bin'))
        """
        Feedforward prediction
        """    
        # Feed-forward inference as initialization
        with torch.no_grad():
            ff_outputs = ff_model(batch_ffres['images'], preprocessed=True, gt_dict=batch_ffres)
        #Print("FeedForward inference done.")

        cam_metrics = eval_cameras(gt_extrinsics=batch_ffres['extrinsics'], pred_extrinsics=ff_outputs['extrinsics'], 
                                  gt_intrinsics=batch_ffres['intrinsics'], pred_intrinsics=ff_outputs['intrinsics'])
        eval_logger.write(cam_metrics, prefix=cfg.feedforward_config.model, dataset_key=batch_ffres['dataset_name'], seq_key=batch_ffres['seq_name'])         


        pts_metrics, ff_points_aligned, sim3d_mat = eval_points(gt_points=batch_ffres['points'], pred_points=ff_outputs['points'], eval_mask=batch_ffres['point_masks'], 
                    umeyama=True, 
                    unit=batch_cpu[0].get('unit', ('cm',100)),
                    umeyama_max_error=batch_cpu[0].get('umeyama_max_error', 0.03),
                    save_errormap=cfg.common_config.save_vis, output_filename=os.path.join(output_dir, 'ff_errormap.ply'))
        eval_logger.write(pts_metrics, prefix=cfg.feedforward_config.model, dataset_key=batch_ffres['dataset_name'], seq_key=batch_ffres['seq_name'])
        
        if cfg.common_config.save_vis:
            save_images_as_grid(batch_ffres['images'], os.path.join(output_dir, 'input_images.png'), num_per_row=4)
            save_xyzrgb_to_ply(points=ff_points_aligned, rgb=ff_outputs['images_ff'], filename=os.path.join(output_dir, 'ff_points.ply'))
            save_xyzrgb_to_ply(points=batch_ffres['points'][batch_ffres['point_masks']], rgb=ff_outputs['images_ff'][batch_ffres['point_masks']], filename=os.path.join(output_dir, 'gt_points.ply'))
        if cfg.common_config.save_outputs:
            torch.save(ff_outputs, os.path.join(output_dir, 'ff_outputs.bin'))

        if cfg.common_config.run_sfm:
            """
            SfM 
            Dense Matching + Sparse BA + Direct linear Triangulation
            """
            t0 = time.time()
            sfm_outputs = run_sfm(batch_rawres[0]['images'], ff_outputs, match_models, cfg, gt=batch_ffres, output_dir=output_dir)
            t1 = time.time()
            eval_logger.write({'runtime': t1 - t0}, prefix='sfm', dataset_key=batch_ffres['dataset_name'], seq_key=batch_ffres['seq_name'])
            if 'match_metrics' in sfm_outputs:
                eval_logger.write(sfm_outputs['match_metrics'], prefix='sfm', dataset_key=batch_ffres['dataset_name'], seq_key=batch_ffres['seq_name'])
            if sfm_outputs['camera_success']:
                cam_metrics = eval_cameras(gt_extrinsics=batch_ffres['extrinsics'], pred_extrinsics=sfm_outputs['extrinsics'], 
                                        gt_intrinsics=batch_ffres['intrinsics'], pred_intrinsics=sfm_outputs['intrinsics'])
                eval_logger.write(cam_metrics, prefix='sfm', dataset_key=batch_ffres['dataset_name'], seq_key=batch_ffres['seq_name'])
                if 'extr_rotcov' in sfm_outputs:
                    eval_logger.write({
                        'sfm/extr_rotcov_mean': sfm_outputs['extr_rotcov'].mean(), 
                        'sfm/extr_transcov_mean': sfm_outputs['extr_transcov'].mean(),
                        'sfm/extr_rotcov_max': sfm_outputs['extr_rotcov'].max(), 
                        'sfm/extr_transcov_max': sfm_outputs['extr_transcov'].max()}, prefix='sfm', dataset_key=batch_ffres['dataset_name'], seq_key=batch_ffres['seq_name'])
            if sfm_outputs['points_success']:
                pts_metrics, sfm_points_aligned, _ = eval_points(gt_points=batch_ffres['points'], pred_points=sfm_outputs['points'], eval_mask=sfm_outputs['point_masks'] & batch_ffres['point_masks'],
                            umeyama=True, 
                            unit=batch_cpu[0].get('unit', ('cm',100)),
                            umeyama_max_error=batch_cpu[0].get('umeyama_max_error', 0.03),
                            save_errormap=cfg.common_config.save_vis, output_filename=os.path.join(output_dir, 'sfm_errormap.ply'))
                pts_metrics['points/ratio(%)'] = round(sfm_outputs['point_masks'].float().mean().item()*100,1)
                eval_logger.write(pts_metrics, prefix='sfm', dataset_key=batch_ffres['dataset_name'], seq_key=batch_ffres['seq_name'])
                
                if cfg.common_config.save_vis:
                    save_xyzrgb_to_ply(points=sfm_outputs['points'][sfm_outputs['point_masks']], rgb=ff_outputs['images_ff'][sfm_outputs['point_masks']], filename=os.path.join(output_dir, 'sfm_dlt_points.ply'))
                if cfg.common_config.save_outputs:
                    torch.save(sfm_outputs, os.path.join(output_dir, 'sfm_dlt_outputs.bin'))

            else:
                Print("SfM failed.")
        eval_logger.save(False)

    eval_logger.save(True)
    return



if __name__ == "__main__":
    main()