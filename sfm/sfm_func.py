import torch 
import os
import numpy as np
import PIL
import hydra
# import pycolmap

from matching import  match_images
from utils.basic import Print
from utils.to_pycolmap import batch_torch_matrix_to_pycolmap
from utils.geometry import compute_epipolar_errors, homo
from tqdm import tqdm
def run_sfm(images, ff_outputs, match_models, cfg, gt=None, output_dir=None):
    import pycolmap
    images_ff = ff_outputs['images_ff']  # B,H,W,3
    N = images_ff.shape[0]
    device = images_ff.device
    ff_h, ff_w = images_ff.shape[1:3]
    output_dict = {} 

    """
    1. Dense matching
    """
    if isinstance(images, torch.Tensor)==False:
        # first convert to torch tensor 
        images = torch.from_numpy(np.stack([np.array(img) for img in images], axis=0)).float()/255.0  #(N,H,W,3)
        images = images.to(device)
    m_h, m_w = images.shape[1:3]
    sx, sy = ff_w/m_w, ff_h/m_h
    mres_to_fres = torch.tensor([[sx,0,0.5*(sx-1)],[0,sy,0.5*(sy-1)],[0,0,1]], dtype=torch.float32).to(device)  #3,3
    match_results = match_images(
        match_models=match_models,
        images_hr=images.permute(0,3,1,2),  #N,3,H,W
        lr_h=ff_h, lr_w=ff_w,
        hr_to_lr=mres_to_fres
    )

    if cfg.common_config.reduce_memory:
        del match_models
        torch.cuda.empty_cache()
        
    M_ba = (match_results['pred_scores']>cfg.ba_config.score_thresh) & (match_results['pred_cycle_error']<cfg.ba_config.cycle_err_thresh)
    M_dlt = (match_results['pred_scores']>cfg.dlt_config.score_thresh) & (match_results['pred_cycle_error']<cfg.dlt_config.cycle_err_thresh)

    """
    2. Sparse ba (TODO: support known camera poses)
    """
    # 2.1 Chooes tracks for BA
    M_ba = M_ba.reshape(N,-1) #Ntgt, Nsrc*H*W
    query_sp_scores = match_results['sp_scores'].view(-1)  #Nsrc*H*W
    tracknum_perview = torch.zeros(N, dtype=torch.int32).to(device)
    selected = torch.zeros(M_ba.shape[1], dtype=torch.bool).to(device) #Num_tracks=Nsrc*H*W
    assert M_ba.shape[1] == (N*ff_h*ff_w)
    for ni in tqdm(range(N), desc="Selecting tracks for BA"):
        to_select_num = cfg.ba_config.mintrack_per_view - tracknum_perview[ni]
        if to_select_num<=0:
            continue
        candidate_tracks = M_ba[ni]&(M_ba.sum(axis=0)>=2)&(~selected) # visible in this view, at least visible in 2 views, not selected, (Ntracks,)
        candidate_ids = torch.where(candidate_tracks)[0]
        if len(candidate_ids)==0:
            continue
        candidate_sps = query_sp_scores[candidate_ids]
        selected_ids = candidate_ids[torch.argsort(candidate_sps, descending=True)][:to_select_num]
        selected[selected_ids] = True
        tracknum_perview += M_ba[:,selected_ids].sum(axis=1).to(torch.int32)
    Print(f"The number of tracks covering each view: {[nn.item() for nn in tracknum_perview]}")
    tracks_ba = match_results['pred_matches_lr'].view(N,-1,2)[:,selected]  #N, Ntracks_ba, 2
    tracks_mask_ba = M_ba[:,selected]  #Ntgt, Ntracks_ba
    assert tracks_mask_ba.sum(axis=0).min()>=2, "Each track should be visible in at least two views for BA."
    pts3d_ba = ff_outputs['points'].reshape(-1,3)[selected]  #Ntracks_ba, 3 (Takes the ff's prediction at query positions as initialization)

    if cfg.match_config.save_vis:
        from matching.vis_match import vis_matches, vis_matches_in_pairs
        from utils.tracks import extract_tracks_from_points
        vis_matches(images=images_ff.permute(0,3,1,2),  #N,3,H,W 
            matches=tracks_ba,
            visibility=tracks_mask_ba, # visualize the matches with queries from image i
            filename=os.path.join(output_dir, f'matches_for_ba.png'), 
            vis_mask=tracks_mask_ba.sum(axis=0)>=2, # visualize len>=2 tracks (Ntgt,Num_tracks) (Actually it is all zero)
            vis_num_track=10)

    calibrated = cfg.ba_config.get('calibrated', False)
    ba_intrinsics = gt['intrinsics'][:,:3,:3].to(device).float() if calibrated else ff_outputs['intrinsics'][:,:3,:3]
    reconstruction = batch_torch_matrix_to_pycolmap(
        points3d = pts3d_ba,
        tracks = tracks_ba+0.5, masks = tracks_mask_ba,
        extrinsics = ff_outputs['extrinsics'][:,:3,:4],
        intrinsics = ba_intrinsics+0.5,
        image_size = [ff_w, ff_h],
        camera_type = cfg.ba_config.camera_type,
        shared_camera = cfg.ba_config.shared_camera,
    )
    xyz_before_ba = torch.from_numpy(np.stack([reconstruction.points3D[i].xyz for i in sorted(reconstruction.points3D.keys())])).to(device).float()
    refine_focal = not calibrated and cfg.ba_config.get('refine_focal_length', True)
    # refine_pp = not calibrated
    if cfg.ba_config.loss_function_type == 'cauchy':
        loss_function_type = pycolmap.LossFunctionType.CAUCHY
    else:
        loss_function_type = pycolmap.LossFunctionType.TRIVIAL
    ba_options = pycolmap.BundleAdjustmentOptions(
        loss_function_type=loss_function_type, loss_function_scale=cfg.ba_config.loss_function_scale,
        refine_focal_length=refine_focal)
    ba_config = pycolmap.BundleAdjustmentConfig()
    for img_id in reconstruction.images.keys():
        ba_config.add_image(img_id)
    bundle_adjuster = pycolmap.create_default_bundle_adjuster(ba_options, ba_config, reconstruction)
    bundle_adjuster.solve()



    output_dict['intrinsics'] = torch.zeros_like(ff_outputs['intrinsics'])
    output_dict['extrinsics'] = torch.zeros_like(ff_outputs['extrinsics'])
    for i in range(1,N+1): # Image ids in pycolmap start from 1
        if cfg.ba_config.camera_type == "PINHOLE":
            fx, fy, cx, cy = reconstruction.cameras[reconstruction.images[i].camera_id].params[:4]
        else:
            focal, cx, cy = reconstruction.cameras[reconstruction.images[i].camera_id].params[:3]
            fx = fy = focal
        output_dict['intrinsics'][i-1, 0, 0] = fx
        output_dict['intrinsics'][i-1, 1, 1] = fy
        output_dict['intrinsics'][i-1, 0, 2] = ff_w/2-0.5
        output_dict['intrinsics'][i-1, 1, 2] = ff_h/2-0.5 #Center principal point!
        output_dict['intrinsics'][i-1, 2, 2] = 1.0
        if True or pycolmap.__version__ == '3.13.0':
            rigid3d = reconstruction.images[i].cam_from_world().matrix() # (3,4)
            #or rigid3d=reconstruction.frames[i].rig_from_world.matrix()
        else:
            rigid3d = reconstruction.images[i].cam_from_world.matrix() # (3,4)
        output_dict['extrinsics'][i-1,:3,:] = torch.from_numpy(rigid3d).to(device).float()
        if output_dict['extrinsics'].shape[1] == 4:
            output_dict['extrinsics'][i-1,3,3] = 1.0
    output_dict['camera_success'] = True


    """
    3. Direct linear Triangulation
    """
    intrinsic_dlt, extrinsic_dlt = output_dict['intrinsics'].to(device), output_dict['extrinsics'].to(device) 
    P = torch.einsum('nij,njk->nik', intrinsic_dlt, extrinsic_dlt[:,:3,:4]).float() #N,3,4
    camR, camT = extrinsic_dlt[:,:3,:3], extrinsic_dlt[:,:3,-1]
    camC = -torch.einsum('nij,nj->ni', camR.permute(0,2,1), camT) #(N,3)
    # filter matches with epipolar error (TODO to make it more efficient)
    w2c_s, K_s = output_dict['extrinsics'].to(device), output_dict['intrinsics'].to(device)
    M_dlt = M_dlt.to(device)
    dlt_num = (M_dlt.sum(axis=0)>=2).sum().item()

    M_ba_debug = M_ba.reshape(N,N,-1).clone()
    for ni in range(N):
        w2c_0, K_0 = output_dict['extrinsics'][ni], output_dict['intrinsics'][ni]
        matches = match_results['pred_matches_lr'][:,ni].view(N,ff_h,ff_w,2) #(N, H,W, 2)
        dis_a, dis_b = compute_epipolar_errors(w2c_0, w2c_s, K_0, K_s, matches.view(N,ff_h,ff_w,2)) #(N,H,W)
        epipolar_errors_msk = (dis_a < cfg.dlt_config.max_epipolar_error) & (dis_b < cfg.dlt_config.max_epipolar_error) 
        epipolar_errors_msk[ni,:,:] = True  #self-view
        epipolar_errors_msk = epipolar_errors_msk.view(N, ff_h*ff_w) #(Ntgt, H*W)
        M_dlt[:,ni] = M_dlt[:,ni] & epipolar_errors_msk  #(Ntgt, H*W)
        M_ba_debug[:,ni] = M_ba_debug[:,ni] & epipolar_errors_msk  #(Ntgt, H*W)
    M_dlt = M_dlt.reshape(N,-1) #Ntgt, Nsrc*H*W
    remaining_tracks = (M_dlt.sum(axis=0)>=2)  #(Nsrc*H*W,)
    print(f"Number of tracks used for DLT triangulation: {remaining_tracks.sum().item()}")
    print("Epipolar error discard ratio: ", 1-(M_dlt.sum(axis=0)>=2).sum().item()/dlt_num)
    tracks_dlt = match_results['pred_matches_lr'].view(N,-1,2)[:,remaining_tracks].to(device)  #Nview, Ntracks_dlt, 2
    tracks_mask_dlt = M_dlt[:,remaining_tracks]  #Nview, Ntracks_dlt

    weights = tracks_mask_dlt.float()
    max_pts_num = cfg.dlt_config.get('batch_size', 500000)
    num_chunk = (tracks_dlt.shape[1]+max_pts_num-1)//max_pts_num
    xyz_in_img, count_in_img = torch.zeros(N*ff_h*ff_w, 3).to(device), torch.zeros(N*ff_h*ff_w).to(device) # Our target
    for chunk_id in tqdm(range(num_chunk), desc='DLT triangulation'):
        start_idx = chunk_id*max_pts_num
        end_idx = min((chunk_id+1)*max_pts_num, tracks_dlt.shape[1])
        tracks_dlt_chunk = tracks_dlt[:,start_idx:end_idx]  #(Nview, Ntracks_chunk, 2)
        tracks_mask_dlt_chunk = tracks_mask_dlt[:,start_idx:end_idx]  #(Nview, Ntracks_chunk)
        
        Ai_chunk = tracks_dlt_chunk[...,None] * P[:,None,2:3,:] - P[:,None,:2,:] #(Nview,Ntracks,2,4)
        weights_chunk = weights[:,start_idx:end_idx] #(Nview, Ntracks_chunk)
        AitAi = Ai_chunk.permute(0,1,3,2) @ Ai_chunk #(Nview,Ntracks_chunk,4,4) 
        AitAi = AitAi * weights_chunk[:,:,None,None] #(Nview,Ntracks_chunk,4,4)
        AitAi_sum = AitAi.sum(axis=0) #(Ntracks_chunk,4,4)
        _, eigenvectors_chunk = torch.linalg.eigh(AitAi_sum) #(Ntracks_chunk,4), (Ntracks_chunk,4,4)
        pt3d_chunk = eigenvectors_chunk[:,:,0] #(Ntracks_chunk,4) Ascending order, the first column vector
        
        # Filter points with invalid solutions
        xyz = pt3d_chunk[:,:3] / pt3d_chunk[:,3:4] #(Npts,3)
        filter1 = pt3d_chunk[:,3].abs()>1e-10  #valid solution
        xyz = xyz[filter1]
        tracks_dlt_chunk = tracks_dlt_chunk[:,filter1]
        tracks_mask_dlt_chunk = tracks_mask_dlt_chunk[:,filter1]

        # Filter points based on reprojection error
        xy_reproj_chunk = torch.einsum('nij,mj->nmi', P, homo(xyz))
        xy_reproj_chunk = xy_reproj_chunk[:,:,:2]/xy_reproj_chunk[:,:,2:3]
        reproj_error_chunk = torch.norm(xy_reproj_chunk-tracks_dlt_chunk, dim=-1)
        reproj_error_mean_chunk = (reproj_error_chunk*tracks_mask_dlt_chunk).sum(0)/tracks_mask_dlt_chunk.sum(0)  
        filter_reproj = (reproj_error_mean_chunk < cfg.dlt_config.max_reproj_error)
        xyz = xyz[filter_reproj]
        tracks_dlt_chunk = tracks_dlt_chunk[:,filter_reproj]
        tracks_mask_dlt_chunk = tracks_mask_dlt_chunk[:,filter_reproj]
        if filter_reproj.sum()==0:
            continue
        # Filter points based on triangulation angles
        # We need further batch operations here. quadratic to the number of views
        max_pts_num2 = int(min(max_pts_num//N, tracks_dlt_chunk.shape[1]))
        num_chunk2 = (tracks_dlt_chunk.shape[1]+max_pts_num2-1)//max_pts_num2
        for chunk_id2 in range(num_chunk2):
            start_idx2 = chunk_id2*max_pts_num2 
            end_idx2 = min((chunk_id2+1)*max_pts_num2, tracks_dlt_chunk.shape[1])
            rays_chunk = xyz[None,start_idx2:end_idx2,...] - camC[:,None,:] #(N,Ntracks_chunk,3)
            rays_chunk = rays_chunk / torch.linalg.norm(rays_chunk, axis=-1, keepdim=True) #(N,Ntracks_chunk,3)
            cos_angles = (rays_chunk[None,:,:,:]*rays_chunk[:,None,:,:]).sum(-1) #(N,N,Ntracks_chunk)-> (Nview1, Nview2, Ntracks_chunk)
            angle_radians = torch.acos(torch.clamp(cos_angles, -0.9999, 0.9999))
            angle_degs = torch.rad2deg(angle_radians)
            vismask_pairwise = tracks_mask_dlt_chunk[None,:, start_idx2:end_idx2] & tracks_mask_dlt_chunk[:,None,start_idx2:end_idx2]  #(Nview1, Nview2, Ntracks_chunk)
            angle_degs = angle_degs * vismask_pairwise.float() # Set invisible views to 0 angle 
            max_angle_degs_chunk = angle_degs.view(-1, end_idx2-start_idx2).max(0).values #(Ntracks_chunk,)
            if chunk_id2==0:
                max_angle_degs = max_angle_degs_chunk
            else:
                max_angle_degs = torch.cat([max_angle_degs, max_angle_degs_chunk], 0)
        filter_angle = (max_angle_degs > cfg.dlt_config.min_tri_angle)
        xyz = xyz[filter_angle]
        tracks_dlt_chunk = tracks_dlt_chunk[:,filter_angle]
        tracks_mask_dlt_chunk = tracks_mask_dlt_chunk[:,filter_angle]

        xyz_for_img_chunk = xyz.unsqueeze(0).tile(N,1,1)[tracks_mask_dlt_chunk] #(N,Npts_chunk,3) -> (Nobs_chunk, 3)
        view_index_chunk = torch.arange(N).to(device).unsqueeze(1).tile(1,xyz.shape[0])[tracks_mask_dlt_chunk]  #(N,Npts) -> (Nobs_chunk,)
        index2d_in_img_chunk = tracks_dlt_chunk[tracks_mask_dlt_chunk].round().long() # (Nobs_chunk,2-xy)
        index1d_in_scene_chunk = view_index_chunk*ff_w*ff_h + index2d_in_img_chunk[:,1].clamp(0,ff_h-1)*ff_w + index2d_in_img_chunk[:,0].clamp(0,ff_w-1)  #(Nobs_chunk,)
        xyz_in_img.scatter_reduce_(dim=0, index=index1d_in_scene_chunk.unsqueeze(1).expand(-1,3), src=xyz_for_img_chunk, reduce='sum', include_self=True)
        count_in_img.scatter_reduce_(dim=0, index=index1d_in_scene_chunk, src=torch.ones_like(index1d_in_scene_chunk).float(), reduce='sum', include_self=True)
    
    xyz_in_img = xyz_in_img / count_in_img.clamp(min=1)[:,None]
    dlt_xyz, count_in_img = xyz_in_img.view(N,ff_h,ff_w,3), count_in_img.view(N,ff_h,ff_w)
    dlt_mask = (count_in_img>0)  #(N,H,W)

    if dlt_mask.sum()==0:
        print('No valid DLT points after assigning to image pixels')
        output_dict['points_success'] = False
        return output_dict

    output_dict['points'] = dlt_xyz
    output_dict['point_masks'] = dlt_mask
    output_dict['points_success'] = True

    return output_dict
    



