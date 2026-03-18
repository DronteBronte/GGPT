import torch 
import os
import numpy as np
# import pycolmap





def aggregate_chunks(ff_pts_all, ff_pts_conf_all, msks_in_scene, scene):
    _, N, H, W = msks_in_scene.shape

    # index = msks_in_scene.nonzero(as_tuple=False)  # num_chunk, n_view, H, W 
    # index = index[:,1]*W*H + index[:,2]*W + index[:,3]
    # agg_ff_pts = torch.zeros((N*H*W, 3), device=ff_pts_all.device)
    # agg_ff_pts.scatter_reduce_(dim=0,index=index.unsqueeze(-1).repeat(1,3),src=ff_pts_all,reduce='mean',include_self=False)
    # agg_ff_pts = agg_ff_pts.view(N,H,W,3)
    # agg_ff_pts_conf = torch.zeros((N*H*W), device=agg_ff_pts.device)
    # agg_ff_pts_conf.scatter_reduce_(dim=0,index=index,src=ff_pts_conf_all,reduce='mean',include_self=False)
    # agg_ff_pts_conf = agg_ff_pts_conf.view(N,H,W)
    # agg_ff_mask = agg_ff_pts_conf>0

    device = ff_pts_all.device
    num_scene_pts = N * H * W

    pts_sum = torch.zeros((num_scene_pts, 3), device=device, dtype=ff_pts_all.dtype)
    conf_sum = torch.zeros((num_scene_pts,), device=device, dtype=ff_pts_conf_all.dtype)
    counts = torch.zeros((num_scene_pts,), device=device, dtype=ff_pts_conf_all.dtype)

    start_idx = 0
    for chunk_mask in msks_in_scene:
        flat_index = torch.where(chunk_mask.reshape(-1))[0]
        num_chunk_pts = flat_index.numel()
        if num_chunk_pts == 0:
            continue

        end_idx = start_idx + num_chunk_pts
        pts_sum.index_add_(0, flat_index, ff_pts_all[start_idx:end_idx])
        conf_sum.index_add_(0, flat_index, ff_pts_conf_all[start_idx:end_idx])
        counts.index_add_(0, flat_index, torch.ones(num_chunk_pts, device=device, dtype=counts.dtype))
        start_idx = end_idx

    if start_idx != ff_pts_all.shape[0]:
        raise RuntimeError(
            f"aggregate_chunks consumed {start_idx} points, but got {ff_pts_all.shape[0]} predictions."
        )

    agg_ff_mask = counts > 0
    agg_ff_pts = scene['ff_pts'].to(device).reshape(num_scene_pts, 3).clone()
    agg_ff_pts_conf = torch.zeros((num_scene_pts,), device=device, dtype=ff_pts_conf_all.dtype)

    agg_ff_pts[agg_ff_mask] = pts_sum[agg_ff_mask] / counts[agg_ff_mask].unsqueeze(-1)
    agg_ff_pts_conf[agg_ff_mask] = conf_sum[agg_ff_mask] / counts[agg_ff_mask]

    agg_ff_pts = agg_ff_pts.view(N, H, W, 3)
    agg_ff_pts_conf = agg_ff_pts_conf.view(N, H, W)
    return agg_ff_pts, agg_ff_pts_conf, agg_ff_mask.view(N, H, W)

def align_eval_points(B, A, mask, max_error=0.03):
    # Align A to B via Umeyama alignment
    A_aligned, sim3d_mat = umeyama_alignment(B, A, mask=mask, max_error=max_error)
    metric =  rmse_cuda(B[mask], A_aligned[mask])[0]
    return metric, A_aligned

def umeyama_alignment(B, A, mask=None, max_error=0.03):
    import pycolmap
    if mask is None:
        mask = torch.ones_like(A[:,0], dtype=torch.bool)
    src = A[mask].cpu().numpy()
    tgt = B[mask].cpu().numpy()

    max_error = float(os.environ.get('UMEYAMA_MAX_ERROR', max_error))
    while True:
        estimation_options = pycolmap.RANSACOptions(max_error=max_error, min_inlier_ratio=0.8) #In pycolmap, the maxerror in align-via-points is 0.005 and the min_inlier_ratio=0.9
        #print('Running robust alignment via points with max_error:', max_error)
        sim3d = pycolmap.estimate_sim3d_robust(src, tgt, estimation_options=estimation_options)
        if sim3d is None:
            if max_error > 0.3:
                print(f"Aligning points failed, max_error is too large: {max_error}, give up using robust alignment")
                sim3d = pycolmap.estimate_sim3d(src, tgt) # Use non-robust alignment
                break
            else:
                max_error *= 3
                print(f"Aligning points failed, increasing max_error to {max_error}")
        else:
            break
    #Do some reshaping
    Ashape0 = A.shape
    Bshape0 = B.shape
    if len(A.shape)!=2:
        A = A.reshape(-1,3)
    if len(B.shape)!=2:
        B = B.reshape(-1,3)
    if sim3d is None:
        A_aligned = A
        sim3d_mat = torch.eye(4, device=A.device).float()[:3,:]
    else:
        A_homo = torch.cat([A, torch.ones_like(A[:,-1:])], axis=1)
        sim3d_mat = torch.from_numpy(sim3d['tgt_from_src'].matrix()).to(A_homo.device).float()
        A_aligned = torch.einsum('ij,nj->ni', sim3d_mat, A_homo)
    A_aligned = A_aligned.reshape(Ashape0)
    return A_aligned, sim3d_mat


def rmse_cuda(gt_points, rec_points):
    rmses = torch.sqrt(torch.sum((gt_points - rec_points) ** 2, axis=-1))
    rmse_mean = torch.mean(rmses)
    rmse_median = torch.median(rmses)

    res = {'rmse_mean(cm)': rmse_mean*100, 'rmse_median(cm)': rmse_median*100,} 

    # unit: mm
    for unit, div in zip(['cm'], [100]): #zip(['mm', 'cm'], [1000, 100]):
        threshold = np.array(list(np.arange(1,11)/div))
        recalls = [] 
        for th in threshold:
            recall = torch.sum(rmses < th) / rmses.shape[0]
            recalls.append(recall)
        recalls = torch.tensor(recalls)
        auc = torch.cumsum(recalls, axis=0)/ torch.arange(1, 11, device=recalls.device)  # Normalize by number of points
        for i, th in enumerate(threshold):
            th_ = int(th*div)
            if th_ in [1,3,5,10]:#(i+1)%2==0: #2,4,6,8,10
                res[f'{unit}-auc@{th_:02d}{unit}(%)'] = auc[i]*100
                #res[f'{unit}-recall@{th_:02d}{unit}(%)'] = recalls[i]*100
    res = {k:round(float(v),4) for k, v in res.items()}
    return res,  rmses
