from omegaconf import OmegaConf
from utils.geometry import closed_form_inverse_K
from matching.densematcher import match_dense
import sys

def init_match_models(model_name, device):
    if type(model_name) != str:
        model_name = OmegaConf.to_object(model_name)
    if type(model_name) in [list]:
        models = {}
        for mn in model_name:
            models[mn] = init_match_model_single(mn, device)
        return models
    else:
        return {model_name:init_match_model_single(model_name, device)}

def init_match_model_single(model_name, device):
    if model_name == 'roma':
        from romatch import roma_outdoor
        model = roma_outdoor(device=device,coarse_res=560,symmetric=True).to(device).eval()
    elif 'romav2' in model_name:
        from romav2 import RoMaV2
        model = RoMaV2(device=device).eval()
        setting = model_name.split('-')[-1]
        if setting == 'precise':
            pass
        elif setting == 'subprecise':
            model.H_hr = 800
            model.W_hr = 800
        elif setting == 'base':
            model.apply_setting('base')
            model.bidirectional = True
        elif setting == 'fast':
            model.apply_setting('fast')
            model.bidirectional = True
        else:
            raise NotImplementedError(f"Setting {setting} not implemented for RoMaV2.")
    elif 'ufm' in model_name:
        from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement
        from uniflowmatch.models.ufm import UniFlowMatchConfidence
        if model_name == 'ufm-refine':
            model = UniFlowMatchClassificationRefinement.from_pretrained("matching/models/ufm-refine", local_files_only=True)
            model = model.to(device).eval()
        elif model_name == 'ufm-refine-980':
            model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine-980")
            model = model.to(device).eval()
        elif model_name == 'ufm-base':
            model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base")
            model = model.to(device).eval()
        elif model_name == 'ufm-base-980':
            model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base-980")
            model = model.to(device).eval()
    elif 'mast3r' in model_name:
        raise NotImplementedError("MASt3R is not supported in the current codebase.")
        model = AsymmetricMASt3R.from_pretrained(mast3r_weights_path).to(device).eval()
        retriever = Retriever(retrieval_model, device=device, backbone=model)
        model = (model, retriever)
    else:
        raise NotImplementedError
    return model

import torch
import torch.nn.functional as F
from lightglue import SuperPoint
def homo(x):
    return torch.cat([x, torch.ones_like(x[...,-1:])], axis=-1)


def extract_query_points_lrdense(images_hr, lr_h, lr_w, hr_to_lr):
    device = images_hr.device
    N, C, _, _ = images_hr.shape
    dense_grid = torch.stack(torch.meshgrid(torch.arange(lr_w), torch.arange(lr_h), indexing='xy'), dim=-1).to(device)
    dense_lr = dense_grid.view(-1,2).float()
    lr_to_hr = closed_form_inverse_K(hr_to_lr)
    dense_hr = torch.einsum('ij,nj->ni', lr_to_hr, homo(dense_lr))[:,:2] #(N,2)
    sp_extractor = SuperPoint(nms_radius=0, max_num_keypoints=None, detection_threshold=0, remove_borders=None).to(device)  
    query_points = {"hr": [dense_hr for ni in range(N)], "lr": [dense_lr for ni in range(N)], "scores":[]}
    for ni in range(N):
        sp_results = sp_extractor.extract(images_hr[ni])
        xys_hr = sp_results['keypoints'][0] 
        xys_lr = torch.einsum('ij,nj->ni', hr_to_lr, homo(xys_hr))[...,:2].round().long()
        xys_lr[:,1] = xys_lr[:,1].clamp(0, lr_h-1)
        xys_lr[:,0] = xys_lr[:,0].clamp(0, lr_w-1)
        xys_lr_1d = xys_lr[:,1]*lr_w+xys_lr[:,0]
        xys_lr_1d = xys_lr_1d.clamp(0,lr_h*lr_w)
        scores = torch.zeros([dense_lr.shape[0]],dtype=torch.float32).to(device) #In high resolution
        src = sp_results['keypoint_scores'][0] 
        index = xys_lr_1d
        scores = torch.scatter_reduce(input=scores, dim=0, index=index, src=src, include_self=False, reduce='mean') #reduce to low-resolution
        query_points['scores'].append(scores)
    return query_points

def match_images(match_models, images_hr, lr_h, lr_w, hr_to_lr, output_dir=None):
    """
    Use (multiple) matching models to extract matches.
    images_hr: [N,C,H,W] the input high-resolution images
    images_lr: [N,C,h,w] the target low-resolution images, whose coordinates are to be matched
    hr_to_lr: [N,3,3]  the homography from high-res to low-res images
    """
    device = images_hr.device
    N, C, hr_h, hr_w = images_hr.shape
    query_points = extract_query_points_lrdense(images_hr, lr_h, lr_w, hr_to_lr)

    if len(match_models) == 1:
        match_result = run_single_matcher(list(match_models.keys())[0], list(match_models.values())[0], query_points, images_hr, hr_to_lr, output_dir)
        return {
            'sp_scores': torch.stack(query_points['scores'], axis=0),  #Nsrc,H*W
            'pred_scores': match_result['pred_scores'].view(N,N,lr_h*lr_w).permute(1,0,2).contiguous(),  #Nsrc, Ntgt, H, W ->Ntgt,Nsrc,H,W
            'pred_cycle_error': match_result['pred_cycle_error'].view(N,N,lr_h*lr_w).permute(1,0,2).contiguous(),  #Ntgt,Nsrc,H,W
            'pred_matches_lr': match_result['pred_matches_lr'].view(N,N,lr_h*lr_w,2).permute(1,0,2,3).contiguous(),  #Ntgt,Nsrc,H,W
        }
    else:
        match_result_ensembled = {
            'pred_matches_lr': [],
            'pred_scores': [],
            'pred_cycle_error': [],
        }
        match_result_dict = {}
        for model_name, single_model in match_models.items():
            match_result_single = run_single_matcher(model_name, single_model, query_points, images_hr,  hr_to_lr, output_dir)
            match_result_dict[model_name] = {}
            for key in  match_result_single:
                match_result_dict[model_name][key] = [mm for mm in match_result_single[key]]
            for key in match_result_ensembled:
                match_result_ensembled[key].append(match_result_single[key])
        match_result_ensembled = {key: torch.stack(match_result_ensembled[key], axis=0) for key in match_result_ensembled} #Nmodel,Nsrc,Ntgt,H*W,...
        #For each pairwise track [m,s,t,hw] # sorted by cycle error
        metric_to_sort = match_result_ensembled['pred_cycle_error'] #Nmodel,Nsrc,Ntgt,H*W
        #Set those invalid matches to large value
        metric_to_sort[match_result_ensembled['pred_scores']<1e-5] = 1e6
        selected_idx = torch.argmin(metric_to_sort, dim=0) #(Nsrc,Ntgt,H*W)
        Nmodel = selected_idx.shape[0]
        match_result = {}
        for key in match_result_ensembled:
            #M Nsrc Ntgt H*W
            Ns, Nt, HW, *rest = match_result_ensembled[key].shape[1:]
            index = selected_idx.view(1,Ns,Nt,HW, *([1]*len(rest))).expand(-1,-1,-1,-1,*rest) #(1,Nsrc,Ntgt,H*W,1,...)
            match_result[key] = match_result_ensembled[key].gather(0, index).squeeze(0) #(Nsrc,Ntgt,H*W,...)
            #Turn the first axis to list axis
            match_result[key] = [mm for mm in match_result[key]]
        match_result['source'] = selected_idx #(Nsrc,Ntgt,H*W)
        return {
            'sp_scores': torch.stack(query_points['scores'], axis=0),  #Nsrc,H*W
            'pred_scores': torch.stack(match_result['pred_scores'],axis=1).view(N,N,lr_h*lr_w),  #Ntgt,Nsrc,H,W
            'pred_cycle_error': torch.stack(match_result['pred_cycle_error'],axis=1).view(N,N,lr_h*lr_w),  #Ntgt,Nsrc,H,W
            'pred_matches_lr': torch.stack(match_result['pred_matches_lr'],axis=1).view(N,N,lr_h*lr_w,2),  #Ntgt,Nsrc,H,W
        }

def run_single_matcher(model_name, model, query_points, images_hr,  hr_to_lr, output_dir=None): #output_dir is used by Mast3r
    if 'roma' in model_name or 'ufm' in model_name:
        match_result = match_dense(
            model = model,
            model_name=model_name, 
            query_points_hr=query_points['hr'], images_hr=images_hr, hr_to_lr=hr_to_lr, 
            output_min_resolution=800) #800 is for roma
    elif 'mast3r' in model_name:
        raise NotImplementedError("MASt3R is not supported in the current codebase.")
        cache_dir = os.path.join(output_dir, 'mast3r_cache')
        match_result = match_mast3r(
        model=model[0],retriever=model[1], 
        images_hr=images_hr, images_lr=images_lr_TODO, hr_to_lr=hr_to_lr, 
        cache_dir=cache_dir)
    else:
        raise NotImplementedError
    match_result = {key: torch.stack(match_result[key], axis=0) if isinstance(match_result[key], list) else match_result[key] for key in match_result} #Nsrc Ntgt H W 2
    return match_result