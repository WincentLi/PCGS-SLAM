import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader
from utils import find_ellipsoid_nds_quick, depth_map_2_nds_map, find_pixel_nds_list, losses_nds_sin, cul_weights, decline_current_mask
# 根目录
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将_BASE_DIR插入到sys.path列表的最前面
sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
                                        ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset,
                                        ScannetPPDataset, NeRFCaptureDataset)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion, quat_mult
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

# 指定了几个数据集合，用于初始化参数
def get_dataset(config_dict, basedir, sequence, **kwargs):  # **kwargs允许函数输入额外的参数，额外的参数储存在一个字典中
    if config_dict["dataset_name"].lower() in ["icl"]:  # .lower()是转换小写指令
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels  二维张量
    # torch.meshgrid(..., indexing='xy') 接受这两个一维张量作为输入，并生成两个二维张量：x_grid和y_grid。x_grid中的每个元素都表示对应像素的x坐标，而y_grid中的每个元素都表示对应像素的y坐标。参数indexing='xy'指定了坐标轴的排序方式，其中'xy'表示x坐标是第一个维度（列），y坐标是第二个维度（行），这与许多图像处理库（如OpenCV）的默认行为相匹配。
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud 从相机生成3d点云
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)  # 4维齐次坐标
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]  # @矩阵乘法  最终取前三维，作为世界坐标系下的坐标
    else:
        pts = pts_cam  # 否则就是相机坐标系下的坐标

    # Compute mean squared distance for initializing the scale of the Gaussians 计算高斯尺度初始化的均方距离
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius) 一种很快的初始化模式，远处的高斯半径大
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud 颜色
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, 
                              mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]  # pose is extrinsics

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera  color的shape属性：高、宽、通道数量
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())  # 这里.detach将w2c固定，在反响传递时只获取w2c的值而不会将梯度传播到历史
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)  初始点云
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)  # 压缩为1维张量
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        # 经过下面的操作，mean和旋转矩阵变换到相机坐标系下，并且进行了视角的筛选
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
            #print(f"1深度损失：{losses['depth']}")
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
            #print('\n')
            #print(f"2深度损失：{losses['depth']}")
#############NORMAL LOSS######################################
    ## 获取旋转矩阵和尺度向量，不用rendervar的旋转矩阵因为里面归一化处理了，scales进行了exp处理
    scales_wv = torch.exp(params['log_scales'])  # 世界坐标系下的尺度向量（N，3）
    opacity_gs = torch.sigmoid(params['logit_opacities']).squeeze(1)  # (N, 1)->(N,)
    # Normal loss， 仅仅在mapping模式下进行  TODO: tracking阶段是否需要
    # print(curr_data['depth'].shape)  # (1, 480, 640)
    currect_depth_gt = curr_data['depth'].squeeze(0).detach()  # (480, 640)
    currect_depth_re = depth.squeeze(0)
    current_mask = mask.squeeze(0)
    height, width = currect_depth_gt.shape
    # 超参数
    intrinsics = curr_data['intrinsics']

    # 视角内的高斯
    ## 内参
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]
    ## transformed_gaussians['means3D'] ： tensor转换后的高斯空间位置，这里可以用现成的
    means_x_cv = transformed_gaussians['means3D'][:, 0]  # w2c相机坐标系下(N,)
    means_y_cv = transformed_gaussians['means3D'][:, 1]
    means_z_cv = transformed_gaussians['means3D'][:, 2]

    norm_rotations_cv = transformed_gaussians['unnorm_rotations']
    ## 计算u，v
    means_u = (FX * means_x_cv / means_z_cv + CX).round().int()  # c2uv坐标系下(N,)
    means_v = (FY * means_y_cv / means_z_cv + CY).round().int()

    ## 根据uv和HW制作蒙板，需要根据蒙板调整的有means_u、means_v、unnorm_rotations_wv、scales_wv
    ## 创建蒙板
    mask_u = (means_u >= 0) & (means_u < width)
    mask_v = (means_v >= 0) & (means_v < height)
    combined_mask = mask_u & mask_v
    combined_mask = combined_mask.detach()
    # 初步筛选
    means_u = means_u[combined_mask]  # (n,)
    means_v = means_v[combined_mask]
    opacity_gs = opacity_gs[combined_mask].detach()  # 不更新这个
    scales_wv = scales_wv[combined_mask]  # (n, 3)
    norm_rotations_cv = norm_rotations_cv[combined_mask]
    ## 组合蒙板（两个条件都需要满足）
    #start_time = time.time()
    combined_mask = decline_current_mask(current_mask, means_u, means_v)
    combined_mask = combined_mask.detach()  # (All,)可以用来筛选第一个维度为N的张量，利用蒙板排除多个gs***（N，） combined_mask不需要更新，也不会影响梯度流
    #end_time = time.time()
    #print(f"组合蒙板运行时间： {end_time - start_time}秒")

    #start_time = time.time()
    weights_nd = cul_weights(currect_depth_gt, currect_depth_re, means_u, means_v)
    #end_time = time.time()
    #print(f"计算权重运行时间： {end_time - start_time}秒")

    ## 再次进行筛选
    means_u = means_u[combined_mask]  # (n,)
    means_v = means_v[combined_mask]
    opacity_gs = opacity_gs[combined_mask].detach()  # 不更新这个
    weights_nd = weights_nd[combined_mask]  # (n,)
    scales_wv = scales_wv[combined_mask]  # (n, 3)
    norm_rotations_cv = norm_rotations_cv[combined_mask]

    # 根据uv计算法线真值
    #start_time = time.time()
    nds_map_gt = depth_map_2_nds_map(currect_depth_re, intrinsics)
    normal_direction_gt_cv = find_pixel_nds_list(nds_map_gt, means_u, means_v)  # (n ,3)
    #end_time = time.time()
    #print(f"法线真值运行时间： {end_time - start_time}秒")

    # 求每个高斯核的协方差矩阵
    ### doneTODO：是否需要detach，tracking阶段不需要，mapping阶段需要，
    ###     ：此时的深度损失使用的是平均值，那么nd也可以使用平均值
    ### TODO:preprocess the depth map to nd-map
    ###     : dont use the nd-loss for every iter, only in several ba-process

    #start_time = time.time()
    rotation_matrixs_cv = build_rotation(norm_rotations_cv)  # (n, 3, 3)
    # 计算短轴方向
    min_axis_directions = find_ellipsoid_nds_quick(rotation_matrixs_cv, scales_wv)
    #end_time = time.time()
    #print(f"短轴方向运行时间： {end_time - start_time}秒")

    # 计算损失，用average进行计算
    normal_losses = losses_nds_sin(normal_direction_gt_cv, min_axis_directions)  # (n,)

    # doneTODO：设置尺度：1、filtered_opacity_gs决定太透明的高斯不能起强作用；2、filtered_weights_nd决定nd的量级要和depth相当

    weighted_normal_loss = opacity_gs * weights_nd * normal_losses  # (n,)
    # Normal loss
    if use_l1:
        if tracking:
            losses['normal'] = 0.05 * weighted_normal_loss.sum()  # 均值 由于点少，这里的weighted_normal_loss比较小 TODO：tracking阶段如何衡量，感觉这个对tracking有害
            #print(f"法线损失：{losses['normal']}")
        else:
            losses['normal'] = 0.2 * weighted_normal_loss.mean()  # 均值
            #print(f"法线损失：{losses['normal']}")
    # 是否有一个纹理上的法线约束，比如类似边窗滤波一样，让高斯长轴往固定的8个方向去拟合，注意这个只能是对于较小的高斯的约束
#########################################################
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, 
                      time_idx, mean_sq_dist_method, gaussian_distribution):
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables


def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def rgbd_slam(config: dict):  # : dict限制了config的类型为字典
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:  # 这里的use_depth_loss_thres应该是深度损失阈值
        # 若没有，则添加两个key，不使用深度阈值，且将深度阈值设置较高，这里应该是对应公式（9）
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        # gs-slam的优化过程和深度l1损失和颜色l1损失有关，原始的gs和颜色l1损失和l D-SSIM损失有关
        config['tracking']['visualize_tracking_loss'] = False  # 视觉跟踪损失？
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "isotropic"  # 各项同性，表示使用圆球而不是椭球
    print(f"{config}")

    # Create Output Directories 创建输出目录
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Init WandB 初始化WandB
    # WandB是一个机器学习实验跟踪、可视化和优化的平台
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        os.environ["WANDB_API_KEY"] = 'c35b4ef0397857c7929e4daaeb6751e0eb71b343' # 将引号内的*替换成自己在wandb上的key
        os.environ["WANDB_MODE"] = "offline"
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)

    # Get Device 加载设备
    device = torch.device(config["primary_device"])  # 使用第一个GPU

    # Load Dataset 加载数据集
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]  # 使用默认参数
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])  # 加载参数
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False  # 忽略坏的啥？？
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True  # 拆分训练
    if "densification_image_height" not in dataset_config:  # 致密化尺寸
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:  # 跟踪图像尺寸
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False
    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # Init seperate dataloader for densification if required  是否为致密化单独设置数据加载器
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        densify_dataset=densify_dataset,
                                                                        gaussian_distribution=config['gaussian_distribution'])                                                                                                                  
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames, 
                                                                                        config['scene_radius_depth_ratio'],
                                                                                        config['mean_sq_dist_method'],
                                                                                        gaussian_distribution=config['gaussian_distribution'])
    
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1], 
                                    tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # Load Checkpoint
    if config['load_checkpoint']:
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            color, depth, _, gt_pose = dataset[time_idx]
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
    else:
        checkpoint_time_idx = 0
    
    # Iterate over Scan
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
        # Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose = dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        
        # Initialize Data for Tracking
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx,
                                  'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])

        # Tracking
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                                   plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter)
                if config['use_wandb']:
                    # Report Loss
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                            wandb_run=wandb_run, wandb_step=wandb_tracking_step, wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if config['use_wandb']:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                        wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if seperate_densification_res:
                    # Load RGBD frames incrementally instead of all frames
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                else:
                    densify_curr_data = curr_data

                # Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians(params, variables, densify_curr_data, 
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'], config['gaussian_distribution'])
                post_num_pts = params['means3D'].shape[0]
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Select Keyframes for Mapping
                num_keyframes = config['mapping_window_size']-2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False) 

            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                # Randomly select a frame until current time step amongst keyframes
                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                if selected_rand_keyframe_idx == -1:
                    # Use Current Frame Data
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                else:
                    # Use Keyframe Data
                    iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                    iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                # Loss for current frame
                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True)
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx)
                        else:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx, global_logging=True)
                        else:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
        
        # Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
        # Increment WandB Time Step
        if config['use_wandb']:
            wandb_time_step += 1

        torch.cuda.empty_cache()

    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    if config['use_wandb']:
        wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                       "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                       "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                       "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                       "Final Stats/step": 1})
    
    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])
        else:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    save_params(params, output_dir)

    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # ArgumentParser()对象

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()  # 解析parser的参数，赋给args

    # SourceFileLoader加载Python源文件，不需要import，作用和import一样
    # .load_module(): 这是SourceFileLoader的一个方法，它实际上执行模块的加载过程。
    # 加载成功后，它会返回一个模块对象，这个对象包含了源文件中定义的所有函数、类和变量。
    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    # config是一个超参文件
    seed_everything(seed=experiment.config['seed'])  # seed表识了哪个数据集的哪个序列的序号
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))  # 创建一个config.py内容是args.experiment这个py文件的内容

    rgbd_slam(experiment.config)  # 这里的config是一个字典，限制了运行过程的一些参数，如关键帧间隔
