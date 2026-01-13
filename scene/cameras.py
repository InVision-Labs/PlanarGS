#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from PIL import Image
import os, cv2
import json
from common_utils.general_utils import PILtoTorch
from common_utils.graphics_utils import getWorld2View2, getProjectionMatrix, get_k, ThickenLines

_WARNED_OBJECT_MASK_DIR = False


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, resolution, path, 
                 params, image_name, uid, 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = resolution[0]
        self.image_height = resolution[1]
        image_path = os.path.join(path, "images", image_name[0] + image_name[1])
        geomprior_folder = os.path.join(path, "geomprior")
        depthconf_path = os.path.join(geomprior_folder, "resized_confs", image_name[0] + ".npy")
        planarmask_path = os.path.join(path, "planarprior/mask", image_name[0] + ".npy")
        weights_path = os.path.join(geomprior_folder, "depth_weights.json")

        original_image = Image.open(image_path)
        resized_image = original_image.resize(resolution)
        self.gt_image = PILtoTorch(resized_image, resolution)[:3, ...].clamp(0.0, 1.0).to(self.data_device) # (3, H, W)
        
        canny_mask = cv2.Canny(np.array(resized_image), params.canny_thresh[0], params.canny_thresh[1])/255.  
        canny_masker = torch.from_numpy(canny_mask).clamp(0.0, 1.0).to(self.data_device)
        self.canny_mask = 1 - ThickenLines(canny_masker, kernel_size=5)

        # Optional binary object mask to ignore pixels during rendering / TSDF fusion.
        # Convention: 1 = keep, 0 = mask out. Stored as float tensor (H, W) in [0, 1].
        self.object_mask = None
        if getattr(params, "object_mask_dir", ""):
            mask_dir_raw = params.object_mask_dir
            candidate_dirs = []
            if os.path.isabs(mask_dir_raw):
                candidate_dirs.append(mask_dir_raw)
            else:
                # Common cases:
                # - relative to dataset root (source_path): "sam2_masks/obj_000004"
                # - relative to current working directory: "scenes/.../sam2_masks/obj_000004"
                candidate_dirs.append(os.path.join(path, mask_dir_raw))
                candidate_dirs.append(os.path.abspath(mask_dir_raw))
            mask_dir = next((d for d in candidate_dirs if os.path.exists(d)), None)
            if mask_dir is None:
                global _WARNED_OBJECT_MASK_DIR
                if not _WARNED_OBJECT_MASK_DIR:
                    print(f"[Warning] object_mask_dir='{mask_dir_raw}' not found. Tried: {candidate_dirs}")
                    _WARNED_OBJECT_MASK_DIR = True
                mask_dir = None

            stem = image_name[0]
            if mask_dir is not None:
                candidates = [
                    os.path.join(mask_dir, stem + ".npy"),
                    os.path.join(mask_dir, stem + ".png"),
                    os.path.join(mask_dir, stem + ".jpg"),
                    os.path.join(mask_dir, stem + ".jpeg"),
                ]
                mask_path = next((p for p in candidates if os.path.exists(p)), None)
                if mask_path is not None:
                    if mask_path.endswith(".npy"):
                        m = np.load(mask_path)
                        m = m.squeeze()
                    else:
                        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        m = cv2.resize(m, resolution, interpolation=cv2.INTER_NEAREST)
                        if m.dtype != np.float32:
                            m = m.astype(np.float32)
                        # Accept 0/255 masks or 0/1 masks
                        if m.max() > 1.5:
                            m = m / 255.0
                        m = np.clip(m, 0.0, 1.0)
                        if getattr(params, "object_mask_invert", False):
                            m = 1.0 - m
                        self.object_mask = torch.from_numpy(m).to(self.data_device)

        if os.path.exists(planarmask_path):
            planar_mask = cv2.resize(np.load(planarmask_path).squeeze(0), resolution, interpolation=cv2.INTER_NEAREST)
            self.planarmask = torch.from_numpy(planar_mask).to(self.data_device).to(torch.int64)   
        else:
            self.planarmask = None

        if os.path.exists(geomprior_folder): 
            self.priordepth, priornormal = LoadGeomprior(geomprior_folder, image_name[0], resolution, self.data_device)
            self.priornormal = priornormal.permute(2,0,1) 

            depthconf_mask = (np.load(depthconf_path) > params.conf_thresh).astype(int)
            depthconf = cv2.resize(depthconf_mask, resolution, interpolation=cv2.INTER_NEAREST)
            self.depth_conf = torch.from_numpy(depthconf).clamp(0.0, 1.0).to(self.data_device) 
            if params.use_weights:
                with open(weights_path, 'r') as f:
                    weights = json.load(f)
                self.depth_weight = torch.tensor(weights[image_name[0]]).to(self.data_device) 
            else:
                self.depth_weight = None
        else:
            self.priordepth, self.depth_weight, self.depth_conf, self.priornormal = None, None, None, None

        self.K, self.inv_K = get_k(FoVx, FoVy, self.image_height, self.image_width, scale)
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()  #4x4
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()  #4x4
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)  #4x4
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        
    
    def get_rays(self, scale=1.0):
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
                    [(ix-self.Cx/scale) / self.Fx * scale,
                    (iy-self.Cy/scale) / self.Fy * scale,
                    torch.ones_like(ix)], -1).float().cuda()
        return rays_d


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def LoadGeomprior(geomprior_folder, image_name, resolution, device="cuda"):
    priordepth_path = os.path.join(geomprior_folder, "aligned_depth", image_name + ".npy")
    priornormal_path = os.path.join(geomprior_folder, "prior_normal", image_name + ".npy")
    priordepth = cv2.resize(np.load(priordepth_path), resolution, interpolation=cv2.INTER_NEAREST)
    priornormal = cv2.resize(np.load(priornormal_path), resolution, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(priordepth).to(device), torch.from_numpy(priornormal).to(device)