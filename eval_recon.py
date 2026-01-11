# adapted from https://github.com/cvg/nice-slam
import numpy as np
import open3d as o3d
import torch
import trimesh
import cv2
import os 
import sys
import json
from planar.cull_mesh import cull_mesh, mask_mesh
from argparse import ArgumentParser
from arguments import ModelParams, PriorParams
from scipy.spatial import cKDTree as KDTree


def completion_ratio(gt_points, rec_points, dist_th): 
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(float))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)  
    acc = np.mean(distances)
    return acc, distances


def completion(gt_points, rec_points):
    rec_points_kd_tree = KDTree(rec_points)
    distances, _ = rec_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp, distances


def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)


def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances, indices


def evaluate_recon_metric(model, prp, method, custom_mesh_path=None):  
    """
    3D reconstruction metric.

    """
    print(">>> Step 1: Loading Meshes")
    if method == 'planargs':
        rec_meshfile = os.path.join(model.model_path, "mesh", "tsdf_fusion_post.ply")
        output_dir = os.path.join(model.model_path, "mesh_eval")
    else:
        rec_meshfile = custom_mesh_path
        output_dir = os.path.join(os.path.dirname(rec_meshfile), "mesh_eval")

    gt_meshfile = os.path.join(model.source_path, "mesh.ply")

    if not os.path.exists(rec_meshfile):
        sys.exit(f"Error: Reconstructed mesh not found at {rec_meshfile}")
    if not os.path.exists(gt_meshfile):
        sys.exit(f"Error: GT mesh not found at {gt_meshfile}")
    
    print(f"Output Directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    mesh_rec = trimesh.load(rec_meshfile)

    if method == 'dn_splatter':
        print("    -> Applying dn_splatter coordinate correction (XYZ -> XZY, Y-flip)")
        vertices = np.asarray(mesh_rec.vertices)
        transformed_vertices = vertices.copy()
        transformed_vertices[:, [0, 1, 2]] = vertices[:, [0, 2, 1]]
        transformed_vertices[:, 1] *= -1
        mesh_rec.vertices = o3d.utility.Vector3dVector(transformed_vertices)
        
    print(f">>> Step 2: Alignment")
    # Align the coordinates of the ground truth and the reconstructed mesh.
    align_params_path = os.path.join(model.source_path, "align_params.npz")
    print(f"    -> Loading alignment params from: {align_params_path}")
    align_dict = np.load(align_params_path)
    align_transform = align_dict['align_transform']
    align_scale = align_dict['align_scale']

    mesh_rec.apply_scale(align_scale)
    mesh_rec = mesh_rec.apply_transform(align_transform)
    
    print(">>> Step 3: Culling Mesh")
    # Exclude the mesh parts that are not visible from the training viewpoints.
    trans_dict = {
        "Rd": torch.from_numpy(align_transform[:3, :3]).cuda().float(),
        "Td": torch.from_numpy(align_transform[:3, 3:]).cuda().float(),
        "scale": 1 / torch.tensor(align_scale).cuda().float()
    }

    mesh_gt = trimesh.load(gt_meshfile)
    mesh_gt = cull_mesh(mesh_gt, model.source_path, trans_dict, model.eval)
    gt_obb = mesh_gt.bounding_box_oriented
    obb2aabb_transform = np.linalg.inv(gt_obb.transform) 
    mesh_gt.apply_transform(obb2aabb_transform)

    mesh_rec = cull_mesh(mesh_rec, model.source_path, trans_dict, model.eval)
    mesh_rec.apply_transform(obb2aabb_transform)
    
    icp_transform = align_dict['icp_transform']
    mesh_rec = mesh_rec.apply_transform(icp_transform)

    print(">>> Step 4: Masking Outliers")
    # Exclude points outside the ground-truth bounding box.
    mesh_rec = mask_mesh(mesh_gt, mesh_rec, model.model_path, prp.maskmesh_thresh)
    mesh_rec.export(os.path.join(output_dir, f"mask_mesh.ply"))
    mesh_gt.export(os.path.join(output_dir, f"gt_mesh.ply"))

    print(">>> Step 5: Calculating Metrics")
    rec_pc = trimesh.sample.sample_surface(mesh_rec, prp.mesh_sample)
    rec_pc_tri = trimesh.PointCloud(vertices =rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, prp.mesh_sample)
    gt_pc_tri = trimesh.PointCloud(vertices = gt_pc[0])

    accuracy_rec, dist_d2s = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec, dist_s2d = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, prp.fscore_thresh)
    precision_ratio_rec = completion_ratio(rec_pc_tri.vertices, gt_pc_tri.vertices, prp.fscore_thresh)

    # f-score has a threshold of 5cm
    precision_ratio_rec = np.mean((dist_d2s < prp.fscore_thresh).astype(float)) 
    completion_ratio_rec = np.mean((dist_s2d < prp.fscore_thresh).astype(float)) 
    fscore = 2 * precision_ratio_rec * completion_ratio_rec / (completion_ratio_rec + precision_ratio_rec)
    
    # normal consistency 
    pointcloud_pred, idx = mesh_rec.sample(prp.mesh_sample, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normal_pred = mesh_rec.face_normals[idx]

    pointcloud_trgt, idx = mesh_gt.sample(prp.mesh_sample, return_index=True)
    pointcloud_trgt = pointcloud_trgt.astype(np.float32)
    normal_trgt = mesh_gt.face_normals[idx]

    _, index1 = nn_correspondance(pointcloud_pred, pointcloud_trgt)
    _, index2 = nn_correspondance(pointcloud_trgt, pointcloud_pred)

    normal_acc = np.abs((normal_pred * normal_trgt[index2.reshape(-1)]).sum(axis=-1)).mean()
    normal_comp = np.abs((normal_trgt * normal_pred[index1.reshape(-1)]).sum(axis=-1)).mean()
    normal_avg = (normal_acc + normal_comp) * 0.5

    # convert to cm
    accuracy_rec *= 100  
    completion_rec *= 100  
    # convert to %
    completion_ratio_rec *= 100 
    precision_ratio_rec *= 100  
    fscore *= 100
    normal_acc *= 100
    normal_comp *= 100
    normal_avg *= 100

    metrics = {
        "accuracy": accuracy_rec,
        "completion": completion_rec,
        "precision_ratio": precision_ratio_rec,
        "completion_ratio": completion_ratio_rec,
        "chamfer_distance": (accuracy_rec + completion_rec)/2,
        "fscore": fscore,
        "normal_average": normal_avg
    }
    
    print(f"Chamfer Distance: {metrics['chamfer_distance']:.4f}")
    print(f"F-Score: {metrics['fscore']:.4f}")
    print(f"Normal Consistency: {metrics['normal_average']:.4f}")

    file_path = os.path.join(output_dir, "output" + ".json")
    with open(file_path, 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    print(">>> Step 6: Visualizing Error")
    stl_alpha = (dist_s2d.clip(max = prp.vis_dist) / prp.vis_dist).reshape(-1, 1)
    im_gray = (stl_alpha * 255).astype(np.uint8)
    stl_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)[:,0,[2, 0, 1]] / 255.
    write_vis_pcd(os.path.join(output_dir, 'mesh_s2d.ply'), gt_pc_tri.vertices, stl_color)


if __name__ == '__main__':
    parser = ArgumentParser(description="Eval reconstructed mesh parameters")
    model = ModelParams(parser, sentinel=True)
    prp = PriorParams(parser)
    parser.add_argument('--method', type=str, default='planargs', help='Reconstruction method name')
    parser.add_argument('--recon_mesh_path', type=str, default=None, help="Explicit path to the mesh file. Only required if method is NOT 'planargs'.")
    args = parser.parse_args()
    
                                                                                                                                                         
    evaluate_recon_metric(model.extract(args), prp.extract(args), method=args.method, custom_mesh_path=args.recon_mesh_path)

