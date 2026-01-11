import numpy as np
import os,sys,json,cv2
import open3d as o3d
import torch
import trimesh
from argparse import ArgumentParser
from arguments import ModelParams, PriorParams
from planar.cull_mesh import cull_mesh
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat, read_extrinsics_text, read_points3D_binary

def o3d_icp_alignment(source_mesh_trimesh, target_mesh_trimesh, threshold, max_iter=2000):
    """
    Performs fine-grained Point-to-Point ICP alignment using Open3D.
    
    Args:
        source_mesh_trimesh: Source mesh to align (Trimesh object).
        target_mesh_trimesh: Target/GT mesh (Trimesh object).
        threshold: Distance threshold for ICP.
        max_iter: Maximum number of iterations.
        
    Returns:
        transformation: (4, 4) alignment transformation matrix.
    """

    # Convert Trimesh objects to Open3D TriangleMesh
    o3d_gt_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(target_mesh_trimesh.vertices),
        triangles=o3d.utility.Vector3iVector(target_mesh_trimesh.faces)
    )
    o3d_rec_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(source_mesh_trimesh.vertices),
        triangles=o3d.utility.Vector3iVector(source_mesh_trimesh.faces)
    )

    # Extract PointClouds for ICP
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)

    # Execute ICP registration (allowing scaling)
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc, 
        o3d_gt_pc, 
        threshold, 
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True), 
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )
    return reg_p2p.transformation

def select_reliable_reference_images(images_dict, points3D_array, top_k=5):
    """
    Selects the top-K most reliable reference images as localization anchors based on reprojection error.
    
    Args:
        images_dict: COLMAP images dictionary.
        points3D_array: COLMAP 3D points data.
        top_k: Number of best images to return.
    
    Returns:
        topk_images: List of Image objects with the lowest errors.
    """
    # Build a map from 3D Point ID to error
    point3D_error_map = {int(p[0]): p[1] for p in points3D_array}
    image_scores = []

    for image in images_dict.values():
        # Get all valid 3D point IDs observed by this image
        valid_ids = image.point3D_ids[image.point3D_ids != -1]
        valid_errors = [point3D_error_map[pid] for pid in valid_ids if pid in point3D_error_map]
        # Skip if too few observation points (<30), indicating unreliable pose
        if len(valid_errors) < 30:
            continue
        avg_error = np.mean(valid_errors)
        image_scores.append((avg_error, image))
    image_scores.sort(key=lambda x: x[0])
    topk_images = [item[1] for item in image_scores[:top_k]]
    return topk_images

def compute_alignment_scannetpp(colmap_path, model_path, gt_data_path):
    """
    Computes coarse alignment (Scale + Transform) for the ScanNet++ dataset.
    Logic:
    1. Find the view with the minimum error in COLMAP.
    2. Calculate the ratio (Scale) between rendered depth and GT depth.
    3. Combine COLMAP poses and GT poses to compute the final transformation matrix.
    """
    gs_render_depth_path = os.path.join(model_path,'train','ours_30000')

    # Read COLMAP data
    images_dict = read_extrinsics_binary(f"{colmap_path}/sparse/images.bin")
    _, _, points3D  = read_points3D_binary(f"{colmap_path}/sparse/points3D.bin")
    
    # Read GT data (ScanNet++ provides GT in colmap format)
    gt_images_dict = read_extrinsics_text(f"{gt_data_path}/colmap/images.txt")
    
    # Select best views
    topk_images = select_reliable_reference_images(images_dict, points3D)

    for best_image in topk_images:
        print(f"Using reference image: {best_image.name}")
        rotation = qvec2rotmat(best_image.qvec)
        translation = best_image.tvec.reshape(3, 1)
        T_rec2cam = np.eye(4)
        T_rec2cam[:3, :3] = rotation
        T_rec2cam[:3, 3:4] = translation

        # --- Compute Scale ---
        scale_list = []
        for key in images_dict:
            img_name = images_dict[key].name
            depth_render_path = f'{gs_render_depth_path}/renders_depth/{img_name.replace("JPG","npy")}'
            depth_gt_path = f'{gt_data_path}/depth/{img_name.replace("JPG","png")}'
            
            if not os.path.exists(depth_render_path) or not os.path.exists(depth_gt_path):
                continue

            # Load and compute median ratio
            depth_render = np.load(depth_render_path)
            depth_gt = cv2.imread(depth_gt_path,cv2.IMREAD_ANYDEPTH)/1000.0 # mm to meter
            
            scale = np.median(depth_gt[depth_gt>0]) / np.median(depth_render[depth_render>0])
            scale_list.append(scale)

        if not scale_list:
            raise RuntimeError(f"Fatal Error: No valid depth pairs found in {gs_render_depth_path} or {depth_gt_path}")
        
        avg_scale = np.median(scale_list)
        # print("median scale ",np.median(scale_list),scale_list.max(),scale_list.min(),scale_list.mean())    
        
        # --- Compute Transform ---
        for gt_key in gt_images_dict:
            # Find the corresponding GT image
            if best_image.name == gt_images_dict[gt_key].name:
                gt_img = gt_images_dict[gt_key]
                
                # T_gt_world_to_cam: GT World -> Camera
                gt_R = qvec2rotmat(gt_img.qvec)
                gt_t = gt_img.tvec.reshape(3, 1)
                T_gt_world2cam = np.eye(4)
                T_gt_world2cam[:3, :3] = gt_R
                T_gt_world2cam[:3, 3:4] = gt_t
                
                # T_cam_to_gt_world: Camera -> GT World
                T_cam2gt_world = np.linalg.inv(T_gt_world2cam)
                
                # Apply Scale to the translation component of the reconstruction (s * t)
                # Note: In COLMAP models T = [R|t], X_cam = R*X_world + t
                # We are scaling the entire reconstructed world.
                T_rec2cam[:3, 3] *= avg_scale
                
                # Final transform: Recon (Scaled) -> Cam -> GT World
                final_transform = T_cam2gt_world @ T_rec2cam
                return final_transform, avg_scale

def compute_alignment_replica(colmap_path, model_path, gt_data_path):
    """
    Computes coarse alignment for the Replica dataset.
    Replica uses `traj.txt` to store camera poses (c2w).
    """
    gs_render_depth_path = os.path.join(model_path,'train','ours_30000')
    images_dict = read_extrinsics_binary(f"{colmap_path}/sparse/images.bin")
    _, _, points3D  = read_points3D_binary(f"{colmap_path}/sparse/points3D.bin")
    
    topk_images = select_reliable_reference_images(images_dict, points3D)
    
    # Load GT trajectory
    gt_poses_c2w = np.loadtxt(f"{gt_data_path}/traj.txt")

    for best_image in topk_images:
        rotation = qvec2rotmat(best_image.qvec)
        translation = best_image.tvec.reshape(3, 1)
        T_rec2cam = np.eye(4)
        T_rec2cam[:3, :3] = rotation
        T_rec2cam[:3, 3:4] = translation
        # Replica dataset ID mapping assumption: pose_id = 20 * (colmap_id - 1) + 10
        pose_id = 20 * int(best_image.id-1) + 10
        T_cam2world_gt = gt_poses_c2w[pose_id].reshape(4, 4)

        scale_list = []
        for key in images_dict:
            # --- ID Mapping Logic ---
            # The original Replica dataset contains 2000 images.
            # This dataset was uniformly downsampled to 100 images (Stride = 2000 / 100 = 20).
            # We map the COLMAP ID (subset 1-100) back to the original GT pose ID.
            pose_id = 20 * int(key-1) + 10
            img_name = images_dict[key].name

            depth_render_path = f'{gs_render_depth_path}/renders_depth/{img_name.replace("jpg","npy")}'
            depth_gt_path = f'{gt_data_path}/depth/depth{pose_id:06}.png'
            
            if not os.path.exists(depth_render_path) or not os.path.exists(depth_gt_path):
                continue
            
            depth_render = np.load(depth_render_path) 
            depth_gt = cv2.imread(depth_gt_path,cv2.IMREAD_UNCHANGED)/6553.5
            scale = np.median(depth_gt[depth_gt>0]) / np.median(depth_render[depth_render>0])
            if scale > 0:
                scale_list.append(scale)

        if not scale_list:
            raise RuntimeError(f"Fatal Error: No valid depth pairs found in {gs_render_depth_path} or {depth_gt_path}")
        
        avg_scale = np.median(scale_list)
        # print("median scale ",np.median(scale_list),scale_list.max(),scale_list.min(),scale_list.mean())    
        
        T_rec2cam[:3, 3] *= avg_scale
        final_transform = T_cam2world_gt @ T_rec2cam
        return final_transform, avg_scale

def compute_alignment_mushroom(colmap_path, model_path, gt_data_path):
    """
    Computes coarse alignment for the Mushroom iPhone dataset.
    Involves additional coordinate system transformations (iPhone ARKit vs COLMAP).
    """
    gs_render_depth_path = os.path.join(model_path,'train','ours_30000')
    images_dict = read_extrinsics_binary(f"{colmap_path}/sparse/images.bin")
    _, _, points3D  = read_points3D_binary(f"{colmap_path}/sparse/points3D.bin")
    
    topk_images = select_reliable_reference_images(images_dict, points3D)
    
    for best_image in topk_images:
        print(f"Ref Image: {best_image.name}")
        # T_rec2cam
        rotation = qvec2rotmat(best_image.qvec)
        translation = best_image.tvec.reshape(3, 1)
        T_rec2cam = np.eye(4)
        T_rec2cam[:3, :3] = rotation
        T_rec2cam[:3, 3:4] = translation

        scale_list = []
        for key in images_dict:
            img_name = images_dict[key].name
            depth_render_path = f'{gs_render_depth_path}/renders_depth/{img_name.replace("jpg","npy")}'
            depth_gt_path = f'{gt_data_path}/depth/{img_name.replace("jpg","png")}'
            
            if not os.path.exists(depth_render_path) or not os.path.exists(depth_gt_path):
                continue
            
            depth_render = np.load(depth_render_path)
            depth_gt = cv2.imread(depth_gt_path,cv2.IMREAD_ANYDEPTH)/1000.0
            
            scale = np.median(depth_gt[depth_gt>0]) / np.median(depth_render[depth_render>0])
            scale_list.append(scale)

        if not scale_list:
            raise RuntimeError(f"Fatal Error: No valid depth pairs found in {gs_render_depth_path} or {depth_gt_path}")
        
        avg_scale = np.median(scale_list)
        # print("median scale ",np.median(scale_list),scale_list.max(),scale_list.min(),scale_list.mean())    

        # Read specific transformation files
        cam_params = json.load(open(os.path.join(gt_data_path,"transformations_colmap.json")))
        # iPhone GT usually requires an additional ICP calibration matrix
        T_colmapgt_to_gt = np.array(json.load(open(os.path.join(gt_data_path, "icp_iphone.json")))["gt_transformation"]).reshape(4, 4)
        frames = cam_params["frames"]
        for frame in frames:
            if best_image.name in str(frame["file_path"]):
                T_cam2colmapgt = np.array(frame["transform_matrix"]).reshape(4, 4)
                # Flip iPhone/COLMAP coordinate axes (usually Y/Z)
                T_cam2colmapgt[0:3, 1:3] *= -1
                T_rec2cam[:3,3]*= avg_scale        
                final_transform = T_colmapgt_to_gt @ T_cam2colmapgt @ T_rec2cam
                return final_transform, avg_scale

def run_preprocessing_pipeline(dataset_type, model, gt_data_path):
    """
    Main Pipeline:
    1. Load reconstructed Mesh and GT Mesh.
    2. Compute Coarse Alignment.
    3. Cull GT Mesh to match the reconstructed area.
    4. Perform ICP Refinement.
    5. Save alignment parameters for subsequent evaluation scripts.
    """
    print(">>> Step 1: Loading Meshes")
    rec_meshfile = os.path.join(model.model_path, "mesh", "tsdf_fusion_post.ply") 
    gt_meshfile = os.path.join(model.source_path, "mesh.ply")

    if not os.path.exists(rec_meshfile):
        sys.exit(f"Error: Reconstructed mesh not found at {rec_meshfile}")
    if not os.path.exists(gt_meshfile):
        sys.exit(f"Error: GT mesh not found at {gt_meshfile}")
    
    mesh_rec = trimesh.load(rec_meshfile)
    
    print(f">>> Step 2: Coarse Alignment ({dataset_type})")
    align_transform = None
    align_scale = 1.0

    if dataset_type == "mushroom":
        align_transform, align_scale = compute_alignment_mushroom(model.source_path, model.model_path, gt_data_path)
    elif dataset_type == "replica":
        align_transform, align_scale = compute_alignment_replica(model.source_path, model.model_path, gt_data_path)
    elif dataset_type == "scannetpp":
        align_transform, align_scale = compute_alignment_scannetpp(model.source_path, model.model_path, gt_data_path)
    else:
        sys.exit(f"Error: Unknown dataset type '{dataset_type}'")

    # Apply coarse alignment to reconstructed Mesh
    mesh_rec.apply_scale(align_scale)
    mesh_rec = mesh_rec.apply_transform(align_transform)

    # Save initial alignment parameters
    align_params_dict = {
        'align_transform': align_transform,
        'align_scale': align_scale
    }

    print(">>> Step 3: Culling Ground Truth Mesh")
    # Prepare transformation matrix for Culling
    # Culling logic is usually done in the original GT space, checking which parts are within the camera frustum via transformation.
    trans_dict = {
        "Rd": torch.from_numpy(align_transform[:3, :3]).float().cuda(),
        "Td": torch.from_numpy(align_transform[:3, 3:]).float().cuda(),
        "scale": 1/align_scale
    }    
    mesh_gt = trimesh.load(gt_meshfile)
    # Cull invisible parts of GT based on view
    mesh_gt = cull_mesh(mesh_gt, model.source_path, trans_dict, model.eval)
    # Use GT's Oriented Bounding Box (OBB) to normalize coordinates to Axis-Aligned (AABB) space to facilitate ICP convergence
    gt_obb = mesh_gt.bounding_box_oriented
    obb2aabb_transform = np.linalg.inv(gt_obb.transform)

    mesh_gt.apply_transform(obb2aabb_transform)
    mesh_rec.apply_transform(obb2aabb_transform)

    align_params_dict['obb2aabb_transform'] = obb2aabb_transform
    # Calculate Mesh size to dynamically set ICP thresholds
    scene_size = np.linalg.norm(mesh_gt.bounding_box_oriented.extents)


    print(">>> Step 4: ICP Refinement")
    # Two-stage ICP: Coarse -> Fine
    icp_thresholds = [0.35 * scene_size / 7.6, 0.1 * scene_size / 7.6]

    print(f"   ICP Pass 1 (Thresh: {icp_thresholds[0]:.4f})...")
    icp_mat_1 = o3d_icp_alignment(mesh_rec, mesh_gt, threshold=icp_thresholds[0])
    mesh_rec.apply_transform(icp_mat_1)

    print(f"   ICP Pass 2 (Thresh: {icp_thresholds[1]:.4f})...")
    icp_mat_2 = o3d_icp_alignment(mesh_rec, mesh_gt, threshold=icp_thresholds[1])
    mesh_rec.apply_transform(icp_mat_2)
    
    # Combine all ICP transformations
    final_icp_transform = icp_mat_2 @ icp_mat_1
    align_params_dict['icp_transform'] = final_icp_transform
    
    # Save final results
    output_path = os.path.join(model.source_path, 'align_params.npz')
    np.savez(output_path, **align_params_dict)
    
    print(f"Preprocessing Done. Params saved to: {output_path}")

if __name__ == '__main__':
    parser = ArgumentParser(description="Pre-process alignment parameters (scale & transform) for mesh evaluation")
    
    model = ModelParams(parser, sentinel=True)
    prp = PriorParams(parser)
    
    parser.add_argument('--dataset_type', type=str)
    parser.add_argument('--gt_data_path', type=str)
    
    args = parser.parse_args()
    
    run_preprocessing_pipeline(
        args.dataset_type, 
        model.extract(args), 
        args.gt_data_path
    )
