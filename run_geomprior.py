import os
import json
from argparse import ArgumentParser
from arguments import ModelParams, PriorParams
from geomprior.dataloader import GroupAlign
from scene.dataset_readers import readColmapSceneInfo

def get_numeric_part(filename):
    return int(''.join(filter(str.isdigit, filename)))

def _list_images_sorted(image_dir):
    files = [
        f
        for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f))
        and os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    return sorted(files, key=get_numeric_part)


def _make_groups(files, group_size):
    total_files = len(files)
    num_groups = (total_files + group_size - 1) // group_size
    groups = [[] for _ in range(num_groups)]
    for i, file in enumerate(files):
        group_idx = i % num_groups
        groups[group_idx].append(file)
    return groups


def _stage_mapanything_groups(scene_dir, image_dir, output_dir, group_size):
    """
    Create DUSt3R-compatible group folders from MapAnything outputs.

    Expected MapAnything inputs in scene_dir:
      - ma_pts3d/<basename>.npy  (H, W, 3) or (3, H, W) float32
      - ma_conf/<basename>.npy   (H, W) float32

    Writes:
      - output_dir/_group*/depth/<basename>.npy  depth map (Z in camera coordinates)
      - output_dir/_group*/confs/<basename>.npy  confidence map (copied)
    """
    import numpy as np  # pyright: ignore[reportMissingImports]

    pts3d_dir = os.path.join(scene_dir, "ma_pts3d")
    conf_dir = os.path.join(scene_dir, "ma_conf")
    if not (os.path.isdir(pts3d_dir) and os.path.isdir(conf_dir)):
        raise FileNotFoundError(
            f"MapAnything priors not found. Expected dirs: {pts3d_dir} and {conf_dir}"
        )

    files = _list_images_sorted(image_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"No images found in {image_dir}")

    groups = _make_groups(files, group_size)
    manifest = {
        "prior_source": "mapanything",
        "image_dir": image_dir,
        "ma_pts3d_dir": pts3d_dir,
        "ma_conf_dir": conf_dir,
        "group_size": group_size,
        "num_images": len(files),
        "num_groups": len(groups),
    }
    with open(os.path.join(output_dir, "prior_source.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    for idx, group in enumerate(groups, 1):
        group_folder = os.path.join(output_dir, f"_group{idx}")
        depth_folder = os.path.join(group_folder, "depth")
        confs_folder = os.path.join(group_folder, "confs")
        os.makedirs(depth_folder, exist_ok=True)
        os.makedirs(confs_folder, exist_ok=True)

        for img_file in group:
            basename = os.path.splitext(os.path.basename(img_file))[0]
            pts3d_path = os.path.join(pts3d_dir, basename + ".npy")
            conf_path = os.path.join(conf_dir, basename + ".npy")
            out_depth_path = os.path.join(depth_folder, basename + ".npy")
            out_conf_path = os.path.join(confs_folder, basename + ".npy")

            if not os.path.exists(out_depth_path):
                pts = np.load(pts3d_path)
                if pts.ndim != 3:
                    raise ValueError(f"Unexpected ma_pts3d shape for {pts3d_path}: {pts.shape}")
                if pts.shape[-1] == 3:
                    depth = pts[..., 2]
                elif pts.shape[0] == 3:
                    depth = pts[2, ...]
                else:
                    raise ValueError(f"Unexpected ma_pts3d shape for {pts3d_path}: {pts.shape}")
                np.save(out_depth_path, depth.astype(np.float32))

            if not os.path.exists(out_conf_path):
                conf = np.load(conf_path).astype(np.float32)
                np.save(out_conf_path, conf)

    print("Finish depth predicting (MapAnything priors).")


# Sampling-based grouping is used to ensure that each image group covers the entire scene.
def GroupFiles(data_dir, output_dir, ckpt, group_size, vis, prior_source="auto"):
    """
    prior_source:
      - "auto": use MapAnything if ma_pts3d/ma_conf exist, else fall back to DUSt3R
      - "mapanything": use MapAnything outputs, no DUSt3R dependency
      - "dust3r": always run DUSt3R
    """
    scene_dir = os.path.dirname(data_dir.rstrip("/"))
    has_ma = os.path.isdir(os.path.join(scene_dir, "ma_pts3d")) and os.path.isdir(
        os.path.join(scene_dir, "ma_conf")
    )

    if prior_source in ("auto", "mapanything") and has_ma:
        _stage_mapanything_groups(scene_dir, data_dir, output_dir, group_size)
        return

    if prior_source == "mapanything" and not has_ma:
        raise FileNotFoundError(
            f"prior_source=mapanything requested, but ma_pts3d/ma_conf not found in {scene_dir}"
        )

    # Lazy import so users can run MapAnything-only flow without DUSt3R installed.
    from geomprior.run_dust3r import DUSt3R

    files = _list_images_sorted(data_dir)
    groups = _make_groups(files, group_size)
    for idx, group in enumerate(groups, 1):
        group_folder = os.path.join(output_dir, f"_group{idx}")
        os.makedirs(group_folder, exist_ok=True)
        DUSt3R(
            data_dir,
            group_folder,
            ckpt,
            group,
            vis,
        )  # Set vis=True to save the reconstructed dense point cloud from DUSt3R.
    print("Finish depth predicting (DUSt3R).")


def GeomPrior(model, prep, group_size, vis, skip_model, skip_align):
    datapath = model.source_path
    ckpt = prep.ckpt_mv
    gp_data_path = os.path.join(datapath, "geomprior")
    os.makedirs(gp_data_path, exist_ok=True)
    image_path = os.path.join(datapath, "rgb")
    # COLMAP reader expects <scene>/images; some pipelines store frames in <scene>/rgb.
    images_dir = os.path.join(datapath, "images")
    if not os.path.exists(images_dir) and os.path.isdir(image_path):
        try:
            os.symlink(image_path, images_dir)
        except FileExistsError:
            pass
    # depth generation
    if not skip_model:
        GroupFiles(image_path, gp_data_path, ckpt, group_size, vis, prior_source="auto")
    # depth align and resize
    if not skip_align:
        if os.path.exists(os.path.join(datapath, "sparse")):
            scene_info = readColmapSceneInfo(datapath, eval=False)
            cam_infos = scene_info.train_cameras
            GroupAlign(prep, cam_infos, scene_info.points3d, gp_data_path, vis)  # Set vis=True to visualize prior depth & normal
        else:
            assert False, "Could not recognize scene type!"


if __name__ == '__main__':
    parser = ArgumentParser(description="Generate geometric priors script parameters")
    model = ModelParams(parser, sentinel=True)
    prp = PriorParams(parser)
    parser.add_argument('--group_size', type=str,help='number of images in each group', default=40)
    parser.add_argument("--vis", action="store_true") 
    parser.add_argument("--skip_model", action="store_true")
    parser.add_argument("--skip_align", action="store_true")
    args = parser.parse_args()

    GeomPrior(model.extract(args), prp.extract(args), int(args.group_size), args.vis, args.skip_model, args.skip_align)