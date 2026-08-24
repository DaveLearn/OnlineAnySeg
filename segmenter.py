"""OnlineAnySeg DEG bridge module.

Adapts OnlineAnySeg's CropFormer + voxel-hash mask-merging pipeline to the DEG
external segmenter contract:

1. write the observations as a ScanNet-layout temp dataset (the upstream
   custom-dataset loader is broken; the ScanNet loader is the supported path);
2. run stage 1 (CropFormer masks + CLIP embeddings) in the `cropformer` pixi
   environment via ``pixi run -e cropformer mask_predict``;
3. run stage 2 (``main.py`` voxel-hash merging) as a subprocess of this
   environment with cwd at the repo root (upstream paths are cwd-relative);
4. composite per-frame instance label images directly from the method's own
   merged 2D masks using the exported ``ori_mask_list`` provenance (no
   raycasting), and transfer the refined 3D result (post boundary-processing
   and DBSCAN) onto the deg mesh vertices by nearest neighbour with the same
   0.15 m bound upstream's own evaluation uses.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

from initializerdefs import (
    InstanceMaskObjectsDef,
    ObjectSegmentations,
    ObservationFrame,
    Observations,
    SceneSetup,
    runtime_start,
    runtime_stop,
)
from psdframe import Frame

logger = logging.getLogger("onlineanyseg-segmenter")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_TEMPLATE = PROJECT_ROOT / "config" / "deg_cropformer.yaml"
DEFAULT_CROPFORMER_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "cropformer" / "Mask2Former_hornet_3x_576d0b.pth"
DEFAULT_CLIP_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "clip" / "open_clip_pytorch_model.bin"

# NN transfer bound from the reconstructed cloud to the deg mesh; matches the
# distance_upper_bound upstream's own evaluation uses (eval/evaluate_seqs.py).
DEFAULT_NN_DISTANCE_BOUND = 0.15


@dataclass
class TempScene:
    data_root: Path  # <work>/data
    scene_dir: Path  # <work>/data/<scene_id>
    frames_dir: Path  # <work>/data/<scene_id>/frames
    scene_id: str
    export_to_frame: Dict[int, ObservationFrame]
    canonical_k: np.ndarray  # shared intrinsic all written frames were resampled to


def get_dataset_frame_from_observation_frame(observation_frame: ObservationFrame) -> Frame:
    return Frame(
        id=observation_frame.id,
        name=observation_frame.name,
        color=torch.tensor(observation_frame.color).cuda(),
        X_WV=torch.tensor(observation_frame.X_WV),
        K=torch.tensor(observation_frame.K),
        depth=(torch.tensor(observation_frame.depth).cuda() if observation_frame.depth is not None else None),
    )


# ---------------------------------------------------------------------------
# Precommitted pose-only frame ordering (identical rule to SAM2Object)
# ---------------------------------------------------------------------------


def _order_frames_for_video(frames: List[Frame], scene: SceneSetup) -> List[Frame]:
    """Order unordered fixed cameras into the smoothest available ring "video".

    Precommitted rule (pose-only, no labels/results):
    1. up-axis n = normalized ground-plane normal from scene.ground_plane;
       if no plane is available, the camera-center PCA axis of smallest
       variance is used instead.
    2. c = mean of camera centers.
    3. zero direction x0 = world +X projected into the plane (world +Y if +X is
       within 1e-6 of parallel to n); y0 = n × x0.
    4. each camera's key is atan2((p - c)·y0, (p - c)·x0); sort ascending.
    5. ties (exact float equality) break by (p - c)·n, then frame id.
    """
    centers = np.stack([f.X_WV.cpu().numpy()[:3, 3] for f in frames], axis=0)

    if scene.ground_plane is not None:
        normal = np.array(scene.ground_plane[:3], dtype=np.float64)
    else:
        centered = centers - centers.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[-1]
    normal = normal / np.linalg.norm(normal)

    centroid = centers.mean(axis=0)

    x_axis = np.array([1.0, 0.0, 0.0])
    x_in_plane = x_axis - np.dot(x_axis, normal) * normal
    if np.linalg.norm(x_in_plane) < 1e-6:
        y_axis = np.array([0.0, 1.0, 0.0])
        x_in_plane = y_axis - np.dot(y_axis, normal) * normal
    x0 = x_in_plane / np.linalg.norm(x_in_plane)
    y0 = np.cross(normal, x0)

    def sort_key(idx_frame):
        idx, frame = idx_frame
        rel = centers[idx] - centroid
        azimuth = math.atan2(float(np.dot(rel, y0)), float(np.dot(rel, x0)))
        height = float(np.dot(rel, normal))
        return (azimuth, height, frame.id)

    ordered = [f for _, f in sorted(enumerate(frames), key=sort_key)]
    logger.info("Pose-only frame order: %s", [f.name for f in ordered])
    return ordered


# ---------------------------------------------------------------------------
# Workspace filtering + table identification (verbatim deg parity helpers)
# ---------------------------------------------------------------------------


def _erode_voxel_grid_xy(voxel_grid: o3d.geometry.VoxelGrid, layers: int) -> o3d.geometry.VoxelGrid:
    if layers <= 0 or not voxel_grid.has_voxels():
        return voxel_grid

    voxel_indices = [tuple(int(idx) for idx in voxel.grid_index) for voxel in voxel_grid.get_voxels()]
    xy_occupied = {(x, y) for x, y, _ in voxel_indices}

    for _ in range(layers):
        if not xy_occupied:
            break
        prev_xy = xy_occupied
        xy_occupied = {
            (x, y) for (x, y) in prev_xy if ((x - 1, y) in prev_xy and (x + 1, y) in prev_xy and (x, y - 1) in prev_xy and (x, y + 1) in prev_xy)
        }

    for voxel_index in voxel_indices:
        if (voxel_index[0], voxel_index[1]) not in xy_occupied:
            voxel_grid.remove_voxel(voxel_index)

    return voxel_grid


def get_workspace_voxels(scene: SceneSetup, shrink_xy_m: float = 0.04) -> o3d.geometry.VoxelGrid:
    table_xyz = scene.ground_gaussians.xyz
    table_plane = scene.ground_plane
    table_normal = np.array([table_plane[0], table_plane[1], table_plane[2]])
    table_pcd_extruded = np.array(table_xyz).copy()

    desired_height = 1.0
    below_table_height = 0.10
    voxel_size = 0.02
    iters = int(np.ceil(desired_height / voxel_size))
    for i in range(iters):
        new_points = table_xyz + table_normal * voxel_size * i
        table_pcd_extruded = np.append(table_pcd_extruded, new_points, axis=0)

    below_table_iters = int(np.ceil(below_table_height / voxel_size))
    for i in range(below_table_iters):
        table_pcd_extruded = np.append(table_pcd_extruded, table_xyz - table_normal * voxel_size * (i + 1), axis=0)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(table_pcd_extruded))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    layers = max(0, int(np.round(shrink_xy_m / voxel_grid.voxel_size)))
    return _erode_voxel_grid_xy(voxel_grid, layers)


def _compute_workspace_pixel_mask(frame: Frame, workspace_voxels: o3d.geometry.VoxelGrid) -> np.ndarray:
    assert frame.depth is not None
    depth = frame.depth
    h, w = depth.shape
    valid_mask = depth > 0
    if not bool(valid_mask.any()):
        return np.zeros((h, w), dtype=bool)

    y, x = torch.meshgrid(
        torch.arange(h, device=depth.device),
        torch.arange(w, device=depth.device),
        indexing="ij",
    )
    z = depth[valid_mask]
    x_world = (x[valid_mask] - frame.cx) * z / frame.fl_x
    y_world = (y[valid_mask] - frame.cy) * z / frame.fl_y
    points = torch.stack([x_world, y_world, z, torch.ones_like(z)], dim=0)
    world_points = (frame.X_WV_opencv.cuda() @ points).T[:, :3].detach().cpu().numpy()

    included = np.asarray(workspace_voxels.check_if_included(o3d.utility.Vector3dVector(world_points)), dtype=bool)
    workspace_mask = np.zeros((h, w), dtype=bool)
    workspace_mask[valid_mask.detach().cpu().numpy()] = included
    return workspace_mask


def _get_instance_id_mask_for_frame(instance_id: int, masks: Dict[str, np.ndarray], frame: Frame) -> torch.Tensor:
    frame_mask = masks[frame.name]
    instance_mask = frame_mask == instance_id
    return torch.tensor(instance_mask, device=frame.color.device, dtype=torch.bool)


def determine_table_instance_id(
    frames: List[Frame],
    masks: Dict[str, np.ndarray],
    table_plane: Tuple[float, float, float, float],
    object_ids: np.ndarray,
) -> int:
    if len(object_ids) == 0:
        return -1

    table_instance_candidates: List[int] = []
    table_instance_counts: List[int] = []
    for frame in frames:
        assert frame.depth is not None
        h, w = frame.depth.shape
        y, x = torch.meshgrid(
            torch.arange(h, device=frame.depth.device),
            torch.arange(w, device=frame.depth.device),
            indexing="ij",
        )
        valid_mask = frame.depth > 0
        z = frame.depth
        x_world = (x - frame.cx) * z / frame.fl_x
        y_world = (y - frame.cy) * z / frame.fl_y
        points = torch.stack([x_world, y_world, z, torch.ones_like(z)], dim=0)
        points = frame.X_WV_opencv.cuda() @ points.reshape(4, -1)
        points = points.reshape(4, h, w)

        a, b, c, d = table_plane
        plane_dist = (a * points[0] + b * points[1] + c * points[2] + d) / math.sqrt(a * a + b * b + c * c)
        table_mask = torch.abs(plane_dist) < 0.02
        table_mask = table_mask & valid_mask

        for instance_id in object_ids:
            inst_id_val = int(instance_id.item()) if hasattr(instance_id, "item") else int(instance_id)
            instance_mask = _get_instance_id_mask_for_frame(inst_id_val, masks, frame)
            instance_mask_valid = instance_mask & valid_mask
            instance_mask_near_table = instance_mask & table_mask
            valid_count = instance_mask_valid.sum()
            if valid_count > 0 and instance_mask_near_table.sum() / valid_count > 0.7:
                table_instance_candidates.append(inst_id_val)
                table_instance_counts.append(instance_mask_near_table.sum().item())

    if len(table_instance_candidates) == 0:
        logger.warning("No table candidates found")
        return -1

    best_idx = int(np.argmax(table_instance_counts))
    return table_instance_candidates[best_idx]


# ---------------------------------------------------------------------------
# Temp dataset in the ScanNet layout OnlineAnySeg's ScannetDataset consumes
# ---------------------------------------------------------------------------


def _remap_grid(k_src: np.ndarray, k_dst: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """cv2.remap grids that resample an image captured with k_src as if it had
    been captured with k_dst (same camera center/orientation, different focal
    length/principal point only -- exact for a pinhole model, since per-pixel
    depth-along-optical-axis is invariant to K).
    """
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x = (us - k_dst[0, 2]) / k_dst[0, 0]
    y = (vs - k_dst[1, 2]) / k_dst[1, 1]
    map_x = (x * k_src[0, 0] + k_src[0, 2]).astype(np.float32)
    map_y = (y * k_src[1, 1] + k_src[1, 2]).astype(np.float32)
    return map_x, map_y


def _resample_frame_to_intrinsic(
    color: np.ndarray, depth: np.ndarray, k_src: np.ndarray, k_dst: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample an RGB-D frame from its own intrinsic to a shared canonical
    intrinsic. Needed because OnlineAnySeg's core reconstruction (TSDF
    integration and 2D-mask backprojection alike) uses a single global
    intrinsic for every frame in the sequence -- it has no per-frame intrinsic
    support anywhere, unlike deg's heterogeneous fixed-camera work cells.
    Pixels outside the source camera's field of view after the remap are
    legitimately invalid (dst FOV may exceed src FOV) and come back as 0.
    """
    h, w = depth.shape
    map_x, map_y = _remap_grid(k_src, k_dst, h, w)
    color_out = cv2.remap(color, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # nearest-neighbour for depth: avoids blending near/far values across object boundaries
    depth_out = cv2.remap(depth, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return color_out, depth_out


def _resample_label_from_intrinsic(label: np.ndarray, k_src: np.ndarray, k_dst: np.ndarray) -> np.ndarray:
    """Inverse of _resample_frame_to_intrinsic for label maps: given a label
    image in k_src's (canonical) pixel grid, resample it into k_dst's (a
    frame's own native) pixel grid, so the exported InstanceMaskObjectsDef
    aligns with each frame's actual native imagery. Same (input_K, output_K)
    argument convention as _remap_grid/_resample_frame_to_intrinsic.
    """
    h, w = label.shape
    map_x, map_y = _remap_grid(k_src, k_dst, h, w)
    return cv2.remap(label, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _sanitize_depth(depth: np.ndarray) -> np.ndarray:
    """Zero out non-finite depth (sensor no-return regions). Left unsanitized,
    inf survives a *1000 + clip(0, 65535) round trip as a finite-but-wrong
    65.535 m value (usually caught by depth_far filtering downstream, but not
    guaranteed), and NaN hits undefined behaviour under astype(uint16) --
    exactly the class of bug fixed in dependencies/MaskClustering's
    utils/mask_backprojection.py::prepare_depth_for_backprojection.
    """
    return np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)


def _sanitize_scene_id(raw_id: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    cleaned = "".join(ch if ch in allowed else "_" for ch in raw_id)
    cleaned = cleaned.strip("_")
    return cleaned or "scene"


def _write_scannet_temp_dataset(frames: List[Frame], scene_id: str, work_root: Path) -> TempScene:
    """ScanNet frame layout with an extra frames/ level:
    <work>/data/<scene_id>/frames/{color,depth,pose,intrinsic}. ScannetDataset
    derives seq_name from the path's second-to-last component, which then
    equals scene_id. No mesh ply is written: the loader would pick up any
    *_vh_clean_2.ply next to frames/, and the deg mesh is only needed
    in-process for the vertex-label transfer.

    OnlineAnySeg's core reconstruction (TSDF integration in Scene_rep AND 2D-mask
    backprojection in voxelized_points.turn_mask_to_voxel) uses a single global
    intrinsic for every frame -- it has no per-frame intrinsic support anywhere,
    unlike deg's heterogeneous fixed-camera work cells. Every frame is resampled
    to a shared canonical intrinsic (frame 0's, after the precommitted pose-only
    ordering) before being written, so the geometry OnlineAnySeg reconstructs is
    self-consistent across views. Depth is also sanitized for non-finite values
    (sensor no-return regions) before the mm/uint16 conversion -- otherwise inf
    survives as a finite-but-wrong 65.535 m value and NaN hits undefined
    behaviour under astype(uint16); same class of bug fixed in
    dependencies/MaskClustering's utils/mask_backprojection.py.
    """
    data_root = work_root / "data"
    scene_dir = data_root / scene_id
    frames_dir = scene_dir / "frames"
    color_dir = frames_dir / "color"
    depth_dir = frames_dir / "depth"
    pose_dir = frames_dir / "pose"
    intrinsic_dir = frames_dir / "intrinsic"

    for path in (color_dir, depth_dir, pose_dir, intrinsic_dir):
        path.mkdir(parents=True, exist_ok=True)

    if len(frames) == 0:
        raise RuntimeError("No frames to export")

    canonical_k = frames[0].K.cpu().numpy()
    reference_shape = frames[0].depth.shape if frames[0].depth is not None else None
    for frame in frames[1:]:
        if not np.allclose(frame.K.cpu().numpy(), canonical_k, atol=1e-3):
            logger.info(
                "Frame %s intrinsics differ from frame %s; resampling to the shared canonical intrinsic before writing.",
                frame.name,
                frames[0].name,
            )
        if frame.depth is not None and reference_shape is not None and frame.depth.shape != reference_shape:
            raise RuntimeError("OnlineAnySeg requires all frames to share one resolution")

    k4 = np.eye(4, dtype=np.float64)
    k4[:3, :3] = canonical_k
    np.savetxt(intrinsic_dir / "intrinsic_depth.txt", k4, fmt="%.8f")

    export_to_frame: Dict[int, ObservationFrame] = {}
    for idx, frame in enumerate(frames):
        color = (frame.color.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        depth = frame.depth.cpu().numpy() if frame.depth is not None else None
        if depth is None:
            raise RuntimeError("Depth is required for the OnlineAnySeg pipeline")
        depth = _sanitize_depth(depth)

        frame_k = frame.K.cpu().numpy()
        if not np.allclose(frame_k, canonical_k, atol=1e-3):
            color, depth = _resample_frame_to_intrinsic(color, depth, frame_k, canonical_k)

        cv2.imwrite(str(color_dir / f"{idx}.jpg"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        depth_mm = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(str(depth_dir / f"{idx}.png"), depth_mm)

        pose = frame.X_WV_opencv.cpu().numpy()
        np.savetxt(pose_dir / f"{idx}.txt", pose, fmt="%.8f")

        export_to_frame[idx] = ObservationFrame(
            id=frame.id,
            name=frame.name,
            color=frame.color.cpu().numpy(),
            X_WV=frame.X_WV.cpu().numpy(),
            K=frame.K.cpu().numpy(),  # native K -- used to inverse-warp exported labels back
            depth=frame.depth.cpu().numpy() if frame.depth is not None else None,
        )

    logger.info("Wrote temp ScanNet dataset to %s (%d frames)", scene_dir, len(frames))
    return TempScene(
        data_root=data_root,
        scene_dir=scene_dir,
        frames_dir=frames_dir,
        scene_id=scene_id,
        export_to_frame=export_to_frame,
        canonical_k=canonical_k,
    )


def _instantiate_config(template_path: Path, output_path: Path, img_h: int, img_w: int, mask_weight_threshold: int, min_instance_points: int) -> None:
    import yaml

    with open(template_path) as f:
        cfg = yaml.safe_load(f)
    cfg["cam"]["img_h"] = int(img_h)
    cfg["cam"]["img_w"] = int(img_w)
    cfg["seg"]["mask_weight_threshold"] = int(mask_weight_threshold)
    cfg["seg"]["min_instance_points"] = int(min_instance_points)
    with open(output_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Subprocess stages
# ---------------------------------------------------------------------------


def _scrubbed_child_env() -> Dict[str, str]:
    """Environment for child processes that re-activate their own pixi env.

    This process runs inside the default pixi environment, whose activation
    exports LD_LIBRARY_PATH / compiler flags pointing at itself. Handing those
    to `pixi run -e cropformer` mixes two environments' libraries into one
    process (the failure mode documented in scripts/build_cropformer_ops.sh),
    so they are cleared and the child's own activation supplies its own.
    """
    env = os.environ.copy()
    for var in ("CFLAGS", "CXXFLAGS", "LDFLAGS", "TORCH_CUDA_ARCH_LIST", "LD_LIBRARY_PATH", "PYTHONPATH"):
        env.pop(var, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _run_stage1_mask_predict(
    temp_scene: TempScene,
    instance_root: Path,
    img_h: int,
    img_w: int,
    clip_checkpoint: Path,
    cropformer_checkpoint: Path,
    confidence_threshold: float,
    min_mask_pixel_size: int,
) -> Path:
    if not cropformer_checkpoint.exists():
        raise FileNotFoundError(
            f"CropFormer checkpoint not found: {cropformer_checkpoint}. "
            "Run `pixi run -e cropformer download_cropformer_checkpoint` once (needs HF_TOKEN; the checkpoint is gated) "
            "or place the checkpoint manually."
        )

    pixi = shutil.which("pixi") or "pixi"
    cmd = [
        pixi,
        "run",
        "--frozen",
        "-e",
        "cropformer",
        "mask_predict",
        "--",
        "--root",
        str(temp_scene.data_root),
        "--seq_name",
        temp_scene.scene_id,
        "--image_path_pattern",
        "frames/color/*.jpg",
        "--seg_interval",
        "1",
        "--output_root",
        str(instance_root),
        "--dst_h",
        str(img_h),
        "--dst_w",
        str(img_w),
        "--pretrained_path",
        str(clip_checkpoint),
        "--confidence_threshold",
        str(confidence_threshold),
        "--min_mask_pixel_size",
        str(min_mask_pixel_size),
        "--opts",
        "MODEL.WEIGHTS",
        str(cropformer_checkpoint),
    ]
    logger.info("Running stage 1 (CropFormer + CLIP): %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=_scrubbed_child_env(),
        stdout=sys.stderr,
        stderr=sys.stderr,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Stage 1 (mask_predict) failed with exit code {result.returncode}")

    instance_dir = instance_root / temp_scene.scene_id
    if not (instance_dir / "mask").is_dir():
        raise RuntimeError(f"Stage 1 produced no mask directory under {instance_dir}")
    return instance_dir


def _run_stage2_main(temp_scene: TempScene, instance_dir: Path, config_path: Path, output_root: Path) -> Path:
    cmd = [
        sys.executable,
        "main.py",
        "-c",
        str(config_path),
        "-d",
        str(temp_scene.frames_dir),
        "-i",
        str(instance_dir),
        "--seq_name",
        temp_scene.scene_id,
        "-o",
        str(output_root),
    ]
    logger.info("Running stage 2 (voxel-hash merging): %s", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),  # upstream paths (third_party/FCGF, geo_extractor_path) are cwd-relative
        env=env,
        stdout=sys.stderr,
        stderr=sys.stderr,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Stage 2 (main.py) failed with exit code {result.returncode}")
    return output_root / temp_scene.scene_id


# ---------------------------------------------------------------------------
# Output conversion
# ---------------------------------------------------------------------------


def _load_stage2_outputs(result_dir: Path) -> Tuple[Optional[np.ndarray], List[List[Tuple[int, int]]], Optional[np.ndarray]]:
    """Returns (pred_masks (n_points, n_inst) bool | None, per-instance ori mask lists, recon points | None)."""
    npz_path = result_dir / "ckpt_final.npz"
    ori_path = result_dir / "ckpt_final_ori_masks.json"
    ply_path = result_dir / "final.ply"

    ori_mask_lists: List[List[Tuple[int, int]]] = []
    if ori_path.exists():
        with open(ori_path) as f:
            ori_mask_lists = [[(int(fid), int(mid)) for fid, mid in inst] for inst in json.load(f)]

    if not npz_path.exists():
        # upstream writes no npz when zero instances survive; the provenance json
        # distinguishes "no instances" (present, empty) from a failed run
        if not ori_path.exists():
            raise RuntimeError(f"Stage 2 outputs missing under {result_dir} (neither {npz_path.name} nor {ori_path.name})")
        logger.warning("OnlineAnySeg exported zero instances (no %s)", npz_path.name)
        return None, ori_mask_lists, None

    data = np.load(npz_path)
    pred_masks = data["pred_masks"].astype(bool)
    if pred_masks.shape[1] != len(ori_mask_lists):
        raise RuntimeError(
            f"Provenance/instance count mismatch: pred_masks has {pred_masks.shape[1]} instances, "
            f"{ori_path.name} has {len(ori_mask_lists)}"
        )

    recon_points = None
    if ply_path.exists():
        recon_points = np.asarray(o3d.io.read_point_cloud(str(ply_path)).points, dtype=np.float64)
        if recon_points.shape[0] != pred_masks.shape[0]:
            raise RuntimeError(f"final.ply has {recon_points.shape[0]} points but pred_masks has {pred_masks.shape[0]} rows")
    return pred_masks, ori_mask_lists, recon_points


def _composite_instance_groups(
    ori_mask_lists: List[List[Tuple[int, int]]],
    instance_dir: Path,
    export_to_frame: Dict[int, ObservationFrame],
    canonical_k: np.ndarray,
    frames: List[Frame],
    workspace_masks: Dict[int, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Paint per-frame label images from the method's own merged 2D masks.

    Collision-free by construction: each raw (frame, mask_id) belongs to
    exactly one final instance. Label i+1 corresponds to pred_masks column i.
    Stage 1 (CropFormer) ran on frames resampled to the shared canonical
    intrinsic (see _write_scannet_temp_dataset), so masks are composited in
    that space and then resampled back to each frame's own native intrinsic --
    matching the pixel grid workspace_masks and the rest of the deg pipeline
    expect -- before the workspace filter is applied.
    """
    export_frame_ids = sorted(export_to_frame)
    mask_dir = instance_dir / "mask"
    seg_images: Dict[int, Optional[np.ndarray]] = {}
    for frame_id in export_frame_ids:
        seg_path = mask_dir / f"{frame_id}.png"
        seg_images[frame_id] = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED) if seg_path.exists() else None

    h, w = frames[0].depth.shape
    canonical_labels: Dict[int, np.ndarray] = {frame_id: np.zeros((h, w), dtype=np.int32) for frame_id in export_frame_ids}

    for instance_idx, ori_list in enumerate(ori_mask_lists):
        label = instance_idx + 1
        for frame_id, mask_id in ori_list:
            seg = seg_images.get(frame_id)
            if seg is None:
                continue
            canonical_labels[frame_id][seg == mask_id] = label

    result_by_export: Dict[int, np.ndarray] = {}
    for frame_id in export_frame_ids:
        native_k = export_to_frame[frame_id].K
        label = canonical_labels[frame_id]
        if not np.allclose(native_k, canonical_k, atol=1e-3):
            label = _resample_label_from_intrinsic(label, canonical_k, native_k)
        label = np.where(workspace_masks[frame_id], label, 0)
        result_by_export[frame_id] = label

    return {export_to_frame[frame_id].name: result_by_export[frame_id] for frame_id in export_frame_ids}


def _transfer_labels_to_mesh_vertices(
    pred_masks: Optional[np.ndarray],
    recon_points: Optional[np.ndarray],
    mesh_vertices: np.ndarray,
    nn_distance_bound: float,
) -> np.ndarray:
    """Refined 3D result (post boundary-processing + DBSCAN) onto the deg mesh.

    Per-point labels resolve overlaps smallest-instance-wins (paint larger
    instances first so smaller ones overwrite), then transfer by nearest
    neighbour with upstream's own eval bound. Unmatched vertices stay 0.
    """
    vertex_labels = np.zeros(len(mesh_vertices), dtype=np.int32)
    if pred_masks is None or recon_points is None or pred_masks.shape[1] == 0:
        return vertex_labels

    point_labels = np.zeros(pred_masks.shape[0], dtype=np.int32)
    instance_sizes = pred_masks.sum(axis=0)
    for instance_idx in np.argsort(-instance_sizes):
        point_labels[pred_masks[:, instance_idx]] = instance_idx + 1

    labeled = point_labels > 0
    if not labeled.any():
        return vertex_labels

    tree = cKDTree(recon_points[labeled])
    distances, indices = tree.query(mesh_vertices, distance_upper_bound=nn_distance_bound)
    matched = np.isfinite(distances)
    vertex_labels[matched] = point_labels[labeled][indices[matched]]
    return vertex_labels


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def initialize_scene(
    observations: Observations,
    scene: SceneSetup,
    intermediate_outputs_path: Optional[Path] = None,
    mesh_path: Optional[Path] = None,
    config_template: Path = DEFAULT_CONFIG_TEMPLATE,
    cropformer_checkpoint: Path = DEFAULT_CROPFORMER_CHECKPOINT,
    clip_checkpoint: Path = DEFAULT_CLIP_CHECKPOINT,
    confidence_threshold: float = 0.6,
    min_mask_pixel_size: int = 500,
    mask_weight_threshold: int = 1,
    min_instance_points: int = 200,
    nn_distance_bound: float = DEFAULT_NN_DISTANCE_BOUND,
) -> ObjectSegmentations:
    # NB: both stages run as subprocesses, so their in-subprocess model loads
    # (CropFormer, CLIP, FCGF) cannot be excluded from this level and are
    # counted as compute (same situation as the MaskClustering adapter).
    _rt = runtime_start("onlineanyseg", scene=observations.id, n_frames=len(observations.frames))
    raw_frames = [get_dataset_frame_from_observation_frame(frame) for frame in observations.frames]
    if not raw_frames:
        raise ValueError("No frames in observations")

    if mesh_path is None:
        raise ValueError("mesh_path is required")
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found at {mesh_path}")
    logger.info("Loading mesh from %s", mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(mesh_vertices) == 0:
        raise RuntimeError("Mesh is empty")

    logger.info("Building workspace voxels ...")
    workspace_voxels = get_workspace_voxels(scene)

    # precommitted pose-only ordering; index i in the temp dataset (and in the
    # ori_mask_list frame ids) refers to frames[i] in this order
    frames = _order_frames_for_video(raw_frames, scene)

    if intermediate_outputs_path is not None:
        work_root = Path(intermediate_outputs_path) / "onlineanyseg_work"
    else:
        work_root = Path(tempfile.mkdtemp(prefix="onlineanyseg_"))
    work_root.mkdir(parents=True, exist_ok=True)

    scene_id = _sanitize_scene_id(observations.id or "scene")
    temp_scene = _write_scannet_temp_dataset(frames, scene_id, work_root)

    h, w = frames[0].depth.shape
    config_path = work_root / "deg_cropformer.yaml"
    _instantiate_config(config_template, config_path, img_h=h, img_w=w, mask_weight_threshold=mask_weight_threshold, min_instance_points=min_instance_points)

    instance_dir = _run_stage1_mask_predict(
        temp_scene,
        instance_root=work_root / "instance",
        img_h=h,
        img_w=w,
        clip_checkpoint=Path(clip_checkpoint),
        cropformer_checkpoint=Path(cropformer_checkpoint),
        confidence_threshold=confidence_threshold,
        min_mask_pixel_size=min_mask_pixel_size,
    )

    result_dir = _run_stage2_main(temp_scene, instance_dir, config_path, output_root=work_root / "output")

    pred_masks, ori_mask_lists, recon_points = _load_stage2_outputs(result_dir)
    n_instances = len(ori_mask_lists)
    logger.info("OnlineAnySeg exported %d instances", n_instances)

    workspace_masks_by_frame = {frame_id: _compute_workspace_pixel_mask(frames[frame_id], workspace_voxels) for frame_id in sorted(temp_scene.export_to_frame)}

    instance_groups = _composite_instance_groups(
        ori_mask_lists,
        instance_dir,
        temp_scene.export_to_frame,
        temp_scene.canonical_k,
        frames,
        workspace_masks_by_frame,
    )

    vertex_labels = _transfer_labels_to_mesh_vertices(pred_masks, recon_points, mesh_vertices, nn_distance_bound)

    # -- shared parity filtering (identical to the other baselines) --
    all_label_ids = np.array(
        sorted({int(label_id) for mask in instance_groups.values() for label_id in np.unique(mask) if label_id > 0}),
        dtype=np.int32,
    )

    frame_counts = {label_id: 0 for label_id in all_label_ids}
    for mask in instance_groups.values():
        for label_id in all_label_ids:
            if np.any(mask == label_id):
                frame_counts[label_id] += 1

    # Require instances to be seen in multiple (3) views.
    min_frame_count = 3
    valid_ids = np.array([label_id for label_id, count in frame_counts.items() if count >= min_frame_count], dtype=np.int32)
    logger.info("Labels in >= %d frames: %d / %d", min_frame_count, len(valid_ids), len(all_label_ids))
    for frame_name in instance_groups:
        instance_groups[frame_name][~np.isin(instance_groups[frame_name], valid_ids)] = 0

    table_id = determine_table_instance_id(frames, instance_groups, scene.ground_plane, valid_ids)
    logger.info("Table instance id: %d", table_id)
    if table_id > 0:
        valid_ids = valid_ids[valid_ids != table_id]
        for frame_name in instance_groups:
            instance_groups[frame_name][instance_groups[frame_name] == table_id] = 0

    for frame_name in instance_groups:
        instance_groups[frame_name][~np.isin(instance_groups[frame_name], valid_ids)] = 0

    # keep the 3D channel's instance set identical to the 2D channel's
    vertex_labels[~np.isin(vertex_labels, valid_ids)] = 0

    frame_ids: List[int] = []
    pixel_masks: List[np.ndarray] = []
    results_path = None
    if intermediate_outputs_path is not None:
        results_path = Path(intermediate_outputs_path) / "instances"
        results_path.mkdir(parents=True, exist_ok=True)

    for obs_frame in observations.frames:
        frame_ids.append(obs_frame.id)
        mask = instance_groups.get(obs_frame.name, np.zeros((h, w), dtype=np.int32))
        pixel_masks.append(mask)
        if results_path is not None:
            cv2.imwrite(str(results_path / f"{obs_frame.name}.png"), mask.astype(np.uint16))

    instance_mask_objects = InstanceMaskObjectsDef(frame_ids=frame_ids, pixel_object_ids=pixel_masks)
    logger.info("Initialized %d objects (after table removal)", len(valid_ids))
    runtime_stop(_rt)

    return ObjectSegmentations(object_segmentations=instance_mask_objects, mesh_vertex_instance_ids=vertex_labels)
