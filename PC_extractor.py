import numpy as np
import torch

from third_party.FCGF.model.resunet import ResUNetBN2C
from third_party.FCGF.util.misc import extract_features, extract_features_mdfy
from third_party.FCGF.util.visualization import get_colored_point_cloud_feature

from tool.geometric_helpers import aggregate_pts_feature


class PointFeatureExtractor:
    def __init__(self, cfg, voxel_grids, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        self.voxel_grids = voxel_grids

        self.voxel_size = self.cfg["pc_extractor"]["voxel_size"]  # default: 0.025
        self.feature_dim = self.cfg["pc_extractor"]["feature_dim"]  # default: 16
        self.ckpt_path = self.cfg["pc_extractor"]["geo_extractor_path"]
        self.min_pts_num = self.cfg["pc_extractor"].get("min_pts_num", 50)  # min scene points inside a mask's voxels for a geometric feature; default: 50
        self.model = self.create_load_model()


    def create_load_model(self):
        checkpoint = torch.load(self.ckpt_path)
        model = ResUNetBN2C(1, self.feature_dim, normalize_feature=True, conv1_kernel_size=3, D=3)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model = model.to(self.device)
        print("[INFO] Pointcloud geometric feature extractor model has been loaded from: %s" % self.ckpt_path)
        return model


    # @brief: extract per-point geometric feature for input pointcloud;
    # @param pts_xyz: Tensor(n, 3)/ndarray(n, 3), dtype=float64;
    #-@return: extracted per-point feature, Tensor(n, feature_dim), dtype=float32.
    def infer_pts_feature(self, pts_xyz):
        if isinstance(pts_xyz, torch.Tensor):
            pts_xyz = pts_xyz.cpu().numpy().astype("float64")
        else:
            pts_xyz = pts_xyz.astype("float64")

        # *** firstly input points will be downsampled(voxelized), then voxelized points will be sent to extract per-point feature
        # @return xyz_down: voxelized point coordinates, ndarray(ds_pts_num, 3);
        # @return feature: per-point feature of each voxelized point, Tensor(ds_pts_num, feature_dim);
        # @return indices_inverse: mapping table. For each original point, its corresponding voxelized point, Tensor(pts_num, );
        xyz_down, feature, indices_inverse = extract_features_mdfy(self.model, xyz=pts_xyz, voxel_size=self.voxel_size, device=self.device, skip_check=True)

        raw_pts_feature = feature[indices_inverse]
        return raw_pts_feature


    # @brief: extract currently reconstructed pointcloud, and extract per-point feature;
    # @param pts_xyz: input pointcloud, ndarray(n, 3);
    # @param return_pc:
    #-@return success_flag: whether current VoxelBlockGrid can extract pointcloud by Marching Cubes, bool;
    #-@return pts_feature: extracted per-point feature, Tensor(n, feature_dim), dtype=float32;
    #-@return pts_xyz: extracted pointcloud, Tensor(n, 3), dtype=float32.
    def get_feature_pc(self, pts_xyz, return_pc=True):
        if pts_xyz.shape[0] > 0:
            success_flag = True
        else:
            return False, None, None

        pts_feature = self.infer_pts_feature(pts_xyz)
        if return_pc:
            if isinstance(pts_xyz, np.ndarray):
                pts_xyz = torch.from_numpy(pts_xyz).to(self.device)
            return success_flag, pts_feature, pts_xyz
        else:
            return success_flag, pts_feature, None


    # @brief: extract geometric feature for each given mask;
    # @param mask_voxel_coord_list: voxel coordinates of each given mask, list of Tensor(n_i, 3), dtype=float32;
    # @param scene_points: extracted reconstructed pointcloud(if provided), Tensor(N, 3)/None;
    # @param pts_feature: extracted feature pointcloud(if provided), Tensor(N, feature_dim)/None;
    #-@return extract_flag: whether at least 1 mask's geometric feature is extracted successfully, bool;
    #-@return mask_geo_feature_list: each mask's geometric feature, list of Tensor(feature_dim, ), dtype=float32;
    #-@return pts_feature: extracted per-point feature, Tensor(n, feature_dim), dtype=float32;
    #-@return scene_points: input pointcloud, Tensor(n, 3), dtype=float32.
    def get_masks_geometric_features(self, mask_voxel_coord_list, scene_points, pts_feature=None, dist_aggr=False, min_pts_num=None):
        if min_pts_num is None:
            min_pts_num = self.min_pts_num
        # Step 1: extract feature pointcloud (if needed)
        if pts_feature is not None and scene_points.shape[0] == pts_feature.shape[0]:
            # if feature pointcloud is provided, skip extracting feature pc
            success_flag = True
        else:
            try:
                success_flag, pts_feature, scene_points = self.get_feature_pc(scene_points)
            except MemoryError as me:
                print(3)

        if not success_flag:
            return False, None, None, None

        # Step 2: for each extracted scene point, compute its corresponding voxel coordinates and voxel index
        voxelized_scene_points = self.voxel_grids.world_coords2voxel_coords(scene_points).to(self.device)
        point_voxel_indices = self.voxel_grids.voxel_coords2voxel_indices(voxelized_scene_points)  # Tensor(scene_pts_num, )

        # Step 3: for each mask, compute scene points contained by it, and extract their per-point feature
        mask_num = len(mask_voxel_coord_list)
        masks_voxel_indices_list = [self.voxel_grids.voxel_coords2voxel_indices(mask_voxel_coords) for mask_voxel_coords in mask_voxel_coord_list]

        mask_geo_feature_list = []
        extract_flag = False
        for i in range(mask_num):
            # 2.1: compute corresponding points of this mask in scene points
            scene_pts_mask = torch.isin(point_voxel_indices, masks_voxel_indices_list[i])  # Tensor(scene_pts_num, ), dtype=bool
            if torch.count_nonzero(scene_pts_mask) < min_pts_num:
                mask_geo_feature_list.append(None)
                continue

            # 2.2: get mask geometric feature from per-point geometric features (***probably need to modify)
            mask_pts_xyz = scene_points[scene_pts_mask]
            mask_pts_feature = pts_feature[scene_pts_mask]  # per-point feature of this mask, Tensor(corr_pts_num, feature_dim)

            if dist_aggr:
                mask_feature = aggregate_pts_feature(mask_pts_xyz, mask_pts_feature)  # weighted sum (weight inversely proportional to distance to centroid)
            else:
                mask_feature = torch.mean(mask_pts_feature, dim=0)  # Average pooling

            mask_geo_feature_list.append(mask_feature)
            extract_flag = True

        return extract_flag, mask_geo_feature_list, pts_feature, scene_points
