import glob
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d
from natsort import natsorted
import json

from tool.geometric_helpers import compose_transformations


def get_dataset(data_location, instance_dir, cfg, device="cuda:0"):
    if cfg["dataset"] == "scannet":
        dataset = ScannetDataset
    if cfg["dataset"] == "scenenn":
        dataset = SceneNNDataset
    if cfg["dataset"] == "my_dataset":
        dataset = MyDataset

    return dataset(data_location, instance_dir, cfg, device)


class SceneNNDataset(Dataset):
    def __init__(self, data_location, instance_dir, cfg, device="cuda:0"):
        self.cfg = cfg
        self.data_location = data_location
        self.seq_name = self.data_location.split("/")[-2]
        self.instance_dir = instance_dir
        self.mask_image_dir = os.path.join(self.instance_dir, "mask")
        self.mask_embed_dir = os.path.join(self.instance_dir, "mask_embeddings")
        self.mask_basename_list = natsorted( os.listdir(self.mask_image_dir) )
        self.mask_embed_basename_list = natsorted( os.listdir(self.mask_embed_dir) )
        self.device = device
        self.target_h = cfg["cam"]["img_h"]
        self.target_w = cfg["cam"]["img_w"]
        self.depth_scale = cfg["cam"]["depth_scale"]
        self.depth_near = cfg["cam"]["depth_near"] if cfg["cam"]["depth_near"] > 0 else -1
        self.depth_far = cfg["cam"]["depth_far"] if cfg["cam"]["depth_far"] > 0 else -1
        intrinsic_file = os.path.join(self.data_location, "intrinsic", "intrinsic_depth.txt")
        cam_intrinsic = np.loadtxt(intrinsic_file)[:3, :3].astype("float32")  # ndarray(3, 3), dtype=float32
        self.cam_intrinsic = torch.from_numpy(cam_intrinsic).to(self.device)
        self.frame_num = self.get_scene_pose_num()
        self.last_seg_frame_id = self.get_last_seg_frame_id(self.cfg["mapping"]["keyframe_freq"])
        self.poses = self.get_poses()  # default: relative poses to first pose, Tensor(n, 4, 4)
        self.bbox = None
        self.gt_ply_file = self.find_gt_ply(self.data_location)
        if self.cfg["cam"]["bound"]:
            self.load_bound()
        self.min_max_xyz = self.get_bbox(self.gt_ply_file)
        self.pinhole_cam_intrinsic = self.get_intrinsics(self.cam_intrinsic.cpu().numpy())  # o3d.camera.PinholeCameraIntrinsic obj


    def get_scene_pose_num(self):
        pose_list = os.listdir(os.path.join(self.data_location, "pose"))
        pose_list = natsorted(pose_list)
        return len(pose_list)

    def __len__(self):
        return self.frame_num

    def find_gt_ply(self, dir):
        gt_ply_files = glob.glob( os.path.join(dir, "*.ply") )
        if len(gt_ply_files) == 1:
            return gt_ply_files[0]
        else:
            return None

    def get_bbox(self, ply_file):
        if ply_file is None or not os.path.exists(ply_file):
            min_max_xyz = [[0., 10.], [0., 10.], [0., 5.]]
        else:
            pc = o3d.io.read_point_cloud(ply_file)
            min_xyz = pc.get_min_bound().astype("float32")
            max_xyz = pc.get_max_bound().astype("float32")
            min_max_xyz = np.stack([min_xyz, max_xyz], axis=-1).tolist()
        return min_max_xyz

    def get_last_seg_frame_id(self, seg_interval=0):
        if seg_interval <= 0:
            seg_interval = self.cfg["seg"]["seg_add_interval"]

        seg_frame_ids = [ int(mask_base_name[5:-4]) for mask_base_name in self.mask_basename_list ]
        last_seg_frame_id = seg_frame_ids[0]
        for seg_frame_id in seg_frame_ids[::-1]:
            if (seg_frame_id - 1) % seg_interval == 0:
                last_seg_frame_id = seg_frame_id
                break
        return last_seg_frame_id

        # @brief: load bounding box of GT mesh
    def load_bound(self):
        gt_mesh_path = self.gt_ply_file
        if gt_mesh_path is not None and os.path.exists(gt_mesh_path):
            gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
            self.bbox = gt_mesh.get_oriented_bounding_box()  # open3d.geometry.OrientedBoundingBox obj

    def load_poses(self):
        poses = []
        posefiles = natsorted( glob.glob( os.path.join(self.data_location, "pose/*.txt") ) )
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile).astype("float32"))
            poses.append(_pose)
        poses = torch.stack(poses, dim=0).to(self.device)
        return poses

    def get_poses(self, relative=False):
        poses = self.load_poses()  # Tensor(n, 4, 4)
        self.first_pose_c2w = poses[0]
        if relative:  # default
            pose_first = poses[0]  # Tensor(4, 4)
            pose_first_inv = torch.inverse(pose_first).unsqueeze(0).repeat(poses.shape[0], 1, 1)  # Tensor(n, 4, 4)
            final_poses = compose_transformations(pose_first_inv, poses)
        else:
            final_poses = poses
        return final_poses

    def get_intrinsics(self, intrinsic_mat):
        intrinsic_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_cam_parameters.set_intrinsics(self.target_w, self.target_h, intrinsic_mat[0, 0], intrinsic_mat[1, 1], intrinsic_mat[0, 2], intrinsic_mat[1, 2])
        return intrinsic_cam_parameters


    def __getitem__(self, index):
        fixed_digit_index = str(index + 1).zfill(5)  # int --> str (fill 0 until 5 digits), starting from 00001
        color_img_path = os.path.join(self.data_location, "image", "image%s.png" % fixed_digit_index)
        depth_img_path = os.path.join(self.data_location, "depth", "depth%s.png" % fixed_digit_index)

        # Step 1: load segmentation image (and semantic feature of each detected mask)
        mask_image_basename = "image%s.png" % fixed_digit_index
        mask_embed_basename = "image%s.pt" % fixed_digit_index
        if mask_image_basename in self.mask_basename_list and mask_embed_basename in self.mask_embed_basename_list:
            instance_path = os.path.join(self.mask_image_dir, mask_image_basename)
            segmentation = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)  # ndarray(H, W), dtype=uint8
            segmentation = torch.from_numpy(segmentation).to(self.device)
            self.latest_seg_img = segmentation

            instance_embed_path = os.path.join(self.mask_embed_dir, mask_embed_basename)
            mask_embeddings = torch.load(instance_embed_path).to(self.device)
            seg_flag = True
        else:
            segmentation = torch.zeros_like(self.latest_seg_img).to(self.device)
            mask_embeddings = torch.zeros((1, self.cfg["mask"]["feature_dim"])).to(self.device)
            seg_flag = False

        # Step 2: load color/depth image
        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
        depth_img = depth_img.astype("float32") / self.depth_scale

        color_img = cv2.imread(color_img_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # BGR --> RGB

        color_img = torch.from_numpy(color_img).to(self.device) / 255  # Tensor(j, w, 3), [0, 1], dtype=float32
        depth_img = torch.from_numpy(depth_img).to(self.device)  # Tensor(h, w)

        # depth clipping
        if self.depth_near > 0 and self.depth_far > 0:
            depth_mask = ((depth_img > self.depth_near) & (depth_img < self.depth_far))
            depth_img = torch.where(depth_mask, depth_img, torch.zeros_like(depth_img))

        pose_matrix = self.poses[index]  # pose_c2w, Tensor(4, 4)
        return color_img, depth_img, pose_matrix, segmentation, mask_embeddings, seg_flag


class ScannetDataset(Dataset):
    def __init__(self, data_location, instance_dir, cfg, device="cuda:0"):
        self.cfg = cfg
        self.data_location = data_location
        self.seq_name = self.data_location.split("/")[-2]
        self.instance_dir = instance_dir
        self.mask_image_dir = os.path.join(self.instance_dir, "mask")
        self.mask_embed_dir = os.path.join(self.instance_dir, "mask_embeddings")
        self.mask_basename_list = natsorted( os.listdir(self.mask_image_dir) )
        self.mask_embed_basename_list = natsorted( os.listdir(self.mask_embed_dir) )
        self.device = device
        self.target_h = cfg["cam"]["img_h"]
        self.target_w = cfg["cam"]["img_w"]
        self.depth_scale = cfg["cam"]["depth_scale"]
        self.depth_near = cfg["cam"]["depth_near"] if cfg["cam"]["depth_near"] > 0 else -1
        self.depth_far = cfg["cam"]["depth_far"] if cfg["cam"]["depth_far"] > 0 else -1
        intrinsic_file = os.path.join(self.data_location, "intrinsic", "intrinsic_depth.txt")
        cam_intrinsic = np.loadtxt(intrinsic_file)[:3, :3].astype("float32")  # ndarray(3, 3), dtype=float32
        self.cam_intrinsic = torch.from_numpy(cam_intrinsic).to(self.device)
        self.frame_num = self.get_scene_img_num()
        self.last_seg_frame_id = self.get_last_seg_frame_id(cfg["seg"]["seg_add_interval"])
        self.poses = self.get_poses()  # default: relative poses to first pose, Tensor(n, 4, 4)
        self.bbox = None
        if self.cfg["cam"]["bound"]:
            self.load_bound()
        self.gt_ply_file = self.find_gt_ply( os.path.join(data_location, "../") )
        self.min_max_xyz = self.get_bbox(self.gt_ply_file)

        # crop image edge
        self.h_crop = cfg["cam"]["h_crop"]
        self.w_crop = cfg["cam"]["w_crop"]
        if self.h_crop > 0 and self.w_crop > 0:
            self.target_h -= 2 * self.h_crop
            self.target_w -= 2 * self.w_crop
            self.cam_intrinsic[0, 2] -= self.w_crop
            self.cam_intrinsic[1, 2] -= self.h_crop

        self.pinhole_cam_intrinsic = self.get_intrinsics(self.cam_intrinsic.cpu().numpy())  # o3d.camera.PinholeCameraIntrinsic obj

        # guard: __getitem__ falls back to latest_seg_img on frames without a seg result;
        # initialize it so the very first frame lacking one doesn't crash
        self.latest_seg_img = torch.zeros((self.target_h, self.target_w), dtype=torch.uint8, device=self.device)


    def get_scene_img_num(self):
        color_img_list = os.listdir(os.path.join(self.data_location, "color"))
        color_img_list.sort(key=lambda x:int(x[:-4]))
        return len(color_img_list)

    def __len__(self):
        return self.frame_num

    def find_gt_ply(self, dir):
        gt_ply_files = glob.glob( os.path.join(dir, "*_vh_clean_2.ply") )
        if len(gt_ply_files) == 1:
            return gt_ply_files[0]
        else:
            return None

    def get_bbox(self, ply_file):
        if ply_file is None or not os.path.exists(ply_file):
            min_max_xyz = [[0., 10.], [0., 10.], [0., 5.]]
        else:
            pc = o3d.io.read_point_cloud(ply_file)
            min_xyz = pc.get_min_bound().astype("float32")
            max_xyz = pc.get_max_bound().astype("float32")
            min_max_xyz = np.stack([min_xyz, max_xyz], axis=-1).tolist()
        return min_max_xyz

    def get_last_seg_frame_id(self, seg_interval=0):
        if seg_interval <= 0:
            seg_interval = self.cfg["seg"]["seg_add_interval"]

        seg_frame_ids = [ int(mask_base_name[:-4]) for mask_base_name in self.mask_basename_list ]
        last_seg_frame_id = seg_frame_ids[0]
        for seg_frame_id in seg_frame_ids[::-1]:
            if seg_frame_id % seg_interval == 0:
                last_seg_frame_id = seg_frame_id
                break
        return last_seg_frame_id

    # @brief: load bounding box of GT mesh
    def load_bound(self):
        gt_mesh_path = os.path.join(self.data_location, "../", "%s_vh_clean_2.ply" % self.seq_name)
        if os.path.exists(gt_mesh_path):
            gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
            self.bbox = gt_mesh.get_oriented_bounding_box()  # open3d.geometry.OrientedBoundingBox obj

    def load_poses(self):
        poses = []
        posefiles = natsorted( glob.glob( os.path.join(self.data_location, "pose/*.txt") ) )
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile).astype("float32"))
            poses.append(_pose)
        poses = torch.stack(poses, dim=0).to(self.device)
        return poses

    def get_poses(self, relative=False):
        poses = self.load_poses()  # Tensor(n, 4, 4)
        self.first_pose_c2w = poses[0]
        if relative:  # default
            pose_first = poses[0]  # Tensor(4, 4)
            pose_first_inv = torch.inverse(pose_first).unsqueeze(0).repeat(poses.shape[0], 1, 1)  # Tensor(n, 4, 4)
            final_poses = compose_transformations(pose_first_inv, poses)
        else:
            final_poses = poses
        return final_poses

    def get_intrinsics(self, intrinsic_mat):
        intrinsic_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_cam_parameters.set_intrinsics(self.target_w, self.target_h, intrinsic_mat[0, 0], intrinsic_mat[1, 1], intrinsic_mat[0, 2], intrinsic_mat[1, 2])
        return intrinsic_cam_parameters

    def __getitem__(self, index):
        color_img_path = os.path.join(self.data_location, "color", "%d.jpg" % index)
        depth_img_path = os.path.join(self.data_location, "depth", "%d.png" % index)

        # Step 1: load segmentation image (and semantic feature of each detected mask)
        mask_image_basename = "%d.png" % index
        mask_embed_basename = "%d.pt" % index
        if mask_image_basename in self.mask_basename_list and mask_embed_basename in self.mask_embed_basename_list:
            instance_path = os.path.join(self.mask_image_dir, mask_image_basename)
            segmentation = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)  # ndarray(H, W), dtype=uint8
            segmentation = torch.from_numpy(segmentation).to(self.device)
            if self.h_crop > 0 and self.w_crop > 0:
                segmentation = segmentation[self.h_crop:-self.h_crop, self.w_crop:-self.w_crop]
            self.latest_seg_img = segmentation

            instance_embed_path = os.path.join(self.mask_embed_dir, mask_embed_basename)
            mask_embeddings = torch.load(instance_embed_path).to(self.device)
            seg_flag = True
        else:
            segmentation = torch.zeros_like(self.latest_seg_img).to(self.device)
            mask_embeddings = torch.zeros((1, self.cfg["mask"]["feature_dim"])).to(self.device)
            seg_flag = False

        # Step 2: load color/depth image
        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
        depth_img = depth_img.astype("float32") / self.depth_scale

        color_img = cv2.imread(color_img_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # BGR --> RGB
        # color_img = cv2.imread(color_img_path)
        if color_img.shape[0] != depth_img.shape[0] or color_img.shape[1] != depth_img.shape[1]:
            color_img = cv2.resize(color_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        # color_img = color_img[:, :, ::-1]  # BGR2RGB

        color_img = torch.from_numpy(color_img).to(self.device) / 255  # Tensor(j, w, 3), [0, 1], dtype=float32
        depth_img = torch.from_numpy(depth_img).to(self.device)  # Tensor(h, w)

        # depth clipping
        if self.depth_near > 0 and self.depth_far > 0:
            depth_mask = ( (depth_img > self.depth_near) & (depth_img < self.depth_far) )
            depth_img = torch.where(depth_mask, depth_img, torch.zeros_like(depth_img))

        if self.h_crop > 0 and self.w_crop > 0:
            color_img = color_img[self.h_crop:-self.h_crop, self.w_crop:-self.w_crop, :]
            depth_img = depth_img[self.h_crop:-self.h_crop, self.w_crop:-self.w_crop]

        pose_matrix = self.poses[index]  # Tensor(4, 4)
        return color_img, depth_img, pose_matrix, segmentation, mask_embeddings, seg_flag


class MyDataset(Dataset):
    def __init__(self, data_location, instance_dir, cfg, device="cuda:0"):
        self.cfg = cfg
        self.data_location = data_location
        self.seq_name = self.data_location.split("/")[-2]
        self.instance_dir = instance_dir
        self.mask_image_dir = os.path.join(self.instance_dir, "mask")
        self.mask_embed_dir = os.path.join(self.instance_dir, "mask_embeddings")
        self.mask_basename_list = natsorted( os.listdir(self.mask_image_dir) )
        self.mask_embed_basename_list = natsorted( os.listdir(self.mask_embed_dir) )
        self.device = device
        self.target_h = cfg["cam"]["img_h"]
        self.target_w = cfg["cam"]["img_w"]
        self.depth_scale = cfg["cam"]["depth_scale"]
        self.depth_near = cfg["cam"]["depth_near"] if cfg["cam"]["depth_near"] > 0 else -1
        self.depth_far = cfg["cam"]["depth_far"] if cfg["cam"]["depth_far"] > 0 else -1
        intrinsic_file = os.path.join(self.data_location, "intrinsic_depth.txt")
        cam_intrinsic = np.loadtxt(intrinsic_file)[:3, :3].astype("float32")  # ndarray(3, 3), dtype=float32
        self.cam_intrinsic = torch.from_numpy(cam_intrinsic).to(self.device)
        self.frame_num = self.get_scene_img_num()
        self.last_seg_frame_id = self.get_last_seg_frame_id(cfg["seg"]["seg_add_interval"])
        self.poses = self.get_poses()  # default: relative poses to first pose, Tensor(n, 4, 4)
        self.bbox = None

        # crop image edge
        self.h_crop = cfg["cam"]["h_crop"]
        self.w_crop = cfg["cam"]["w_crop"]
        if self.h_crop > 0 and self.w_crop > 0:
            self.target_h -= 2 * self.h_crop
            self.target_w -= 2 * self.w_crop
            self.cam_intrinsic[0, 2] -= self.w_crop
            self.cam_intrinsic[1, 2] -= self.h_crop

        self.pinhole_cam_intrinsic = self.get_intrinsics(self.cam_intrinsic.cpu().numpy())  # o3d.camera.PinholeCameraIntrinsic obj
        self.rgb_files, self.depth_files = self.get_rgbd()


    def get_scene_img_num(self):
        color_img_list = os.listdir(os.path.join(self.data_location, "color"))
        color_img_list = natsorted(color_img_list)
        return len(color_img_list)

    def __len__(self):
        return self.frame_num

    def get_last_seg_frame_id(self, seg_interval=0):
        if seg_interval <= 0:
            seg_interval = self.cfg["seg"]["seg_add_interval"]

        seg_frame_ids = [ int(mask_base_name[:4]) for mask_base_name in self.mask_basename_list ]
        last_seg_frame_id = seg_frame_ids[0]
        for seg_frame_id in seg_frame_ids[::-1]:
            if seg_frame_id % seg_interval == 0:
                last_seg_frame_id = seg_frame_id
                break
        return last_seg_frame_id

    def load_poses(self):
        poses = []
        posefiles = natsorted( glob.glob( os.path.join(self.data_location, "poses/*.txt") ) )
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile).astype("float32"))
            poses.append(_pose)
        poses = torch.stack(poses, dim=0).to(self.device)
        return poses

    def get_poses(self, relative=False):
        poses = self.load_poses()  # Tensor(n, 4, 4)
        self.first_pose_c2w = poses[0]
        if relative:  # default
            pose_first = poses[0]  # Tensor(4, 4)
            pose_first_inv = torch.inverse(pose_first).unsqueeze(0).repeat(poses.shape[0], 1, 1)  # Tensor(n, 4, 4)
            final_poses = compose_transformations(pose_first_inv, poses)
        else:
            final_poses = poses
        return final_poses

    def get_intrinsics(self, intrinsic_mat):
        intrinsic_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_cam_parameters.set_intrinsics(self.target_w, self.target_h, intrinsic_mat[0, 0], intrinsic_mat[1, 1], intrinsic_mat[0, 2], intrinsic_mat[1, 2])
        return intrinsic_cam_parameters

    def get_rgbd(self):
        rgb_dir = os.path.join(self.data_location, "color")
        depth_dir = os.path.join(self.data_location, "depth")
        rgb_files = natsorted( os.listdir(rgb_dir) )
        depth_files = natsorted( os.listdir(depth_dir) )
        return rgb_files, depth_files

    def __getitem__(self, index):
        color_img_path = self.rgb_files[index]
        depth_img_path = self.depth_files[index]

        # Step 1: load segmentation image (and semantic feature of each detected mask)
        mask_image_basename = "%d.png" % index
        mask_embed_basename = "%d.pt" % index
        if mask_image_basename in self.mask_basename_list and mask_embed_basename in self.mask_embed_basename_list:
            instance_path = os.path.join(self.mask_image_dir, mask_image_basename)
            segmentation = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)  # ndarray(H, W), dtype=uint8
            segmentation = torch.from_numpy(segmentation).to(self.device)
            if self.h_crop > 0 and self.w_crop > 0:
                segmentation = segmentation[self.h_crop:-self.h_crop, self.w_crop:-self.w_crop]
            self.latest_seg_img = segmentation

            instance_embed_path = os.path.join(self.mask_embed_dir, mask_embed_basename)
            mask_embeddings = torch.load(instance_embed_path).to(self.device)
            seg_flag = True
        else:
            segmentation = torch.zeros_like(self.latest_seg_img).to(self.device)
            mask_embeddings = torch.zeros((1, self.cfg["mask"]["feature_dim"])).to(self.device)
            seg_flag = False

        # Step 2: load color/depth image
        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
        depth_img = depth_img.astype("float32") / self.depth_scale

        color_img = cv2.imread(color_img_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # BGR --> RGB
        if color_img.shape[0] != depth_img.shape[0] or color_img.shape[1] != depth_img.shape[1]:
            color_img = cv2.resize(color_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        color_img = torch.from_numpy(color_img).to(self.device) / 255  # Tensor(j, w, 3), [0, 1], dtype=float32
        depth_img = torch.from_numpy(depth_img).to(self.device)  # Tensor(h, w)

        # depth clipping
        if self.depth_near > 0 and self.depth_far > 0:
            depth_mask = ( (depth_img > self.depth_near) & (depth_img < self.depth_far) )
            depth_img = torch.where(depth_mask, depth_img, torch.zeros_like(depth_img))

        if self.h_crop > 0 and self.w_crop > 0:
            color_img = color_img[self.h_crop:-self.h_crop, self.w_crop:-self.w_crop, :]
            depth_img = depth_img[self.h_crop:-self.h_crop, self.w_crop:-self.w_crop]

        pose_matrix = self.poses[index]  # Tensor(4, 4)
        return color_img, depth_img, pose_matrix, segmentation, mask_embeddings, seg_flag
