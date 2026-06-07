# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import os
import cv2
import json
import math
import shutil
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R


class CloudUtils:
    @staticmethod
    def create_open3d_cloud(points, voxel_size):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size)
        return pcd

    @staticmethod
    def extract_cloud_from_image(P_matrix, im, T, min_dist=0.2, max_dist=50, scale=1000.0):
        p_inv = np.linalg.inv(P_matrix)
        im = im / scale
        im[im<min_dist] = 1e-3
        im[im>max_dist] = 1e-3

        T_z = np.concatenate((T, np.expand_dims(1.0/im, axis=0)), axis=0).reshape(4, -1)
        P = np.multiply(im.reshape(1, -1), p_inv.dot(T_z)).T[:,:3]
        return P

class CameraUtils:
    @staticmethod
    def compute_pixel_tensor(x_nums, y_nums):
        T = np.zeros([3, x_nums, y_nums])
        for u in range(x_nums):
            for v in range(y_nums):
                T[:, u, v] = np.array([u, v, 1.0])
        return T

    @staticmethod
    def compute_e_matrix(odom, is_flat_ground, cameraR, cameraT):
        Rc, C = CameraUtils.compute_camera_pose(odom, is_flat_ground, cameraR, cameraT)
        C = C.reshape(-1,1)
        Rc_t = Rc.as_matrix().T
        E = np.concatenate((Rc_t, -Rc_t.dot(C)), axis=1)
        E = np.concatenate((E, np.array([0.0, 0.0, 0.0, 1.0]).reshape(1,-1)), axis=0)
        return E

    @staticmethod
    def compute_camera_pose(odom, is_flat_ground, cameraR, cameraT):
        """Return camera-to-world rotation and camera center.

        This intentionally mirrors the legacy ``compute_e_matrix`` convention:
        the camera translation is added in the odom/world frame instead of being
        rotated by the base pose.  Keeping this convention avoids changing the
        existing depth and Open3D-TSDF paths while allowing the raycast mapper to
        use an explicit camera-to-world pose.
        """
        odom = np.asarray(odom, dtype=float).copy()
        Rc = R.from_quat(odom[3:])
        if is_flat_ground:
            euler = Rc.as_euler('xyz', degrees=False)
            euler[1] = 0.0
            Rc = R.from_euler("xyz", euler, degrees=False)
        Rc = Rc * cameraR
        C = odom[:3] + cameraT
        return Rc, C

class DataUtils:
    @staticmethod
    def read_odom_list(odom_path):
        odom_list = []
        avg_height = 0.0
        with open(odom_path) as f:
            lines = f.readlines()
            for line in lines:
                odom = np.fromstring(line[1:-1], dtype=float, sep=', ')
                avg_height = avg_height + odom[2]
                odom_list.append(list(odom))
            avg_height =  avg_height / len(lines)
        return odom_list, avg_height

    @staticmethod
    def read_intrinsic(intrinsic_path):
        with open(intrinsic_path) as f:
            lines = f.readlines()
            elems = np.fromstring(lines[0][1:-1], dtype=float, sep=', ')
        if len(elems) == 12:
            P = np.array(elems).reshape(3, 4)
            K = np.concatenate((P, np.array([0.0, 0.0, 0.0, 1.0]).reshape(1,-1)), axis=0)
        else:
            K = np.array(elems).reshape(4, 4)
        return K

    @staticmethod
    def read_extrinsic(extrinsic_path):
        with open(extrinsic_path) as f:
            lines = f.readlines()
            elems = np.fromstring(lines[0][1:-1], dtype=float, sep=', ')
        CR = R.from_quat(elems[:4])
        CT = np.array(elems[4:])
        return CR, CT

    @staticmethod
    def load_point_cloud(path):
        return o3d.io.read_point_cloud(path)

    @staticmethod
    def list_indexed_files(folder_path, suffix):
        if not os.path.isdir(folder_path):
            return []

        indexed_files = []
        for name in os.listdir(folder_path):
            stem, ext = os.path.splitext(name)
            if ext.lower() != suffix.lower():
                continue
            try:
                idx = int(stem)
            except ValueError:
                continue
            indexed_files.append((idx, os.path.join(folder_path, name)))
        indexed_files.sort(key=lambda item: item[0])
        return [path for _, path in indexed_files]

    @staticmethod
    def prepare_output_folders(out_path, image_type):
        depth_im_path = os.path.join(out_path, image_type)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            os.makedirs(depth_im_path)
            os.makedirs(os.path.join(out_path, "maps", "cloud"))
            os.makedirs(os.path.join(out_path, "maps", "data"))
            os.makedirs(os.path.join(out_path, "maps", "params"))
        elif os.path.exists(depth_im_path):  # remove existing files
            for efile in os.listdir(depth_im_path):
                os.remove(os.path.join(depth_im_path, efile))
        else:
            os.makedirs(depth_im_path)
        os.makedirs(os.path.join(out_path, "maps", "cloud"), exist_ok=True)
        os.makedirs(os.path.join(out_path, "maps", "data"), exist_ok=True)
        os.makedirs(os.path.join(out_path, "maps", "params"), exist_ok=True)
        return None
    
    @staticmethod
    def load_images(start_id, end_id, root_path, image_type):
        im_arr_list = []
        im_path = os.path.join(root_path, image_type)
        for idx in range(start_id, end_id):
            path = os.path.join(im_path, str(idx) + ".png")
            im = cv2.imread(path, cv2.IMREAD_ANYDEPTH).T
            im_arr_list.append(im)
        print("total number of images for reconstruction: {}".format(len(im_arr_list)))
        return im_arr_list

    @staticmethod
    def save_images(out_path, im_arr_list, image_type, is_transpose=True):
        for idx, img in enumerate(im_arr_list):
            if is_transpose:
                img = img.T
            cv2.imwrite(os.path.join(out_path, image_type, f"{idx}.png"), img)
        return None

    @staticmethod
    def save_odom_list(out_path, odom_list, start_id, num_images):
        with open(os.path.join(out_path, "odom_ground_truth.txt"), 'w') as f:
            for i in range(start_id, start_id + num_images):
                f.write(str(odom_list[i]) + "\n")
        return None

    @staticmethod
    def save_extrinsic(out_path, cameraR, cameraT):
        with open(os.path.join(out_path, "camera_extrinsic.txt"), 'w') as f:
            f.write(str(list(cameraR.as_quat()) + list(cameraT)) + "\n")
        return None

    @staticmethod
    def save_intrinsic(out_path, K):
        with open(os.path.join(out_path, "depth_intrinsic.txt"), 'w') as f:
            f.write(str(K.flatten().tolist()) + "\n")
        return None

    @staticmethod
    def save_point_cloud(out_path, pcd):
        o3d.io.write_point_cloud(os.path.join(out_path, "cloud.ply"), pcd)  # save point cloud
        return None

    @staticmethod
    def copy_if_exists(src_path, dst_path):
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)
            return True
        return False

    @staticmethod
    def pose_to_matrix(pose):
        T = np.eye(4)
        T[:3, :3] = R.from_quat(pose[3:]).as_matrix()
        T[:3, 3] = np.array(pose[:3])
        return T

    @staticmethod
    def extrinsic_to_matrix(rotation, translation):
        T = np.eye(4)
        T[:3, :3] = rotation.as_matrix()
        T[:3, 3] = translation
        return T

    @staticmethod
    def transform_points(points, transform):
        if points.shape[0] == 0:
            return points
        points_h = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
        return (transform.dot(points_h.T)).T[:, :3]

    @staticmethod
    def normalize_array_to_uint8(array, transpose=True):
        image = np.asarray(array, dtype=np.float32)
        if transpose:
            image = np.flipud(image.T)
        finite_mask = np.isfinite(image)
        if not np.any(finite_mask):
            return np.zeros(image.shape, dtype=np.uint8)
        finite_values = image[finite_mask]
        min_value = np.min(finite_values)
        max_value = np.max(finite_values)
        if max_value - min_value < 1e-6:
            return np.zeros(image.shape, dtype=np.uint8)
        image = (image - min_value) / (max_value - min_value)
        image[~finite_mask] = 0.0
        return np.uint8(np.clip(image * 255.0, 0, 255))

    @staticmethod
    def save_array_visualization(array, output_path, colormap=None, transpose=True):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image = DataUtils.normalize_array_to_uint8(array, transpose=transpose)
        if colormap is not None:
            image = cv2.applyColorMap(image, colormap)
        cv2.imwrite(output_path, image)
        return output_path

    @staticmethod
    def save_image_overview(items, output_path, tile_width=360, image_height=240, label_height=34):
        if len(items) == 0:
            return None

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cols = min(3, len(items))
        rows = int(np.ceil(len(items) / cols))
        tile_height = image_height + label_height
        canvas = np.ones((rows * tile_height, cols * tile_width, 3), dtype=np.uint8) * 255

        for idx, (label, image_path) in enumerate(items):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue

            scale = min(tile_width / image.shape[1], image_height / image.shape[0])
            new_width = max(1, int(image.shape[1] * scale))
            new_height = max(1, int(image.shape[0] * scale))
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            row = idx // cols
            col = idx % cols
            x0 = col * tile_width + (tile_width - new_width) // 2
            y0 = row * tile_height + label_height + (image_height - new_height) // 2
            canvas[y0:y0 + new_height, x0:x0 + new_width] = image

            label_origin = (col * tile_width + 10, row * tile_height + 23)
            cv2.putText(canvas, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (30, 30, 30), 1, cv2.LINE_AA)

        cv2.imwrite(output_path, canvas)
        return output_path

class TSDF_Creator:
    def __init__(self, input_path, voxel_size, robot_height, robot_size,
                 clear_dist=1.0, ground_z=0.0, ground_height=0.25,
                 trajectory_clear_radius=0.0, obstacle_reassign_threshold=0.3,
                 free_space_threshold=0.7):
        self.initialize_path_and_properties(input_path, voxel_size, robot_height, robot_size,
                                            clear_dist, ground_z, ground_height,
                                            trajectory_clear_radius,
                                            obstacle_reassign_threshold,
                                            free_space_threshold)
        self.initialize_point_clouds()

    def initialize_path_and_properties(self, input_path, voxel_size, robot_height, robot_size,
                                       clear_dist, ground_z, ground_height,
                                       trajectory_clear_radius,
                                       obstacle_reassign_threshold,
                                       free_space_threshold):
        self.input_path = input_path
        self.is_map_ready = False
        self.clear_dist = clear_dist
        self.voxel_size = voxel_size
        self.robot_height = robot_height
        self.robot_size = robot_size
        self.ground_z = ground_z
        self.ground_height = ground_height
        self.trajectory_clear_radius = trajectory_clear_radius
        self.obstacle_reassign_threshold = obstacle_reassign_threshold
        self.free_space_threshold = free_space_threshold
        self._trajectory_clear_indices = None

    def initialize_point_clouds(self):
        self.obs_pcd = o3d.geometry.PointCloud()
        self.free_pcd = o3d.geometry.PointCloud()
        
    def update_point_cloud(self, P_obs, P_free, is_downsample=False):
        self.obs_pcd.points  = o3d.utility.Vector3dVector(P_obs)
        self.free_pcd.points = o3d.utility.Vector3dVector(P_free)
        self.downsample_point_cloud(is_downsample)
        self.obs_points   = np.asarray(self.obs_pcd.points)
        self.free_points  = np.asarray(self.free_pcd.points)
        
    def read_point_from_file(self, file_name, is_filter=True):
        file_path = os.path.join(self.input_path, file_name)
        pcd_load = DataUtils.load_point_cloud(file_path)
        
        print("Running terrain analysis...")
        obs_p, free_p = self.terrain_analysis(np.asarray(pcd_load.points), self.ground_height)
        self.update_point_cloud(obs_p, free_p, is_downsample=True)
        
        if is_filter:
            obs_p = self.filter_cloud(self.obs_points, num_nbs=50, std_ratio=2.0)
            self.update_point_cloud(obs_p, free_p)
        
        self.update_map_params()
        
    def downsample_point_cloud(self, is_downsample):
        if is_downsample:
            self.obs_pcd  = self.obs_pcd.voxel_down_sample(self.voxel_size)
            self.free_pcd = self.free_pcd.voxel_down_sample(self.voxel_size * 0.85)
            
    def update_map_params(self):
        self._handle_no_points()
        self._set_map_limits_and_start_coordinates()
        self._log_map_initialization()
        self.is_map_ready = True
        
    def terrain_analysis(self, input_points, ground_height=0.25):
        print("terrain analysis thresholds: ground_z=%.3f, free=[%.3f, %.3f], obstacle=(%.3f, %.3f)" % (
            self.ground_z,
            self.ground_z - ground_height,
            self.ground_z + ground_height,
            self.ground_z + ground_height,
            self.ground_z + self.robot_height * 1.5))
        obs_points, free_points = self._initialize_point_arrays(input_points)
        obs_idx = free_idx = 0
        
        for p in input_points:
            p_height = p[2]
            if self._is_obstacle(p_height, ground_height):
                obs_points[obs_idx, :] = p
                obs_idx += 1
            elif self._is_free_space(p_height, ground_height):
                free_points[free_idx, :] = p
                free_idx += 1

        return obs_points[:obs_idx, :], free_points[:free_idx, :]
    
    def create_TSDF_map(self, sigma_smooth=2.5):
        if not self.is_map_ready:
            print("create tsdf map fails, no points received.")
            return

        free_map = np.ones([self.num_x, self.num_y])
        obs_map = self._create_obstacle_map()
        obs_map = self._clear_trajectory_corridor(obs_map, fill_value=0.0, map_name="occupancy")

        # create free place map
        free_I = self._index_array_of_points(self.free_points)
        free_map = self._create_free_space_map(free_I, free_map, sigma_smooth)

        free_map[obs_map > self.obstacle_reassign_threshold] = 1.0 # re-assign obstacles if they are in free space
        free_map = self._clear_trajectory_corridor(free_map, fill_value=0.0, map_name="free-space")
        print("occupancy map generation completed.")

        # Distance Transform
        tsdf_array = self._distance_transform_and_smooth(free_map, sigma_smooth)
        self.occupancy_array = obs_map
        self.free_array = free_map
        self.tsdf_array = tsdf_array

        viz_points = np.concatenate((self.obs_points, self.free_points), axis=0)
        # TODO: Use true terrain analysis module
        ground_array = np.ones([self.num_x, self.num_y]) * 0.0

        return [tsdf_array, viz_points, ground_array], [self.start_x, self.start_y], [self.voxel_size, self.clear_dist]

    def save_map_visualizations(self, root_path, map_name):
        if not hasattr(self, "tsdf_array") or not hasattr(self, "occupancy_array"):
            print("save map visualizations failed, TSDF map has not been created.")
            return None

        viz_dir = os.path.join(root_path, "maps", "viz")
        heatmap_path = os.path.join(viz_dir, map_name + "_heatmap.png")
        occupancy_path = os.path.join(viz_dir, map_name + "_occupancy.png")
        DataUtils.save_array_visualization(self.tsdf_array, heatmap_path, colormap=cv2.COLORMAP_JET)
        DataUtils.save_array_visualization(self.occupancy_array, occupancy_path, colormap=None)
        print("Map visualizations saved.")
        return {"heatmap": heatmap_path, "occupancy": occupancy_path}
    
    def filter_cloud(self, points, num_nbs=100, std_ratio=1.0):
        pcd = self._convert_to_point_cloud(points)
        filtered_pcd = self._remove_statistical_outliers(pcd, num_nbs, std_ratio)
        return np.asarray(filtered_pcd.points)
    
    def visualize_cloud(self, pcd):
        o3d.visualization.draw_geometries([pcd])
    
    def _convert_to_point_cloud(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def _remove_statistical_outliers(self, pcd, num_nbs, std_ratio):
        filtered_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=num_nbs, std_ratio=std_ratio)
        return filtered_pcd
    
    def _create_obstacle_map(self):
        obs_map = np.zeros([self.num_x, self.num_y])
        obs_I = self._index_array_of_points(self.obs_points)
        for i in obs_I:
            obs_map[i[0], i[1]] = 1.0
        obs_map = gaussian_filter(obs_map, sigma=self.robot_size / self.voxel_size)
        obs_map /= np.max(obs_map + 1e-5) # normalize
        return obs_map
    
    def _create_free_space_map(self, free_I, free_map, sigma_smooth):
        for i in free_I:
            if i[0] >= 0 and i[0] < self.num_x and i[1] >= 0 and i[1] < self.num_y:
                free_map[i[0], i[1]] = 0
        free_map = gaussian_filter(free_map, sigma=sigma_smooth)
        free_map /= np.max(free_map) # normalize
        free_map[free_map < self.free_space_threshold] = 0 # 0.683% is the probability of standard normal distribution
        return free_map

    def _load_trajectory_positions(self):
        odom_path = os.path.join(self.input_path, "odom_ground_truth.txt")
        if not os.path.exists(odom_path):
            return np.zeros((0, 2))

        positions = []
        with open(odom_path) as f:
            for line in f:
                odom = np.fromstring(line.strip()[1:-1], dtype=float, sep=',')
                if odom.shape[0] >= 2:
                    positions.append(odom[:2])
        if len(positions) == 0:
            return np.zeros((0, 2))
        return np.asarray(positions)

    def _get_trajectory_clear_indices(self):
        if self._trajectory_clear_indices is not None:
            return self._trajectory_clear_indices
        if self.trajectory_clear_radius is None or self.trajectory_clear_radius <= 0:
            self._trajectory_clear_indices = np.zeros((0, 2), dtype=int)
            return self._trajectory_clear_indices

        positions = self._load_trajectory_positions()
        if positions.shape[0] == 0:
            self._trajectory_clear_indices = np.zeros((0, 2), dtype=int)
            return self._trajectory_clear_indices

        center_indices = np.round((positions - np.array([self.start_x, self.start_y])) / self.voxel_size).astype(int)
        radius_cells = int(np.ceil(self.trajectory_clear_radius / self.voxel_size))
        offsets = []
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if np.hypot(dx, dy) * self.voxel_size <= self.trajectory_clear_radius:
                    offsets.append((dx, dy))

        clear_cells = set()
        for center in center_indices:
            for dx, dy in offsets:
                ix = int(center[0] + dx)
                iy = int(center[1] + dy)
                if ix >= 0 and ix < self.num_x and iy >= 0 and iy < self.num_y:
                    clear_cells.add((ix, iy))
        if len(clear_cells) == 0:
            self._trajectory_clear_indices = np.zeros((0, 2), dtype=int)
        else:
            self._trajectory_clear_indices = np.asarray(sorted(clear_cells), dtype=int)
        print("trajectory corridor clear cells: %d, radius: %.3f m" % (
            self._trajectory_clear_indices.shape[0], self.trajectory_clear_radius))
        return self._trajectory_clear_indices

    def _clear_trajectory_corridor(self, array, fill_value=0.0, map_name="map"):
        clear_indices = self._get_trajectory_clear_indices()
        if clear_indices.shape[0] == 0:
            return array
        array[clear_indices[:, 0], clear_indices[:, 1]] = fill_value
        print("trajectory corridor cleared in %s map." % map_name)
        return array
    
    def _distance_transform_and_smooth(self, free_map, sigma_smooth, is_log=True):
        dt_map = ndimage.distance_transform_edt(free_map)
        tsdf_array = gaussian_filter(dt_map, sigma=sigma_smooth)
        if is_log:
            tsdf_array = np.log(tsdf_array + 1.00001)
        return tsdf_array
    
    def _index_array_of_points(self, points):
        I = np.round((points[:, :2] - np.array([self.start_x, self.start_y])) / self.voxel_size).astype(int)
        return I
    
    def _initialize_point_arrays(self, input_points):
        return np.zeros(input_points.shape), np.zeros(input_points.shape)

    def _is_obstacle(self, p_height, ground_height):
        return (p_height > self.ground_z + ground_height) and (p_height < self.ground_z + self.robot_height * 1.5)

    def _is_free_space(self, p_height, ground_height):
        return p_height < self.ground_z + ground_height and p_height > self.ground_z - ground_height
        
    def _handle_no_points(self):
        if (self.obs_points.shape[0] == 0):
            print("No points received.")
            return
        
    def _set_map_limits_and_start_coordinates(self):
        max_x, max_y, _ = np.amax(self.obs_points, axis=0) + self.clear_dist
        min_x, min_y, _ = np.amin(self.obs_points, axis=0) - self.clear_dist
        self.num_x = np.ceil((max_x - min_x) / self.voxel_size / 10).astype(int) * 10
        self.num_y = np.ceil((max_y - min_y) / self.voxel_size / 10).astype(int) * 10
        self.start_x = (max_x + min_x) / 2.0 - self.num_x / 2.0 * self.voxel_size
        self.start_y = (max_y + min_y) / 2.0 - self.num_y / 2.0 * self.voxel_size

    def _log_map_initialization(self):
        print("tsdf map initialized, with size: %d, %d" %(self.num_x, self.num_y))
        

class DepthReconstruction:
    def __init__(self, input_path, out_path, start_id, iters, voxel_size, max_range, is_max_iter=True):
        self._initialize_paths(input_path, out_path)
        self._initialize_parameters(voxel_size, max_range, is_max_iter)
        self._read_camera_params()
        
        # odom list read
        self.odom_list, self._avg_height = DataUtils.read_odom_list(self.input_path + "/odom_ground_truth.txt")
        
        N = len(self.odom_list)
        self.start_id = 0 if self.is_max_iter else start_id
        self.end_id = N if self.is_max_iter else min(start_id + iters, N)
        
        self.is_constructed = False
        print("Ready to read depth data.")

    # public methods
    def depth_map_reconstruction(self, is_output=False, is_flat_ground=False):
        self.im_arr_list = DataUtils.load_images(self.start_id, self.end_id, self.input_path, "depth")

        x_nums, y_nums = self.im_arr_list[0].shape
        T = CameraUtils.compute_pixel_tensor(x_nums, y_nums)
        pixel_nums = x_nums * y_nums

        print("start reconstruction...")
        self.points = np.zeros([(self.end_id - self.start_id + 1) * pixel_nums, 3])

        for idx, im in enumerate(self.im_arr_list):
            odom = self.odom_list[idx + self.start_id].copy()
            if is_flat_ground:
                odom[2] = self._avg_height
            E = CameraUtils.compute_e_matrix(odom, is_flat_ground, self.cameraR, self.cameraT)
            P_matrix = self.K.dot(E)
            if is_output:
                print("Extracting points from image: ", idx + self.start_id)
            self.points[idx * pixel_nums: (idx + 1) * pixel_nums, :] = CloudUtils.extract_cloud_from_image(
                P_matrix, im, T, max_dist=self.max_range)

        print("creating open3d geometry point cloud...")
        self.pcd = CloudUtils.create_open3d_cloud(self.points, self.voxel_size)
        self.is_constructed = True
        print("construction completed.")
    
    def show_point_cloud(self):
        if not self.is_constructed:
            print("no reconstructed cloud")
        o3d.visualization.draw_geometries([self.pcd])  # visualize point cloud
        
    def save_reconstructed_data(self, image_type="depth"):
        if not self.is_constructed:
            print("save points failed, no reconstructed cloud!")
            
        print("save output files to: " + self.out_path)
        DataUtils.prepare_output_folders(self.out_path, image_type)
        
        DataUtils.save_images(self.out_path, self.im_arr_list, image_type)
        DataUtils.save_odom_list(self.out_path, self.odom_list, self.start_id, len(self.im_arr_list))
        DataUtils.save_extrinsic(self.out_path, self.cameraR, self.cameraT)
        DataUtils.save_intrinsic(self.out_path, self.K)
        DataUtils.save_point_cloud(self.out_path, self.pcd)  # save point cloud
        print("saved cost map data.")
        
    @property
    def avg_height(self):
        return self._avg_height
    
    # private methods
    def _initialize_paths(self, input_path, out_path):
        self.input_path = input_path
        self.out_path = out_path

    def _initialize_parameters(self, voxel_size, max_range, is_max_iter):
        self.voxel_size = voxel_size
        self.is_max_iter = is_max_iter
        self.max_range = max_range

    def _read_camera_params(self):
        # Get Camera Parameters
        self.K = DataUtils.read_intrinsic(self.input_path + "/depth_intrinsic.txt")
        self.cameraR, self.cameraT = DataUtils.read_extrinsic(self.input_path + "/camera_extrinsic.txt")


class DepthTSDFReconstruction:
    """Integrate depth images into an Open3D TSDF volume before 2D map creation.

    Unlike the legacy ``DepthReconstruction`` path that back-projects all depth
    pixels into a stacked point cloud, this path uses Open3D's TSDF integration.
    The integration performs projective ray updates/carving in the volume, which
    is a better match for testing whether raycast-based depth fusion produces a
    cleaner occupancy/heatmap target.
    """

    def __init__(self, input_path, out_path, start_id, iters, voxel_size, max_range,
                 is_max_iter=True, tsdf_trunc=0.2, frame_stride=1):
        self._initialize_paths(input_path, out_path)
        self._initialize_parameters(voxel_size, max_range, is_max_iter, tsdf_trunc, frame_stride)
        self._read_camera_params()

        self.odom_list, self._avg_height = DataUtils.read_odom_list(self.input_path + "/odom_ground_truth.txt")
        self.depth_files = DataUtils.list_indexed_files(os.path.join(self.input_path, "depth"), ".png")
        if len(self.depth_files) == 0:
            raise RuntimeError("No indexed depth/*.png files found for depth TSDF reconstruction.")

        N = min(len(self.odom_list), len(self.depth_files))
        if N == 0:
            raise RuntimeError("No synchronized depth and odom samples found.")
        self.start_id = 0 if self.is_max_iter else start_id
        self.end_id = N if self.is_max_iter else min(start_id + iters, N)
        self.is_constructed = False
        print("Ready to integrate depth data with Open3D TSDF.")

    def depth_tsdf_reconstruction(self, is_output=False, is_flat_ground=False):
        first_depth = cv2.imread(self.depth_files[self.start_id], cv2.IMREAD_ANYDEPTH)
        if first_depth is None:
            raise RuntimeError("Failed to read depth image: %s" % self.depth_files[self.start_id])
        height, width = first_depth.shape[:2]

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height,
            float(self.K[0, 0]), float(self.K[1, 1]),
            float(self.K[0, 2]), float(self.K[1, 2]))
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.tsdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
        dummy_color = o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8))

        integrated = 0
        print("start Open3D depth TSDF integration, stride: %d, truncation: %.3f" % (
            self.frame_stride, self.tsdf_trunc))
        for idx in range(self.start_id, self.end_id, self.frame_stride):
            depth_np = cv2.imread(self.depth_files[idx], cv2.IMREAD_ANYDEPTH)
            if depth_np is None:
                continue
            if depth_np.shape[:2] != (height, width):
                depth_np = cv2.resize(depth_np, (width, height), interpolation=cv2.INTER_NEAREST)
            if not np.any(depth_np > 0):
                continue

            depth_image = o3d.geometry.Image(depth_np)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                dummy_color,
                depth_image,
                depth_scale=1000.0,
                depth_trunc=float(self.max_range),
                convert_rgb_to_intensity=False)

            odom = self.odom_list[idx].copy()
            if is_flat_ground:
                odom[2] = self._avg_height
            extrinsic = CameraUtils.compute_e_matrix(odom, is_flat_ground, self.cameraR, self.cameraT)
            volume.integrate(rgbd, intrinsic, extrinsic)
            integrated += 1
            if is_output and integrated % 25 == 0:
                print("Integrated depth frame: ", idx)

        if integrated == 0:
            raise RuntimeError("Depth TSDF integration produced no integrated frames.")

        print("extracting point cloud from Open3D TSDF volume...")
        self.pcd = volume.extract_point_cloud()
        self.pcd = self.pcd.voxel_down_sample(self.voxel_size)
        self.is_constructed = True
        print("depth TSDF integration completed, frames integrated: %d, points: %d" % (
            integrated, np.asarray(self.pcd.points).shape[0]))

    def show_point_cloud(self):
        if not self.is_constructed:
            print("no reconstructed cloud")
        o3d.visualization.draw_geometries([self.pcd])

    def save_reconstructed_data(self, image_type="depth"):
        if not self.is_constructed:
            print("save points failed, no reconstructed cloud!")
            return

        print("save depth-TSDF output files to: " + self.out_path)
        DataUtils.prepare_output_folders(self.out_path, image_type)
        self._copy_depth_images_for_training(image_type)
        DataUtils.save_odom_list(self.out_path, self.odom_list, self.start_id, self.end_id - self.start_id)
        DataUtils.save_extrinsic(self.out_path, self.cameraR, self.cameraT)
        DataUtils.save_intrinsic(self.out_path, self.K)
        DataUtils.save_point_cloud(self.out_path, self.pcd)
        print("saved depth-TSDF cost map data.")

    @property
    def avg_height(self):
        return self._avg_height

    def _initialize_paths(self, input_path, out_path):
        self.input_path = input_path
        self.out_path = out_path

    def _initialize_parameters(self, voxel_size, max_range, is_max_iter, tsdf_trunc, frame_stride):
        self.voxel_size = voxel_size
        self.max_range = max_range
        self.is_max_iter = is_max_iter
        self.tsdf_trunc = tsdf_trunc
        self.frame_stride = max(1, int(frame_stride))

    def _read_camera_params(self):
        self.K = DataUtils.read_intrinsic(self.input_path + "/depth_intrinsic.txt")
        self.cameraR, self.cameraT = DataUtils.read_extrinsic(self.input_path + "/camera_extrinsic.txt")

    def _copy_depth_images_for_training(self, image_type):
        for out_idx, in_idx in enumerate(range(self.start_id, self.end_id)):
            src = self.depth_files[in_idx]
            dst = os.path.join(self.out_path, image_type, str(out_idx) + ".png")
            shutil.copyfile(src, dst)


class DepthRaycast2DMapper:
    """Build a 2D training map by directly voting along depth rays.

    The Open3D depth-TSDF path first extracts a surface point cloud and then
    reuses ``TSDF_Creator``'s surface projection.  That tends to close narrow
    gaps because final free-space evidence is inferred from near-ground points
    and then smoothed.  This mapper keeps the projective evidence in the final
    BEV map: each valid depth sample casts free-space votes from the camera to
    just before the measured endpoint, while endpoints in the obstacle height
    band cast hit votes.
    """

    def __init__(self, input_path, out_path, start_id, iters, voxel_size, max_range,
                 is_max_iter=True, frame_stride=1, ray_pixel_stride=8,
                 hit_pixel_stride=3, bounds_pixel_stride=8, min_depth=0.35,
                 depth_scale=1000.0, clear_dist=1.0, endpoint_clearance=0.08,
                 min_hit_votes=2, min_free_votes=2, hit_ratio_threshold=0.35,
                 ray_thickness=1, ignore_bottom_rows=0,
                 occupancy_inflation_radius=0.0,
                 obstacle_min_component_cells=1,
                 unknown_cost=1.0,
                 collision_cost_transition=0.05,
                 collision_cost_power=1.0,
                 free_carve_min_votes=0,
                 free_carve_max_hit_ratio=0.5,
                 surface_thinning_radius=0.0):
        self._initialize_paths(input_path, out_path)
        self._initialize_parameters(voxel_size, max_range, is_max_iter, frame_stride,
                                    ray_pixel_stride, hit_pixel_stride,
                                    bounds_pixel_stride, min_depth, depth_scale,
                                    clear_dist, endpoint_clearance, min_hit_votes,
                                    min_free_votes, hit_ratio_threshold,
                                    ray_thickness, ignore_bottom_rows,
                                    occupancy_inflation_radius,
                                    obstacle_min_component_cells,
                                    unknown_cost,
                                    collision_cost_transition,
                                    collision_cost_power,
                                    free_carve_min_votes,
                                    free_carve_max_hit_ratio,
                                    surface_thinning_radius)
        self._read_camera_params()

        self.odom_list, self._avg_height = DataUtils.read_odom_list(self.input_path + "/odom_ground_truth.txt")
        self.depth_files = DataUtils.list_indexed_files(os.path.join(self.input_path, "depth"), ".png")
        if len(self.depth_files) == 0:
            raise RuntimeError("No indexed depth/*.png files found for direct depth raycast mapping.")

        N = min(len(self.odom_list), len(self.depth_files))
        if N == 0:
            raise RuntimeError("No synchronized depth and odom samples found.")
        self.start_id = 0 if self.is_max_iter else start_id
        self.end_id = N if self.is_max_iter else min(start_id + iters, N)
        self.is_constructed = False
        print("Ready to build direct 2D depth raycast map.")

    def depth_raycast_map_reconstruction(self, is_flat_ground=False, robot_height=0.4,
                                         ground_z=0.0, ground_height=0.18,
                                         robot_size=0.05, sigma_smooth=0.8):
        self.is_flat_ground = is_flat_ground
        self.robot_height = robot_height
        self.ground_z = ground_z
        self.ground_height = ground_height
        self.robot_size = robot_size
        self.sigma_smooth = sigma_smooth

        self._estimate_map_bounds()
        self.hit_votes = np.zeros((self.num_x, self.num_y), dtype=np.uint32)
        self.free_votes = np.zeros((self.num_x, self.num_y), dtype=np.uint32)
        self._hit_point_blocks = []
        self._free_point_blocks = []

        processed = 0
        print("start direct depth raycast voting, frame stride: %d, ray pixel stride: %d, hit pixel stride: %d" % (
            self.frame_stride, self.ray_pixel_stride, self.hit_pixel_stride))
        print("raycast thresholds: min_depth=%.3f max_depth=%.3f ground_z=%.3f obstacle=(%.3f, %.3f)" % (
            self.min_depth, self.max_range, self.ground_z,
            self.ground_z + self.ground_height,
            self.ground_z + self.robot_height * 1.5))
        print("raycast map policy: precise occupancy inflation=%.3f m, robot clearance radius=%.3f m, transition=%.3f m" % (
            self.occupancy_inflation_radius, self.robot_size, self.collision_cost_transition))
        print("raycast wall thinning: free-carve votes=%d max-hit-ratio=%.3f, surface radius=%.3f m" % (
            self.free_carve_min_votes, self.free_carve_max_hit_ratio, self.surface_thinning_radius))

        for idx in range(self.start_id, self.end_id, self.frame_stride):
            depth_np = cv2.imread(self.depth_files[idx], cv2.IMREAD_ANYDEPTH)
            if depth_np is None:
                continue

            Rc, C = self._camera_pose(idx)
            hit_points = self._depth_to_world_points(depth_np, self.hit_pixel_stride, Rc, C)
            if hit_points.shape[0] > 0:
                self._accumulate_hit_votes(hit_points)

            ray_points = self._depth_to_world_points(depth_np, self.ray_pixel_stride, Rc, C)
            if ray_points.shape[0] > 0:
                self._accumulate_free_votes(C, ray_points)

            processed += 1
            if processed % 50 == 0:
                print("raycast processed frames: %d" % processed)

        if processed == 0:
            raise RuntimeError("Direct depth raycast mapping processed no frames.")

        self._create_arrays_from_votes()
        self.is_constructed = True
        print("direct depth raycast map completed, frames: %d, hit cells: %d, free cells: %d" % (
            processed, int(np.count_nonzero(self.occupancy_array)), int(np.count_nonzero(self.free_binary))))

    def save_reconstructed_data(self, image_type="depth"):
        if not self.is_constructed:
            print("save raycast output failed, map has not been built!")
            return

        print("save direct-raycast output files to: " + self.out_path)
        DataUtils.prepare_output_folders(self.out_path, image_type)
        self._copy_depth_images_for_training(image_type)
        DataUtils.save_odom_list(self.out_path, self.odom_list, self.start_id, self.end_id - self.start_id)
        DataUtils.save_extrinsic(self.out_path, self.cameraR, self.cameraT)
        DataUtils.save_intrinsic(self.out_path, self.K)
        DataUtils.save_point_cloud(self.out_path, self.pcd)
        self._save_vote_debug_arrays()
        print("saved direct-raycast cost map data.")

    def show_point_cloud(self):
        if not self.is_constructed:
            print("no direct-raycast visualization cloud")
            return
        o3d.visualization.draw_geometries([self.pcd])

    def create_TSDF_map(self):
        if not self.is_constructed:
            print("create direct-raycast tsdf map fails, map has not been built.")
            return None
        ground_array = np.zeros((self.num_x, self.num_y), dtype=np.float32)
        return [self.tsdf_array, self.viz_points, ground_array], [self.start_x, self.start_y], [self.voxel_size, self.clear_dist]

    def save_map_visualizations(self, root_path, map_name):
        if not self.is_constructed:
            print("save raycast map visualizations failed, map has not been built.")
            return None

        viz_dir = os.path.join(root_path, "maps", "viz")
        heatmap_path = os.path.join(viz_dir, map_name + "_heatmap.png")
        occupancy_path = os.path.join(viz_dir, map_name + "_occupancy.png")
        free_path = os.path.join(viz_dir, map_name + "_free.png")
        hit_votes_path = os.path.join(viz_dir, map_name + "_hit_votes.png")
        free_votes_path = os.path.join(viz_dir, map_name + "_free_votes.png")
        hit_ratio_path = os.path.join(viz_dir, map_name + "_hit_ratio.png")
        collision_path = os.path.join(viz_dir, map_name + "_collision_cost.png")
        unknown_path = os.path.join(viz_dir, map_name + "_unknown.png")
        raw_occupancy_path = os.path.join(viz_dir, map_name + "_occupancy_raw.png")
        carved_occupancy_path = os.path.join(viz_dir, map_name + "_occupancy_carved.png")
        thinned_occupancy_path = os.path.join(viz_dir, map_name + "_occupancy_thinned.png")

        DataUtils.save_array_visualization(self.tsdf_array, heatmap_path, colormap=cv2.COLORMAP_JET)
        DataUtils.save_array_visualization(self.occupancy_array, occupancy_path, colormap=None)
        DataUtils.save_array_visualization(self.raw_occupancy_array, raw_occupancy_path, colormap=None)
        DataUtils.save_array_visualization(self.carved_occupancy_array, carved_occupancy_path, colormap=None)
        DataUtils.save_array_visualization(self.thinned_occupancy_array, thinned_occupancy_path, colormap=None)
        DataUtils.save_array_visualization(self.free_binary.astype(np.float32), free_path, colormap=None)
        DataUtils.save_array_visualization(np.log1p(self.hit_votes.astype(np.float32)), hit_votes_path, colormap=cv2.COLORMAP_MAGMA)
        DataUtils.save_array_visualization(np.log1p(self.free_votes.astype(np.float32)), free_votes_path, colormap=cv2.COLORMAP_VIRIDIS)
        DataUtils.save_array_visualization(self.hit_ratio, hit_ratio_path, colormap=cv2.COLORMAP_JET)
        DataUtils.save_array_visualization(self.collision_cost_array, collision_path, colormap=cv2.COLORMAP_JET)
        DataUtils.save_array_visualization(self.unknown_array.astype(np.float32), unknown_path, colormap=None)
        print("Direct raycast map visualizations saved.")
        return {"heatmap": heatmap_path, "occupancy": occupancy_path,
                "free": free_path, "hit_votes": hit_votes_path,
                "free_votes": free_votes_path, "hit_ratio": hit_ratio_path,
                "collision": collision_path, "unknown": unknown_path,
                "occupancy_raw": raw_occupancy_path,
                "occupancy_carved": carved_occupancy_path,
                "occupancy_thinned": thinned_occupancy_path}

    @property
    def avg_height(self):
        return self._avg_height

    def _initialize_paths(self, input_path, out_path):
        self.input_path = input_path
        self.out_path = out_path

    def _initialize_parameters(self, voxel_size, max_range, is_max_iter, frame_stride,
                               ray_pixel_stride, hit_pixel_stride, bounds_pixel_stride,
                               min_depth, depth_scale, clear_dist, endpoint_clearance,
                               min_hit_votes, min_free_votes, hit_ratio_threshold,
                               ray_thickness, ignore_bottom_rows,
                               occupancy_inflation_radius,
                               obstacle_min_component_cells,
                               unknown_cost,
                               collision_cost_transition,
                               collision_cost_power,
                               free_carve_min_votes,
                               free_carve_max_hit_ratio,
                               surface_thinning_radius):
        self.voxel_size = voxel_size
        self.max_range = max_range
        self.is_max_iter = is_max_iter
        self.frame_stride = max(1, int(frame_stride))
        self.ray_pixel_stride = max(1, int(ray_pixel_stride))
        self.hit_pixel_stride = max(1, int(hit_pixel_stride))
        self.bounds_pixel_stride = max(1, int(bounds_pixel_stride))
        self.min_depth = float(min_depth)
        self.depth_scale = float(depth_scale)
        self.clear_dist = float(clear_dist)
        self.endpoint_clearance = float(endpoint_clearance)
        self.min_hit_votes = max(1, int(min_hit_votes))
        self.min_free_votes = max(1, int(min_free_votes))
        self.hit_ratio_threshold = float(hit_ratio_threshold)
        self.ray_thickness = max(1, int(ray_thickness))
        self.ignore_bottom_rows = max(0, int(ignore_bottom_rows))
        self.occupancy_inflation_radius = max(0.0, float(occupancy_inflation_radius))
        self.obstacle_min_component_cells = max(1, int(obstacle_min_component_cells))
        self.unknown_cost = max(0.0, float(unknown_cost))
        self.collision_cost_transition = max(0.0, float(collision_cost_transition))
        self.collision_cost_power = max(0.1, float(collision_cost_power))
        self.free_carve_min_votes = max(0, int(free_carve_min_votes))
        self.free_carve_max_hit_ratio = float(free_carve_max_hit_ratio)
        self.surface_thinning_radius = max(0.0, float(surface_thinning_radius))

    def _read_camera_params(self):
        self.K = DataUtils.read_intrinsic(self.input_path + "/depth_intrinsic.txt")
        self.cameraR, self.cameraT = DataUtils.read_extrinsic(self.input_path + "/camera_extrinsic.txt")

    def _camera_pose(self, idx):
        odom = self.odom_list[idx].copy()
        if self.is_flat_ground:
            odom[2] = self._avg_height
        return CameraUtils.compute_camera_pose(odom, self.is_flat_ground, self.cameraR, self.cameraT)

    def _depth_to_world_points(self, depth_np, pixel_stride, Rc, C):
        height, width = depth_np.shape[:2]
        row_end = max(0, height - self.ignore_bottom_rows)
        if row_end == 0:
            return np.zeros((0, 3), dtype=np.float32)

        v = np.arange(0, row_end, pixel_stride, dtype=np.int32)
        u = np.arange(0, width, pixel_stride, dtype=np.int32)
        uu, vv = np.meshgrid(u, v)
        depth = depth_np[vv, uu].astype(np.float32) / self.depth_scale
        valid = np.logical_and(depth >= self.min_depth, depth <= self.max_range)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32)

        z = depth[valid]
        u_valid = uu[valid].astype(np.float32)
        v_valid = vv[valid].astype(np.float32)
        fx = float(self.K[0, 0])
        fy = float(self.K[1, 1])
        cx = float(self.K[0, 2])
        cy = float(self.K[1, 2])
        x = (u_valid - cx) * z / fx
        y = (v_valid - cy) * z / fy
        points_cam = np.stack((x, y, z), axis=1)
        return (Rc.as_matrix().dot(points_cam.T)).T + C.reshape(1, 3)

    def _estimate_map_bounds(self):
        min_xy = np.array([np.inf, np.inf], dtype=np.float64)
        max_xy = np.array([-np.inf, -np.inf], dtype=np.float64)
        frames_with_points = 0

        for idx in range(self.start_id, self.end_id, self.frame_stride):
            depth_np = cv2.imread(self.depth_files[idx], cv2.IMREAD_ANYDEPTH)
            if depth_np is None:
                continue
            Rc, C = self._camera_pose(idx)
            min_xy = np.minimum(min_xy, C[:2])
            max_xy = np.maximum(max_xy, C[:2])
            points = self._depth_to_world_points(depth_np, self.bounds_pixel_stride, Rc, C)
            if points.shape[0] == 0:
                continue
            frames_with_points += 1
            min_xy = np.minimum(min_xy, np.min(points[:, :2], axis=0))
            max_xy = np.maximum(max_xy, np.max(points[:, :2], axis=0))

        if frames_with_points == 0 or not np.all(np.isfinite(min_xy)) or not np.all(np.isfinite(max_xy)):
            raise RuntimeError("Direct depth raycast mapping could not estimate map bounds.")

        min_xy -= self.clear_dist
        max_xy += self.clear_dist
        self.num_x = int(np.ceil((max_xy[0] - min_xy[0]) / self.voxel_size / 10.0) * 10)
        self.num_y = int(np.ceil((max_xy[1] - min_xy[1]) / self.voxel_size / 10.0) * 10)
        self.num_x = max(10, self.num_x)
        self.num_y = max(10, self.num_y)
        self.start_x = (max_xy[0] + min_xy[0]) / 2.0 - self.num_x / 2.0 * self.voxel_size
        self.start_y = (max_xy[1] + min_xy[1]) / 2.0 - self.num_y / 2.0 * self.voxel_size
        print("direct raycast map initialized, with size: %d, %d" % (self.num_x, self.num_y))

    def _points_to_indices(self, points):
        indices = np.round((points[:, :2] - np.array([self.start_x, self.start_y])) / self.voxel_size).astype(np.int32)
        valid = np.logical_and.reduce((indices[:, 0] >= 0,
                                       indices[:, 0] < self.num_x,
                                       indices[:, 1] >= 0,
                                       indices[:, 1] < self.num_y))
        return indices, valid

    def _accumulate_hit_votes(self, points):
        obstacle_mask = np.logical_and(points[:, 2] > self.ground_z + self.ground_height,
                                       points[:, 2] < self.ground_z + self.robot_height * 1.5)
        hit_points = points[obstacle_mask]
        if hit_points.shape[0] == 0:
            return
        hit_indices, valid = self._points_to_indices(hit_points)
        hit_indices = hit_indices[valid]
        if hit_indices.shape[0] == 0:
            return
        np.add.at(self.hit_votes, (hit_indices[:, 0], hit_indices[:, 1]), 1)
        if len(self._hit_point_blocks) < 200:
            self._hit_point_blocks.append(hit_points[valid][::max(1, hit_points[valid].shape[0] // 2500 + 1)])

    def _accumulate_free_votes(self, camera_center, points):
        start = np.round((camera_center[:2] - np.array([self.start_x, self.start_y])) / self.voxel_size).astype(np.int32)
        if start[0] < 0 or start[0] >= self.num_x or start[1] < 0 or start[1] >= self.num_y:
            return
        end_indices, valid = self._points_to_indices(points)
        end_indices = end_indices[valid]
        if end_indices.shape[0] == 0:
            return

        # A per-frame mask gives at most one free-space vote per cell per frame,
        # preventing high-density image regions from overwhelming the occupancy
        # evidence while preserving visibility through narrow gaps.
        free_mask = np.zeros((self.num_x, self.num_y), dtype=np.uint8)
        stop_margin_cells = max(1, int(np.ceil(self.endpoint_clearance / self.voxel_size)))
        unique_endpoints = np.unique(end_indices, axis=0)
        for end in unique_endpoints:
            clear_end = self._shorten_ray_endpoint(start, end, stop_margin_cells)
            if clear_end is None:
                continue
            cv2.line(free_mask,
                     (int(start[1]), int(start[0])),
                     (int(clear_end[1]), int(clear_end[0])),
                     1,
                     thickness=self.ray_thickness)
        self.free_votes += free_mask.astype(np.uint32)

        if len(self._free_point_blocks) < 200:
            free_points = points[valid]
            self._free_point_blocks.append(free_points[::max(1, free_points.shape[0] // 1500 + 1)])

    def _shorten_ray_endpoint(self, start, end, margin_cells):
        delta = end.astype(np.float32) - start.astype(np.float32)
        length = float(np.linalg.norm(delta))
        if length <= margin_cells + 1.0:
            return None
        clear_end = start.astype(np.float32) + delta * ((length - margin_cells) / length)
        clear_end = np.round(clear_end).astype(np.int32)
        clear_end[0] = np.clip(clear_end[0], 0, self.num_x - 1)
        clear_end[1] = np.clip(clear_end[1], 0, self.num_y - 1)
        return clear_end

    def _create_arrays_from_votes(self):
        total_votes = self.hit_votes.astype(np.float32) + self.free_votes.astype(np.float32)
        self.hit_ratio = np.divide(self.hit_votes.astype(np.float32), total_votes + 1e-6)
        free_evidence = self.free_votes >= self.min_free_votes
        raw_occupancy = np.logical_and(self.hit_votes >= self.min_hit_votes,
                                       self.hit_ratio >= self.hit_ratio_threshold)
        carved_occupancy = self._carve_occupancy_with_free_evidence(raw_occupancy)
        thinned_occupancy = self._thin_occupancy_to_observed_surface(carved_occupancy,
                                                                     free_evidence)
        filtered_occupancy = self._remove_small_obstacle_components(thinned_occupancy)
        precise_occupancy = self._inflate_binary_mask(filtered_occupancy,
                                                      self.occupancy_inflation_radius)

        self.raw_occupancy_array = raw_occupancy.astype(np.float32)
        self.carved_occupancy_array = carved_occupancy.astype(np.float32)
        self.thinned_occupancy_array = thinned_occupancy.astype(np.float32)
        self.free_binary = np.logical_and(free_evidence, np.logical_not(precise_occupancy))
        self.unknown_array = np.logical_not(np.logical_or(self.free_binary,
                                                          precise_occupancy))

        self.collision_cost_array = self._create_robot_clearance_cost(precise_occupancy)
        cost_map = np.zeros((self.num_x, self.num_y), dtype=np.float32)
        if self.unknown_cost > 0.0:
            cost_map[self.unknown_array] = self.unknown_cost
        cost_map = np.maximum(cost_map, self.collision_cost_array)
        if self.sigma_smooth > 0:
            cost_map = gaussian_filter(cost_map, sigma=self.sigma_smooth)
        self.tsdf_array = np.clip(cost_map, 0.0, 1.0).astype(np.float32)
        self.occupancy_array = precise_occupancy.astype(np.float32)
        self.free_array = np.logical_not(self.free_binary).astype(np.float32)

        hit_points = np.concatenate(self._hit_point_blocks, axis=0) if self._hit_point_blocks else np.zeros((0, 3))
        free_points = np.concatenate(self._free_point_blocks, axis=0) if self._free_point_blocks else np.zeros((0, 3))
        if hit_points.shape[0] + free_points.shape[0] == 0:
            self.viz_points = np.zeros((1, 3), dtype=np.float32)
        else:
            self.viz_points = np.concatenate((hit_points, free_points), axis=0)
        self.pcd = CloudUtils.create_open3d_cloud(self.viz_points, self.voxel_size)

        self.raycast_stats = {
            "voxel_size": self.voxel_size,
            "robot_size": self.robot_size,
            "occupancy_inflation_radius": self.occupancy_inflation_radius,
            "obstacle_min_component_cells": self.obstacle_min_component_cells,
            "unknown_cost": self.unknown_cost,
            "collision_cost_transition": self.collision_cost_transition,
            "free_carve_min_votes": self.free_carve_min_votes,
            "free_carve_max_hit_ratio": self.free_carve_max_hit_ratio,
            "surface_thinning_radius": self.surface_thinning_radius,
            "num_x": int(self.num_x),
            "num_y": int(self.num_y),
            "start_x": float(self.start_x),
            "start_y": float(self.start_y),
            "hit_cells_raw": int(np.count_nonzero(raw_occupancy)),
            "hit_cells_after_free_carve": int(np.count_nonzero(carved_occupancy)),
            "hit_cells_after_surface_thinning": int(np.count_nonzero(thinned_occupancy)),
            "hit_cells_filtered": int(np.count_nonzero(filtered_occupancy)),
            "hit_cells_inflated": int(np.count_nonzero(precise_occupancy)),
            "robot_collision_cells_over_0_5": int(np.count_nonzero(self.collision_cost_array >= 0.5)),
            "unknown_cells": int(np.count_nonzero(self.unknown_array)),
            "free_cells": int(np.count_nonzero(self.free_binary)),
            "max_hit_votes": int(np.max(self.hit_votes)) if self.hit_votes.size else 0,
            "max_free_votes": int(np.max(self.free_votes)) if self.free_votes.size else 0,
            "min_hit_votes": self.min_hit_votes,
            "min_free_votes": self.min_free_votes,
            "hit_ratio_threshold": self.hit_ratio_threshold,
        }

    def _carve_occupancy_with_free_evidence(self, occupancy):
        if self.free_carve_min_votes <= 0:
            return occupancy.copy()
        carve_mask = np.logical_and(self.free_votes >= self.free_carve_min_votes,
                                    self.hit_ratio <= self.free_carve_max_hit_ratio)
        carved = occupancy.copy()
        carved[carve_mask] = False
        return carved

    def _thin_occupancy_to_observed_surface(self, occupancy, free_evidence):
        if self.surface_thinning_radius <= 0.0 or not np.any(occupancy):
            return occupancy.copy()
        if not np.any(free_evidence):
            return occupancy.copy()

        # Keep only obstacle cells close to observed free space. This removes
        # wall thickness caused by endpoint noise / multi-view accumulation,
        # while preserving the visible obstacle surface used by the planner.
        dist_to_free = ndimage.distance_transform_edt(np.logical_not(free_evidence)) * self.voxel_size
        surface_mask = dist_to_free <= self.surface_thinning_radius
        return np.logical_and(occupancy, surface_mask)

    def _inflate_binary_mask(self, mask, radius_m):
        radius_cells = max(0, int(np.ceil(float(radius_m) / self.voxel_size)))
        if radius_cells <= 0:
            return mask.copy()
        return ndimage.binary_dilation(mask, structure=self._disk_structure(radius_cells))

    def _remove_small_obstacle_components(self, mask):
        if self.obstacle_min_component_cells <= 1 or not np.any(mask):
            return mask.copy()
        labels, count = ndimage.label(mask)
        if count == 0:
            return mask.copy()
        sizes = np.bincount(labels.ravel())
        keep = sizes >= self.obstacle_min_component_cells
        keep[0] = False
        return keep[labels]

    def _create_robot_clearance_cost(self, occupancy):
        if not np.any(occupancy):
            return np.zeros((self.num_x, self.num_y), dtype=np.float32)
        clearance = ndimage.distance_transform_edt(np.logical_not(occupancy)) * self.voxel_size
        if self.robot_size <= 0.0:
            cost = occupancy.astype(np.float32)
        elif self.collision_cost_transition <= 0.0:
            cost = (clearance <= self.robot_size).astype(np.float32)
        else:
            # A soft footprint cost keeps narrow but feasible gaps continuous in
            # the heatmap. The 0.5 iso-cost line is the true robot radius.
            x = (clearance - self.robot_size) / self.collision_cost_transition
            x = np.clip(x, -60.0, 60.0)
            cost = 1.0 / (1.0 + np.exp(x))
            cost = np.power(cost, self.collision_cost_power)
        cost[occupancy] = 1.0
        return cost.astype(np.float32)

    def _disk_structure(self, radius_cells):
        y, x = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
        return (x * x + y * y) <= radius_cells * radius_cells

    def _copy_depth_images_for_training(self, image_type):
        for out_idx, in_idx in enumerate(range(self.start_id, self.end_id)):
            src = self.depth_files[in_idx]
            dst = os.path.join(self.out_path, image_type, str(out_idx) + ".png")
            shutil.copyfile(src, dst)

    def _save_vote_debug_arrays(self):
        debug_dir = os.path.join(self.out_path, "maps", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        np.save(os.path.join(debug_dir, "hit_votes.npy"), self.hit_votes)
        np.save(os.path.join(debug_dir, "free_votes.npy"), self.free_votes)
        np.save(os.path.join(debug_dir, "hit_ratio.npy"), self.hit_ratio)
        np.save(os.path.join(debug_dir, "occupancy_raw.npy"), self.raw_occupancy_array)
        np.save(os.path.join(debug_dir, "occupancy_carved.npy"), self.carved_occupancy_array)
        np.save(os.path.join(debug_dir, "occupancy_thinned.npy"), self.thinned_occupancy_array)
        np.save(os.path.join(debug_dir, "occupancy_final.npy"), self.occupancy_array)
        np.save(os.path.join(debug_dir, "collision_cost.npy"), self.collision_cost_array)
        with open(os.path.join(debug_dir, "raycast_stats.json"), "w") as f:
            json.dump(self.raycast_stats, f, indent=2, sort_keys=True)


class ScanReconstruction:
    """Fuse collected, de-skewed scan point clouds into the map cloud.

    The data collector already stores synchronized ``scan/<idx>.ply`` files.
    This class provides an alternative to depth-image back-projection: transform
    each scan into the world frame, voxel down-sample the fused cloud, then save
    it as the same ``cloud.ply`` consumed by ``TSDF_Creator``.
    """

    VALID_TRANSFORM_MODES = ("sensor_to_world", "base_to_world", "world")

    def __init__(self, input_path, out_path, start_id, iters, voxel_size, max_range,
                 is_max_iter=True, transform_mode="sensor_to_world"):
        self._initialize_paths(input_path, out_path)
        self._initialize_parameters(voxel_size, max_range, is_max_iter, transform_mode)
        self._read_params()

        self.odom_list, self._avg_height = DataUtils.read_odom_list(self.input_path + "/odom_ground_truth.txt")
        self.scan_files = DataUtils.list_indexed_files(os.path.join(self.input_path, "scan"), ".ply")
        if len(self.scan_files) == 0:
            raise RuntimeError("No indexed scan/*.ply files found for scan reconstruction.")

        N = min(len(self.odom_list), len(self.scan_files))
        if N == 0:
            raise RuntimeError("No synchronized scan and odom samples found.")
        self.start_id = 0 if self.is_max_iter else start_id
        self.end_id = N if self.is_max_iter else min(start_id + iters, N)

        self.is_constructed = False
        print("Ready to read de-skewed scan data.")

    def scan_cloud_reconstruction(self, is_output=False):
        point_blocks = []
        T_base_scan = DataUtils.extrinsic_to_matrix(self.scanR, self.scanT) if self.scanR is not None else None

        print("start scan reconstruction with transform mode: %s" % self.transform_mode)
        for idx in range(self.start_id, self.end_id):
            if is_output:
                print("Extracting points from scan: ", idx)

            pcd = DataUtils.load_point_cloud(self.scan_files[idx])
            points = np.asarray(pcd.points)
            if points.shape[0] == 0:
                continue

            if self.transform_mode == "world":
                points = self._range_filter(points, origin=np.asarray(self.odom_list[idx][:3]))
            else:
                points = self._range_filter(points)
            if points.shape[0] == 0:
                continue

            if self.transform_mode == "sensor_to_world":
                if T_base_scan is None:
                    raise RuntimeError("scan_transform_mode=sensor_to_world requires scan_extrinsic.txt.")
                T_world_base = DataUtils.pose_to_matrix(self.odom_list[idx])
                points = DataUtils.transform_points(points, T_world_base.dot(T_base_scan))
            elif self.transform_mode == "base_to_world":
                T_world_base = DataUtils.pose_to_matrix(self.odom_list[idx])
                points = DataUtils.transform_points(points, T_world_base)
            elif self.transform_mode == "world":
                # The incoming scan was already registered in the fixed/world frame.
                pass
            else:
                raise RuntimeError("Unsupported scan transform mode: %s" % self.transform_mode)

            point_blocks.append(points)

        if len(point_blocks) == 0:
            raise RuntimeError("Scan reconstruction produced no points.")

        self.points = np.concatenate(point_blocks, axis=0)
        print("creating open3d geometry point cloud from de-skewed scans...")
        self.pcd = CloudUtils.create_open3d_cloud(self.points, self.voxel_size)
        self.is_constructed = True
        print("scan construction completed.")

    def show_point_cloud(self):
        if not self.is_constructed:
            print("no reconstructed cloud")
        o3d.visualization.draw_geometries([self.pcd])

    def save_reconstructed_data(self, image_type="depth"):
        if not self.is_constructed:
            print("save points failed, no reconstructed cloud!")
            return

        print("save scan-based output files to: " + self.out_path)
        DataUtils.prepare_output_folders(self.out_path, image_type)

        self._copy_depth_images_for_training(image_type)
        DataUtils.save_odom_list(self.out_path, self.odom_list, self.start_id, self.end_id - self.start_id)
        self._copy_or_save_camera_params()
        DataUtils.copy_if_exists(os.path.join(self.input_path, "scan_extrinsic.txt"),
                                 os.path.join(self.out_path, "scan_extrinsic.txt"))
        DataUtils.save_point_cloud(self.out_path, self.pcd)
        print("saved scan-based cost map data.")

    @property
    def avg_height(self):
        return self._avg_height

    def _initialize_paths(self, input_path, out_path):
        self.input_path = input_path
        self.out_path = out_path

    def _initialize_parameters(self, voxel_size, max_range, is_max_iter, transform_mode):
        self.voxel_size = voxel_size
        self.max_range = max_range
        self.is_max_iter = is_max_iter
        self.transform_mode = transform_mode
        if self.transform_mode not in self.VALID_TRANSFORM_MODES:
            raise ValueError("scan_transform_mode must be one of %s" % (self.VALID_TRANSFORM_MODES,))

    def _read_params(self):
        self.K = DataUtils.read_intrinsic(self.input_path + "/depth_intrinsic.txt")
        self.cameraR, self.cameraT = DataUtils.read_extrinsic(self.input_path + "/camera_extrinsic.txt")
        scan_extrinsic_path = self.input_path + "/scan_extrinsic.txt"
        if os.path.exists(scan_extrinsic_path):
            self.scanR, self.scanT = DataUtils.read_extrinsic(scan_extrinsic_path)
        elif self.transform_mode == "sensor_to_world":
            raise RuntimeError("scan_transform_mode=sensor_to_world requires scan_extrinsic.txt: %s" % scan_extrinsic_path)
        else:
            self.scanR, self.scanT = None, None

    def _range_filter(self, points, origin=None):
        if self.max_range is None or self.max_range <= 0:
            return points
        if origin is None:
            ranges = np.linalg.norm(points, axis=1)
        else:
            ranges = np.linalg.norm(points - origin.reshape(1, 3), axis=1)
        return points[ranges <= self.max_range]

    def _copy_depth_images_for_training(self, image_type):
        depth_dir = os.path.join(self.input_path, "depth")
        if not os.path.isdir(depth_dir):
            print("No depth folder found; scan cloud saved without training images.")
            return

        for out_idx, in_idx in enumerate(range(self.start_id, self.end_id)):
            src = os.path.join(depth_dir, str(in_idx) + ".png")
            dst = os.path.join(self.out_path, image_type, str(out_idx) + ".png")
            if os.path.exists(src):
                shutil.copyfile(src, dst)

    def _copy_or_save_camera_params(self):
        copied_camera = DataUtils.copy_if_exists(os.path.join(self.input_path, "camera_extrinsic.txt"),
                                                 os.path.join(self.out_path, "camera_extrinsic.txt"))
        copied_depth = DataUtils.copy_if_exists(os.path.join(self.input_path, "depth_intrinsic.txt"),
                                                os.path.join(self.out_path, "depth_intrinsic.txt"))
        if not copied_camera:
            DataUtils.save_extrinsic(self.out_path, self.cameraR, self.cameraT)
        if not copied_depth:
            DataUtils.save_intrinsic(self.out_path, self.K)
        
        
