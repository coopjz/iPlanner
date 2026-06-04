# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import os
import json
from tsdf_map import TSDF_Map
from esdf_mapping import TSDF_Creator, DepthReconstruction

if __name__ == '__main__':
    
    root_folder = os.getenv('EXPERIMENT_DIRECTORY', os.getcwd())
    # Load parameters from json file
    config_path = os.getenv('IP_DATA_GENERATION_CONFIG',
                             os.path.join(os.path.dirname(root_folder), 'config', 'data_generation.json'))
    print("Data generation config:", config_path)
    with open(config_path) as json_file:
        parameters = json.load(json_file)
    
    folder_name = parameters.get('folder_name', "CollectedData")
    folder_path = os.path.join(*[root_folder, "data"])
    ids_path = os.path.join(folder_path, parameters.get('collect_list', "collect_list.txt"))
    
    if not folder_name == "":
        folder_path = os.path.join(folder_path, folder_name)
    env_list = []
    with open(ids_path) as f:
        lines = f.readlines()
        for line in lines:
            env_list.append(line.rstrip())
    print("Env List: ", env_list)

    outfolder_name = parameters.get('outfolder_name', "TrainingData")
    output_folder = os.path.join(*[root_folder, "data", outfolder_name])

    image_type = parameters.get('image_type', "depth")
    voxel_size = parameters.get('voxel_size', 0.05)
    robot_size = parameters.get('robot_size', 0.3)  # the inflated robot radius
    map_name = parameters.get('map_name', "tsdf1")
    is_max_iter = parameters.get('is_max_iter', True)
    max_depth_range = parameters.get('max_depth_range', 10.0)
    is_flat_ground = parameters.get('is_flat_ground', True)
    is_visualize = parameters.get('is_visualize', False)
    robot_height_override = parameters.get('robot_height', None)
    ground_z_override = parameters.get('ground_z', None)
    ground_z_offset = parameters.get('ground_z_offset', None)

    for env_name in env_list:
        root_path = os.path.join(*[folder_path, env_name])
        image_path = os.path.join(root_path, 'depth')

        total_data_n = len([name for name in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, name))])
        print("================= Reconstruction of env: %s =================="%(env_name))
        out_path = os.path.join(output_folder, env_name)
        
        depth_constructor = DepthReconstruction(root_path, out_path, 0, 100, voxel_size*0.9, max_depth_range, is_max_iter)
        depth_constructor.depth_map_reconstruction(is_flat_ground=is_flat_ground)
        depth_constructor.save_reconstructed_data(image_type=image_type)
        avg_height = depth_constructor.avg_height
        robot_height = avg_height if robot_height_override is None else robot_height_override
        if ground_z_override is not None:
            ground_z = ground_z_override
        elif ground_z_offset is not None:
            ground_z = avg_height + ground_z_offset
        else:
            ground_z = 0.0
        print("Average Height: ", avg_height)
        print("TSDF robot height: ", robot_height)
        print("TSDF ground z: ", ground_z)
        if is_visualize:
            depth_constructor.show_point_cloud()

        # Construct the 2D cost map
        tsdf_creator = TSDF_Creator(out_path, voxel_size=voxel_size, robot_size=robot_size,
                                    robot_height=robot_height, ground_z=ground_z)
        tsdf_creator.read_point_from_file("cloud.ply")
        data, coord, params = tsdf_creator.create_TSDF_map()
        if is_visualize:
            tsdf_creator.visualize_cloud(tsdf_creator.obs_pcd)
            tsdf_creator.visualize_cloud(tsdf_creator.free_pcd)

        # Save the esdf map
        tsdf_map = TSDF_Map()
        tsdf_map.DirectLoadMap(data, coord, params)
        tsdf_map.SaveTSDFMap(out_path, map_name)
        if is_visualize:
            tsdf_map.ShowTSDFMap(cost_map=True)
