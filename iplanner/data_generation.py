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
from esdf_mapping import (DataUtils, TSDF_Creator, DepthReconstruction,
                          DepthTSDFReconstruction, DepthRaycast2DMapper,
                          ScanReconstruction)

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
    cloud_source = parameters.get('cloud_source',
                                  parameters.get('reconstruction_source',
                                                 parameters.get('map_source', "depth"))).lower()
    voxel_size = parameters.get('voxel_size', 0.05)
    robot_length = parameters.get('robot_length', None)
    robot_width = parameters.get('robot_width', None)
    if 'robot_size' in parameters:
        robot_size = float(parameters['robot_size'])  # collision/inflation radius in meters
    elif robot_width is not None:
        robot_size = float(robot_width) * 0.5
    else:
        robot_size = 0.3
    map_name = parameters.get('map_name', "tsdf1")
    is_max_iter = parameters.get('is_max_iter', True)
    max_depth_range = parameters.get('max_depth_range', 10.0)
    depth_tsdf_truncation = parameters.get('depth_tsdf_truncation', voxel_size * 4.0)
    depth_tsdf_frame_stride = parameters.get('depth_tsdf_frame_stride', 1)
    max_scan_range = parameters.get('max_scan_range', max_depth_range)
    scan_transform_mode = parameters.get('scan_transform_mode', "sensor_to_world")
    is_flat_ground = parameters.get('is_flat_ground', True)
    is_visualize = parameters.get('is_visualize', False)
    save_visualizations = parameters.get('save_visualizations', False)
    visualization_prefix = parameters.get('visualization_prefix', map_name)
    robot_height_override = parameters.get('robot_height', None)
    ground_z_override = parameters.get('ground_z', None)
    ground_z_offset = parameters.get('ground_z_offset', None)
    terrain_ground_height = parameters.get('terrain_ground_height',
                                           parameters.get('ground_height', 0.25))
    trajectory_clear_radius = parameters.get('trajectory_clear_radius', 0.0)
    map_sigma_smooth = parameters.get('map_sigma_smooth', 2.5)
    obstacle_reassign_threshold = parameters.get('obstacle_reassign_threshold', 0.3)
    free_space_threshold = parameters.get('free_space_threshold', 0.7)
    depth_raycast_frame_stride = parameters.get('depth_raycast_frame_stride',
                                                parameters.get('depth_tsdf_frame_stride', 1))
    depth_raycast_ray_pixel_stride = parameters.get('depth_raycast_ray_pixel_stride', 8)
    depth_raycast_hit_pixel_stride = parameters.get('depth_raycast_hit_pixel_stride', 3)
    depth_raycast_bounds_pixel_stride = parameters.get('depth_raycast_bounds_pixel_stride',
                                                       max(depth_raycast_ray_pixel_stride,
                                                           depth_raycast_hit_pixel_stride))
    depth_raycast_min_depth = parameters.get('depth_raycast_min_depth', 0.35)
    depth_raycast_depth_scale = parameters.get('depth_raycast_depth_scale', 1000.0)
    depth_raycast_endpoint_clearance = parameters.get('depth_raycast_endpoint_clearance', 0.08)
    depth_raycast_min_hit_votes = parameters.get('depth_raycast_min_hit_votes', 2)
    depth_raycast_min_free_votes = parameters.get('depth_raycast_min_free_votes', 2)
    depth_raycast_hit_ratio_threshold = parameters.get('depth_raycast_hit_ratio_threshold', 0.35)
    depth_raycast_ray_thickness = parameters.get('depth_raycast_ray_thickness', 1)
    depth_raycast_ignore_bottom_rows = parameters.get('depth_raycast_ignore_bottom_rows', 0)
    depth_raycast_occupancy_inflation_radius = parameters.get('depth_raycast_occupancy_inflation_radius', 0.0)
    depth_raycast_obstacle_min_component_cells = parameters.get('depth_raycast_obstacle_min_component_cells', 1)
    depth_raycast_unknown_cost = parameters.get('depth_raycast_unknown_cost', 1.0)
    depth_raycast_collision_cost_transition = parameters.get('depth_raycast_collision_cost_transition', 0.05)
    depth_raycast_collision_cost_power = parameters.get('depth_raycast_collision_cost_power', 1.0)
    depth_raycast_free_carve_min_votes = parameters.get('depth_raycast_free_carve_min_votes', 0)
    depth_raycast_free_carve_max_hit_ratio = parameters.get('depth_raycast_free_carve_max_hit_ratio', 0.5)
    depth_raycast_surface_thinning_radius = parameters.get('depth_raycast_surface_thinning_radius', 0.0)
    heatmap_overview_items = []
    occupancy_overview_items = []

    for env_name in env_list:
        root_path = os.path.join(*[folder_path, env_name])
        print("================= Reconstruction of env: %s =================="%(env_name))
        out_path = os.path.join(output_folder, env_name)

        is_direct_depth_raycast = cloud_source in (
            "depth_raycast_2d", "depth_raycast2d", "raycast_2d",
            "raycast_depth_2d", "direct_depth_raycast")

        if cloud_source == "depth":
            reconstructor = DepthReconstruction(root_path, out_path, 0, 100, voxel_size*0.9, max_depth_range, is_max_iter)
            reconstructor.depth_map_reconstruction(is_flat_ground=is_flat_ground)
            reconstructor.save_reconstructed_data(image_type=image_type)
            avg_height = reconstructor.avg_height
        elif cloud_source in ("depth_tsdf", "raycast_depth", "raycast_depth_tsdf", "open3d_depth_tsdf"):
            reconstructor = DepthTSDFReconstruction(root_path, out_path, 0, 100, voxel_size*0.9,
                                                    max_depth_range, is_max_iter,
                                                    depth_tsdf_truncation,
                                                    depth_tsdf_frame_stride)
            reconstructor.depth_tsdf_reconstruction(is_flat_ground=is_flat_ground)
            reconstructor.save_reconstructed_data(image_type=image_type)
            avg_height = reconstructor.avg_height
        elif cloud_source in ("scan", "pointcloud", "point_cloud", "deskewed_scan"):
            reconstructor = ScanReconstruction(root_path, out_path, 0, 100, voxel_size*0.9, max_scan_range,
                                               is_max_iter, scan_transform_mode)
            reconstructor.scan_cloud_reconstruction()
            reconstructor.save_reconstructed_data(image_type=image_type)
            avg_height = reconstructor.avg_height
        elif is_direct_depth_raycast:
            reconstructor = DepthRaycast2DMapper(root_path, out_path, 0, 100, voxel_size,
                                                max_depth_range, is_max_iter,
                                                depth_raycast_frame_stride,
                                                depth_raycast_ray_pixel_stride,
                                                depth_raycast_hit_pixel_stride,
                                                depth_raycast_bounds_pixel_stride,
                                                depth_raycast_min_depth,
                                                depth_raycast_depth_scale,
                                                clear_dist=1.0,
                                                endpoint_clearance=depth_raycast_endpoint_clearance,
                                                min_hit_votes=depth_raycast_min_hit_votes,
                                                min_free_votes=depth_raycast_min_free_votes,
                                                hit_ratio_threshold=depth_raycast_hit_ratio_threshold,
                                                ray_thickness=depth_raycast_ray_thickness,
                                                ignore_bottom_rows=depth_raycast_ignore_bottom_rows,
                                                occupancy_inflation_radius=depth_raycast_occupancy_inflation_radius,
                                                obstacle_min_component_cells=depth_raycast_obstacle_min_component_cells,
                                                unknown_cost=depth_raycast_unknown_cost,
                                                collision_cost_transition=depth_raycast_collision_cost_transition,
                                                collision_cost_power=depth_raycast_collision_cost_power,
                                                free_carve_min_votes=depth_raycast_free_carve_min_votes,
                                                free_carve_max_hit_ratio=depth_raycast_free_carve_max_hit_ratio,
                                                surface_thinning_radius=depth_raycast_surface_thinning_radius)
            avg_height = reconstructor.avg_height
        else:
            raise ValueError("Unsupported cloud_source '%s'. Use 'depth', 'depth_tsdf', 'depth_raycast_2d', or 'scan'." % cloud_source)

        robot_height = avg_height if robot_height_override is None else robot_height_override
        if ground_z_override is not None:
            ground_z = ground_z_override
        elif ground_z_offset is not None:
            ground_z = avg_height + ground_z_offset
        else:
            ground_z = 0.0
        print("Average Height: ", avg_height)
        if robot_length is not None or robot_width is not None:
            print("TSDF robot footprint length/width: ", robot_length, robot_width)
            if robot_width is not None and abs(robot_size - float(robot_width) * 0.5) > 1e-6:
                print("[WARN] robot_size is a radius; robot_width / 2 is %.3f m but robot_size is %.3f m." % (
                    float(robot_width) * 0.5, robot_size))
        print("TSDF robot footprint radius: ", robot_size)
        print("TSDF robot height: ", robot_height)
        print("TSDF ground z: ", ground_z)
        print("TSDF terrain ground height: ", terrain_ground_height)
        print("TSDF trajectory clear radius: ", trajectory_clear_radius)
        print("TSDF map sigma smooth: ", map_sigma_smooth)
        print("TSDF obstacle reassign threshold: ", obstacle_reassign_threshold)
        print("TSDF free space threshold: ", free_space_threshold)
        if is_visualize:
            reconstructor.show_point_cloud()

        if is_direct_depth_raycast:
            print("Direct depth raycast min depth: ", depth_raycast_min_depth)
            print("Direct depth raycast ray pixel stride: ", depth_raycast_ray_pixel_stride)
            print("Direct depth raycast hit pixel stride: ", depth_raycast_hit_pixel_stride)
            print("Direct depth raycast min hit/free votes: ", depth_raycast_min_hit_votes, depth_raycast_min_free_votes)
            print("Direct depth raycast hit ratio threshold: ", depth_raycast_hit_ratio_threshold)
            print("Direct depth raycast precise occupancy inflation radius: ", depth_raycast_occupancy_inflation_radius)
            print("Direct depth raycast soft robot clearance transition: ", depth_raycast_collision_cost_transition)
            print("Direct depth raycast free carve: ", depth_raycast_free_carve_min_votes, depth_raycast_free_carve_max_hit_ratio)
            print("Direct depth raycast surface thinning radius: ", depth_raycast_surface_thinning_radius)
            reconstructor.depth_raycast_map_reconstruction(is_flat_ground=is_flat_ground,
                                                           robot_height=robot_height,
                                                           ground_z=ground_z,
                                                           ground_height=terrain_ground_height,
                                                           robot_size=robot_size,
                                                           sigma_smooth=map_sigma_smooth)
            reconstructor.save_reconstructed_data(image_type=image_type)
            data, coord, params = reconstructor.create_TSDF_map()
        else:
            # Construct the 2D cost map
            tsdf_creator = TSDF_Creator(out_path, voxel_size=voxel_size, robot_size=robot_size,
                                        robot_height=robot_height, ground_z=ground_z,
                                        ground_height=terrain_ground_height,
                                        trajectory_clear_radius=trajectory_clear_radius,
                                        obstacle_reassign_threshold=obstacle_reassign_threshold,
                                        free_space_threshold=free_space_threshold)
            tsdf_creator.read_point_from_file("cloud.ply")
            data, coord, params = tsdf_creator.create_TSDF_map(sigma_smooth=map_sigma_smooth)
            if is_visualize:
                tsdf_creator.visualize_cloud(tsdf_creator.obs_pcd)
                tsdf_creator.visualize_cloud(tsdf_creator.free_pcd)

        # Save the esdf map
        tsdf_map = TSDF_Map()
        tsdf_map.DirectLoadMap(data, coord, params)
        tsdf_map.SaveTSDFMap(out_path, map_name)
        if save_visualizations:
            if is_direct_depth_raycast:
                viz_paths = reconstructor.save_map_visualizations(out_path, map_name)
            else:
                viz_paths = tsdf_creator.save_map_visualizations(out_path, map_name)
            if viz_paths is not None:
                heatmap_overview_items.append((env_name, viz_paths["heatmap"]))
                occupancy_overview_items.append((env_name, viz_paths["occupancy"]))
        if is_visualize:
            tsdf_map.ShowTSDFMap(cost_map=True)

    if save_visualizations:
        heatmap_overview_path = os.path.join(output_folder, visualization_prefix + "_heatmap_overview.png")
        occupancy_overview_path = os.path.join(output_folder, visualization_prefix + "_occupancy_overview.png")
        DataUtils.save_image_overview(heatmap_overview_items, heatmap_overview_path)
        DataUtils.save_image_overview(occupancy_overview_items, occupancy_overview_path)
        print("Map visualization overviews saved.")
