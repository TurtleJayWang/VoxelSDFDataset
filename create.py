import os
import shutil
from argparse import ArgumentParser
import json
import subprocess
import random
from tempfile import NamedTemporaryFile
import logging
import mesh_to_sdf as mts
from voxypy.models import Entity
import trimesh
import glob

import numpy as np

def generate_random_rotation_matrix():
    q = np.random.normal(0, 1, 4)
    q = q / np.linalg.norm(q)  # Normalize to get a unit quaternion
    
    w, x, y, z = q
    rotation_matrix = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y, 0],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x, 0],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y, 0],
        [0, 0, 0, 0]
    ])
    
    return rotation_matrix

def normalize_mesh(mesh, unit_sphere=True):
    """
    Normalize a mesh by centering it at origin and scaling to fit within a unit sphere or cube.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The input mesh to normalize
    unit_sphere : bool, optional
        If True, normalize to unit sphere. If False, normalize to unit cube
        Default is True
        
    Returns:
    --------
    trimesh.Trimesh
        Normalized copy of the input mesh
    """
    # Create a copy of the mesh to avoid modifying the original
    normalized = mesh.copy()
    
    # Center the mesh at origin
    center = normalized.vertices.mean(axis=0)
    normalized.vertices -= center
    
    # Scale the mesh
    if unit_sphere:
        # Scale to fit in unit sphere (radius = 1)
        scale = np.max(np.linalg.norm(normalized.vertices, axis=1))
    else:
        # Scale to fit in unit cube (max dimension = 1)
        scale = np.max(np.abs(normalized.vertices))
    
    if scale > 0:
        normalized.vertices /= scale
        
    return normalized

def create_dataset(huggingface_token, shapenet_categories, shapenet_download_dir, processed_data_dir,
                   num_augment_data=4,
                   num_sdf_samples=250000, 
                   cuda_voxelizer_path="thirdparty/cuda_voxelizer/cuda_voxelizer", input_voxel_grid_size=64,
                   test_validation_ratio=(0.8, 0.2)
):
    def download_category(category):
        if os.path.exists(os.path.join(shapenet_download_dir, category)):
            return
        cwd = os.getcwd()
        os.chdir(shapenet_download_dir)
        os.system(f'wget --header="Authorization: Bearer ${huggingface_token}" https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/{category}.zip')
        os.system("unzip -o '*.zip'")
        os.system("rm '*.zip'")
        os.chdir(cwd)
        
    def process_model(normalized_mesh, i):
        def voxelize_model_cuda_voxelizer(mesh_file_path):
            resolution = input_voxel_grid_size
            
            assert not os.path.exists(cuda_voxelizer_path), "Failed to find cuda voxelizer"

            subprocess.run(
                [cuda_voxelizer_path, "-f", mesh_file_path, "-s", str(resolution)], 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            entity = Entity().from_file(mesh_file_path + f"_{resolution}.vox")
            voxel_array = entity.get_dense()
            voxel_array = np.pad(voxel_array, pad_width=1, mode="constant", constant_values=0)
            os.remove(mesh_file_path + f"_{resolution}.vox")
            return voxel_array

        if i != 0:
            normalized_mesh.apply_scale(random.random() * 2)
            normalized_mesh.apply_transform(generate_random_rotation_matrix())

        tempf = NamedTemporaryFile(suffix=".obj")
        normalized_mesh.export(file_obj=tempf.name)

        logging.info("Sampling SDF...")
        points, sdfs = mts.sample_sdf_near_surface(normalized_mesh, number_of_points=num_sdf_samples)
        logging.info("Done")

        # Get the voxelized object
        logging.info("Voxelizing...")
        voxel_array = voxelize_model_cuda_voxelizer(tempf.name)
        logging.info("Done")
        logging.info("-" * 50)

        tempf.close()

        return points, sdfs, voxel_array
    
    def create_and_save_category_data(category):
        category_dir = os.path.join(shapenet_download_dir, category)
        data = []

        np_data_folder = os.path.join(processed_data_dir, "np_data")
        if not os.path.exists(np_data_folder):
            os.mkdir(np_data_folder)

        for model_file in glob.iglob(os.path.join(category_dir, "**/*.obj"), recursive=True):
            normalized_model = normalize_mesh(trimesh.load(model_file, force="mesh"))
            for i in range(num_augment_data):
                points, sdfs, voxel_grid = process_model(normalized_model, i)
                points = np.array(points)
                sdfs = np.array(sdfs)
                name = lambda type : f"{category}_{os.path.splitext(os.path.split(model_file)[-1])[0]}_{i}_{type}"
                np.save(name("points"), points)
                np.save(name("sdfs"), sdfs)
                np.save(name("voxel_grid"), voxel_grid)
                data.append({ 
                    "points" : name("points") + ".npy", 
                    "sdfs": name("sdfs") + ".npy", 
                    "voxel_grid" : name("voxel_grid") + f"_{input_voxel_grid_size}" + ".npy" 
                })
        
        n = len(data)
        split = [list(range(len(data)))]
        random.seed(42)
        random.shuffle(split)
        split_ratio = test_validation_ratio[0] / (test_validation_ratio[0] + test_validation_ratio[1])
        split = [
            split[0 : int(split_ratio * n)], 
            split[int(split_ratio * n) : n]
        ]

        return {
            "models" : data,
            "split" : split
        }

    for category in shapenet_categories:
        download_category(category)

    dataset_json = {}

    if os.path.exists(os.path.join(processed_data_dir, "dataset.json")):
        with open(os.path.join(processed_data_dir, "dataset.json"), "r") as f:
            dataset_json = json.load(f)

    for category in shapenet_categories:
        category_data = dataset_json.get(category, {})
        if len(category_data): continue

        category_data = create_and_save_category_data(category)
        dataset_json[category] = category_data

    with open(os.path.join(processed_data_dir, "dataset.json")) as f:
        json.dump(dataset_json, f)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-c", "--config")
    parser.add_argument("-t", "--token")
    args = parser.parse_args()

    config = {}
    with open(parser.config, "r") as f:
        config = json.load(f)
    
    create_dataset(
        args.token, config["shapenet_categories"], config["shapenet_download_dir"], config["processed_data_dir"], 
        config.get("num_augment_data", 1), config.get("num_sdf_samples", 250000), 
        config.get("cuda_voxelizer_path", "thirdparty/cuda_voxelizer/cuda_voxelizer"), config.get("input_voxel_grid_size", 64),
        config.get("test_validation_ratio", [0.8, 0.2])
    )