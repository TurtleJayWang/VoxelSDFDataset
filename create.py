import os
import shutil
from argparse import ArgumentParser
import json
import subprocess
import random
from tempfile import NamedTemporaryFile
import logging
from mesh_to_sdf import *
from midvoxio.voxio import vox_to_arr, viz_vox
from voxypy.models import Entity
import trimesh
import glob

import numpy as np
import requests
from tqdm import tqdm
import sys

import binvox_rw

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

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

def make_directories(path):
    fullpath = os.getcwd()
    for folder in os.path.split(path):
        fullpath = os.path.join(fullpath, folder)
        if not os.path.exists(fullpath):
            os.mkdir(fullpath)

def create_dataset(huggingface_token, shapenet_categories, shapenet_download_dir, processed_data_dir,
                   num_augment_data=4,
                   num_sdf_samples=250000, 
                   cuda_voxelizer_path="thirdparty/cuda_voxelizer/cuda_voxelizer", input_voxel_grid_size=64,
                   test_validation_ratio=(0.8, 0.2)
):
    def download_category(category):
        if os.path.exists(os.path.join(shapenet_download_dir, category)):
            return
        make_directories(shapenet_download_dir)

        url = f"https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/{category}.zip"
        headers = {"Authorization": f"Bearer {huggingface_token}"}
        response = requests.get(url, headers=headers, stream=True)
        
        if response.status_code == 200:
            logging.info(f"Downloading \"{url}\"...")
            with open(os.path.join(shapenet_download_dir, f"{category}.zip"), "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"Done")
        else:
            logging.error(f"Failed to download {category}.zip: {response.status_code}")
        
        for zip_file in glob.glob(os.path.join(shapenet_download_dir, "*.zip")):
            logging.info(f"Unzipping {category}...")
            shutil.unpack_archive(zip_file, extract_dir=shapenet_download_dir)
            os.remove(zip_file)
            logging.info("Done")
        
    def process_model(normalized_mesh, i):
        def voxelize_model_cuda_voxelizer(mesh_file_path):
            resolution = input_voxel_grid_size
            
            assert os.path.exists(cuda_voxelizer_path), "Failed to find cuda voxelizer"

            subprocess.run(
                [cuda_voxelizer_path, "-f", mesh_file_path, "-s", str(resolution), "-o", "binvox"], 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            voxel_array = np.zeros((resolution, resolution, resolution))
            with open(mesh_file_path + f"_{resolution}.binvox", "rb") as f:
                voxel_model = binvox_rw.read_as_3d_array(f)
                voxel_array = voxel_model.data
            os.remove(mesh_file_path + f"_{resolution}.binvox")
            return voxel_array

        if i != 0:
            normalized_mesh.apply_scale(random.random() * 2)
            normalized_mesh.apply_transform(generate_random_rotation_matrix())

        tempf = NamedTemporaryFile(suffix=".obj")
        normalized_mesh.export(file_obj=tempf.name)

        print("\tSampling SDF...")
        points, sdfs = sample_sdf_near_surface(normalized_mesh, number_of_points=num_sdf_samples)
        
        # Get the voxelized object
        print("\tVoxelizing...")
        voxel_array = voxelize_model_cuda_voxelizer(tempf.name)
        
        tempf.close()

        return points, sdfs, voxel_array
    
    def create_and_save_category_data(category):
        category_dir = os.path.join(shapenet_download_dir, category)
        data = []

        np_data_folder = os.path.join(processed_data_dir, "np_data")
        make_directories(np_data_folder)
    
        for i, model_file in enumerate(glob.iglob(os.path.join(category_dir, "**/*.obj"), recursive=True)):
            normalized_model = 0
            print("-" * 100)
            print(f"Processing {model_file}, the {i}th model in the category")
            for i in range(num_augment_data):
                print("-" * 100)
                print(f"Augment {i} (In total of {num_augment_data})")

                name = lambda type : f"{category}_{splitall(model_file)[-3]}_{i}_{type}"

                if os.path.exists(os.path.join(np_data_folder, name("fulldata") + ".npz")):
                    fulldata = np.load(os.path.join(np_data_folder, name("fulldata") + ".npz"))
                    if fulldata["voxel_grid"].shape == (64, 64, 64) \
                        and fulldata["points"].shape == (num_sdf_samples, 3) \
                        and fulldata["sdfs"].size == num_sdf_samples:
                        data.append(name("fulldata") + ".npz")
                        print(f"Processed data exists, skip (In {os.path.join(np_data_folder, name("fulldata") + ".npz")})")
                        continue

                if normalized_model == 0:
                    normalized_model = normalize_mesh(trimesh.load(model_file, force="mesh"))

                points, sdfs, voxel_grid = process_model(normalized_model, i)
                points = np.array(points)
                sdfs = np.array(sdfs)

                print("\tSaving data...")
                np.savez_compressed(
                    os.path.join(np_data_folder, name("fulldata")), 
                    points=points, sdfs=sdfs, voxel_grid=voxel_grid
                )
                print(f"\tData has been written to {os.path.join(np_data_folder, name("fulldata"))}.npz")

                data.append(name("fulldata") + ".npz")
                
                print("Done", flush=True)
        
        n = len(data)
        split = list(range(n))
        random.seed(20)
        random.shuffle(split)
        split_ratio = test_validation_ratio[0] / (test_validation_ratio[0] + test_validation_ratio[1])
        split = [
            split[0 : int(split_ratio * n)], 
            split[int(split_ratio * n):]
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

        category_data = create_and_save_category_data(category)
        dataset_json[category] = category_data

    with open(os.path.join(processed_data_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-c", "--config")
    parser.add_argument("-t", "--token")
    args = parser.parse_args()

    config = {}
    with open(args.config, "r") as f:
        config = json.load(f)

    create_dataset(
        args.token, config["shapenet_categories"], config["shapenet_download_dir"], config["processed_data_dir"], 
        config.get("num_augment_data", 1), config.get("num_sdf_samples", 250000), 
        config.get("cuda_voxelizer_path", "thirdparty/cuda_voxelizer/cuda_voxelizer"), config.get("input_voxel_grid_size", 64),
        config.get("test_validation_ratio", [0.8, 0.2])
    )