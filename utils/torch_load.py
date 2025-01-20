import torch
from torch.utils.data import Dataset, DataLoader

import json
import os
import random

import numpy as np

class VoxelSDFDataset(Dataset):
    def __init__(self, dataset_dir, dataset_config_file, num_sdf_samples_per_minor_batch, used_categories="All", is_train=True):
        self.dataset_dir = dataset_dir
        self.is_train = is_train
        self.num_sdf_samples_per_minor_batch = num_sdf_samples_per_minor_batch
    
        with open(dataset_config_file, "r") as f:
            self.config = json.load(f)

        self.models = []
        with open(os.path.join(dataset_dir, "dataset.json")) as f:
            self.data_json : dict = json.load(f)

        categories = []
        if used_categories == "All":
            categories = list(self.data_json.values())
        elif isinstance(categories, list):
            categories = used_categories
        
        for category_info in categories:
            category_models = category_info["models"]
            for i in category_info["split"][0 if is_train else 1]:
                self.models.append(category_models[i])

    def __getitem__(self, index):
        model_index, minor_batch_index = divmod(index, self.config["num_sdf_samples"] // self.num_sdf_samples_per_minor_batch)
        model_npz_filename = self.models[model_index]
        points, sdfs, voxel_grid = self.load_np_files(model_npz_filename)
        minor_batch_begin_index = self.num_sdf_samples_per_minor_batch * minor_batch_index
        minor_batch_end_index = self.num_sdf_samples_per_minor_batch * (minor_batch_index + 1)
        return voxel_grid, points[minor_batch_begin_index:minor_batch_end_index], sdfs[minor_batch_begin_index:minor_batch_end_index]

    def __len__(self):
        return len(self.models) * (self.config["num_sdf_samples"] // self.num_sdf_samples_per_minor_batch)

    def load_np_files(self, model_npz_filename) -> tuple: # Returns are torch Tensors
        np_file = os.path.join(self.dataset_dir, model_npz_filename)
        model_fulldata = np.load(np_file)
        points = model_fulldata["points"]
        sdfs = model_fulldata["sdfs"]
        voxel_grid = model_fulldata["voxel_grid"]
        return torch.from_numpy(points), torch.from_numpy(sdfs).unsqueeze(1), torch.from_numpy(voxel_grid).float()
    
def create_test_validation_data_loader(dataset_dir, batch_size, dataset_config_file, num_sdf_samples_per_minor_batch):
    train_dataset = VoxelSDFDataset(dataset_dir, dataset_config_file, num_sdf_samples_per_minor_batch, is_train=True)
    validation_dataset = VoxelSDFDataset(dataset_dir, dataset_config_file, num_sdf_samples_per_minor_batch, is_train=False)
    return DataLoader(train_dataset, batch_size, shuffle=True), DataLoader(validation_dataset, batch_size, shuffle=False)
