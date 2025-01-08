import torch
from torch.utils.data import Dataset, DataLoader

import json
import os
import random

import numpy as np

class VoxelSDFDataset(Dataset):
    def __init__(self, dataset_dir, dataset_config_file, num_sdf_samples_per_item, is_train=True):
        self.dataset_dir = dataset_dir
        self.is_train = is_train
        self.num_sdf_samples_per_item = num_sdf_samples_per_item
    
        with open(dataset_config_file, "r") as f:
            self.config = json.load(f)

        self.models = []
        with open(os.path.join(dataset_dir, "dataset.json")) as f:
            self.data_json : dict = json.load(f)
        for category_info in self.data_json.values():
            category_models = category_info["models"]
            for i in category_info["split"][0 if is_train else 1]:
                self.models.append(category_models[i])

    def __getitem__(self, index):
        model_index, minor_batch_index = divmod(index, self.config["num_sdf_samples"] // self.num_sdf_samples_per_item)
        model = self.models[model_index]
        points, sdfs, voxel_grid = self.load_np_files(model)
        minor_batch_begin_index = self.num_sdf_samples_per_item * minor_batch_index
        minor_batch_end_index = (self.num_sdf_samples_per_item + 1) * minor_batch_index
        return voxel_grid, points[minor_batch_begin_index, minor_batch_end_index], sdfs[minor_batch_begin_index, minor_batch_end_index]

    def __len__(self):
        return len(self.models) * (self.config["num_sdf_samples"] // self.num_sdf_samples_per_item)

    def load_np_files(self, model_data_dict) -> tuple: # Returns are torch Tensors
        np_file = lambda name : os.path.join(self.dataset_dir, "np_data", name)
        points = np.load(np_file(model_data_dict["points"]))
        sdfs   = np.load(np_file(model_data_dict["sdfs"]))
        voxel_grid = np.load(np_file(model_data_dict["voxel_grid"]))
        return torch.from_numpy(points), torch.from_numpy(sdfs).unsqueeze(1), torch.from_numpy(voxel_grid).float()
    
def create_test_validation_data_loader(dataset_dir, batch_size, dataset_config_file, num_sdf_samples_per_item):
    train_dataset = VoxelSDFDataset(dataset_dir, dataset_config_file, num_sdf_samples_per_item, is_train=True)
    validation_dataset = VoxelSDFDataset(dataset_dir, dataset_config_file, num_sdf_samples_per_item, is_train=False)
    return DataLoader(train_dataset, batch_size, shuffle=True), DataLoader(validation_dataset, batch_size, shuffle=False)
