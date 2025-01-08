# VoxelSDFDataset
A Dataset and its utility scripts for Vox2SDF

## How to use
### 1. Preprocess the data
1. Create a codespace from this repository
2. Run ```bash cuda_voxelizer_install.sh```
3. Run ```python create.py --token <Your own huggingface token> --config config.json``` to create the dataset. Your can modify the config.json to change where the output is placed. (Make sure the account to which the token belongs have access to ShapeNet)