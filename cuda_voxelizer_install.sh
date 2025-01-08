cd thirdparty
# Install Trimesh
sudo apt update
sudo apt install mesa-common-dev -y
sudo apt install libgl1-mesa-dev libglu1-mesa-dev
sudo apt install -y libgl1-mesa-glx
sudo apt install libxi-dev
cd trimesh2; make
cd ..
# Install cuda_voxelizer
mkdir cuda_voxelizer/build
cmake .. -DTrimesh2_INCLUDE_DIR="../../trimesh2/include" -DTrimesh2_LINK_DIR="../../trimesh2/lib.Linux64"
cmake --build .
mv cuda_voxelizer ../cuda_voxelizer
