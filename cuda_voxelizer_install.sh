cd thirdparty
cd trimesh2; make
cd ..
# Install cuda_voxelizer
mkdir cuda_voxelizer/build
cd cuda_voxelizer/build
cmake .. -DTrimesh2_INCLUDE_DIR="../../trimesh2/include" -DTrimesh2_LINK_DIR="../../trimesh2/lib.Linux64"
cmake --build .
mv cuda_voxelizer ../cuda_voxelizer
