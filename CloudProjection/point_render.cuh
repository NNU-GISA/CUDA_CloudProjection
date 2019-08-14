#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void GPU_DepthProject(cudaArray* point_clouds, int num_points,
	cudaArray* tar_intrinsic, cudaArray* tar_Pose, int tar_width, int tar_height,
	int* mutex_map,float near, float far, float max_splatting_size,
	float* out_depth, unsigned int* out_index, cudaStream_t cuda_streams);