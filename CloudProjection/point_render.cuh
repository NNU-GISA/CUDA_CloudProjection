#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void GPU_DepthProject(cudaArray* point_clouds, int num_points,
	cudaArray* tar_intrinsic, cudaArray* tar_Pose, int tar_width, int tar_height,
	int* mutex_map,float near, float far, float max_splatting_size,
	float* out_depth, int* out_index, cudaStream_t cuda_streams);


void GPU_PCPR_backward(
	cudaArray* grad_feature_image, //(batch, dim, height, width)
	cudaArray* index,        //(batch, height, width)
	cudaArray* num_points,     // (batch)
	cudaArray* out_grad_feature_points, // (dim, total points)
	cudaArray* out_grad_default_feature, // (dim, 1)
	int height, int width,
	int num_batch, int feature_dim, int total_num
);