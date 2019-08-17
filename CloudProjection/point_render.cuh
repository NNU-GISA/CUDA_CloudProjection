#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/extension.h>

void GPU_PCPR(
	torch::Tensor in_points, //(num_points,3)
	torch::Tensor tar_intrinsic, torch::Tensor tar_Pose, 
	float near, float far, float max_splatting_size,
	torch::Tensor out_depth, torch::Tensor out_index); // (tar_heigh ,tar_width)

void GPU_PCPR_backward(
    torch::Tensor grad_feature_image, //(batch, dim, heigh, width)
    torch::Tensor index,        //(batch, heigh, width)
    torch::Tensor num_points,     // (batch)
    torch::Tensor out_grad_feature_points, // (dim, total points)
	torch::Tensor out_grad_default_feature, // (dim, 1)
    int total_num
    );