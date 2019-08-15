#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/extension.h>

void GPU_PCPR(
	torch::Tensor in_points, //(num_points,3)
	torch::Tensor tar_intrinsic, torch::Tensor tar_Pose, 
	float near, float far, float max_splatting_size,
	torch::Tensor out_depth, torch::Tensor out_index); // (tar_heigh ,tar_width)