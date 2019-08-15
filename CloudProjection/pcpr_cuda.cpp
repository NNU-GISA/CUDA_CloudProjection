#include <torch/extension.h>
#include <vector>
#include "point_render.cuh"


// CUDA forward declarations

std::vector<torch::Tensor> pcpr_cuda_forward(
    torch::Tensor in_points, //(num_points,3)
    torch::Tensor tar_intrinsic, torch::Tensor tar_Pose,
    torch::Tensor out_depth, torch::Tensor out_index, // (tar_heigh ,tar_width)
    float near, float far, float max_splatting_size
    );

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) AT_ASSERTM(x.type().scalarType()==torch::ScalarType::Float, #x " must be a float tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

std::vector<torch::Tensor> pcpr_cuda_forward(
    torch::Tensor in_points, //(num_points,3)
    torch::Tensor tar_intrinsic, torch::Tensor tar_Pose,
    torch::Tensor out_depth, torch::Tensor out_index, // (tar_heigh ,tar_width)
    float near, float far, float max_splatting_size
    ) 
{
  CHECK_INPUT(in_points); CHECK_FLOAT(in_points);
  CHECK_INPUT(tar_intrinsic); CHECK_FLOAT(tar_intrinsic);
  CHECK_INPUT(tar_Pose); CHECK_FLOAT(tar_Pose);
  CHECK_INPUT(out_depth); CHECK_FLOAT(out_depth);
  CHECK_INPUT(out_index);

  AT_ASSERTM(out_depth.size(0)== out_index.size(0), "out_depth and out_index must be the same size");
  AT_ASSERTM(out_depth.size(1)== out_index.size(1), "out_depth and out_index must be the same size");

  GPU_PCPR(
	in_points, //(num_points,3)
	tar_intrinsic, tar_Pose, 
	near, far, max_splatting_size,
	out_depth, out_index);
 

  return {out_depth, out_index};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pcpr_cuda_forward, "PCPR forward (CUDA)");
}