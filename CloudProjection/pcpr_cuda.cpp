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

std::vector<torch::Tensor> pcpr_cuda_backward(
    torch::Tensor grad_feature_image, //(batch, dim, heigh, width)
    torch::Tensor index,        //(batch, heigh, width)
    torch::Tensor num_points,     // (batch) - GPU
    torch::Tensor out_grad_feature_points, // (dim, total points)
    torch::Tensor out_grad_default_feature, // (dim, 1)
    int total_num
    );

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) AT_ASSERTM(x.type().scalarType()==torch::ScalarType::Float, #x " must be a float tensor")
#define CHECK_Int(x) AT_ASSERTM(x.type().scalarType()==torch::ScalarType::Int, #x " must be a Int tensor")
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
  CHECK_INPUT(out_index); CHECK_Int(out_depth);

  AT_ASSERTM(out_depth.size(0)== out_index.size(0), "out_depth and out_index must be the same size");
  AT_ASSERTM(out_depth.size(1)== out_index.size(1), "out_depth and out_index must be the same size");

  GPU_PCPR(
	in_points, //(num_points,3)
	tar_intrinsic, tar_Pose, 
	near, far, max_splatting_size,
	out_depth, out_index);
 

  return {out_depth, out_index};
}


std::vector<torch::Tensor> pcpr_cuda_backward(
    torch::Tensor grad_feature_image, //(batch, dim, heigh, width)
    torch::Tensor index,        //(batch, heigh, width)
    torch::Tensor num_points,     // (batch) - GPU
    torch::Tensor out_grad_feature_points, // (dim, total points)
    torch::Tensor out_grad_default_feature, // (dim, 1)
    int total_num
    )
{
  CHECK_INPUT(grad_feature_image); CHECK_FLOAT(grad_feature_image);
  CHECK_INPUT(index); CHECK_Int(index);
  CHECK_INPUT(num_points); CHECK_Int(num_points);
  CHECK_INPUT(out_grad_feature_points); CHECK_FLOAT(out_grad_feature_points);
  CHECK_INPUT(out_grad_default_feature); CHECK_FLOAT(out_grad_default_feature);

  AT_ASSERTM(grad_feature_image.size(0)== index.size(0), "grad_feature_image and index must be the same batch size");
  AT_ASSERTM(index.size(0)== num_points.size(0), "grad_feature_image and num_points must be the same batch size");

  GPU_PCPR_backward(
    grad_feature_image, //(batch, dim, heigh, width)
    index,        //(batch, heigh, width)
    num_points,     // (batch)
    out_grad_feature_points, // (dim, total points)
	  out_grad_default_feature, // (dim, 1)
    total_num
    );


  return {out_grad_feature_points};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pcpr_cuda_forward, "PCPR forward (CUDA)");
  m.def("backward", &pcpr_cuda_backward, "PCPR backward (CUDA)");
}

