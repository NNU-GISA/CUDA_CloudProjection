#include "point_render.cuh"
#include <stdio.h>

#include "helper_math.h"


struct Matrix4x4
{
public:
	float4 col[4];
	__device__
		Matrix4x4()
	{
		col[0] = col[1] = col[2] = col[3] = make_float4(0, 0, 0, 0);
	}
	__device__
		Matrix4x4(float3 a, float3 b, float3 c, float3 d)
	{
		col[0].x = a.x;
		col[0].y = a.y;
		col[0].z = a.z;
		col[0].w = 0;

		col[1].x = b.x;
		col[1].y = b.y;
		col[1].z = b.z;
		col[1].w = 0;

		col[2].x = c.x;
		col[2].y = c.y;
		col[2].z = c.z;
		col[2].w = 0;

		col[3].x = d.x;
		col[3].y = d.y;
		col[3].z = d.z;
		col[3].w = 1;
	}

	__device__
		Matrix4x4 transpose() const
	{
		Matrix4x4 res;

		res.col[0].x = col[0].x;
		res.col[0].y = col[1].x;
		res.col[0].z = col[2].x;
		res.col[0].w = col[3].x;

		res.col[1].x = col[0].y;
		res.col[1].y = col[1].y;
		res.col[1].z = col[2].y;
		res.col[1].w = col[3].y;

		res.col[2].x = col[0].z;
		res.col[2].y = col[1].z;
		res.col[2].z = col[2].z;
		res.col[2].w = col[3].z;

		res.col[3].x = 0;
		res.col[3].y = 0;
		res.col[3].z = 0;
		res.col[3].w = 1;
		return res;

	}
	__device__
		Matrix4x4 inv() const
	{
		Matrix4x4 res;
		res.col[0].x = col[0].x;
		res.col[0].y = col[1].x;
		res.col[0].z = col[2].x;
		res.col[0].w = 0;

		res.col[1].x = col[0].y;
		res.col[1].y = col[1].y;
		res.col[1].z = col[2].y;
		res.col[1].w = 0;

		res.col[2].x = col[0].z;
		res.col[2].y = col[1].z;
		res.col[2].z = col[2].z;
		res.col[2].w = 0;

		res.col[3].x = -dot(col[0], col[3]);
		res.col[3].y = -dot(col[1], col[3]);
		res.col[3].z = -dot(col[2], col[3]);
		res.col[3].w = 1;
		return res;
	}

	__device__
		static	Matrix4x4 RotateX(float rad)
	{
		Matrix4x4 res;
		res.col[0].x = 1;
		res.col[0].y = 0;
		res.col[0].z = 0;
		res.col[0].w = 0;

		res.col[1].x = 0;
		res.col[1].y = cos(rad);
		res.col[1].z = sin(rad);
		res.col[1].w = 0;

		res.col[2].x = 0;
		res.col[2].y = -sin(rad);
		res.col[2].z = cos(rad);
		res.col[2].w = 0;

		res.col[3].x = 0;
		res.col[3].y = 0;
		res.col[3].z = 0;
		res.col[3].w = 1;
		return res;
	}
};



typedef struct CamPoseNode
{
	float3 norm, Xaxis, Yaxis, offset;
	__device__
		Matrix4x4 getRT() const
	{
		return Matrix4x4(Xaxis, Yaxis, norm, offset);
	}

}CamPose;



typedef struct CamIntrinsic
{
	float3 r[3];

	__device__
		Matrix4x4 getMatrix(float scale = 1.0) const
	{
		Matrix4x4 res;
		res.col[0].x = r[0].x * scale;
		res.col[0].y = r[1].x * scale;
		res.col[0].z = r[2].x * scale;
		res.col[0].w = 0;

		res.col[1].x = r[0].y * scale;
		res.col[1].y = r[1].y * scale;
		res.col[1].z = r[2].y * scale;
		res.col[1].w = 0;

		res.col[2].x = r[0].z * scale;
		res.col[2].y = r[1].z * scale;
		res.col[2].z = r[2].z;
		res.col[2].w = 0;

		res.col[3].x = 0;
		res.col[3].y = 0;
		res.col[3].z = 0;
		res.col[3].w = 1;
		return res;
	}
	__device__
		float4 PointInverse(float x, float y, float scale = 1.0)
	{
		float xx = (x - r[0].z * scale) / (r[0].x * scale);
		float yy = (y - r[1].z * scale) / (r[1].y * scale);
		return make_float4(xx, yy, 1, 1);
	}

};


namespace math
{
	__device__
		float4 MatrixMul(const Matrix4x4& mat, float4& x)
	{
		Matrix4x4 res = mat.transpose();
		float4 ans;
		ans.x = dot(res.col[0], x);
		ans.y = dot(res.col[1], x);
		ans.z = dot(res.col[2], x);
		ans.w = dot(res.col[3], x);

		ans = ans / ans.w;
		return ans;
	}
}


__global__
void DepthProject(float3 * point_clouds, int num_points,
	CamIntrinsic* tar_intrinsic, CamPose* tar_Pose, int tar_width, int tar_heigh,
	int * mutex_map, float near, float far, float max_splatting_size,
	float* out_depth, unsigned int* out_index)
{
	int ids = blockDim.x * blockIdx.x + threadIdx.x; //  index of point


	if (ids > num_points) 
		return;


	// Cache camera parameters
	 CamPose _tarcamPose = *tar_Pose;
	 CamIntrinsic _tarcamIntrinsic = *tar_intrinsic;


	float4 p = make_float4(point_clouds[ids], 1.0);

	Matrix4x4 camT = _tarcamPose.getRT();
	camT = camT.inv();
	float4 camp = math::MatrixMul(camT, p);



	float tdepth = -camp.z;

	if (tdepth < 0)
		return;
	camp = math::MatrixMul(_tarcamIntrinsic.getMatrix(), camp);

	camp = camp / camp.w;
	camp = camp / camp.z;



	// splatting radius

	float rate = (tdepth - near) / (far - near);
	rate = 1.0 - rate;
	rate = max(rate, 0.0);
	rate = min(rate, 1.0);
	

	float radius = max_splatting_size * rate;

	// splatting
	for (int xx = round(camp.x - radius); xx <= round(camp.x + radius); ++xx)
	{
		for (int yy = round(camp.y - radius); yy <= round(camp.y + radius); ++yy)
		{
			if (xx < 0 || xx >= tar_width || yy < 0 || yy >= tar_heigh)
				return;

			int ind = yy * tar_width + xx ;

			if (out_depth[ind] > 0 && out_depth[ind] <= tdepth)
				continue;

			bool isSet = false;
			do
			{
				if ((isSet = atomicCAS(mutex_map + ind, 0, 1)) == false)
				{
					// critical section goes here
					if (out_depth[ind] > tdepth || out_depth[ind]==0)
					{
						out_depth[ind] = tdepth;
						out_index[ind] = ids + 1; // 0 denote empty
					}
				}
				if (isSet)
				{
					mutex_map[ind] = 0;
				}
			} while (!isSet);

		}
	}

}

void GPU_DepthProject(cudaArray * point_clouds, int num_points,
	cudaArray* tar_intrinsic, cudaArray* tar_Pose, int tar_width, int tar_heigh,
	int* mutex_map, float near, float far, float max_splatting_size,
	float* out_depth, unsigned int* out_index, cudaStream_t cuda_streams)
{
	dim3 dimBlock(256,1);
	dim3 dimGrid(num_points / dimBlock.x + 1, 1);

	cudaMemsetAsync(out_depth, 0, tar_width * tar_heigh * sizeof(float), cuda_streams);
	cudaMemsetAsync(out_index, 0, tar_width * tar_heigh * sizeof(unsigned int), cuda_streams);

	DepthProject << <dimGrid, dimBlock, 0, cuda_streams >> > ((float3*)point_clouds, num_points,
		(CamIntrinsic*)tar_intrinsic, (CamPose*)tar_Pose, tar_width, tar_heigh,
		mutex_map, near, far, max_splatting_size,
		out_depth, out_index );

}

