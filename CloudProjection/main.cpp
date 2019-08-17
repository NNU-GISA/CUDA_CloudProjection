#include"point_render.cuh"
#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>

int main()
{
	std::ifstream ifs("G:/data/vertices.txt");
	ifs.sync_with_stdio(0);
	std::vector<float> data;
	float tmp;

	std::cout << ifs.is_open() << std::endl;

	int cnt = 0;
	while (!ifs.eof())
	{
		ifs >> tmp;
		data.push_back(tmp);
		++cnt;
		//std::cout << tmp << std::endl;
	}

	cnt = cnt / 3;

	std::cout << " read [" << cnt  << "] points " << data.size()<< std::endl;

	float campose[] = { 0.738746 ,-0.338934 ,-0.582562 ,0.619214 ,0.000001 ,0.785223 ,
				  -0.266138, -0.940810 ,0.209873 ,-32.617535 ,26.210785 ,23.938950 };

	float angle = 0.691111/2;
	int tar_width = 1280;
	int tar_heigh = 720;

	float intrisic[9] = { 0 };

	intrisic[0] = (float)tar_width / 2 / atan(angle);
	intrisic[2] = (float)tar_width / 2;
	intrisic[4] = intrisic[0];
	intrisic[5] = (float)tar_heigh / 2;
	intrisic[8] = 1.0;

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			std::cout << intrisic[i * 3 + j] << " ";
		}
		std::cout << std::endl;
	}
		
	int ret_val;


	cudaArray* gpu_points;
	ret_val = cudaMalloc(&gpu_points, sizeof(float) * cnt *3);

	ret_val = cudaMemcpy(gpu_points, (const void*)(data.data()), sizeof(float) * cnt*3, cudaMemcpyHostToDevice);
	if (ret_val != cudaSuccess)
	{
		std::cout<< ("error on copy.\n");
		exit(-1);
	}


	cudaArray* gpu_intrinsic;
	ret_val = cudaMalloc(&gpu_intrinsic, sizeof(float) * 9);
	ret_val = cudaMemcpy(gpu_intrinsic, (const void*)(intrisic), sizeof(float) *9, cudaMemcpyHostToDevice);
	if (ret_val != cudaSuccess)
	{
		std::cout << ("error on copy.\n");
		exit(-1);
	}


	cudaArray* gpu_campose;
	ret_val = cudaMalloc(&gpu_campose, sizeof(float) * 12);
	ret_val = cudaMemcpy(gpu_campose, (const void*)(campose), sizeof(float) * 12, cudaMemcpyHostToDevice);
	if (ret_val != cudaSuccess)
	{
		std::cout << ("error on copy.\n");
		exit(-1);
	}

	int* gpu_mutex_map;
	ret_val = cudaMalloc(&gpu_mutex_map, sizeof(int) * tar_width* tar_heigh);


	float* gpu_depthmap;
	ret_val = cudaMalloc(&gpu_depthmap, sizeof(float) * tar_width * tar_heigh);

	int* out_index;
	ret_val = cudaMalloc(&out_index, sizeof( int) * tar_width * tar_heigh);




	GPU_DepthProject(gpu_points, cnt,
		gpu_intrinsic, gpu_campose,  tar_width,  tar_heigh,
		gpu_mutex_map,40,50, 2.5,
		gpu_depthmap, out_index, 0);

	float* cpu_depth = new float[tar_width * tar_heigh];
	ret_val = cudaMemcpy(cpu_depth, (const void*)(gpu_depthmap), sizeof(float) * tar_width * tar_heigh, cudaMemcpyDeviceToHost);
	if (ret_val != cudaSuccess)
	{
		std::cout << ("error on copy.\n");
		exit(-1);
	}

	std::ofstream ofs("G:/data/depth.txt");

	for (int i = 0; i < tar_heigh; ++i)
	{
		for (int j = 0; j < tar_width; ++j)
		{
			ofs << cpu_depth[i * tar_width + j] << " ";
		}

		ofs << std::endl;
	}
	ofs.close();

	delete[] cpu_depth;
	//------------------------------------------------------------
	//  Test Backward
	// ------------------------------------------------------
	int feature_dim = 128;

	float* grad_feature_image = new float[feature_dim * tar_width * tar_heigh];

	for (int i = 0; i < feature_dim * tar_width * tar_heigh; ++i)
		grad_feature_image[i] = 1.0;

	int num_points = cnt;

	cudaArray* gpu_grad_feature_image;
	cudaMalloc(&gpu_grad_feature_image, sizeof(float)* feature_dim* tar_width* tar_heigh);
	cudaMemcpy(gpu_grad_feature_image, grad_feature_image, sizeof(float)* feature_dim* tar_width* tar_heigh, cudaMemcpyHostToDevice);

	cudaArray* gpu_num_points;
	cudaMalloc(&gpu_num_points, sizeof(int));
	cudaMemcpy(gpu_num_points, &num_points, sizeof(int), cudaMemcpyHostToDevice);

	cudaArray*gpu_out_grad_feature_points;
	cudaMalloc(&gpu_out_grad_feature_points, sizeof(float)*feature_dim* num_points);

	cudaArray* gpu_out_grad_default_feature;
	cudaMalloc(&gpu_out_grad_default_feature, sizeof(float)* feature_dim);




	GPU_PCPR_backward(
		gpu_grad_feature_image, //(batch, dim, height, width)
		(cudaArray*)out_index,        //(batch, height, width)
		gpu_num_points,     // (batch)
		gpu_out_grad_feature_points, // (dim, total points)
		gpu_out_grad_default_feature, // (dim, 1)
		tar_heigh, tar_width,
		1, feature_dim, num_points
	);

	cudaMemcpy(grad_feature_image, gpu_out_grad_feature_points, sizeof(float)* feature_dim* num_points,cudaMemcpyDeviceToHost);

	std::ofstream oofs("G:/data/grad_feature.txt");

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < num_points; ++j)
		{
			oofs << grad_feature_image[i * num_points + j] << " ";
		}

		oofs << std::endl;
	}
	oofs.close();



	return 0;

}