/*
  Image Histogram Equalization with GPU & Shared Memory
  Víctor Rendón Suárez
  A01022462
*/
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

__global__ void make_histogram(unsigned char* input, unsigned int *histogram, int cols, int rows)
{
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Only valid threads perform memory I/O
	if ((xIndex < cols) && (yIndex < rows)){
		const int tid = yIndex * cols + xIndex;
		atomicAdd(&histogram[(int)input[tid]], 1);
	}
}

__global__ void normalize_histogram(unsigned int *histogram, unsigned int *norm_histogram, float n_constant)
{
	const int idx = threadIdx.x;
	for (int i = 0; i < idx; i++)
		norm_histogram[idx] += histogram[i];

	norm_histogram[idx] *= n_constant;
}

__global__ void make_output(unsigned char* input, unsigned char* output, unsigned int *histogram, int cols, int rows)
{
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Only valid threads perform memory I/O
	if ((xIndex < cols) && (yIndex < rows)){
		const int tid = yIndex * cols + xIndex;
		output[tid] = static_cast<unsigned char>(histogram[(int)input[tid]]);
	}
}

void equalize_image(const cv::Mat& input, cv::Mat& output)
{
	// Calculate total number of bytes of input and output image
	size_t bytes = input.step * input.rows;
	size_t histogram_bytes = sizeof(int)*256;
	unsigned char *d_input, *d_output;
	unsigned int *histogram, *norm_histogram;

	// Allocate device memory
	cudaMalloc<unsigned char>(&d_input, bytes);
	cudaMalloc<unsigned char>(&d_output, bytes);
	cudaMalloc<unsigned int>(&histogram, histogram_bytes);
	cudaMalloc<unsigned int>(&norm_histogram, histogram_bytes);

	// Copy data from OpenCV input image to device memory
	cudaMemcpy(d_input, input.ptr(), bytes, cudaMemcpyHostToDevice);

	// Specify a reasonable block size for image manipulation
	const dim3 block(32, 32);
	// Calculate grid size to cover the whole image
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));

	// Histogram equalization
	auto start_time = std::chrono::high_resolution_clock::now();
	make_histogram<<<grid, block>>>(d_input, histogram, input.cols, input.rows);
	normalize_histogram<<<1, block>>>(histogram, norm_histogram, 255 / (float)(input.cols * input.rows));
	make_output<<<grid, block>>>(d_input, d_output, norm_histogram, input.cols, input.rows);
	auto end_time = std::chrono::high_resolution_clock::now();
	chrono::duration<float, milli> duration_ms = end_time - start_time;
	printf("Histogram equalization w/GPU, time elapsed: %f ms\n", duration_ms.count());

	// Synchronize to check for any kernel launch errors
	cudaDeviceSynchronize();
	// Copy back data from destination device meory to OpenCV output image
	cudaMemcpy(output.ptr(), d_output, bytes, cudaMemcpyDeviceToHost);

	// Free the device memory
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(histogram);
	cudaFree(norm_histogram);
}

int main(int argc, char *argv[])
{
	string input_image;
	// Use default image if none specified in command
	if (argc < 2)
		input_image = "Images/woman3.jpg";
	else
		input_image = argv[1];

	cv::Mat input = cv::imread(input_image, CV_LOAD_IMAGE_GRAYSCALE);
	if (input.empty()) {
		cout << "Error: Specified image not found." << std::endl;
		return -1;
	}

	// Output image
	cv::Mat output(input.rows, input.cols, input.type());

	// Call wrapper function
	equalize_image(input, output);

	// Resize and show images
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);
	cv::resizeWindow("Input", 1000, 600);
	cv::resizeWindow("Output", 1000, 600);
	imshow("Input", input);
	imshow("Output", output);

	// Close with key press
	cv::waitKey();

	return 0;
}
