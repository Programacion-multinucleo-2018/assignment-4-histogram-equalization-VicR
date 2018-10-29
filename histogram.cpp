/*
  Image Histogram Equalization with CPU
  Víctor Rendón Suárez
  A01022462
*/
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;

void equalize_image(const cv::Mat& input, cv::Mat& output)
{
	unsigned int histogram[256] = {};
	int histogram_s[256] = {};
	float size = input.rows * input.cols;
	int actual;

	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			actual = (int)input.at<uchar>(y,x);
			histogram[actual]++;
		}
	}
	for(int y = 0; y < 256;y++) {
		for(int x = 0; x <= y; x++) {
			histogram_s[y] += histogram[x];
		}
	}
	for(int y = 0; y < 256;y++) {
		histogram_s[y] = histogram_s[y]*(255/size);
	}
	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			actual = (int)input.at<uchar>(y,x);
			output.at<uchar>(y,x) = histogram_s[actual];
		}
	}
}

int main(int argc, char *argv[])
{
	string input_image;
	// Use default image if none specified in command
	if (argc < 2)
		input_image = "Images/woman2.jpg";
	else
		input_image = argv[1];

	cv::Mat input = cv::imread(input_image, CV_LOAD_IMAGE_GRAYSCALE);
	if (input.empty()) {
		cout << "Error: Specified image not found." << std::endl;
		return -1;
	}

	// Output image
	cv::Mat output(input.rows, input.cols, input.type());

	// Histogram equalization, measure time elapsed
	auto start_time = std::chrono::high_resolution_clock::now();
	equalize_image(input, output);
	auto end_time = std::chrono::high_resolution_clock::now();
	chrono::duration<float, milli> duration_ms = end_time - start_time;
	printf("Histogram equalization w/CPU, time elapsed: %f ms\n", duration_ms.count());

	// Resize and show images
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);
	cv::resizeWindow("Input", 1000, 600);
	cv::resizeWindow("Output", 1000, 600);
	imshow("Input", input);
	imshow("Output", output);
	// Close with keypress
	cv::waitKey();

	return 0;
}
