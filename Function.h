#include<opencv2\core\core.hpp>
#include<iostream>
using namespace std;

namespace Elysia
{

	cv::Mat sigmoid(cv::Mat& x);

	cv::Mat tanh(cv::Mat& x);

	cv::Mat ReLU(cv::Mat& x);

	cv::Mat derivativeFunction(cv::Mat& fx, string func_type);

	void calcLoss(cv::Mat& output, cv::Mat& target, cv::Mat& output_error, float& loss);
}