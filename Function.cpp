#include "Function.h"

using namespace std;

namespace Elysia
{
	cv::Mat sigmoid(cv::Mat& x)
	{
		cv::Mat exp_x, fx;
		cv::exp(-x, exp_x);
		fx = 1.0 / (1.0 + exp_x);
		return fx;
	}

	cv::Mat tanh(cv::Mat& x)
	{
		cv::Mat exp_x_, exp_x, fx;
		cv::exp(-x, exp_x_);
		cv::exp(x, exp_x);
		fx = (exp_x - exp_x_) / (exp_x + exp_x_);
		return fx;
	}

	cv::Mat ReLU(cv::Mat& x)
	{
		cv::Mat fx = x;
		for (int i = 0; i < fx.rows; i++)
		{
			for (int j = 0; j < fx.cols; j++)
			{
				if (fx.at<float>(i, j) < 0)
				{
					fx.at<float>(i, j) = 0;
				}
			}
		}
		return fx;
	}

	cv::Mat derivativeFunction(cv::Mat& fx, string func_type) //计算相应激活函数的导数
	{
		cv::Mat dx;
		if (func_type == "sigmoid")
		{
			dx = sigmoid(fx).mul((1 - sigmoid(fx)));
		}
		if (func_type == "tanh")
		{
			cv::Mat tanh_2;
			pow(tanh(fx), 2., tanh_2);
			dx = 1 - tanh_2;
		}
		if (func_type == "ReLU")
		{
			dx = fx;
			for (int i = 0; i < fx.rows; i++)
			{
				for (int j = 0; j < fx.cols; j++)
				{
					if (fx.at<float>(i, j) > 0)
					{
						dx.at<float>(i, j) = 1;
					}
				}
			}
		}
		return dx;
	}

	void calcLoss(cv::Mat& output, cv::Mat& target, cv::Mat& output_error, float& loss) //损失计算 均方误差MSE
	{
		if (target.empty())
		{
			cout << "Can't find the target cv::Matrix" << endl;
			return;
		}
		output_error = target - output;
		cv::Mat err_sqrare;
		pow(output_error, 2., err_sqrare);
		cv::Scalar err_sqr_sum = sum(err_sqrare);
		loss = err_sqr_sum[0] / (float)(output.rows); //误差矩阵和/样本数
	}
}