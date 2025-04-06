#pragma once
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <iomanip>
#include "Function.h"

using namespace std;

namespace Elysia{
	class Net
	{
	public:
		vector<int> layer_neuron_num; //存储每层神经元个数
		string activation_function = "sigmoid";
		int output_interval = 10; //输出间隔
		float learning_rate;
		float accuracy = 0.;
		vector<double> loss_vec; //存储loss
		float fine_tune_factor = 1.01; //微调因子

	protected:
		vector<cv::Mat> layer; //存储每层神经元权值
		vector<cv::Mat> weights;
		vector<cv::Mat> bias;
		vector<cv::Mat> delta_err; //存储每层误差

		cv::Mat output_error; //存储输出误差
		cv::Mat target;
		cv::Mat board; //绘制损失曲线
		float loss;

	public:
		Net() {};
		~Net() {};

		void initNet(vector<int> layer_neuron_num_);
		void initWeights(int type = 0, double a = 0, double b = 0.1);
		void initBias(cv::Scalar& bias);
		void forward();
		void backward();

		void train(cv::Mat input, cv::Mat target, float accuracy_threshold); //基于准确率阈值
		void train(cv::Mat input, cv::Mat target_, float loss_threshold, bool draw_loss_curve = false); //基于损失阈值，可绘制损失曲线
		void test(cv::Mat& input, cv::Mat& target_);
		int predict_one(cv::Mat& input);
		vector<int> predict(cv::Mat& input);

		void save(string filename);
		void load(string filename);

	protected:
		void initWeight(cv::Mat& dst, int type, double a, double b);
		cv::Mat activationFunction(cv::Mat& x, string func_type);
		void deltaError(); //计算误差增量
		void updateWeights(); //更新权重

	};

	void get_input_label(string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start = 0); //获取输入和标签

	void draw_curve(cv::Mat& board, vector<double> points); //绘制损失曲线

}

