#include "Net.h"
#include <opencv2/imgproc.hpp>

namespace Elysia 
{
	//初始化神经网络
	void Net::initNet(vector<int> layer_neuron_num_)
	{
		layer_neuron_num = layer_neuron_num_;
		//生成每层的矩阵
		layer.resize(layer_neuron_num.size()); //调整 layer 向量的大小，使其与层数相匹配
		for (int i = 0;i < layer.size();++i)
		{
			layer[i].create(layer_neuron_num[i], 1, CV_32FC1); //行数为该层神经元数量，列数为1，数据类型为 CV_32FC1（单通道32位浮点）
		}
		cout << "成功生成网络" << endl;

		//生成权重和偏置矩阵
		weights.resize(layer.size() - 1);
		bias.resize(layer.size() - 1);
		for (int i = 0; i < (layer.size() - 1); ++i)
		{
			weights[i].create(layer[i + 1].rows, layer[i].rows, CV_32FC1); //行数为下一层的神经元数量，列数为当前层的神经元数量
			bias[i] = cv::Mat::zeros(layer[i + 1].rows, 1, CV_32FC1); 
		}
		cout << "成功生成权重矩阵和偏差" << endl;
		cout << "初始化网络已完成" << endl;
	}

	//初始化权重
	void Net::initWeight(cv::Mat& dst, int type, double a, double b)
	{
		if(type == 0) randn(dst, a, b); //高斯分布
		else randu(dst, a, b); //均匀分布
	}
	void Net::initWeights(int type, double a, double b)
	{
		for (int i = 0; i < weights.size(); ++i)
		{
			initWeight(weights[i], type, a, b);
		}
	}

	//初始化偏置
	void Net::initBias(cv::Scalar& bias_)
    {
        for (int i = 0; i < bias.size(); ++i)
        {
            bias[i] = bias_;
        }
    }

	//前向传播
	void Net::forward()
    {
		for (int i = 0;i < layer_neuron_num.size() - 1;++i)
		{
			cv::Mat product = weights[i] * layer[i] + bias[i];
			layer[i + 1] = activationFunction(product,activation_function);
		}
		calcLoss(layer[layer.size() - 1], target, output_error, loss);
    }

	//激活函数
	cv::Mat Net::activationFunction(cv::Mat& x, string func_type)
	{
		activation_function = func_type;
		cv::Mat fx;
		if (func_type == "sigmoid")
        {
			fx = sigmoid(x);
        }
		if (func_type == "tanh")
        {
			fx = tanh(x);
        }
        if (func_type == "ReLU")
        {
			fx = ReLU(x);
        }
		return fx;
	}

	//反向传播
	void Net::backward()
    {
		//calcLoss(layer[layer.size() - 1], target, output_error, loss); //计算输出误差和目标函数
		deltaError(); //计算delta误差
		updateWeights(); 
    }

	//更新权重
	void Net::updateWeights()
    {
		for (int i = 0;i < weights.size();++i)
		{
			cv::Mat delta_weights = learning_rate * (delta_err[i] * layer[i].t());
			cv::Mat delta_bias = learning_rate * delta_err[i];
			weights[i] = weights[i] + delta_weights;
			bias[i] = bias[i] + delta_bias;
		}
    }

	//计算delta误差
	void Net::deltaError()
    {
		delta_err.resize(layer.size()-1);
		for (int i = delta_err.size() - 1;i >= 0;--i) //从输出层开始，逐层向前计算误差增量
		{
			delta_err[i].create(layer[i + 1].size(), layer[i + 1].type()); //大小和类型与当前层的输出矩阵 layer[i + 1] 相同
			cv::Mat dx = derivativeFunction(layer[i + 1], activation_function); //计算当前层输出 layer[i + 1] 关于激活函数的导数
			if (i == delta_err.size() - 1)
			{
				//误差增量等于激活函数的导数 dx 与输出误差 output_error 的逐元素相乘
				delta_err[i] = dx.mul(output_error);
			}
			else
			{
				//下一层的权重矩阵 weights[i + 1] 的转置与下一层的误差增量 delta_err[i + 1] 相乘
				//将结果与当前层激活函数的导数 dx 逐元素相乘，得到当前层的误差增量
				cv::Mat weight = weights[i+1];
				cv::Mat weight_t = weight.t();
				cv::Mat delta_err_1 = delta_err[i+1];
				delta_err[i] = dx.mul(weight_t * delta_err_1);
			}
		}
    }

	//训练 基于损失阈值
	void Net::train(cv::Mat input, cv::Mat target_, float loss_threshold, bool draw_loss_curve)
	{
		if (input.empty())
		{
			cout << "输出为空" << endl;
			return;
		}
		cout << "开始训练" << endl;

		cv::Mat sample;
		if (input.rows == (layer[0].rows) && input.cols == 1) //单个样本输入（input列数为1）
		{
			target = target_;
			sample = input;
			layer[0] = sample;
			forward();
			//backward();
			int num_of_train = 0;
			while (loss > loss_threshold)
			{
				backward();
				forward();
				num_of_train++;
				if (num_of_train % 500 == 0)
				{
					cout<<"训练 "<<num_of_train<<" 次"<<endl;
					cout<<"Loss:"<<loss<<endl;
				}
			}
			cout << endl << "训练 " << num_of_train << " 次" << endl;
			cout << "Loss:" << loss << endl;
			cout<<"训练成功"<<endl;	
		}
		else if (input.rows == (layer[0].rows) && input.cols > 1)
		{
			double batch_loss = loss_threshold + 0.01;
			int epoch = 0;
			while (batch_loss > loss_threshold)
			{
				batch_loss = 0.;
				for (int i = 0; i < input.cols; ++i)
				{
					target = target_.col(i);
					sample = input.col(i);
					layer[0] = sample;

					forward();
					backward();

					batch_loss += loss;
				}

				loss_vec.push_back(batch_loss); //记录每轮的损失情况

				if (loss_vec.size() >= 2 && draw_loss_curve)
				{
					draw_curve(board, loss_vec);
				}
				epoch++;
				if (epoch % output_interval == 0)
				{
					cout << "Number of epoch: " << epoch << endl;
					cout << "Loss sum: " << batch_loss << endl;
				}
				if (epoch % 100 == 0)
				{
					learning_rate *= fine_tune_factor;
				}
			}
			cout << endl << "Number of epoch: " << epoch << endl;
			cout << "Loss sum: " << batch_loss << endl;
			cout << "训练成功" << endl;
		}
		else
		{
			cout << "输入不匹配" << endl;
		}
	}

	//训练 基于准确率阈值
	void Net::train(cv::Mat input, cv::Mat target_, float accuracy_threshold)
	{
		if (input.empty())
		{
			cout << "输入为空" << endl;
			return;
		}

		cout << "开始训练" << endl;

		cv::Mat sample;
		if (input.rows == (layer[0].rows) && input.cols == 1)
		{
			target = target_;
			sample = input;
			layer[0] = sample;
			forward();
			int num_of_train = 0;
			while (accuracy < accuracy_threshold)
			{
				backward();
				forward();
				num_of_train++;
				if (num_of_train % 500 == 0)
				{
					cout << "训练 " << num_of_train << " 次" << endl;
					cout << "Loss: " << loss << endl;
				}
			}
			cout << endl << "训练 " << num_of_train << " 次" << endl;
			cout << "Loss: " << loss << endl;
			cout << "训练成功" << endl;
		}
		else if (input.rows == (layer[0].rows) && input.cols > 1)
		{
			double batch_loss = 0.;
			int epoch = 0;
			while (accuracy < accuracy_threshold)
			{
				batch_loss = 0.;
				for (int i = 0; i < input.cols; ++i)
				{
					target = target_.col(i);
					sample = input.col(i);

					layer[0] = sample;
					forward();
					batch_loss += loss;
					backward();
				}
				test(input, target_);
				epoch++;
				if (epoch % 10 == 0)
				{
					cout << "Number of epoch: " << epoch << endl;
					cout << "Loss sum: " << batch_loss << endl;
				}
				if (epoch % 100 == 0)
				{
					learning_rate *= fine_tune_factor;
				}
			}
			cout << endl << "Number of epoch: " << epoch << endl;
			cout << "Loss sum: " << batch_loss << endl;
			cout << "训练成功" << endl;
		}
		else
		{
			cout << "输入不匹配" << endl;
		}
	}

	//测试
	void Net::test(cv::Mat& input, cv::Mat& target_)
	{
		if (input.empty())
		{
			cout << "输入为空" << endl;
			return;
		}
		cout << endl << "开始预测" << endl;

		if (input.rows == (layer[0].rows) && input.cols == 1)
		{
			int predict_number = predict_one(input);

			cv::Point target_maxLoc;
			minMaxLoc(target_, NULL, NULL, NULL, &target_maxLoc, cv::noArray());
			int target_number = target_maxLoc.y;

			cout << "Predict: " << predict_number << endl;
			cout << "Target:  " << target_number << endl;
			cout << "Loss: " << loss << endl;
		}
		else if (input.rows == (layer[0].rows) && input.cols > 1)
		{
			double loss_sum = 0;
			int right_num = 0;
			cv::Mat sample;
			for (int i = 0; i < input.cols; ++i)
			{
				sample = input.col(i);
				int predict_number = predict_one(sample);
				loss_sum += loss;

				target = target_.col(i);
				cv::Point target_maxLoc; //CV点类存储目标数据矩阵里最大值所在的位置
				minMaxLoc(target, NULL, NULL, NULL, &target_maxLoc, cv::noArray()); //在矩阵中查找最小值和最大值，以及它们的位置
				//void cv::minMaxLoc(InputArray src, double* minVal, double* maxVal = 0, Point* minLoc = 0, Point* maxLoc = 0, InputArray mask = noArray());
				//参数：输入矩阵 指向存储最小/大值的变量的指针 指向存储最小/大值位置的 cv::Point 对象的指针 可选的掩码矩阵，用于指定查找范围
				//noArray()用于在调用 OpenCV 函数时表示某个参数不需要输入数组
				int target_number = target_maxLoc.y; //纵坐标值就对应着目标的类别编号

				cout << "Test sample: " << i << "   " << "Predict: " << predict_number << endl;
				cout << "Test sample: " << i << "   " << "Target:  " << target_number << endl << endl;
				if (predict_number == target_number)
				{
					right_num++;
				}
			}
			accuracy = (double)right_num / input.cols;
			cout << "Loss sum: " << loss_sum << endl;
			cout << "accuracy: " << accuracy << endl;
		}
		else
		{
			cout << "输入不匹配" << endl;
			return;
		}
	}

	//预测
	int Net::predict_one(cv::Mat& input)
	{
		if (input.empty())
		{
			cout << "输入为空" << endl;
			return -1;
		}

		if (input.rows == (layer[0].rows) && input.cols == 1)
		{
			layer[0] = input;
			forward();

			cv::Mat layer_out = layer[layer.size() - 1]; 
			cv::Point predict_maxLoc;

			minMaxLoc(layer_out, NULL, NULL, NULL, &predict_maxLoc, cv::noArray());
			return predict_maxLoc.y;
		}
		else
		{
			cout << "请单独给出一个示例并确保输入匹配" << endl;
			return -1;
		}
	}
	vector<int> Net::predict(cv::Mat& input)
	{
		vector<int> predicted_labels;
		if (input.rows == (layer[0].rows) && input.cols > 1)
		{
			for (int i = 0; i < input.cols; ++i)
			{
				cv::Mat sample = input.col(i);
				int predicted_label = predict_one(sample);
				predicted_labels.push_back(predicted_label);
			}
		}
		return predicted_labels;
	}

	//读取样本和标签 
	void get_input_label(string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start)
	{
		cv::FileStorage fs;
		fs.open(filename, cv::FileStorage::READ);
		cv::Mat input_, target_;
		fs["input"] >> input_;
		fs["target"] >> target_;
		fs.release();
		input = input_(cv::Rect(start, 0, sample_num, input_.rows));
		label = target_(cv::Rect(start, 0, sample_num, target_.rows));
	}

	//保存模型
	void Net::save(string filename)
	{
		cv::FileStorage model(filename, cv::FileStorage::WRITE);
		model << "layer_neuron_num" << layer_neuron_num;
		model << "learning_rate" << learning_rate;
		model << "activation_function" << activation_function;

		for (int i = 0; i < weights.size(); i++)
		{
			string weight_name = "weight_" + to_string(i);
			model << weight_name << weights[i];
		}
		model.release();
	}

	//加载模型
	void Net::load(std::string filename)
	{
		cv::FileStorage fs;
		fs.open(filename, cv::FileStorage::READ);
		cv::Mat input_, target_;

		fs["layer_neuron_num"] >> layer_neuron_num;
		initNet(layer_neuron_num);

		for (int i = 0; i < weights.size(); i++)
		{
			std::string weight_name = "weight_" + std::to_string(i);
			fs[weight_name] >> weights[i];
		}

		fs["learning_rate"] >> learning_rate;
		fs["activation_function"] >> activation_function;

		fs.release();
	}

	//绘图
	void draw_curve(cv::Mat& board, std::vector<double> points)
	{
		cv::Mat board_(620, 1000, CV_8UC3, cv::Scalar::all(200)); //创建一个大小为 620x1000 的 8 位无符号 3 通道图像 board_，并将其所有像素初始化为 200
		board = board_;
		cv::line(board, cv::Point(0, 550), cv::Point(1000, 550), cv::Scalar(0, 0, 0), 2); //线宽为 2 像素
		cv::line(board, cv::Point(50, 0), cv::Point(50, 1000), cv::Scalar(0, 0, 0), 2);

		for (size_t i = 0; i < points.size() - 1; i++)
		{
			cv::Point pt1(50 + i * 2, (int)(548 - points[i]));
			cv::Point pt2(50 + i * 2 + 1, (int)(548 - points[i + 1]));
			cv::line(board, pt1, pt2, cv::Scalar(0, 0, 255), 2);
			if (i >= 1000)
			{
				return;
			}
		}
		cv::imshow("Loss", board);
		cv::waitKey(10); //等待 10 毫秒
	}
}