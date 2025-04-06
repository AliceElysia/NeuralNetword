#include <opencv2\opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int csv2xml()
{
	Ptr<ml::TrainData> trainData = ml::TrainData::loadFromCSV("train.csv", 0);
	if (!trainData || trainData->getNSamples() == 0)
	{
		cout << "加载CSV文件失败" << endl;
		return -1;
	}
	Mat data = trainData->getSamples();
	cout << "数据读取成功" << endl;

	Mat input_ = data(Rect(1, 1, 784, data.rows - 1)).t();
	//Mat label_ = data(Rect(0, 1, 1, data.rows - 1));
	Mat label_ = trainData->getResponses();

	Mat target_(10, input_.cols, CV_32F, Scalar::all(0.)); //初始化为0

	Mat digit(28, 28, CV_32FC1); //28*28浮点型矩阵
	Mat col_0 = input_.col(3);
	float label0 = label_.at<float>(3, 0);
	cout << label0;
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			digit.at<float>(i, j) = col_0.at<float>(i * 28 + j); //一维向量转换为二维矩阵
		}
	}

	for (int i = 0; i < label_.rows; ++i) //label_.rows为样本的数量
	{
		float label_num = label_.at<float>(i, 0);
		target_.at<float>(label_num, i) = label_num;
	}

	//归一化
	Mat input_normalized(input_.size(), input_.type());
	for (int i = 0; i < input_.rows; ++i)
	{
		for (int j = 0; j < input_.cols; ++j) 
		{
			input_normalized.at<float>(i, j) = input_.at<float>(i, j) / 255.;
		}
	}

	string filename = "input_label_0-9.xml";
	FileStorage fs(filename, FileStorage::WRITE); //写入模式
	fs << "input" << input_normalized;
	fs << "target" << target_; // Write cv::Mat
	fs.release();


	Mat input_1000 = input_normalized(Rect(0, 0, 10000, input_normalized.rows));
	Mat target_1000 = target_(Rect(0, 0, 10000, target_.rows));

	string filename2 = "input_label_0-9_10000.xml";
	FileStorage fs2(filename2, FileStorage::WRITE);

	fs2 << "input" << input_1000;
	fs2 << "target" << target_1000; // Write cv::Mat
	fs2.release();

	return 0;
}