#include"Net.h"

using namespace std;
using namespace cv;
using namespace Elysia;

int main(int argc, char* argv[])
{
	//设置每层神经元个数
	vector<int> layer_neuron_num = { 784,100,10 };

	//初始化网络与权重
	Net net;
	net.initNet(layer_neuron_num);
	net.initWeights(0, 0., 0.01);
	Scalar scalar(0.5);
	net.initBias(scalar);

	//获取训练集与测试集
	Mat input, label, test_input, test_label;
	int sample_number = 800;
	get_input_label("data/input_label_1000.xml", input, label, sample_number, 0);
	get_input_label("data/input_label_1000.xml", test_input, test_label, 200, 800);

	//设置训练参数
	float loss_threshold = 0.5;
	net.learning_rate = 0.3;
	net.output_interval = 10;
	net.activation_function = "sigmoid";

	//训练，绘制损失曲线并测试训练好的网络
	net.train(input, label, loss_threshold, false);
	net.test(test_input, test_label);

	//保存模型
	net.save("models/model_sigmoid_800_200.xml");

	getchar();
	return 0;
}