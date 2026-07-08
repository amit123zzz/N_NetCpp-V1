#include<iostream>

#include"../lib/NuralNet/Dataset.h"
#include"../lib/NuralNet/DataLoader.h"
#include"../lib/Module/Loss.h"
#include"../lib/NuralNet/NuralNet.h"
#include"../lib/NuralNet/SGD.h"

#include<fstream>

#include"../lib/NuralNet/Serialization.hpp"

#include<numeric>
#include<random>

void train(DataLoader& dataLoader, NuralNet& model, CrossEntropyLoss& crl, SGD& optimizer) {

	//std::ofstream _out{ "out_file" };

	std::size_t _logInterval = 100;
	std::size_t _batchCount = 0;
	std::size_t _seenSamples = 0;

	std::size_t _batchNo{};

	for (const auto& batch : dataLoader) {
		std::shared_ptr<Tensor> _totalLoss = nullptr;
		std::size_t _batchSize = batch.size();

		for (const auto& [label, tensor] : batch) {
			//std::cout << *tensor << std::endl;
			auto _output = model(tensor);
			//std::cout << *_output << std::endl;
			auto _loss = crl(_output, label);
			//model.printLinear1(_out);

 			if (_totalLoss == nullptr) {
				_totalLoss = _loss;
			}
			else {
				_totalLoss = (*_totalLoss) + _loss;
			}
			_seenSamples += 1;
		}

		//_batchNo++;
		//std::cout << std::format("{} batch done", _batchNo) << std::endl;

		_totalLoss->item() /= _batchSize;

		if (_batchCount % _logInterval == 0) {
			std::string _outStr = std::format(
				"Loss : {} [{} / {}]", _totalLoss->item(), _seenSamples,
				dataLoader.amountSamples());

			//_out << _outStr;

			std::cout << _outStr << std::endl;
		}

		_totalLoss->backward();

		optimizer.step();
		optimizer.zeroGrad();
		_batchCount += 1;
	}
}

void test(DataLoader& dataLoader, NuralNet& model, CrossEntropyLoss& lossFn) {
	float _runningLoss = 0.0f;
	std::size_t _correct{};
	std::size_t _numSamples{};

	for (const auto& batch : dataLoader) {
		for (const auto& [label, tensor] : batch) {
			auto _output = model(tensor);
			if (_output->argmax() == label) {
				_correct += 1;
			}
			_runningLoss += lossFn(_output, label)->item();
			_numSamples += 1;
		}
	}

	float _accuracy = static_cast<float>(_correct) / static_cast<float>(_numSamples);
	float _avgLoss = _runningLoss / _numSamples;

	std::cout << std::format("Test Error: \n accuracy : {} \n avg. loss : {}", _accuracy * 100.0f, _avgLoss) << std::endl;
}

void trainNewMNISTModel() {
	MNIST _trainMNIST{ "E:/ML_TRAINING/train-images-idx3-ubyte",
		"E:/ML_TRAINING/train-labels.idx1-ubyte" };

	MNIST _testMNIST{
		"E:/ML_TRAINING/t10k-images.idx3-ubyte",
		"E:/ML_TRAINING/t10k-labels.idx1-ubyte"
	};

	std::cout << "DataSet loaded" << std::endl;

	std::int32_t _batchSize{ 10 };

	DataLoader _trainDataLoader{ &_trainMNIST, _batchSize };
	DataLoader _testDataLoader{ &_testMNIST, _batchSize };

	NuralNet _model{};
	CrossEntropyLoss _lossFn{};
	float _learningRate = 0.001f;

	SGD _optimizer{ _model.parameters(), _learningRate };

	std::cout << "Training started..." << std::endl;

	std::uint32_t _epochs{ 3 };
	for (std::uint32_t epoch{}; epoch < _epochs; ++epoch) {
		std::cout << std::format("[Epoch: {} / {}] Training...", epoch, _epochs) << std::endl;
		train(_trainDataLoader, _model, _lossFn, _optimizer);
		std::cout << std::format("[Epoch: {} / {}] Testing...", epoch, _epochs) << std::endl;
		test(_testDataLoader, _model, _lossFn);
	}

	std::cout << "Training ended..." << std::endl;

	auto _stateDict = _model.stateDictionary();
	save(_stateDict, "mnist.nn");
}

void inference_on_saved_model()
{
	NuralNet model;
	std::cout << "Loading model..." << std::endl;
	auto loaded_state_dict = load("mnist.nn");
	model.loadStateDictionary(loaded_state_dict);

	std::cout << "Loading test set..." << std::endl;
	// MNIST mnist_test =
	//     MNIST("data/MNIST/raw/t10k-images-idx3-ubyte", "data/MNIST/raw/t10k-labels-idx1-ubyte");
	MNIST mnist_test{
		"E:/ML_TRAINING/t10k-images.idx3-ubyte",
		"E:/ML_TRAINING/t10k-labels.idx1-ubyte"
	};
	int n_samples = 100;

	std::vector<int> all_indices(mnist_test.getLength());
	std::iota(all_indices.begin(), all_indices.end(), 0);
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(all_indices.begin(), all_indices.end(), g);
	std::vector<int> indices(all_indices.begin(), all_indices.begin() + n_samples);

	for (int i = 0; i < n_samples; i++)
	{
		std::cout << "Sample " << i << " of " << n_samples << std::endl;
		std::pair<int, std::shared_ptr<Tensor>> sample_image = mnist_test.getItem(indices[i]);
		visualizeImage(sample_image.second);
		auto output = model(sample_image.second);
		int predicted_class = output->argmax();
		std::cout << "Predicted class: " << mnist_test.labelToClass(predicted_class) << std::endl;
		std::cout << "Actual class: " << mnist_test.labelToClass(sample_image.first) << std::endl;
		std::cout << "----------------------------" << std::endl;
	}
}

void main() {

	try
	{
		trainNewMNISTModel();
		//inference_on_saved_model();
	}
	catch (Error& e) {
		std::cout << e.what() << std::endl;
		std::cout << e.trace() << std::endl;
	}

}
