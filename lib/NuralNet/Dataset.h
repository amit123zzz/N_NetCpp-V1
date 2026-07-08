#pragma once

#include"../Tensor/Tensor.h"
#include<utility>
#include<memory>
#include<string>
#include<vector>
#include<cstdint>

#include<iostream>

class Dataset
{
public:
	virtual std::pair<std::int32_t, std::shared_ptr<Tensor>> getItem(std::uint32_t index) = 0;
	virtual std::int32_t getLength() = 0;

private:

};

class MNIST : public Dataset {
public:
	MNIST(const std::string& dataPath, const std::string& labelPath);
	std::pair<std::int32_t, std::shared_ptr<Tensor>> getItem(std::uint32_t index) override;
	std::int32_t getLength() override;
	std::string labelToClass(std::uint32_t label) const;

private:
	std::vector<std::vector<std::vector<float>>> images_;
	std::vector<std::int32_t> labels_;
	std::vector < std::string> classes_ = {
		"zero", "one", "two", "three", "four", "five",
		"six", "seven", "eight", "nine"
	};

	void readMNIST(const std::string& path);
	void readMNISTLabels(const std::string& path);

	std::int32_t reverseInt(std::int32_t num) const;

	float convertToFloat(unsigned char c) const;
};

inline void visualizeImage(std::shared_ptr<Tensor> image) {
	for (std::size_t i{}; i < image->shape()[0]; ++i) {
		for (std::size_t j{}; j < image->shape()[1]; ++j) {
			float _pxl = (*image)[i, j];
			std::cout << (_pxl > 0.75 ? '@' :
				_pxl > 0.5 ? '#' : _pxl > 0.25 ? '+' :
				_pxl > 0.1 ? '.' : ' ');
		}

		std::cout << '\n';
	}
}

