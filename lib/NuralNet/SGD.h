#pragma once

#include"../Tensor/Tensor.h"

#include<memory>
#include<vector>
#include<string>
#include<utility>

class SGD
{
public:
	SGD(const std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>& params, float lr);
	void step();
	void zeroGrad();

private:
	std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> param_;
	float learningRate_;
};

