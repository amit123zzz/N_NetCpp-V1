#pragma once

#include"../Module/Module.h"
#include"../Module/Flatten.h"
#include"../Module/Linear.h"
#include"../Module/ReLU.h"

#include<fstream>

#include<memory>

class NuralNet : public Module
{
public:
	NuralNet();

	std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

	void printLinear1(std::ofstream& out);

private:
	std::shared_ptr<Flatten> flatten_ = std::make_shared<Flatten>();
	std::shared_ptr<Linear> linear1_ = std::make_shared<Linear>(28 * 28, 512);
	std::shared_ptr<Linear> linear2_ = std::make_shared<Linear>(512, 512);
	std::shared_ptr<Linear> linear3_ = std::make_shared<Linear>(512, 10);

	std::shared_ptr<ReLU> relu_ = std::make_shared<ReLU>();
};