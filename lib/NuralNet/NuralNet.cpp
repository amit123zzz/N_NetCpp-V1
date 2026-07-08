#include "NuralNet.h"

NuralNet::NuralNet()
{
	registerModule("linear1", linear1_);
	registerModule("linear2", linear2_);
	registerModule("linear3", linear3_);
}

std::shared_ptr<Tensor> NuralNet::forward(const std::shared_ptr<Tensor>& input)
{
	std::shared_ptr<Tensor> _flat = (*flatten_)(input);
	std::shared_ptr<Tensor> _linear1 = (*linear1_)(_flat);
	std::shared_ptr<Tensor> _relu1 = (*relu_)(_linear1);
	std::shared_ptr<Tensor> _linear2 = (*linear2_)(_relu1);
	std::shared_ptr<Tensor> _relu2 = (*relu_)(_linear2);
	std::shared_ptr<Tensor> _linear3 = (*linear3_)(_relu2);

	return _linear3;
}

void NuralNet::printLinear1(std::ofstream& out)
{
	//out << "Weight 1 : " << std::endl;
	//out << *(linear1_->weight()) << std::endl;
	out << "Bias 1 : " << std::endl;
	out << *(linear1_->bias()) << std::endl;
}
