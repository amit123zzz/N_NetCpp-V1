#include "Linear.h"

#include<random>

Linear::Linear(std::size_t inputFeatures, std::size_t outputFeatures, std::size_t seed) :
	inputFeatures_{ inputFeatures }, outputFeatures_{ outputFeatures }, seed_{ seed },
	weights_{
		std::make_shared<Tensor>(
			std::vector<std::vector<float>>(inputFeatures, std::vector<float>(outputFeatures, 0.0f)),
			true
		)
	},
	bias_{ std::make_shared<Tensor>(std::vector<float>(outputFeatures, 0.0f), true) }
{
	registerParameter("weight", weights_);
	registerParameter("bias", bias_);

	resetParameters();
}

void Linear::resetParameters() {
	float _gain = std::sqrt(2.0f);
	std::size_t _fnIn = inputFeatures_;
	float _bond = _gain * std::sqrt(3.0f / _fnIn);

	std::mt19937 _generator{ static_cast<std::uint32_t>(seed_) };

	for (std::size_t i{}; i < weights_->shape()[0]; ++i) {
		for (std::size_t j{}; j < weights_->shape()[1]; ++j) {
			(*weights_)[i, j] = std::uniform_real_distribution<float>(-_bond, _bond)(_generator);
		}
	}
}

std::shared_ptr<Tensor> Linear::forward(const std::shared_ptr<Tensor>& input)
{
	std::shared_ptr<Tensor> _xW = (*input) * weights_;
	return (*_xW) + bias_;
}

std::shared_ptr<Tensor> Linear::weight() const
{
	return weights_;
}

std::shared_ptr<Tensor> Linear::bias() const
{
	return bias_;
}
