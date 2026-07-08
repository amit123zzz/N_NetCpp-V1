#pragma once

#include"Module.h"

class Linear : public Module
{
public:
	Linear(std::size_t inputFeatures, std::size_t outputFeatures, std::size_t seed = 7);
	void resetParameters();
	std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

	std::shared_ptr<Tensor> weight() const;
	std::shared_ptr<Tensor> bias() const;

private:
	std::shared_ptr<Tensor> weights_;
	std::shared_ptr<Tensor> bias_;
	std::size_t inputFeatures_;
	std::size_t outputFeatures_;
	std::size_t seed_;
};

