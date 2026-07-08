#pragma once

#include"Module.h"

#include"SoftMax.h"

class Loss : public Module
{
public:
	std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
	virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::size_t target);
	std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input, std::size_t target);
private:

};

class NLLLoss : public Loss {
public:
	std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::size_t target) override;

private:

};

class CrossEntropyLoss : public Loss {
public:
	std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::size_t target) override;

private:

};

