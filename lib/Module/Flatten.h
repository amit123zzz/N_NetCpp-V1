#pragma once

#include"Module.h"

class Flatten : public Module
{
public:
	std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

private:

};

