#include "Loss.h"

std::shared_ptr<Tensor> Loss::forward(const std::shared_ptr<Tensor>& input)
{
    throw Error(std::format("Loss expects an input and a target"));
}

std::shared_ptr<Tensor> Loss::forward(std::shared_ptr<Tensor> input, std::size_t target)
{
    throw Error("forward not implemented.");
}

std::shared_ptr<Tensor> Loss::operator()(std::shared_ptr<Tensor> input, std::size_t target)
{
    return forward(input, target);
}

std::shared_ptr<Tensor> NLLLoss::forward(std::shared_ptr<Tensor> input, std::size_t target)
{
    if(input->shape().size() != 1) {
        throw Error("NLLLoss accepts an 1D input tensor");
    }

    if (target >= input->numElement()) {
        throw Error("NLLLoss target out of bound");
    }

    float _prob = std::max((*input)[target], 1e-12f);
    float _loss = -std::log(_prob);

    if (input->requiresGrad()) {
        std::vector<std::shared_ptr<Tensor>> _parents{ input };

        std::function<void(const std::vector<float>&)> _gradFn =
            [input, target](const std::vector<float>& gradOutput) {
            //std::cout << "[NLLLoss backward] input[target]=" << (*input)[target]
            //    << " gradOutput[0]=" << gradOutput[0] << std::endl;
            std::vector<float> _gradInput{};
            _gradInput.reserve(input->numElement());
            for (std::size_t i{}; i < input->numElement(); ++i) {
                if (i == target) {
                    _gradInput.emplace_back(static_cast<float>(gradOutput[0] * 
                        (-1.0f / (*input)[i])));
                }
                else {
                    _gradInput.emplace_back(0.0f);
                }
            }
            
            {
                std::lock_guard<std::mutex> lock(input->gradMutex());
                input->addToGrad(_gradInput);
            }
            };

        return std::make_shared<Tensor>(std::move(_loss), true, std::move(_gradFn), std::move(_parents));
    }

    return std::make_shared<Tensor>(std::move(_loss));
}

std::shared_ptr<Tensor> CrossEntropyLoss::forward(std::shared_ptr<Tensor> input, std::size_t target)
{
    if (input->shape().size()!= 1) {
        throw Error("Cross Entropy Loss accepts an 1D tensor");
    }

    if (target >= input->numElement()) {
        throw Error("Cross entropy loss index is out of bound");
    }

    SoftMax _softMax{};
    NLLLoss _nllLoss{};

    //std::cout << *input << std::endl;

    std::shared_ptr<Tensor> _softMaxOutput = _softMax(input);

    //std::cout << *_softMaxOutput << std::endl;

    return _nllLoss(_softMaxOutput, target);
}
