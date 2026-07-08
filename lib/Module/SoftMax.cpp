#include "SoftMax.h"

std::shared_ptr<Tensor> SoftMax::forward(const std::shared_ptr<Tensor>& input)
{
    if (input->shape().size() == 0) {
        float _result{ 1.0f };

        if (input->requiresGrad()) {
            std::vector<std::shared_ptr<Tensor>> _parents{ input };
            std::function<void(const std::vector<float>&)> _gradFn =
                [input](const std::vector<float>& gradOther) {
                
                {
                    std::lock_guard<std::mutex> lock(input->gradMutex());
                    input->addToGrad({ 0.0f });
                }
                };
            return std::make_shared<Tensor>(_result, true, std::move(_gradFn), std::move(_parents));
        }

        return std::make_shared<Tensor>(_result);
    }

    // 1D

    if (input->shape().size() == 1) {
        float _maxVal = (*input)[0];

        for (std::size_t i{}; i < input->numElement(); ++i) {
            if ((*input)[i] > _maxVal) {
                _maxVal = (*input)[i];
            }
        }

        std::vector<float> _s{};
        _s.reserve(input->numElement());
        float _sumExp{};
        for (std::size_t i{}; i < input->numElement(); ++i) {
            _sumExp += std::exp((*input)[i] - _maxVal);
        }

        for (std::size_t i{}; i < input->numElement(); ++i) {
            _s.emplace_back(static_cast<float>(std::exp((*input)[i] - _maxVal) / _sumExp));
        }

        if (input->requiresGrad()) {
            std::vector<std::shared_ptr<Tensor>> _parents{ input };
            std::function<void(const std::vector<float>&)> _gradFn =
                [input, _s](const std::vector<float>& gradOutput) {
                // Step 1: Calculate a shared scalar (Dot Product of gradOutput and _s)
                    // This loop runs in O(N) time
                float sum_grad_out_times_s = 0.0f;
                for (std::size_t i = 0; i < input->numElement(); ++i) {
                    sum_grad_out_times_s += gradOutput[i] * _s[i];
                }

                // Step 2: Compute input gradient in a single O(N) pass
                std::vector<float> _gradInput(input->numElement());
                for (std::size_t j = 0; j < input->numElement(); ++j) {
                    _gradInput[j] = _s[j] * (gradOutput[j] - sum_grad_out_times_s);
                }
                
                {
                    std::lock_guard<std::mutex> lock(input->gradMutex());
                    input->addToGrad(std::move(_gradInput));
                }
                };

            return std::make_shared<Tensor>(std::move(_s), true, std::move(_gradFn), std::move(_parents));
        }

        return std::make_shared<Tensor>(std::move(_s));
    }
    
}
