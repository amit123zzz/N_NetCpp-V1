#include "ReLU.h"
#include<execution>
#include<algorithm>

std::shared_ptr<Tensor> ReLU::forward(const std::shared_ptr<Tensor>& input)
{
    if (input->shape().size() == 0) {
        float _result = input->item() > 0 ? input->item() : 0.0f;

        if (input->requiresGrad()) {
            std::vector<std::shared_ptr<Tensor>> _parents{ input };
            std::function<void(const std::vector<float>&)> _gradFn =
                [input](const std::vector<float>& gradOutput) {
                float _amount = input->item() > 0 ? gradOutput[0] : 0.0f;
                input->addToGrad({_amount});
                };

            return std::make_shared<Tensor>(_result, true, std::move(_gradFn), std::move(_parents));

        }

        return std::make_shared<Tensor>(_result);
    }
    else if (input->shape().size() == 1) {
        const std::size_t N = input->shape()[0];

        // 1. Allocate exact memory size instantly
        std::vector<float> _result(N);

        // 2. Fetch raw flat pointers to completely bypass tracking overhead inside threads
        const float* const input_ptr = input->data().data();
        float* const res_ptr = _result.data();

        // 3. Parallel Element-wise ReLU
        std::transform(
            std::execution::par,           // Enable multi-threading
            input_ptr,                     // Start of input range
            input_ptr + N,                 // End of input range
            res_ptr,                       // Destination output location
            [](float val) {                // Lambda operation per element
                return val > 0.0f ? val : 0.0f;
            }
        );

        if (input->requiresGrad()) {
            std::vector<std::shared_ptr<Tensor>> _parents{ input };
            std::function<void(const std::vector<float>&)> _gradFn =
                [input](const std::vector<float>& gradOutput) {
                const std::size_t N = input->numElement();

                // 1. Allocate exact memory size instantly
                std::vector<float> _gradInput(N);

                // 2. Fetch raw flat pointers to completely bypass tracking overhead inside threads
                const float* const input_ptr = input->data().data();
                const float* const grad_out_ptr = gradOutput.data();
                float* const grad_in_ptr = _gradInput.data();

                // 3. Parallel Element-wise ReLU Gradient Calculation
                std::transform(
                    std::execution::par,           // Enable multi-threading
                    input_ptr,                     // Start of first input range (input activations)
                    input_ptr + N,                 // End of first input range
                    grad_out_ptr,                  // Start of second input range (incoming gradients)
                    grad_in_ptr,                   // Destination output location
                    [](float input_val, float grad_out_val) { // Lambda operation per element
                        return input_val > 0.0f ? grad_out_val : 0.0f;
                    }
                );

                {
                    std::lock_guard<std::mutex> lock(input->gradMutex());
                    input->addToGrad(std::move(_gradInput));
                }
                };
            
            return std::make_shared<Tensor>(_result, true, std::move(_gradFn), std::move(_parents));
        }

        return std::make_shared<Tensor>(_result);
    }
    else {
        // 1. Pre-allocate rows
        const std::size_t rows = input->shape()[0];
        const std::size_t cols = input->shape()[1];
        std::vector<std::vector<float>> _result(rows, std::vector<float>(cols));

        // 2. Create an index vector to iterate over in parallel
        std::vector<std::size_t> row_indices(rows);
        std::iota(row_indices.begin(), row_indices.end(), 0);

        // 3. Run parallel for_each over the row indices
        std::for_each(std::execution::par, row_indices.begin(), row_indices.end(), [&](std::size_t i) {
            for (std::size_t j = 0; j < cols; ++j) {
                float _entry = (*input)[i, j] > 0 ? (*input)[i, j] : 0.0f;
                _result[i][j] = _entry;
            }
            });

        if (input->requiresGrad()) {
            std::vector<std::shared_ptr<Tensor>> _parents{ input };
            std::function<void(const std::vector<float>&)> _gradFn =
                [input](const std::vector<float>& gradOutput) {
                const std::size_t size = input->numElement();
                std::vector<float> _gradInput(size); // Pre-allocate memory upfront

                // Safely parallel transform across the flat data
                std::transform(
                    std::execution::par,
                    input->data().begin(), input->data().end(), // Source data boundaries
                    gradOutput.begin(),                         // Second input stream boundary
                    _gradInput.begin(),                         // Output target destination
                    [](float input_val, float grad_val) {
                        return input_val > 0 ? grad_val : 0.0f;
                    }
                );

                {
                    std::lock_guard<std::mutex> lock(input->gradMutex());
                    input->addToGrad(_gradInput);
                }
                };

            return std::make_shared<Tensor>(_result, true, _gradFn, _parents);
        }

        return std::make_shared<Tensor>(_result);
    }
}
