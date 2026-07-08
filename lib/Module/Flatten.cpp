#include "Flatten.h"

#include<mutex>

std::shared_ptr<Tensor> Flatten::forward(const std::shared_ptr<Tensor>& input)
{
    std::vector<float> _ret = std::move(input->data());

    if (input->requiresGrad()) {
        std::vector < std::shared_ptr<Tensor>> _parents{ input };
        std::function<void(const std::vector<float>&)> _gradFn =
            [input](const std::vector<float>& gradOther) {
            input->addToGrad(gradOther);
            {
                std::lock_guard<std::mutex> lock(input->gradMutex());
                input->addToGrad(gradOther);
            }
            };

        return std::make_shared<Tensor>(std::move(_ret), true, std::move(_gradFn), std::move(_parents));
    }

    return std::make_shared<Tensor>(std::move(_ret));
}
