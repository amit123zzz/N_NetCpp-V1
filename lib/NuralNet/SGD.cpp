#include "SGD.h"

#include<execution>
#include<algorithm>

SGD::SGD(const std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>& params, float lr) :
	param_{ params }, learningRate_{ lr }
{
}

void SGD::step()
{
	for (auto& [name, param] : param_) {
		//for (std::size_t i{}; i < param->numElement(); ++i) {
		//	param->data()[i] -= learningRate_ * param->grad()[i];
		//}
		const std::size_t N = param->numElement();

		// 1. Hoist pointers out of the loop! (CRITICAL)
		// This allows the compiler to see contiguous memory and auto-vectorize.
		float* const w_ptr = param->data().data();
		const float* const g_ptr = param->grad().data();
		const float lr = learningRate_;

		// Parallel execution for large weight matrices
		std::transform(
			std::execution::par,
			w_ptr, w_ptr + N,          // First input range (Weights)
			g_ptr,                     // Second input start (Gradients)
			w_ptr,                     // Destination output (Overwrite weights)
			[lr](float w, float g) {   // Lambda for the math
				return w - lr * g;
			}
		);
	}
}

void SGD::zeroGrad()
{
	for (auto& [name, param] : param_) {
		param->zeroGrad();
	}
}
