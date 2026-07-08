#include "Tensor.h"

#include<stdexcept>

#include<format>

#include<numeric>
#include<execution>


Tensor::Tensor(float data, bool requiresGrad,
	std::function<void(const std::vector<float>&)> gradFn,
	std::vector<std::shared_ptr<Tensor>> parents):
	data_{data}, shape_{}, stride_{}, requiresgrad_{requiresGrad}, 
	gradFn_{gradFn}, parents_{parents}
{
	if (requiresgrad_) {
		zeroGrad();
	}
}

Tensor::Tensor(const std::vector<float>& data, bool requiresGrad,
	std::function<void(const std::vector<float>&)> gradFn,
	std::vector<std::shared_ptr<Tensor>> parents) :
	data_{ data }, shape_{ data.size() }, stride_{1}, requiresgrad_{requiresGrad},
	gradFn_{gradFn}, parents_{parents}
{
	if (requiresgrad_) {
		zeroGrad();
	}
}

Tensor::Tensor(const std::vector<std::vector<float>>& data, bool requiresGrad,
	std::function<void(const std::vector<float>&)> gradFn,
	std::vector<std::shared_ptr<Tensor>> parents):
	shape_{data.size(), data[0].size()}, stride_{data[0].size(), 1}, requiresgrad_{requiresGrad},
	gradFn_{gradFn}, parents_{parents}
{
	std::size_t _nExpectedColumns{ data[0].size() };
	for (std::size_t i{}; i < data.size(); ++i) {
		if (data[i].size() != _nExpectedColumns) {
			throw Error("Dimension mismatch in Tensor class");
		}
	}

	//for (std::size_t i{}; i < data.size(); ++i) {
	//	//for (std::size_t j{}; j < data[i].size(); ++j) {
	//	//	data_.emplace_back(data[i][j]);
	//	//}

	//}

	data_.reserve(data.size() * data[0].size());

	for (const auto& row : data) {
		data_.insert(data_.end(), row.begin(), row.end());
	}

	if (requiresgrad_) {
		zeroGrad();
	}
}

const float& Tensor::item() const
{
	if (data_.size() == 1) {
		return data_[0];
	}
	else {
		throw Error("item() can be only called for scalar tensors");
	}
}

float& Tensor::item()
{
	if (data_.size() == 1) {
		return data_[0];
	}
	else {
		throw Error("item() can be only called for scalar tensors");
	}
}

const float& Tensor::operator[](std::size_t index) const
{
	if (shape_.size() == 0) {
		throw Error("scalar tensor cannot be accessed via [] operator in Tensor class");
	}

	if (shape_.size() == 1) {
		if (index >= shape_[0]) {
			throw Error(std::format("Index {} is out of bound, Max size {} in Tensor class\n", index, shape_[0]));
		}

		return data_[index];
	}

	throw Error("This is 1D tensor. Use 2 indices for 2D tensor in Tensor class.");
}

float& Tensor::operator[](std::size_t index)
{
	if (shape_.size() == 0) {
		throw Error("scalar tensor cannot be accessed via [] operator in Tensor class");
	}

	if (shape_.size() == 1) {
		if (index >= shape_[0]) {
			throw Error(std::format("Index {} is out of bound, Max size {} in Tensor class\n", index, shape_[0]));
		}

		return data_[index];
	}

	throw Error("This is 1D tensor. Use 2 indices for 2D tensor.");
}

const float& Tensor::operator[](std::size_t index1, std::size_t index2) const
{
	if (shape_.size() == 2) {
		if (index1 >= shape_[0]) {
			throw Error(std::format("Row index {} out of bound of {} max size in Tensor class", index1, shape_[0]));
		}

		if (index2 >= shape_[1]) {
			throw Error(std::format("Column index {} out of bound of {} max size in Tensor class", index2, shape_[1]));
		}

		return data_[index1 * stride_[0] + index2 * stride_[1]];
	}

	throw Error("Can only double index into 2D tensors");
}

float& Tensor::operator[](std::size_t index1, std::size_t index2)
{
	if (shape_.size() == 2) {
		if (index1 >= shape_[0]) {
			throw Error(std::format("Row index {} out of bound of {} max size in Tensor class", index1, shape_[0]));
		}

		if (index2 >= shape_[1]) {
			throw Error(std::format("Column index {} out of bound of {} max size in Tensor class", index2, shape_[1]));
		}

		return data_[index1 * stride_[0] + index2 * stride_[1]];
	}

	throw Error("Can only double index into 2D tensors");
}

std::ostream& operator<<(std::ostream& os, const Tensor& tObj) {
	if (tObj.shape_.size() == 0) {
		os << tObj.data_[0];
	}
	else if (tObj.shape_.size() == 1) {
		os << std::format("{}", tObj.data_);
	}
	else {
		std::string _str = "[ ";
		for (std::size_t i{}; i < tObj.shape_[0]; ++i) {
			_str += "[ ";
			for (std::size_t j{}; j < tObj.shape_[1]; ++j) {
				_str += std::format("{}",tObj[i, j]);
				if (j != tObj.shape_[1] - 1) {
					_str += ", ";
				}
			}

			_str += " ]";

			if (i != tObj.shape_[0] - 1) {
				_str += ", ";
			}
		}

		_str += " ]";

		os << _str;
	}

	return os;
}

const std::vector<std::size_t>& Tensor::shape() const
{
	return shape_;
}

const std::vector<std::size_t>& Tensor::stride() const
{
	return stride_;
}

const std::vector<float>& Tensor::data() const
{
	return data_;
}

std::vector<float>& Tensor::data()
{
	return data_;
}

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other)
{
	// scalar + scalar
	if (shape_.size() == 0 && other->shape_.size() == 0) {
		float _result = data_[0] + other->data_[0];
		if (requiresgrad_ || other->requiresgrad_) {
			auto _self = shared_from_this();
			std::vector<std::shared_ptr<Tensor>> _parents{ _self, other };
			std::function<void(const std::vector<float>&)> _gradFn =
				[_self, other](const std::vector<float>& gradoutput) {

					{
						std::lock_guard<std::mutex> lock(_self->grad_mutex);
						_self->addToGrad({ gradoutput[0] });
					}
					{
						std::lock_guard<std::mutex> lock(other->grad_mutex);
						other->addToGrad({ gradoutput[0] });
					}
				};

			return std::make_shared<Tensor>(_result, true,std::move( _gradFn), std::move(_parents));
		}
		return std::make_shared<Tensor>(_result);
	}

	// scalar + 1D
	if (shape_.size() == 0 && other->shape_.size() == 1) {
		std::vector<float> _result{};
		_result.reserve(other->data_.size());
		for (const auto& d : other->data_) {
			_result.emplace_back(static_cast<float>(d + data_[0]));
		}

		if (requiresgrad_ || other->requiresgrad_) {
			auto _self = shared_from_this();
			std::vector<std::shared_ptr<Tensor>> _parents{ _self, other };
			std::function<void(const std::vector<float>&)> _gradFn =
				[_self, other](const std::vector<float>& gradoutput) {
				
				float _gradSelf = 0.0f;

				for (const auto& g : gradoutput) {
					_gradSelf += g;
				}

				{
					std::lock_guard<std::mutex> lock(_self->grad_mutex);
					_self->addToGrad({ _gradSelf });
				}
				{
					std::lock_guard<std::mutex> lock(other->grad_mutex);
					other->addToGrad(gradoutput);
				}

				};

			return std::make_shared<Tensor>(_result, true, _gradFn, _parents);
		}

		return std::make_shared<Tensor>(_result);
	}

	// Scalar + 2D
	if (shape_.size() == 0 && other->shape_.size() == 2) {
		std::vector<std::vector<float>> _result{};
		_result.reserve(other->shape_[0]);

		for (std::size_t i{}; i < other->shape_[0]; ++i) {
			std::vector<float> _resultI{};
			_resultI.reserve(other->shape_[1]);
			for (std::size_t j{}; j < other->shape_[1]; ++j) {
				_resultI.emplace_back(static_cast<float>(data_[0] + (*other)[i, j]));
			}
			_result.emplace_back(std::move(_resultI));
		}

		if (requiresgrad_ || other->requiresgrad_) {
			auto _self = shared_from_this();
			std::vector<std::shared_ptr<Tensor>> _parents{ _self, other };
			std::function<void(const std::vector<float>&)> _gradFn =
				[_self, other](const std::vector<float>& gradoutput) {

				float _gradSelf = 0.0f;

				for (const auto& g : gradoutput) {
					_gradSelf += g;
				}

				{
					std::lock_guard<std::mutex> lock(_self->grad_mutex);
					_self->addToGrad({ _gradSelf });
				}
				{
					std::lock_guard<std::mutex> lock(other->grad_mutex);
					other->addToGrad(gradoutput);
				}

				};

			return std::make_shared<Tensor>(_result, true, _gradFn, _parents);
		}

		return std::make_shared<Tensor>(_result);
	}

	//1D + Scalar
	if (shape_.size() == 1 && other->shape_.size() == 0) {
		std::vector<float> _result{};
		_result.reserve(data_.size());
		for (const auto& d : data_) {
			_result.emplace_back(static_cast<float>(d + other->data_[0]));
		}

		if (requiresgrad_ || other->requiresgrad_) {
			auto _self = shared_from_this();
			std::vector<std::shared_ptr<Tensor>> _parents{ _self, other };
			std::function<void(const std::vector<float>&)> _gradFn =
				[_self, other](const std::vector<float>& gradoutput) {

				float _gradSelf = 0.0f;

				for (const auto& g : gradoutput) {
					_gradSelf += g;
				}

				{
					std::lock_guard<std::mutex> lock(_self->grad_mutex);
					_self->addToGrad(gradoutput);
				}
				{
					std::lock_guard<std::mutex> lock(other->grad_mutex);
					other->addToGrad({ _gradSelf });
				}

				};

			return std::make_shared<Tensor>(_result, true, _gradFn, _parents);
		}

		return std::make_shared<Tensor>(_result);
	}

	// 2D + Scalar
	if (shape_.size() == 2 && other->shape_.size() == 0) {
		std::vector<std::vector<float>> _result{};
		_result.reserve(shape_[1]);

		for (std::size_t i{}; i < shape_[1]; ++i) { // TODO
			std::vector<float> _resultI{};
			_result[i].reserve(shape_[0]);
			for (std::size_t j{}; j < shape_[0]; ++j) {
				_resultI.emplace_back(static_cast<float>(other->data_[0] + operator[](i, j)));
			}
			_result.emplace_back(std::move(_resultI));
		}

		if (requiresgrad_ || other->requiresgrad_)
		{
			std::shared_ptr<Tensor> self = shared_from_this();
			std::vector<std::shared_ptr<Tensor>> parents{ self, other };
			std::function<void(const std::vector<float>&)> gradfn =
				[self, other](const std::vector<float>& grad_output)
				{
					// propagate child grad
					
					// broadcast in forward == sum in backward
					float grad_other = 0.0f;
					for (std::size_t i = 0; i < grad_output.size(); i++)
					{
						grad_other += grad_output[i];
					}

					{
						std::lock_guard<std::mutex> lock(self->grad_mutex);
						self->addToGrad(grad_output);
					}
					{
						std::lock_guard<std::mutex> lock(other->grad_mutex);
						other->addToGrad({ grad_other });
					}
				};
			return std::make_shared<Tensor>(_result, true, gradfn, parents);
		}

		return std::make_shared<Tensor>(_result);
	}

	// 1D + 1D
	if (shape_[0] != other->shape_[0]) {
		throw Error(std::format("First dimention mismatch. {} vs {} in Tensor class", 
			shape_[0], other->shape_[0]));
	}

	if (shape_.size() == 1) {
		const std::size_t N = shape_[0];

		// 1. Instantly allocate exact memory size upfront 
		std::vector<float> _result(N);

		// 2. Fetch raw pointers to bypass vector tracking overhead inside threads
		const float* const self_ptr = data_.data();
		const float* const other_ptr = other->data_.data();
		float* const res_ptr = _result.data();

		// 3. Parallel Vector Addition
		std::transform(
			std::execution::par,       // Enable multi-threading
			self_ptr,                  // Start of first array
			self_ptr + N,              // End of first array
			other_ptr,                 // Start of second array
			res_ptr,                   // Destination output location
			std::plus<float>()         // Operation: res_ptr[i] = self_ptr[i] + other_ptr[i]
		);

		if (requiresgrad_ || other->requiresgrad_) {
			auto _self = shared_from_this();
			std::vector<std::shared_ptr<Tensor>> _parents{ _self, other };
			std::function<void(const std::vector<float>&)> _gradFn =
				[_self, other](const std::vector<float>& gradoutput) {

				{
					std::lock_guard<std::mutex> lock(_self->grad_mutex);
					_self->addToGrad(gradoutput);
				}
				{
					std::lock_guard<std::mutex> lock(other->grad_mutex);
					other->addToGrad(gradoutput);
				}

				};

			return std::make_shared<Tensor>(std::move(_result), true, std::move(_gradFn), std::move(_parents));
		}

		return std::make_shared<Tensor>(std::move(_result));
	}
	else {
		if (shape_[1] != other->shape_[1]) {
			throw Error(std::format("Second dimension mismatch. {} vs {} in Tensor class", 
				shape_[1], other->shape_[1]));
		}

		std::vector<std::vector<float>> _result{};
		_result.reserve(shape_[0]);

		for (std::size_t i{}; i < shape_[0]; ++i) {
			std::vector<float> _resultI{};
			_resultI.reserve(shape_[1]);
			for (std::size_t j{}; j < shape_[1]; ++j) {
				_resultI.emplace_back(static_cast<float>(operator[](i, j) + (*other)[i, j]));
			}
			_result.emplace_back(std::move(_resultI));
		}

		if (requiresgrad_ || other->requiresgrad_) {
			auto _self = shared_from_this();
			std::vector<std::shared_ptr<Tensor>> _parents{ _self, other };
			std::function<void(const std::vector<float>&)> _gradFn =
				[_self, other](const std::vector<float>& gradoutput) {

				{
					std::lock_guard<std::mutex> lock(_self->grad_mutex);
					_self->addToGrad(gradoutput);
				}
				{
					std::lock_guard<std::mutex> lock(other->grad_mutex);
					other->addToGrad(gradoutput);
				}

				};

			return std::make_shared<Tensor>(_result, true, _gradFn, _parents);
		}

		return std::make_shared<Tensor>(_result);
	}
}

std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other)
{
	if (shape_.size() == 0 || other->shape_.size() == 0) {
		throw Error("Matrix multiplication is not available for Linear tensors");
	}

	if (shape_[shape_.size() - 1] != other->shape_[0]) {
		throw Error("Last dimension of first tensor doesn't match the first dimension of fist tensor");
	}
	// 1D * 1D
	if (shape_.size() == 1 && other->shape_.size() == 1) {
		//float _result{};
		//for (std::size_t i{}; i < shape_[0]; ++i) {
		//	_result += data_[i] * other->data_[i];
		//}

		const std::size_t _n = shape_[0];

		float _result = std::transform_reduce(
			std::execution::par,
			data_.data(),
			data_.data() + _n,
			other->data_.data(),
			0.0f,
			std::plus<>(),
			std::multiplies<>()
		);

		if (requiresgrad_ || other->requiresgrad_) {
			auto _self = shared_from_this();
			std::vector<std::shared_ptr<Tensor>> _parent{ _self, other };
			const std::size_t _count = _self->numElement();
			std::function<void(const std::vector<float>&)> _gradFn =
				[_self, other, _count](const std::vector<float>& gradOutput) {
				std::vector<float> _gradSelf{};
				_gradSelf.reserve(_count);
				std::vector<float> _gradOther{};
				_gradOther.reserve(_count);
				const float _g0 = gradOutput[0];
				for (std::size_t i{}; i < _count; ++i) {
					_gradSelf.emplace_back(static_cast<float>((*other)[i] * _g0));
					_gradOther.emplace_back(static_cast<float>((*_self)[i] * _g0));
				}

				{
					std::lock_guard<std::mutex> lock(_self->grad_mutex);
					_self->addToGrad(std::move(_gradSelf));
				}
				{
					std::lock_guard<std::mutex> lock(other->grad_mutex);
					other->addToGrad(std::move(_gradOther));
				}
				};

			return std::make_shared<Tensor>(_result, true, std::move(_gradFn), std::move(_parent));
		}

		return std::make_shared<Tensor>(_result);
	}
	else if (shape_.size() == 2 && other->shape_.size() == 1) {
		// 2D * 1D
		std::vector<float> _result{};
		_result.reserve(shape_[0]);
		for (std::size_t i{}; i < shape_[0]; ++i) {
			float _res{};
			for (std::size_t j{}; j < shape_[1]; ++j) {
				_res += operator[](i, j) * other->data_[j];
			}

			_result.emplace_back(_res);
		}

		if (requiresgrad_ || other->requiresgrad_) {
			auto _self = shared_from_this();
			std::vector<std::shared_ptr<Tensor>> _parent{ _self, other };
			std::function<void(const std::vector<float>&)> _gradFn =
				[_self, other](const std::vector<float>& gradOutput) {

				std::vector<float> _gradSelf{};
				_gradSelf.reserve(_self->shape_[0] * _self->shape_[1]);

				for (std::size_t i{}; i < _self->shape_[0]; ++i) {
					for (std::size_t j{}; j < _self->shape_[1]; ++j) {
						_gradSelf.emplace_back(static_cast<float>((*other)[j] * gradOutput[i]));
					}
				}

				std::vector<float> _gradOther{};
				_gradOther.reserve(other->shape_[0]);

				for (std::size_t i{}; i < other->shape_[0]; ++i) {
					float _gradOtherI{};
					for (std::size_t j{}; j < _self->shape_[0]; ++j) {
						_gradOtherI += (*_self)[j, i] * gradOutput[j];
					}

					_gradOther.emplace_back(_gradOtherI);
				}

				{
					std::lock_guard<std::mutex> lock(_self->grad_mutex);
					_self->addToGrad(std::move(_gradSelf));
				}
				{
					std::lock_guard<std::mutex> lock(other->grad_mutex);
					other->addToGrad(std::move(_gradOther));
				}
				};

			return std::make_shared<Tensor>(_result, true, _gradFn, _parent);
		}

		return std::make_shared<Tensor>(_result);
	}
	else if(shape_.size() == 1 && other->shape_.size() == 2){ // todo
		// 1D * 2D
		const std::size_t K = shape_[0];         // Vector length / Matrix rows
		const std::size_t N = other->shape_[1];  // Matrix columns

		// 1. Instantly allocate exact memory upfront (Required for parallel indexing)
		std::vector<float> _result(N);

		// 2. Create a column index tracking vector [0, 1, 2, ..., N-1]
		std::vector<std::size_t> col_indices(N);
		std::iota(col_indices.begin(), col_indices.end(), 0);

		const float* const self_ptr = data_.data();        // Vector pointer
		const float* const other_ptr = other->data_.data(); // Flat 2D matrix pointer

		// 3. Parallelize across columns
		std::for_each(std::execution::par, col_indices.begin(), col_indices.end(), [&](std::size_t i) {
			float _res = 0.0f;

			// Inner loop: Must handle the stride using flat pointer math
			for (std::size_t j = 0; j < K; ++j) {
				// row 'j' * total_columns 'N' + current_column 'i'
				_res += self_ptr[j] * other_ptr[j * N + i];
			}

			_result[i] = _res; // Thread-safe: 'i' is unique per thread task
			});

		if (requiresgrad_ || other->requiresgrad_)
		{
			std::shared_ptr<Tensor> self = shared_from_this();
			std::vector<std::shared_ptr<Tensor>> parents{ self, other };

			auto gradfn = [self, other](const std::vector<float>& grad_output)
				{
					// Gather structural dimensions
					const std::size_t rows_self = self->shape()[0];  // K
					const std::size_t rows_other = other->shape()[0]; // K
					const std::size_t cols_other = other->shape()[1]; // N

					// Extract raw flat pointers for high-performance streaming inside threads
					const float* const self_ptr = self->data().data();
					const float* const other_ptr = other->data().data();
					const float* const grad_out_ptr = grad_output.data();

					// =================================================================
					// PART 1: Parallelize grad_self Computation (Matrix-Vector MatMul)
					// =================================================================
					// Allocate exact memory size upfront
					std::vector<float> grad_self(rows_self);

					// Create a row tracking index vector [0, 1, 2, ..., rows_self-1]
					std::vector<std::size_t> row_indices_self(rows_self);
					std::iota(row_indices_self.begin(), row_indices_self.end(), 0);

					// Parallel execution across self rows
					std::for_each(std::execution::par, row_indices_self.begin(), row_indices_self.end(), [&](std::size_t i) {
						float grad_self_i = 0.0f;
						const std::size_t row_offset = i * cols_other;

						// Horizontal row-by-row streaming (Cache-friendly)
						for (std::size_t j = 0; j < cols_other; ++j) {
							grad_self_i += other_ptr[row_offset + j] * grad_out_ptr[j];
						}
						grad_self[i] = grad_self_i; // Safe: Index 'i' is unique to this thread
						});

					// =================================================================
					// PART 2: Parallelize grad_other Computation (Outer Product)
					// =================================================================
					// Allocate full flat layout size upfront
					std::vector<float> grad_other(rows_other * cols_other);

					// Create a row tracking index vector for other matrix [0, 1, 2, ..., rows_other-1]
					std::vector<std::size_t> row_indices_other(rows_other);
					std::iota(row_indices_other.begin(), row_indices_other.end(), 0);

					// Parallel execution across other rows
					std::for_each(std::execution::par, row_indices_other.begin(), row_indices_other.end(), [&](std::size_t i) {
						const float self_val = self_ptr[i];
						const std::size_t row_offset = i * cols_other;

						// Fill the current row of grad_other in parallel
						for (std::size_t j = 0; j < cols_other; ++j) {
							grad_other[row_offset + j] = self_val * grad_out_ptr[j]; // Safe: row_offset + j is unique
						}
						});

					// Move gradients back into the Tensors
					{
						std::lock_guard<std::mutex> lock(self->grad_mutex);
						self->addToGrad(std::move(grad_self));
					}
					{
						std::lock_guard<std::mutex> lock(other->grad_mutex);
						other->addToGrad(std::move(grad_other));
					}
				};
			return std::make_shared<Tensor>(std::move(_result), true, std::move(gradfn), std::move(parents));
		}

		return std::make_shared<Tensor>(std::move(_result));
	}
	else {
		if (other->shape_.size() < 2) {
			throw Error("Expected second tensor to have 2 dimensions for the operations of Tensors");
		}

		std::vector<std::vector<float>> _result{};
		_result.reserve(shape_[0]);

		for (std::size_t i{}; i < shape_[0]; ++i) {
			std::vector<float> _resultI{};
			_resultI.reserve(other->shape_[1]);
			for (std::size_t j{}; j < other->shape_[1]; ++j) {
				float _resultIJ{};
				for (std::size_t k{}; k < shape_[1]; ++k) {
					_resultIJ += operator[](i, k) * (*other)[k, j];
				}
				_resultI.emplace_back(_resultIJ);
			}
			_result.emplace_back(std::move(_resultI));
		}

		if (requiresgrad_ || other->requiresgrad_) {
			auto _self = shared_from_this();
			std::vector<std::shared_ptr<Tensor>> _parent{ _self, other };
			std::function<void(const std::vector<float>&)> _gradFn =
				[_self, other](const std::vector<float>& gradOutput) {

				std::vector<float> _gradSelf{};
				_gradSelf.reserve(_self->shape_[0] * _self->shape_[1]);

				for (std::size_t i{}; i < _self->shape_[0]; ++i) {
					for (std::size_t j{}; j < _self->shape_[1]; ++j) {

						float _gradSelfIJ{};

						for (std::size_t k{}; k < other->shape_[1]; ++k) {
							_gradSelfIJ += (*other)[j, k] * gradOutput[i * other->shape_[1] + k];
						}
						_gradSelf.emplace_back(_gradSelfIJ);
					}
				}

				std::vector<float> _gradOther{};
				_gradOther.reserve(other->shape_[0] * other->shape_[1]);

				for (std::size_t i{}; i < other->shape_[0]; ++i) {
					for (std::size_t j{}; j < other->shape_[1]; ++j) {

						float _gradSelfIJ{};

						for (std::size_t k{}; k < _self->shape_[0]; ++k) {
							_gradSelfIJ += (*_self)[k, i] * gradOutput[k * other->shape_[1] + j];
						}
						_gradOther.emplace_back(_gradSelfIJ);
					}
				}

				{
					std::lock_guard<std::mutex> lock(_self->grad_mutex);
					_self->addToGrad(std::move(_gradSelf));
				}
				{
					std::lock_guard<std::mutex> lock(other->grad_mutex);
					other->addToGrad(std::move(_gradOther));
				}
				};

			return std::make_shared<Tensor>(_result, true, _gradFn, _parent);
		}

		return std::make_shared<Tensor>(_result);
	}
}

const bool& Tensor::requiresGrad() const
{
	return requiresgrad_;
}

const std::vector<float>& Tensor::grad() const
{
	return grad_;
}

std::vector<float>& Tensor::grad()
{
	return grad_;
}

void Tensor::addToGrad(const std::vector<float>& gradUpdate)
{
	if (!requiresgrad_) {
		return;
	}

	if (grad_.size() != gradUpdate.size()) {
		throw Error("Gradiant size mismatch for Tensors");
	}

	//for (std::size_t i{}; i < gradUpdate.size(); ++i) {
	//	grad_[i] += gradUpdate[i];
	//}

	std::transform(
		std::execution::par,
		gradUpdate.begin(),
		gradUpdate.end(),
		grad_.begin(),
		grad_.begin(),
		std::plus<float>()
	);
}

void Tensor::zeroGrad()
{
	grad_ = std::vector<float>(data_.size(), 0.0f);
}

std::size_t Tensor::numElement() const
{
	return data_.size();
}

//void Tensor::backward()
//{
//	if (!requiresgrad_) {
//		throw Error("Element doesnt require grad for Tensors");
//	}
//
//	if (shape_.size() != 0) {
//		throw Error("Grad can only be calculated for scalar outputs for Tensors");
//	}
//
//	resetGraphVisit();
//	grad_ = { 1.0 };
//
//	backward_();
//}

std::size_t Tensor::argmax() const
{
	return std::distance(data_.begin(), std::max_element(data_.begin(), data_.end()));
}

std::mutex& Tensor::gradMutex() 
{
	return grad_mutex;
}

//void Tensor::backward_()
//{
//	if (!requiresgrad_) {
//		return;
//	}
//
//	if (visited_) {
//		return;
//	}
//
//	visited_ = true;
//
//	if (gradFn_) {
//		gradFn_(grad_);
//	}
//
//	for (auto& parent : parents_) {
//		parent->backward_();
//	}
//}
//
//void Tensor::resetGraphVisit()
//{
//	if (!visited_) {
//		return;
//	}
//
//	visited_ = false;
//
//	for (std::size_t i{}; i < parents_.size(); ++i) {
//		parents_[i]->resetGraphVisit();
//	}
//}

// Phase 1: Count how many downstream nodes (children) depend on each parent

// Helper to build dependency counts accurately across the entire DAG
void Tensor::countDependencies()
{
	// Use visited_ to ensure we only process each unique node once during counting
	if (visited_) {
		return;
	}
	visited_ = true;

	for (auto& parent : parents_) {
		// Increment how many children depend on this parent
		parent->pending_dependencies.fetch_add(1, std::memory_order_relaxed);

		// Recurse down to grandparents
		parent->countDependencies();
	}
}

// Helper to clear the visited flags after counting is done
void Tensor::clearVisitedFlags()
{
	if (!visited_) {
		return;
	}
	visited_ = false;
	for (auto& parent : parents_) {
		parent->clearVisitedFlags();
	}
}

void Tensor::parallel_backward_step(std::vector<std::future<void>>& shared_futures, std::mutex& futures_mutex)
{
	// 1. Run gradient function (Keep this outside of locks!)
	if (gradFn_) {
		gradFn_(grad_);
	}

	// 2. Notify parents
	for (auto& parent : parents_) {
		// fetch_sub returns the PREVIOUS value. If it was 1, it becomes 0 (ready).
		if (parent->pending_dependencies.fetch_sub(1, std::memory_order_acq_rel) == 1) {

			if (!parent->requiresgrad_) {
				continue;
			}

			// OPTIMIZATION A: Prepare the async task outside of the lock.
			// This avoids holding the lock during the heavy OS thread-creation phase.
			auto future_task = std::async(std::launch::async, [p = parent, &shared_futures, &futures_mutex]() {
				p->parallel_backward_step(shared_futures, futures_mutex);
				});

			// OPTIMIZATION B: Only lock for the microsecond it takes to push to the vector
			{
				std::lock_guard<std::mutex> lock(futures_mutex);
				shared_futures.push_back(std::move(future_task));
			}
		}
	}
}
void Tensor::backward()
{
	if (!requiresgrad_) {
		throw std::runtime_error("Element doesn't require grad for Tensors");
	}

	if (shape_.size() != 0) {
		throw std::runtime_error("Grad can only be calculated for scalar outputs for Tensors");
	}

	// Phase 1: Cleanly count dependencies across the entire graph structure
	clearVisitedFlags(); // Ensure fresh state
	countDependencies();
	clearVisitedFlags(); // Clean up flags so they are ready for the next backward() call

	// Phase 2: Initialize root gradient
	grad_ = { 1.0f };

	// Phase 3: Prepare tracking infrastructure for our threads
	std::vector<std::future<void>> shared_futures;
	std::mutex futures_mutex;

	// Phase 4: Start the parallel execution chain
	parallel_backward_step(shared_futures, futures_mutex);

	// Phase 5: CRITICAL BARRIER
	// Loop continuously until all dynamically spawned tasks have completed.
	// This stops the training loop from advancing to optimizer.step() too early!
	bool tasks_remaining = true;
	while (tasks_remaining) {
		std::future<void> next_task;
		{
			std::lock_guard<std::mutex> lock(futures_mutex);
			if (!shared_futures.empty()) {
				next_task = std::move(shared_futures.back());
				shared_futures.pop_back();
			}
			else {
				tasks_remaining = false;
			}
		}
		if (next_task.valid()) {
			next_task.get(); // Blocks until this specific branch task is completely done
			tasks_remaining = true; // Check if worker threads added more ready parents
		}
	}
}
