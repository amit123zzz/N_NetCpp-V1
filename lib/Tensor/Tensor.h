#pragma once

#include<vector>

#include<iostream>

#include<string>
#include<memory>

#include<functional>

#include"../err/Error.hpp"

#include<thread>
#include<queue>
#include<mutex>
#include<atomic>
#include<future>

class Tensor : public std::enable_shared_from_this<Tensor>
{
public:
	Tensor(float, bool requiresGrad = false,
		std::function<void(const std::vector<float>&)> gradFn = nullptr,
		std::vector<std::shared_ptr<Tensor>> parents = {});

	Tensor(const std::vector<float>&, bool requiresGrad = false,
		std::function<void(const std::vector<float>&)> gradFn = nullptr,
		std::vector<std::shared_ptr<Tensor>> parents = {});

	Tensor(const std::vector<std::vector<float>>&, bool requiresGrad = false,
		std::function<void(const std::vector<float>&)> gradFn = nullptr,
		std::vector<std::shared_ptr<Tensor>> parents = {});

	const float& item() const;

	float& item();

	const float& operator[](std::size_t index) const;
	float& operator[](std::size_t index);
	const float& operator[](std::size_t index1, std::size_t index2) const;
	float& operator[](std::size_t index1, std::size_t index2);

	friend std::ostream& operator<<(std::ostream& os, const Tensor& tObj);

	const std::vector<std::size_t>& shape() const;
	const std::vector<std::size_t>& stride() const;
	const std::vector<float>& data() const;
	std::vector<float>& data();

	std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
	std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other);

	const bool& requiresGrad() const;
	const std::vector<float>& grad() const;
	std::vector<float>& grad();
	void addToGrad(const std::vector<float>& gradUpdate);

	void zeroGrad();
	std::size_t numElement() const;

	void backward();

	std::size_t argmax() const;

	std::mutex& gradMutex();

private:
	std::vector<float> data_;
	std::vector<std::size_t> shape_;
	std::vector<std::size_t> stride_;
	std::vector<float> grad_;
	std::function<void(const std::vector<float>&)> gradFn_;
	std::vector < std::shared_ptr<Tensor>> parents_;
	bool requiresgrad_;
	void backward_();
	bool visited_ = false;

	void resetGraphVisit();
	void countDependencies();
	void clearVisitedFlags();
	void parallel_backward_step(std::vector<std::future<void>>& shared_futures, std::mutex& futures_mutex);

	std::atomic<int> pending_dependencies{ 0 }; // How many children we are waiting for
	std::mutex grad_mutex;
};
