#include "DataLoader.h"

#include<algorithm>
#include<numeric>

#include<random>

DataLoader::DataLoader(Dataset* dataSet, std::int32_t batchSize, bool shuffle):
	dataSet_{dataSet}, batchSize_{batchSize}
{
	indices_.resize(dataSet_->getLength());
	std::iota(indices_.begin(), indices_.end(), 0);

	if (shuffle) {
		std::random_device _rd{};
		std::mt19937 _mt{ _rd() };
		std::shuffle(indices_.begin(), indices_.end(), _mt);
	}
}

DataLoader::Iterator DataLoader::begin()
{
	return Iterator(this, 0);
}

DataLoader::Iterator DataLoader::end()
{
	return Iterator(this, dataSet_->getLength());
}

std::size_t DataLoader::batchSize() const
{
	return batchSize_;
}

std::size_t DataLoader::amountSamples() const
{
	return dataSet_->getLength();
}

std::size_t DataLoader::amountBatches() const
{
	return (dataSet_->getLength() + batchSize_ - 1) / batchSize_;
}

DataLoader::Iterator::Iterator(DataLoader* dl, std::int32_t index):
	dataLoader_{dl}, index_{index}
{
}

void DataLoader::Iterator::operator++()
{
	index_ += dataLoader_->batchSize_;
}

std::vector<std::pair<std::int32_t, std::shared_ptr<Tensor>>> DataLoader::Iterator::operator*()
{
	std::vector<std::pair<std::int32_t, std::shared_ptr<Tensor>>> _batch{};

	for (std::size_t i{}; i < dataLoader_->batchSize_; ++i) {
		_batch.emplace_back(
			dataLoader_->dataSet_->getItem(dataLoader_->indices_[index_ + i])
		);
	}

	return _batch;
}

bool DataLoader::Iterator::operator!=(const Iterator& other)
{
	return index_ != other.index_;
}
