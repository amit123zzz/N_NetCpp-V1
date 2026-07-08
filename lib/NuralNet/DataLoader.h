#pragma once

#include"Dataset.h"
#include<memory>
#include<utility>
#include<vector>
#include<cstdint>

class DataLoader
{
public:
	DataLoader(Dataset* dataSet, std::int32_t batchSize, bool shuffle = true);

	class Iterator {
	public:
		Iterator(DataLoader* dl, std::int32_t index);
		void operator++();
		std::vector<std::pair<std::int32_t, std::shared_ptr<Tensor>>> operator*();
		bool operator!=(const Iterator& other);

	private:
		DataLoader* dataLoader_;
		std::int32_t index_;
	};

	Iterator begin();
	Iterator end();

	std::size_t batchSize() const;
	std::size_t amountSamples() const;
	std::size_t amountBatches() const;

private:
	Dataset* dataSet_;
	std::int32_t batchSize_;
	std::vector<std::int32_t> indices_;
};

