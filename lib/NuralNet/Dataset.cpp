#include "Dataset.h"

#include<fstream>
#include<filesystem>
#include<numeric>
#include<algorithm>
#include<execution>

MNIST::MNIST(const std::string& dataPath, const std::string& labelPath)
{
	readMNIST(dataPath);
	readMNISTLabels(labelPath);
}

std::pair<std::int32_t, std::shared_ptr<Tensor>> MNIST::getItem(std::uint32_t index)
{
	return std::make_pair(
		labels_[index], std::make_shared<Tensor>(images_[index])
		);
}

std::int32_t MNIST::getLength()
{
	return images_.size();
}

std::string MNIST::labelToClass(std::uint32_t label) const
{
	return classes_[label];
}

void MNIST::readMNIST(const std::string& path)
{
	std::fstream _file{ path, std::ios::in | std::ios::binary };
	if (!_file) {
		throw Error(std::format("Cannot open file {}", path));
	}

	std::int32_t _magicNumber{};
	std::int32_t _numberOfImages{};
	std::int32_t _rows{};
	std::int32_t _columns{};

	std::int32_t _arr[4]{};

	_file.read(reinterpret_cast<char*>(_arr), 4 * sizeof(std::int32_t));
	
	_magicNumber = reverseInt(_arr[0]);

	if (_magicNumber != 2051) {
		throw Error("Invalid mnist form.");
	}

	_numberOfImages = reverseInt(_arr[1]);
	_rows = reverseInt(_arr[2]);
	_columns = reverseInt(_arr[3]);

	std::size_t _pixelAmount = static_cast<std::size_t>(_numberOfImages) * _rows * _columns;
	std::vector<unsigned char> _buffer(_pixelAmount);
	_file.read(reinterpret_cast<char*>(_buffer.data()), _pixelAmount);

	//images_.reserve(_numberOfImages);

	//std::size_t _bufferIdx{};

	//for (std::size_t i{}; i < _numberOfImages; ++i) {
	//	std::vector<std::vector<float>> _image(_rows, std::vector<float>(_columns));
	//	for (std::size_t r{}; r < _rows; ++r) {
	//		for (std::size_t c{}; c < _columns; ++c) {
	//			_image[r][c] = convertToFloat(_buffer[_bufferIdx++]);
	//		}
	//	}
	//	images_.emplace_back(std::move(_image));
	//}
	// Create an index sequence [0, 1, 2, ..., _numberOfImages - 1]

	images_.resize(_numberOfImages);

	std::vector<std::size_t> indices(_numberOfImages);
	std::iota(indices.begin(), indices.end(), 0);

	std::size_t _imageSize = static_cast<std::size_t>(_rows) * _columns;

	// Parallelize processing across all CPU cores
	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i) {
		std::vector<std::vector<float>> _image(_rows, std::vector<float>(_columns));

		// Calculate exactly where this image's chunk starts in the bulk buffer
		std::size_t imageStartIdx = i * _imageSize;

		for (std::size_t r{}; r < _rows; ++r) {
			std::size_t rowStartIdx = imageStartIdx + (r * _columns);
			for (std::size_t c{}; c < _columns; ++c) {
				_image[r][c] = convertToFloat(_buffer[rowStartIdx + c]);
			}
		}

		// Safe to assign directly because the outer vector elements are independent
		images_[i] = std::move(_image);
		});
}

void MNIST::readMNISTLabels(const std::string& path)
{
	std::ifstream _file{ path, std::ios::in | std::ios::binary };
	if (!_file) {
		throw Error(std::format("Cannot open file {}", path));
	}

	std::int32_t _arr[2]{};
	_file.read(reinterpret_cast<char*>(_arr), 2 * sizeof(std::int32_t));

	std::int32_t _magicNumber = reverseInt(_arr[0]);
	if (_magicNumber != 2049) {
		throw Error("Invalid MNIST label file");
	}

	std::int32_t _numOfItems = reverseInt(_arr[1]);

	// 1. Allocate a temporary buffer to hold the 1-byte raw labels
	std::vector<unsigned char> tempBuffer(_numOfItems);
	_file.read(reinterpret_cast<char*>(tempBuffer.data()), _numOfItems);

	// 2. Prepare the destination vector
	labels_.resize(_numOfItems);

	// 3. Bulk convert/copy the 1-byte chars into 4-byte int32_t values
	std::copy(tempBuffer.begin(), tempBuffer.end(), labels_.begin());
	
}

std::int32_t MNIST::reverseInt(std::int32_t num) const
{
	unsigned char _c1, _c2, _c3, _c4;

	_c1 = num & 255;
	_c2 = (num >> 8) & 255;
	_c3 = (num >> 16) & 255;
	_c4 = (num >> 24) & 255;

	return ((std::int32_t)_c1 << 24) + ((std::int32_t)_c2 << 16) + ((std::int32_t)_c3 << 8) +
		(std::int32_t)_c4;
}

float MNIST::convertToFloat(unsigned char c) const
{
	return static_cast<float>(c) / 255.0f;
}
