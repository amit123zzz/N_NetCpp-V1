#pragma once

#include"../Tensor/Tensor.h"

#include<filesystem>
#include<unordered_map>
#include<string>
#include<memory>
#include<vector>
#include<iostream>
#include<fstream>

const std::uint32_t MAGIC_NUM = 777;

inline void save(const std::unordered_map<std::string, std::shared_ptr<Tensor>>& stateDict,
	const std::string& filename) {
	std::ofstream _file{ filename, std::ios::binary };
	_file.write(reinterpret_cast<const char*>(&MAGIC_NUM), sizeof(std::uint32_t));

	for (const auto& [name, w] : stateDict) {
		std::size_t _nameLen = name.length();
		_file.write(reinterpret_cast<const char*>(&_nameLen), sizeof(std::size_t));
		_file.write(name.data(), _nameLen);

		std::size_t _shapeLength = w->shape().size();
		_file.write(reinterpret_cast<const char*>(&_shapeLength), sizeof(std::size_t));
		_file.write(reinterpret_cast<const char*>(w->shape().data()), _shapeLength * sizeof(std::size_t));

		std::size_t _dataLength = w->numElement();
		_file.write(reinterpret_cast<const char*>(&_dataLength), sizeof(std::size_t));
		_file.write(reinterpret_cast<const char*>(w->data().data()), _dataLength * sizeof(float));
	}
}

inline std::unordered_map<std::string, std::shared_ptr<Tensor>> load(const std::string& filename)
{
    std::unordered_map<std::string, std::shared_ptr<Tensor>> state_dict;
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open " + filename);
    }

    int magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(int));
    if (magic != MAGIC_NUM)
    {
        throw std::runtime_error("Bad fle format: wrong magic number");
    }

    while (file.peek() != EOF)
    {
        size_t name_len = 0;
        if (!file.read(reinterpret_cast<char*>(&name_len), sizeof(size_t)))
            break;

        std::string weight_name(name_len, '\0');
        file.read(weight_name.data(), name_len);

        size_t shape_length = 0;
        file.read(reinterpret_cast<char*>(&shape_length), sizeof(size_t));

        std::vector<size_t> shape(shape_length);
        file.read(reinterpret_cast<char*>(shape.data()), shape_length * sizeof(size_t));

        size_t data_length = 0;
        file.read(reinterpret_cast<char*>(&data_length), sizeof(size_t));

        std::vector<float> raw(data_length);
        file.read(reinterpret_cast<char*>(raw.data()), data_length * sizeof(float));

        std::shared_ptr<Tensor> tensor;

        if (shape_length == 0)
        {
            tensor = std::make_shared<Tensor>(raw[0]);
        }
        else if (shape_length == 1)
        {
            tensor = std::make_shared<Tensor>(raw);
        }
        else if (shape_length == 2)
        {
            std::vector<std::vector<float>> data_2d(shape[0], std::vector<float>(shape[1]));
            for (size_t i = 0; i < shape[0]; i++)
            {
                for (size_t j = 0; j < shape[1]; j++)
                {
                    data_2d[i][j] = raw[i * shape[1] + j];
                }
            }
            tensor = std::make_shared<Tensor>(data_2d);
        }
        else
        {
            throw std::runtime_error("Unsupported tensor dimensionality: " +
                std::to_string(shape_length));
        }

        state_dict[weight_name] = tensor;
    }

    return state_dict;
}
