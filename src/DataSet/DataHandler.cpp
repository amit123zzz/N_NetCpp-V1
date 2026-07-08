#include "DataHandler.h"

#include<print>

DataHandler::DataHandler()
{
    dataSet_ = new std::vector<Data*>;
    testData_ = new std::vector<Data*>;
    trainingData_ = new std::vector<Data*>;
    validationData_ = new std::vector<Data*>;
}

DataHandler::~DataHandler()
{

}

void DataHandler::readFeatureVectors(const std::string& path)
{
    std::uint32_t _header[4]{};
    unsigned char _bytes[4]{};

    std::ifstream _file{ path };
    if (!_file) {
        std::print("{} is not found. \n", path);
        exit(1);
    }

    for (std::uint32_t i{}; i < 4; ++i) {
        if (_file.read(reinterpret_cast<char*>(_bytes), sizeof(_bytes))) {
            _header[i] = convertToLittleEndian(_bytes);
        }
    }

    std::print("Data read from file {}\n", path);

    std::uint32_t _imageSize = _header[2] * _header[3];

    for (std::uint32_t i{}; i < _header[1]; ++i) {
        Data* _d = new Data();
        uint8_t _element[1]{};
        for (std::uint32_t j{}; j < _imageSize; ++j) {
            if (_file.read(reinterpret_cast<char*>(_element), sizeof(_element))) {
                _d->appendToFeatureVector(_element[0]);
            }
            else {
                std::print("Error reading from file {}\n", path);
                exit(1);
            }
        }

        dataSet_->emplace_back(_d);
    }

    std::print("successfully read feature vectors\n");
}

void DataHandler::readFeatureLabels(const std::string& path)
{
    std::uint32_t _header[2]{};
    unsigned char _bytes[2]{};

    std::ifstream _file{ path };
    if (!_file) {
        std::print("{} is not found. \n", path);
        exit(1);
    }

    for (std::uint32_t i{}; i < 2; ++i) {
        if (_file.read(reinterpret_cast<char*>(_bytes), sizeof(_bytes))) {
            _header[i] = convertToLittleEndian(_bytes);
        }
    }

    std::print("Data read from file {}\n", path);

    for (std::uint32_t i{}; i < _header[1]; ++i) {
        uint8_t _element[1]{};
        if (_file.read(reinterpret_cast<char*>(_element), sizeof(_element))) {
            dataSet_->at(i)->setlabel(_element[0]);
        }
        else {
            std::print("Error reading from file {}\n", path);
            exit(1);
        }

    }

    std::print("successfully read and stored feature labels\n");
}

void DataHandler::splitData()
{
    std::unordered_set<std::int32_t> _usedIndices{};

    std::int32_t _trainSize = dataSet_->size() * TRAIN_SET_PERCENT;
    std::int32_t _testSize = dataSet_->size() * TEST_SET_PERCENT;
    std::int32_t _validSize = dataSet_->size() * VALIDATION_PERCENT;

    // Train data
    std::int32_t _count{};
    while (_count < _trainSize) {
        std::int32_t _randomIdx = std::rand() % dataSet_->size();
        if (_usedIndices.find(_randomIdx) == _usedIndices.end()) {
            trainingData_->emplace_back(dataSet_->at(_randomIdx));
            _usedIndices.insert(_randomIdx);
            _count++;
        }
    }

    // Test data
    _count = 0;
    while (_count < _testSize) {
        std::int32_t _randomIdx = std::rand() % dataSet_->size();
        if (_usedIndices.find(_randomIdx) == _usedIndices.end()) {
            testData_->emplace_back(dataSet_->at(_randomIdx));
            _usedIndices.insert(_randomIdx);
            _count++;
        }
    }

    // Train data
    _count = 0;
    while (_count < _validSize) {
        std::int32_t _randomIdx = std::rand() % dataSet_->size();
        if (_usedIndices.find(_randomIdx) == _usedIndices.end()) {
            validationData_->emplace_back(dataSet_->at(_randomIdx));
            _usedIndices.insert(_randomIdx);
            _count++;
        }
    }

    std::print("Training Data : {}, Test Data : {}, Validation Data : {}", 
        trainingData_->size(), testData_->size(), validationData_->size());
}

void DataHandler::countClasses()
{
    std::uint32_t _count{};

    for (std::uint32_t i{}; i < dataSet_->size(); ++i) {
        if (classMap_.find(dataSet_->at(i)->getLabel()) == classMap_.end()) {
            classMap_[dataSet_->at(i)->getLabel()] = _count;
            dataSet_->at(i)->setEnumeratedLabel(_count);
            _count++;
        }
    }

    numClasses_ = _count;

    std::print("Successfully extracted {} unique classes", numClasses_);
}

std::uint32_t DataHandler::convertToLittleEndian(const unsigned char* bytes)
{
    return std::uint32_t((bytes[0] << 24) | (bytes[1] << 16) |
                (bytes[2] << 8) | (bytes[3]));
}

std::vector<Data*>* DataHandler::getTrainingData() const
{
    return trainingData_;
}

std::vector<Data*>* DataHandler::getTestData() const
{
    return testData_;
}

std::vector<Data*>* DataHandler::getValidationData() const
{
    return validationData_;
}
