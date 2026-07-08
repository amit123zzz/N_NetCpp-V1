#pragma once

#include<fstream>
#include"Data.h"
#include<string>
#include<map>
#include<unordered_set>

#define TRAIN_SET_PERCENT 0.75
#define TEST_SET_PERCENT 0.20
#define VALIDATION_PERCENT 0.05

class DataHandler
{
public:
	DataHandler();

	~DataHandler();

	void readFeatureVectors(const std::string&);
	void readFeatureLabels(const std::string&);
	
	void splitData();
	void countClasses();

	std::uint32_t convertToLittleEndian(const unsigned char* bytes);

	std::vector<Data*>* getTrainingData() const;
	std::vector<Data*>* getTestData() const;
	std::vector<Data*>* getValidationData() const;

private:
	std::vector<Data*>* dataSet_;
	std::vector<Data*>* trainingData_;
	std::vector<Data*>* testData_;
	std::vector<Data*>* validationData_;

	std::int32_t numClasses_;
	std::int32_t featureVectorSize_;
	std::map<std::uint8_t, std::int32_t> classMap_;
};

