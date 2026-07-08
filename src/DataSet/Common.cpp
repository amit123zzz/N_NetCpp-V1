#include "Common.h"

void CommonData::setTrainingData(std::vector<Data*>* data)
{
	trainingData_ = data;
}

void CommonData::setTestData(std::vector<Data*>* data)
{
	testData_ = data;
}

void CommonData::setValidationData(std::vector<Data*>* data)
{
	validationData_ = data;
}