#pragma once

#include<vector>
#include"Data.h"

class CommonData
{
public:
	void setTrainingData(std::vector<Data*>* data);
	void setTestData(std::vector<Data*>* data);
	void setValidationData(std::vector<Data*>* data);


protected:
	std::vector<Data*>* trainingData_;
	std::vector<Data*>* testData_;
	std::vector<Data*>* validationData_;

};

