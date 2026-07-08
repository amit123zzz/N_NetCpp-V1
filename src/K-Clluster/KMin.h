#pragma once

#include<vector>
#include"../DataSet/Data.h"

#include"../DataSet/Common.h"

class KMin : public CommonData
{
public:
	KMin();
	KMin(std::uint32_t);
	~KMin();


	void findKNearest(Data* queryPoint);

	void setK(std::uint32_t);

	std::int32_t predict();

	double calculateDistance(Data* querytPoint, Data* input);
	double validatePerformance();
	double testPerformance();

private:
	std::uint32_t k_;

	std::vector<Data*>* neighbours_;
	
	
	
};

