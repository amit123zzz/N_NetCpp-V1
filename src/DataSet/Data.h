#pragma once

#include<cstdint>
#include<vector>
#include<print>

class Data
{
public:
	Data();
	~Data();

	void setFeatureVector(std::vector<std::uint8_t>*);
	void appendToFeatureVector(std::uint8_t);
	void setlabel(std::uint8_t);
	void setEnumeratedLabel(std::int32_t);

	std::int32_t getFeatureVectorSize() const;

	std::uint8_t getLabel() const;

	std::uint8_t getEnumeratedLabel() const;

	std::vector<std::uint8_t>* getFeatureVector() const;

private:
	std::vector<std::uint8_t>* featureVector_;
	std::uint8_t label_;
	std::int32_t enumLabel_;
};

