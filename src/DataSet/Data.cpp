#include "Data.h"

Data::Data()
{
    featureVector_ = new std::vector<std::uint8_t>;
}

Data::~Data()
{
}

void Data::setFeatureVector(std::vector<std::uint8_t>* vec)
{
    featureVector_ = vec;
}

void Data::appendToFeatureVector(std::uint8_t value)
{
    featureVector_->emplace_back(value);
}

void Data::setlabel(std::uint8_t value)
{
    label_ = value;
}

void Data::setEnumeratedLabel(std::int32_t value)
{
    enumLabel_ = value;
}

std::int32_t Data::getFeatureVectorSize() const
{
    return featureVector_->size();
}

std::uint8_t Data::getLabel() const
{
    return label_;
}

std::uint8_t Data::getEnumeratedLabel() const
{
    return enumLabel_;
}

std::vector<std::uint8_t>* Data::getFeatureVector() const
{
    return featureVector_;
}
