#include "KMin.h"

#include<limits>

#include<map>

KMin::KMin()
{
}

KMin::KMin(std::uint32_t val)
{
    k_ = val;
}

KMin::~KMin()
{
}

void KMin::findKNearest(Data* queryPoint)
{
    neighbours_ = new std::vector<Data*>;
    double _min = std::numeric_limits<double>::max();
    double _previousMin = _min;
    std::uint32_t _index{};
    
    for (std::uint32_t i{}; i < k_; ++i) {
        if (i == 0) {
            for (std::uint32_t j{}; j < trainingData_->size(); ++j) {
                double _dist = calculateDistance(queryPoint, trainingData_->at(j));
                //trainingData_->at(j)->setDistance(_dist);
                if (_dist < _min) {
                    _min = _dist;
                    _index = j;
                }
                neighbours_->emplace_back(trainingData_->at(_index));
                _previousMin = _min;
                _min = std::numeric_limits<double>::max();
            }
        }
        else {
            for (std::uint32_t j{}; j < trainingData_->size(); ++j) {
                double _dist{};
                //double _dist = trainingData_->at(j)->getDistance();
                if (_dist > _previousMin && _dist < _min) {
                    _min = _dist;
                    _index = j;
                }
                neighbours_->emplace_back(trainingData_->at(_index));
                _previousMin = _min;
                _min = std::numeric_limits<double>::max();
            }
        }
    }
}

void KMin::setK(std::uint32_t value)
{
    k_ = value;
}

std::int32_t KMin::predict()
{
    std::map<std::uint8_t, std::uint32_t> _classFreq{};
    for (std::uint32_t i{}; i < neighbours_->size(); ++i) {
        if (_classFreq.find(neighbours_->at(i)->getLabel()) == _classFreq.end()) {
            _classFreq[neighbours_->at(i)->getLabel()] = 1;
        }
        else {
            _classFreq[neighbours_->at(i)->getLabel()]++;
        }
    }

    std::int32_t _best{};
    std::int32_t _max{};

    for (const auto& [key, value] : _classFreq) {
        if (value > _max) {
            _max = value;
            _best = key;
        }
    }

    neighbours_->clear();
    return _best;
}

double KMin::calculateDistance(Data* querytPoint, Data* input)
{
    return 0.0;
}

double KMin::validatePerformance()
{
    return 0.0;
}

double KMin::testPerformance()
{
    return 0.0;
}
