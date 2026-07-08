#pragma once

#include<vector>
#include<map>
#include<utility>
#include<unordered_map>
#include<string>
#include<memory>
#include"../Tensor/Tensor.h"

class Module
{
public:
	virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input);
	std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& input);
	std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> parameters() const;
	std::unordered_map<std::string, std::shared_ptr<Tensor>> stateDictionary() const;
	void loadStateDictionary(std::unordered_map<std::string, std::shared_ptr<Tensor>>& stateDict);

protected:
	std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> parameters_;
	std::vector<std::pair<std::string, std::shared_ptr<Module>>> modules_;

	void registerParameter(const std::string& name, std::shared_ptr<Tensor> param);
	void registerModule(const std::string& name, std::shared_ptr<Module> module);

};
