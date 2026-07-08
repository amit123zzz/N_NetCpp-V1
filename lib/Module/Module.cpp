#include "Module.h"

std::shared_ptr<Tensor> Module::forward(const std::shared_ptr<Tensor>& input)
{
    throw Error("forward is not implemented");
}

std::shared_ptr<Tensor> Module::operator()(const std::shared_ptr<Tensor>& input)
{
    return forward(input);
}

std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> Module::parameters() const
{
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> _ret{};

    for (auto& [name, param] : parameters_) {
        _ret.emplace_back(name, param);
    }

    for (auto& [mName, module] : modules_) {
        for (auto& [pName, param] : module->parameters()) {
            std::string _name = mName.empty() ? pName : std::format("{}.{}", mName, pName);
            _ret.emplace_back(_name, param);
        }
    }

    return _ret;
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> Module::stateDictionary() const
{
    std::unordered_map<std::string, std::shared_ptr<Tensor>> _ret;

    for (const auto& [name, param] : parameters()) {
        _ret[name] = param;
    }

    return _ret;
}

void Module::loadStateDictionary(std::unordered_map<std::string, std::shared_ptr<Tensor>>& stateDict)
{
    for (const auto& [name, param] : parameters()) {
        auto _it = stateDict.find(name);

        if (_it == stateDict.end()) {
            std::cerr << std::format("Warning : Parameters {} not found in stateDict", name) << std::endl;
            continue;
        }

        std::shared_ptr<Tensor> _storedParams = _it->second;

        if (param->shape() != _storedParams->shape()) {
            throw Error(std::format("Parameter {} has different shape in state dictionary", name));
        }

        param->data() = _storedParams->data();
    }
}

void Module::registerParameter(const std::string& name, std::shared_ptr<Tensor> param)
{
    for (const auto& [nm, prm] : parameters_) {
        if (nm == name) {
            throw Error(std::format("Parameter name {} alraedy exists", name));
        }
    }

    parameters_.emplace_back(name, param);
}

void Module::registerModule(const std::string& name, std::shared_ptr<Module> module)
{
    for (auto& [nm, prm] : modules_) {
        if (nm == name) {
            throw Error(std::format("Module name {} alraedy exists", name));
        }
    }

    modules_.emplace_back(name, module);
}
