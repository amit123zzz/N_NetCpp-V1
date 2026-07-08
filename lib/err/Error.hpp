#pragma once

#include<stdexcept>
#include<format>
#include<stacktrace>
#include<iostream>
#include<string>

class Error : public std::runtime_error {
public:
	Error(const std::string& message) :
		std::runtime_error{ message }, trace_{ std::stacktrace::current(1) } {}

	std::string trace() const {
		return std::to_string(trace_);
	}

private:
	std::stacktrace trace_;
};
