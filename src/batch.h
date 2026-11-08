#pragma once

#include <cstdint>
#include "training_windows.h"

struct Batch {
	nnet::Tensor targets;
	nnet::Tensor predictors;
};

Batch windowToBatch(const Training_Window& window);

std::vector<std::vector<<std::size_t>> oneHotEncoder(const std::vector<std::uint32_t>& tokens) {
	std::vector<std::vector<std::size_t>> out(tokens.size(), std::vector<std::size_t>(256, 0));
	for (int token = 0; token < tokens.size(); token++) {
      		out[token][tokens[token]] = 1;
		}
	return out;
	}
      		
