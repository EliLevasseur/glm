#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "nnet/core/tensor.h"
#include "training_windows.h"

struct Batch {
	nnet::Tensor targets;
	nnet::Tensor predictors;
};

Batch windowToBatch(const TrainingWindows& windows);
// only the windows at windowIndices, in that order
Batch windowToBatch(const TrainingWindows& windows, const std::vector<std::size_t>& windowIndices);
