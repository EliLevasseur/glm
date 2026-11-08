#pragma once

#include <cstdint>
#include <vector>

#include "nnet/core/tensor.h"
#include "training_windows.h"

struct Batch {
	nnet::Tensor targets;
	nnet::Tensor predictors;
};

Batch windowToBatch(const TrainingWindows& windows);
