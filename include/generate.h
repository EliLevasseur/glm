#pragma once

#include <cstddef>
#include <random>
#include <string>

#include "batch.h"
#include "tokenizer.h"

#include "nnet/nn/sequential.h"

std::string generateText(const nnet::Sequential& Model, const std::string& prompt, std::size_t length, std::mt19937& generator);

