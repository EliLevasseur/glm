#pragma once

#include <memory>
#include <random>

#include "nnet/nn/sequential.h"

// training and writing both build the model here, so the checkpoint always fits
std::unique_ptr<nnet::Sequential> buildModel(std::mt19937& weightGenerator);
