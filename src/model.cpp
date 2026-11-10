#include "model.h"

#include <vector>

#include "config.h"

#include "nnet/nn/dense.h"
#include "nnet/nn/embedding.h"
#include "nnet/nn/module.h"

std::unique_ptr<nnet::Sequential> buildModel(std::mt19937& weightGenerator) {
	// the embedding turns each window of ids into contextSize * embeddingSize numbers
	const std::size_t inputSize = config::contextSize * config::embeddingSize;
	const std::size_t hiddenSize = config::hiddenSize;
	const std::size_t outputSize = config::vocabularySize;

	std::vector<std::unique_ptr<nnet::Unary_Module>> components;

	// look up each character's learned row, map them to hidden features, then predict the next byte
	components.push_back(std::make_unique<nnet::Embedding>(config::vocabularySize, config::embeddingSize, weightGenerator));
	components.push_back(std::make_unique<nnet::Dense>(inputSize, hiddenSize, weightGenerator));
	components.push_back(std::make_unique<nnet::ReLU>());
	components.push_back(std::make_unique<nnet::Dense>(hiddenSize, outputSize, weightGenerator));

	return std::make_unique<nnet::Sequential>(std::move(components));
}
