#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <random>

#include "text_file.h"
#include "tokenizer.h"
#include "training_windows.h"
#include "batch.h"

#include "nnet/nn/module.h"
#include "nnet/nn/dense.h"
#include "nnet/nn/sequential.h"
#include "nnet/core/autograd/backward.h"
#include "nnet/optim/optimizer.h"

int main() {
	const int randomSeed = 42;
	const nnet::Tensor::Shape networkShape = {8 * 256, 128, 256};
	const std::size_t batchSize = 32;
	const std::string bookPath = "alice.txt";
	std::size_t epochs = 10;

	std::mt19937 weightGenerator(randomSeed);
	std::vector<std::unique_ptr<nnet::Unary_Module>> components;

	// map the one-hot context tokens to hidden features, then predict the next byte
	components.push_back(std::make_unique<nnet::Dense>(networkShape[0], networkShape[1], weightGenerator));
	components.push_back(std::make_unique<nnet::ReLU>());
	components.push_back(std::make_unique<nnet::Dense>(networkShape[1], networkShape[2], weightGenerator));

	nnet::Sequential model(std::move(components));
	nnet::AdamOptimizer optimizer(model.parameters(), 0.01);

	std::string book = readTextFile(bookPath);

	std::vector<std::uint32_t> tokens = encode(book);
	const std::size_t split = tokens.size() * 9 / 10;

	// split before building windows so training and validation contexts do not overlap
	std::vector<std::uint32_t> trainTokens(tokens.begin(), tokens.begin() + split);
	std::vector<std::uint32_t> validationTokens(tokens.begin() + split, tokens.end());

	TrainingWindows trainWindows = buildWindows(trainTokens);
	TrainingWindows validationWindows = buildWindows(validationTokens);

	if (trainWindows.inputs.empty() || validationWindows.inputs.empty()) {
		std::cerr << "The book needs enough tokens for training and validation windows\n";
		return 1;
	}

	for (std::size_t epoch = 0; epoch < epochs; epoch++) {
		double totalLoss = 0.0;
		auto epochStart = std::chrono::steady_clock::now();

		for (std::size_t start = 0; start < trainWindows.inputs.size(); start += batchSize) {
			// keep the final batch within the dataset even if it has fewer than 32 windows
			const std::size_t end = std::min(start + batchSize, trainWindows.inputs.size());
			TrainingWindows windows{
				{trainWindows.targets.begin() + start, trainWindows.targets.begin() + end},
				{trainWindows.inputs.begin() + start, trainWindows.inputs.begin() + end}
			};
			Batch batch = windowToBatch(windows);

			model.zeroGrad();

			// data leaves do not need gradients; the model parameters still receive them
			auto input = nnet::makeLeaf(batch.predictors, false);
			auto prediction = model.forward(input);
			auto target = nnet::makeLeaf(batch.targets, false);
			auto loss = nnet::softmaxCrossEntropy(prediction, target);

			// weight each batch by its size so a short final batch adds to the loss correctly
			totalLoss += loss->data.at({0}) * static_cast<double>(end - start);
			nnet::backward(loss);
			optimizer.step();

			if (((start / batchSize) + 1) % 100 == 0) {
				std::cout << "Epoch " << epoch + 1 << " | Windows " << end << '/' << trainWindows.inputs.size() << '\n' << std::flush;
			}
		}

		double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epochStart).count();
		std::cout << "Epoch " << epoch + 1 << " | Training loss: " << totalLoss / trainWindows.inputs.size()
			<< " | Seconds: " << seconds << '\n' << std::flush;
	}


	return 0;
}
