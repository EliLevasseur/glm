#include "book_trainer.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <vector>

#include "config.h"
#include "generate.h"
#include "model.h"

#include "nnet/core/autograd/backward.h"
#include "nnet/core/autograd/operations.h"
#include "nnet/nn/checkpoint.h"
#include "nnet/train/trainer.h"


static std::unique_ptr<nnet::Sequential> freshModel() {
	std::mt19937 weightGenerator(config::randomSeed);
	return buildModel(weightGenerator);
}

BookTrainer::BookTrainer(const std::string& checkpointPath)
	: model_(freshModel()),
	  optimizer_(model_->parameters(), config::learningRate),
	  shuffleGenerator_(config::randomSeed),
	  checkpointPath_(checkpointPath) {
	// every run starts fresh and gets its own numbered file
	std::cout << "Starting a fresh model, saving this run to " << checkpointPath_ << '\n';
}

void BookTrainer::train(const Book& book, std::size_t epochs) {
	// shuffled every epoch so batches mix passages from the whole book
	std::vector<std::size_t> windowOrder(book.trainWindows.count());
	std::iota(windowOrder.begin(), windowOrder.end(), 0);

	// start from the current loss, so an epoch only gets saved if it beats what we already have
	double bestValidationLoss = validationLoss(book);
	std::size_t epochsWithoutImprovement = 0;
	std::cout << book.name << " | Validation loss before training: " << bestValidationLoss << '\n';

	for (std::size_t epoch = 0; epoch < epochs; epoch++) {
		double totalLoss = 0.0;
		auto epochStart = std::chrono::steady_clock::now();
		std::shuffle(windowOrder.begin(), windowOrder.end(), shuffleGenerator_);

		for (std::size_t start = 0; start < windowOrder.size(); start += config::batchSize) {
			// keep the final batch within the dataset even if it has fewer than batchSize windows
			const std::size_t end = std::min(start + config::batchSize, windowOrder.size());
			const std::vector<std::size_t> batchIndices(windowOrder.begin() + start, windowOrder.begin() + end);
			Batch batch = windowToBatch(book.trainWindows, batchIndices);

			model_->zeroGrad();

			// data leaves do not need gradients; the model parameters still receive them (hence the false parameters)
			auto input = nnet::makeLeaf(batch.predictors, false);
			auto prediction = model_->forward(input);
			auto target = nnet::makeLeaf(batch.targets, false);
			auto loss = nnet::softmaxCrossEntropy(prediction, target);

			// weight each batch by its size so a short final batch adds to the loss correctly
			totalLoss += loss->data.at({0}) * static_cast<double>(end - start);
			nnet::backward(loss);
			optimizer_.step();

			if (((start / config::batchSize) + 1) % config::progressEvery == 0) {
				std::cout << "Epoch " << epoch + 1 << " | Windows " << end << '/' << windowOrder.size() << '\n' << std::flush;
			}
		}
		double currentValidationLoss = validationLoss(book);
		double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epochStart).count();
		std::cout << book.name << " | Epoch " << epoch + 1 << " | Training loss: " << totalLoss / windowOrder.size()
			<< " | Validation loss: " << currentValidationLoss << " | Seconds: " << seconds << '\n' << std::flush;

		// keep the best model, not just the last one
		if (currentValidationLoss < bestValidationLoss) {
			bestValidationLoss = currentValidationLoss;
			epochsWithoutImprovement = 0;
			nnet::saveParameters(*model_, checkpointPath_);
			std::cout << "Saved best model so far to " << checkpointPath_ << '\n';
		} else {
			epochsWithoutImprovement++;
		}

		// same seed every epoch, so changes in the text come from the model
		std::mt19937 checkerShuffler(config::randomSeed);
		std::cout << "Sentence attempt " << epoch + 1 << ": " << write(config::samplePrompt, config::sampleLength, checkerShuffler) << "\n\n";

		// once validation stops improving, more epochs only memorize the book
		if (epochsWithoutImprovement >= config::patience) {
			std::cout << "No new best for " << config::patience << " epochs, stopping " << book.name << " early\n";
			break;
		}
	}

	// carry on from the best weights, not the last epoch's. The checkpoint holds this
	// book's best epoch, or the starting weights if no epoch beat them.
	if (std::filesystem::exists(checkpointPath_)) {
		nnet::loadParameters(*model_, checkpointPath_);
	}
}

double BookTrainer::validationLoss(const Book& book) const {
	return nnet::evaluateLoss(*model_, book.validationBatch.predictors, book.validationBatch.targets, nnet::softmaxCrossEntropy);
}

std::string BookTrainer::write(const std::string& prompt, std::size_t length, std::mt19937& generator) const {
	return generateText(*model_, prompt, length, generator);
}
