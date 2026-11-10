#pragma once

#include <cstddef>
#include <memory>
#include <random>
#include <string>

#include "book.h"

#include "nnet/nn/sequential.h"
#include "nnet/optim/optimizer.h"

// The model and everything that carries over from one book to the next.
class BookTrainer {
public:
	// starts a fresh model; the run's best epoch gets saved to checkpointPath
	explicit BookTrainer(const std::string& checkpointPath);

	// shuffled epochs on one book, saving whenever its validation loss improves
	void train(const Book& book, std::size_t epochs);
	double validationLoss(const Book& book) const;
	std::string write(const std::string& prompt, std::size_t length, std::mt19937& generator) const;

private:
	// members are built in this order, and the optimizer needs the model's parameters
	std::unique_ptr<nnet::Sequential> model_;
	nnet::AdamOptimizer optimizer_;
	std::mt19937 shuffleGenerator_;
	std::string checkpointPath_;
};
