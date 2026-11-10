#include "generate.h"

#include <stdexcept>
#include <utility>

#include "config.h"

static_assert(config::temperature > 0.0, "temperature must be above 0, it divides the scores");

std::string generateText(const nnet::Sequential& Model, const std::string& prompt, std::size_t length, std::mt19937& generator) {
	// the first window needs a full context to read
	if (prompt.size() < config::contextSize) {
		throw std::invalid_argument("The prompt needs at least " + std::to_string(config::contextSize) + " characters");
	}

	std::vector<std::uint32_t> ids = encode(prompt);
	nnet::NoGrad noGrad;   // no graph needed for the whole function
	//

	for (std::size_t i = 0; i < length; i++) {
		// the last contextSize ids are the window, plus a dummy target so it counts as one window
		std::vector<std::uint32_t> windowTokens(ids.end() - config::contextSize, ids.end());
		windowTokens.push_back(0);

		// only the predictors get used, the dummy target is ignored
		Batch batch = windowToBatch(TrainingWindows{std::move(windowTokens)}, {0});

		nnet::Value scores = Model.forward(nnet::makeLeaf(batch.predictors, false));
		// dividing the scores by the temperature before softmax sharpens (below 1) or flattens (above 1) the choice
		nnet::Tensor probabilities = nnet::softmax(scores->data * (1.0 / config::temperature));

		std::discrete_distribution<std::uint32_t> pickCharacter(probabilities.getData().begin(), probabilities.getData().end());
		ids.push_back(pickCharacter(generator));
}
	return decode(ids);
}
