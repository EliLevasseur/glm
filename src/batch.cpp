#include "batch.h"

#include <numeric>
#include <stdexcept>
#include <utility>

#include "config.h"

Batch windowToBatch(const TrainingWindows& windows) {
	std::vector<std::size_t> allWindows(windows.count());
	std::iota(allWindows.begin(), allWindows.end(), 0);
	return windowToBatch(windows, allWindows);
}

Batch windowToBatch(const TrainingWindows& windows, const std::vector<std::size_t>& windowIndices) {
	const std::size_t contextSize = config::contextSize;
	std::vector<double> flatTargets;
	std::vector<double> flatInputs;
	flatTargets.resize(windowIndices.size() * config::vocabularySize, 0.0);
	flatInputs.resize(windowIndices.size() * contextSize, 0.0);

	// inputs are the ids themselves, the Embedding layer looks them up;
	// targets stay one-hot because softmaxCrossEntropy compares against them
	for (std::size_t row = 0; row < windowIndices.size(); row++) {
		const std::size_t windowIndex = windowIndices[row];
		if (windowIndex >= windows.count()) {
			throw std::out_of_range("Window index is past the last window");
		}

		// window i is read straight out of the text: its target sits right after it
		flatTargets[row * config::vocabularySize + windows.tokens[windowIndex + contextSize]] = 1.0;

		for (std::size_t slot = 0; slot < contextSize; slot++) {
			flatInputs[row * contextSize + slot] = windows.tokens[windowIndex + slot];
		}
	}

	return Batch {
		nnet::Tensor({windowIndices.size(), config::vocabularySize}, std::move(flatTargets)),
		nnet::Tensor({windowIndices.size(), contextSize}, std::move(flatInputs))
	};
}
