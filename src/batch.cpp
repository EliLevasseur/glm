#include "batch.h"

#include <stdexcept>
#include <utility>

Batch windowToBatch(const TrainingWindows& windows) {
	if (windows.inputs.empty() || windows.inputs.size() != windows.targets.size()) {
		throw std::invalid_argument("Expected matching, nonempty inputs and targets");
	}

	const std::size_t contextSize = windows.inputs.front().size();
	std::vector<double> flatTargets;
	std::vector<double> flatInputs;
	flatTargets.resize(windows.targets.size() * 256, 0.0);
	flatInputs.resize(windows.inputs.size() * contextSize * 256, 0.0);

	// write directly into the flat tensors to avoid allocating temporary one-hot rows
	for (std::size_t row = 0; row < windows.targets.size(); row++) {
		if (windows.targets[row] >= 256) {
			throw std::out_of_range("Target token must be below 256");
		}
		flatTargets[row * 256 + windows.targets[row]] = 1.0;
	}

	for (std::size_t row = 0; row < windows.inputs.size(); row++) {
		const auto& window = windows.inputs[row];
		if (window.size() != contextSize) {
			throw std::invalid_argument("Windows must have equal lengths");
		}
		for (std::size_t token = 0; token < contextSize; token++) {
			if (window[token] >= 256) {
				throw std::out_of_range("Input token must be below 256");
			}
			flatInputs[(row * contextSize + token) * 256 + window[token]] = 1.0;
		}
	}

	return Batch {
		nnet::Tensor({windows.targets.size(), 256}, std::move(flatTargets)),
		nnet::Tensor({windows.inputs.size(), contextSize * 256}, std::move(flatInputs))
	};
}
