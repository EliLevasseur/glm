#include "training_windows.h"

#include <cstddef>
#include <utility>

TrainingWindows buildWindows(const std::vector<std::uint32_t>& tokens) {
	std::size_t windowStart = 0;
	std::size_t targetIndex = 8;
	std::vector<std::uint32_t> targets;
	std::vector<std::vector<std::uint32_t>> inputs;
	std::vector<std::uint32_t> window;

	// the context tokens before targetIndex predict the token at targetIndex
	while (targetIndex < tokens.size()) {
		window.clear();
		for (std::size_t i = windowStart; i < targetIndex; i++) {
			window.push_back(tokens[i]);
		}
		inputs.push_back(window);
		targets.push_back(tokens[targetIndex]);
		targetIndex++;
		windowStart++;
	}
	return TrainingWindows{std::move(targets), std::move(inputs)};
}
