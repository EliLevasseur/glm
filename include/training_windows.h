#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "config.h"

// The text, stored once. Window i is tokens[i] to tokens[i + contextSize - 1],
// and its target is the token right after it, tokens[i + contextSize].
struct TrainingWindows {
	std::vector<std::uint32_t> tokens;

	// every position with a full window before its target is one window
	std::size_t count() const {
		return tokens.size() > config::contextSize ? tokens.size() - config::contextSize : 0;
	}
};

TrainingWindows buildWindows(const std::vector<std::uint32_t>& tokens);
