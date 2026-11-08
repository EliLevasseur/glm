#pragma once

#include <cstdint>
#include <vector>

struct TrainingWindows {
	std::vector<std::uint32_t> targets;
	const std::vector<std::vector<std::uint32_t>> inputs;
};

TrainingWindows buildWindows(const std::vector<std::uint32_t>& tokens);
