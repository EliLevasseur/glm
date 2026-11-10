#include "checkpoints.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>

#include "config.h"

// the largest n with a file named n.ckpt in the folder, or 0 if there are none
static std::size_t highestCheckpointNumber() {
	const std::filesystem::path folder = config::checkpointFolder;
	if (!std::filesystem::exists(folder)) {
		return 0;
	}

	std::size_t highest = 0;
	for (const auto& entry : std::filesystem::directory_iterator(folder)) {
		const std::string stem = entry.path().stem().string();
		const bool isNumbered = entry.path().extension() == ".ckpt" && !stem.empty() &&
			std::all_of(stem.begin(), stem.end(), [](unsigned char c) { return std::isdigit(c); });
		if (entry.is_regular_file() && isNumbered) {
			highest = std::max(highest, static_cast<std::size_t>(std::stoull(stem)));
		}
	}
	return highest;
}

std::string checkpointPathFor(std::size_t number) {
	return (std::filesystem::path(config::checkpointFolder) / (std::to_string(number) + ".ckpt")).string();
}

std::string nextCheckpointPath() {
	std::filesystem::create_directories(config::checkpointFolder);
	return checkpointPathFor(highestCheckpointNumber() + 1);
}

std::string latestCheckpointPath() {
	const std::size_t highest = highestCheckpointNumber();
	if (highest == 0) {
		throw std::runtime_error("No checkpoints in " + config::checkpointFolder + " yet, train first with ./build/glm");
	}
	return checkpointPathFor(highest);
}
