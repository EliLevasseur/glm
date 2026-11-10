#include <algorithm>
#include <cctype>
#include <exception>
#include <iostream>
#include <random>
#include <string>

#include "checkpoints.h"
#include "config.h"
#include "generate.h"
#include "model.h"

#include "nnet/nn/checkpoint.h"

static bool isWholeNumber(const std::string& text) {
	return !text.empty() && std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isdigit(c); });
}

// usage: ./build/glm_write [checkpoint number] ["your prompt here"] [length]
// without a number it uses the newest checkpoint
int main(int argc, char* argv[]) {
	// the first argument picks a checkpoint only if it's a plain number
	int nextArgument = 1;
	std::string checkpointPath;
	try {
		if (argc > 1 && isWholeNumber(argv[1])) {
			checkpointPath = checkpointPathFor(std::stoul(argv[1]));
			nextArgument = 2;
		} else {
			checkpointPath = latestCheckpointPath();
		}
	} catch (const std::exception& error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
	const std::string prompt = argc > nextArgument ? argv[nextArgument] : config::samplePrompt;
	const std::size_t length = argc > nextArgument + 1 ? std::stoul(argv[nextArgument + 1]) : config::writeLength;

	// the starting weights get replaced by the checkpoint, so any seed works here
	std::mt19937 weightGenerator(config::randomSeed);
	std::unique_ptr<nnet::Sequential> model = buildModel(weightGenerator);

	try {
		nnet::loadParameters(*model, checkpointPath);
	} catch (const std::exception& error) {
		std::cerr << "Could not load " << checkpointPath << ": " << error.what() << '\n';
		return 1;
	}
	std::cerr << "Writing with " << checkpointPath << '\n';

	// a new seed each run, so every run writes something different
	std::mt19937 sampleGenerator(std::random_device{}());
	try {
		std::cout << generateText(*model, prompt, length, sampleGenerator) << '\n';
	} catch (const std::exception& error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
	return 0;
}
