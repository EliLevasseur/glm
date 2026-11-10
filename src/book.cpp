#include "book.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <utility>
#include <vector>

#include "config.h"
#include "text_file.h"
#include "tokenizer.h"

Book loadBook(const std::string& path) {
	const std::string name = std::filesystem::path(path).stem().string();
	std::vector<std::uint32_t> tokens = encode(readTextFile(path));
	const std::size_t split = static_cast<std::size_t>(static_cast<double>(tokens.size()) * config::trainingFraction);

	// split before building windows so training and validation contexts do not overlap
	TrainingWindows trainWindows = buildWindows({tokens.begin(), tokens.begin() + split});
	TrainingWindows validationWindows = buildWindows({tokens.begin() + split, tokens.end()});

	if (trainWindows.count() == 0 || validationWindows.count() == 0) {
		throw std::runtime_error(name + " needs enough text for training and validation windows");
	}

	// a fixed, evenly spaced sample: it covers the whole validation section, it's the same
	// every run, and its one-hot targets stay small however big the book is
	const std::size_t sampleSize = std::min(config::validationSampleSize, validationWindows.count());
	const std::size_t step = validationWindows.count() / sampleSize;
	std::vector<std::size_t> sample;
	sample.reserve(sampleSize);
	for (std::size_t i = 0; i < sampleSize; i++) {
		sample.push_back(i * step);
	}

	Batch validationBatch = windowToBatch(validationWindows, sample);
	return Book{name, std::move(trainWindows), std::move(validationWindows), std::move(validationBatch)};
}
