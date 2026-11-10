#pragma once

#include <cstddef>
#include <string>

// everything you'd change between runs lives here
namespace config {

	// data
	inline const std::string bookPath = "data/tinystories.txt";   // used when glm gets no book arguments
	inline constexpr double trainingFraction = 0.9;   // the rest of the book is validation
	inline constexpr std::size_t validationSampleSize = 20000;  // windows checked each epoch, evenly spaced

	// model
	// changing these needs a fresh training run, old checkpoints won't load
	inline constexpr std::size_t contextSize = 32;      // characters the model sees at once
	inline constexpr std::size_t embeddingSize = 16;    // learned numbers per character
	inline constexpr std::size_t hiddenSize = 128;
	inline constexpr std::size_t vocabularySize = 256;  // one slot per byte value

	// training
	inline constexpr unsigned int randomSeed = 42;
	inline constexpr std::size_t epochs = 10;           // the most it will run, early stopping usually ends sooner
	inline constexpr std::size_t patience = 3;           // stop after this many epochs without a new best
	inline constexpr std::size_t batchSize = 128;
	inline constexpr double learningRate = 0.002;        // a bit higher to go with the bigger batches
	inline constexpr std::size_t progressEvery = 1000;  // batches between progress lines

	// each training run saves its best epoch as the next number here: 1.ckpt, 2.ckpt, ...
	inline const std::string checkpointFolder = "checkpoints";

	// writing, prompts need at least contextSize characters
	inline const std::string samplePrompt = "The rabbit hopped over the grass to eat the carr";
	inline constexpr std::size_t sampleLength = 100;    // printed after each epoch
	inline constexpr std::size_t writeLength = 300;     // default for glm_write
	// below 1 makes likely characters likelier and rare ones rarer; much below 0.5 gets repetitive
	inline constexpr double temperature = 0.6;

}
