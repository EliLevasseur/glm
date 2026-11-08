#include <iostream>

#include "text_file.h"
#include "tokenizer.h"
#include "training_windows.h"


int main() {
	std::string book = readTextFile("data/alice.txt");
	std::vector<std::uint32_t> tokens = encode(book);
	TrainingWindows data = buildWindows(tokens);
	return 0;
}
