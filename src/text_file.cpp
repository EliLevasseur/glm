#include "text_file.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

std::string readTextFile(const std::string& path) {
	// binary so the bytes come back exactly as they are on disk
	std::ifstream file(path, std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("Critical Error: Could not open the file '" + path + "'");
	}

	std::ostringstream contents;
	contents << file.rdbuf();
	return contents.str();
}
