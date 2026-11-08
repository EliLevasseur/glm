#include "operators/one_hot.h"

std::vector<std::vector<double>> oneHotEncoder(const std::vector<std::uint32_t>& tokens) {
	std::vector<std::vector<double>> out(tokens.size(), std::vector<double>(256, 0));
	for (std::size_t token = 0; token < tokens.size(); token++) {
		out[token].at(tokens[token]) = 1;
	}
	return out;
}

std::vector<double> oneHotEncoder(const std::uint32_t token) {
	std::vector<double> out(256, 0);
	out.at(token) = 1;
	return out;
}

