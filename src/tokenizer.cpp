#include "tokenizer.h"

std::vector<std::uint32_t> encode(const std::string& text) {
	std::vector<std::uint32_t> tokens;
	tokens.reserve(text.size());
	for (const auto& character : text) {
		// unsigned char so bytes above 127 come out as 128-255
		tokens.push_back(static_cast<unsigned char>(character));
	}
	return tokens;
}


std::string decode(const std::vector<std::uint32_t>& tokens) {
	std::string outstr = "";	
    	for (auto& token : tokens) {
	  outstr += static_cast<char>(token);
		}
	return outstr;
}
      		
