#pragma once

#include <cstdint>
#include <string>
#include <vector>

std::string decode(const std::vector<std::uint32_t>& tokens);
std::vector<std::uint32_t> encode(const std::string& text);
