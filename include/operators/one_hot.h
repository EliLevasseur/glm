#pragma once

#include <cstdint>
#include <vector>

std::vector<std::vector<double>> oneHotEncoder(const std::vector<std::uint32_t>& tokens);
std::vector<double> oneHotEncoder(std::uint32_t token);
