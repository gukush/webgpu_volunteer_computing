#pragma once
#include <string>
#include <vector>
#include <cstdint>

std::string base64_encode(const std::vector<uint8_t>& data);
std::vector<uint8_t> base64_decode(const std::string& encoded);