#pragma once
// Small POSIX file utilities.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace ninfer {

// Reads an entire binary file. Throws std::runtime_error on failure.
std::vector<uint8_t> read_file_bytes(const std::string& path);

// Reads a whole text file into a std::string.
std::string read_file_text(const std::string& path);

// Reads at most `bytes` from `path` starting at `offset`.
std::vector<uint8_t> read_file_range(const std::string& path, uint64_t offset,
                                     uint64_t bytes);

bool file_exists(const std::string& path);

// Joins path components with '/' (POSIX).
std::string path_join(const std::string& a, const std::string& b);

}  // namespace ninfer
