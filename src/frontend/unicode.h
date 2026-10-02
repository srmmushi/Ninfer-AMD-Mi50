#pragma once
// Minimal Unicode support for the BPE pretokenizer: UTF-8 decoding and
// General-Category approximations for Letter (L) and Number (N), replacing
// the vendored utf8proc of the original ninfer.

#include <cstdint>
#include <string>

namespace ninfer {

// Decodes the UTF-8 code point at `pos` (byte index). Advances `pos` past it.
// Invalid sequences decode as U+FFFD and advance one byte.
uint32_t utf8_decode(const std::string& s, size_t& pos);

// Encodes a code point as UTF-8 and appends to `out`.
void utf8_encode(uint32_t cp, std::string& out);

// True if the code point belongs to Unicode general category L* (letter).
bool is_unicode_letter(uint32_t cp);

// True if the code point belongs to Unicode general category N* (number).
bool is_unicode_number(uint32_t cp);

}  // namespace ninfer
