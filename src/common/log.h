#pragma once
// Minimal leveled logging, written to stderr (mirrors ninfer product logging:
// answer content on stdout, diagnostics on stderr).

#include <cstdarg>
#include <cstdio>
#include <string>

namespace ninfer {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

inline LogLevel& global_log_level() {
  static LogLevel level = LogLevel::Info;
  return level;
}

inline void log_printf(LogLevel level, const char* fmt, ...) {
  if (level < global_log_level()) return;
  static const char* tags[] = {"DEBUG", "INFO", "WARN", "ERROR"};
  std::fprintf(stderr, "[%s] ", tags[static_cast<int>(level)]);
  va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);
  std::fprintf(stderr, "\n");
}

#define LOG_DEBUG(...) ::ninfer::log_printf(::ninfer::LogLevel::Debug, __VA_ARGS__)
#define LOG_INFO(...)  ::ninfer::log_printf(::ninfer::LogLevel::Info,  __VA_ARGS__)
#define LOG_WARN(...)  ::ninfer::log_printf(::ninfer::LogLevel::Warn,  __VA_ARGS__)
#define LOG_ERROR(...) ::ninfer::log_printf(::ninfer::LogLevel::Error, __VA_ARGS__)

}  // namespace ninfer
