#ifndef BIEVR_LIO_LOG_H_
#define BIEVR_LIO_LOG_H_

// resbie: a one-sink logger in place of log++. LOG(D|I|W|E, a << b) writes
// "LEVEL a b" as one line to stdout when the level is enabled; the message
// is not formatted otherwise. Errors and warnings are coloured on a terminal.

#include <unistd.h>

#include <atomic>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace bievr::log {

enum class Level { kDebug, kInfo, kWarn, kError };
inline constexpr Level D = Level::kDebug;
inline constexpr Level I = Level::kInfo;
inline constexpr Level W = Level::kWarn;
inline constexpr Level E = Level::kError;

inline std::atomic<int> min_level{static_cast<int>(Level::kInfo)};

inline void setLevel(Level level) { min_level.store(static_cast<int>(level), std::memory_order_relaxed); }

inline bool enabled(Level level) {
  return static_cast<int>(level) >= min_level.load(std::memory_order_relaxed);
}

inline void write(Level level, const std::string& message) {
  static constexpr const char* kPrefix[] = {"DEBUG ", "INFO  ", "WARN  ", "ERROR "};
  static constexpr const char* kColor[] = {"", "", "\033[33m", "\033[31m"};
  static const bool tty = isatty(STDOUT_FILENO);
  static std::mutex mutex;
  const int i = static_cast<int>(level);
  const bool color = tty && kColor[i][0] != '\0';
  std::lock_guard<std::mutex> lock(mutex);
  std::cout << (color ? kColor[i] : "") << kPrefix[i] << message << (color ? "\033[0m" : "")
            << std::endl;
}

}  // namespace bievr::log

#define LOG(level, message)                                     \
  do {                                                          \
    if (::bievr::log::enabled(::bievr::log::level)) {           \
      std::ostringstream log_stream_;                           \
      log_stream_ << message;                                   \
      ::bievr::log::write(::bievr::log::level, log_stream_.str()); \
    }                                                           \
  } while (0)

#endif  // BIEVR_LIO_LOG_H_
