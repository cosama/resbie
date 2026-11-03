#pragma once

// The log level: a run option, not part of the config. debug adds a line per
// sweep with the filter diagnostics (resbie/odometry.h).

#include <algorithm>
#include <iterator>
#include <string>

#include "bievr_lio/log.h"

namespace resbie {

// Log level names, in the order of bievr::log::Level.
inline constexpr const char* kLogLevels[] = {"debug", "info", "warn", "error"};

// Sets the process-wide log level (it is one global sink). False for an
// unknown name.
inline bool setLogLevel(const std::string& level) {
  const auto it = std::find(std::begin(kLogLevels), std::end(kLogLevels), level);
  if (it == std::end(kLogLevels)) return false;
  bievr::log::setLevel(static_cast<bievr::log::Level>(std::distance(std::begin(kLogLevels), it)));
  return true;
}

inline bool debugEnabled() { return bievr::log::enabled(bievr::log::D); }

}  // namespace resbie
