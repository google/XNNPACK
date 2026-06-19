// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_LOG_H_
#define XNNPACK_YNNPACK_BASE_LOG_H_

#include <iostream>

namespace ynn {

class null_logger {
 public:
  template <typename T>
  null_logger& operator<<(const T&) {
    return *this;
  }
};

class logger {
 public:
  logger(const char* file, int line) : file_(file), line_(line) {}
  ~logger() { std::cerr << " (" << file_ << ":" << line_ << ")\n"; }

  template <typename T>
  logger& operator<<(const T& x) {
    std::cerr << x;
    return *this;
  }

 protected:
  const char* file_;
  int line_;
};

class fatal_logger : public logger {
 public:
  fatal_logger(const char* file, int line) : logger(file, line) {}
  ~fatal_logger() {
    std::cerr << " (" << file_ << ":" << line_ << ")\n";
    std::abort();
  }
};

#define YNN_LOG_LEVEL_NONE 0
#define YNN_LOG_LEVEL_FATAL 1
#define YNN_LOG_LEVEL_ERROR 2
#define YNN_LOG_LEVEL_WARNING 3
#define YNN_LOG_LEVEL_INFO 4
#define YNN_LOG_LEVEL_DEBUG 5

#ifndef YNN_LOG_LEVEL
#define YNN_LOG_LEVEL YNN_LOG_DEBUG
#endif

#if YNN_LOG_LEVEL >= YNN_LOG_LEVEL_ERROR
inline logger log_error(const char* file, int line) {
  return logger(file, line);
}
#else
inline null_logger log_error(const char*, int) { return null_logger(); }
#endif  // YNN_LOG_LEVEL >= YNN_LOG_LEVEL_ERROR

#if YNN_LOG_LEVEL >= YNN_LOG_LEVEL_WARNING
inline logger log_warning(const char* file, int line) {
  return logger(file, line);
}
#else
inline null_logger log_warning(const char*, int) { return null_logger(); }
#endif  // YNN_LOG_LEVEL >= YNN_LOG_LEVEL_WARNING

#if YNN_LOG_LEVEL >= YNN_LOG_LEVEL_INFO
inline logger log_info(const char* file, int line) {
  return logger(file, line);
}
#else
inline null_logger log_info(const char*, int) { return null_logger(); }
#endif  // YNN_LOG_LEVEL >= YNN_LOG_LEVEL_INFO

#if YNN_LOG_LEVEL >= YNN_LOG_LEVEL_DEBUG
inline logger log_debug(const char* file, int line) {
  return logger(file, line);
}
#else
inline null_logger log_debug(const char*, int) { return null_logger(); }
#endif  // YNN_LOG_LEVEL >= YNN_LOG_LEVEL_DEBUG

#define YNN_LOG_FATAL() ynn::fatal_logger(__FILE__, __LINE__)
#define YNN_LOG_ERROR() ynn::log_error(__FILE__, __LINE__)
#define YNN_LOG_WARNING() ynn::log_warning(__FILE__, __LINE__)
#define YNN_LOG_INFO() ynn::log_info(__FILE__, __LINE__)
#define YNN_LOG_DEBUG() ynn::log_debug(__FILE__, __LINE__)

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_LOG_H_
