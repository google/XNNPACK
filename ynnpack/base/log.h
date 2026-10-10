// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_LOG_H_
#define XNNPACK_YNNPACK_BASE_LOG_H_

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "include/xnnpack.h"

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
  logger(enum xnn_log_level level, const char* function, const char* file,
         int line)
      : level_(level),
        function_(function),
        file_(file),
        line_(line),
        flushed_(false) {}

  logger(enum xnn_log_level level, const char* file, int line)
      : logger(level, nullptr, file, line) {}

  ~logger() { flush(); }

  void flush() {
    if (!flushed_) {
      flushed_ = true;
      if (file_ != nullptr) {
        if (function_ != nullptr) {
          stream_ << " (" << function_ << ", " << file_ << ":" << line_ << ")";
        } else {
          stream_ << " (" << file_ << ":" << line_ << ")";
        }
      }
      xnn_log_callback_fn callback = xnn_get_log_callback();
      if (callback != nullptr) {
        callback(level_, file_, line_, stream_.str().c_str(),
                 xnn_get_log_callback_user_data());
      } else {
        std::cerr << stream_.str() << "\n";
      }
    }
  }

  template <typename T>
  logger& operator<<(const T& x) {
    stream_ << x;
    return *this;
  }

 protected:
  enum xnn_log_level level_;
  const char* function_;
  const char* file_;
  int line_;
  std::ostringstream stream_;
  bool flushed_;
};

class fatal_logger : public logger {
 public:
  fatal_logger(const char* function, const char* file, int line)
      : logger(xnn_log_level_fatal, function, file, line) {}
  fatal_logger(const char* file, int line)
      : logger(xnn_log_level_fatal, nullptr, file, line) {}
  ~fatal_logger() {
    flush();
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
inline logger log_error(const char* function, const char* file, int line) {
  return logger(xnn_log_level_error, function, file, line);
}
inline logger log_error(const char* file, int line) {
  return logger(xnn_log_level_error, nullptr, file, line);
}
#else
inline null_logger log_error(const char*, const char*, int) {
  return null_logger();
}
inline null_logger log_error(const char*, int) { return null_logger(); }
#endif  // YNN_LOG_LEVEL >= YNN_LOG_LEVEL_ERROR

#if YNN_LOG_LEVEL >= YNN_LOG_LEVEL_WARNING
inline logger log_warning(const char* function, const char* file, int line) {
  return logger(xnn_log_level_warning, function, file, line);
}
inline logger log_warning(const char* file, int line) {
  return logger(xnn_log_level_warning, nullptr, file, line);
}
#else
inline null_logger log_warning(const char*, const char*, int) {
  return null_logger();
}
inline null_logger log_warning(const char*, int) { return null_logger(); }
#endif  // YNN_LOG_LEVEL >= YNN_LOG_LEVEL_WARNING

#if YNN_LOG_LEVEL >= YNN_LOG_LEVEL_INFO
inline logger log_info(const char* function, const char* file, int line) {
  return logger(xnn_log_level_info, function, file, line);
}
inline logger log_info(const char* file, int line) {
  return logger(xnn_log_level_info, nullptr, file, line);
}
#else
inline null_logger log_info(const char*, const char*, int) {
  return null_logger();
}
inline null_logger log_info(const char*, int) { return null_logger(); }
#endif  // YNN_LOG_LEVEL >= YNN_LOG_LEVEL_INFO

#if YNN_LOG_LEVEL >= YNN_LOG_LEVEL_DEBUG
inline logger log_debug(const char* function, const char* file, int line) {
  return logger(xnn_log_level_debug, function, file, line);
}
inline logger log_debug(const char* file, int line) {
  return logger(xnn_log_level_debug, nullptr, file, line);
}
#else
inline null_logger log_debug(const char*, const char*, int) {
  return null_logger();
}
inline null_logger log_debug(const char*, int) { return null_logger(); }
#endif  // YNN_LOG_LEVEL >= YNN_LOG_LEVEL_DEBUG

#define YNN_LOG_FATAL() ynn::fatal_logger(__FUNCTION__, __FILE__, __LINE__)
#define YNN_LOG_ERROR() ynn::log_error(__FUNCTION__, __FILE__, __LINE__)
#define YNN_LOG_WARNING() ynn::log_warning(__FUNCTION__, __FILE__, __LINE__)
#define YNN_LOG_INFO() ynn::log_info(__FUNCTION__, __FILE__, __LINE__)
#define YNN_LOG_DEBUG() ynn::log_debug(__FUNCTION__, __FILE__, __LINE__)

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_LOG_H_
