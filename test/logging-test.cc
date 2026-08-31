// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/log.h"
#include "ynnpack/base/log.h"

namespace {

struct LogEntry {
  enum xnn_log_level level;
  std::string file;
  int line;
  std::string message;
};

class LoggingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    entries_.clear();
    xnn_set_log_callback(&TestLogCallback, this);
  }

  void TearDown() override { xnn_set_log_callback(nullptr, nullptr); }

  static void TestLogCallback(enum xnn_log_level level, const char* file,
                              int line, const char* message, void* user_data) {
    auto* self = static_cast<LoggingTest*>(user_data);
    self->entries_.push_back(LogEntry{
        level,
        file != nullptr ? file : "",
        line,
        message != nullptr ? message : "",
    });
  }

  std::vector<LogEntry> entries_;
};

TEST_F(LoggingTest, GetSetCallback) {
  EXPECT_EQ(xnn_get_log_callback(), &TestLogCallback);
  EXPECT_EQ(xnn_get_log_callback_user_data(), this);

  xnn_set_log_callback(nullptr, nullptr);
  EXPECT_EQ(xnn_get_log_callback(), nullptr);
  EXPECT_EQ(xnn_get_log_callback_user_data(), nullptr);
}

TEST_F(LoggingTest, XnnpackLoggingCallback) {
#if XNN_LOG_LEVEL >= XNN_LOG_ERROR
  xnn_log_error("Test error with int %d and string %s", 123, "hello");
  ASSERT_GE(entries_.size(), 1);
  EXPECT_EQ(entries_.back().level, xnn_log_level_error);
  EXPECT_NE(
      entries_.back().message.find("Test error with int 123 and string hello"),
      std::string::npos);
#endif

#if XNN_LOG_LEVEL >= XNN_LOG_WARNING
  xnn_log_warning("Test warning %d", 456);
  ASSERT_GE(entries_.size(), 2);
  EXPECT_EQ(entries_.back().level, xnn_log_level_warning);
  EXPECT_NE(entries_.back().message.find("Test warning 456"),
            std::string::npos);
#endif

#if XNN_LOG_LEVEL >= XNN_LOG_INFO
  xnn_log_info("Test info %d", 789);
  ASSERT_GE(entries_.size(), 3);
  EXPECT_EQ(entries_.back().level, xnn_log_level_info);
  EXPECT_NE(entries_.back().message.find("Test info 789"), std::string::npos);
#endif

#if XNN_LOG_LEVEL >= XNN_LOG_DEBUG
  xnn_log_debug("Test debug %d", 101112);
  ASSERT_GE(entries_.size(), 4);
  EXPECT_EQ(entries_.back().level, xnn_log_level_debug);
  EXPECT_NE(entries_.back().message.find("Test debug 101112"),
            std::string::npos);
#endif
}

TEST_F(LoggingTest, YnnpackLoggingCallback) {
#if YNN_LOG_LEVEL >= YNN_LOG_LEVEL_ERROR
  size_t prev_count = entries_.size();
  YNN_LOG_ERROR() << "YNN error message: " << 42;
  ASSERT_GT(entries_.size(), prev_count);
  EXPECT_EQ(entries_.back().level, xnn_log_level_error);
  EXPECT_NE(entries_.back().message.find("YNN error message: 42"),
            std::string::npos);
  EXPECT_NE(entries_.back().file.find("logging-test.cc"), std::string::npos);
  EXPECT_GT(entries_.back().line, 0);
#endif

#if YNN_LOG_LEVEL >= YNN_LOG_LEVEL_WARNING
  prev_count = entries_.size();
  YNN_LOG_WARNING() << "YNN warning message: " << 99;
  ASSERT_GT(entries_.size(), prev_count);
  EXPECT_EQ(entries_.back().level, xnn_log_level_warning);
  EXPECT_NE(entries_.back().message.find("YNN warning message: 99"),
            std::string::npos);
#endif

#if YNN_LOG_LEVEL >= YNN_LOG_LEVEL_INFO
  prev_count = entries_.size();
  YNN_LOG_INFO() << "YNN info message: " << 100;
  ASSERT_GT(entries_.size(), prev_count);
  EXPECT_EQ(entries_.back().level, xnn_log_level_info);
  EXPECT_NE(entries_.back().message.find("YNN info message: 100"),
            std::string::npos);
#endif

#if YNN_LOG_LEVEL >= YNN_LOG_LEVEL_DEBUG
  prev_count = entries_.size();
  YNN_LOG_DEBUG() << "YNN debug message: " << 200;
  ASSERT_GT(entries_.size(), prev_count);
  EXPECT_EQ(entries_.back().level, xnn_log_level_debug);
  EXPECT_NE(entries_.back().message.find("YNN debug message: 200"),
            std::string::npos);
#endif
}

}  // namespace
