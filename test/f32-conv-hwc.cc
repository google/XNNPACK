// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/conv.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microparams-init.h"
#include "test/conv-hwc-microkernel-tester.h"

#define XNN_TEST_CONVHWC_INPUT_WIDTH_EQ(                                                                               \
  ukernel, arch_flags, kernel_sizes, subsamplings, padding_right_, padding_left_, input_channel, output_channel_tile,  \
  input_widths, init_params, ...)                                                                                      \
  TEST(ukernel, input_width_eq)                                                                                        \
  {                                                                                                                    \
    auto tester = ConvHWCMicrokernelTester()                                                                           \
                    .kernel_size(kernel_sizes)                                                                         \
                    .subsampling(subsamplings)                                                                         \
                    .input_channels(input_channel)                                                                     \
                    .output_channels_tile(output_channel_tile)                                                         \
                    .output_channels(output_channel_tile)                                                              \
                    .input_width(input_widths)                                                                         \
                    .input_height(kernel_sizes);                                                                       \
    if (padding_left_ == 1 && padding_right_ == 1) {                                                                   \
      tester.padding_width(padding_right_);                                                                            \
    }                                                                                                                  \
    else if (padding_left_ == 0 && padding_right_ == 1) {                                                              \
      tester.padding_right(padding_right_);                                                                            \
    }                                                                                                                  \
    tester.Test(ukernel, init_params);                                                                                 \
  }

#define XNN_TEST_CONVHWC_INPUT_WIDTH_DIV(                                                                              \
  ukernel, arch_flags, kernel_sizes, subsamplings, padding_right_, padding_left_, input_channel, output_channel_tile,  \
  input_widths, init_params, ...)                                                                                      \
  TEST(ukernel, input_width_div)                                                                                       \
  {                                                                                                                    \
    for (size_t input_widths_ = input_widths * 2; input_widths_ < (input_widths * 8);                                  \
         input_widths_ += input_widths * 3) {                                                                          \
      auto tester = ConvHWCMicrokernelTester()                                                                         \
                      .kernel_size(kernel_sizes)                                                                       \
                      .subsampling(subsamplings)                                                                       \
                      .input_channels(input_channel)                                                                   \
                      .output_channels_tile(output_channel_tile)                                                       \
                      .output_channels(output_channel_tile)                                                            \
                      .input_width(input_widths_)                                                                      \
                      .input_height(kernel_sizes);                                                                     \
      if (padding_left_ == 1 && padding_right_ == 1) {                                                                 \
        tester.padding_width(padding_right_);                                                                          \
      }                                                                                                                \
      else if (padding_left_ == 0 && padding_right_ == 1) {                                                            \
        tester.padding_right(padding_right_);                                                                          \
      }                                                                                                                \
      tester.Test(ukernel, init_params);                                                                               \
    }                                                                                                                  \
  }

#define XNN_TEST_CONVHWC_INPUT_WIDTH_LT(                                                                               \
  ukernel, arch_flags, kernel_sizes, subsamplings, padding_right_, padding_left_, input_channel, output_channel_tile,  \
  input_widths, init_params, ...)                                                                                      \
  TEST(ukernel, input_width_lt)                                                                                        \
  {                                                                                                                    \
    for (size_t input_widths_ = (padding_left_ ? 1 : 2); input_widths_ < input_widths; input_widths_++) {              \
      auto tester = ConvHWCMicrokernelTester()                                                                         \
                      .kernel_size(kernel_sizes)                                                                       \
                      .subsampling(subsamplings)                                                                       \
                      .input_channels(input_channel)                                                                   \
                      .output_channels_tile(output_channel_tile)                                                       \
                      .output_channels(output_channel_tile)                                                            \
                      .input_width(input_widths_)                                                                      \
                      .input_height(kernel_sizes);                                                                     \
      if (padding_left_ == 1 && padding_right_ == 1) {                                                                 \
        tester.padding_width(padding_right_);                                                                          \
      }                                                                                                                \
      else if (padding_left_ == 0 && padding_right_ == 1) {                                                            \
        tester.padding_right(padding_right_);                                                                          \
      }                                                                                                                \
      tester.Test(ukernel, init_params);                                                                               \
    }                                                                                                                  \
  }

#define XNN_TEST_CONVHWC_INPUT_WIDTH_GT(                                                                               \
  ukernel, arch_flags, kernel_sizes, subsamplings, padding_right_, padding_left_, input_channel, output_channel_tile,  \
  input_widths, init_params, ...)                                                                                      \
  TEST(ukernel, input_width_gt)                                                                                        \
  {                                                                                                                    \
    for (size_t input_widths_ = input_widths + 1; input_widths_ < input_widths * 2; input_widths_++) {                 \
      auto tester = ConvHWCMicrokernelTester()                                                                         \
                      .kernel_size(kernel_sizes)                                                                       \
                      .subsampling(subsamplings)                                                                       \
                      .input_channels(input_channel)                                                                   \
                      .output_channels_tile(output_channel_tile)                                                       \
                      .output_channels(output_channel_tile)                                                            \
                      .input_width(input_widths_)                                                                      \
                      .input_height(kernel_sizes);                                                                     \
      if (padding_left_ == 1 && padding_right_ == 1) {                                                                 \
        tester.padding_width(padding_right_);                                                                          \
      }                                                                                                                \
      else if (padding_left_ == 0 && padding_right_ == 1) {                                                            \
        tester.padding_right(padding_right_);                                                                          \
      }                                                                                                                \
      tester.Test(ukernel, init_params);                                                                               \
    }                                                                                                                  \
  }

#define XNN_TEST_CONVHWC_OUTPUT_CHANNELS_LT(                                                                           \
  ukernel, arch_flags, kernel_sizes, subsamplings, padding_right_, padding_left_, input_channel, output_channel_tile,  \
  input_widths, init_params, ...)                                                                                      \
  TEST(ukernel, output_channels_lt)                                                                                    \
  {                                                                                                                    \
    for (size_t output_channels = 1; output_channels < output_channel_tile; output_channels++) {                       \
      for (size_t input_widths_ = (padding_left_ ? 1 : 2); input_widths_ < input_widths * 8;                           \
           input_widths_ += (input_widths * 2 - 1)) {                                                                  \
        auto tester = ConvHWCMicrokernelTester()                                                                       \
                        .kernel_size(kernel_sizes)                                                                     \
                        .subsampling(subsamplings)                                                                     \
                        .input_channels(input_channel)                                                                 \
                        .output_channels_tile(output_channel_tile)                                                     \
                        .output_channels(output_channels)                                                              \
                        .input_width(input_widths_)                                                                    \
                        .input_height(kernel_sizes);                                                                   \
        if (padding_left_ == 1 && padding_right_ == 1) {                                                               \
          tester.padding_width(padding_right_);                                                                        \
        }                                                                                                              \
        else if (padding_left_ == 0 && padding_right_ == 1) {                                                          \
          tester.padding_right(padding_right_);                                                                        \
        }                                                                                                              \
        tester.Test(ukernel, init_params);                                                                             \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_CONVHWC_OUTPUT_CHANNELS_DIV(                                                                          \
  ukernel, arch_flags, kernel_sizes, subsamplings, padding_right_, padding_left_, input_channel, output_channel_tile,  \
  input_widths, init_params, ...)                                                                                      \
  TEST(ukernel, output_channels_div)                                                                                   \
  {                                                                                                                    \
    for (size_t output_channels = output_channel_tile * 2; output_channels <= output_channel_tile * 4;                 \
         output_channels += output_channel_tile) {                                                                     \
      for (size_t input_widths_ = (padding_left_ ? 1 : 2); input_widths_ < input_widths * 8;                           \
           input_widths_ += (input_widths * 2 - 1)) {                                                                  \
        auto tester = ConvHWCMicrokernelTester()                                                                       \
                        .kernel_size(kernel_sizes)                                                                     \
                        .subsampling(subsamplings)                                                                     \
                        .input_channels(input_channel)                                                                 \
                        .output_channels_tile(output_channel_tile)                                                     \
                        .output_channels(output_channels)                                                              \
                        .input_width(input_widths_)                                                                    \
                        .input_height(kernel_sizes);                                                                   \
        if (padding_left_ == 1 && padding_right_ == 1) {                                                               \
          tester.padding_width(padding_right_);                                                                        \
        }                                                                                                              \
        else if (padding_left_ == 0 && padding_right_ == 1) {                                                          \
          tester.padding_right(padding_right_);                                                                        \
        }                                                                                                              \
        tester.Test(ukernel, init_params);                                                                             \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_CONVHWC_OUTPUT_CHANNELS_GT(                                                                           \
  ukernel, arch_flags, kernel_sizes, subsamplings, padding_right_, padding_left_, input_channel, output_channel_tile,  \
  input_widths, init_params, ...)                                                                                      \
  TEST(ukernel, output_channels_gt)                                                                                    \
  {                                                                                                                    \
    for (size_t output_channels = output_channel_tile + 1; output_channels < output_channel_tile * 2;                  \
         output_channels++) {                                                                                          \
      for (size_t input_widths_ = (padding_left_ ? 1 : 2); input_widths_ < input_widths * 8;                           \
           input_widths_ += (input_widths * 2 - 1)) {                                                                  \
        auto tester = ConvHWCMicrokernelTester()                                                                       \
                        .kernel_size(kernel_sizes)                                                                     \
                        .subsampling(subsamplings)                                                                     \
                        .input_channels(input_channel)                                                                 \
                        .output_channels_tile(output_channel_tile)                                                     \
                        .output_channels(output_channels)                                                              \
                        .input_width(input_widths_)                                                                    \
                        .input_height(kernel_sizes);                                                                   \
        if (padding_left_ == 1 && padding_right_ == 1) {                                                               \
          tester.padding_width(padding_right_);                                                                        \
        }                                                                                                              \
        else if (padding_left_ == 0 && padding_right_ == 1) {                                                          \
          tester.padding_right(padding_right_);                                                                        \
        }                                                                                                              \
        tester.Test(ukernel, init_params);                                                                             \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_CONVHWC_INPUT_HEIGHT_LT(                                                                              \
  ukernel, arch_flags, kernel_sizes, subsamplings, padding_right_, padding_left_, input_channel, output_channel_tile,  \
  input_widths, init_params, ...)                                                                                      \
  TEST(ukernel, input_height_lt)                                                                                       \
  {                                                                                                                    \
    for (size_t input_heights = 1; input_heights < 3; input_heights++) {                                               \
      for (size_t output_channels = 1; output_channels < output_channel_tile * 2;                                      \
           output_channels += output_channel_tile - 1) {                                                               \
        for (size_t input_widths_ = (padding_left_ ? 1 : 2); input_widths_ < input_widths * 8;                         \
             input_widths_ += (input_widths * 2 - 1)) {                                                                \
          auto tester = ConvHWCMicrokernelTester()                                                                     \
                          .kernel_size(kernel_sizes)                                                                   \
                          .subsampling(subsamplings)                                                                   \
                          .input_channels(input_channel)                                                               \
                          .output_channels_tile(output_channel_tile)                                                   \
                          .output_channels(output_channels)                                                            \
                          .input_width(input_widths_)                                                                  \
                          .input_height(input_heights);                                                                \
          if (padding_left_ == 0) {                                                                                    \
            tester.padding_right(1);                                                                                   \
            tester.padding_height(1);                                                                                  \
          }                                                                                                            \
          else {                                                                                                       \
            tester.padding(1);                                                                                         \
          }                                                                                                            \
          tester.Test(ukernel, init_params);                                                                           \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_CONVHWC_INPUT_HEIGHT_GT(                                                                              \
  ukernel, arch_flags, kernel_sizes, subsamplings, padding_right_, padding_left_, input_channel, output_channel_tile,  \
  input_widths, init_params, ...)                                                                                      \
  TEST(ukernel, input_height_gt)                                                                                       \
  {                                                                                                                    \
    for (size_t input_heights = 4; input_heights <= 9; input_heights++) {                                              \
      for (size_t output_channels = 1; output_channels < output_channel_tile * 2;                                      \
           output_channels += output_channel_tile - 1) {                                                               \
        for (size_t input_widths_ = (padding_left_ ? 1 : 2); input_widths_ < input_widths * 8;                         \
             input_widths_ += (input_widths * 2 - 1)) {                                                                \
          auto tester = ConvHWCMicrokernelTester()                                                                     \
                          .kernel_size(kernel_sizes)                                                                   \
                          .subsampling(subsamplings)                                                                   \
                          .input_channels(input_channel)                                                               \
                          .output_channels_tile(output_channel_tile)                                                   \
                          .output_channels(output_channels)                                                            \
                          .input_width(input_widths_)                                                                  \
                          .input_height(input_heights);                                                                \
          if (padding_left_ == 1 && padding_right_ == 1) {                                                             \
            tester.padding_width(padding_right_);                                                                      \
          }                                                                                                            \
          else if (padding_left_ == 0 && padding_right_ == 1) {                                                        \
            tester.padding_right(padding_right_);                                                                      \
          }                                                                                                            \
          tester.Test(ukernel, init_params);                                                                           \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_CONVHWC_PADDING_TOP(                                                                                  \
  ukernel, arch_flags, kernel_sizes, subsamplings, padding_right_, padding_left_, input_channel, output_channel_tile,  \
  input_widths, init_params, ...)                                                                                      \
  TEST(ukernel, padding_top)                                                                                           \
  {                                                                                                                    \
    for (size_t padding_tops = 0; padding_tops <= 1; padding_tops++) {                                                 \
      for (size_t output_channels = 1; output_channels < output_channel_tile * 2;                                      \
           output_channels += output_channel_tile - 1) {                                                               \
        for (size_t input_widths_ = (padding_left_ ? 1 : 2); input_widths_ < input_widths * 8;                         \
             input_widths_ += (input_widths * 2 - 1)) {                                                                \
          auto tester = ConvHWCMicrokernelTester()                                                                     \
                          .kernel_size(kernel_sizes)                                                                   \
                          .subsampling(subsamplings)                                                                   \
                          .input_channels(input_channel)                                                               \
                          .output_channels_tile(output_channel_tile)                                                   \
                          .output_channels(output_channels)                                                            \
                          .input_width(input_widths_)                                                                  \
                          .input_height(9);                                                                            \
          if (padding_left_ == 1 && padding_right_ == 1) {                                                             \
            tester.padding_width(padding_right_);                                                                      \
          }                                                                                                            \
          else if (padding_left_ == 0 && padding_right_ == 1) {                                                        \
            tester.padding_right(padding_right_);                                                                      \
          }                                                                                                            \
          tester.padding_top(padding_tops).Test(ukernel, init_params);                                                 \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_CONVHWC_PADDING_BOTTOM(                                                                               \
  ukernel, arch_flags, kernel_sizes, subsamplings, padding_right_, padding_left_, input_channel, output_channel_tile,  \
  input_widths, init_params, ...)                                                                                      \
  TEST(ukernel, padding_bottom)                                                                                        \
  {                                                                                                                    \
    for (size_t padding_bottoms = 0; padding_bottoms <= 1; padding_bottoms++) {                                        \
      for (size_t output_channels = 1; output_channels < output_channel_tile * 2;                                      \
           output_channels += output_channel_tile - 1) {                                                               \
        for (size_t input_widths_ = (padding_left_ ? 1 : 2); input_widths_ < input_widths * 8;                         \
             input_widths_ += (input_widths * 2 - 1)) {                                                                \
          auto tester = ConvHWCMicrokernelTester()                                                                     \
                          .kernel_size(kernel_sizes)                                                                   \
                          .subsampling(subsamplings)                                                                   \
                          .input_channels(input_channel)                                                               \
                          .output_channels_tile(output_channel_tile)                                                   \
                          .output_channels(output_channels)                                                            \
                          .input_width(input_widths_)                                                                  \
                          .input_height(9);                                                                            \
          if (padding_left_ == 1 && padding_right_ == 1) {                                                             \
            tester.padding_width(padding_right_);                                                                      \
          }                                                                                                            \
          else if (padding_left_ == 0 && padding_right_ == 1) {                                                        \
            tester.padding_right(padding_right_);                                                                      \
          }                                                                                                            \
          tester.padding_bottom(padding_bottoms).Test(ukernel, init_params);                                           \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_CONVHWC_OUTPUT_Y_START(                                                                               \
  ukernel, arch_flags, kernel_sizes, subsamplings, padding_right_, padding_left_, input_channel, output_channel_tile,  \
  input_widths, init_params, ...)                                                                                      \
  TEST(ukernel, output_y_start)                                                                                        \
  {                                                                                                                    \
    for (size_t output_y_starts = 1; output_y_starts <= 3; output_y_starts++) {                                        \
      for (size_t output_channels = 1; output_channels < output_channel_tile * 2;                                      \
           output_channels += output_channel_tile - 1) {                                                               \
        for (size_t input_widths_ = (padding_left_ ? 1 : 2); input_widths_ < input_widths * 8;                         \
             input_widths_ += (input_widths * 2 - 1)) {                                                                \
          auto tester = ConvHWCMicrokernelTester()                                                                     \
                          .kernel_size(kernel_sizes)                                                                   \
                          .subsampling(subsamplings)                                                                   \
                          .input_channels(input_channel)                                                               \
                          .output_channels_tile(output_channel_tile)                                                   \
                          .output_channels(output_channels)                                                            \
                          .input_width(input_widths_)                                                                  \
                          .input_height(9)                                                                             \
                          .output_y_start(output_y_starts);                                                            \
          if (padding_left_ == 1 && padding_right_ == 1) {                                                             \
            tester.padding_width(padding_right_);                                                                      \
          }                                                                                                            \
          else if (padding_left_ == 0 && padding_right_ == 1) {                                                        \
            tester.padding_right(padding_right_);                                                                      \
          }                                                                                                            \
          tester.Test(ukernel, init_params);                                                                           \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_CONVHWC_OUTPUT_Y_END(                                                                                 \
  ukernel, arch_flags, kernel_sizes, subsamplings, padding_right_, padding_left_, input_channel, output_channel_tile,  \
  input_widths, init_params, ...)                                                                                      \
  TEST(ukernel, output_y_end)                                                                                          \
  {                                                                                                                    \
    for (size_t output_y_ends = 2; output_y_ends < 5; output_y_ends++) {                                               \
      for (size_t output_channels = 1; output_channels < output_channel_tile * 2;                                      \
           output_channels += output_channel_tile - 1) {                                                               \
        for (size_t input_widths_ = (padding_left_ ? 1 : 2); input_widths_ < input_widths * 8;                         \
             input_widths_ += (input_widths * 2 - 1)) {                                                                \
          auto tester = ConvHWCMicrokernelTester()                                                                     \
                          .kernel_size(kernel_sizes)                                                                   \
                          .subsampling(subsamplings)                                                                   \
                          .input_channels(input_channel)                                                               \
                          .output_channels_tile(output_channel_tile)                                                   \
                          .output_channels(output_channels)                                                            \
                          .input_width(input_widths_)                                                                  \
                          .input_height(9)                                                                             \
                          .output_y_end(output_y_ends);                                                                \
          if (padding_left_ == 1 && padding_right_ == 1) {                                                             \
            tester.padding_width(padding_right_);                                                                      \
          }                                                                                                            \
          else if (padding_left_ == 0 && padding_right_ == 1) {                                                        \
            tester.padding_right(padding_right_);                                                                      \
          }                                                                                                            \
          tester.Test(ukernel, init_params);                                                                           \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_CONVHWC_QMIN(                                                                                         \
  ukernel, arch_flags, kernel_sizes, subsamplings, padding_right_, padding_left_, input_channel, output_channel_tile,  \
  input_widths, init_params, ...)                                                                                      \
  TEST(ukernel, qmin)                                                                                                  \
  {                                                                                                                    \
    for (size_t output_channels = 1; output_channels < output_channel_tile * 2;                                        \
         output_channels += output_channel_tile - 1) {                                                                 \
      for (size_t input_widths_ = (padding_left_ ? 1 : 2); input_widths_ < input_widths * 8;                           \
           input_widths_ += (input_widths * 2 - 1)) {                                                                  \
        auto tester = ConvHWCMicrokernelTester()                                                                       \
                        .kernel_size(kernel_sizes)                                                                     \
                        .subsampling(subsamplings)                                                                     \
                        .input_channels(input_channel)                                                                 \
                        .output_channels_tile(output_channel_tile)                                                     \
                        .output_channels(output_channels)                                                              \
                        .input_width(input_widths_)                                                                    \
                        .input_height(6)                                                                               \
                        .qmin(128);                                                                                    \
        if (padding_left_ == 1 && padding_right_ == 1) {                                                               \
          tester.padding_width(padding_right_);                                                                        \
        }                                                                                                              \
        else if (padding_left_ == 0 && padding_right_ == 1) {                                                          \
          tester.padding_right(padding_right_);                                                                        \
        }                                                                                                              \
        tester.Test(ukernel, init_params);                                                                             \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_CONVHWC_QMAX(                                                                                         \
  ukernel, arch_flags, kernel_sizes, subsamplings, padding_right_, padding_left_, input_channel, output_channel_tile,  \
  input_widths, init_params, ...)                                                                                      \
  TEST(ukernel, qmax)                                                                                                  \
  {                                                                                                                    \
    for (size_t output_channels = 1; output_channels < output_channel_tile * 2;                                        \
         output_channels += output_channel_tile - 1) {                                                                 \
      for (size_t input_widths_ = (padding_left_ ? 1 : 2); input_widths_ < input_widths * 8;                           \
           input_widths_ += (input_widths * 2 - 1)) {                                                                  \
        auto tester = ConvHWCMicrokernelTester()                                                                       \
                        .kernel_size(kernel_sizes)                                                                     \
                        .subsampling(subsamplings)                                                                     \
                        .input_channels(input_channel)                                                                 \
                        .output_channels_tile(output_channel_tile)                                                     \
                        .output_channels(output_channels)                                                              \
                        .input_width(input_widths_)                                                                    \
                        .input_height(6)                                                                               \
                        .qmax(128);                                                                                    \
        if (padding_left_ == 1 && padding_right_ == 1) {                                                               \
          tester.padding_width(padding_right_);                                                                        \
        }                                                                                                              \
        else if (padding_left_ == 0 && padding_right_ == 1) {                                                          \
          tester.padding_right(padding_right_);                                                                        \
        }                                                                                                              \
        tester.Test(ukernel, init_params);                                                                             \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_UKERNEL_WITH_PARAMS(                                                                                       \
  arch_flags, ukernel, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,    \
  input_widths, datatype, params_type, init_params)                                                                    \
  XNN_TEST_CONVHWC_INPUT_WIDTH_EQ(                                                                                     \
    ukernel, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,  \
    input_widths, init_params);                                                                                        \
  XNN_TEST_CONVHWC_INPUT_WIDTH_DIV(                                                                                    \
    ukernel, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,  \
    input_widths, init_params);                                                                                        \
  XNN_TEST_CONVHWC_INPUT_WIDTH_LT(                                                                                     \
    ukernel, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,  \
    input_widths, init_params);                                                                                        \
  XNN_TEST_CONVHWC_INPUT_WIDTH_GT(                                                                                     \
    ukernel, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,  \
    input_widths, init_params);                                                                                        \
  XNN_TEST_CONVHWC_OUTPUT_CHANNELS_LT(                                                                                 \
    ukernel, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,  \
    input_widths, init_params);                                                                                        \
  XNN_TEST_CONVHWC_OUTPUT_CHANNELS_DIV(                                                                                \
    ukernel, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,  \
    input_widths, init_params);                                                                                        \
  XNN_TEST_CONVHWC_OUTPUT_CHANNELS_GT(                                                                                 \
    ukernel, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,  \
    input_widths, init_params);                                                                                        \
  XNN_TEST_CONVHWC_INPUT_HEIGHT_LT(                                                                                    \
    ukernel, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,  \
    input_widths, init_params);                                                                                        \
  XNN_TEST_CONVHWC_INPUT_HEIGHT_GT(                                                                                    \
    ukernel, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,  \
    input_widths, init_params);                                                                                        \
  XNN_TEST_CONVHWC_PADDING_TOP(                                                                                        \
    ukernel, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,  \
    input_widths, init_params);                                                                                        \
  XNN_TEST_CONVHWC_PADDING_BOTTOM(                                                                                     \
    ukernel, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,  \
    input_widths, init_params);                                                                                        \
  XNN_TEST_CONVHWC_OUTPUT_Y_START(                                                                                     \
    ukernel, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,  \
    input_widths, init_params);                                                                                        \
  XNN_TEST_CONVHWC_OUTPUT_Y_END(                                                                                       \
    ukernel, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,  \
    input_widths, init_params);                                                                                        \
  XNN_TEST_CONVHWC_QMIN(                                                                                               \
    ukernel, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,  \
    input_widths, init_params);                                                                                        \
  XNN_TEST_CONVHWC_QMAX(                                                                                               \
    ukernel, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,  \
    input_widths, init_params);
#include "f32-conv-hwc/f32-conv-hwc.h"
#undef XNN_UKERNEL_WITH_PARAMS
