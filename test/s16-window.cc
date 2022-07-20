// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/s16-window.yaml
//   Generator: tools/generate-window-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/window.h>
#include "window-microkernel-tester.h"


TEST(S16_WINDOW__SCALAR_X1, channels_eq_1) {
  WindowMicrokernelTester()
    .rows(1)
    .channels(1)
    .Test(xnn_s16_window_ukernel__scalar_x1);
}

TEST(S16_WINDOW__SCALAR_X1, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x1);
  }
}

TEST(S16_WINDOW__SCALAR_X1, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__scalar_x1);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X1, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_window_ukernel__scalar_x1);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X1, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(1)
      .shift(shift)
      .Test(xnn_s16_window_ukernel__scalar_x1);
  }
}


TEST(S16_WINDOW__SCALAR_X2, channels_eq_2) {
  WindowMicrokernelTester()
    .rows(1)
    .channels(2)
    .Test(xnn_s16_window_ukernel__scalar_x2);
}

TEST(S16_WINDOW__SCALAR_X2, channels_div_2) {
  for (size_t channels = 4; channels < 20; channels += 2) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x2);
  }
}

TEST(S16_WINDOW__SCALAR_X2, channels_lt_2) {
  for (size_t channels = 1; channels < 2; channels++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x2);
  }
}

TEST(S16_WINDOW__SCALAR_X2, channels_gt_2) {
  for (size_t channels = 3; channels < 4; channels++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x2);
  }
}

TEST(S16_WINDOW__SCALAR_X2, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__scalar_x2);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X2, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_window_ukernel__scalar_x2);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X2, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(2)
      .shift(shift)
      .Test(xnn_s16_window_ukernel__scalar_x2);
  }
}


TEST(S16_WINDOW__SCALAR_X4, channels_eq_4) {
  WindowMicrokernelTester()
    .rows(1)
    .channels(4)
    .Test(xnn_s16_window_ukernel__scalar_x4);
}

TEST(S16_WINDOW__SCALAR_X4, channels_div_4) {
  for (size_t channels = 8; channels < 40; channels += 4) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x4);
  }
}

TEST(S16_WINDOW__SCALAR_X4, channels_lt_4) {
  for (size_t channels = 1; channels < 4; channels++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x4);
  }
}

TEST(S16_WINDOW__SCALAR_X4, channels_gt_4) {
  for (size_t channels = 5; channels < 8; channels++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x4);
  }
}

TEST(S16_WINDOW__SCALAR_X4, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__scalar_x4);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X4, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_window_ukernel__scalar_x4);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X4, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(4)
      .shift(shift)
      .Test(xnn_s16_window_ukernel__scalar_x4);
  }
}
