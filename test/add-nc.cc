// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "add-operator-tester.h"


TEST(ADD_NC_Q8, unit_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestQ8();
  }
}

TEST(ADD_NC_Q8, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(1)
      .channels(channels)
      .qmin(128)
      .iterations(3)
      .TestQ8();
  }
}

TEST(ADD_NC_Q8, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(1)
      .channels(channels)
      .qmax(128)
      .iterations(3)
      .TestQ8();
  }
}

TEST(ADD_NC_Q8, unit_batch_with_a_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float a_scale = 1.0e-2f; a_scale < 1.0e+2f; a_scale *= 10.0f) {
      AddOperatorTester()
        .batch_size(1)
        .channels(channels)
        .a_scale(a_scale)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, unit_batch_with_b_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float b_scale = 1.0e-2f; b_scale < 1.0e+2f; b_scale *= 10.0f) {
      AddOperatorTester()
        .batch_size(1)
        .channels(channels)
        .b_scale(b_scale)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, unit_batch_with_y_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float y_scale = 1.0e-2f; y_scale < 1.0e+2f; y_scale *= 10.0f) {
      AddOperatorTester()
        .batch_size(1)
        .channels(channels)
        .y_scale(y_scale)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, unit_batch_with_a_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t a_zero_point = 0; a_zero_point <= 255; a_zero_point += 51) {
      AddOperatorTester()
        .batch_size(1)
        .channels(channels)
        .a_zero_point(uint8_t(a_zero_point))
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, unit_batch_with_b_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t b_zero_point = 0; b_zero_point <= 255; b_zero_point += 51) {
      AddOperatorTester()
        .batch_size(1)
        .channels(channels)
        .b_zero_point(uint8_t(b_zero_point))
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, unit_batch_with_y_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
      AddOperatorTester()
        .batch_size(1)
        .channels(channels)
        .y_zero_point(uint8_t(y_zero_point))
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, small_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestQ8();
  }
}

TEST(ADD_NC_Q8, small_batch_with_a_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .a_stride(129)
      .iterations(3)
      .TestQ8();
  }
}

TEST(ADD_NC_Q8, small_batch_with_b_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .b_stride(123)
      .iterations(3)
      .TestQ8();
  }
}

TEST(ADD_NC_Q8, small_batch_with_y_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .y_stride(117)
      .iterations(3)
      .TestQ8();
  }
}

TEST(ADD_NC_Q8, small_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .qmin(128)
      .iterations(3)
      .TestQ8();
  }
}

TEST(ADD_NC_Q8, small_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .qmax(128)
      .iterations(3)
      .TestQ8();
  }
}

TEST(ADD_NC_Q8, small_batch_with_a_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float a_scale = 1.0e-2f; a_scale < 1.0e+2f; a_scale *= 10.0f) {
      AddOperatorTester()
        .batch_size(3)
        .channels(channels)
        .a_scale(a_scale)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, small_batch_with_b_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float b_scale = 1.0e-2f; b_scale < 1.0e+2f; b_scale *= 10.0f) {
      AddOperatorTester()
        .batch_size(3)
        .channels(channels)
        .b_scale(b_scale)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, small_batch_with_y_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float y_scale = 1.0e-2f; y_scale < 1.0e+2f; y_scale *= 10.0f) {
      AddOperatorTester()
        .batch_size(3)
        .channels(channels)
        .y_scale(y_scale)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, small_batch_with_a_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t a_zero_point = 0; a_zero_point <= 255; a_zero_point += 51) {
      AddOperatorTester()
        .batch_size(3)
        .channels(channels)
        .a_zero_point(uint8_t(a_zero_point))
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, small_batch_with_b_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t b_zero_point = 0; b_zero_point <= 255; b_zero_point += 51) {
      AddOperatorTester()
        .batch_size(3)
        .channels(channels)
        .b_zero_point(uint8_t(b_zero_point))
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, small_batch_with_y_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
      AddOperatorTester()
        .batch_size(3)
        .channels(channels)
        .y_zero_point(uint8_t(y_zero_point))
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, strided_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .a_stride(129)
      .b_stride(123)
      .y_stride(117)
      .iterations(3)
      .TestQ8();
  }
}

TEST(ADD_NC_Q8, strided_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .a_stride(129)
      .b_stride(123)
      .y_stride(117)
      .qmin(128)
      .iterations(3)
      .TestQ8();
  }
}

TEST(ADD_NC_Q8, strided_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .a_stride(129)
      .b_stride(123)
      .y_stride(117)
      .qmax(128)
      .iterations(3)
      .TestQ8();
  }
}

TEST(ADD_NC_Q8, strided_batch_with_a_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float a_scale = 1.0e-2f; a_scale < 1.0e+2f; a_scale *= 10.0f) {
      AddOperatorTester()
        .batch_size(3)
        .channels(channels)
        .a_stride(129)
        .b_stride(123)
        .y_stride(117)
        .a_scale(a_scale)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, strided_batch_with_b_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float b_scale = 1.0e-2f; b_scale < 1.0e+2f; b_scale *= 10.0f) {
      AddOperatorTester()
        .batch_size(3)
        .channels(channels)
        .a_stride(129)
        .b_stride(123)
        .y_stride(117)
        .b_scale(b_scale)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, strided_batch_with_y_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float y_scale = 1.0e-2f; y_scale < 1.0e+2f; y_scale *= 10.0f) {
      AddOperatorTester()
        .batch_size(3)
        .channels(channels)
        .a_stride(129)
        .b_stride(123)
        .y_stride(117)
        .y_scale(y_scale)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, strided_batch_with_a_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t a_zero_point = 0; a_zero_point <= 255; a_zero_point += 51) {
      AddOperatorTester()
        .batch_size(3)
        .channels(channels)
        .a_stride(129)
        .b_stride(123)
        .y_stride(117)
        .a_zero_point(uint8_t(a_zero_point))
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, strided_batch_with_b_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t b_zero_point = 0; b_zero_point <= 255; b_zero_point += 51) {
      AddOperatorTester()
        .batch_size(3)
        .channels(channels)
        .a_stride(129)
        .b_stride(123)
        .y_stride(117)
        .b_zero_point(uint8_t(b_zero_point))
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_Q8, strided_batch_with_y_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
      AddOperatorTester()
        .batch_size(3)
        .channels(channels)
        .a_stride(129)
        .b_stride(123)
        .y_stride(117)
        .y_zero_point(uint8_t(y_zero_point))
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(ADD_NC_F32, unit_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestF32();
  }
}

TEST(ADD_NC_F32, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(1)
      .channels(channels)
      .qmin(128)
      .iterations(3)
      .TestF32();
  }
}

TEST(ADD_NC_F32, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(1)
      .channels(channels)
      .qmax(128)
      .iterations(3)
      .TestF32();
  }
}

TEST(ADD_NC_F32, small_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestF32();
  }
}

TEST(ADD_NC_F32, small_batch_with_a_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .a_stride(129)
      .iterations(3)
      .TestF32();
  }
}

TEST(ADD_NC_F32, small_batch_with_b_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .b_stride(123)
      .iterations(3)
      .TestF32();
  }
}

TEST(ADD_NC_F32, small_batch_with_y_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .y_stride(117)
      .iterations(3)
      .TestF32();
  }
}

TEST(ADD_NC_F32, small_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .qmin(128)
      .iterations(3)
      .TestF32();
  }
}

TEST(ADD_NC_F32, small_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .qmax(128)
      .iterations(3)
      .TestF32();
  }
}

TEST(ADD_NC_F32, strided_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .a_stride(129)
      .b_stride(123)
      .y_stride(117)
      .iterations(3)
      .TestF32();
  }
}

TEST(ADD_NC_F32, strided_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .a_stride(129)
      .b_stride(123)
      .y_stride(117)
      .qmin(128)
      .iterations(3)
      .TestF32();
  }
}

TEST(ADD_NC_F32, strided_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
      .batch_size(3)
      .channels(channels)
      .a_stride(129)
      .b_stride(123)
      .y_stride(117)
      .qmax(128)
      .iterations(3)
      .TestF32();
  }
}
