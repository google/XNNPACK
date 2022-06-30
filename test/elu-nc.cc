// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "elu-operator-tester.h"


TEST(ELU_NC_F16, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ELUOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestF16();
  }
}

TEST(ELU_NC_F16, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestF16();
  }
}

TEST(ELU_NC_F16, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestF16();
  }
}

TEST(ELU_NC_F16, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestF16();
  }
}

TEST(ELU_NC_F16, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestF16();
  }
}

TEST(ELU_NC_F16, small_batch_with_alpha) {
  for (size_t batch_size = 1; batch_size <= 3; batch_size += 2) {
    for (size_t channels = 1; channels < 100; channels += 15) {
      for (float alpha = 1.0e-4f; alpha < 1.0f; alpha *= 3.14159265f) {
        ELUOperatorTester()
          .batch_size(3)
          .channels(channels)
          .alpha(alpha)
          .iterations(1)
          .TestF16();
      }
    }
  }
}


TEST(ELU_NC_F32, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ELUOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestF32();
  }
}

TEST(ELU_NC_F32, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestF32();
  }
}

TEST(ELU_NC_F32, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestF32();
  }
}

TEST(ELU_NC_F32, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestF32();
  }
}

TEST(ELU_NC_F32, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestF32();
  }
}

TEST(ELU_NC_F32, small_batch_with_alpha) {
  for (size_t batch_size = 1; batch_size <= 3; batch_size += 2) {
    for (size_t channels = 1; channels < 100; channels += 15) {
      for (float alpha = 1.0e-4f; alpha < 1.0f; alpha *= 3.14159265f) {
        ELUOperatorTester()
          .batch_size(3)
          .channels(channels)
          .alpha(alpha)
          .iterations(1)
          .TestF32();
      }
    }
  }
}


TEST(ELU_NC_QS8, unit_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestQS8();
  }
}

TEST(ELU_NC_QS8, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(1)
      .channels(channels)
      .qmin(128)
      .iterations(3)
      .TestQS8();
  }
}

TEST(ELU_NC_QS8, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(1)
      .channels(channels)
      .qmax(128)
      .iterations(3)
      .TestQS8();
  }
}

TEST(ELU_NC_QS8, unit_batch_with_input_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float input_scale = 1.0e-2f; input_scale < 1.0e+2f; input_scale *= 10.0f) {
      ELUOperatorTester()
        .batch_size(1)
        .channels(channels)
        .input_scale(input_scale)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(ELU_NC_QS8, unit_batch_with_input_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      ELUOperatorTester()
        .batch_size(1)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(ELU_NC_QS8, small_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestQS8();
  }
}

TEST(ELU_NC_QS8, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestQS8();
  }
}

TEST(ELU_NC_QS8, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestQS8();
  }
}

TEST(ELU_NC_QS8, small_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .qmin(128)
      .iterations(3)
      .TestQS8();
  }
}

TEST(ELU_NC_QS8, small_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .qmax(128)
      .iterations(3)
      .TestQS8();
  }
}

TEST(ELU_NC_QS8, small_batch_with_input_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float input_scale = 1.0e-2f; input_scale < 1.0e+2f; input_scale *= 10.0f) {
      ELUOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_scale(input_scale)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(ELU_NC_QS8, small_batch_with_input_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      ELUOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(ELU_NC_QS8, small_batch_with_alpha) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float alpha = 1.0e-4f; alpha < 1.0f; alpha *= 3.14159265f) {
      ELUOperatorTester()
        .batch_size(3)
        .channels(channels)
        .alpha(alpha)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(ELU_NC_QS8, strided_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestQS8();
  }
}

TEST(ELU_NC_QS8, strided_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .qmin(128)
      .iterations(3)
      .TestQS8();
  }
}

TEST(ELU_NC_QS8, strided_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ELUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .qmax(128)
      .iterations(3)
      .TestQS8();
  }
}

TEST(ELU_NC_QS8, strided_batch_with_input_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float input_scale = 1.0e-2f; input_scale < 1.0e+2f; input_scale *= 10.0f) {
      ELUOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .output_stride(117)
        .input_scale(input_scale)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(ELU_NC_QS8, strided_batch_with_input_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      ELUOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .output_stride(117)
        .input_zero_point(uint8_t(input_zero_point))
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(ELU_NC_QS8, strided_batch_with_alpha) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float alpha = 1.0e-4f; alpha < 1.0f; alpha *= 3.14159265f) {
      ELUOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .output_stride(117)
        .alpha(alpha)
        .iterations(1)
        .TestQS8();
    }
  }
}
