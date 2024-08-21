// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "unary-operator-tester.h"
#include "pthreadpool.h"

namespace xnnpack {


class ClzOperatorTester : public UnaryOperatorTester {
 public:
  ClzOperatorTester() : UnaryOperatorTester() {
    range_s32_ = {std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()};
  }

 protected:
  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override {
    return 0;
  }
  int32_t RefFunc(int32_t x) const override {
    int32_t clz = 0;
    int32_t value = x;
    if (value == 0)
      clz = 32;
    else if (value < 0)
      clz = 0;
    else {
      while ((value & 0x80000000) == 0) {
        clz++;
        value <<= 1;
      }
    }
    return clz;
  }

  CREATE_OP_OVERRIDES_S32(clz);
};

CREATE_UNARY_INT32_TESTS(S32, ClzOperatorTester);

};  // namespace xnnpack
