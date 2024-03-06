// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include <cmath>

#include "unary-operator-tester.h"

namespace xnnpack {

class FloorOperatorTester : public UnaryOperatorTester {
 public:
  FloorOperatorTester() : UnaryOperatorTester() {
    range_f32_ = {0.0f, 5.0f};
    range_f16_ = {0.0f, 5.0f};
  }

 protected:
  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override { return std::floor(x); }

  CREATE_OP_OVERRIDES_F32(floor);
  CREATE_OP_OVERRIDES_F16(floor);
};

CREATE_UNARY_FLOAT_TESTS(F32, FloorOperatorTester);
CREATE_UNARY_FLOAT_TESTS(RunF32, FloorOperatorTester);
#ifndef XNN_EXCLUDE_F16_TESTS
CREATE_UNARY_FLOAT_TESTS(F16, FloorOperatorTester);
#endif  // XNN_EXCLUDE_F16_TESTS

};  // namespace xnnpack
