// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "unary-operator-tester.h"

namespace xnnpack {

class LogOperatorTester : public UnaryOperatorTester {
 public:
  LogOperatorTester() : UnaryOperatorTester() {
    range_f32_ = {0.f, 10.0f};
    range_f16_ = {0.f, 10.0f};
  }

 protected:
  float AbsTolF32(float y_ref) const override {
    return std::max(
        2 * std::numeric_limits<float>::epsilon(),
        std::abs(y_ref) * 6 * std::numeric_limits<float>::epsilon());
  };

  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override { return std::log(x); }

  CREATE_STANDARD_OP_OVERRIDES_F32(log);
};

CREATE_UNARY_FLOAT_TESTS(F32, LogOperatorTester);

};  // namespace xnnpack
