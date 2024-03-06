// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "unary-operator-tester.h"

namespace xnnpack {

class SquareOperatorTester : public UnaryOperatorTester {
 public:
  SquareOperatorTester() : UnaryOperatorTester() {
    range_f32_ = {-1.0f, 1.0f};
    range_f16_ = {-1.0f, 1.0f};
  }

 protected:
  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override { return x * x; }

  // Computes the absolute tolerance for a reference value `y_ref`. Tests will
  // fail when `std::abs(y - y_ref) > AbsTol32(y_ref)`. Note that for `fp16`
  // tests, both `y` and `y_ref` will be converted to `float` for the tolerance
  // evaluation.
  float AbsTolF32(float y_ref) const override { return 0.0f; }
  float AbsTolF16(float y_ref) const override {
    return std::max(1.0e-4f, std::abs(y_ref) * 5.0e-3f);
  }

  CREATE_OP_OVERRIDES_F32(square);
  CREATE_OP_OVERRIDES_F16(square);
};

CREATE_UNARY_FLOAT_TESTS(F32, SquareOperatorTester);
CREATE_UNARY_FLOAT_TESTS(RunF32, SquareOperatorTester);
#ifndef XNN_EXCLUDE_F16_TESTS
CREATE_UNARY_FLOAT_TESTS(F16, SquareOperatorTester);
#endif  // XNN_EXCLUDE_F16_TESTS

};  // namespace xnnpack
