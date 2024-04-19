// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <cmath>
#include <cstdlib>
#include <limits>

#include "unary-operator-tester.h"

namespace xnnpack {

class SquareRootOperatorTester : public UnaryOperatorTester {
 public:
  SquareRootOperatorTester() : UnaryOperatorTester() {
    range_f32_ = {0.0f, 0.5f};
    range_f16_ = {0.1f, 5.0f};
  }

 protected:
  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override { return std::sqrt(x); }

  // Computes the absolute tolerance for a reference value `y_ref`. Tests will
  // fail when `std::abs(y - y_ref) > AbsTol32(y_ref)`. Note that for `fp16`
  // tests, both `y` and `y_ref` will be converted to `float` for the tolerance
  // evaluation.
  float AbsTolF32(float y_ref) const override {
    return std::abs(y_ref) * 2.0f * std::numeric_limits<float>::epsilon();
  }
  float AbsTolF16(float y_ref) const override {
    return std::abs(y_ref) * 5.0e-3f;
  }

  CREATE_OP_OVERRIDES_F32(square_root);
  CREATE_OP_OVERRIDES_F16(square_root);
};

CREATE_UNARY_FLOAT_TESTS(F32, SquareRootOperatorTester);
CREATE_UNARY_FLOAT_TESTS(RunF32, SquareRootOperatorTester);
#ifndef XNN_EXCLUDE_F16_TESTS
CREATE_UNARY_FLOAT_TESTS(F16, SquareRootOperatorTester);
#endif  // XNN_EXCLUDE_F16_TESTS

};  // namespace xnnpack
