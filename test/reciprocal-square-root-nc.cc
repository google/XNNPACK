// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <cmath>
#include <cstdlib>
#include <limits>

#include "unary-operator-tester.h"

namespace xnnpack {

class ReciprocalSquareRootOperatorTester : public UnaryOperatorTester {
 public:
  ReciprocalSquareRootOperatorTester() : UnaryOperatorTester() {
    range_f32_ = {0.001f, 5.0f};
    range_f16_ = {0.001f, 5.0f};
  }

 protected:
  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override { return 1.0f / std::sqrt(x); }

  // Computes the absolute tolerance for a reference value `y_ref`. Tests will
  // fail when `std::abs(y - y_ref) > AbsTol32(y_ref)`. Note that for `fp16`
  // tests, both `y` and `y_ref` will be converted to `float` for the tolerance
  // evaluation.
  float AbsTolF32(float y_ref) const override {
    return y_ref * std::numeric_limits<float>::epsilon() * 2;
  }
  float AbsTolF16(float y_ref) const override {
    return std::abs(y_ref) * 5.0e-3f;
  }

  CREATE_OP_OVERRIDES_F32(reciprocal_square_root);
};

CREATE_UNARY_FLOAT_TESTS(F32, ReciprocalSquareRootOperatorTester);
CREATE_UNARY_FLOAT_TESTS(RunF32, ReciprocalSquareRootOperatorTester);

};  // namespace xnnpack
