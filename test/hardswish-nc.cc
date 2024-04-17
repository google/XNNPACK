// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <algorithm>
#include <cstdlib>

#include "unary-operator-tester.h"

namespace xnnpack {

class HardSwishOperatorTester : public UnaryOperatorTester {
 public:
  HardSwishOperatorTester() : UnaryOperatorTester() {
    range_f32_ = {-1.0f, 1.0f};
    range_f16_ = {-1.0f, 1.0f};
  }

 protected:
  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override {
    return x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
    ;
  }

  // Computes the absolute tolerance for a reference value `y_ref`. Tests will
  // fail when `std::abs(y - y_ref) > AbsTol32(y_ref)`. Note that for `fp16`
  // tests, both `y` and `y_ref` will be converted to `float` for the tolerance
  // evaluation.
  float AbsTolF32(float y_ref) const override {
    return std::max(1.0e-7f, std::abs(y_ref) * 1.0e-6f);
  };
  float AbsTolF16(float y_ref) const override {
    return std::max(1.0e-3f, std::abs(y_ref) * 1.0e-2f);
  };

  CREATE_OP_OVERRIDES_F32(hardswish);
  CREATE_OP_OVERRIDES_F16(hardswish);
};

CREATE_UNARY_FLOAT_TESTS(F32, HardSwishOperatorTester);
CREATE_UNARY_FLOAT_TESTS(RunF32, HardSwishOperatorTester);
#ifndef XNN_EXCLUDE_F16_TESTS
CREATE_UNARY_FLOAT_TESTS(F16, HardSwishOperatorTester);
#endif  // XNN_EXCLUDE_F16_TESTS

};  // namespace xnnpack
