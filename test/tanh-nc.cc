// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "unary-operator-tester.h"

namespace xnnpack {

class TanhOperatorTester : public UnaryOperatorTester {
 public:
  TanhOperatorTester() : UnaryOperatorTester() {
    range_f32_ = {-10.0f, 10.0f};
    range_f16_ = {-5.0f, 5.0f};
  }

 protected:
  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override { return std::tanh(x); }

  // Computes the absolute tolerance for a reference value `y_ref`. Tests will
  // fail when `std::abs(y - y_ref) > AbsTol32(y_ref)`. Note that for `fp16`
  // tests, both `y` and `y_ref` will be converted to `float` for the tolerance
  // evaluation.
  float AbsTolF32(float) const override { return 5e-6f; }
  float AbsTolF16(float y_ref) const override {
    return std::max(1.0e-4f, std::abs(y_ref) * 5.0e-3f);
  }
  float AbsTolQS8(float) const override { return 0.6f; };
  float AbsTolQU8(float) const override { return 0.6f; };

  CREATE_OP_OVERRIDES_F32(tanh);
  CREATE_OP_OVERRIDES_F16(tanh);
  CREATE_OP_OVERRIDES_QS8(tanh);
  CREATE_OP_OVERRIDES_QU8(tanh);
};

CREATE_UNARY_FLOAT_TESTS(F32, TanhOperatorTester);
CREATE_UNARY_FLOAT_TESTS(RunF32, TanhOperatorTester);
#ifndef XNN_EXCLUDE_F16_TESTS
CREATE_UNARY_FLOAT_TESTS(F16, TanhOperatorTester);
#endif  // XNN_EXCLUDE_F16_TESTS

CREATE_UNARY_QUANTIZED_TESTS(QS8, TanhOperatorTester);
CREATE_UNARY_QUANTIZED_TESTS(QU8, TanhOperatorTester);

};  // namespace xnnpack
