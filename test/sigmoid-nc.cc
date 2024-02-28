// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "unary-operator-tester.h"

namespace xnnpack {

class SigmoidOperatorTester : public UnaryOperatorTester {
 public:
  SigmoidOperatorTester() : UnaryOperatorTester() {
    range_f32_ = {-25.0f, 25.0f};
    range_f16_ = {-25.0f, 25.0f};
    output_scale(1.0f / 256.0f);
    output_zero_point(0);
  }

 protected:
  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override {
    return 1.0 / (1.0 + std::exp(static_cast<double>(-x)));
  }

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

  CREATE_OP_OVERRIDES_F32(sigmoid);
  CREATE_OP_OVERRIDES_F16(sigmoid);
  CREATE_OP_OVERRIDES_QS8(sigmoid);
  CREATE_OP_OVERRIDES_QU8(sigmoid);
};

CREATE_UNARY_FLOAT_TESTS(F32, SigmoidOperatorTester);
CREATE_UNARY_FLOAT_TESTS(RunF32, SigmoidOperatorTester);
#ifndef XNN_EXCLUDE_F16_TESTS
CREATE_UNARY_FLOAT_TESTS(F16, SigmoidOperatorTester);
#endif  // XNN_EXCLUDE_F16_TESTS

CREATE_UNARY_QUANTIZED_TESTS(QS8, SigmoidOperatorTester);
CREATE_UNARY_QUANTIZED_TESTS(QU8, SigmoidOperatorTester);

};  // namespace xnnpack
