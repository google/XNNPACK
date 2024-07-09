// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cmath>

#include "unary-operator-tester.h"

namespace xnnpack {

class GELUOperatorTester : public UnaryOperatorTester {
 public:
  GELUOperatorTester() : UnaryOperatorTester() { range_f32_ = {-20.0f, 20.0f}; }

 protected:
  float AbsTolF32(float y_ref) const override {
    return std::max(std::abs(y_ref) * 5.0e-6f, 5.0e-6f);
  };

  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override {
    return x * 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
  }

  CREATE_OP_OVERRIDES_F32(gelu);
};

CREATE_UNARY_FLOAT_TESTS(F32, GELUOperatorTester);
CREATE_UNARY_FLOAT_TESTS(RunF32, GELUOperatorTester);

};  // namespace xnnpack
