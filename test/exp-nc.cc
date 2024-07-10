// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstdlib>

#include "unary-operator-tester.h"

namespace xnnpack {

class ExpOperatorTester : public UnaryOperatorTester {
 public:
  ExpOperatorTester() : UnaryOperatorTester() {
    range_f32_ = {-10.f, 10.0f};
    range_f16_ = {-10.f, 10.0f};
  }

 protected:
  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override { return std::exp(x); }

  CREATE_STANDARD_OP_OVERRIDES_F32(exp);
};

CREATE_UNARY_FLOAT_TESTS(F32, ExpOperatorTester);

};  // namespace xnnpack
