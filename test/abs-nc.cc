// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstdlib>

#include "unary-operator-tester.h"

namespace xnnpack {

class AbsOperatorTester : public UnaryOperatorTester {
 public:
  AbsOperatorTester() : UnaryOperatorTester() {
    range_f32_ = {-1.0f, 1.0f};
    range_f16_ = {-1.0f, 1.0f};
  }

 protected:
  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override { return std::abs(x); }

  CREATE_OP_OVERRIDES_F32(abs);
  CREATE_OP_OVERRIDES_F16(abs);
};

CREATE_UNARY_FLOAT_TESTS(F32, AbsOperatorTester);
CREATE_UNARY_FLOAT_TESTS(RunF32, AbsOperatorTester);
#ifndef XNN_EXCLUDE_F16_TESTS
CREATE_UNARY_FLOAT_TESTS(F16, AbsOperatorTester);
#endif  // XNN_EXCLUDE_F16_TESTS

};  // namespace xnnpack
