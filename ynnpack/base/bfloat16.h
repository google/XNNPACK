// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_BFLOAT16_H_
#define XNNPACK_YNNPACK_BASE_BFLOAT16_H_

#include <cstdint>

#include "ynnpack/base/bit_cast.h"

namespace ynn {

class bfloat16 {
 private:
  // We need this hoop jumping to enable implementing a constexpr `from_bits`.
  struct zero_initializer {};
  explicit constexpr bfloat16(zero_initializer) : bits_(0) {}

 public:
  bfloat16() = default;
  bfloat16(float x)  // NOLINT
      : bits_(bit_cast<uint32_t>(x * rounding_multiplier) >> 16) {}

  operator float() const {  // NOLINT
    return bit_cast<float>(static_cast<uint32_t>(bits_) << 16);
  }

  static constexpr bfloat16 from_bits(uint16_t bits) {
    bfloat16 result{zero_initializer{}};
    result.bits_ = bits;
    return result;
  }

  constexpr uint16_t to_bits() const { return bits_; }

  bool is_zero() const {
    // Check for +/- zero (0x0000/0x8000). uint16 overflow is well defined to
    // wrap around.
    return static_cast<uint16_t>(bits_ * 2) == 0;
  }

  // Not private due to -Werror=class-memaccess, which can't be disabled:
  // - via a --copt, because it seems to have no effect.
  // - via .bazelrc, because it then applies to C code, and the compiler says
  //   this flag is not valid in C.
  uint16_t bits_;

  // When rounding a float to bfloat16, we want to add 1 to the bit after the
  // last bit in the mantissa. We can make the floating point hardware do this
  // for us, by multiplying by 1 + 0.5*epsilon:
  // a*rounding_multiplier = a*(1 + 0.5*epsilon) = a + a*0.5*epsilon.
  // 0.5*epsilon is a power of 2, so this is effectively moving the leading 1
  // from the mantissa to the bit after the last bit of the mantissa, and then
  // adding it.
  static constexpr float rounding_multiplier = 1.0f + 0.5f / 128.0f;
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_BFLOAT16_H_
