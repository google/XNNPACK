// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_BFLOAT16_H_
#define XNNPACK_YNNPACK_BASE_BFLOAT16_H_

#include <cmath>
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
  bfloat16(float x) {  // NOLINT
    const uint32_t u = bit_cast<uint32_t>(x);
    if ((u << 1) > 0xFF000000u) {
      // Handle the denormal case.
      bits_ = (u >> 16) | 1;
    } else {
      bits_ = (u + 0x7FFF + ((u >> 16) & 1)) >> 16;
    }
  }

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
};

inline bool isfinite(bfloat16 x) { return (x.to_bits() & 0x7f80) != 0x7f80; }
inline bool isnan(bfloat16 x) {
  return !isfinite(x) && (x.to_bits() & 0x007f) != 0;
}
inline bool isinf(bfloat16 x) {
  return !isfinite(x) && (x.to_bits() & 0x007f) == 0;
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_BFLOAT16_H_
