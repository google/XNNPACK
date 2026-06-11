// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_FP8_H_
#define XNNPACK_YNNPACK_BASE_FP8_H_

#include <cstdint>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/bit_cast.h"
#include "ynnpack/base/half.h"

namespace ynn {

namespace internal {

inline uint8_t half_to_e5m2(half h) {
  // fp16 has the same exponent as e5m2.
  uint16_t fp16 = h.to_bits();
  uint32_t sign = (fp16 >> 8) & 0x80;
  uint32_t temp = fp16 & 0x7FFF;
  if (temp >= 0x7C01) {
    return sign | 0x7F;  // NaN
  }
  temp += 0x007F + ((temp >> 8) & 1);
  return sign | (temp >> 8);
}

inline half e5m2_to_half(uint8_t bits) {
  return half::from_bits(static_cast<uint16_t>(bits) << 8);
}

inline uint8_t float_to_e5m2(float f) { return half_to_e5m2(half(f)); }

inline uint8_t float_to_e4m3(float f) {
  uint32_t w = bit_cast<uint32_t>(f);
  uint32_t sign = (w >> 24) & 0x80;
  uint32_t nonsign = w & 0x7FFFFFFF;

  if (nonsign >= 0x7F800000) {
    return sign | 0x7F;  // NaN
  }

  float abs_f = bit_cast<float>(nonsign);

  if (abs_f > 448.0f) {
    return sign | 0x7E;  // Cap to max normal
  }

  if (abs_f < 0.0009765625f) {  // 2^-10
    return sign | 0;
  }

  if (abs_f < 0.015625f) {          // 2^-6
    float scaled = abs_f * 512.0f;  // / 2^-9
    int i = (int)scaled;
    float diff = scaled - i;
    if (diff > 0.5f) {
      i++;
    } else if (diff == 0.5f) {
      if (i % 2 != 0) {
        i++;
      }
    }
    return sign | i;
  }

  uint32_t exp = (nonsign >> 23) & 0xFF;
  uint32_t mant = nonsign & 0x7FFFFF;

  int e4m3_exp = (int)exp - 120;

  uint32_t rounding_bias = 0x0007FFFF + ((mant >> 20) & 1);
  uint32_t rounded_mant = mant + rounding_bias;

  if (rounded_mant & 0x800000) {
    e4m3_exp++;
    rounded_mant &= 0x7FFFFF;
  }

  return sign | (e4m3_exp << 3) | (rounded_mant >> 20);
}

inline uint8_t half_to_e4m3(half h) {
  return float_to_e4m3(static_cast<float>(h));
}

inline bfloat16 e5m2_to_bf16(uint8_t bits) {
  return static_cast<bfloat16>(static_cast<float>(e5m2_to_half(bits)));
}

inline uint8_t bf16_to_e5m2(bfloat16 f) {
  return float_to_e5m2(static_cast<float>(f));
}

inline half e4m3_to_half(uint8_t bits) {
  uint16_t sign = (static_cast<uint16_t>(bits) & 0x80) << 8;
  uint16_t exp = (bits & 0x78) >> 3;
  uint16_t mant = bits & 0x07;

  if (exp == 0) {
    static constexpr uint16_t subnormal_table[8] = {
        0x0000, 0x1800, 0x1C00, 0x1E00, 0x2000, 0x2100, 0x2200, 0x2300};
    return half::from_bits(sign | subnormal_table[mant]);
  }

  if (exp == 15 && mant == 7) {
    return half::nan();
  }

  return half::from_bits(sign | ((exp + 8) << 10) | (mant << 7));
}

inline bfloat16 e4m3_to_bf16(uint8_t bits) {
  return static_cast<bfloat16>(static_cast<float>(e4m3_to_half(bits)));
}

inline uint8_t bf16_to_e4m3(bfloat16 f) {
  return float_to_e4m3(static_cast<float>(f));
}

}  // namespace internal

class fp8_e5m2 {
 private:
  struct zero_initializer {};
  explicit constexpr fp8_e5m2(zero_initializer) : bits_(0) {}

 public:
  fp8_e5m2() = default;
  fp8_e5m2(float f) : bits_(internal::float_to_e5m2(f)) {}    // NOLINT
  fp8_e5m2(half f) : bits_(internal::half_to_e5m2(f)) {}      // NOLINT
  fp8_e5m2(bfloat16 f) : bits_(internal::bf16_to_e5m2(f)) {}  // NOLINT

  operator bfloat16() const { return internal::e5m2_to_bf16(bits_); }  // NOLINT
  operator half() const { return internal::e5m2_to_half(bits_); }      // NOLINT
  operator float() const { return internal::e5m2_to_half(bits_); }     // NOLINT

  static constexpr fp8_e5m2 from_bits(uint8_t bits) {
    fp8_e5m2 result{zero_initializer{}};
    result.bits_ = bits;
    return result;
  }

  constexpr uint8_t to_bits() const { return bits_; }

  bool is_zero() const { return (bits_ & 0x7F) == 0; }

  // Constants
  static constexpr fp8_e5m2 epsilon() { return from_bits(0x34); }  // 2^-2
  static constexpr fp8_e5m2 infinity() { return from_bits(0x7C); }
  static constexpr fp8_e5m2 min() { return from_bits(0xFB); }  // -max
  static constexpr fp8_e5m2 max() { return from_bits(0x7B); }  // 57344
  static constexpr fp8_e5m2 smallest_normal() {
    return from_bits(0x04);  // 2^-14
  }
  static constexpr fp8_e5m2 min_identity() { return from_bits(0x7C); }
  static constexpr fp8_e5m2 max_identity() { return from_bits(0xFC); }
  static constexpr fp8_e5m2 sum_identity() { return from_bits(0); }

  uint8_t bits_;
};

class fp8_e4m3 {
 private:
  struct zero_initializer {};
  explicit constexpr fp8_e4m3(zero_initializer) : bits_(0) {}

 public:
  fp8_e4m3() = default;
  fp8_e4m3(float f) : bits_(internal::float_to_e4m3(f)) {}    // NOLINT
  fp8_e4m3(half f) : bits_(internal::half_to_e4m3(f)) {}      // NOLINT
  fp8_e4m3(bfloat16 f) : bits_(internal::bf16_to_e4m3(f)) {}  // NOLINT

  operator bfloat16() const { return internal::e4m3_to_bf16(bits_); }  // NOLINT
  operator half() const { return internal::e4m3_to_half(bits_); }      // NOLINT
  operator float() const { return internal::e4m3_to_bf16(bits_); }     // NOLINT

  static constexpr fp8_e4m3 from_bits(uint8_t bits) {
    fp8_e4m3 result{zero_initializer{}};
    result.bits_ = bits;
    return result;
  }

  constexpr uint8_t to_bits() const { return bits_; }

  bool is_zero() const { return (bits_ & 0x7F) == 0; }

  // Constants (E4M3FN)
  static constexpr fp8_e4m3 epsilon() { return from_bits(0x20); }  // 2^-3
  // E4M3FN has no infinity encoding; NaN is the conventional substitute
  // (matches ml_dtypes float8_e4m3fn).
  static constexpr fp8_e4m3 infinity() { return from_bits(0x7F); }
  static constexpr fp8_e4m3 min() { return from_bits(0xFE); }  // -448
  static constexpr fp8_e4m3 max() { return from_bits(0x7E); }  // 448
  static constexpr fp8_e4m3 smallest_normal() {
    return from_bits(0x08);  // 2^-6
  }
  static constexpr fp8_e4m3 min_identity() { return max(); }
  static constexpr fp8_e4m3 max_identity() { return min(); }
  static constexpr fp8_e4m3 sum_identity() { return from_bits(0); }

  uint8_t bits_;
};

inline bool isfinite(fp8_e5m2 x) { return (x.to_bits() & 0x7c) != 0x7c; }
inline bool isnan(fp8_e5m2 x) {
  return !isfinite(x) && (x.to_bits() & 0x03) != 0;
}
inline bool isinf(fp8_e5m2 x) {
  return !isfinite(x) && (x.to_bits() & 0x03) == 0;
}

// E4M3FN has no infinity: exponent 15 with mantissa < 7 encodes the finite
// values 256..448, and only 0x7F/0xFF is NaN.
inline bool isfinite(fp8_e4m3 x) { return (x.to_bits() & 0x7F) != 0x7F; }
inline bool isnan(fp8_e4m3 x) { return (x.to_bits() & 0x7F) == 0x7F; }
inline bool isinf(fp8_e4m3 x) { return false; }

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_FP8_H_
