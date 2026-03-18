// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_HALF_H_
#define XNNPACK_YNNPACK_BASE_HALF_H_

#include <cstdint>

// We want to use _Float16 if the compiler supports it fully, but it's
// tricky to do this detection; there are compiler versions that define the
// type in broken ways. We're only going to bother using it if the support is
// known to be at least a robust f16<->f32 conversion, which generally means a
// recent version of Clang or GCC, x86 or ARM or RISC-V architectures, and
// (in some cases) the right architecture flags specified on the command line.

#ifndef YNN_ARCH_FLOAT16

// Some non-GCC compilers define __GNUC__, but we only want to detect the Real
// Thing
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && \
    !defined(__INTEL_LLVM_COMPILER)
#define YNN_GNUC_ACTUAL __GNUC__
#else
#define YNN_GNUC_ACTUAL 0
#endif

#if (defined(__i386__) || defined(__x86_64__)) && defined(__SSE2__) && \
    defined(__FLT16_MAX__) && defined(__F16C__) &&                     \
    ((__clang_major__ >= 15 && !defined(_MSC_VER)) || (YNN_GNUC_ACTUAL >= 12))
#define YNN_ARCH_FLOAT16 1
#endif

#if (defined(YNN_ARCH_ARM) && !defined(_MSC_VER)) && \
    defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
#define YNN_ARCH_FLOAT16 1
#endif

#if defined(__riscv) && defined(__riscv_zvfh) && __clang__ >= 1600
#define YNN_ARCH_FLOAT16 1
#endif

#ifndef YNN_ARCH_FLOAT16
#define YNN_ARCH_FLOAT16 0
#endif

#endif  // YNN_ARCH_FLOAT16

#if YNN_ARCH_FLOAT16

#include <cmath>

#include "ynnpack/base/bit_cast.h"

namespace ynn {

class half {
 public:
  half() = default;
  constexpr half(float x) : value_(static_cast<_Float16>(x)) {}  // NOLINT

  constexpr operator float() const { return value_; }  // NOLINT

  static half from_bits(uint16_t bits) {
    half result;
    result.value_ = bit_cast<_Float16>(bits);
    return result;
  }

  uint16_t to_bits() const { return bit_cast<uint16_t>(value_); }

  bool is_zero() const { return value_ == 0.0f; }

  // These definitions are imprecise because we want them to be constexpr, and
  // the various tools for doing that are not constepxr (bit_cast,
  // std::numeric_limits, etc.).
  static constexpr half epsilon() { return 0.0009765625f; }
  static constexpr half infinity() { return INFINITY; }
  static constexpr half min() { return -65504.0f; }
  static constexpr half max() { return 65504.0f; }
  static constexpr half smallest_normal() { return 0.00006103515625f; }
  static constexpr half min_identity() { return INFINITY; }
  static constexpr half max_identity() { return -INFINITY; }
  static constexpr half sum_identity() { return 0.0f; }

  // Not private due to -Werror=class-memaccess, which can't be disabled:
  // - via a --copt, because it seems to have no effect.
  // - via .bazelrc, because it then applies to C code, and the compiler says
  //   this flag is not valid in C.
  _Float16 value_;
};

}  // namespace ynn

#else  // YNN_ARCH_FLOAT16

#include "ynnpack/base/fp16.h"

namespace ynn {

class half {
 private:
  // We need this hoop jumping to enable implementing a constexpr `from_bits`.
  struct zero_initializer {};
  explicit constexpr half(zero_initializer) : bits_(0) {}

 public:
  half() = default;
  half(float x) : bits_(fp16_ieee_from_fp32_value(x)) {}  // NOLINT

  operator float() const { return fp16_ieee_to_fp32_value(bits_); }  // NOLINT

  static constexpr half from_bits(uint16_t bits) {
    half result{zero_initializer{}};
    result.bits_ = bits;
    return result;
  }

  constexpr uint16_t to_bits() const { return bits_; }

  bool is_zero() const {
    // Check for +/- zero (0x0000/0x8000). uint16 overflow is well defined to
    // wrap around.
    return static_cast<uint16_t>(bits_ * 2) == 0;
  }

  static constexpr half epsilon() {
    return half::from_bits(0x1400);  // 2^-10 = 0.0009765625
  }
  static constexpr half infinity() { return from_bits(0x7c00); }
  static constexpr half min() { return from_bits(0xfbff); }
  static constexpr half max() { return from_bits(0x7bff); }
  static constexpr half smallest_normal() {
    return from_bits(0x0400);  // 2^-14
  }
  static constexpr half min_identity() { return from_bits(0x7c00); }
  static constexpr half max_identity() { return from_bits(0xfc00); }
  static constexpr half sum_identity() { return from_bits(0); }

  // Not private due to -Werror=class-memaccess, which can't be disabled:
  // - via a --copt, because it seems to have no effect.
  // - via .bazelrc, because it then applies to C code, and the compiler says
  //   this flag is not valid in C.
  uint16_t bits_;
};

}  // namespace ynn

#endif  // YNN_ARCH_FLOAT16

#endif  // XNNPACK_YNNPACK_BASE_HALF_H_
