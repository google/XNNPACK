// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_TRANSPOSE_TRANSPOSE_IMPL_H_
#define XNNPACK_YNNPACK_KERNELS_TRANSPOSE_TRANSPOSE_IMPL_H_

#include <arm_neon.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"

namespace ynn {

static std::array<uint8x16_t, 2> interleave(std::integral_constant<size_t, 64>,
                                            std::array<uint8x16_t, 2> x) {
  return {{
      vcombine_u8(vget_low_u8(x[0]), vget_low_u8(x[1])),
      vcombine_u8(vget_high_u8(x[0]), vget_high_u8(x[1])),
  }};
}
static std::array<uint8x16_t, 2> interleave(std::integral_constant<size_t, 32>,
                                            std::array<uint8x16_t, 2> x) {
  uint32x4x2_t x01 =
      vzipq_u32(vreinterpretq_u32_u8(x[0]), vreinterpretq_u32_u8(x[1]));
  return {{vreinterpretq_u8_u32(x01.val[0]), vreinterpretq_u8_u32(x01.val[1])}};
}
static std::array<uint8x16_t, 2> interleave(std::integral_constant<size_t, 16>,
                                            std::array<uint8x16_t, 2> x) {
  uint16x8x2_t x01 =
      vzipq_u16(vreinterpretq_u16_u8(x[0]), vreinterpretq_u16_u8(x[1]));
  return {{vreinterpretq_u8_u16(x01.val[0]), vreinterpretq_u8_u16(x01.val[1])}};
}
static std::array<uint8x16_t, 2> interleave(std::integral_constant<size_t, 8>,
                                            std::array<uint8x16_t, 2> x) {
  uint8x16x2_t x01 = vzipq_u8(x[0], x[1]);
  return {{x01.val[0], x01.val[1]}};
}
static std::array<uint8x16_t, 2> interleave(std::integral_constant<size_t, 4>,
                                            std::array<uint8x16_t, 2> x) {
  return interleave(std::integral_constant<size_t, 8>{},
                    {vbslq_u8(vdupq_n_u8(0xf0), vshlq_n_u8(x[1], 4), x[0]),
                     vbslq_u8(vdupq_n_u8(0xf0), x[1], vshrq_n_u8(x[0], 4))});
}

static std::array<uint8x8_t, 2> interleave(std::integral_constant<size_t, 32>,
                                           std::array<uint8x8_t, 2> x) {
  uint32x2x2_t x01 =
      vzip_u32(vreinterpret_u32_u8(x[0]), vreinterpret_u32_u8(x[1]));
  return {{vreinterpret_u8_u32(x01.val[0]), vreinterpret_u8_u32(x01.val[1])}};
}
static std::array<uint8x8_t, 2> interleave(std::integral_constant<size_t, 16>,
                                           std::array<uint8x8_t, 2> x) {
  uint16x4x2_t x01 =
      vzip_u16(vreinterpret_u16_u8(x[0]), vreinterpret_u16_u8(x[1]));
  return {{vreinterpret_u8_u16(x01.val[0]), vreinterpret_u8_u16(x01.val[1])}};
}
static std::array<uint8x8_t, 2> interleave(std::integral_constant<size_t, 8>,
                                           std::array<uint8x8_t, 2> x) {
  uint8x8x2_t x01 = vzip_u8(x[0], x[1]);
  return {{x01.val[0], x01.val[1]}};
}
static std::array<uint8x8_t, 2> interleave(std::integral_constant<size_t, 4>,
                                           std::array<uint8x8_t, 2> x) {
  return interleave(std::integral_constant<size_t, 8>{},
                    {vbsl_u8(vdup_n_u8(0xf0), vshl_n_u8(x[1], 4), x[0]),
                     vbsl_u8(vdup_n_u8(0xf0), x[1], vshr_n_u8(x[0], 4))});
}

template <size_t M>
static std::array<uint8x16_t, M> load(
    std::array<uint8x16_t, M>, const void* a, size_t stride, size_t m,
    std::integral_constant<size_t, 16> /*n_bytes*/) {
  assert(m > 0);
  assert(m <= M);
  std::array<uint8x16_t, M> x;
  x[0] = vld1q_u8(reinterpret_cast<const uint8_t*>(a));
  for (size_t i = 1; i < M; ++i) {
    x[i] = i < m ? vld1q_u8(reinterpret_cast<const uint8_t*>(
                       offset_bytes(a, i * stride)))
                 : vdupq_n_u8(0);
  }
  return x;
}

template <size_t M>
static void store(std::array<uint8x16_t, M> x, void* a, size_t stride, size_t m,
                  std::integral_constant<size_t, 16> /*n_bytes*/) {
  assert(m > 0);
  assert(m <= M);
  vst1q_u8(reinterpret_cast<uint8_t*>(a), x[0]);
  for (size_t i = 1; i < M; ++i) {
    if (i < m) {
      vst1q_u8(reinterpret_cast<uint8_t*>(offset_bytes(a, i * stride)), x[i]);
    }
  }
}

template <size_t M>
static std::array<uint8x8_t, M> load(
    std::array<uint8x8_t, M>, const void* a, size_t stride, size_t m,
    std::integral_constant<size_t, 8> /*n_bytes*/) {
  assert(m > 0);
  assert(m <= M);
  std::array<uint8x8_t, M> x;
  x[0] = vld1_u8(reinterpret_cast<const uint8_t*>(a));
  for (size_t i = 1; i < M; ++i) {
    x[i] = i < m ? vld1_u8(reinterpret_cast<const uint8_t*>(
                       offset_bytes(a, i * stride)))
                 : vdup_n_u8(0);
  }
  return x;
}

template <size_t M>
static void store(std::array<uint8x8_t, M> x, void* a, size_t stride, size_t m,
                  std::integral_constant<size_t, 8> /*n_bytes*/) {
  assert(m > 0);
  assert(m <= M);
  vst1_u8(reinterpret_cast<uint8_t*>(a), x[0]);
  for (size_t i = 1; i < M; ++i) {
    if (i < m) {
      vst1_u8(reinterpret_cast<uint8_t*>(offset_bytes(a, i * stride)), x[i]);
    }
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_TRANSPOSE_TRANSPOSE_IMPL_H_
