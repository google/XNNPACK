// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_DOT_ARM64_SME_INTERNAL_H_
#define XNNPACK_YNNPACK_KERNELS_DOT_ARM64_SME_INTERNAL_H_

#include <arm_sme.h>

#include <cstddef>
#include <cstdint>

#include "ynnpack/base/base.h"

namespace ynn {

#define YNN_SME_HELPER __arm_streaming __arm_inout("za")

// Helpers to call the right SME intrinsics from overloads based on the type.
inline size_t svcnt(float) { return svcntsw(); }
inline size_t svcnt(int32_t) { return svcntsw(); }
inline size_t svcnt(bfloat16_t) { return svcntsh(); }
inline size_t svcnt(float16_t) { return svcntsh(); }
inline size_t svcnt(int8_t) { return svcntsb(); }

template <uint64_t tile>
YNN_ALWAYS_INLINE void svmopa(svbool_t pn, svbool_t pm, svfloat32_t zn,
                              svfloat32_t zm) YNN_SME_HELPER {
  return svmopa_za32_m(tile, pn, pm, zn, zm);
}
template <uint64_t tile>
YNN_ALWAYS_INLINE void svmopa(svbool_t pn, svbool_t pm, svbfloat16_t zn,
                              svbfloat16_t zm) YNN_SME_HELPER {
  return svmopa_za32_bf16_m(tile, pn, pm, zn, zm);
}
template <uint64_t tile>
YNN_ALWAYS_INLINE void svmopa(svbool_t pn, svbool_t pm, svfloat16_t zn,
                              svfloat16_t zm) YNN_SME_HELPER {
  return svmopa_za32_f16_m(tile, pn, pm, zn, zm);
}
template <uint64_t tile>
YNN_ALWAYS_INLINE void svmopa(svbool_t pn, svbool_t pm, svint8_t zn,
                              svint8_t zm) YNN_SME_HELPER {
  return svmopa_za32_s8_m(tile, pn, pm, zn, zm);
}

YNN_ALWAYS_INLINE svbool_t svwhilelt(int64_t a, int64_t b,
                                     float) YNN_SME_HELPER {
  return svwhilelt_b32(a, b);
}
YNN_ALWAYS_INLINE svbool_t svwhilelt(int64_t a, int64_t b,
                                     int32_t) YNN_SME_HELPER {
  return svwhilelt_b32(a, b);
}
YNN_ALWAYS_INLINE svbool_t svwhilelt(int64_t a, int64_t b,
                                     bfloat16_t) YNN_SME_HELPER {
  return svwhilelt_b16(a, b);
}
YNN_ALWAYS_INLINE svbool_t svwhilelt(int64_t a, int64_t b,
                                     float16_t) YNN_SME_HELPER {
  return svwhilelt_b16(a, b);
}
YNN_ALWAYS_INLINE svbool_t svwhilelt(int64_t a, int64_t b,
                                     int8_t) YNN_SME_HELPER {
  return svwhilelt_b8(a, b);
}

YNN_ALWAYS_INLINE svbool_t svptrue(float) YNN_SME_HELPER {
  return svptrue_b32();
}
YNN_ALWAYS_INLINE svbool_t svptrue(int32_t) YNN_SME_HELPER {
  return svptrue_b32();
}
YNN_ALWAYS_INLINE svbool_t svptrue(bfloat16_t) YNN_SME_HELPER {
  return svptrue_b16();
}
YNN_ALWAYS_INLINE svbool_t svptrue(float16_t) YNN_SME_HELPER {
  return svptrue_b16();
}
YNN_ALWAYS_INLINE svbool_t svptrue(int8_t) YNN_SME_HELPER {
  return svptrue_b8();
}

YNN_ALWAYS_INLINE svcount_t svctrue(float) YNN_SME_HELPER {
  return svptrue_c32();
}
YNN_ALWAYS_INLINE svcount_t svctrue(int32_t) YNN_SME_HELPER {
  return svptrue_c32();
}
YNN_ALWAYS_INLINE svcount_t svctrue(bfloat16_t) YNN_SME_HELPER {
  return svptrue_c16();
}
YNN_ALWAYS_INLINE svcount_t svctrue(float16_t) YNN_SME_HELPER {
  return svptrue_c16();
}
YNN_ALWAYS_INLINE svcount_t svctrue(int8_t) YNN_SME_HELPER {
  return svptrue_c8();
}

// Helpers for ZA loads/stores
template <uint32_t tile>
YNN_ALWAYS_INLINE void svld1_hor_za(uint32_t slice, svbool_t p, const float* ptr) YNN_SME_HELPER {
  svld1_hor_za32(tile, slice, p, ptr);
}
template <uint32_t tile>
YNN_ALWAYS_INLINE void svld1_hor_za(uint32_t slice, svbool_t p, const int32_t* ptr) YNN_SME_HELPER {
  svld1_hor_za32(tile, slice, p, ptr);
}
template <uint32_t tile>
YNN_ALWAYS_INLINE void svld1_hor_za(uint32_t slice, svbool_t p, const bfloat16_t* ptr) YNN_SME_HELPER {
  svld1_hor_za16(tile, slice, p, ptr);
}
template <uint32_t tile>
YNN_ALWAYS_INLINE void svld1_hor_za(uint32_t slice, svbool_t p, const float16_t* ptr) YNN_SME_HELPER {
  svld1_hor_za16(tile, slice, p, ptr);
}
template <uint32_t tile>
YNN_ALWAYS_INLINE void svld1_hor_za(uint32_t slice, svbool_t p, const int8_t* ptr) YNN_SME_HELPER {
  svld1_hor_za8(tile, slice, p, ptr);
}

template <uint32_t tile>
YNN_ALWAYS_INLINE void svst1_ver_za(uint32_t slice, svbool_t p, void* ptr) YNN_SME_HELPER {
  svst1_ver_za32(tile, slice, p, ptr);
}
template <uint32_t tile>
YNN_ALWAYS_INLINE void svst1_ver_za(uint32_t slice, svbool_t p, int32_t* ptr) YNN_SME_HELPER {
  svst1_ver_za32(tile, slice, p, ptr);
}
template <uint32_t tile>
YNN_ALWAYS_INLINE void svst1_ver_za(uint32_t slice, svbool_t p, bfloat16_t* ptr) YNN_SME_HELPER {
  svst1_ver_za16(tile, slice, p, ptr);
}
template <uint32_t tile>
YNN_ALWAYS_INLINE void svst1_ver_za(uint32_t slice, svbool_t p, float16_t* ptr) YNN_SME_HELPER {
  svst1_ver_za16(tile, slice, p, ptr);
}
template <uint32_t tile>
YNN_ALWAYS_INLINE void svst1_ver_za(uint32_t slice, svbool_t p, int8_t* ptr) YNN_SME_HELPER {
  svst1_ver_za8(tile, slice, p, ptr);
}

// Helpers to call SME2 multi-vector loads and creations via overloading.
YNN_ALWAYS_INLINE svfloat32x4_t svld1_x4_impl(svcount_t c, const float* p) YNN_SME_HELPER {
  return svld1_x4(c, p);
}
YNN_ALWAYS_INLINE svbfloat16x4_t svld1_x4_impl(svcount_t c, const bfloat16_t* p) YNN_SME_HELPER {
  return svld1_x4(c, p);
}
YNN_ALWAYS_INLINE svfloat16x4_t svld1_x4_impl(svcount_t c, const float16_t* p) YNN_SME_HELPER {
  return svld1_x4(c, p);
}
YNN_ALWAYS_INLINE svint8x4_t svld1_x4_impl(svcount_t c, const int8_t* p) YNN_SME_HELPER {
  return svld1_x4(c, p);
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_DOT_ARM64_SME_INTERNAL_H_