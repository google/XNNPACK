// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_sme.h>

#include <cstddef>
#include <cstdint>

#include "ynnpack/base/base.h"
#include "ynnpack/kernels/transpose/arm64_sme_internal.h"
#include "ynnpack/kernels/transpose/transpose.h"

#ifndef YNN_DISABLE_SME

namespace ynn {

namespace {

struct SmeLoader {
  template <uint64_t Tile0, uint64_t Tile1>
  static YNN_ALWAYS_INLINE void load_2svl(
      size_t r, const void* a_r, size_t svl_bytes) YNN_SME_TRANSPOSE_HELPER {
    const svbool_t p32 = svptrue_b32();
    svld1_hor_za32(Tile0, r, p32, a_r);
    svld1_hor_za32(Tile1, r, p32, internal::sme_offset_bytes(a_r, svl_bytes));
  }

  static YNN_ALWAYS_INLINE void load_4svl(
      size_t r, const void* a_r, size_t svl_bytes) YNN_SME_TRANSPOSE_HELPER {
    const svbool_t p32 = svptrue_b32();
    svld1_hor_za32(0, r, p32, a_r);
    svld1_hor_za32(1, r, p32, internal::sme_offset_bytes(a_r, svl_bytes));
    svld1_hor_za32(2, r, p32, internal::sme_offset_bytes(a_r, 2 * svl_bytes));
    svld1_hor_za32(3, r, p32, internal::sme_offset_bytes(a_r, 3 * svl_bytes));
  }
};

}  // namespace

void transpose_x32_sme(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                       const void* a, size_t stride_x, void* x) {
  internal::transpose_x32_sme_generic<SmeLoader>(m, n, n_bytes_a, stride_a, a,
                                                 stride_x, x);
}

}  // namespace ynn

#endif  // YNN_DISABLE_SME
