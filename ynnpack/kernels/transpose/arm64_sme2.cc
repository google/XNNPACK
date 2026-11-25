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

// In the 8-bit ZA view, horizontal slices 4*r + {0, 1, 2, 3} map to slice r of
// ZA32 tiles {0, 1, 2, 3}, allowing multi-vector ld1b_x{2,4} + mova_vg{2,4}
// loads with zero inter-register moves.
struct Sme2Loader {
  template <uint64_t Tile0, uint64_t Tile1>
  static YNN_ALWAYS_INLINE void load_2svl(
      size_t r, const void* a_r, size_t) YNN_SME_TRANSPOSE_HELPER {
    static_assert(Tile1 == Tile0 + 1);
    svwrite_hor_za8_u8_vg2(
        0, 4 * r + Tile0,
        svld1_u8_x2(svptrue_c8(), static_cast<const uint8_t*>(a_r)));
  }

  static YNN_ALWAYS_INLINE void load_4svl(
      size_t r, const void* a_r, size_t) YNN_SME_TRANSPOSE_HELPER {
    svwrite_hor_za8_u8_vg4(
        0, 4 * r, svld1_u8_x4(svptrue_c8(), static_cast<const uint8_t*>(a_r)));
  }
};

}  // namespace

void transpose_x32_sme2(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                        const void* a, size_t stride_x, void* x) {
  internal::transpose_x32_sme_generic<Sme2Loader>(m, n, n_bytes_a, stride_a, a,
                                                  stride_x, x);
}

}  // namespace ynn

#endif  // YNN_DISABLE_SME
