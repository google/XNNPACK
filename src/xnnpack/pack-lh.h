// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_PACK_LH_H_
#define XNNPACK_SRC_XNNPACK_PACK_LH_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_UKERNEL(arch_flags, ukernel, size_fn, packed_offset_fn)        \
  XNN_INTERNAL void ukernel(size_t m, size_t k, size_t mr, size_t kr,      \
                            size_t sr, size_t m_idx_start, const float* x, \
                            size_t x_stride, void* y);                     \
                                                                           \
  XNN_INTERNAL size_t size_fn(size_t m, size_t k, size_t mr, size_t kr,    \
                              size_t sr);                                  \
                                                                           \
  XNN_INTERNAL size_t packed_offset_fn(size_t m, size_t k, size_t mr,      \
                                       size_t kr, size_t sr);

#include "src/x32-pack-lh/x32-pack-lh.inc"

#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, size_fn, packed_offset_fn)          \
  XNN_INTERNAL void ukernel(size_t m, size_t k, size_t mr, size_t kr,        \
                            size_t sr, size_t m_idx_start,                   \
                            const xnn_float16* x, size_t x_stride, void* y); \
                                                                             \
  XNN_INTERNAL size_t size_fn(size_t m, size_t k, size_t mr, size_t kr,      \
                              size_t sr);                                    \
                                                                             \
  XNN_INTERNAL size_t packed_offset_fn(size_t m, size_t k, size_t mr,        \
                                       size_t kr, size_t sr);

#include "src/x16-pack-lh/x16-pack-lh.inc"

#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, size_fn, packed_offset_fn)         \
  XNN_INTERNAL void ukernel(size_t m, size_t k, size_t mr, size_t kr,       \
                            size_t sr, size_t m_idx_start, const int8_t* x, \
                            size_t x_stride, void* y);                      \
                                                                            \
  XNN_INTERNAL size_t size_fn(size_t m, size_t k, size_t mr, size_t kr,     \
                              size_t sr);                                   \
                                                                            \
  XNN_INTERNAL size_t packed_offset_fn(size_t m, size_t k, size_t mr,       \
                                       size_t kr, size_t sr);

#include "src/x8-pack-lh/x8-pack-lh.inc"

#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, igemm_ukernel, igemm_size_fn,                 \
                    igemm_packed_offset_fn)                                   \
  XNN_INTERNAL void igemm_ukernel(                                            \
      size_t m, size_t kc, size_t ks, size_t mr_packed, size_t kr, size_t sr, \
      const void** a, size_t a_offset, const void* zero, void* lhs_packed);   \
                                                                              \
  XNN_INTERNAL size_t igemm_size_fn(size_t m, size_t kc, size_t ks,           \
                                    size_t mr_packed, size_t kr, size_t sr);  \
                                                                              \
  XNN_INTERNAL size_t igemm_packed_offset_fn(                                 \
      size_t m, size_t kc, size_t ks, size_t mr_packed, size_t kr, size_t sr);

#include "src/x8-pack-lh/x8-pack-lh-igemm.inc"

#undef XNN_UKERNEL

// These "configs" are defined here because they do not follow established
// conventions of configs and microkernels (the "microkernels" depend on other
// configs and microkernels), which would need to have special treatment to
// avoid a circular dependency.
XNN_INTERNAL const struct xnn_pack_lh_config*
xnn_init_f16_qdint8_pack_lh_config();
XNN_INTERNAL const struct xnn_pack_lh_config*
xnn_init_f16_qduint8_pack_lh_config();
XNN_INTERNAL const struct xnn_pack_lh_config*
xnn_init_f32_qdint8_pack_lh_config();
XNN_INTERNAL const struct xnn_pack_lh_config*
xnn_init_f32_qduint8_pack_lh_config();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_PACK_LH_H_
