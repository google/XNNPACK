// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif


#define XNN_UKERNEL(arch_flags, ukernel, nr_, kr_, sr_, kblock, nr_scale) \
  XNN_INTERNAL void ukernel(                                              \
      size_t g,                                                           \
      size_t nc,                                                          \
      size_t kc,                                                          \
      size_t nr,                                                          \
      size_t kr,                                                          \
      size_t sr,                                                          \
      const int8_t* weights,                                              \
      const uint32_t* bias,                                               \
      const void* scale,                                                  \
      int8_t* packed_weights,                                             \
      size_t extra_bytes,                                                 \
      const void* params);

#define XNN_GIO_UKERNEL(arch_flags, ukernel, nr_, kr_, sr_, kblock, nr_scale) \
  XNN_INTERNAL void ukernel(                                                  \
      size_t g,                                                               \
      size_t nc,                                                              \
      size_t kc,                                                              \
      size_t nr,                                                              \
      size_t kr,                                                              \
      size_t sr,                                                              \
      size_t k_stride,                                                        \
      const int8_t* weights,                                                  \
      const uint32_t* bias,                                                   \
      const void* scale,                                                      \
      int8_t* packed_weights,                                                 \
      size_t extra_bytes,                                                     \
      const void* params);

#include "x8-packw/x8-packw.h"

#undef XNN_UKERNEL
#undef XNN_GIO_UKERNEL

#define XNN_QS8_UKERNEL(arch_flags, ukernel, nr_, kr_, sr_, kblock, nr_scale, izp) \
  XNN_INTERNAL void ukernel(                                                       \
      size_t g,                                                                    \
      size_t nc,                                                                   \
      size_t kc,                                                                   \
      size_t nr,                                                                   \
      size_t kr,                                                                   \
      size_t sr,                                                                   \
      const int8_t* weights,                                                       \
      const int32_t* bias,                                                         \
      const void* scale,                                                           \
      int8_t* packed_weights,                                                      \
      size_t extra_bytes,                                                          \
      const void* params);

#define XNN_QS8_GIO_UKERNEL(arch_flags, ukernel, nr_, kr_, sr_, kblock, nr_scale, izp)  \
  XNN_INTERNAL void ukernel(                                                            \
      size_t g,                                                                         \
      size_t nc,                                                                        \
      size_t kc,                                                                        \
      size_t nr,                                                                        \
      size_t kr,                                                                        \
      size_t sr,                                                                        \
      size_t k_stride,                                                                  \
      const int8_t* weights,                                                            \
      const int32_t* bias,                                                              \
      const void* scale,                                                                \
      int8_t* packed_weights,                                                           \
      size_t extra_bytes,                                                               \
      const void* params);

#include "qs8-packw/qs8-packw.h"

#undef XNN_QS8_UKERNEL
#undef XNN_QS8_GIO_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, nr_, kr_, sr_, kblock, nr_scale) \
  XNN_INTERNAL void ukernel(                                              \
      size_t g,                                                           \
      size_t nc,                                                          \
      size_t kc,                                                          \
      size_t nr,                                                          \
      size_t kr,                                                          \
      size_t sr,                                                          \
      const uint16_t* weights,                                            \
      const uint16_t* bias,                                               \
      const void* scale,                                                  \
      uint16_t* packed_weights,                                           \
      size_t extra_bytes,                                                 \
      const void* params);                                                \

#include "x16-packw/x16-packw.h"

#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, nr_, kr_, sr_, kblock, nr_scale) \
  XNN_INTERNAL void ukernel(                                              \
      size_t g,                                                           \
      size_t nc,                                                          \
      size_t kc,                                                          \
      size_t nr,                                                          \
      size_t kr,                                                          \
      size_t sr,                                                          \
      const uint32_t* weights,                                            \
      const uint32_t* bias,                                               \
      const void* scale,                                                  \
      uint32_t* packed_weights,                                           \
      size_t extra_bytes,                                                 \
      const void* params);                                                \

#define XNN_GIO_UKERNEL(arch_flags, ukernel, nr_, kr_, sr_, kblock, nr_scale) \
  XNN_INTERNAL void ukernel(                                                  \
      size_t g,                                                               \
      size_t nc,                                                              \
      size_t kc,                                                              \
      size_t nr,                                                              \
      size_t kr,                                                              \
      size_t sr,                                                              \
      size_t k_stride,                                                        \
      const uint32_t* weights,                                                \
      const uint32_t* bias,                                                   \
      const void* scale,                                                      \
      uint32_t* packed_weights,                                               \
      size_t extra_bytes,                                                     \
      const void* params);                                                    \

#include "x32-packw/x32-packw.h"

#undef XNN_UKERNEL
#undef XNN_GIO_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, nr_, kr_, sr_, kblock, nr_scale) \
  XNN_INTERNAL void ukernel(                                              \
      size_t g,                                                           \
      size_t nc,                                                          \
      size_t kc,                                                          \
      size_t nr,                                                          \
      size_t kr,                                                          \
      size_t sr,                                                          \
      const uint8_t* k,                                                   \
      const int32_t* b,                                                   \
      const float* scale,                                                 \
      void* packed_weights,                                               \
      size_t extra_bytes,                                                 \
      const struct xnn_qs8_qc4w_packing_params* params);

#include "qs8-qc4w-packw/qs8-qc4w-packw.h"

#undef XNN_UKERNEL


#ifdef __cplusplus
}  // extern "C"
#endif
