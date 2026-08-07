// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/pack-lh.h"
#include "src/xnnpack/packq.h"

static struct xnn_pack_lh_config qp8_pack_lh_config = {0};
static struct xnn_pack_lh_config x8_pack_lh_config = {0};
static struct xnn_pack_lh_config x16_pack_lh_config = {0};
static struct xnn_pack_lh_config x32_pack_lh_config = {0};
static struct xnn_pack_lh_config x8_igemm_pack_lh_config = {0};
static struct xnn_pack_lh_config x32_igemm_pack_lh_config = {0};
static struct xnn_pack_lh_config x16_igemm_pack_lh_config = {0};

XNN_INIT_ONCE_GUARD(qp8_pack_lh);
XNN_INIT_ONCE_GUARD(x8_pack_lh);
XNN_INIT_ONCE_GUARD(x16_pack_lh);
XNN_INIT_ONCE_GUARD(x32_pack_lh);
XNN_INIT_ONCE_GUARD(x32_igemm_pack_lh);
XNN_INIT_ONCE_GUARD(x8_igemm_pack_lh);
XNN_INIT_ONCE_GUARD(x16_igemm_pack_lh);

#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
static void x16_pack_lh_direct(
    size_t m, size_t k, size_t unused_mr, size_t unused_kr, size_t unused_sr,
    size_t unused_m_idx_start, const void* lhs, size_t lhs_stride,
    void* lhs_packed) {
  (void)unused_mr;
  (void)unused_kr;
  (void)unused_sr;
  (void)unused_m_idx_start;
  const size_t row_size = k * sizeof(xnn_float16);
  for (size_t row = 0; row < m; row++) {
    memcpy((void*)((uintptr_t)lhs_packed + row * row_size),
           (const void*)((uintptr_t)lhs + row * lhs_stride), row_size);
  }
}

static size_t x16_pack_lh_direct_size(
    size_t m, size_t k, size_t unused_mr, size_t unused_kr,
    size_t unused_sr) {
  (void)unused_mr;
  (void)unused_kr;
  (void)unused_sr;
  return m * k * sizeof(xnn_float16);
}
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI

static void init_qp8_pack_lh_config(void) {
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
  qp8_pack_lh_config.pack_lh_fn =
      (xnn_pack_lh_ukernel_fn)xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2;
#else
  qp8_pack_lh_config.pack_lh_fn =
      (xnn_pack_lh_ukernel_fn)xnn_x8_packq_f32qp8_ukernel__scalar_u1;
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
  qp8_pack_lh_config.size_fn =
      (xnn_pack_lh_size_fn)xnn_x8_packq_f32qp8_packed_size;
  qp8_pack_lh_config.offset_fn =
      (xnn_pack_lh_offset_fn)xnn_x8_packq_f32qp8_packed_offset;
  qp8_pack_lh_config.log2_input_element_size = XNN_LOG2_SIZEOF_FLOAT;
  qp8_pack_lh_config.log2_packed_element_size = 0;
}

const struct xnn_pack_lh_config* xnn_init_qp8_pack_lh_config() {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qp8_pack_lh);
  return &qp8_pack_lh_config;
}

static void init_x32_pack_lh_config(void) {
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  assert(hardware_config != NULL);
  (void)hardware_config;
#if XNN_ENABLE_ARM_SME2
  if ((hardware_config->arch_flags & xnn_arch_arm_sme2)) {
    x32_pack_lh_config.pack_lh_fn =
        (xnn_pack_lh_ukernel_fn)xnn_x32_pack_lh_ukernel__neonsme2;
    x32_pack_lh_config.size_fn =
        (xnn_pack_lh_size_fn)xnn_x32_pack_lh_size__neonsme2;
    x32_pack_lh_config.offset_fn =
        (xnn_pack_lh_offset_fn)xnn_x32_pack_lh_offset__neonsme2;
  }
#endif  // XNN_ENABLE_ARM_SME2
#if XNN_ENABLE_ARM_SME
  if ((hardware_config->arch_flags & xnn_arch_arm_sme)) {
    x32_pack_lh_config.pack_lh_fn =
        (xnn_pack_lh_ukernel_fn)xnn_x32_pack_lh_ukernel__neonsme;
    x32_pack_lh_config.size_fn =
        (xnn_pack_lh_size_fn)xnn_x32_pack_lh_size__neonsme;
    x32_pack_lh_config.offset_fn =
        (xnn_pack_lh_offset_fn)xnn_x32_pack_lh_offset__neonsme;
  }
#endif  // XNN_ENABLE_ARM_SME
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
  x32_pack_lh_config.log2_input_element_size = 2;
  x32_pack_lh_config.log2_packed_element_size = 2;
  x32_pack_lh_config.gemv_noop = true;
}

const struct xnn_pack_lh_config* xnn_init_x32_pack_lh_config() {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(x32_pack_lh);
  return &x32_pack_lh_config;
}

static void init_x32_igemm_pack_lh_config(void) {
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  assert(hardware_config != NULL);
  (void)hardware_config;
#if XNN_ENABLE_ARM_SME2
    if ((hardware_config->arch_flags & xnn_arch_arm_sme2)) {
        x32_igemm_pack_lh_config.pack_lh_for_igemm_fn  = (xnn_pack_lh_igemm_ukernel_fn) xnn_x32_pack_lh_ukernel__igemm_neonsme2;
        x32_igemm_pack_lh_config.size_for_igemm_fn  = (xnn_pack_lh_igemm_size_fn) xnn_x32_pack_lh_size__igemm_neonsme2;
        x32_igemm_pack_lh_config.offset_for_igemm_fn  = (xnn_pack_lh_igemm_offset_fn) xnn_x32_pack_lh_offset__igemm_neonsme2;
    }
#endif  // XNN_ENABLE_ARM_SME2
#if XNN_ENABLE_ARM_SME
    if ((hardware_config->arch_flags & xnn_arch_arm_sme)) {
        x32_igemm_pack_lh_config.pack_lh_for_igemm_fn  = (xnn_pack_lh_igemm_ukernel_fn) xnn_x32_pack_lh_ukernel__igemm_neonsme;
        x32_igemm_pack_lh_config.size_for_igemm_fn  = (xnn_pack_lh_igemm_size_fn) xnn_x32_pack_lh_size__igemm_neonsme;
        x32_igemm_pack_lh_config.offset_for_igemm_fn  = (xnn_pack_lh_igemm_offset_fn) xnn_x32_pack_lh_offset__igemm_neonsme;
    }
#endif  // XNN_ENABLE_ARM_SME
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
  x32_igemm_pack_lh_config.log2_input_element_size = 2;
  x32_igemm_pack_lh_config.log2_packed_element_size = 2;
  x32_igemm_pack_lh_config.gemv_noop = true;
}

const struct xnn_pack_lh_config* xnn_init_x32_igemm_pack_lh_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(x32_igemm_pack_lh);
  return &x32_igemm_pack_lh_config;
}

static void init_x16_pack_lh_config(void) {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  assert(hardware_config != NULL);
  (void)hardware_config;
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
  x16_pack_lh_config.pack_lh_fn = x16_pack_lh_direct;
  x16_pack_lh_config.size_fn = x16_pack_lh_direct_size;
  x16_pack_lh_config.offset_fn = x16_pack_lh_direct_size;
#if XNN_ENABLE_ARM_SME
  if ((hardware_config->arch_flags & xnn_arch_arm_sme)) {
    x16_pack_lh_config.pack_lh_fn =
        (xnn_pack_lh_ukernel_fn)xnn_x16_pack_lh_ukernel__neonsme;
    x16_pack_lh_config.size_fn =
        (xnn_pack_lh_size_fn)xnn_x16_pack_lh_size__neonsme;
    x16_pack_lh_config.offset_fn =
        (xnn_pack_lh_offset_fn)xnn_x16_pack_lh_offset__neonsme;
  }
#endif  // XNN_ENABLE_ARM_SME
#if XNN_ENABLE_ARM_SME2
  if ((hardware_config->arch_flags & xnn_arch_arm_sme2)) {
/* IGEMM SME packer is not used for generic x16 pack_lh. */
    x16_pack_lh_config.pack_lh_fn =
        (xnn_pack_lh_ukernel_fn)xnn_x16_pack_lh_ukernel__neonsme2;
    x16_pack_lh_config.size_fn =
        (xnn_pack_lh_size_fn)xnn_x16_pack_lh_size__neonsme2;
    x16_pack_lh_config.offset_fn =
        (xnn_pack_lh_offset_fn)xnn_x16_pack_lh_offset__neonsme2;
  }
#endif  // XNN_ENABLE_ARM_SME2
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
  x16_pack_lh_config.log2_input_element_size = 1;
  x16_pack_lh_config.log2_packed_element_size = 1;
  x16_pack_lh_config.gemv_noop = true;
}

const struct xnn_pack_lh_config* xnn_init_x16_pack_lh_config() {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(x16_pack_lh);
  return &x16_pack_lh_config;
}

static void init_x8_pack_lh_config(void) {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  assert(hardware_config != NULL);
  (void)hardware_config;
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
#if XNN_ENABLE_ARM_SME
  if ((hardware_config->arch_flags & xnn_arch_arm_sme)) {
    x8_pack_lh_config.pack_lh_fn = (xnn_pack_lh_ukernel_fn) xnn_x8_pack_lh_ukernel__neonsme;
    x8_pack_lh_config.size_fn = (xnn_pack_lh_size_fn) xnn_x8_pack_lh_size__neonsme;
    x8_pack_lh_config.offset_fn = (xnn_pack_lh_offset_fn) xnn_x8_pack_lh_offset__neonsme;
  }
#endif  // XNN_ENABLE_ARM_SME
#if XNN_ENABLE_ARM_SME2
  if ((hardware_config->arch_flags & xnn_arch_arm_sme2)) {
    x8_pack_lh_config.pack_lh_fn = (xnn_pack_lh_ukernel_fn)xnn_x8_pack_lh_ukernel__neonsme2;
    x8_pack_lh_config.size_fn = (xnn_pack_lh_size_fn)xnn_x8_pack_lh_size__neonsme2;
    x8_pack_lh_config.offset_fn = (xnn_pack_lh_offset_fn)xnn_x8_pack_lh_offset__neonsme2;
  }
#endif  // XNN_ENABLE_ARM_SME2
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
  x8_pack_lh_config.log2_input_element_size = 0;
  x8_pack_lh_config.log2_packed_element_size = 0;
  x8_pack_lh_config.gemv_noop = true;
}

const struct xnn_pack_lh_config* xnn_init_x8_pack_lh_config() {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(x8_pack_lh);
  return &x8_pack_lh_config;
}

static void init_x8_igemm_pack_lh_config(void) {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  assert(hardware_config != NULL);
  (void)hardware_config;
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
#if XNN_ENABLE_ARM_SME2
  if ((hardware_config->arch_flags & xnn_arch_arm_sme2)) {
    x8_igemm_pack_lh_config.pack_lh_for_igemm_fn =
        (xnn_pack_lh_igemm_ukernel_fn)xnn_x8_pack_lh_ukernel__igemm_neonsme2;
    x8_igemm_pack_lh_config.size_for_igemm_fn =
        (xnn_pack_lh_igemm_size_fn)xnn_x8_pack_lh_size__igemm_neonsme2;
    x8_igemm_pack_lh_config.offset_for_igemm_fn =
        (xnn_pack_lh_igemm_offset_fn)xnn_x8_pack_lh_offset__igemm_neonsme2;
  }
#endif  // XNN_ENABLE_ARM_SME2
#if XNN_ENABLE_ARM_SME
  if ((hardware_config->arch_flags & xnn_arch_arm_sme)) {
    x8_igemm_pack_lh_config.pack_lh_for_igemm_fn = (xnn_pack_lh_igemm_ukernel_fn) xnn_x8_pack_lh_ukernel__igemm_neonsme;
    x8_igemm_pack_lh_config.size_for_igemm_fn = (xnn_pack_lh_igemm_size_fn) xnn_x8_pack_lh_size__igemm_neonsme;
    x8_igemm_pack_lh_config.offset_for_igemm_fn = (xnn_pack_lh_igemm_offset_fn) xnn_x8_pack_lh_offset__igemm_neonsme;
  }
#endif  // XNN_ENABLE_ARM_SME
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
  x8_igemm_pack_lh_config.log2_input_element_size = 0;
  x8_igemm_pack_lh_config.log2_packed_element_size = 0;
}

const struct xnn_pack_lh_config* xnn_init_x8_igemm_pack_lh_config() {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(x8_igemm_pack_lh);
  return &x8_igemm_pack_lh_config;
}

static void init_x16_igemm_pack_lh_config(void) {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  assert(hardware_config != NULL);
  (void)hardware_config;
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
#if XNN_ENABLE_ARM_SME2
  if ((hardware_config->arch_flags & xnn_arch_arm_sme2)) {
    x16_igemm_pack_lh_config.pack_lh_for_igemm_fn =
        (xnn_pack_lh_igemm_ukernel_fn)xnn_x16_pack_lh_ukernel__igemm_neonsme2;
    x16_igemm_pack_lh_config.size_for_igemm_fn =
        (xnn_pack_lh_igemm_size_fn)xnn_x16_pack_lh_size__igemm_neonsme2;
    x16_igemm_pack_lh_config.offset_for_igemm_fn =
        (xnn_pack_lh_igemm_offset_fn)xnn_x16_pack_lh_offset__igemm_neonsme2;
  }
#endif  // XNN_ENABLE_ARM_SME2
#if XNN_ENABLE_ARM_SME
  if ((hardware_config->arch_flags & xnn_arch_arm_sme)) {
    x16_igemm_pack_lh_config.pack_lh_for_igemm_fn =
        (xnn_pack_lh_igemm_ukernel_fn)xnn_x16_pack_lh_ukernel__igemm_neonsme;
    x16_igemm_pack_lh_config.size_for_igemm_fn =
        (xnn_pack_lh_igemm_size_fn)xnn_x16_pack_lh_size__igemm_neonsme;
    x16_igemm_pack_lh_config.offset_for_igemm_fn =
        (xnn_pack_lh_igemm_offset_fn)xnn_x16_pack_lh_offset__igemm_neonsme;
  }
#endif  // XNN_ENABLE_ARM_SME
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
  x16_igemm_pack_lh_config.log2_input_element_size = 1;
  x16_igemm_pack_lh_config.log2_packed_element_size = 1;
}

const struct xnn_pack_lh_config* xnn_init_x16_igemm_pack_lh_config() {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(x16_igemm_pack_lh);
  return &x16_igemm_pack_lh_config;
}
