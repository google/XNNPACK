// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/pack-lh.h"
#include "src/xnnpack/packq.h"

static struct xnn_pack_lh_config f16_qdint8_pack_lh_config = {0};
static struct xnn_pack_lh_config f16_qduint8_pack_lh_config = {0};
static struct xnn_pack_lh_config f32_qdint8_pack_lh_config = {0};
static struct xnn_pack_lh_config f32_qduint8_pack_lh_config = {0};
static struct xnn_pack_lh_config qp8_pack_lh_config = {0};
static struct xnn_pack_lh_config x8_pack_lh_config = {0};
static struct xnn_pack_lh_config x16_pack_lh_config = {0};
static struct xnn_pack_lh_config x32_pack_lh_config = {0};
static struct xnn_pack_lh_config x8_igemm_pack_lh_config = {0};

XNN_INIT_ONCE_GUARD(f16_qdint8_pack_lh);
XNN_INIT_ONCE_GUARD(f16_qduint8_pack_lh);
XNN_INIT_ONCE_GUARD(f32_qdint8_pack_lh);
XNN_INIT_ONCE_GUARD(f32_qduint8_pack_lh);
XNN_INIT_ONCE_GUARD(qp8_pack_lh);
XNN_INIT_ONCE_GUARD(x8_pack_lh);
XNN_INIT_ONCE_GUARD(x16_pack_lh);
XNN_INIT_ONCE_GUARD(x32_pack_lh);
XNN_INIT_ONCE_GUARD(x8_igemm_pack_lh);

static void init_f16_qdint8_pack_lh_config(void) {
  f16_qdint8_pack_lh_config.pack_lh_fn =
      (xnn_pack_lh_ukernel_fn)xnn_pack_lh_f16_qdint8;
  f16_qdint8_pack_lh_config.size_fn =
      (xnn_pack_lh_size_fn)xnn_pack_lh_fx_qd8_packed_size;
  f16_qdint8_pack_lh_config.offset_fn =
      (xnn_pack_lh_offset_fn)xnn_pack_lh_fx_qd8_packed_offset;
  f16_qdint8_pack_lh_config.log2_input_element_size = XNN_LOG2_SIZEOF_HALF;
  f16_qdint8_pack_lh_config.log2_packed_element_size = 0;
}

const struct xnn_pack_lh_config* xnn_init_f16_qdint8_pack_lh_config() {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_qdint8_pack_lh);
  return &f16_qdint8_pack_lh_config;
}

static void init_f16_qduint8_pack_lh_config(void) {
  f16_qduint8_pack_lh_config.pack_lh_fn =
      (xnn_pack_lh_ukernel_fn)xnn_pack_lh_f16_qduint8;
  f16_qduint8_pack_lh_config.size_fn =
      (xnn_pack_lh_size_fn)xnn_pack_lh_fx_qd8_packed_size;
  f16_qduint8_pack_lh_config.offset_fn =
      (xnn_pack_lh_offset_fn)xnn_pack_lh_fx_qd8_packed_offset;
  f16_qduint8_pack_lh_config.log2_input_element_size = XNN_LOG2_SIZEOF_HALF;
  f16_qduint8_pack_lh_config.log2_packed_element_size = 0;
}

const struct xnn_pack_lh_config* xnn_init_f16_qduint8_pack_lh_config() {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_qduint8_pack_lh);
  return &f16_qduint8_pack_lh_config;
}

static void init_f32_qdint8_pack_lh_config(void) {
  f32_qdint8_pack_lh_config.pack_lh_fn =
      (xnn_pack_lh_ukernel_fn)xnn_pack_lh_f32_qdint8;
  f32_qdint8_pack_lh_config.size_fn =
      (xnn_pack_lh_size_fn)xnn_pack_lh_fx_qd8_packed_size;
  f32_qdint8_pack_lh_config.offset_fn =
      (xnn_pack_lh_offset_fn)xnn_pack_lh_fx_qd8_packed_offset;
  f32_qdint8_pack_lh_config.log2_input_element_size = XNN_LOG2_SIZEOF_FLOAT;
  f32_qdint8_pack_lh_config.log2_packed_element_size = 0;
}

const struct xnn_pack_lh_config* xnn_init_f32_qdint8_pack_lh_config() {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_qdint8_pack_lh);
  return &f32_qdint8_pack_lh_config;
}

static void init_f32_qduint8_pack_lh_config(void) {
  f32_qduint8_pack_lh_config.pack_lh_fn =
      (xnn_pack_lh_ukernel_fn)xnn_pack_lh_f32_qduint8;
  f32_qduint8_pack_lh_config.size_fn =
      (xnn_pack_lh_size_fn)xnn_pack_lh_fx_qd8_packed_size;
  f32_qduint8_pack_lh_config.offset_fn =
      (xnn_pack_lh_offset_fn)xnn_pack_lh_fx_qd8_packed_offset;
  f32_qduint8_pack_lh_config.log2_input_element_size = XNN_LOG2_SIZEOF_FLOAT;
  f32_qduint8_pack_lh_config.log2_packed_element_size = 0;
}

const struct xnn_pack_lh_config* xnn_init_f32_qduint8_pack_lh_config() {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_qduint8_pack_lh);
  return &f32_qduint8_pack_lh_config;
}

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
#if XNN_ENABLE_ARM_SME2 || XNN_ENABLE_ARM_SME
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  assert(hardware_config != NULL);
  if (hardware_config->arch_flags & xnn_arch_arm_sme) {
    x32_pack_lh_config.pack_lh_fn =
        (xnn_pack_lh_ukernel_fn)xnn_x32_pack_lh_ukernel__neonsme;
    x32_pack_lh_config.size_fn =
        (xnn_pack_lh_size_fn)xnn_x32_pack_lh_size__neonsme;
    x32_pack_lh_config.offset_fn =
        (xnn_pack_lh_offset_fn)xnn_x32_pack_lh_offset__neonsme;
  }
#endif  // XNN_ENABLE_ARM_SME2 || XNN_ENABLE_ARM_SME
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

static void init_x16_pack_lh_config(void) {
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
#if XNN_ENABLE_ARM_SME2
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  assert(hardware_config != NULL);
  if ((hardware_config->arch_flags & xnn_arch_arm_sme2)) {
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
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
#if XNN_ENABLE_ARM_SME2
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  assert(hardware_config != NULL);
  if ((hardware_config->arch_flags & xnn_arch_arm_sme2)) {
    x8_pack_lh_config.pack_lh_fn =
        (xnn_pack_lh_ukernel_fn)xnn_x8_pack_lh_ukernel__neonsme2;
    x8_pack_lh_config.size_fn =
        (xnn_pack_lh_size_fn)xnn_x8_pack_lh_size__neonsme2;
    x8_pack_lh_config.offset_fn =
        (xnn_pack_lh_offset_fn)xnn_x8_pack_lh_offset__neonsme2;
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
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
#if XNN_ENABLE_ARM_SME2
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  assert(hardware_config != NULL);
  if ((hardware_config->arch_flags & xnn_arch_arm_sme2)) {
    x8_igemm_pack_lh_config.pack_lh_for_igemm_fn =
        (xnn_pack_lh_igemm_ukernel_fn)xnn_x8_pack_lh_ukernel__igemm_neonsme2;
    x8_igemm_pack_lh_config.size_for_igemm_fn =
        (xnn_pack_lh_igemm_size_fn)xnn_x8_pack_lh_size__igemm_neonsme2;
    x8_igemm_pack_lh_config.offset_for_igemm_fn =
        (xnn_pack_lh_igemm_offset_fn)xnn_x8_pack_lh_offset__igemm_neonsme2;
  }
#endif  // XNN_ENABLE_ARM_SME2
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
