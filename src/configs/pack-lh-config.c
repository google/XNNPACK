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

static struct xnn_pack_lh_config x8_pack_lh_config = {0};
static struct xnn_pack_lh_config x16_pack_lh_config = {0};
static struct xnn_pack_lh_config x32_pack_lh_config = {0};

XNN_INIT_ONCE_GUARD(x8_pack_lh);
XNN_INIT_ONCE_GUARD(x16_pack_lh);
XNN_INIT_ONCE_GUARD(x32_pack_lh);

static void init_x32_pack_lh_config(void) {
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
    #if XNN_ENABLE_ARM_SME2
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
      if (hardware_config->use_arm_sme2) {
        x32_pack_lh_config.ukernel = (xnn_pack_lh_ukernel_fn) xnn_x32_pack_lh_ukernel__neonsme2;
        x32_pack_lh_config.size_fn = (xnn_pack_lh_size_fn) xnn_x32_pack_lh_size__neonsme2;
        x32_pack_lh_config.offset_fn = (xnn_pack_lh_offset_fn) xnn_x32_pack_lh_offset__neonsme2;
      }
    #endif  // XNN_ENABLE_ARM_SME2
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
}

const struct xnn_pack_lh_config* xnn_init_x32_pack_lh_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(x32_pack_lh);
  return &x32_pack_lh_config;
}

static void init_x16_pack_lh_config(void) {
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
    #if XNN_ENABLE_ARM_SME2
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
      if (hardware_config->use_arm_sme2) {
        x16_pack_lh_config.ukernel = (xnn_pack_lh_ukernel_fn) xnn_x16_pack_lh_ukernel__neonsme2;
        x16_pack_lh_config.size_fn = (xnn_pack_lh_size_fn) xnn_x16_pack_lh_size__neonsme2;
        x16_pack_lh_config.offset_fn = (xnn_pack_lh_offset_fn) xnn_x16_pack_lh_offset__neonsme2;
      }
    #endif  // XNN_ENABLE_ARM_SME2
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
}

const struct xnn_pack_lh_config* xnn_init_x16_pack_lh_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(x16_pack_lh);
  return &x16_pack_lh_config;
}

static void init_x8_pack_lh_config(void) {
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
    #if XNN_ENABLE_ARM_SME2
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
      if (hardware_config->use_arm_sme2) {
        x8_pack_lh_config.ukernel = (xnn_pack_lh_ukernel_fn) xnn_x8_pack_lh_ukernel__neonsme2;
        x8_pack_lh_config.size_fn = (xnn_pack_lh_size_fn) xnn_x8_pack_lh_size__neonsme2;
        x8_pack_lh_config.offset_fn = (xnn_pack_lh_offset_fn) xnn_x8_pack_lh_offset__neonsme2;
      }
    #endif  // XNN_ENABLE_ARM_SME2
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
}

const struct xnn_pack_lh_config* xnn_init_x8_pack_lh_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(x8_pack_lh);
  return &x8_pack_lh_config;
}
