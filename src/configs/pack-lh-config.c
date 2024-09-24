// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdio.h>

#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/init-once.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/pack-lh.h"

static struct xnn_pack_lh_config x32_pack_lh_config = {0};

XNN_INIT_ONCE_GUARD(x32_pack_lh);

static void init_x32_pack_lh_config(void) {
    //const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    //assert(hardware_config != NULL);
  #if XNN_ENABLE_KLEIDIAI
    //if (hardware_config->use_arm_sme2) {
      //kai_run_lhs_pack_f32p2vlx1_f32_sme
      x32_pack_lh_config.ukernel = (xnn_x32_pack_lh_ukernel_fn) xnn_x32_pack_lh_ukernel__neonsme2_u2;
      x32_pack_lh_config.size_fn = (xnn_x32_pack_lh_size_fn) xnn_x32_pack_lh_size__neonsme2_u2;
    //}
  #endif
}

const struct xnn_pack_lh_config* xnn_init_x32_pack_lh_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(x32_pack_lh);
  return &x32_pack_lh_config;
}
