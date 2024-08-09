// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/init-once.h"
#include "xnnpack/lut.h"


static struct xnn_lut32norm_config u8_lut32norm_config = {0};

XNN_INIT_ONCE_GUARD(u8_lut32norm);

static void init_u8_lut32norm_config(void) {
  u8_lut32norm_config.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
}

const struct xnn_lut32norm_config* xnn_init_u8_lut32norm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(u8_lut32norm);
  return &u8_lut32norm_config;
}
