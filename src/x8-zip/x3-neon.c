// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <xnnpack/zip.h>


void xnn_x8_zip_x3_ukernel__neon(
    size_t n,
    const uint8_t* input,
    uint8_t* output)
{
  const uint8_t* x = input;
  const uint8_t* y = (const uint8_t*) ((uintptr_t) x + n);
  const uint8_t* z = (const uint8_t*) ((uintptr_t) y + n);
  uint8_t* o = output;

  if (n >= 8) {
    do {
      uint8x8x3_t vxyz;
      vxyz.val[0] = vld1_u8(x); x += 8;
      vxyz.val[1] = vld1_u8(y); y += 8;
      vxyz.val[2] = vld1_u8(z); z += 8;
      vst3_u8(o, vxyz); o += 24;
      n -= 8;
    } while (n >= 8);
    if (n != 0) {
      const size_t address_increment = n - 8;
      uint8x8x3_t vxyz;
      vxyz.val[0] = vld1_u8(x + address_increment);
      vxyz.val[1] = vld1_u8(y + address_increment);
      vxyz.val[2] = vld1_u8(z + address_increment);
      vst3_u8((uint8_t*) ((uintptr_t) o + address_increment * 3), vxyz);
    }
  } else {
    do {
      const uint8_t vx = *x++;
      const uint8_t vy = *y++;
      const uint8_t vz = *z++;
      o[0] = vx;
      o[1] = vy;
      o[2] = vz;
      o += 3;
    } while (--n != 0);
  }
}
