# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp32 arm dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.arm import arm_neon


class arm_neon_bf16_bf16_fp32(arm_neon):
  def __init__(self, arch="neon", tile_shape=(1, 4, 2)):
    super().__init__(arch, "bf16_bf16_fp32", "float", tile_shape)
    self.a_type = "bfloat16"
    self.b_type = "bfloat16"


class arm_neonbf16_bf16_bf16_fp32_k2(arm_neon_bf16_bf16_fp32):
  def __init__(self, arch="neonbf16", tile_shape=(1, 4, 2)):
    super().__init__(arch, tile_shape)

  def header(self):
    return super().header() + """

using bfloat16 = __bf16;

namespace {

YNN_INTRINSIC bfloat16x4_t unaligned_load_broadcast_2xbf16(const bfloat16* ptr) {
    float value;
    memcpy(&value, ptr, sizeof(float));
    return vreinterpret_bf16_f32(vdup_n_f32(value));
}

}  // namespace
"""

  def load_a_tile_k_tail(self, i, k, nk):
    if k % nk != 0:
      return ""
    a_ptr = self.a_ptr(i, k)
    a = f"a_{i}_{k}"
    if nk == 8:
      return f"bfloat16x8_t {a} = vld1q_bf16({a_ptr});\n"
    elif nk == 4:
      return f"bfloat16x4_t {a} = vld1_bf16({a_ptr});\n"
    else:
      return f"bfloat16x4_t {a} = unaligned_load_broadcast_2xbf16({a_ptr});\n"

  def load_b_tile(self, k, j):
    return f"bfloat16x8_t b_{k}_{j} = vld1q_bf16({self.b_ptr(k, j)});\n"

  def product(self, i, j, k):
    c = f"c_{i}_{j}"
    b = f"b_{k}_{j}"
    if self.block_shape[2] == 8:
      return f"{c} = vbfdotq_laneq_f32({c}, {b}, a_{i}_{0}, {k//2});\n"
    elif self.block_shape[2] == 4:
      return f"{c} = vbfdotq_lane_f32({c}, {b}, a_{i}_{0}, {k//2});\n"
    else:
      return f"{c} = vbfdotq_lane_f32({c}, {b}, a_{i}_{k}, 0);\n"
