# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp32 arm dot kernel generators."""

from ynnpack.kernels.dot.generator.arm import arm_neon


class arm_neon_fp32(arm_neon):
  def __init__(self):
    super().__init__("neon", "fp32", "float", (1, 4, 1))
    self.a_type = "float"
    self.b_type = "float"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def header(self):
    return super().header() + """

namespace {

YNN_INTRINSIC float32x4_t unaligned_load_broadcast(const void* ptr) {
    float value;
    memcpy(&value, ptr, sizeof(float));
    return vdupq_n_f32(value);
}

}  // namespace
"""

  def load_a_tile_k_tail(self, i, k, nk):
    if k % nk != 0:
      return ""
    if nk == 4:
      return f"float32x4_t a_{i}_{k} = vld1q_f32({self.a_ptr(i, k)});\n"
    elif nk == 2:
      return f"float32x2_t a_{i}_{k} = vld1_f32({self.a_ptr(i, k)});\n"
    elif nk == 1:
      return f"float32x4_t a_{i}_{k} = unaligned_load_broadcast({self.a_ptr(i, k)});\n"

  def load_b_tile(self, k, j):
    return (
        f"float32x4_t b_{k}_{j} = vld1q_f32({self.b_ptr(k, j)});\n"
    )


class arm64_neon_fp32(arm_neon_fp32):
  def __init__(self):
    super().__init__()

  def product(self, i, j, k):
    if self.block_shape[2] == 4:
      return f"c_{i}_{j} = vfmaq_laneq_f32(c_{i}_{j}, b_{k}_{j}, a_{i}_{(k//4)*4}, {k%4});\n"
    if self.block_shape[2] == 2:
      return f"c_{i}_{j} = vfmaq_lane_f32(c_{i}_{j}, b_{k}_{j}, a_{i}_{(k//2)*2}, {k%2});\n"
    if self.block_shape[2] == 1:
      return f"c_{i}_{j} = vfmaq_f32(c_{i}_{j}, b_{k}_{j}, a_{i}_{k});\n"
