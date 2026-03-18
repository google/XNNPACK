# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp32 arm dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.arm import arm_neon


class arm_neon_bf16_bf16_fp32(arm_neon):
  def __init__(self):
    super().__init__("neon", "bf16_bf16_fp32", "float", (1, 4, 1))
    self.a_type = "bfloat16"
    self.b_type = "bfloat16"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def header(self):
    return super().header() + """

namespace {

struct bfloat16 {
  std::uint16_t value;
};

YNN_INTRINSIC float32x4_t unaligned_load_broadcast_bf16(const bfloat16* ptr) {
    uint16_t value;
    memcpy(&value, ptr, sizeof(uint16_t));
    return vreinterpretq_f32_u32(vshll_n_u16(vdup_n_u16(value), 16));
}

YNN_INTRINSIC float32x4_t load_bf16x4(const bfloat16* ptr) {
    return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(reinterpret_cast<const uint16_t*>(ptr)), 16));
}

}  // namespace
"""

  def load_a_tile_k_tail(self, i, k, nk):
    if k % nk != 0:
      return ""
    if nk == 4:
      return f"float32x4_t a_{i}_{k} = load_bf16x4({self.a_ptr(i, k)});\n"
    elif nk == 1:
      return (
          f"float32x4_t a_{i}_{k} ="
          f" unaligned_load_broadcast_bf16({self.a_ptr(i, k)});\n"
      )

  def load_b_tile(self, k, j):
    return f"float32x4_t b_{k}_{j} = load_bf16x4({self.b_ptr(k, j)});\n"


class arm64_neon_bf16_bf16_fp32(arm_neon_bf16_bf16_fp32):
  def __init__(self):
    super().__init__()

  def product(self, i, j, k):
    if self.block_shape[2] == 4:
      return f"c_{i}_{j} = vfmaq_laneq_f32(c_{i}_{j}, b_{k}_{j}, a_{i}_{(k//4)*4}, {k%4});\n"
    if self.block_shape[2] == 1:
      return f"c_{i}_{j} = vfmaq_f32(c_{i}_{j}, b_{k}_{j}, a_{i}_{k});\n"
