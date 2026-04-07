# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp32 arm dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.arm import arm_neon
from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels


class arm_neon_bf16_bf16_fp32(arm_neon):
  def __init__(self):
    super().__init__("neon", "bf16_bf16_fp32", "float", (1, 4, 1))
    self.a_type = "bfloat16"
    self.b_type = "bfloat16"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def header(self):
    return super().header() + """

namespace {

using bfloat16 = uint16_t;

YNN_INTRINSIC float32x4_t bf16_to_f32(uint16x4_t x) {
  return vreinterpretq_f32_u32(vshll_n_u16(x, 16));
}

}  // namespace
"""

  def load_a_tile_k_tail(self, i, k, nk):
    a_ptr = self.a_ptr(i, k)
    if nk == 1:
      return f"float32x4_t a_{i}_{k} = bf16_to_f32(vdup_n_u16(*{a_ptr}));\n"
    elif k % 4 == 0:
      assert nk % 4 == 0
      return f"float32x4_t a_{i}_{k} = bf16_to_f32(vld1_u16({a_ptr}));\n"
    else:
      return ""

  def load_b_tile(self, k, j):
    b_ptr = self.b_ptr(k, j)
    return f"float32x4_t b_{k}_{j} = bf16_to_f32(vld1_u16({b_ptr}));\n"


class arm64_neon_bf16_bf16_fp32(arm_neon_bf16_bf16_fp32):
  def product(self, i, j, k):
    a_ik = f"a_{i}_{(k//4)*4}"
    b_kj = f"b_{k}_{j}"
    c_ij = f"c_{i}_{j}"
    _, _, block_k = self.block_shape
    if block_k == 1:
      return f"{c_ij} = vfmaq_f32({c_ij}, {b_kj}, {a_ik});\n"
    else:
      assert block_k % 4 == 0
      return f"{c_ij} = vfmaq_laneq_f32({c_ij}, {b_kj}, {a_ik}, {k%4});\n"


generate_dot_kernels(
    arm64_neon_bf16_bf16_fp32(),
    [
        (1, 32, 4),
        (2, 32, 4),
        (3, 32, 4),
        (1, 16, 4),
        (2, 16, 4),
        (3, 16, 4),
        (4, 16, 4),
        (5, 16, 4),
        (1, 8, 4),
        (4, 8, 4),
        (6, 8, 4),
        (8, 8, 4),
        (8, 4, 4),
    ],
)
