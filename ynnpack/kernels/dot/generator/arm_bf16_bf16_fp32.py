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
    self.a_load_type = "uint16_t"
    self.b_type = "bfloat16"
    self.b_load_type = "uint16_t"
    self.flags += ["dot_flag::consistent_arithmetic"]
    self.flags += ["dot_flag::unaligned_b"]

  def b_alignment_bytes(self):
    return 2

  def header(self):
    return super().header() + """

namespace {

YNN_INTRINSIC float32x4_t bf16_to_f32(uint16x4_t x) {
  return vreinterpretq_f32_u32(vshll_n_u16(x, 16));
}

}  // namespace
"""

  def load_a_tile(self, i, k):
    if k % 4 == 0:
      return (
          f"float32x4_t a_{i}_{k} ="
          f" bf16_to_f32(vld1_u16({self.a_ptr(i, k)}));\n"
      )
    return ""

  def load_a_tile_tail(self, i, k):
    return (
        f"float32x4_t a_{i}_{k} ="
        f" bf16_to_f32(vdup_n_u16(*{self.a_ptr(i, k)}));\n"
    )

  def load_b_tile(self, k, j):
    b_ptr = self.b_ptr(k, j)
    return f"float32x4_t b_{k}_{j} = bf16_to_f32(vld1_u16({b_ptr}));\n"

  def load_b_tile_tail(self, k, j):
    b_ptr = self.b_ptr(k, j)
    size = f"min(sizeof(uint16x4_t), sub_sat(N, {j}) * sizeof({self.b_type}))"
    return f"""\
uint16x4_t b_raw_{k}_{j} = vdup_n_u16(0);
std::memcpy(&b_raw_{k}_{j}, {b_ptr}, {size});
float32x4_t b_{k}_{j} = bf16_to_f32(b_raw_{k}_{j});
"""


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
        (1, 64, 4),
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
