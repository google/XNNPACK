# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp32 arm dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.arm import arm_neon
from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels


class arm_neon_fp32(arm_neon):

  def __init__(self, arch="neon", tile_shape=(1, 4, 1)):
    super().__init__(arch, "fp32", "float", tile_shape)
    self.a_type = "float"
    self.b_type = "float"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def load_a_tile_k_tail(self, i, k, nk):
    a_ptr = self.a_ptr(i, k)
    if nk == 1:
      return f"float32x4_t a_{i}_{k} = vdupq_n_f32(*{a_ptr});\n"
    elif nk == 2:
      return f"float32x2_t a_{i}_{k} = vld1_f32({a_ptr});\n"
    elif k % 4 == 0:
      assert nk % 4 == 0
      return f"float32x4_t a_{i}_{k} = vld1q_f32({a_ptr});\n"
    else:
      return ""

  def load_b_tile(self, k, j):
    return f"float32x4_t b_{k}_{j} = vld1q_f32({self.b_ptr(k, j)});\n"


class arm64_neon_fp32(arm_neon_fp32):
  def product(self, i, j, k):
    c_ij = f"c_{i}_{j}"
    b_kj = f"b_{k}_{j}"
    a_ik = f"a_{i}_{(k//4)*4}"
    _, _, block_k = self.block_shape
    if block_k == 1:
      return f"{c_ij} = vfmaq_f32({c_ij}, {b_kj}, {a_ik});\n"
    elif block_k == 2:
      return f"{c_ij} = vfmaq_lane_f32({c_ij}, {b_kj}, {a_ik}, {k%2});\n"
    else:
      assert block_k % 4 == 0
      return f"{c_ij} = vfmaq_laneq_f32({c_ij}, {b_kj}, {a_ik}, {k%4});\n"


generate_dot_kernels(
    arm64_neon_fp32(),
    [
        (1, 32, 4),
        (2, 32, 4),
        (3, 32, 4),
        (2, 16, 4),
        (3, 16, 4),
        (4, 16, 4),
        (5, 16, 4),
        (4, 8, 4),
        (6, 8, 4),
        (8, 8, 4),
        (8, 4, 4),
    ],
)
