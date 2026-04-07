# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp64 arm dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.arm import arm_neon
from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels


class arm_neon_fp64(arm_neon):
  def __init__(self):
    super().__init__("neon", "fp64", "double", (1, 2, 1))
    self.a_type = "double"
    self.b_type = "double"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def load_a_tile_k_tail(self, i, k, nk):
    a_ptr = self.a_ptr(i, k)
    if nk == 1:
      return f"float64x2_t a_{i}_{k} = vdupq_n_f64(*{a_ptr});\n"
    elif k % 2 == 0:
      return f"float64x2_t a_{i}_{k} = vld1q_f64({a_ptr});\n"
    else:
      return ""

  def load_b_tile(self, k, j):
    return f"float64x2_t b_{k}_{j} = vld1q_f64({self.b_ptr(k, j)});\n"


class arm64_neon_fp64(arm_neon_fp64):
  def product(self, i, j, k):
    c_ij = f"c_{i}_{j}"
    b_kj = f"b_{k}_{j}"
    a_ik = f"a_{i}_{(k//2)*2}"
    _, _, block_k = self.block_shape
    if block_k == 1:
      return f"{c_ij} = vfmaq_f64({c_ij}, {b_kj}, {a_ik});\n"
    else:
      assert block_k % 2 == 0
      return f"{c_ij} = vfmaq_laneq_f64({c_ij}, {b_kj}, {a_ik}, {k%2});\n"


generate_dot_kernels(
    arm64_neon_fp64(),
    [
        (1, 16, 2),
        (2, 16, 2),
        (3, 16, 2),
        (1, 8, 2),
        (2, 8, 2),
        (3, 8, 2),
        (4, 8, 2),
        (5, 8, 2),
        (1, 4, 2),
        (4, 4, 2),
        (6, 4, 2),
        (8, 4, 2),
        (8, 2, 2),
    ],
)
