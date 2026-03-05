# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp64 arm dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.arm import arm_neon


class arm_neon_fp64(arm_neon):
  def __init__(self):
    super().__init__("neon", "fp64", "double", (1, 2, 1))
    self.a_type = "double"
    self.b_type = "double"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def header(self):
    return super().header() + """

namespace {

YNN_INTRINSIC float64x2_t unaligned_load_broadcast(const void* ptr) {
    double value;
    memcpy(&value, ptr, sizeof(double));
    return vdupq_n_f64(value);
}

}  // namespace
"""

  def load_a_tile_k_tail(self, i, k, nk):
    if k % nk != 0:
      return ""
    if nk == 2:
      return f"float64x2_t a_{i}_{k} = vld1q_f64({self.a_ptr(i, k)});\n"
    elif nk == 1:
      return (
          f"float64x2_t a_{i}_{k} ="
          f" unaligned_load_broadcast({self.a_ptr(i, k)});\n"
      )

  def load_b_tile(self, k, j):
    return f"float64x2_t b_{k}_{j} = vld1q_f64({self.b_ptr(k, j)});\n"


class arm64_neon_fp64(arm_neon_fp64):
  def __init__(self):
    super().__init__()

  def product(self, i, j, k):
    if self.block_shape[2] == 2:
      return (
          f"c_{i}_{j} = vfmaq_laneq_f64(c_{i}_{j}, b_{k}_{j}, a_{i}_{(k//2)*2},"
          f" {k%2});\n"
      )
    if self.block_shape[2] == 1:
      return f"c_{i}_{j} = vfmaq_f64(c_{i}_{j}, b_{k}_{j}, a_{i}_{k});\n"
