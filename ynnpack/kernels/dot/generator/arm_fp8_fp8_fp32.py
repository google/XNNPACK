# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp8 arm dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.arm import arm_neon
from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels


class arm_fp8_fp8_fp32(arm_neon):
  def __init__(self, arch, fp8_type, tile_shape):
    dot_type = f"{fp8_type}_{fp8_type}_fp32"
    super().__init__(arch, dot_type, "float", tile_shape)
    self.fp8_type = fp8_type
    self.a_type = fp8_type
    self.b_type = fp8_type
    self.flags += ["dot_flag::consistent_arithmetic"]

  def header(self):
    return super().header() + """

namespace {

using fp8_e5m2 = uint8_t;
using fp8_e4m3 = uint8_t;

YNN_INTRINSIC uint32_t unaligned_load_uint8x4(const uint8_t* ptr) {
    uint32_t value;
    memcpy(&value, ptr, sizeof(uint32_t));
    return value;
}

}  // namespace
"""

  def begin_func(self, func_name):
    if self.fp8_type == "fp8_e5m2":
      fpm_format = "__ARM_FPM_E5M2"
    elif self.fp8_type == "fp8_e4m3":
      fpm_format = "__ARM_FPM_E4M3"
    else:
      raise ValueError(f"Unsupported fp8_type: {self.fp8_type}")

    return super().begin_func(func_name) + f"""
  fpm_t fpm = __arm_fpm_init();
  fpm = __arm_set_fpm_src1_format(fpm, {fpm_format});
  fpm = __arm_set_fpm_src2_format(fpm, {fpm_format});
"""

  def load_a_tile_k_tail(self, i, k, nk):
    if k % 8 != 0:
      return ""
    if nk == 8:
      return (
          f"mfloat8x8_t a_{i}_{k} ="
          f" vreinterpret_mf8_u8(vld1_u8({self.a_ptr(i, k)}));\n"
      )
    else:
      a_x4 = f"unaligned_load_uint8x4({self.a_ptr(i, k)})"
      return (
          f"mfloat8x8_t a_{i}_{k} = vreinterpret_mf8_u32(vdup_n_u32({a_x4}));\n"
      )

  def load_b_tile(self, k, j):
    return (
        f"mfloat8x16_t b_{k}_{j} ="
        f" vreinterpretq_mf8_u8(vld1q_u8({self.b_ptr(k, j, 'uint8_t')}));\n"
    )

  def product(self, i, j, k):
    c_ij = f"c_{i}_{j}"
    b_kj = f"b_{k}_{j}"
    a_ik = f"a_{i}_{(k//8)*8}"
    lane = (k // 4) % 2
    return (
        f"{c_ij} = vdotq_lane_f32_mf8_fpm({c_ij}, {b_kj}, {a_ik}, {lane},"
        " fpm);\n"
    )


class arm64_neonfp8dot4_fp8_e5m2_fp8_e5m2_fp32(arm_fp8_fp8_fp32):
  def __init__(self):
    super().__init__("neonfp8dot4", "fp8_e5m2", (1, 4, 4))


class arm64_neonfp8dot4_fp8_e4m3_fp8_e4m3_fp32(arm_fp8_fp8_fp32):
  def __init__(self):
    super().__init__("neonfp8dot4", "fp8_e4m3", (1, 4, 4))


# Generate kernels
shapes = [
    (1, 32, 8),
    (2, 32, 8),
    (3, 32, 8),
    (1, 16, 8),
    (2, 16, 8),
    (3, 16, 8),
    (4, 16, 8),
    (5, 16, 8),
    (1, 8, 8),
    (8, 8, 8),
    (10, 8, 8),
    (8, 4, 8),
]

generate_dot_kernels(arm64_neonfp8dot4_fp8_e5m2_fp8_e5m2_fp32(), shapes)
generate_dot_kernels(arm64_neonfp8dot4_fp8_e4m3_fp8_e4m3_fp32(), shapes)
