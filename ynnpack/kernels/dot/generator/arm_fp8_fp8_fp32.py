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

  def header(self):
    return super().header() + """

namespace {

using fp8_e5m2 = uint8_t;
using fp8_e4m3 = uint8_t;

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


class arm_neonfp8dot4_fp8_fp8_fp32(arm_fp8_fp8_fp32):

  def __init__(self, arch, fp8_type, tile_shape):
    super().__init__(arch, fp8_type, tile_shape)
    self.flags += ["dot_flag::consistent_arithmetic"]

  def header(self):
    return super().header() + """

namespace {

YNN_INTRINSIC uint32_t unaligned_load_uint8x4(const uint8_t* ptr) {
    uint32_t value;
    memcpy(&value, ptr, sizeof(uint32_t));
    return value;
}

}  // namespace
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


class arm64_neonfp8dot4_fp8_e5m2_fp8_e5m2_fp32(arm_neonfp8dot4_fp8_fp8_fp32):
  def __init__(self):
    super().__init__("neonfp8dot4", "fp8_e5m2", (1, 4, 4))


class arm64_neonfp8dot4_fp8_e4m3_fp8_e4m3_fp32(arm_neonfp8dot4_fp8_fp8_fp32):
  def __init__(self):
    super().__init__("neonfp8dot4", "fp8_e4m3", (1, 4, 4))


class arm64_neonf8mm8_fp8_fp8_fp32(arm_fp8_fp8_fp32):

  def __init__(self, arch, fp8_type, tile_shape):
    super().__init__(arch, fp8_type, tile_shape)
    self.flags += ["dot_flag::transpose_a"]

  # This is one of the trickier generators. mmla is a 2x8*8x2 matrix multiply.
  # We handle it in 2x4 tiles, with 2 accumulator registers per tile.

  def load_a_tile(self, i, k):
    return (
        f"const mfloat8x16_t a_{i}_{k} ="
        f" vreinterpretq_mf8_u8(vld1q_u8({self.a_ptr(i, k)}));\n"
    )

  def load_b_tile(self, k, j):
    return f"""
mfloat8x16_t b_{k}_{j+0} = vreinterpretq_mf8_u8(vld1q_u8({self.b_ptr(k, j+0)}));
mfloat8x16_t b_{k}_{j+2} = vreinterpretq_mf8_u8(vld1q_u8({self.b_ptr(k, j+2)}));
"""

  def init_c_tile(self, i, j):
    return f"""
float32x4_t c_{i}_{j+0} = vdupq_n_f32(0);
float32x4_t c_{i}_{j+2} = vdupq_n_f32(0);
"""

  def product(self, i, j, k):
    return f"""
c_{i}_{j+0} = vmmlaq_f32_mf8_fpm(c_{i}_{j+0}, a_{i}_{k}, b_{k}_{j+0}, fpm);
c_{i}_{j+2} = vmmlaq_f32_mf8_fpm(c_{i}_{j+2}, a_{i}_{k}, b_{k}_{j+2}, fpm);
  """

  def add_c_tile(self, i, j):
    return f"""
c_{i+0}_{j} = vaddq_f32(c_{i+0}_{j}, vld1q_f32({self.c_in_ptr(i+0, j)}));
c_{i+1}_{j} = vaddq_f32(c_{i+1}_{j}, vld1q_f32({self.c_in_ptr(i+1, j)}));
"""

  def store_c_tile(self, i, j):
    # Write row 1 first, in case i is clamped by M.
    return f"""
vst1q_f32({self.c_out_ptr(i+1, j)}, c_{i+1}_{j});
vst1q_f32({self.c_out_ptr(i+0, j)}, c_{i+0}_{j});
"""

  def finalize_c_tile(self, i, j):
    c0 = f"c_{i}_{j+0}"
    c2 = f"c_{i}_{j+2}"
    return f"""
float32x4_t c_{i+1}_{j} = vcombine_f32(vget_high_f32({c0}), vget_high_f32({c2}));
c_{i+0}_{j} = vcombine_f32(vget_low_f32({c0}), vget_low_f32({c2}));
"""


class arm64_neonf8mm8_fp8_e5m2_fp8_e5m2_fp32(arm64_neonf8mm8_fp8_fp8_fp32):

  def __init__(self):
    super().__init__("neonf8mm8", "fp8_e5m2", (2, 4, 8))


class arm64_neonf8mm8_fp8_e4m3_fp8_e4m3_fp32(arm64_neonf8mm8_fp8_fp8_fp32):

  def __init__(self):
    super().__init__("neonf8mm8", "fp8_e4m3", (2, 4, 8))


# Generate kernels
fp8dot4_shapes = [
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

generate_dot_kernels(arm64_neonfp8dot4_fp8_e5m2_fp8_e5m2_fp32(), fp8dot4_shapes)
generate_dot_kernels(arm64_neonfp8dot4_fp8_e4m3_fp8_e4m3_fp32(), fp8dot4_shapes)

f8mm8_shapes = [
    (2, 32, 8),
    (4, 16, 8),
    (6, 16, 8),
    (6, 8, 8),
    (8, 8, 8),
    (10, 8, 8),
    (16, 4, 8),
]

generate_dot_kernels(arm64_neonf8mm8_fp8_e5m2_fp8_e5m2_fp32(), f8mm8_shapes)
generate_dot_kernels(arm64_neonf8mm8_fp8_e4m3_fp8_e4m3_fp32(), f8mm8_shapes)
