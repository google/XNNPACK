# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for int8 x int4 arm dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.arm import arm_neon
from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels


class arm_int8_int4_int32(arm_neon):

  def __init__(self, arch, tile_shape):
    super().__init__(arch, "int8_int4_int32", "int32_t", tile_shape)
    self.a_type = "int8_t"
    self.b_type = "int4x2"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def b_tile_size_bytes(self):
    return f"{self.tile_shape[1] * self.tile_shape[2] // 2}"


class arm_neondot_int8_int4_int32(arm_int8_int4_int32):

  def __init__(self):
    super().__init__("neondot", (1, 4, 8))

  # We can permute B, or we can permute A. If we are loading a lot of tiles of
  # A, it's better to permute B, or vice versa.
  def shuffle_a(self):
    block_m, _, _ = self.block_shape
    return block_m < 8

  def begin_func(self, func_name):
    result = super().begin_func(func_name)
    if self.shuffle_a():
      result += "  const int8_t a_table[] = {0, 2, 4, 6, 1, 3, 5, 7};\n"
    return result

  def load_a_tile(self, i, k):
    result = f"int8x8_t a_{i}_{k} = vld1_s8({self.a_ptr(i, k)});\n"
    if self.shuffle_a():
      result += f"a_{i}_{k} = vtbl1_s8(a_{i}_{k}, vld1_s8(a_table));\n"
    return result

  def load_b_tile(self, k, j):
    if self.shuffle_a():
      return f"""
int8x16_t b_{k}_{j} = vld1q_s8({self.b_ptr(k, j, "int8_t")});
int8x16_t b_{k+1}_{j} = vshrq_n_s8(b_{k}_{j}, 4);
b_{k+0}_{j} = vshrq_n_s8(vshlq_n_s8(b_{k}_{j}, 4), 4);
"""
    else:
      return f"""
int8x16_t b_{k}_{j}_eo = vld1q_s8({self.b_ptr(k, j, "int8_t")});
int8x16_t b_{k}_{j}_o = vshrq_n_s8(b_{k}_{j}_eo, 4);
int8x16_t b_{k}_{j}_e = vshrq_n_s8(vshlq_n_s8(b_{k}_{j}_eo, 4), 4);
int8x16x2_t zipped_{k}_{j} = vzipq_s8(b_{k}_{j}_e, b_{k}_{j}_o);
int32x4x2_t unzipped_{k}_{j} = vuzpq_s32(vreinterpretq_s32_s8(zipped_{k}_{j}.val[0]), vreinterpretq_s32_s8(zipped_{k}_{j}.val[1]));
int8x16_t b_{k+0}_{j} = vreinterpretq_s8_s32(unzipped_{k}_{j}.val[0]);
int8x16_t b_{k+4}_{j} = vreinterpretq_s8_s32(unzipped_{k}_{j}.val[1]);
"""

  def product(self, i, j, k):
    c_ij = f"c_{i}_{j}"
    a = f"a_{i}_{k}"
    if self.shuffle_a():
      return f"""
{c_ij} = vdotq_lane_s32({c_ij}, b_{k+0}_{j}, {a}, 0);
{c_ij} = vdotq_lane_s32({c_ij}, b_{k+1}_{j}, {a}, 1);
"""
    else:
      return f"""
{c_ij} = vdotq_lane_s32({c_ij}, b_{k+0}_{j}, {a}, 0);
{c_ij} = vdotq_lane_s32({c_ij}, b_{k+4}_{j}, {a}, 1);
"""


class arm64_neoni8mm_int8_int4_int32(arm_int8_int4_int32):

  def __init__(self):
    super().__init__("neoni8mm", (2, 4, 8))
    self.flags += ["dot_flag::transpose_a"]

  def load_a_tile(self, i, k):
    return f"const int8x16_t a_{i}_{k} = vld1q_s8({self.a_ptr(i, k)});\n"

  def load_b_tile(self, k, j):
    return f"""
int8x16_t b_{k}_{j}_eo = vld1q_s8({self.b_ptr(k, j, "int8_t")});
int8x16_t b_{k}_{j}_o = vshrq_n_s8(b_{k}_{j}_eo, 4);
int8x16_t b_{k}_{j}_e = vshrq_n_s8(vshlq_n_s8(b_{k}_{j}_eo, 4), 4);
int8x16_t b_{k}_{j+0} = vzip1q_s8(b_{k}_{j}_e, b_{k}_{j}_o);
int8x16_t b_{k}_{j+2} = vzip2q_s8(b_{k}_{j}_e, b_{k}_{j}_o);
"""

  def init_c_tile(self, i, j):
    return f"""
int32x4_t c_{i}_{j+0} = vdupq_n_s32(0);
int32x4_t c_{i}_{j+2} = vdupq_n_s32(0);
"""

  def product(self, i, j, k):
    return f"""
c_{i}_{j+0} = vmmlaq_s32(c_{i}_{j+0}, a_{i}_{k}, b_{k}_{j+0});
c_{i}_{j+2} = vmmlaq_s32(c_{i}_{j+2}, a_{i}_{k}, b_{k}_{j+2});
"""

  def add_c_tile(self, i, j):
    return f"""
c_{i+0}_{j} = vaddq_s32(c_{i+0}_{j}, vld1q_s32({self.c_in_ptr(i+0, j)}));
c_{i+1}_{j} = vaddq_s32(c_{i+1}_{j}, vld1q_s32({self.c_in_ptr(i+1, j)}));
"""

  def store_c_tile(self, i, j):
    return f"""
vst1q_s32({self.c_out_ptr(i+1, j)}, c_{i+1}_{j});
vst1q_s32({self.c_out_ptr(i+0, j)}, c_{i+0}_{j});
"""

  def finalize_c_tile(self, i, j):
    c0 = f"c_{i}_{j+0}"
    c2 = f"c_{i}_{j+2}"
    return f"""
int32x4_t c_{i+1}_{j} = vcombine_s32(vget_high_s32({c0}), vget_high_s32({c2}));
c_{i+0}_{j} = vcombine_s32(vget_low_s32({c0}), vget_low_s32({c2}));
"""


generate_dot_kernels(
    arm_neondot_int8_int4_int32(),
    [
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
        (8, 4, 8),
    ],
)

generate_dot_kernels(
    arm64_neoni8mm_int8_int4_int32(),
    [
        (2, 32, 8),
        (4, 16, 8),
        (6, 16, 8),
        (6, 8, 8),
        (8, 8, 8),
        (10, 8, 8),
        (16, 4, 8),
    ],
)
