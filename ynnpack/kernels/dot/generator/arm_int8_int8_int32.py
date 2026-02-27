# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for int8 arm dot kernel generators."""

from ynnpack.kernels.dot.generator.arm import arm_neon


class arm_int8_int8_int32(arm_neon):
  def __init__(self, arch, tile_shape):
    super().__init__(arch, "int8_int8_int32", "int32_t", tile_shape)
    self.a_type = "int8_t"
    self.b_type = "int8_t"
    self.flags += ["dot_flag::consistent_arithmetic"]


class arm_neon_int8_int8_int32(arm_int8_int8_int32):
  def __init__(self, arch="neon", tile_shape=(1, 4, 1)):
    super().__init__(arch, tile_shape)

  def load_a_tile_k_tail(self, i, k, nk):
    if k % nk != 0:
      return ""
    if nk == 8:
      return f"int16x8_t a_{i}_{k} = vmovl_s8(vld1_s8({self.a_ptr(i, k)}));\n"
    else:
      assert(nk == 1)
      return f"int16x8_t a_{i}_{k} = vdupq_n_s16(*{self.a_ptr(i, k)});\n"

  def load_b_tile(self, k, j):
    if self.b_chunk_n < 8 or j + 8 > self.block_shape[1]:
      # We can only load 4 values.
      int8 = f"vreinterpret_s8_s32(vdup_n_s32(*{self.b_ptr(k, j, 'int32_t')}))"
      return f"int16x8_t b_{k}_{j} = vmovl_s8({int8});\n"
    elif j % 8 == 0:
      # The next 8 values are all in bounds of the block.
      return f"int16x8_t b_{k}_{j} = vmovl_s8(vld1_s8({self.b_ptr(k, j)}));\n"
    else:
      return ""

  def product(self, i, j, k):
    if self.b_chunk_n < 8 or j + 8 > self.block_shape[1]:
      a = f"vget_{'low' if k % 8 < 4 else 'high'}_s16(a_{i}_{(k//8)*8})"
      b = f"vget_low_s16(b_{k}_{(j//4)*4})"
      return f"c_{i}_{j} = vmlal_lane_s16(c_{i}_{j}, {b}, {a}, {k%4});\n"
    else:
      a = f"vget_{'low' if k % 8 < 4 else 'high'}_s16(a_{i}_{(k//8)*8})"
      b = f"vget_{'low' if j % 8 < 4 else 'high'}_s16(b_{k}_{(j//8)*8})"
      return f"c_{i}_{j} = vmlal_lane_s16(c_{i}_{j}, {b}, {a}, {k%4});\n"


class arm_neondot_int8_int8_int32(arm_int8_int8_int32):
  def __init__(self):
    super().__init__("neondot", (1, 4, 4))

  def header(self):
    return super().header() + """

namespace {

YNN_INTRINSIC int32_t unaligned_load_int8x4(const void* ptr) {
    int32_t value;
    memcpy(&value, ptr, sizeof(int32_t));
    return value;
}

}  // namespace
"""

  def load_a_tile_k_tail(self, i, k, nk):
    if k % 8 != 0:
      return ""
    if nk == 8:
      return f"int8x8_t a_{i}_{k} = vld1_s8({self.a_ptr(i, k)});\n"
    else:
      return f"int8x8_t a_{i}_{k} = vreinterpret_s8_s32(vdup_n_s32(unaligned_load_int8x4({self.a_ptr(i, k)})));\n"

  def load_b_tile(self, k, j):
    return f"int8x16_t b_{k}_{j} = vld1q_s8({self.b_ptr(k, j)});\n"

  def product(self, i, j, k):
    # TODO: arm64 has vdtoq_laneq_s32, so we can unroll by 16 instead of 8.
    return (
        f"c_{i}_{j} = vdotq_lane_s32(c_{i}_{j}, b_{k}_{j}, a_{i}_{(k//8)*8},"
        f" {(k//4)%2});\n"
    )


class arm_neoni8mm_int8_int8_int32(arm_int8_int8_int32):
  def __init__(self):
    super().__init__("neoni8mm", (2, 4, 8))
    self.flags += ["dot_flag::transpose_a"]

  def header(self):
    return "#include <tuple>\n" + super().header() + """

namespace {

YNN_INTRINSIC std::tuple<int32x4_t, int32x4_t> transpose2x2_x64(int32x4_t x0, int32x4_t x1) {
  return {vcombine_s32(vget_low_s32(x0), vget_low_s32(x1)),
          vcombine_s32(vget_high_s32(x0), vget_high_s32(x1))};
}

}  // namespace
"""

  # This is one of the trickier generators. mmla is a 2x8*8x2 matrix multiply.
  # We handle it in 2x4 tiles, with 2 accumulator registers per tile.

  def load_a_tile(self, i, k):
    return f"""
const int8x16_t a_{i}_{k} = vld1q_s8({self.a_ptr(i, k)});
"""

  def load_b_tile(self, k, j):
    return f"""
int8x16_t b_{k}_{j+0} = vld1q_s8({self.b_ptr(k, j+0)});
int8x16_t b_{k}_{j+2} = vld1q_s8({self.b_ptr(k, j+2)});
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
    # Write row 1 first, in case i is clamped by M.
    return f"""
vst1q_s32({self.c_out_ptr(i+1, j)}, c_{i+1}_{j});
vst1q_s32({self.c_out_ptr(i+0, j)}, c_{i+0}_{j});
"""

  def finalize_c_tile(self, i, j):
    return f"""
int32x4_t c_{i+1}_{j+0};
std::tie(c_{i}_{j+0}, c_{i+1}_{j+0}) = transpose2x2_x64(c_{i}_{j+0}, c_{i}_{j+2});
"""
