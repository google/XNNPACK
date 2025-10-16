"""Specializations for int8 arm dot kernel generators."""

from ynnpack.kernels.dot.generator.arm import arm_neon


class arm_int8_int8_int32(arm_neon):
  def __init__(self, arch, tile_shape):
    super().__init__(arch, "int8_int8_int32", "int32_t", tile_shape)
    self.a_type = "int8_t"
    self.b_type = "int8_t"


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
    super().__init__("neoni8mm", (1, 4, 8))

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
  # We handle it in 2x8 tiles, with 4 accumulator registers per tile. However,
  # the tile shape is 1x4, to get the base class's handling of adding/storing c,
  # the tile handlers need to ignore the unaligned invocations, and handle the
  # whole 2x8 tile.

  def load_a_tile_k_tail(self, i, k, nk):
    if i % 2 != 0: return ""
    if k % nk != 0:
      return ""
    if nk == 16:
      return f"""
uint64x2x2_t a_{i}_{k}_u64;
a_{i}_{k}_u64 = vld2q_lane_u64({self.a_ptr(i+0, k, "uint64_t")}, a_{i}_{k}_u64, 0);
a_{i}_{k}_u64 = vld2q_lane_u64({self.a_ptr(i+1, k, "uint64_t")}, a_{i}_{k}_u64, 1);
const int8x16_t a_{i}_{k+0} = vreinterpretq_s8_u64(a_{i}_{k}_u64.val[0]);
const int8x16_t a_{i}_{k+8} = vreinterpretq_s8_u64(a_{i}_{k}_u64.val[1]);
"""
    else:
      assert(nk <= 8)
      return f"""
uint64x2_t a_{i}_{k}_u64 = vld1q_dup_u64({self.a_ptr(i+0, k, "uint64_t")});
const int8x16_t a_{i}_{k} = vreinterpretq_s8_u64(vld1q_lane_u64({self.a_ptr(i+1, k, "uint64_t")}, a_{i}_{k}_u64, 1));
"""

  def load_b_tile(self, k, j):
    if j % 4 != 0: return ""
    return f"""
int8x16_t b_{k}_{j+0} = vld1q_s8({self.b_ptr(k, j+0)});
int8x16_t b_{k}_{j+2} = vld1q_s8({self.b_ptr(k, j+2)});
"""

  def init_c_tile(self, i, j):
    if i % 2 != 0 or j % 4 != 0: return ""
    return f"""
int32x4_t c_{i}_{j+0} = vdupq_n_s32(0);
int32x4_t c_{i}_{j+2} = vdupq_n_s32(0);
"""

  def product(self, i, j, k):
    if i % 2 != 0 or j % 4 != 0: return ""
    return f"""
c_{i}_{j+0} = vmmlaq_s32(c_{i}_{j+0}, a_{i}_{k}, b_{k}_{j+0});
c_{i}_{j+2} = vmmlaq_s32(c_{i}_{j+2}, a_{i}_{k}, b_{k}_{j+2});
  """

  def finalize_c_tile(self, i, j):
    if i % 2 != 0 or j % 4 != 0: return ""
    return f"""
int32x4_t c_{i+1}_{j+0};
std::tie(c_{i}_{j+0}, c_{i+1}_{j+0}) = transpose2x2_x64(c_{i}_{j+0}, c_{i}_{j+2});
"""
