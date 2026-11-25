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
    # Signed 4-bit nibbles are unpacked as (nibble << 4) in [-128, 112], scaling
    # the int32 dot accumulation by 16x before shifting right by 4 in
    # finalize_c_tile. Each term is at most |-128 * -128| = 16384, so chunks of
    # up to 65536 elements are guaranteed never to overflow int32_t.
    self.max_k = 65536

  def header(self):
    return super().header() + """
namespace {

YNN_INTRINSIC void prefetch_l1(const void* ptr) {
#if defined(YNN_ARCH_ARM64) && !defined(__ANDROID__) && !defined(__APPLE__) && defined(__GNUC__)
  __builtin_prefetch(ptr, 0, 3);
#else
  (void)ptr;
#endif
}

YNN_INTRINSIC int8x16_t shuffle_a_k16(int8x16_t a, uint8x16_t table) {
#ifdef YNN_ARCH_ARM64
  return vqtbl1q_s8(a, table);
#else
  const int8x8_t tbl = vreinterpret_s8_u8(vget_low_u8(table));
  return vcombine_s8(vtbl1_s8(vget_low_s8(a), tbl),
                     vtbl1_s8(vget_high_s8(a), tbl));
#endif
}

YNN_INTRINSIC int8x8_t shuffle_a_k8(int8x8_t a, uint8x16_t table) {
#ifdef YNN_ARCH_ARM64
  return vqtbl1_s8(vcombine_s8(a, vdup_n_s8(0)), vget_low_u8(table));
#else
  return vtbl1_s8(a, vreinterpret_s8_u8(vget_low_u8(table)));
#endif
}

template <int Lane>
YNN_INTRINSIC int32x4_t dot_laneq_s32(int32x4_t c, int8x16_t b, int8x16_t a) {
#ifdef YNN_ARCH_ARM64
  return vdotq_laneq_s32(c, b, a, Lane);
#else
  if constexpr (Lane < 2) {
    return vdotq_lane_s32(c, b, vget_low_s8(a), Lane);
  } else {
    return vdotq_lane_s32(c, b, vget_high_s8(a), Lane - 2);
  }
#endif
}

}  // namespace
"""

  # We can permute B, or we can permute A. If we are loading a lot of tiles of
  # A, it's better to permute B, or vice versa.
  def shuffle_a(self):
    block_m, _, _ = self.block_shape
    return block_m < 8

  def use_paired_b_load(self):
    block_m, block_n, _ = self.block_shape
    # Pair two 16-byte B loads (j and j+4) when B is contiguous across N tiles
    # and enough NEON registers remain to avoid spills (3x32 uses 24 c + 3 a).
    c_regs = block_m * (block_n // self.tile_shape[1])
    return (
        self.shuffle_a()
        and self.b_chunk_n >= 8
        and block_n >= 8
        and c_regs + block_m <= 26
    )

  def begin_func(self, func_name):
    result = super().begin_func(func_name)
    result += (
        "  const int8x16_t b_mask = vdupq_n_s8(static_cast<int8_t>(0xf0));\n"
    )
    if self.shuffle_a():
      result += (
          "  const uint8_t a_table_data[] ="
          " {0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15};\n"
          "  const uint8x16_t a_table = vld1q_u8(a_table_data);\n"
      )
    return result

  def finalize_c_tile(self, i, j):
    return f"c_{i}_{j} = vshrq_n_s32(c_{i}_{j}, 4);\n"

  def _b_k_ptr(self, k, j, ty="int8_t"):
    jo = (j // self.b_chunk_n) * self.b_chunk_n
    ji = j - jo
    base = f"B_k1_{jo}" if k == 0 else f"B_k_{k}_{jo}"
    return (
        f"reinterpret_cast<const {ty}*>(offset_bytes({base},"
        f" {ji} * {self.b_tile_stride_n()}))"
    )

  def _load_a(self, i, k16):
    if k16:
      val = f"vld1q_s8({self.a_ptr(i, 0)})"
      if self.shuffle_a():
        val = f"shuffle_a_k16({val}, a_table)"
      return f"int8x16_t a_{i}_0 = {val};\n"
    val = f"vld1_s8({self.a_ptr(i, 0)})"
    if self.shuffle_a():
      val = f"shuffle_a_k8({val}, a_table)"
    return f"int8x8_t a_{i}_0 = {val};\n"

  def _emit_b_group(self, k, js, k16):
    block_m, _, _ = self.block_shape
    tile_m = self.tile_shape[0]
    result = ""
    for j in js:
      result += f"int8x16_t b_{k}_{j}_raw = vld1q_s8({self._b_k_ptr(k, j)});\n"
    for j in js:
      if self.shuffle_a():
        result += f"""int8x16_t b_{k+1}_{j} = vandq_s8(b_{k}_{j}_raw, b_mask);
int8x16_t b_{k+0}_{j} = vshlq_n_s8(b_{k}_{j}_raw, 4);
"""
      else:
        result += f"""int8x16_t b_{k}_{j}_o = vandq_s8(b_{k}_{j}_raw, b_mask);
int8x16_t b_{k}_{j}_e = vshlq_n_s8(b_{k}_{j}_raw, 4);
int8x16x2_t zipped_{k}_{j} = vzipq_s8(b_{k}_{j}_e, b_{k}_{j}_o);
int32x4x2_t unzipped_{k}_{j} = vuzpq_s32(vreinterpretq_s32_s8(zipped_{k}_{j}.val[0]), vreinterpretq_s32_s8(zipped_{k}_{j}.val[1]));
int8x16_t b_{k+0}_{j} = vreinterpretq_s8_s32(unzipped_{k}_{j}.val[0]);
int8x16_t b_{k+1}_{j} = vreinterpretq_s8_s32(unzipped_{k}_{j}.val[1]);
"""
    lane0 = (k // 4) if k16 else 0
    for i in range(0, block_m, tile_m):
      for step in (0, 1):
        for j in js:
          b_reg = f"b_{k + step}_{j}"
          if k16:
            result += (
                f"c_{i}_{j} = dot_laneq_s32<{lane0 + step}>(c_{i}_{j},"
                f" {b_reg}, a_{i}_0);\n"
            )
          else:
            result += (
                f"c_{i}_{j} = vdotq_lane_s32(c_{i}_{j}, {b_reg}, a_{i}_0,"
                f" {step});\n"
            )
    return result

  def generate_block(self, a_tail=False, b_tail=False):
    block_m, block_n, block_k = self.block_shape
    tile_m, tile_n, tile_k = self.tile_shape
    k16 = block_k == 16
    assert block_k in (8, 16)

    result = "".join(
        self._load_a(i, k16) for i in range(0, block_m, tile_m)
    ) + "\n"

    paired = self.use_paired_b_load()
    for k in range(0, block_k, tile_k):
      if k > 0:
        for jo in range(0, block_n, self.b_chunk_n):
          result += (
              f"const void* B_k_{k}_{jo} = offset_bytes(B_k1_{jo}, {k} *"
              " B_stride_k1);\n"
          )
      if paired and block_m == 1 and block_n >= 32 and k16:
        b_base = "B_k1_0" if k == 0 else f"B_k_{k}_0"
        b_slice_bytes = block_n * tile_k // 2
        result += (
            f"const void* B_pf_{k} = offset_bytes({b_base}, 128 *"
            " B_stride_k1);\n"
        )
        for off in range(0, b_slice_bytes, 64):
          ptr = f"B_pf_{k}" if off == 0 else f"offset_bytes(B_pf_{k}, {off})"
          result += f"prefetch_l1({ptr});\n"
      if paired:
        # Pair contiguous B loads on ARM64 (32 vector regs); keep ARM32 at 1
        # tile per step because ARM32 NEON has only 16 128-bit registers.
        result += "#ifdef YNN_ARCH_ARM64\n"
        for j in range(0, block_n, tile_n * 2):
          result += self._emit_b_group(k, (j, j + tile_n), k16)
        result += "#else\n"
        for j in range(0, block_n, tile_n):
          result += self._emit_b_group(k, (j,), k16)
        result += "#endif\n"
      else:
        for j in range(0, block_n, tile_n):
          result += self._emit_b_group(k, (j,), k16)
    return result + "\n"


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
        (1, 32, 16),
        (2, 32, 16),
        (3, 32, 16),
        (1, 16, 16),
        (2, 16, 16),
        (3, 16, 16),
        (4, 16, 16),
        (5, 16, 16),
        (1, 8, 16),
        (8, 8, 16),
        (8, 4, 16),
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
