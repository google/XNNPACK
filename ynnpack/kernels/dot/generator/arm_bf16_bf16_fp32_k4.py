# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for bf16 arm dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.arm import arm_neon


class arm_neonbf16_bf16_bf16_fp32_k4(arm_neon):
  def __init__(self):
    super().__init__("neonbf16", "bf16_bf16_fp32", "float", (2, 4, 4))
    self.a_type = "bfloat16"
    self.b_type = "bfloat16"
    self.flags += ["dot_flag::transpose_a"]

  def header(self):
    return "#include <tuple>\n" + super().header() + """

namespace {

using bfloat16 = bfloat16_t;

YNN_INTRINSIC std::tuple<float32x4_t, float32x4_t> transpose2x2_x64(float32x4_t x0, float32x4_t x1) {
  return {vcombine_f32(vget_low_f32(x0), vget_low_f32(x1)),
          vcombine_f32(vget_high_f32(x0), vget_high_f32(x1))};
}

}  // namespace
"""

  # This is one of the trickier generators. mmla is a 2x4*4x2 matrix multiply.
  # We handle it in 2x4 tiles, with 2 accumulator registers per tile.

  def load_a_tile(self, i, k):
    return f"""
const bfloat16x8_t a_{i}_{k} = vld1q_bf16({self.a_ptr(i, k)});
"""

  def load_b_tile(self, k, j):
    return f"""
bfloat16x8_t b_{k}_{j+0} = vld1q_bf16({self.b_ptr(k, j+0)});
bfloat16x8_t b_{k}_{j+2} = vld1q_bf16({self.b_ptr(k, j+2)});
"""

  def init_c_tile(self, i, j):
    return f"""
float32x4_t c_{i}_{j+0} = vdupq_n_f32(0);
float32x4_t c_{i}_{j+2} = vdupq_n_f32(0);
"""

  def product(self, i, j, k):
    return f"""
c_{i}_{j+0} = vbfmmlaq_f32(c_{i}_{j+0}, a_{i}_{k}, b_{k}_{j+0});
c_{i}_{j+2} = vbfmmlaq_f32(c_{i}_{j+2}, a_{i}_{k}, b_{k}_{j+2});
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
    c1 = f"c_{i}_{j+2}"
    return f"""
float32x4_t c_{i+1}_{j} = vcombine_f32(vget_high_f32({c0}), vget_high_f32({c1}));
c_{i+0}_{j} = vcombine_f32(vget_low_f32({c0}), vget_low_f32({c1}));
"""
