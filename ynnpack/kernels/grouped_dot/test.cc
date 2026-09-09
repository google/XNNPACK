// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/kernels/dot/dot.h"
#include "ynnpack/kernels/dot/pack.h"
#include "ynnpack/kernels/grouped_dot/grouped_dot.h"

namespace ynn {
namespace {

// Reference implementation of grouped_dot.
void ReferenceGroupedDot(
    size_t E, const int32_t* expert_counts, const int32_t* offsets,
    const float* input_a,
    const float* input_b,  // Unpacked weights [E, D_in, D_out]
    float* output, size_t D_in, size_t D_out) {
  for (size_t e = 0; e < E; ++e) {
    int32_t count = expert_counts[e];
    if (count == 0) continue;

    int32_t offset = offsets[e];
    const float* A_e = input_a + offset * D_in;
    const float* B_e = input_b + e * D_in * D_out;
    float* C_e = output + offset * D_out;

    for (int i = 0; i < count; ++i) {
      for (size_t d_out = 0; d_out < D_out; ++d_out) {
        float sum = 0.0f;
        for (size_t d_in = 0; d_in < D_in; ++d_in) {
          // B_e is shape [D_in, D_out]
          sum += A_e[i * D_in + d_in] * B_e[d_in * D_out + d_out];
        }
        C_e[i * D_out + d_out] = sum;
      }
    }
  }
}

// Helper to align value.
size_t align_up(size_t v, size_t alignment) {
  return (v + alignment - 1) / alignment * alignment;
}

TEST(GroupedDotTest, Basic) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  size_t E = 8;
  size_t N = 16;
  size_t K = 4;  // top-k
  size_t NK = N * K;
  size_t D_in = 64;
  size_t D_out = 128;

  // 1. Generate routing.
  std::vector<int32_t> expert_indices(NK);
  std::uniform_int_distribution<int32_t> exp_dist(0, E - 1);
  for (size_t i = 0; i < NK; ++i) {
    expert_indices[i] = exp_dist(rng);
  }

  // Compute counts and offsets.
  std::vector<int32_t> expert_counts(E, 0);
  for (size_t i = 0; i < NK; ++i) {
    expert_counts[expert_indices[i]]++;
  }
  std::vector<int32_t> offsets(E + 1, 0);
  for (size_t e = 0; e < E; ++e) {
    offsets[e + 1] = offsets[e] + expert_counts[e];
  }

  // 2. Generate inputs.
  std::vector<float> input_a(NK * D_in);
  for (auto& v : input_a) v = dist(rng);

  std::vector<float> input_b(E * D_in * D_out);
  for (auto& v : input_b) v = dist(rng);

  // 3. Get dot kernel.
  dot_type type{ynn_type_fp32, ynn_type_fp32, ynn_type_fp32};
  // We don't restrict shape, let get_dot_kernel choose.
  dot_kernel kernel = get_dot_kernel(type);
  ASSERT_NE(kernel.kernel, nullptr);

  size_t tile_k = kernel.tile_k;
  size_t tile_n = kernel.tile_n;

  // 4. Pack weights.
  // packed_b shape per expert: [ceil_div(D_in, tile_k), align_up(D_out,
  // tile_n), tile_k]
  size_t num_k_blocks = (D_in + tile_k - 1) / tile_k;
  size_t aligned_n = align_up(D_out, tile_n);
  size_t packed_b_stride = aligned_n * tile_k * sizeof(float);
  size_t expert_packed_size = num_k_blocks * packed_b_stride;

  std::vector<char> packed_b(E * expert_packed_size, 0);

  packer p(/*transpose=*/false, /*elem_size_bits=*/32, tile_k, aligned_n);

  for (size_t e = 0; e < E; ++e) {
    const float* B_e = input_b.data() + e * D_in * D_out;
    char* packed_B_e = packed_b.data() + e * expert_packed_size;

    // pack expects input shape [m, n] which is [D_in, D_out]
    // input_stride is stride of m dimension (which is D_out elements = D_out *
    // sizeof(float) bytes). output_stride is stride of output first dimension
    // (packed_b_stride).
    p.pack(D_in, D_out, D_out * sizeof(float), B_e, packed_b_stride, 0,
           packed_B_e);
  }

  // 5. Run reference.
  std::vector<float> ref_output(NK * D_out, 0.0f);
  ReferenceGroupedDot(E, expert_counts.data(), offsets.data(), input_a.data(),
                      input_b.data(), ref_output.data(), D_in, D_out);

  // 6. Run kernel.
  std::vector<float> ynn_output(NK * D_out, 0.0f);
  grouped_dot(E, expert_counts.data(), offsets.data(), input_a.data(),
              packed_b.data(), ynn_output.data(), D_in, D_out, packed_b_stride,
              kernel);

  // 7. Compare.
  for (size_t i = 0; i < NK * D_out; ++i) {
    EXPECT_NEAR(ynn_output[i], ref_output[i], 1e-4f) << "At index " << i;
  }
}

}  // namespace
}  // namespace ynn
