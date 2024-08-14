#include "gemm-microkernel-tester.h"

#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <fp16/bitcasts.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/config-types.h"
#include "xnnpack/gemm.h"
#include "xnnpack/math.h"
#include "xnnpack/memory.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/pack.h"
#include "xnnpack/packq.h"
#include "xnnpack/quantization.h"
#include "xnnpack/requantization.h"
#include "replicable_random_device.h"

TEST_P(GemmTest, Test) {
  const GemmTestParams& params = GetParam();
  GemmMicrokernelTester tester = params.tester;

  // Make sure that we can execute this test.
  if (params.isa_check) {
    params.isa_check();
    if (IsSkipped()) {
      return;
    }
  }

  // Loop over the `k`, `m`, and `n` values, if required.
  for (size_t k = params.loop_k_.from; k <= params.loop_k_.to;
       k = params.loop_k_.next(k)) {
    if (params.loop_k_.is_set) {
      tester.k(k);
    }
    for (size_t m = params.loop_m_.from; m <= params.loop_m_.to;
         m = params.loop_m_.next(m)) {
      if (params.loop_m_.is_set) {
        tester.m(m);
      }
      for (size_t n = params.loop_n_.from; n <= params.loop_n_.to;
           n = params.loop_n_.next(n)) {
        if (params.loop_n_.is_set) {
          tester.n(n);
        }
        for (size_t zi = params.loop_zi_.from; zi <= params.loop_zi_.to;
             zi = params.loop_zi_.next(zi)) {
          if (params.loop_zi_.is_set) {
            tester.zero_index(zi);
          }
          for (size_t bzp = params.loop_bzp_.from; bzp <= params.loop_bzp_.to;
               bzp = params.loop_bzp_.next(bzp)) {
            if (params.loop_bzp_.is_set) {
              tester.b_zero_point(bzp);
            }
            for (size_t bl = params.loop_bl_.from; bl <= tester.k() / 2;
               bl = params.loop_bl_.next(bl)) {
               if (params.loop_bl_.is_set) {
                // Require block size to divide (padded) column size.
                if (round_up_po2(k, params.loop_bl_.step) % bl != 0) {
                  continue;
                }
                tester.bl(bl);
              }

              // Call the test function.
              params.test_func(tester);
            }
          }
        }
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_qd8_f16_qc8w_igemm_ukernel_fn igemm,
  xnn_init_f16_minmax_params_fn init_params,
  xnn_pack_qs8_igemm_fn pack) const
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f), std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f), std::ref(rng));
  auto w8rng = std::bind(
      std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()),
      std::ref(rng));

  std::vector<float> input(mr() * k());
  std::vector<int8_t> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<xnn_qd8_quantization_params> quantization_params(1 + XNN_EXTRA_QUANTIZATION_PARAMS);
  std::vector<int8_t> b(n() * ks() * k());
  std::vector<float> bias(n());
  std::vector<float> kernel_scale(n());
  std::vector<int8_t, AlignedAllocator<int8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(ks() * packed_n() * packed_k() + packed_n() * (sizeof(int32_t) + sizeof(float) * 2));
  std::vector<uint16_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n(), 0);
  std::vector<int8_t> junk(k() + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<const int8_t*> im2col(mr() * ks());

  std::fill(junk.begin(), junk.end(), 0xA5);
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    const auto minmax = std::minmax_element(input.begin(), input.begin() + mr() * k());
    quantization_params[0] = xnn_f32_qd8_asymmetric_quantization_params(*minmax.first, *minmax.second);
    const float inv_scale = 1.f / quantization_params[0].inv_scale;
    for (size_t i = 0; i < mr(); ++i) {
      const float* input_ptr = &input[i * k()];
      for (size_t j = 0; j < k(); ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(scaled_input, float(std::numeric_limits<int8_t>::max()
                                                           - quantization_params[0].zero_point));
        scaled_input = std::max<float>(scaled_input, float(std::numeric_limits<int8_t>::min()
                                                           - quantization_params[0].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) + long(quantization_params[0].zero_point));
      }
    }
    std::generate(b.begin(), b.end(), std::ref(w8rng));

    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));
    std::fill(c.begin(), c.end(), UINT16_C(0x7E00));

    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_packing_params packing_params = { /*input_zero_point=*/1 };
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), /*bias=*/nullptr, /*scale=*/nullptr, packed_w.data(), 2 * sizeof(float) * nr(), &packing_params);
    // Fill in packed kernel scale
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      0,
      kernel_scale.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(float))));

    // Fill in packed bias
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      0,
      bias.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k() * sizeof(int8_t) + 2 * sizeof(float))));

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = a.data() + a_stride() * m_index - a_offset();
      }
    }
    std::shuffle(im2col.begin(), im2col.end(), rng);
    const size_t k_stride =  round_up_po2(k(), kr() * sr());
    std::vector<int8_t> zero_points(k_stride + XNN_EXTRA_BYTES, quantization_params[0].zero_point);
    const int8_t* zero_sentinel = (const int8_t*) &packing_params;
    const int8_t* zero_data = zero_points.data();
    if (zero_index() != SIZE_MAX) {
      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        im2col[ks_index * mr() + zero_index()] = zero_sentinel;
      }
    }
    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = m(); m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = junk.data();
      }
    }
    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            if (im2col[ks_index * mr() + m_index] != zero_sentinel) {
              c_ref[m_index * n() + n_index] +=
                (int32_t(im2col[ks_index * mr() + m_index][k_index + a_offset()]) - quantization_params[0].zero_point) *
                int32_t(b[(n_index * ks() + ks_index) * k() + k_index]);
            }
          }
        }
        c_ref[m_index * n() + n_index] *= quantization_params[0].inv_scale * kernel_scale[n_index];
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min =
        qmin() == std::numeric_limits<uint8_t>::min() ? -std::numeric_limits<float>::infinity()
        : accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
    const float c_max =
        qmax() == std::numeric_limits<uint8_t>::max() ? +std::numeric_limits<float>::infinity()
        : accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

    // Prepare parameters.
    xnn_f16_minmax_params params;
    init_params(&params,
      fp16_ieee_from_fp32_value(c_min),
      fp16_ieee_from_fp32_value(c_max));

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], c_max), c_min);
      }
    }

    igemm(m(), n(), k(), ks() * mr() * sizeof(void*),
        im2col.data(), static_cast<const void*>(packed_w.data()),
        c.data(), cm_stride() * sizeof(uint16_t), cn_stride() * sizeof(uint16_t),
        a_offset() * sizeof(uint8_t), zero_sentinel, zero_data,
        &params, quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux AArch64.
        const float tolerance = std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 1.0e-2f);
        EXPECT_NEAR(fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << "), optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_qd8_f32_qc8w_igemm_ukernel_fn igemm,
  xnn_init_f32_minmax_params_fn init_params,
  xnn_pack_qs8_igemm_fn pack) const
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f), std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f), std::ref(rng));
  auto w8rng = std::bind(
      std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()),
      std::ref(rng));

  std::vector<float> input(mr() * k());
  std::vector<int8_t> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<xnn_qd8_quantization_params> quantization_params(1 + XNN_EXTRA_QUANTIZATION_PARAMS);
  std::vector<int8_t> b(n() * ks() * k());
  std::vector<float> bias(n());
  std::vector<float> kernel_scale(n());
  std::vector<int8_t, AlignedAllocator<int8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(ks() * packed_n() * packed_k() + packed_n() * (sizeof(int32_t) + sizeof(float) * 2));
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<float> c_ref(m() * n(), 0);
  std::vector<int8_t> junk(k() + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<const int8_t*> im2col(mr() * ks());

  std::fill(junk.begin(), junk.end(), 0xA5);
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    const auto minmax = std::minmax_element(input.begin(), input.begin() + mr() * k());
    quantization_params[0] = xnn_f32_qd8_asymmetric_quantization_params(*minmax.first, *minmax.second);
    const float inv_scale = 1.f / quantization_params[0].inv_scale;
    for (size_t i = 0; i < mr(); ++i) {
      const float* input_ptr = &input[i * k()];
      for (size_t j = 0; j < k(); ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(scaled_input, float(std::numeric_limits<int8_t>::max()
                                                           - quantization_params[0].zero_point));
        scaled_input = std::max<float>(scaled_input, float(std::numeric_limits<int8_t>::min()
                                                           - quantization_params[0].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) + long(quantization_params[0].zero_point));
      }
    }
    std::generate(b.begin(), b.end(), std::ref(w8rng));

    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));
    std::fill(c.begin(), c.end(), nanf(""));

    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_packing_params packing_params = { /*input_zero_point=*/1 };
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), /*bias=*/nullptr, /*scale=*/nullptr, packed_w.data(), 2 * sizeof(float) * nr(), &packing_params);
    // Fill in packed kernel scale
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      0,
      kernel_scale.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(float))));

    // Fill in packed bias
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      0,
      bias.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k() * sizeof(int8_t) + 2 * sizeof(float))));

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = a.data() + a_stride() * m_index - a_offset();
      }
    }
    std::shuffle(im2col.begin(), im2col.end(), rng);
    const size_t k_stride =  round_up_po2(k(), kr() * sr());
    std::vector<int8_t> zero_points(k_stride + XNN_EXTRA_BYTES, quantization_params[0].zero_point);
    const int8_t* zero_sentinel = (const int8_t*) &packing_params;
    const int8_t* zero_data = zero_points.data();
    if (zero_index() != SIZE_MAX) {
      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        im2col[ks_index * mr() + zero_index()] = zero_sentinel;
      }
    }
    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = m(); m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = junk.data();
      }
    }
    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            if (im2col[ks_index * mr() + m_index] != zero_sentinel) {
              c_ref[m_index * n() + n_index] +=
                (int32_t(im2col[ks_index * mr() + m_index][k_index + a_offset()]) - quantization_params[0].zero_point) *
                int32_t(b[(n_index * ks() + ks_index) * k() + k_index]);
            }
          }
        }
        c_ref[m_index * n() + n_index] *= quantization_params[0].inv_scale * kernel_scale[n_index];
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min =
        qmin() == std::numeric_limits<uint8_t>::min() ? -std::numeric_limits<float>::infinity()
        : accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
    const float c_max =
        qmax() == std::numeric_limits<uint8_t>::max() ? +std::numeric_limits<float>::infinity()
        : accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, c_min, c_max);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], c_max), c_min);
      }
    }

    igemm(m(), n(), k(), ks() * mr() * sizeof(void*),
        im2col.data(), static_cast<const void*>(packed_w.data()),
        c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
        a_offset() * sizeof(uint8_t), zero_sentinel, zero_data,
        &params, quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux AArch64.
        const float tolerance = std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f);
        EXPECT_NEAR(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << " (accumulator = " << acc[i * n() + j]
            << "), optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_qu8_gemm_minmax_ukernel_fn gemm,
  xnn_init_qu8_conv_minmax_params_fn init_params,
  xnn_pack_qu8_gemm_fn pack,
  xnn_qu8_requantize_fn requantize) const
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto u8rng = std::bind(
    std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), std::ref(rng));

  std::vector<uint8_t> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<uint8_t> b(n() * k());
  std::vector<int32_t> bias(n());
  std::vector<uint8_t, AlignedAllocator<uint8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k() + packed_n() * sizeof(int32_t) / sizeof(uint8_t));
  std::vector<uint8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<uint8_t> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(u8rng));
    std::generate(b.begin(), b.end(), std::ref(u8rng));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), b_zero_point());
    const xnn_qu8_packing_params packing_params = { a_zero_point(), b_zero_point() };
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0, &packing_params);

    // Compute 32-bit results and output quantization arguments.
    std::fill(acc.begin(), acc.end(), 0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          acc[m_index * n() + n_index] +=
              (int32_t(a[m_index * a_stride() + k_index]) - int32_t(a_zero_point())) *
              (int32_t(b[n_index * k() + k_index]) - int32_t(b_zero_point()));
        }
        acc[m_index * n() + n_index] += bias[n_index];
      }
    }

    const int32_t accumulated_min = *std::min_element(acc.cbegin(), acc.cend());
    const int32_t accumulated_max = *std::max_element(acc.cbegin(), acc.cend());
    const double c_scale = uint32_t(accumulated_max - accumulated_min) >= 256 ? double(uint32_t(accumulated_max - accumulated_min)) / 255.0 : 1.00001;
    const uint8_t c_zero_point = uint8_t(std::max(std::min(
      lrint(127.5 - 0.5 * double(accumulated_min + accumulated_max) / c_scale),
      long(std::numeric_limits<uint8_t>::max())), long(std::numeric_limits<uint8_t>::min())));

    const float requantization_scale = 1.0f / float(c_scale);
    union xnn_qu8_conv_minmax_params quantization_params;
    init_params(&quantization_params,
      b_zero_point(), requantization_scale, c_zero_point, qmin(), qmax());

    gemm(
      m(), n(), k(),
      a.data(), a_stride() * sizeof(uint8_t),
      packed_w.data(),
      c.data(), cm_stride() * sizeof(uint8_t), cn_stride() * sizeof(uint8_t),
      &quantization_params);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = requantize(
          acc[m_index * n() + n_index], requantization_scale, c_zero_point, qmin(), qmax());
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_LE(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(qmax()));
        EXPECT_GE(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(qmin()));
        EXPECT_EQ(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(c_ref[i * n() + j]))
            << "at " << i << ", " << j << ": reference = " << (uint32_t) c_ref[i * n() + j]
            << " (accumulator = " << acc[i * n() + j]
            << "), optimized = " << (uint32_t) c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", requantization scale = " << requantization_scale << ", output zero point = " << int32_t(c_zero_point);
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_qu8_igemm_minmax_ukernel_fn igemm,
  xnn_init_qu8_conv_minmax_params_fn init_params,
  xnn_pack_qu8_igemm_fn pack,
  xnn_qu8_requantize_fn requantize)
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto u8rng = std::bind(
    std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), std::ref(rng));

  std::vector<uint8_t> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<uint8_t> b(n() * ks() * k());
  std::vector<uint8_t, AlignedAllocator<uint8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(ks() * packed_n() * packed_k() + packed_n() * sizeof(int32_t) / sizeof(uint8_t));
  std::vector<int32_t> bias(n());
  std::vector<uint8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<uint8_t> c_ref(m() * n());
  std::vector<uint8_t> junk(k() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<const uint8_t*> im2col(mr() * ks());

  std::fill(junk.begin(), junk.end(), 0xA5);

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(u8rng));
    std::generate(b.begin(), b.end(), std::ref(u8rng));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), b_zero_point());
    const xnn_qu8_packing_params packing_params = { a_zero_point(), b_zero_point() };
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0, &packing_params);

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = a.data() + a_stride() * m_index - a_offset();
      }
    }
    std::shuffle(im2col.begin(), im2col.end(), rng);
    if (zero_index() != SIZE_MAX) {
      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        im2col[ks_index * mr() + zero_index()] = a.data();
      }
    }
    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = m(); m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = junk.data();
      }
    }

    // Compute 32-bit results and output quantization arguments.
    std::fill(acc.begin(), acc.end(), 0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            if (im2col[ks_index * mr() + m_index] == a.data()) {
              acc[m_index * n() + n_index] +=
                (int32_t(im2col[ks_index * mr() + m_index][k_index]) - int32_t(a_zero_point())) *
                (int32_t(b[(n_index * ks() + ks_index) * k() + k_index]) - int32_t(b_zero_point()));
            } else {
              acc[m_index * n() + n_index] +=
                (int32_t(im2col[ks_index * mr() + m_index][k_index + a_offset()]) - int32_t(a_zero_point())) *
                (int32_t(b[(n_index * ks() + ks_index) * k() + k_index]) - int32_t(b_zero_point()));
            }
          }
        }
        acc[m_index * n() + n_index] += bias[n_index];
      }
    }

    const int32_t accumulated_min = *std::min_element(acc.cbegin(), acc.cend());
    const int32_t accumulated_max = *std::max_element(acc.cbegin(), acc.cend());
    const double c_scale = uint32_t(accumulated_max - accumulated_min) >= 256 ? double(uint32_t(accumulated_max - accumulated_min)) / 255.0 : 1.00001;
    const uint8_t c_zero_point = uint8_t(std::max(std::min(
      lrint(127.5 - 0.5 * double(accumulated_min + accumulated_max) / c_scale),
      long(std::numeric_limits<uint8_t>::max())), long(std::numeric_limits<uint8_t>::min())));

    const float requantization_scale = 1.0f / float(c_scale);
    union xnn_qu8_conv_minmax_params quantization_params;
    init_params(&quantization_params,
      b_zero_point(), requantization_scale, c_zero_point, qmin(), qmax());

    const uint8_t* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : nullptr;

    igemm(
      m(), n(), k(), ks() * mr() * sizeof(void*),
      im2col.data(), packed_w.data(),
      c.data(), cm_stride() * sizeof(uint8_t), cn_stride() * sizeof(uint8_t),
      a_offset() * sizeof(uint8_t), zero_pointer,
      &quantization_params);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = requantize(
          acc[m_index * n() + n_index], requantization_scale, c_zero_point, qmin(), qmax());
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_LE(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(qmax()));
        EXPECT_GE(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(qmin()));
        EXPECT_EQ(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(c_ref[i * n() + j]))
            << "at " << i << ", " << j << ": reference = " << uint32_t(c_ref[i * n() + j])
            << " (accumulator = " << acc[i * n() + j]
            << "), optimized = " << (uint32_t) c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", requantization scale = " << requantization_scale << ", output zero point = " << int32_t(c_zero_point);
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_qs8_qc8w_gemm_minmax_ukernel_fn gemm,
  xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
  xnn_pack_qs8_gemm_fn pack,
  xnn_qs8_requantize_fn requantize) const
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));
  auto w8rng = std::bind(
    std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<int8_t> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<int8_t> b(n() * k());
  std::vector<int32_t> bias(n());
  std::vector<int8_t, AlignedAllocator<int8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k() + packed_n() * (sizeof(int32_t) + sizeof(float)) / sizeof(int8_t));
  std::vector<int8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<float> scale(n());
  std::vector<int8_t> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(i8rng));
    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = { int8_t(a_zero_point() - 0x80) };
    void* const packed_data = packed_w.data();
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
        b.data(), bias.data(), /*scale=*/nullptr, packed_data, nr() * sizeof(float), &packing_params);

    // Compute 32-bit results and output quantization arguments.
    std::fill(acc.begin(), acc.end(), 0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          acc[m_index * n() + n_index] +=
              (int32_t(a[m_index * a_stride() + k_index]) - int32_t(a_zero_point() - 0x80)) *
              int32_t(b[n_index * k() + k_index]);
        }
        acc[m_index * n() + n_index] += bias[n_index];
      }
    }

    const int8_t c_zero_point = -1;
    for (size_t n_index = 0; n_index < n(); n_index++) {
      int32_t accumulated_min = acc[n_index];
      int32_t accumulated_max = acc[n_index];
      for (size_t m_index = 0; m_index < m(); m_index++) {
        accumulated_min = std::min(accumulated_min, acc[m_index * n() + n_index]);
        accumulated_max = std::max(accumulated_max, acc[m_index * n() + n_index]);
      }
      const uint32_t accumulated_range = uint32_t(accumulated_max - accumulated_min);
      const float c_scale = accumulated_range >= 256 ? double(accumulated_range) / 255.0 : 1.00001;
      scale[n_index] = 1.0f / c_scale;
    }

    const size_t type_size = sizeof(int8_t);
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (packed_k() * type_size + (sizeof(int32_t) + sizeof(float))),
      nr() * (packed_k() * type_size + (sizeof(int32_t) + sizeof(float))),
      0,
      scale.data(),
      (void*) ((uintptr_t) packed_data + nr() * (packed_k() * type_size + sizeof(int32_t))));

    union xnn_qs8_qc8w_conv_minmax_params minmax_params;
    init_params(&minmax_params,
      c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

    gemm(
      m(), n(), k(),
      a.data(), a_stride() * sizeof(int8_t),
      packed_data,
      c.data(), cm_stride() * sizeof(int8_t), cn_stride() * sizeof(int8_t),
      &minmax_params);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = requantize(
          acc[m_index * n() + n_index], scale[n_index], c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_LE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmax()) - 0x80);
        EXPECT_GE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmin()) - 0x80);
        EXPECT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(c_ref[i * n() + j]))
            << "at " << i << ", " << j << ": reference = " << int32_t(c_ref[i * n() + j])
            << " (accumulator = " << acc[i * n() + j]
            << "), optimized = " << int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]) << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", requantization scale = " << scale[j] << ", output zero point = " << int32_t(c_zero_point);
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_qs8_qc8w_igemm_minmax_ukernel_fn igemm,
  xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
  xnn_pack_qs8_igemm_fn pack,
  xnn_qs8_requantize_fn requantize) const
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));
  auto w8rng = std::bind(
    std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<int8_t> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<int8_t> b(n() * ks() * k());
  std::vector<int8_t, AlignedAllocator<int8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(ks() * packed_n() * packed_k() + packed_n() * (sizeof(int32_t) + sizeof(float)) / sizeof(int8_t));
  std::vector<int32_t> bias(n());
  std::vector<int8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<float> scale(n());
  std::vector<int8_t> c_ref(m() * n());
  std::vector<int8_t> junk(k() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<const int8_t*> im2col(mr() * ks());

  std::fill(junk.begin(), junk.end(), 0xA5);

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(i8rng));
    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = { int8_t(a_zero_point() - 0x80) };
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), nr() * sizeof(float), &packing_params);

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = a.data() + a_stride() * m_index - a_offset();
      }
    }
    std::shuffle(im2col.begin(), im2col.end(), rng);
    if (zero_index() != SIZE_MAX) {
      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        im2col[ks_index * mr() + zero_index()] = a.data();
      }
    }
    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = m(); m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = junk.data();
      }
    }

    // Compute 32-bit results and output quantization arguments.
    std::fill(acc.begin(), acc.end(), 0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            if (im2col[ks_index * mr() + m_index] == a.data()) {
              acc[m_index * n() + n_index] +=
                (int32_t(im2col[ks_index * mr() + m_index][k_index]) - int32_t(a_zero_point() - 0x80)) *
                int32_t(b[(n_index * ks() + ks_index) * k() + k_index]);
            } else {
              acc[m_index * n() + n_index] +=
                (int32_t(im2col[ks_index * mr() + m_index][k_index + a_offset()]) - int32_t(a_zero_point() - 0x80)) *
                int32_t(b[(n_index * ks() + ks_index) * k() + k_index]);
            }
          }
        }
        acc[m_index * n() + n_index] += bias[n_index];
      }
    }

    const int8_t c_zero_point = -1;
    for (size_t n_index = 0; n_index < n(); n_index++) {
      int32_t accumulated_min = acc[n_index];
      int32_t accumulated_max = acc[n_index];
      for (size_t m_index = 0; m_index < m(); m_index++) {
        accumulated_min = std::min(accumulated_min, acc[m_index * n() + n_index]);
        accumulated_max = std::max(accumulated_max, acc[m_index * n() + n_index]);
      }
      const uint32_t accumulated_range = uint32_t(accumulated_max - accumulated_min);
      const float c_scale = accumulated_range >= 256 ? double(accumulated_range) / 255.0 : 1.00001;
      scale[n_index] = 1.0f / c_scale;
    }

    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k() * sizeof(int8_t) + (sizeof(int32_t) + sizeof(float))),
      nr() * (ks() * packed_k() * sizeof(int8_t) + (sizeof(int32_t) + sizeof(float))),
      0,
      scale.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(int32_t))));

    union xnn_qs8_qc8w_conv_minmax_params minmax_params;
    init_params(&minmax_params,
      c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

    const int8_t* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : nullptr;

    igemm(
      m(), n(), k(), ks() * mr() * sizeof(void*),
      im2col.data(), packed_w.data(),
      c.data(), cm_stride() * sizeof(int8_t), cn_stride() * sizeof(int8_t),
      a_offset() * sizeof(uint8_t), zero_pointer,
      &minmax_params);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = requantize(
          acc[m_index * n() + n_index], scale[n_index], c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_LE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmax()) - 0x80);
        EXPECT_GE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmin()) - 0x80);
        EXPECT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(c_ref[i * n() + j]))
            << "at " << i << ", " << j << ": reference = " << uint32_t(c_ref[i * n() + j])
            << " (accumulator = " << acc[i * n() + j]
            << "), optimized = " << (uint32_t) c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", requantization scale = " << scale[j] << ", output zero point = " << int32_t(c_zero_point);
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_qd8_f16_qc8w_gemm_ukernel_fn gemm,
  xnn_init_f16_minmax_params_fn init_params,
  xnn_pack_qs8_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f), std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f), std::ref(rng));
  auto w8rng = std::bind(
      std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()),
      std::ref(rng));

  std::vector<float> input(m() * k());
  std::vector<int8_t> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<xnn_qd8_quantization_params> quantization_params(mr());
  std::vector<int8_t> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float> kernel_scale(n());
  std::vector<int8_t, AlignedAllocator<int8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k() +
                                                             packed_n() * (sizeof(int32_t) + sizeof(float) * 2));
  std::vector<uint16_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n(), 0);

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    for (size_t i = 0; i < m(); ++i) {
      const float* input_ptr = &input[i * k()];
      const auto minmax = std::minmax_element(input_ptr, input_ptr + k());
      quantization_params[i] = xnn_f32_qd8_asymmetric_quantization_params(*minmax.first, *minmax.second);
      const float inv_scale = 1.f / quantization_params[i].inv_scale;
      for (size_t j = 0; j < k(); ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(scaled_input, float(std::numeric_limits<int8_t>::max()
                                                           - quantization_params[i].zero_point));
        scaled_input = std::max<float>(scaled_input, float(std::numeric_limits<int8_t>::min()
                                                           - quantization_params[i].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) + long(quantization_params[i].zero_point));
      }
    }
    for (size_t i = m(); i < mr(); ++i) {
      quantization_params[i].zero_point = quantization_params[m() - 1].zero_point;
      quantization_params[i].inv_scale = quantization_params[m() - 1].inv_scale;
    }
    std::generate(b.begin(), b.end(), std::ref(w8rng));

    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));
    std::fill(c.begin(), c.end(), UINT16_C(0xDEAD));

    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_packing_params packing_params = { /*input_zero_point=*/1 };
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
      b.data(), /*bias=*/nullptr, /*scale=*/nullptr, packed_w.data(), 2 * sizeof(float) * nr(), &packing_params);
    // Fill in packed kernel scale
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      0,
      kernel_scale.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(float))));

    // Fill in packed bias
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      0,
      bias.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k() * sizeof(int8_t) + 2 * sizeof(float))));

    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        int32_t ksum = 0;
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ksum += b[n_index * k() + k_index];
          c_ref[m_index * n() + n_index] +=
              int32_t(a[m_index * a_stride() + k_index]) *
              int32_t(b[n_index * k() + k_index]);
        }
        c_ref[m_index * n() + n_index] -= (quantization_params[m_index].zero_point * ksum);
        c_ref[m_index * n() + n_index] *= quantization_params[m_index].inv_scale * kernel_scale[n_index];
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min =
        qmin() == std::numeric_limits<uint8_t>::min() ? -std::numeric_limits<float>::infinity()
        : accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
    const float c_max =
        qmax() == std::numeric_limits<uint8_t>::max() ? +std::numeric_limits<float>::infinity()
        : accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

    // Prepare parameters.
    xnn_f16_minmax_params params;
    init_params(&params,
      fp16_ieee_from_fp32_value(c_min),
      fp16_ieee_from_fp32_value(c_max));

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], c_max), c_min);
      }
    }

    gemm(m(), n(), k(),
        a.data(), a_stride() * sizeof(int8_t),
        static_cast<const void*>(packed_w.data()),
        c.data(), cm_stride() * sizeof(uint16_t), cn_stride() * sizeof(uint16_t), &params, quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux AArch64.
        const float tolerance = std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 1.0e-2f);
        EXPECT_NEAR(fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << "), optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_qd8_f32_qc8w_gemm_ukernel_fn gemm,
  xnn_init_f32_minmax_params_fn init_params,
  xnn_pack_qs8_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f), std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f), std::ref(rng));
  auto w8rng = std::bind(
      std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()),
      std::ref(rng));

  std::vector<float> input(m() * k());
  std::vector<int8_t> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<xnn_qd8_quantization_params> quantization_params(mr());
  std::vector<int8_t> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float> kernel_scale(n());
  std::vector<int8_t, AlignedAllocator<int8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k() +
                                                             packed_n() * (sizeof(int32_t) + sizeof(float) * 2));
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<float> c_ref(m() * n(), 0);

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    for (size_t i = 0; i < m(); ++i) {
      const float* input_ptr = &input[i * k()];
      const auto minmax = std::minmax_element(input_ptr, input_ptr + k());
      quantization_params[i] = xnn_f32_qd8_asymmetric_quantization_params(*minmax.first, *minmax.second);
      const float inv_scale = 1.f / quantization_params[i].inv_scale;
      for (size_t j = 0; j < k(); ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(scaled_input, float(std::numeric_limits<int8_t>::max()
                                                           - quantization_params[i].zero_point));
        scaled_input = std::max<float>(scaled_input, float(std::numeric_limits<int8_t>::min()
                                                           - quantization_params[i].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) + long(quantization_params[i].zero_point));
      }
    }
    for (size_t i = m(); i < mr(); ++i) {
      quantization_params[i].zero_point = quantization_params[m() - 1].zero_point;
      quantization_params[i].inv_scale = quantization_params[m() - 1].inv_scale;
    }
    std::generate(b.begin(), b.end(), std::ref(w8rng));

    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));
    std::fill(c.begin(), c.end(), nanf(""));

    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_packing_params packing_params = { /*input_zero_point=*/1 };
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
      b.data(), /*bias=*/nullptr, /*scale=*/nullptr, packed_w.data(), 2 * sizeof(float) * nr(), &packing_params);
    // Fill in packed kernel scale
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      0,
      kernel_scale.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(float))));

    // Fill in packed bias
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
      0,
      bias.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k() * sizeof(int8_t) + 2 * sizeof(float))));

    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        int32_t ksum = 0;
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ksum += b[n_index * k() + k_index];
          c_ref[m_index * n() + n_index] +=
              int32_t(a[m_index * a_stride() + k_index]) *
              int32_t(b[n_index * k() + k_index]);
        }
        c_ref[m_index * n() + n_index] -= (quantization_params[m_index].zero_point * ksum);
        c_ref[m_index * n() + n_index] *= quantization_params[m_index].inv_scale * kernel_scale[n_index];
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min =
        qmin() == std::numeric_limits<uint8_t>::min() ? -std::numeric_limits<float>::infinity()
        : accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
    const float c_max =
        qmax() == std::numeric_limits<uint8_t>::max() ? +std::numeric_limits<float>::infinity()
        : accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, c_min, c_max);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], c_max), c_min);
      }
    }

    gemm(m(), n(), k(),
        a.data(), a_stride() * sizeof(int8_t),
        static_cast<const void*>(packed_w.data()),
        c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float), &params, quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux AArch64.
        const float tolerance = std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f);
        EXPECT_NEAR(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << " (accumulator = " << acc[i * n() + j]
            << "), optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_qd8_f16_qc4w_gemm_ukernel_fn gemm,
  xnn_init_f16_qc4w_minmax_params_fn init_params,
  xnn_pack_qs8_qc4w_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f), std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f), std::ref(rng));
  auto w8rng = std::bind(
      std::uniform_int_distribution<int32_t>(0, std::numeric_limits<uint8_t>::max()),
      std::ref(rng));

  const size_t planes = 2;  // 4 bit is 2 planes - low nibbles and high nibbles
  const size_t k2 =  round_up_po2(k(), 2);  // tester assumes byte aligned rows
  const size_t packed_k2 = round_up_po2(k(), kr() * sr() * planes);  // 2 blocks for nibbles
  const size_t packed_k_bytes = (packed_k2 + 1)/ 2;

  std::vector<float> input(m() * k2);
  std::vector<int8_t> a((m() - 1) * a_stride() + k2 + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<xnn_qd8_quantization_params> quantization_params(mr());
  std::vector<uint8_t> b(n() * k2 / 2);
  std::vector<float> bias(n());
  std::vector<float> kernel_scale(n());
  std::vector<uint8_t, AlignedAllocator<uint8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k_bytes +
                                                               packed_n() * (sizeof(int32_t) + sizeof(float) * 2));
  std::vector<uint16_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<float> c_ref(m() * n(), 0.0f);

  {//for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    for (size_t i = 0; i < m(); ++i) {
      const float* input_ptr = &input[i * k2];
      const auto minmax = std::minmax_element(input_ptr, input_ptr + k2);
      quantization_params[i] = xnn_f32_qd8_asymmetric_quantization_params(*minmax.first, *minmax.second);
      const float inv_scale = 1.f / quantization_params[i].inv_scale;
      for (size_t j = 0; j < k2; ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(scaled_input, float(std::numeric_limits<int8_t>::max()
                                                           - quantization_params[i].zero_point));
        scaled_input = std::max<float>(scaled_input, float(std::numeric_limits<int8_t>::min()
                                                           - quantization_params[i].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) + long(quantization_params[i].zero_point));
      }
    }
    for (size_t i = m(); i < mr(); ++i) {
      quantization_params[i].zero_point = quantization_params[m() - 1].zero_point;
      quantization_params[i].inv_scale = quantization_params[m() - 1].inv_scale;
    }

    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));
    std::fill(c.begin(), c.end(), UINT16_C(0x7E00));
    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_qc4w_packing_params packing_params = { /*input_zero_point=*/1, b_zero_point()};
    pack(/*g=*/1, n(), k2, nr(), kr(), sr(),
      b.data(), /*bias=*/nullptr, /*scale=*/nullptr,
      packed_w.data(), 2 * sizeof(float) * nr(), &packing_params);
    // Fill in packed kernel scale
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k_bytes + 3 * sizeof(float)),
      nr() * (ks() * packed_k_bytes + 3 * sizeof(float)),
      0,
      kernel_scale.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k_bytes + sizeof(float))));

    // Fill in packed bias
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k_bytes + 3 * sizeof(float)),
      nr() * (ks() * packed_k_bytes + 3 * sizeof(float)),
      0,
      bias.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k_bytes + 2 * sizeof(float))));

    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        int32_t ksum = 0;
        for (size_t k_index = 0; k_index < k2; k_index++) {
          const size_t nb_index = (n_index * k2 + k_index) / 2;
          const int32_t bv = int32_t((k_index % 2 == 0) ? (b[nb_index] & UINT8_C(0xF)) : (b[nb_index] >> 4)) - b_zero_point();
          ksum += bv;
          c_ref[m_index * n() + n_index] += int32_t(a[m_index * a_stride() + k_index]) * int32_t(bv);
        }
        c_ref[m_index * n() + n_index] -= (quantization_params[m_index].zero_point * ksum);
        c_ref[m_index * n() + n_index] *= quantization_params[m_index].inv_scale * kernel_scale[n_index];
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min =
        qmin() == std::numeric_limits<uint8_t>::min() ? -std::numeric_limits<float>::infinity()
        : accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
    const float c_max =
        qmax() == std::numeric_limits<uint8_t>::max() ? +std::numeric_limits<float>::infinity()
        : accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

    // Prepare parameters.
    xnn_f16_qc4w_minmax_params params;
    init_params(&params,
      fp16_ieee_from_fp32_value(c_min),
      fp16_ieee_from_fp32_value(c_max), 8);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], c_max), c_min);
      }
    }

    gemm(m(), n(), k2,
        a.data(), a_stride() * sizeof(int8_t),
        static_cast<const void*>(packed_w.data()),
        c.data(), cm_stride() * sizeof(uint16_t), cn_stride() * sizeof(uint16_t), &params, quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux AArch64.
        const float tolerance = std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 1.0e-2f);
        EXPECT_NEAR(fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << " (accumulator = " << acc[i * n() + j]
            << "), optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_qd8_f16_qb4w_gemm_ukernel_fn gemm,
  xnn_init_f16_qb4w_minmax_params_fn init_params,
  xnn_pack_qs8_qb4w_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f), std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f), std::ref(rng));
  auto w8rng = std::bind(
      std::uniform_int_distribution<int32_t>(0, std::numeric_limits<uint8_t>::max()),
      std::ref(rng));

  const size_t planes = 2;  // 4 bit is 2 planes - low nibbles and high nibbles
  const size_t k2 =  round_up_po2(k(), 2);  // tester assumes byte aligned rows

  const size_t packed_k2 = round_up_po2(k(), kr() * sr() * planes);  // 2 blocks for nibbles
  const size_t packed_k_bytes = (packed_k2 + 1)/ 2;
  const size_t num_blocks = packed_k2 / bl();

  std::vector<float> input(m() * k2);
  std::vector<int8_t> a((m() - 1) * a_stride() + k2 + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<xnn_qd8_quantization_params> quantization_params(mr());
  std::vector<uint8_t> b(n() * k2 / 2);
  std::vector<float> bias(n());
  std::vector<uint16_t> kernel_scale2d(n() * k2 / bl());
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> packed_w(packed_n() * packed_k_bytes +
                                                               /* vksum */ packed_n() * sizeof(float) +
                                                               /* scales */ packed_n() * num_blocks * sizeof(uint16_t) +
                                                               /* bias */ packed_n() * sizeof(float));

  std::vector<uint16_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n(), 0);

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    for (size_t i = 0; i < m(); ++i) {
      const float* input_ptr = &input[i * k2];
      const auto minmax = std::minmax_element(input_ptr, input_ptr + k2);
      quantization_params[i] = xnn_f32_qd8_asymmetric_quantization_params(*minmax.first, *minmax.second);
      const float inv_scale = 1.f / quantization_params[i].inv_scale;
      for (size_t j = 0; j < k2; ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(scaled_input, float(std::numeric_limits<int8_t>::max()
                                                           - quantization_params[i].zero_point));
        scaled_input = std::max<float>(scaled_input, float(std::numeric_limits<int8_t>::min()
                                                           - quantization_params[i].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) + long(quantization_params[i].zero_point));
      }
    }
    for (size_t i = m(); i < mr(); ++i) {
      quantization_params[i].zero_point = quantization_params[m() - 1].zero_point;
      quantization_params[i].inv_scale = quantization_params[m() - 1].inv_scale;
    }

    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale2d.begin(), kernel_scale2d.end(), [&]() { return math_cvt_bf16_fp32(scalerng()); });

    std::fill(c.begin(), c.end(), UINT16_C(0x7E00));
    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_qc4w_packing_params packing_params = { /*input_zero_point=*/1, b_zero_point()};

    pack(/*g=*/1, n(), k2, nr(), kr(), sr(), bl(),
      b.data(), /*bias=*/nullptr, /*scale=*/kernel_scale2d.data(),
      packed_w.data(), sizeof(uint16_t) * nr(), sizeof(float) * nr(), &packing_params);

    // Fill in packed kernel scale
    size_t stride =  nr() * (packed_k_bytes + /* scales= */ num_blocks * sizeof(uint16_t) + /* ksum= */ sizeof(float) + /* bias= */ sizeof(float));
    size_t block_stride = (bl() / 2 + sizeof(uint16_t)) * nr();
    size_t start_offset = nr() * (packed_k_bytes / num_blocks + sizeof(float));
    uintptr_t start = (uintptr_t) packed_w.data() + start_offset;
    xnn_init_blockwise_scale_bf16_params(
      n(), nr(), nr(),
      stride,
      stride,
      /*num_blocks=*/ num_blocks,
      /*block_stride=*/ block_stride,
      0,
      kernel_scale2d.data(),
      (void*) start);

    start = (uintptr_t) packed_w.data() + stride - sizeof(float) * nr();
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      stride,
      stride,
      0,
      bias.data(),
      (void*) start);

    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        float kfsum = 0.0;
        for (size_t bl_index=0; bl_index < num_blocks; ++bl_index) {
          int32_t ksum = 0;
          int32_t c_ref_acc = 0;
          for (size_t kr_index = 0; kr_index < bl(); kr_index++) {
            const size_t k_index =  bl_index * bl() + kr_index;
            const size_t nb_index = (n_index * k2 + k_index) / 2;
            const int32_t bv = int32_t((k_index % 2 == 0) ? (b[nb_index] & UINT8_C(0xF)) : (b[nb_index] >> 4)) - b_zero_point();
            ksum += bv;
            c_ref_acc += int32_t(a[m_index * a_stride() + k_index]) * int32_t(bv);
          }
          size_t scale_index = n_index * num_blocks + bl_index;
          float scale = math_cvt_fp32_bf16(kernel_scale2d[scale_index]);
          c_ref[m_index * n() + n_index] += c_ref_acc * scale;
          kfsum += scale * ksum;
        }
        c_ref[m_index * n() + n_index] -= (quantization_params[m_index].zero_point * kfsum);
        c_ref[m_index * n() + n_index] *= quantization_params[m_index].inv_scale;
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min =
        qmin() == std::numeric_limits<uint8_t>::min() ? -std::numeric_limits<float>::infinity()
        : accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
    const float c_max =
        qmax() == std::numeric_limits<uint8_t>::max() ? +std::numeric_limits<float>::infinity()
        : accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

    // Prepare parameters.
    xnn_f16_qb4w_minmax_params params;
    init_params(&params,
      fp16_ieee_from_fp32_value(c_min),
      fp16_ieee_from_fp32_value(c_max),
      8,
      bl());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], c_max), c_min);
      }
    }

    gemm(m(), n(), k2,
        a.data(), a_stride() * sizeof(int8_t),
        static_cast<const void*>(packed_w.data()),
        c.data(), cm_stride() * sizeof(uint16_t), cn_stride() * sizeof(uint16_t), &params, quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux AArch64.
        const float tolerance = std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 1.0e-3f);
        EXPECT_NEAR(fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k2;
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_qd8_f32_qc4w_gemm_ukernel_fn gemm,
  xnn_init_f32_qc4w_minmax_params_fn init_params,
  xnn_pack_qs8_qc4w_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f), std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f), std::ref(rng));
  auto w8rng = std::bind(
      std::uniform_int_distribution<int32_t>(0, std::numeric_limits<uint8_t>::max()),
      std::ref(rng));

  const size_t planes = 2;  // 4 bit is 2 planes - low nibbles and high nibbles
  const size_t k2 =  round_up_po2(k(), 2);  // tester assumes byte aligned rows
  const size_t packed_k2 = round_up_po2(k(), kr() * sr() * planes);  // 2 blocks for nibbles
  const size_t packed_k_bytes = (packed_k2 + 1)/ 2;

  std::vector<float> input(m() * k2);
  std::vector<int8_t> a((m() - 1) * a_stride() + k2 + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<xnn_qd8_quantization_params> quantization_params(mr());
  std::vector<uint8_t> b(n() * k2 / 2);
  std::vector<float> bias(n());
  std::vector<float> kernel_scale(n());
  std::vector<uint8_t, AlignedAllocator<uint8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k_bytes +
                                                               packed_n() * (sizeof(int32_t) + sizeof(float) * 2));
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<float> c_ref(m() * n(), 0);

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    for (size_t i = 0; i < m(); ++i) {
      const float* input_ptr = &input[i * k2];
      const auto minmax = std::minmax_element(input_ptr, input_ptr + k2);
      quantization_params[i] = xnn_f32_qd8_asymmetric_quantization_params(*minmax.first, *minmax.second);
      const float inv_scale = 1.f / quantization_params[i].inv_scale;
      for (size_t j = 0; j < k2; ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(scaled_input, float(std::numeric_limits<int8_t>::max()
                                                           - quantization_params[i].zero_point));
        scaled_input = std::max<float>(scaled_input, float(std::numeric_limits<int8_t>::min()
                                                           - quantization_params[i].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) + long(quantization_params[i].zero_point));
      }
    }
    for (size_t i = m(); i < mr(); ++i) {
      quantization_params[i].zero_point = quantization_params[m() - 1].zero_point;
      quantization_params[i].inv_scale = quantization_params[m() - 1].inv_scale;
    }

    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_qc4w_packing_params packing_params = { /*input_zero_point=*/1, b_zero_point()};
    pack(/*g=*/1, n(), k2, nr(), kr(), sr(),
      b.data(), /*bias=*/nullptr, /*scale=*/nullptr,
      packed_w.data(), 2 * sizeof(float) * nr(), &packing_params);
    // Fill in packed kernel scale
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k_bytes + 3 * sizeof(float)),
      nr() * (ks() * packed_k_bytes + 3 * sizeof(float)),
      0,
      kernel_scale.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k_bytes + sizeof(float))));

    // Fill in packed bias
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k_bytes + 3 * sizeof(float)),
      nr() * (ks() * packed_k_bytes + 3 * sizeof(float)),
      0,
      bias.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k_bytes + 2 * sizeof(float))));

    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        int32_t ksum = 0;
        for (size_t k_index = 0; k_index < k2; k_index++) {
          const size_t nb_index = (n_index * k2 + k_index) / 2;
          const int32_t bv = int32_t((k_index % 2 == 0) ? (b[nb_index] & UINT8_C(0xF)) : (b[nb_index] >> 4)) - b_zero_point();
          ksum += bv;
          c_ref[m_index * n() + n_index] += int32_t(a[m_index * a_stride() + k_index]) * int32_t(bv);
        }
        c_ref[m_index * n() + n_index] -= (quantization_params[m_index].zero_point * ksum);
        c_ref[m_index * n() + n_index] *= quantization_params[m_index].inv_scale * kernel_scale[n_index];
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min =
        qmin() == std::numeric_limits<uint8_t>::min() ? -std::numeric_limits<float>::infinity()
        : accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
    const float c_max =
        qmax() == std::numeric_limits<uint8_t>::max() ? +std::numeric_limits<float>::infinity()
        : accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

    // Prepare parameters.
    xnn_f32_qc4w_minmax_params params;
    init_params(&params, c_min, c_max, 8);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], c_max), c_min);
      }
    }

    gemm(m(), n(), k2,
        a.data(), a_stride() * sizeof(int8_t),
        static_cast<const void*>(packed_w.data()),
        c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float), &params, quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux AArch64.
        const float tolerance = std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f);
        EXPECT_NEAR(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << " (accumulator = " << acc[i * n() + j]
            << "), optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k2;
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_qd8_f32_qb4w_gemm_ukernel_fn gemm,
  xnn_init_f32_qb4w_minmax_params_fn init_params,
  xnn_pack_qs8_qb4w_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f), std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f), std::ref(rng));
  auto w8rng = std::bind(
      std::uniform_int_distribution<int32_t>(0, std::numeric_limits<uint8_t>::max()),
      std::ref(rng));

  const size_t planes = 2;  // 4 bit is 2 planes - low nibbles and high nibbles
  const size_t k2 =  round_up_po2(k(), 2);  // tester assumes byte aligned rows
  const size_t packed_k2 = round_up_po2(k(), kr() * sr() * planes);  // 2 blocks for nibbles
  const size_t packed_k_bytes = (packed_k2 + 1)/ 2;
  const size_t num_blocks = packed_k2 / bl();

  std::vector<float> input(m() * k2);
  std::vector<int8_t> a((m() - 1) * a_stride() + k2 + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<xnn_qd8_quantization_params> quantization_params(mr());
  std::vector<uint8_t> b(n() * k2 / 2);
  std::vector<float> bias(n());
  std::vector<uint16_t> kernel_scale2d(n() * k2 / bl());
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> packed_w(packed_n() * packed_k_bytes +
                                                               /* vksum */ packed_n() * sizeof(float) +
                                                               /* scales */ packed_n() * num_blocks * sizeof(float) +
                                                               /* bias */ packed_n() * sizeof(float));

  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n(), 0);

  for (size_t iteration = 0; iteration < 1 /* iterations() */; iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    for (size_t i = 0; i < m(); ++i) {
      const float* input_ptr = &input[i * k2];
      const auto minmax = std::minmax_element(input_ptr, input_ptr + k2);
      quantization_params[i] = xnn_f32_qd8_asymmetric_quantization_params(*minmax.first, *minmax.second);
      const float inv_scale = 1.f / quantization_params[i].inv_scale;
      for (size_t j = 0; j < k2; ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(scaled_input, float(std::numeric_limits<int8_t>::max()
                                                           - quantization_params[i].zero_point));
        scaled_input = std::max<float>(scaled_input, float(std::numeric_limits<int8_t>::min()
                                                           - quantization_params[i].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) + long(quantization_params[i].zero_point));
      }
    }
    for (size_t i = m(); i < mr(); ++i) {
      quantization_params[i].zero_point = quantization_params[m() - 1].zero_point;
      quantization_params[i].inv_scale = quantization_params[m() - 1].inv_scale;
    }

    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale2d.begin(), kernel_scale2d.end(), [&]() { return math_cvt_bf16_fp32(scalerng()); });

    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_qc4w_packing_params packing_params = { /*input_zero_point=*/1, b_zero_point()};
    pack(/*g=*/1, n(), k2, nr(), kr(), sr(), bl(),
      b.data(), /*bias=*/nullptr, /*scale=*/kernel_scale2d.data(),
      packed_w.data(), sizeof(uint16_t) * nr(), sizeof(float) * nr(), &packing_params);

    // Fill in packed kernel scale
    size_t stride =  nr() * (packed_k_bytes + /* scales= */ num_blocks * sizeof(uint16_t) + /* ksum= */ sizeof(float) + /* bias= */ sizeof(float));
    size_t block_stride = (bl() / 2 + sizeof(uint16_t)) * nr();
    size_t start_offset = nr() * (packed_k_bytes / num_blocks + sizeof(float));
    uintptr_t start = (uintptr_t) packed_w.data() + start_offset;
    xnn_init_blockwise_scale_bf16_params(
      n(), nr(), nr(),
      stride,
      stride,
      /*num_blocks=*/ num_blocks,
      /*block_stride=*/ block_stride,
      0,
      kernel_scale2d.data(),
      (void*) start);

    start = (uintptr_t) packed_w.data() + stride - sizeof(float) * nr();
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      stride,
      stride,
      0,
      bias.data(),
      (void*) start);

    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        float kfsum = 0.0;
        for (size_t bl_index=0; bl_index < num_blocks; ++bl_index) {
          int32_t ksum = 0;
          int32_t c_ref_acc = 0;
          for (size_t kr_index = 0; kr_index < bl(); kr_index++) {
            const size_t k_index =  bl_index * bl() + kr_index;
            const size_t nb_index = (n_index * k2 + k_index) / 2;
            const int32_t bv = int32_t((k_index % 2 == 0) ? (b[nb_index] & UINT8_C(0xF)) : (b[nb_index] >> 4)) - b_zero_point();
            ksum += bv;
            c_ref_acc += int32_t(a[m_index * a_stride() + k_index]) * int32_t(bv);
          }
          size_t scale_index = n_index * num_blocks + bl_index;
          float scale = math_cvt_fp32_bf16(kernel_scale2d[scale_index]);
          c_ref[m_index * n() + n_index] += c_ref_acc * scale;
          kfsum += scale * ksum;
        }
        c_ref[m_index * n() + n_index] -= (quantization_params[m_index].zero_point * kfsum);
        c_ref[m_index * n() + n_index] *= quantization_params[m_index].inv_scale;
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min =
        qmin() == std::numeric_limits<uint8_t>::min() ? -std::numeric_limits<float>::infinity()
        : accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
    const float c_max =
        qmax() == std::numeric_limits<uint8_t>::max() ? +std::numeric_limits<float>::infinity()
        : accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

    // Prepare parameters.
    xnn_f32_qb4w_minmax_params params;
    init_params(&params, c_min, c_max, 8, bl());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], c_max), c_min);
      }
    }

    gemm(m(), n(), k2,
        a.data(), a_stride() * sizeof(int8_t),
        static_cast<const void*>(packed_w.data()),
        c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float), &params, quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux AArch64.
        const float tolerance =
            std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 1.0e-5f);
        EXPECT_NEAR(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k2;
      }
    }
  }
}

void GemmMicrokernelTester::Test(
    xnn_qp8_f32_qc4w_gemm_minmax_ukernel_fn gemm,
    xnn_init_f32_minmax_params_fn init_minmax_params,
    xnn_pack_weights_and_biases_fn pack,
    xnn_packed_stride_weights_and_biases_fn packed_stride) {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f),
                          std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f),
                            std::ref(rng));
  auto w8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             0, std::numeric_limits<uint8_t>::max()),
                         std::ref(rng));

  const size_t k2 = round_up_po2(k(), 2);  // tester assumes byte aligned rows

  std::vector<float> input_f32(m() * k2);
  std::vector<uint8_t> b(n() * k2 / 2);
  std::vector<float> bias(n(), 0.0f);
  std::vector<float> kernel_scale(n());
  std::vector<float> c((mr() - 1) * cm_stride() +
                       ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<float> c_ref(m() * n(), 0);

  // Create a fake `gemm_config` for the packing functions.
  struct xnn_gemm_config gemm_config;
  gemm_config.mr = static_cast<uint8_t>(mr());
  gemm_config.mr_packed = static_cast<uint8_t>(mr_packed());
  gemm_config.nr = static_cast<uint8_t>(nr());
  gemm_config.log2_kr = static_cast<uint8_t>(31 - math_clz_nonzero_u32(kr()));
  gemm_config.log2_sr = static_cast<uint8_t>(31 - math_clz_nonzero_u32(sr()));

  const size_t packed_w_stride =
      packed_stride(&gemm_config, k2, /*k_stride=*/k2, /*extra_bytes=*/0);
  const size_t packed_w_size = packed_w_stride * round_up(n(), nr());
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> packed_w(packed_w_size);

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input_f32.begin(), input_f32.end(), std::ref(f32rng));

    // Quantize the left-hand operand.
    const size_t input_packed_size =
        xnn_x8_packq_f32qp8_packed_size(m(), k2, mr_packed(), kr(), sr());
    std::vector<int8_t> input_qp8(input_packed_size);
    xnn_x8_packq_f32qp8_ukernel__scalar_u1(m(), k2, mr_packed(), kr(), sr(),
                                           /*m_idx_start=*/0, input_f32.data(),
                                           /*lhs_stride=*/k2 * sizeof(float),
                                           input_qp8.data());

    std::generate(b.begin(), b.end(), std::ref(w8rng));
    // std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(packed_w.begin(), packed_w.end(), 0);

    // RHS packing.
    struct xnn_qs8_qc4w_packing_params params;
    params.input_zero_point = 1;
    params.kernel_zero_point = b_zero_point();
    pack(/*flags=*/0, &gemm_config, k2, n(),
         /*groups=*/1, /*k_stride=*/k2,
         /*accumulator_init=*/nullptr,
         /*weights=*/b.data(),
         /*int_extra_data0_fn=*/nullptr,
         /*extra_data0=*/nullptr,
         /*extra_data0_size=*/0,
         /*init_extra_data1_fn=*/
         nullptr,
         /*extra_data1=*/kernel_scale.data(),
         /*extra_data1_size=*/sizeof(float),
         /*packed_weights_ptr=*/packed_w.data(), &params);

    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k2; k_index++) {
          const size_t nb_index = (n_index * k2 + k_index) / 2;
          const int32_t bv =
              static_cast<int32_t>((k_index % 2 == 0)
                                       ? (b[nb_index] & UINT8_C(0xF))
                                       : (b[nb_index] >> 4)) -
              b_zero_point();
          c_ref[m_index * n() + n_index] +=
              xnn_x8_packq_f32qp8_get_dequantized(m_index, k_index,
                                                  input_qp8.data(), k2,
                                                  mr_packed(), kr(), sr()) *
              static_cast<int32_t>(bv);
        }
        c_ref[m_index * n() + n_index] *= kernel_scale[n_index];
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    const float accumulated_min =
        *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max =
        *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min =
        qmin() == std::numeric_limits<uint8_t>::min()
            ? -std::numeric_limits<float>::infinity()
            : accumulated_min + (accumulated_max - accumulated_min) / 255.0f *
                                    static_cast<float>(qmin());
    const float c_max =
        qmax() == std::numeric_limits<uint8_t>::max()
            ? std::numeric_limits<float>::infinity()
            : accumulated_max - (accumulated_max - accumulated_min) / 255.0f *
                                    static_cast<float>(255 - qmax());

    // Prepare parameters.
    xnn_f32_minmax_params minmax_params;
    init_minmax_params(&minmax_params, c_min, c_max);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], c_max), c_min);
      }
    }

    gemm(m(), n(), k2, input_qp8.data(), packed_w.data(), c.data(),
         cm_stride() * sizeof(float), sizeof(float), &minmax_params);

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux
        // AArch64.
        const float tolerance =
            std::max(1.1e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f);
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
                    c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << " (accumulator = " << acc[i * n() + j] << "), optimized = "
            << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k2
            << ", cn_stride = " << cn_stride()
            << ", cm_stride = " << cm_stride();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_qs8_gemm_minmax_ukernel_fn gemm,
  xnn_init_qs8_conv_minmax_params_fn init_params,
  xnn_pack_qs8_gemm_fn pack,
  xnn_qs8_requantize_fn requantize) const
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));
  auto w8rng = std::bind(
    std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<int8_t> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<int8_t> b(n() * k());
  std::vector<int32_t> bias(n());
  std::vector<int8_t, AlignedAllocator<int8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k() + packed_n() * sizeof(int32_t) / sizeof(int8_t));
  std::vector<int8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<int8_t> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(i8rng));
    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = { int8_t(a_zero_point() - 0x80) };
    void* const packed_data = packed_w.data();
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_data, /*extra_bytes=*/0, &packing_params);

    // Compute 32-bit results and output quantization arguments.
    std::fill(acc.begin(), acc.end(), 0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          acc[m_index * n() + n_index] +=
              (int32_t(a[m_index * a_stride() + k_index]) - int32_t(a_zero_point() - 0x80)) *
              int32_t(b[n_index * k() + k_index]);
        }
        acc[m_index * n() + n_index] += bias[n_index];
      }
    }

    const int32_t accumulated_min = *std::min_element(acc.cbegin(), acc.cend());
    const int32_t accumulated_max = *std::max_element(acc.cbegin(), acc.cend());
    const double c_scale = uint32_t(accumulated_max - accumulated_min) >= 256 ? double(uint32_t(accumulated_max - accumulated_min)) / 255.0 : 1.00001;
    const int8_t c_zero_point = int8_t(std::max(std::min(
      lrint(-0.5 - 0.5 * double(accumulated_min + accumulated_max) / c_scale),
      long(std::numeric_limits<int8_t>::max())), long(std::numeric_limits<int8_t>::min())));

    const float requantization_scale = 1.0f / float(c_scale);
    union xnn_qs8_conv_minmax_params quantization_params;
    init_params(&quantization_params,
      requantization_scale, c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

    gemm(
      m(), n(), k(),
      a.data(), a_stride() * sizeof(int8_t),
      packed_data,
      c.data(), cm_stride() * sizeof(int8_t), cn_stride() * sizeof(int8_t),
      &quantization_params);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = requantize(
          acc[m_index * n() + n_index], requantization_scale, c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_LE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmax()) - 0x80);
        EXPECT_GE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmin()) - 0x80);
        EXPECT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(c_ref[i * n() + j]))
            << "at " << i << ", " << j << ": reference = " << int32_t(c_ref[i * n() + j])
            << " (accumulator = " << acc[i * n() + j]
            << "), optimized = " << int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]) << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", requantization scale = " << requantization_scale << ", output zero point = " << int32_t(c_zero_point);
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_qs8_igemm_minmax_ukernel_fn igemm,
  xnn_init_qs8_conv_minmax_params_fn init_params,
  xnn_pack_qs8_igemm_fn pack,
  xnn_qs8_requantize_fn requantize) const
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));
  auto w8rng = std::bind(
    std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<int8_t> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<int8_t> b(n() * ks() * k());
  std::vector<int8_t, AlignedAllocator<int8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(ks() * packed_n() * packed_k() + packed_n() * sizeof(int32_t) / sizeof(int8_t));
  std::vector<int32_t> bias(n());
  std::vector<int8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<int8_t> c_ref(m() * n());
  std::vector<int8_t> junk(k() + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<const int8_t*> im2col(mr() * ks());

  std::fill(junk.begin(), junk.end(), 0xA5);

  {//for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(i8rng));
    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = { int8_t(a_zero_point() - 0x80) };
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0, &packing_params);

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = a.data() + a_stride() * m_index - a_offset();
      }
    }
    std::shuffle(im2col.begin(), im2col.end(), rng);
    if (zero_index() != SIZE_MAX) {
      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        im2col[ks_index * mr() + zero_index()] = a.data();
      }
    }
    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = m(); m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = junk.data();
      }
    }

    // Compute 32-bit results and output quantization arguments.
    std::fill(acc.begin(), acc.end(), 0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            if (im2col[ks_index * mr() + m_index] == a.data()) {
              acc[m_index * n() + n_index] +=
                (int32_t(im2col[ks_index * mr() + m_index][k_index]) - int32_t(a_zero_point() - 0x80)) *
                int32_t(b[(n_index * ks() + ks_index) * k() + k_index]);
            } else {
              acc[m_index * n() + n_index] +=
                (int32_t(im2col[ks_index * mr() + m_index][k_index + a_offset()]) - int32_t(a_zero_point() - 0x80)) *
                int32_t(b[(n_index * ks() + ks_index) * k() + k_index]);
            }
          }
        }
        acc[m_index * n() + n_index] += bias[n_index];
      }
    }

    const int32_t accumulated_min = *std::min_element(acc.cbegin(), acc.cend());
    const int32_t accumulated_max = *std::max_element(acc.cbegin(), acc.cend());
    const double c_scale = uint32_t(accumulated_max - accumulated_min) >= 256 ? double(uint32_t(accumulated_max - accumulated_min)) / 255.0 : 1.00001;
    const uint8_t c_zero_point = uint8_t(std::max(std::min(
      lrint(-0.5 - 0.5 * double(accumulated_min + accumulated_max) / c_scale),
      long(std::numeric_limits<int8_t>::max())), long(std::numeric_limits<int8_t>::min())));

    const float requantization_scale = 1.0f / float(c_scale);
    union xnn_qs8_conv_minmax_params quantization_params;
    init_params(&quantization_params,
      requantization_scale, c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

    const int8_t* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : nullptr;

    igemm(
      m(), n(), k(), ks() * mr() * sizeof(void*),
      im2col.data(), packed_w.data(),
      c.data(), cm_stride() * sizeof(int8_t), cn_stride() * sizeof(int8_t),
      a_offset() * sizeof(uint8_t), zero_pointer,
      &quantization_params);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = requantize(
          acc[m_index * n() + n_index], requantization_scale, c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_LE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmax()) - 0x80);
        EXPECT_GE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmin()) - 0x80);
        EXPECT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(c_ref[i * n() + j]))
            << "at " << i << ", " << j << ": reference = " << uint32_t(c_ref[i * n() + j])
            << " (accumulator = " << acc[i * n() + j]
            << "), optimized = " << (uint32_t) c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", requantization scale = " << requantization_scale << ", output zero point = " << int32_t(c_zero_point);
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_bf16_gemm_minmax_ukernel_fn gemm_minmax,
  xnn_init_bf16_minmax_params_fn init_params,
  xnn_pack_f16_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.5f, 1.0f), std::ref(rng));

  std::vector<uint16_t> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<uint16_t> b(n() * k());
  std::vector<uint16_t, AlignedAllocator<uint16_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k() + packed_n());
  std::vector<uint16_t> bias(n());
  std::vector<uint16_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&] { return fp32_to_bits(f32rng(rng)) >> 16; });
    std::generate(b.begin(), b.end(), [&] { return fp32_to_bits(f32rng(rng)) >> 16; });
    std::generate(bias.begin(), bias.end(), [&] { return fp32_to_bits(f32rng(rng)) >> 16; });
    std::fill(c.begin(), c.end(), UINT16_C(0x7FC0) /* NaN */);
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0, /*params=*/nullptr);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = fp32_from_bits(uint32_t(bias[n_index]) << 16);
        for (size_t k_index = 0; k_index < k(); k_index++) {
          EXPECT_LE(n(), packed_n());
          EXPECT_LT(m_index * n() + n_index, c_ref.size());
          EXPECT_LT(m_index * k() + k_index, a.size());
          c_ref[m_index * n() + n_index] +=
            fp32_from_bits(uint32_t(a[m_index * a_stride() + k_index]) << 16) *
            fp32_from_bits(uint32_t(b[n_index * k() + k_index]) << 16);
        }
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min = fp32_from_bits(fp32_to_bits(accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin())) & UINT32_C(0xFFFF0000));
    const float c_max = fp32_from_bits(fp32_to_bits(accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax())) & UINT32_C(0xFFFF0000));

    // Prepare parameters.
    xnn_bf16_minmax_params params;
    init_params(&params,
      fp32_to_bits(c_min) >> 16,
      fp32_to_bits(c_max) >> 16);

    for (float& c_value : c_ref) {
      c_value = std::max(std::min(c_value, c_max), c_min);
    }

    gemm_minmax(m(), n(), k() * sizeof(uint16_t),
      a.data(), a_stride() * sizeof(uint16_t),
      packed_w.data(),
      c.data(), cm_stride() * sizeof(uint16_t), cn_stride() * sizeof(uint16_t),
      &params);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_NEAR(
            fp32_from_bits(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]) << 16),
            c_ref[i * n() + j],
            std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 3.0e-2f))
          << "at " << i << ", " << j << ": Mr x Nr x Kr = " << mr() << " x " << nr()
          << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_f16_gemm_minmax_ukernel_fn gemm_minmax,
  xnn_init_f16_minmax_params_fn init_params,
  xnn_pack_f16_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  std::vector<uint16_t> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<uint16_t> b(n() * k());
  std::vector<uint16_t, AlignedAllocator<uint16_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k() + packed_n());
  std::vector<uint16_t> bias(n());
  std::vector<uint16_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(f16rng));
    std::generate(b.begin(), b.end(), std::ref(f16rng));
    std::generate(bias.begin(), bias.end(), std::ref(f16rng));
    std::fill(c.begin(), c.end(), UINT16_C(0x7E00) /* NaN */);
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0, /*params=*/nullptr);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          EXPECT_LE(n(), packed_n());
          EXPECT_LT(m_index * n() + n_index, c_ref.size());
          EXPECT_LT(m_index * k() + k_index, a.size());
          c_ref[m_index * n() + n_index] +=
            fp16_ieee_to_fp32_value(a[m_index * a_stride() + k_index]) *
            fp16_ieee_to_fp32_value(b[n_index * k() + k_index]);
        }
        c_ref[m_index * n() + n_index] += fp16_ieee_to_fp32_value(bias[n_index]);
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin())));
    const float c_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax())));

    // Prepare parameters.
    xnn_f16_minmax_params params;
    init_params(&params,
      fp16_ieee_from_fp32_value(c_min),
      fp16_ieee_from_fp32_value(c_max));

    for (float& c_value : c_ref) {
      c_value = std::max(std::min(c_value, c_max), c_min);
    }

    gemm_minmax(m(), n(), k() * sizeof(uint16_t),
      a.data(), a_stride() * sizeof(uint16_t),
      packed_w.data(),
      c.data(), cm_stride() * sizeof(uint16_t), cn_stride() * sizeof(uint16_t),
      &params);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_NEAR(fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), c_ref[i * n() + j], std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 1.0e-2f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]) << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_f16_igemm_minmax_ukernel_fn igemm_minmax,
  xnn_init_f16_minmax_params_fn init_params,
  xnn_pack_f16_igemm_fn pack) const
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  std::vector<uint16_t> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<uint16_t> b(n() * ks() * k());
  std::vector<uint16_t, AlignedAllocator<uint16_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(ks() * packed_k() * packed_n() + packed_n());
  std::vector<uint16_t> bias(n());
  std::vector<uint16_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());
  std::vector<uint16_t> junk(k() + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<const uint16_t*> im2col(mr() * ks());
  std::fill(junk.begin(), junk.end(), UINT16_C(0x7E00) /* NaN */);

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(f16rng));
    std::generate(b.begin(), b.end(), std::ref(f16rng));
    std::generate(bias.begin(), bias.end(), std::ref(f16rng));
    std::fill(c.begin(), c.end(), UINT16_C(0x7E00) /* NaN */);
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0, /*params=*/nullptr);

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = a.data() + a_stride() * m_index - a_offset();
      }
    }
    std::shuffle(im2col.begin(), im2col.end(), rng);
    if (zero_index() != SIZE_MAX) {
      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        im2col[ks_index * mr() + zero_index()] = a.data();
      }
    }
    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = m(); m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = junk.data();
      }
    }

    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            EXPECT_LT(ks_index * mr() + m_index, im2col.size());
            EXPECT_LT(k_index, k());
            EXPECT_LT(k_index, a_stride());
            if (im2col[ks_index * mr() + m_index] == a.data()) {
              c_ref[m_index * n() + n_index] +=
                fp16_ieee_to_fp32_value(im2col[ks_index * mr() + m_index][k_index]) *
                fp16_ieee_to_fp32_value(b[(n_index * ks() + ks_index) * k() + k_index]);
            } else {
              c_ref[m_index * n() + n_index] +=
                fp16_ieee_to_fp32_value(im2col[ks_index * mr() + m_index][k_index + a_offset()]) *
                fp16_ieee_to_fp32_value(b[(n_index * ks() + ks_index) * k() + k_index]);
            }
          }
        }
        c_ref[m_index * n() + n_index] += fp16_ieee_to_fp32_value(bias[n_index]);
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + (accumulated_max - accumulated_min) / 255.0f * uint16_t(qmin())));
    const float c_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - (accumulated_max - accumulated_min) / 255.0f * uint16_t(255 - qmax())));
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = std::min(c_ref[m_index * n() + n_index], c_max);
        c_ref[m_index * n() + n_index] = std::max(c_ref[m_index * n() + n_index], c_min);
      }
    }

    // Prepare parameters.
    xnn_f16_minmax_params params;
    init_params(&params,
      fp16_ieee_from_fp32_value(c_min),
      fp16_ieee_from_fp32_value(c_max));

    for (float& c_value : c_ref) {
      c_value = std::max(std::min(c_value, c_max), c_min);
    }

    const uint16_t* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : nullptr;

    igemm_minmax(
      m(), n(), k() * sizeof(uint16_t), ks() * mr() * sizeof(void*),
      reinterpret_cast<const void**>(im2col.data()), packed_w.data(),
      c.data(), cm_stride() * sizeof(uint16_t), cn_stride() * sizeof(uint16_t),
      a_offset() * sizeof(uint16_t), zero_pointer,
      &params);

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_LE(fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), c_max)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]) << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
        EXPECT_GE(fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), c_min)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]) << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
        EXPECT_NEAR(fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), c_ref[i * n() + j], std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 1.0e-2f))
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]) << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_f32_ppmm_minmax_ukernel_fn ppmm_minmax,
  xnn_init_f32_minmax_params_fn init_params,
  xnn_pack_f32_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  std::vector<float> a(packed_k() * mr());
  std::vector<float> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float, AlignedAllocator<float, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k() + packed_n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0, /*params=*/nullptr);

    for (size_t i = m(); i < mr(); i++) {
      for (size_t l = 0; l < k(); l++) {
        a[l * mr() + i] = a[l * mr() + m() - 1];
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        for (size_t l = 0; l < k(); l++) {
          c_ref[i * n() + j] +=
            a[l * mr() + i] *
            b[j * k() + l];
        }
        c_ref[i * n() + j] += bias[j];
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min = accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
    const float c_max = accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, c_min, c_max);

    for (float& c_value : c_ref) {
      c_value = std::max(std::min(c_value, c_max), c_min);
    }

    ppmm_minmax(m(), n(), k() * sizeof(float),
      a.data(), packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
      &params);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
    xnn_f32_gemm_ukernel_fn gemm,
    xnn_pack_f32_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float, AlignedAllocator<float, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k() + packed_n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0, /*params=*/nullptr);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          c_ref[m_index * n() + n_index] +=
            a[m_index * a_stride() + k_index] *
            b[n_index * k() + k_index];
        }
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    gemm(m(), n(), k() * sizeof(float),
      a.data(), a_stride() * sizeof(float),
      packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
      nullptr);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_f32_gemm_relu_ukernel_fn gemm_relu,
  xnn_pack_f32_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float, AlignedAllocator<float, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k() + packed_n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0, /*params=*/nullptr);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          c_ref[m_index * n() + n_index] +=
            a[m_index * a_stride() + k_index] *
            b[n_index * k() + k_index];
        }
        c_ref[m_index * n() + n_index] = std::max(0.0f, c_ref[m_index * n() + n_index] + bias[n_index]);
      }
    }

    gemm_relu(m(), n(), k() * sizeof(float),
      a.data(), a_stride() * sizeof(float),
      packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
      nullptr);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], 0.0f)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        EXPECT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_f32_gemm_minmax_ukernel_fn gemm_minmax,
  xnn_init_f32_minmax_params_fn init_params,
  xnn_pack_f32_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float, AlignedAllocator<float, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k() + packed_n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0, /*params=*/nullptr);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          c_ref[m_index * n() + n_index] +=
            a[m_index * a_stride() + k_index] *
            b[n_index * k() + k_index];
        }
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min =
        qmin() == std::numeric_limits<uint8_t>::min() ? -std::numeric_limits<float>::infinity()
                    : accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
    const float c_max =
        qmax() == std::numeric_limits<uint8_t>::max() ? +std::numeric_limits<float>::infinity()
                      : accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, c_min, c_max);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], c_max), c_min);
      }
    }

    gemm_minmax(m(), n(), k() * sizeof(float),
      a.data(), a_stride() * sizeof(float),
      packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
      &params);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_LE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_max)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        EXPECT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_min)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        EXPECT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
    xnn_f32_gemm_goi_minmax_ukernel_fn gemm_minmax,
    xnn_init_f32_minmax_params_fn init_params) const
{
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * k());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          c_ref[m_index * n() + n_index] +=
              a[m_index * a_stride() + k_index] *
              b[n_index * k() + k_index];
        }
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min =
        qmin() == std::numeric_limits<uint8_t>::min() ? -std::numeric_limits<float>::infinity()
        : accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
    const float c_max =
        qmax() == std::numeric_limits<uint8_t>::max() ? +std::numeric_limits<float>::infinity()
        : accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, c_min, c_max);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], c_max), c_min);
      }
    }

    gemm_minmax(m(), n(), k() * sizeof(float),
                a.data(), a_stride() * sizeof(float),
                b.data(),
                c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
                &params);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_LE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_max)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        EXPECT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_min)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        EXPECT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}


void GemmMicrokernelTester::Test(
  xnn_f32_qc4w_gemm_minmax_ukernel_fn gemm_minmax,
  xnn_init_f32_qc4w_minmax_params_fn init_params,
  xnn_pack_f32_qc4w_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);
  std::uniform_int_distribution<int32_t> i8dist(-1, std::numeric_limits<uint8_t>::max());

  const size_t k_stride = (k() + 1) / 2;
  const size_t packed_k_bytes = (packed_k() + 1) / 2;
  std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<uint8_t> b(n() * k_stride);
  std::vector<float> bias(n());
  std::vector<float> scale(n());
  std::vector<uint8_t, AlignedAllocator<uint8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k_bytes + packed_n() * sizeof(float) * 2);
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<double> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    std::fill(packed_w.begin(), packed_w.end(), 0);

    std::generate(b.begin(), b.end(), [&]() { return i8dist(rng); });
    // For odd k ensure last nibble is padded with 0
    if (k() & 1) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        const size_t nb_index = n_index * k_stride + k_stride - 1;
        b[nb_index] &= 0xF;
      }
    }

    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), nr() * sizeof(float), /*params=*/nullptr);

    // Fill in packed scale
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k_bytes + (sizeof(float) + sizeof(float))),
      nr() * (ks() * packed_k_bytes + (sizeof(float) + sizeof(float))),
      0,
      scale.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k_bytes + sizeof(float))));

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = double(bias[n_index]);
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          const size_t nb_index = n_index * k_stride + k_index / 2;
          const int16_t bv = int16_t((k_index % 2 == 0) ? (b[nb_index] & UINT8_C(0xF)) : (b[nb_index] >> 4u)) - b_zero_point();

          c_ref[m_index * n() + n_index] +=
            double(a[m_index * a_stride() + k_index]) *
            double(bv);
        }
        c_ref[m_index * n() + n_index] *= double(scale[n_index]);
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min =
        qmin() == std::numeric_limits<uint8_t>::min() ? -std::numeric_limits<float>::infinity()
                    : accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
    const float c_max =
        qmax() == std::numeric_limits<uint8_t>::max() ? +std::numeric_limits<float>::infinity()
                      : accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

    // Prepare parameters.
    xnn_f32_qc4w_minmax_params params;
    init_params(&params, c_min, c_max, b_zero_point());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], double(c_max)), double(c_min));
      }
    }

    gemm_minmax(m(), n(), k() * sizeof(float),  // Note KC measured in bytes of input
      a.data(), a_stride() * sizeof(float),
      packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
      &params);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_LE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_max)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        EXPECT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_min)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        EXPECT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::max(1.0e-5, std::abs(c_ref[i * n() + j]) * 1.0e-6))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
    xnn_f32_qc8w_gemm_ukernel_fn gemm,
    xnn_pack_f32_qs8w_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_int_distribution<int32_t> i8dist(-1, std::numeric_limits<int8_t>::max());

  std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<int8_t> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float> scale(n());
  std::vector<int8_t, AlignedAllocator<int8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k() + packed_n() * sizeof(float) * 2);
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return i8dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    std::fill(packed_w.begin(), packed_w.end(), 0);

    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), nr() * sizeof(float), /*params=*/nullptr);

    // Fill in packed scale
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k() * sizeof(int8_t) + (sizeof(float) + sizeof(float))),
      nr() * (ks() * packed_k() * sizeof(int8_t) + (sizeof(float) + sizeof(float))),
      0,
      scale.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(float))));

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          c_ref[m_index * n() + n_index] +=
            a[m_index * a_stride() + k_index] *
            (float) b[n_index * k() + k_index];
        }
        c_ref[m_index * n() + n_index] += bias[n_index];
        c_ref[m_index * n() + n_index] *= scale[n_index];
      }
    }

    gemm(m(), n(), k() * sizeof(float),  // Note KC measured in bytes of input
      a.data(), a_stride() * sizeof(float),
      packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
       nullptr);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            0.1f);
      }
    }

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_f32_qc8w_gemm_relu_ukernel_fn gemm_relu,
  xnn_pack_f32_qs8w_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_int_distribution<int32_t> i8dist(-1, std::numeric_limits<int8_t>::max());

  std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<int8_t> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float> scale(n());
  std::vector<int8_t, AlignedAllocator<int8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k() + packed_n() * sizeof(float) * 2);
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return i8dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    std::fill(packed_w.begin(), packed_w.end(), 0);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), nr() * sizeof(float), /*params=*/nullptr);

    // Fill in packed scale
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k() * sizeof(int8_t) + (sizeof(float) + sizeof(float))),
      nr() * (ks() * packed_k() * sizeof(int8_t) + (sizeof(float) + sizeof(float))),
      0,
      scale.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(float))));

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          c_ref[m_index * n() + n_index] +=
            a[m_index * a_stride() + k_index] *
            (float) b[n_index * k() + k_index];
        }
        c_ref[m_index * n() + n_index] = std::max(0.0f, c_ref[m_index * n() + n_index] + bias[n_index]);
        c_ref[m_index * n() + n_index] *= scale[n_index];
      }
    }

    gemm_relu(m(), n(), k() * sizeof(float),  // Note KC measured in bytes of input
      a.data(), a_stride() * sizeof(float),
      packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
       nullptr);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], 0.0f)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        EXPECT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_f32_qc8w_gemm_minmax_ukernel_fn gemm_minmax,
  xnn_init_f32_minmax_params_fn init_params,
  xnn_pack_f32_qs8w_gemm_fn pack) const
{
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);
  std::uniform_int_distribution<int32_t> i8dist(-1, std::numeric_limits<int8_t>::max());

  std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<int8_t> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float> scale(n());
  std::vector<int8_t, AlignedAllocator<int8_t, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k() + packed_n() * sizeof(float) * 2);
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<double> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return i8dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    std::fill(packed_w.begin(), packed_w.end(), 0);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), nr() * sizeof(float), /*params=*/nullptr);

    // Fill in packed scale
    xnn_init_qs8_qc8w_scale_fp32_params(
      n(), nr(), nr(),
      nr() * (ks() * packed_k() * sizeof(int8_t) + (sizeof(float) + sizeof(float))),
      nr() * (ks() * packed_k() * sizeof(int8_t) + (sizeof(float) + sizeof(float))),
      0,
      scale.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(float))));

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = double(bias[n_index]);
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          c_ref[m_index * n() + n_index] +=
            double(a[m_index * a_stride() + k_index]) *
            double(b[n_index * k() + k_index]);
        }
        c_ref[m_index * n() + n_index] *= double(scale[n_index]);
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min =
        qmin() == std::numeric_limits<uint8_t>::min() ? -std::numeric_limits<float>::infinity()
                    : accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
    const float c_max =
        qmax() == std::numeric_limits<uint8_t>::max() ? +std::numeric_limits<float>::infinity()
                      : accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, c_min, c_max);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], double(c_max)), double(c_min));
      }
    }

    gemm_minmax(m(), n(), k() * sizeof(float),  // Note KC measured in bytes of input
      a.data(), a_stride() * sizeof(float),
      packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
      &params);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_LE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_max)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        EXPECT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_min)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        EXPECT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::max(1.0e-5, std::abs(c_ref[i * n() + j]) * 1.0e-6))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_f32_gemminc_minmax_ukernel_fn gemminc,
  xnn_init_f32_minmax_params_fn init_params,
  xnn_pack_f32_gemminc_fn pack) const
{
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float, AlignedAllocator<float, XNN_ALLOCATION_ALIGNMENT>> packed_w(packed_n() * packed_k());  // no packed_n()
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());
  std::vector<float, AlignedAllocator<float, XNN_ALLOCATION_ALIGNMENT>> acc(mr() * packed_n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    std::generate(acc.begin(), acc.end(), [&]() { return f32dist(rng); });

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
      b.data(), packed_w.data(), /*params=*/nullptr);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          c_ref[m_index * n() + n_index] +=
            a[m_index * a_stride() + k_index] *
            b[n_index * k() + k_index];
        }
        c_ref[m_index * n() + n_index] += acc[n_index / nr() * nr() * mr() + m_index % mr() * nr() + n_index % nr()];
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min = accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
    const float c_max = accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, c_min, c_max);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], c_max), c_min);
      }
    }

    gemminc(m(), n(), k() * sizeof(float),
      a.data(), a_stride() * sizeof(float),
      packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
      acc.data(),
      &params);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_LE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_max)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        EXPECT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_min)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        EXPECT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
    xnn_f32_igemm_ukernel_fn igemm,
    xnn_pack_f32_igemm_fn pack) const
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  std::vector<float> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * ks() * k());
  std::vector<float, AlignedAllocator<float, XNN_ALLOCATION_ALIGNMENT>> packed_w(ks() * packed_k() * packed_n() + packed_n());
  std::vector<float> bias(n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());
  std::vector<float> junk(k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<const float*> im2col(mr() * ks());
  std::fill(junk.begin(), junk.end(), nanf(""));

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0, /*params=*/nullptr);

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = a.data() + a_stride() * m_index - a_offset();
      }
    }
    std::shuffle(im2col.begin(), im2col.end(), rng);
    if (zero_index() != SIZE_MAX) {
      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        im2col[ks_index * mr() + zero_index()] = a.data();
      }
    }
    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = m(); m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = junk.data();
      }
    }

    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            ASSERT_LT(ks_index * mr() + m_index, im2col.size());
            ASSERT_LT(k_index, k());
            ASSERT_LT(k_index, a_stride());
            if (im2col[ks_index * mr() + m_index] == a.data()) {
              c_ref[m_index * n() + n_index] +=
                (im2col[ks_index * mr() + m_index][k_index]) *
                (b[(n_index * ks() + ks_index) * k() + k_index]);
            } else {
              c_ref[m_index * n() + n_index] +=
                (im2col[ks_index * mr() + m_index][k_index + a_offset()]) *
                (b[(n_index * ks() + ks_index) * k() + k_index]);
            }
          }
        }
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    const float* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : nullptr;

    igemm(
      m(), n(), k() * sizeof(float), ks() * mr() * sizeof(void*),
      im2col.data(), packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
      a_offset() * sizeof(float), zero_pointer,
      nullptr);

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_f32_igemm_relu_ukernel_fn igemm_relu,
  xnn_pack_f32_igemm_fn pack) const
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  std::vector<float> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * ks() * k());
  std::vector<float, AlignedAllocator<float, XNN_ALLOCATION_ALIGNMENT>> packed_w(ks() * packed_k() * packed_n() + packed_n());
  std::vector<float> bias(n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());
  std::vector<float> junk(k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<const float*> im2col(mr() * ks());
  std::fill(junk.begin(), junk.end(), nanf(""));

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0, /*params=*/nullptr);

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = a.data() + a_stride() * m_index - a_offset();
      }
    }
    std::shuffle(im2col.begin(), im2col.end(), rng);
    if (zero_index() != SIZE_MAX) {
      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        im2col[ks_index * mr() + zero_index()] = a.data();
      }
    }
    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = m(); m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = junk.data();
      }
    }

    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            ASSERT_LT(ks_index * mr() + m_index, im2col.size());
            ASSERT_LT(k_index, k());
            ASSERT_LT(k_index, a_stride());
            if (im2col[ks_index * mr() + m_index] == a.data()) {
              c_ref[m_index * n() + n_index] +=
                (im2col[ks_index * mr() + m_index][k_index]) *
                (b[(n_index * ks() + ks_index) * k() + k_index]);
            } else {
              c_ref[m_index * n() + n_index] +=
                (im2col[ks_index * mr() + m_index][k_index + a_offset()]) *
                (b[(n_index * ks() + ks_index) * k() + k_index]);
            }
          }
        }
        c_ref[m_index * n() + n_index] = std::max(0.0f, bias[n_index] + c_ref[m_index * n() + n_index]);
      }
    }

    const float* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : nullptr;

    igemm_relu(
      m(), n(), k() * sizeof(float), ks() * mr() * sizeof(void*),
      im2col.data(), packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
      a_offset() * sizeof(float), zero_pointer,
      nullptr);

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], 0.0f)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
        EXPECT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_f32_igemm_minmax_ukernel_fn igemm_minmax,
  xnn_init_f32_minmax_params_fn init_params,
  xnn_pack_f32_igemm_fn pack) const
{
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  std::vector<float> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * ks() * k());
  std::vector<float, AlignedAllocator<float, XNN_ALLOCATION_ALIGNMENT>> packed_w(ks() * packed_k() * packed_n() + packed_n());
  std::vector<float> bias(n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());
  std::vector<float> junk(k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<const float*> im2col(mr() * ks());
  std::fill(junk.begin(), junk.end(), nanf(""));

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0, /*params=*/nullptr);

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = a.data() + a_stride() * m_index - a_offset();
      }
    }
    std::shuffle(im2col.begin(), im2col.end(), rng);
    if (zero_index() != SIZE_MAX) {
      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        im2col[ks_index * mr() + zero_index()] = a.data();
      }
    }
    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = m(); m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] = junk.data();
      }
    }

    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            ASSERT_LT(ks_index * mr() + m_index, im2col.size());
            ASSERT_LT(k_index, k());
            ASSERT_LT(k_index, a_stride());
            if (im2col[ks_index * mr() + m_index] == a.data()) {
              c_ref[m_index * n() + n_index] +=
                (im2col[ks_index * mr() + m_index][k_index]) *
                (b[(n_index * ks() + ks_index) * k() + k_index]);
            } else {
              c_ref[m_index * n() + n_index] +=
                (im2col[ks_index * mr() + m_index][k_index + a_offset()]) *
                (b[(n_index * ks() + ks_index) * k() + k_index]);
            }
          }
        }
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
    const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
    const float c_min = accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
    const float c_max = accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = std::min(c_ref[m_index * n() + n_index], c_max);
        c_ref[m_index * n() + n_index] = std::max(c_ref[m_index * n() + n_index], c_min);
      }
    }

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, c_min, c_max);

    const float* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : nullptr;

    igemm_minmax(
      m(), n(), k() * sizeof(float), ks() * mr() * sizeof(void*),
      im2col.data(), packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
      a_offset() * sizeof(float), zero_pointer,
      &params);

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        EXPECT_LE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_max)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
        EXPECT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_min)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
        EXPECT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
      }
    }
  }
}

