#include "test/gemm-microkernel-tester.h"

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

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packq.h"
#include "src/xnnpack/quantization.h"
#include "src/xnnpack/requantization.h"
#include "test/replicable_random_device.h"

constexpr int kIterations = 1;

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

void GemmMicrokernelTester::Test(xnn_qd8_f16_qc8w_igemm_ukernel_fn igemm,
                                 xnn_init_f16_minmax_params_fn init_params,
                                 xnn_pack_qs8_igemm_fn pack) const {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f),
                          std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f),
                            std::ref(rng));
  auto w8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));

  xnnpack::Buffer<float> input(mr() * k());
  xnnpack::Buffer<int8_t> a((mr() - 1) * a_stride() + k() +
                            XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<xnn_qd8_quantization_params> quantization_params(
      1 + XNN_EXTRA_QUANTIZATION_PARAMS);
  xnnpack::Buffer<int8_t> b(n() * ks() * k());
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float> kernel_scale(n());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      ks() * packed_n() * packed_k() +
      packed_n() * (sizeof(int32_t) + sizeof(float) * 2));
  xnnpack::Buffer<xnn_float16> c((mr() - 1) * cm_stride() +
                                 ((n() - 1) / nr()) * nr() + (n() - 1) % nr() +
                                 1);
  xnnpack::Buffer<float> c_ref(m() * n(), 0);
  xnnpack::Buffer<int8_t> junk(k() + XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<const int8_t*> im2col(mr() * ks());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    const auto minmax =
        std::minmax_element(input.begin(), input.begin() + mr() * k());
    float inv_scale;
    quantization_params[0] = xnn_f32_qd8_asymmetric_quantization_params(
        *minmax.first, *minmax.second, &inv_scale);
    for (size_t i = 0; i < mr(); ++i) {
      const float* input_ptr = &input[i * k()];
      for (size_t j = 0; j < k(); ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(
            scaled_input, float(std::numeric_limits<int8_t>::max() -
                                quantization_params[0].zero_point));
        scaled_input = std::max<float>(
            scaled_input, float(std::numeric_limits<int8_t>::min() -
                                quantization_params[0].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) +
                                       long(quantization_params[0].zero_point));
      }
    }
    std::generate(b.begin(), b.end(), std::ref(w8rng));

    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));

    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_packing_params packing_params = {/*input_zero_point=*/1};
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(), b.data(), /*bias=*/nullptr,
         /*scale=*/nullptr, packed_w.data(), 2 * sizeof(float) * nr(),
         &packing_params);
    // Fill in packed kernel scale
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(),
        nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
        kernel_scale.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(float))));

    // Fill in packed bias
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(),
        nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
        bias.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() *
                    (ks() * packed_k() * sizeof(int8_t) + 2 * sizeof(float))));

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] =
            a.data() + a_stride() * m_index - a_offset();
      }
    }
    std::shuffle(im2col.begin(), im2col.end(), rng);
    const size_t k_stride = round_up_po2(k(), kr() * sr());
    int32_t zp = quantization_params[0].zero_point;
    if (unsigned_inputs()) {
      zp += 128;
    }
    xnnpack::Buffer<int8_t> zero_points(k_stride + XNN_EXTRA_BYTES, zp);
    const int8_t* zero_sentinel = (const int8_t*)&packing_params;
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
                  (int32_t(im2col[ks_index * mr() + m_index]
                                 [k_index + a_offset()]) -
                   quantization_params[0].zero_point) *
                  int32_t(b[(n_index * ks() + ks_index) * k() + k_index]);
            }
          }
        }
        c_ref[m_index * n() + n_index] *=
            quantization_params[0].inv_scale * kernel_scale[n_index];
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    // Prepare parameters.
    xnn_f16_minmax_params params;
    init_params(&params, static_cast<xnn_float16>(min()),
                static_cast<xnn_float16>(max()));

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], max()), min());
      }
    }

    if (unsigned_inputs()) {
      // Some architectures require that the input be unsigned.
      // Adjust the zero point and flip the sign of the input to mimic adding
      // 128 to the input with correct overflow behaviour.
      for (int i = 0; i < quantization_params.size(); ++i) {
        quantization_params[i].zero_point += 128;
      }
      for (int i = 0; i < a.size(); ++i) {
        a[i] ^= 0x80;
      }
    }
    igemm(m(), n(), k(), ks() * mr() * sizeof(void*), im2col.data(),
          static_cast<const void*>(packed_w.data()), c.data(),
          cm_stride() * sizeof(xnn_float16), nr() * sizeof(xnn_float16),
          a_offset() * sizeof(uint8_t), zero_sentinel, zero_data, &params,
          quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux
        // AArch64.
        const float tolerance =
            std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 1.0e-2f);
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << "), optimized = "
            << (float)c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_qd8_f32_qc8w_igemm_ukernel_fn igemm,
                                 xnn_init_f32_minmax_params_fn init_params,
                                 xnn_pack_qs8_igemm_fn pack) const {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f),
                          std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f),
                            std::ref(rng));
  auto w8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));

  xnnpack::Buffer<float> input(mr() * k());
  xnnpack::Buffer<int8_t> a((mr() - 1) * a_stride() + k() +
                            XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<xnn_qd8_quantization_params> quantization_params(
      1 + XNN_EXTRA_QUANTIZATION_PARAMS);
  xnnpack::Buffer<int8_t> b(n() * ks() * k());
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float> kernel_scale(n());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      ks() * packed_n() * packed_k() +
      packed_n() * (sizeof(int32_t) + sizeof(float) * 2));
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<float> c_ref(m() * n(), 0);
  xnnpack::Buffer<int8_t> junk(k() + XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<const int8_t*> im2col(mr() * ks());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    const auto minmax =
        std::minmax_element(input.begin(), input.begin() + mr() * k());
    float inv_scale;
    quantization_params[0] = xnn_f32_qd8_asymmetric_quantization_params(
        *minmax.first, *minmax.second, &inv_scale);
    for (size_t i = 0; i < mr(); ++i) {
      const float* input_ptr = &input[i * k()];
      for (size_t j = 0; j < k(); ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(
            scaled_input, float(std::numeric_limits<int8_t>::max() -
                                quantization_params[0].zero_point));
        scaled_input = std::max<float>(
            scaled_input, float(std::numeric_limits<int8_t>::min() -
                                quantization_params[0].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) +
                                       long(quantization_params[0].zero_point));
      }
    }
    std::generate(b.begin(), b.end(), std::ref(w8rng));

    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));

    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_packing_params packing_params = {/*input_zero_point=*/1};
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(), b.data(), /*bias=*/nullptr,
         /*scale=*/nullptr, packed_w.data(), 2 * sizeof(float) * nr(),
         &packing_params);
    // Fill in packed kernel scale
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(),
        nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
        kernel_scale.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(float))));

    // Fill in packed bias
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(),
        nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
        bias.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() *
                    (ks() * packed_k() * sizeof(int8_t) + 2 * sizeof(float))));

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] =
            a.data() + a_stride() * m_index - a_offset();
      }
    }
    std::shuffle(im2col.begin(), im2col.end(), rng);
    const size_t k_stride = round_up_po2(k(), kr() * sr());
    int32_t zp = quantization_params[0].zero_point;
    if (unsigned_inputs()) {
      zp += 128;
    }
    xnnpack::Buffer<int8_t> zero_points(k_stride + XNN_EXTRA_BYTES, zp);
    const int8_t* zero_sentinel = (const int8_t*)&packing_params;
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
                  (int32_t(im2col[ks_index * mr() + m_index]
                                 [k_index + a_offset()]) -
                   quantization_params[0].zero_point) *
                  int32_t(b[(n_index * ks() + ks_index) * k() + k_index]);
            }
          }
        }
        c_ref[m_index * n() + n_index] *=
            quantization_params[0].inv_scale * kernel_scale[n_index];
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, min(), max());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], max()), min());
      }
    }

    if (unsigned_inputs()) {
      // Some architectures require that the input be unsigned.
      // Adjust the zero point and flip the sign of the input to mimic adding
      // 128 to the input with correct overflow behaviour.
      for (int i = 0; i < quantization_params.size(); ++i) {
        quantization_params[i].zero_point += 128;
      }
      for (int i = 0; i < a.size(); ++i) {
        a[i] ^= 0x80;
      }
    }
    igemm(m(), n(), k(), ks() * mr() * sizeof(void*), im2col.data(),
          static_cast<const void*>(packed_w.data()), c.data(),
          cm_stride() * sizeof(float), nr() * sizeof(float),
          a_offset() * sizeof(uint8_t), zero_sentinel, zero_data, &params,
          quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux
        // AArch64.
        const float tolerance =
            std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f);
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << "), optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_qu8_gemm_minmax_ukernel_fn gemm,
                                 xnn_init_qu8_conv_minmax_params_fn init_params,
                                 xnn_pack_qu8_gemm_fn pack,
                                 xnn_qu8_requantize_fn requantize) const {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000),
                          std::ref(rng));

  xnnpack::Buffer<uint8_t> a((m() - 1) * a_stride() + k() +
                             XNN_EXTRA_BYTES / sizeof(uint8_t));
  xnnpack::Buffer<uint8_t> b(n() * k());
  xnnpack::Buffer<int32_t> bias(n());
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k() + packed_n() * sizeof(int32_t) / sizeof(uint8_t));
  xnnpack::Buffer<uint8_t> c((mr() - 1) * cm_stride() +
                             ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<int32_t> acc(m() * n());
  xnnpack::Buffer<uint8_t> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    xnnpack::fill_uniform_random_bits(a.data(), a.size(), rng);
    xnnpack::fill_uniform_random_bits(b.data(), b.size(), rng);
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));

    std::fill(packed_w.begin(), packed_w.end(), b_zero_point());
    const xnn_qu8_packing_params packing_params = {a_zero_point(),
                                                   b_zero_point()};
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0,
         &packing_params);

    // Compute 32-bit results and output quantization arguments.
    std::fill(acc.begin(), acc.end(), 0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          acc[m_index * n() + n_index] +=
              (int32_t(a[m_index * a_stride() + k_index]) -
               int32_t(a_zero_point())) *
              (int32_t(b[n_index * k() + k_index]) - int32_t(b_zero_point()));
        }
        acc[m_index * n() + n_index] += bias[n_index];
      }
    }

    const int32_t accumulated_min = *std::min_element(acc.cbegin(), acc.cend());
    const int32_t accumulated_max = *std::max_element(acc.cbegin(), acc.cend());
    const double c_scale =
        uint32_t(accumulated_max - accumulated_min) >= 256
            ? double(uint32_t(accumulated_max - accumulated_min)) / 255.0
            : 1.00001;
    const uint8_t c_zero_point = uint8_t(std::max(
        std::min(lrint(127.5 - 0.5 * double(accumulated_min + accumulated_max) /
                                   c_scale),
                 long(std::numeric_limits<uint8_t>::max())),
        long(std::numeric_limits<uint8_t>::min())));

    const float requantization_scale = 1.0f / float(c_scale);
    union xnn_qu8_conv_minmax_params quantization_params;
    init_params(&quantization_params, b_zero_point(), requantization_scale,
                c_zero_point, qmin(), qmax());

    gemm(m(), n(), k(), a.data(), a_stride() * sizeof(uint8_t), packed_w.data(),
         c.data(), cm_stride() * sizeof(uint8_t), nr() * sizeof(uint8_t),
         &quantization_params);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            requantize(acc[m_index * n() + n_index], requantization_scale,
                       c_zero_point, qmin(), qmax());
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(uint32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  uint32_t(qmax()));
        ASSERT_GE(uint32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  uint32_t(qmin()));
        ASSERT_EQ(uint32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  uint32_t(c_ref[i * n() + j]))
            << "at " << i << ", " << j
            << ": reference = " << (uint32_t)c_ref[i * n() + j]
            << " (accumulator = " << acc[i * n() + j] << "), optimized = "
            << (uint32_t)c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", requantization scale = " << requantization_scale
            << ", output zero point = " << int32_t(c_zero_point);
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_qu8_igemm_minmax_ukernel_fn igemm,
                                 xnn_init_qu8_conv_minmax_params_fn init_params,
                                 xnn_pack_qu8_igemm_fn pack,
                                 xnn_qu8_requantize_fn requantize) {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000),
                          std::ref(rng));

  xnnpack::Buffer<uint8_t> a((mr() - 1) * a_stride() + k() +
                             XNN_EXTRA_BYTES / sizeof(uint8_t));
  xnnpack::Buffer<uint8_t> b(n() * ks() * k());
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      ks() * packed_n() * packed_k() +
      packed_n() * sizeof(int32_t) / sizeof(uint8_t));
  xnnpack::Buffer<int32_t> bias(n());
  xnnpack::Buffer<uint8_t> c((mr() - 1) * cm_stride() +
                             ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<int32_t> acc(m() * n());
  xnnpack::Buffer<uint8_t> c_ref(m() * n());
  xnnpack::Buffer<uint8_t> junk(k() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  xnnpack::Buffer<const uint8_t*> im2col(mr() * ks());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    xnnpack::fill_uniform_random_bits(a.data(), a.size(), rng);
    xnnpack::fill_uniform_random_bits(b.data(), b.size(), rng);
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));

    std::fill(packed_w.begin(), packed_w.end(), b_zero_point());
    const xnn_qu8_packing_params packing_params = {a_zero_point(),
                                                   b_zero_point()};
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0,
         &packing_params);

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] =
            a.data() + a_stride() * m_index - a_offset();
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
                  (int32_t(im2col[ks_index * mr() + m_index][k_index]) -
                   int32_t(a_zero_point())) *
                  (int32_t(b[(n_index * ks() + ks_index) * k() + k_index]) -
                   int32_t(b_zero_point()));
            } else {
              acc[m_index * n() + n_index] +=
                  (int32_t(im2col[ks_index * mr() + m_index]
                                 [k_index + a_offset()]) -
                   int32_t(a_zero_point())) *
                  (int32_t(b[(n_index * ks() + ks_index) * k() + k_index]) -
                   int32_t(b_zero_point()));
            }
          }
        }
        acc[m_index * n() + n_index] += bias[n_index];
      }
    }

    const int32_t accumulated_min = *std::min_element(acc.cbegin(), acc.cend());
    const int32_t accumulated_max = *std::max_element(acc.cbegin(), acc.cend());
    const double c_scale =
        uint32_t(accumulated_max - accumulated_min) >= 256
            ? double(uint32_t(accumulated_max - accumulated_min)) / 255.0
            : 1.00001;
    const uint8_t c_zero_point = uint8_t(std::max(
        std::min(lrint(127.5 - 0.5 * double(accumulated_min + accumulated_max) /
                                   c_scale),
                 long(std::numeric_limits<uint8_t>::max())),
        long(std::numeric_limits<uint8_t>::min())));

    const float requantization_scale = 1.0f / float(c_scale);
    union xnn_qu8_conv_minmax_params quantization_params;
    init_params(&quantization_params, b_zero_point(), requantization_scale,
                c_zero_point, qmin(), qmax());

    const uint8_t* zero_pointer =
        (zero_index() != SIZE_MAX) ? a.data() : nullptr;

    igemm(m(), n(), k(), ks() * mr() * sizeof(void*), im2col.data(),
          packed_w.data(), c.data(), cm_stride() * sizeof(uint8_t),
          nr() * sizeof(uint8_t), a_offset() * sizeof(uint8_t), zero_pointer,
          &quantization_params);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            requantize(acc[m_index * n() + n_index], requantization_scale,
                       c_zero_point, qmin(), qmax());
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(uint32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  uint32_t(qmax()));
        ASSERT_GE(uint32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  uint32_t(qmin()));
        ASSERT_EQ(uint32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  uint32_t(c_ref[i * n() + j]))
            << "at " << i << ", " << j
            << ": reference = " << uint32_t(c_ref[i * n() + j])
            << " (accumulator = " << acc[i * n() + j] << "), optimized = "
            << (uint32_t)c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", requantization scale = " << requantization_scale
            << ", output zero point = " << int32_t(c_zero_point);
      }
    }
  }
}

void GemmMicrokernelTester::Test(
    xnn_qs8_qc8w_gemm_minmax_ukernel_fn gemm,
    xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
    xnn_pack_qs8_gemm_fn pack, xnn_qs8_requantize_fn requantize) const {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000),
                          std::ref(rng));
  auto w8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));

  xnnpack::Buffer<int8_t> a((m() - 1) * a_stride() + k() +
                            XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<int8_t> b(n() * k());
  xnnpack::Buffer<int32_t> bias(n());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k() +
      packed_n() * (sizeof(int32_t) + sizeof(float)) / sizeof(int8_t));
  xnnpack::Buffer<int8_t> c((mr() - 1) * cm_stride() +
                            ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<int32_t> acc(m() * n());
  xnnpack::Buffer<float> scale(n());
  xnnpack::Buffer<int8_t> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    xnnpack::fill_uniform_random_bits(a.data(), a.size(), rng);
    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = {
        int8_t(a_zero_point() - 0x80)};
    void* const packed_data = packed_w.data();
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_data, nr() * sizeof(float), &packing_params);

    // Compute 32-bit results and output quantization arguments.
    std::fill(acc.begin(), acc.end(), 0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          acc[m_index * n() + n_index] +=
              (int32_t(a[m_index * a_stride() + k_index]) -
               int32_t(a_zero_point() - 0x80)) *
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
        accumulated_min =
            std::min(accumulated_min, acc[m_index * n() + n_index]);
        accumulated_max =
            std::max(accumulated_max, acc[m_index * n() + n_index]);
      }
      const uint32_t accumulated_range =
          uint32_t(accumulated_max - accumulated_min);
      const float c_scale = accumulated_range >= 256
                                ? double(accumulated_range) / 255.0
                                : 1.00001;
      scale[n_index] = 1.0f / c_scale;
    }

    const size_t type_size = sizeof(int8_t);
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(),
        nr() * (packed_k() * type_size + (sizeof(int32_t) + sizeof(float))),
        scale.data(),
        (void*)((uintptr_t)packed_data +
                nr() * (packed_k() * type_size + sizeof(int32_t))));

    union xnn_qs8_qc8w_conv_minmax_params minmax_params;
    init_params(&minmax_params, c_zero_point, int8_t(qmin() - 0x80),
                int8_t(qmax() - 0x80));

    gemm(m(), n(), k(), a.data(), a_stride() * sizeof(int8_t), packed_data,
         c.data(), cm_stride() * sizeof(int8_t), nr() * sizeof(int8_t),
         &minmax_params);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = requantize(
            acc[m_index * n() + n_index], scale[n_index], c_zero_point,
            int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(int32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  int32_t(qmax()) - 0x80);
        ASSERT_GE(int32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  int32_t(qmin()) - 0x80);
        ASSERT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  int32_t(c_ref[i * n() + j]))
            << "at " << i << ", " << j
            << ": reference = " << int32_t(c_ref[i * n() + j])
            << " (accumulator = " << acc[i * n() + j] << "), optimized = "
            << int32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()])
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", requantization scale = " << scale[j]
            << ", output zero point = " << int32_t(c_zero_point);
      }
    }
  }
}

void GemmMicrokernelTester::Test(
    xnn_qs8_qc8w_igemm_minmax_ukernel_fn igemm,
    xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
    xnn_pack_qs8_igemm_fn pack, xnn_qs8_requantize_fn requantize) const {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000),
                          std::ref(rng));
  auto w8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));

  xnnpack::Buffer<int8_t> a((mr() - 1) * a_stride() + k() +
                            XNN_EXTRA_BYTES / sizeof(uint8_t));
  xnnpack::Buffer<int8_t> b(n() * ks() * k());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      ks() * packed_n() * packed_k() +
      packed_n() * (sizeof(int32_t) + sizeof(float)) / sizeof(int8_t));
  xnnpack::Buffer<int32_t> bias(n());
  xnnpack::Buffer<int8_t> c((mr() - 1) * cm_stride() +
                            ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<int32_t> acc(m() * n());
  xnnpack::Buffer<float> scale(n());
  xnnpack::Buffer<int8_t> c_ref(m() * n());
  xnnpack::Buffer<int8_t> junk(k() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  xnnpack::Buffer<const int8_t*> im2col(mr() * ks());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    xnnpack::fill_uniform_random_bits(a.data(), a.size(), rng);
    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = {
        int8_t(a_zero_point() - 0x80)};
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(), nr() * sizeof(float),
         &packing_params);

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] =
            a.data() + a_stride() * m_index - a_offset();
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
                  (int32_t(im2col[ks_index * mr() + m_index][k_index]) -
                   int32_t(a_zero_point() - 0x80)) *
                  int32_t(b[(n_index * ks() + ks_index) * k() + k_index]);
            } else {
              acc[m_index * n() + n_index] +=
                  (int32_t(im2col[ks_index * mr() + m_index]
                                 [k_index + a_offset()]) -
                   int32_t(a_zero_point() - 0x80)) *
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
        accumulated_min =
            std::min(accumulated_min, acc[m_index * n() + n_index]);
        accumulated_max =
            std::max(accumulated_max, acc[m_index * n() + n_index]);
      }
      const uint32_t accumulated_range =
          uint32_t(accumulated_max - accumulated_min);
      const float c_scale = accumulated_range >= 256
                                ? double(accumulated_range) / 255.0
                                : 1.00001;
      scale[n_index] = 1.0f / c_scale;
    }

    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(),
        nr() * (ks() * packed_k() * sizeof(int8_t) +
                (sizeof(int32_t) + sizeof(float))),
        scale.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(int32_t))));

    union xnn_qs8_qc8w_conv_minmax_params minmax_params;
    init_params(&minmax_params, c_zero_point, int8_t(qmin() - 0x80),
                int8_t(qmax() - 0x80));

    const int8_t* zero_pointer =
        (zero_index() != SIZE_MAX) ? a.data() : nullptr;

    igemm(m(), n(), k(), ks() * mr() * sizeof(void*), im2col.data(),
          packed_w.data(), c.data(), cm_stride() * sizeof(int8_t),
          nr() * sizeof(int8_t), a_offset() * sizeof(uint8_t), zero_pointer,
          &minmax_params);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = requantize(
            acc[m_index * n() + n_index], scale[n_index], c_zero_point,
            int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(int32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  int32_t(qmax()) - 0x80);
        ASSERT_GE(int32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  int32_t(qmin()) - 0x80);
        ASSERT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  int32_t(c_ref[i * n() + j]))
            << "at " << i << ", " << j
            << ": reference = " << uint32_t(c_ref[i * n() + j])
            << " (accumulator = " << acc[i * n() + j] << "), optimized = "
            << (uint32_t)c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", requantization scale = " << scale[j]
            << ", output zero point = " << int32_t(c_zero_point);
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_qd8_f16_qc8w_gemm_ukernel_fn gemm,
                                 xnn_init_f16_minmax_params_fn init_params,
                                 xnn_pack_qs8_gemm_fn pack) const {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f),
                          std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f),
                            std::ref(rng));
  auto w8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));

  xnnpack::Buffer<float> input(m() * k());
  xnnpack::Buffer<int8_t> a((m() - 1) * a_stride() + k() +
                            XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<xnn_qd8_quantization_params> quantization_params(mr());
  xnnpack::Buffer<int8_t> b(n() * k());
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float> kernel_scale(n());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k() +
      packed_n() * (sizeof(int32_t) + sizeof(float) * 2));
  xnnpack::Buffer<xnn_float16> c((mr() - 1) * cm_stride() +
                                 ((n() - 1) / nr()) * nr() + (n() - 1) % nr() +
                                 1);
  xnnpack::Buffer<float> c_ref(m() * n(), 0);

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    for (size_t i = 0; i < m(); ++i) {
      const float* input_ptr = &input[i * k()];
      const auto minmax = std::minmax_element(input_ptr, input_ptr + k());
      float inv_scale;
      quantization_params[i] = xnn_f32_qd8_asymmetric_quantization_params(
          *minmax.first, *minmax.second, &inv_scale);
      for (size_t j = 0; j < k(); ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(
            scaled_input, float(std::numeric_limits<int8_t>::max() -
                                quantization_params[i].zero_point));
        scaled_input = std::max<float>(
            scaled_input, float(std::numeric_limits<int8_t>::min() -
                                quantization_params[i].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) +
                                       long(quantization_params[i].zero_point));
      }
    }
    for (size_t i = m(); i < mr(); ++i) {
      quantization_params[i].zero_point =
          quantization_params[m() - 1].zero_point;
      quantization_params[i].inv_scale = quantization_params[m() - 1].inv_scale;
    }
    std::generate(b.begin(), b.end(), std::ref(w8rng));

    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));

    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_packing_params packing_params = {/*input_zero_point=*/1};
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(), b.data(), /*bias=*/nullptr,
         /*scale=*/nullptr, packed_w.data(), 2 * sizeof(float) * nr(),
         &packing_params);
    // Fill in packed kernel scale
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(),
        nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
        kernel_scale.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(float))));

    // Fill in packed bias
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(),
        nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
        bias.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() *
                    (ks() * packed_k() * sizeof(int8_t) + 2 * sizeof(float))));

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
        c_ref[m_index * n() + n_index] -=
            (quantization_params[m_index].zero_point * ksum);
        c_ref[m_index * n() + n_index] *=
            quantization_params[m_index].inv_scale * kernel_scale[n_index];
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    // Prepare parameters.
    xnn_f16_minmax_params params;
    init_params(&params, static_cast<xnn_float16>(min()),
                static_cast<xnn_float16>(max()));

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], max()), min());
      }
    }

    if (unsigned_inputs()) {
      // Some architectures require that the input be unsigned.
      // Adjust the zero point and flip the sign of the input to mimic adding
      // 128 to the input with correct overflow behaviour.
      for (int i = 0; i < quantization_params.size(); ++i) {
        quantization_params[i].zero_point += 128;
      }
      for (int i = 0; i < a.size(); ++i) {
        a[i] ^= 0x80;
      }
    }
    gemm(m(), n(), k(), a.data(), a_stride() * sizeof(int8_t),
         static_cast<const void*>(packed_w.data()), c.data(),
         cm_stride() * sizeof(xnn_float16), nr() * sizeof(xnn_float16), &params,
         quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux
        // AArch64.
        const float tolerance =
            std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 1.0e-2f);
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << "), optimized = "
            << (float)c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_qd8_f32_qc8w_gemm_ukernel_fn gemm,
                                 xnn_init_f32_minmax_params_fn init_params,
                                 xnn_pack_qs8_gemm_fn pack) const {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f),
                          std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f),
                            std::ref(rng));
  auto w8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));

  xnnpack::Buffer<float> input(m() * k());
  xnnpack::Buffer<int8_t> a((m() - 1) * a_stride() + k() +
                            XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<xnn_qd8_quantization_params> quantization_params(mr());
  xnnpack::Buffer<int8_t> b(n() * k());
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float> kernel_scale(n());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k() +
      packed_n() * (sizeof(int32_t) + sizeof(float) * 2));
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<int32_t> acc(m() * n());
  xnnpack::Buffer<float> c_ref(m() * n(), 0);

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    for (size_t i = 0; i < m(); ++i) {
      const float* input_ptr = &input[i * k()];
      const auto minmax = std::minmax_element(input_ptr, input_ptr + k());
      float inv_scale;
      quantization_params[i] = xnn_f32_qd8_asymmetric_quantization_params(
          *minmax.first, *minmax.second, &inv_scale);
      for (size_t j = 0; j < k(); ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(
            scaled_input, float(std::numeric_limits<int8_t>::max() -
                                quantization_params[i].zero_point));
        scaled_input = std::max<float>(
            scaled_input, float(std::numeric_limits<int8_t>::min() -
                                quantization_params[i].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) +
                                       long(quantization_params[i].zero_point));
      }
    }
    for (size_t i = m(); i < mr(); ++i) {
      quantization_params[i].zero_point =
          quantization_params[m() - 1].zero_point;
      quantization_params[i].inv_scale = quantization_params[m() - 1].inv_scale;
    }
    std::generate(b.begin(), b.end(), std::ref(w8rng));

    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));

    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_packing_params packing_params = {/*input_zero_point=*/1};
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(), b.data(), /*bias=*/nullptr,
         /*scale=*/nullptr, packed_w.data(), 2 * sizeof(float) * nr(),
         &packing_params);
    // Fill in packed kernel scale
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(),
        nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
        kernel_scale.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(float))));

    // Fill in packed bias
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(),
        nr() * (ks() * packed_k() * sizeof(int8_t) + 3 * sizeof(float)),
        bias.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() *
                    (ks() * packed_k() * sizeof(int8_t) + 2 * sizeof(float))));

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
        c_ref[m_index * n() + n_index] -=
            (quantization_params[m_index].zero_point * ksum);
        c_ref[m_index * n() + n_index] *=
            quantization_params[m_index].inv_scale * kernel_scale[n_index];
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, min(), max());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], max()), min());
      }
    }

    if (unsigned_inputs()) {
      // Some architectures require that the input be unsigned.
      // Adjust the zero point and flip the sign of the input to mimic adding
      // 128 to the input with correct overflow behaviour.
      for (int i = 0; i < quantization_params.size(); ++i) {
        quantization_params[i].zero_point += 128;
      }
      for (int i = 0; i < a.size(); ++i) {
        a[i] ^= 0x80;
      }
    }
    gemm(m(), n(), k(), a.data(), a_stride() * sizeof(int8_t),
         static_cast<const void*>(packed_w.data()), c.data(),
         cm_stride() * sizeof(float), nr() * sizeof(float), &params,
         quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux
        // AArch64.
        const float tolerance =
            std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f);
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << " (accumulator = " << acc[i * n() + j] << "), optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_qd8_f16_qc4w_gemm_ukernel_fn gemm,
                                 xnn_init_f16_qc4w_minmax_params_fn init_params,
                                 xnn_pack_qs8_qc4w_gemm_fn pack) const {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f),
                          std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f),
                            std::ref(rng));
  auto w8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             0, std::numeric_limits<uint8_t>::max()),
                         std::ref(rng));

  const size_t planes = 2;  // 4 bit is 2 planes - low nibbles and high nibbles
  const size_t k2 = round_up_po2(k(), 2);  // tester assumes byte aligned rows
  const size_t packed_k2 =
      round_up_po2(k(), kr() * sr() * planes);  // 2 blocks for nibbles
  const size_t packed_k_bytes = (packed_k2 + 1) / 2;

  xnnpack::Buffer<float> input(m() * k2);
  xnnpack::Buffer<int8_t> a((m() - 1) * a_stride() + k2 +
                            XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<xnn_qd8_quantization_params> quantization_params(mr());
  xnnpack::Buffer<uint8_t> b(n() * k2 / 2);
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float> kernel_scale(n());
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k_bytes +
      packed_n() * (sizeof(int32_t) + sizeof(float) * 2));
  xnnpack::Buffer<xnn_float16> c((mr() - 1) * cm_stride() +
                                 ((n() - 1) / nr()) * nr() + (n() - 1) % nr() +
                                 1);
  xnnpack::Buffer<int32_t> acc(m() * n());
  xnnpack::Buffer<float> c_ref(m() * n(), 0.0f);

  {  // for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    for (size_t i = 0; i < m(); ++i) {
      const float* input_ptr = &input[i * k2];
      const auto minmax = std::minmax_element(input_ptr, input_ptr + k2);
      float inv_scale;
      quantization_params[i] = xnn_f32_qd8_asymmetric_quantization_params(
          *minmax.first, *minmax.second, &inv_scale);
      for (size_t j = 0; j < k2; ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(
            scaled_input, float(std::numeric_limits<int8_t>::max() -
                                quantization_params[i].zero_point));
        scaled_input = std::max<float>(
            scaled_input, float(std::numeric_limits<int8_t>::min() -
                                quantization_params[i].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) +
                                       long(quantization_params[i].zero_point));
      }
    }
    for (size_t i = m(); i < mr(); ++i) {
      quantization_params[i].zero_point =
          quantization_params[m() - 1].zero_point;
      quantization_params[i].inv_scale = quantization_params[m() - 1].inv_scale;
    }

    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));
    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_qc4w_packing_params packing_params = {/*input_zero_point=*/1,
                                                        b_zero_point()};
    pack(/*g=*/1, n(), k2, nr(), kr(), sr(), b.data(), /*bias=*/nullptr,
         /*scale=*/nullptr, packed_w.data(), 2 * sizeof(float) * nr(),
         &packing_params);
    // Fill in packed kernel scale
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(), nr() * (ks() * packed_k_bytes + 3 * sizeof(float)),
        kernel_scale.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() * (ks() * packed_k_bytes + sizeof(float))));

    // Fill in packed bias
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(), nr() * (ks() * packed_k_bytes + 3 * sizeof(float)),
        bias.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() * (ks() * packed_k_bytes + 2 * sizeof(float))));

    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        int32_t ksum = 0;
        for (size_t k_index = 0; k_index < k2; k_index++) {
          const size_t nb_index = (n_index * k2 + k_index) / 2;
          const int32_t bv =
              int32_t((k_index % 2 == 0) ? (b[nb_index] & UINT8_C(0xF))
                                         : (b[nb_index] >> 4)) -
              b_zero_point();
          ksum += bv;
          c_ref[m_index * n() + n_index] +=
              int32_t(a[m_index * a_stride() + k_index]) * int32_t(bv);
        }
        c_ref[m_index * n() + n_index] -=
            (quantization_params[m_index].zero_point * ksum);
        c_ref[m_index * n() + n_index] *=
            quantization_params[m_index].inv_scale * kernel_scale[n_index];
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    // Prepare parameters.
    xnn_f16_qc4w_minmax_params params;
    init_params(&params, static_cast<xnn_float16>(min()),
                static_cast<xnn_float16>(max()), 8);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], max()), min());
      }
    }

    if (unsigned_inputs()) {
      // Some architectures require that the input be unsigned.
      // Adjust the zero point and flip the sign of the input to mimic adding
      // 128 to the input with correct overflow behaviour.
      for (int i = 0; i < quantization_params.size(); ++i) {
        quantization_params[i].zero_point += 128;
      }
      for (int i = 0; i < a.size(); ++i) {
        a[i] ^= 0x80;
      }
    }
    gemm(m(), n(), k2, a.data(), a_stride() * sizeof(int8_t),
         static_cast<const void*>(packed_w.data()), c.data(),
         cm_stride() * sizeof(xnn_float16), nr() * sizeof(xnn_float16), &params,
         quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux
        // AArch64.
        const float tolerance =
            std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 1.0e-2f);
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << " (accumulator = " << acc[i * n() + j] << "), optimized = "
            << (float)c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_qd8_f16_qb4w_gemm_ukernel_fn gemm,
                                 xnn_init_f16_qb4w_minmax_params_fn init_params,
                                 xnn_pack_qs8_qb4w_gemm_fn pack) const {
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f),
                          std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f),
                            std::ref(rng));
  auto w8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             0, std::numeric_limits<uint8_t>::max()),
                         std::ref(rng));

  const size_t planes = 2;  // 4 bit is 2 planes - low nibbles and high nibbles
  const size_t k2 = round_up_po2(k(), 2);  // tester assumes byte aligned rows

  const size_t packed_k2 =
      round_up_po2(k(), kr() * sr() * planes);  // 2 blocks for nibbles
  const size_t packed_k_bytes = (packed_k2 + 1) / 2;
  const size_t num_blocks = packed_k2 / bl();

  xnnpack::Buffer<float> input(m() * k2);
  xnnpack::Buffer<int8_t> a((m() - 1) * a_stride() + k2 +
                            XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<xnn_qd8_quantization_params> quantization_params(mr());
  xnnpack::Buffer<uint8_t> b(n() * k2 / 2);
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<xnn_bfloat16> kernel_scale2d(n() * k2 / bl());
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k_bytes +
      /* vksum */ packed_n() * sizeof(float) +
      /* scales */ packed_n() * num_blocks * sizeof(xnn_bfloat16) +
      /* bias */ packed_n() * sizeof(float));

  xnnpack::Buffer<xnn_float16> c((mr() - 1) * cm_stride() +
                                 ((n() - 1) / nr()) * nr() + (n() - 1) % nr() +
                                 1);
  xnnpack::Buffer<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    for (size_t i = 0; i < m(); ++i) {
      const float* input_ptr = &input[i * k2];
      const auto minmax = std::minmax_element(input_ptr, input_ptr + k2);
      float inv_scale;
      quantization_params[i] = xnn_f32_qd8_asymmetric_quantization_params(
          *minmax.first, *minmax.second, &inv_scale);
      for (size_t j = 0; j < k2; ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(
            scaled_input, float(std::numeric_limits<int8_t>::max() -
                                quantization_params[i].zero_point));
        scaled_input = std::max<float>(
            scaled_input, float(std::numeric_limits<int8_t>::min() -
                                quantization_params[i].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) +
                                       long(quantization_params[i].zero_point));
      }
    }
    for (size_t i = m(); i < mr(); ++i) {
      quantization_params[i].zero_point =
          quantization_params[m() - 1].zero_point;
      quantization_params[i].inv_scale = quantization_params[m() - 1].inv_scale;
    }

    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale2d.begin(), kernel_scale2d.end(),
                  [&]() { return scalerng(); });

    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_qc4w_packing_params packing_params = {/*input_zero_point=*/1,
                                                        b_zero_point()};

    pack(/*g=*/1, n(), k2, nr(), kr(), sr(), bl(), b.data(), /*bias=*/nullptr,
         /*scale=*/kernel_scale2d.data(), packed_w.data(),
         sizeof(xnn_float16) * nr(), sizeof(float) * nr(), &packing_params);

    // Fill in packed kernel scale
    size_t stride =
        nr() *
        (packed_k_bytes + /* scales= */ num_blocks * sizeof(xnn_float16) +
         /* ksum= */ sizeof(float) + /* bias= */ sizeof(float));
    size_t block_stride = (bl() / 2 + sizeof(xnn_float16)) * nr();
    size_t start_offset = nr() * (packed_k_bytes / num_blocks + sizeof(float));
    uintptr_t start = (uintptr_t)packed_w.data() + start_offset;
    xnn_init_blockwise_scale_bf16_params(n(), nr(), stride,
                                         /*num_blocks=*/num_blocks,
                                         /*block_stride=*/block_stride,
                                         kernel_scale2d.data(), (void*)start);

    start = (uintptr_t)packed_w.data() + stride - sizeof(float) * nr();
    xnn_init_qs8_qc8w_scale_fp32_params(n(), nr(), stride, bias.data(),
                                        (void*)start);

    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        float kfsum = 0.0;
        for (size_t bl_index = 0; bl_index < num_blocks; ++bl_index) {
          int32_t ksum = 0;
          int32_t c_ref_acc = 0;
          for (size_t kr_index = 0; kr_index < bl(); kr_index++) {
            const size_t k_index = bl_index * bl() + kr_index;
            const size_t nb_index = (n_index * k2 + k_index) / 2;
            const int32_t bv =
                int32_t((k_index % 2 == 0) ? (b[nb_index] & UINT8_C(0xF))
                                           : (b[nb_index] >> 4)) -
                b_zero_point();
            ksum += bv;
            c_ref_acc +=
                int32_t(a[m_index * a_stride() + k_index]) * int32_t(bv);
          }
          size_t scale_index = n_index * num_blocks + bl_index;
          float scale = kernel_scale2d[scale_index];
          c_ref[m_index * n() + n_index] += c_ref_acc * scale;
          kfsum += scale * ksum;
        }
        c_ref[m_index * n() + n_index] -=
            (quantization_params[m_index].zero_point * kfsum);
        c_ref[m_index * n() + n_index] *=
            quantization_params[m_index].inv_scale;
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    // Prepare parameters.
    xnn_f16_qb4w_minmax_params params;
    init_params(&params, static_cast<xnn_float16>(min()),
                static_cast<xnn_float16>(max()), 8, bl());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], max()), min());
      }
    }

    gemm(m(), n(), k2, a.data(), a_stride() * sizeof(int8_t),
         static_cast<const void*>(packed_w.data()), c.data(),
         cm_stride() * sizeof(xnn_float16), nr() * sizeof(xnn_float16), &params,
         quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux
        // AArch64.
        const float tolerance =
            std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 1.0e-3f);
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << (float)c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k2;
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_qd8_f32_qc4w_gemm_ukernel_fn gemm,
                                 xnn_init_f32_qc4w_minmax_params_fn init_params,
                                 xnn_pack_qs8_qc4w_gemm_fn pack) const {
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
  const size_t packed_k2 =
      round_up_po2(k(), kr() * sr() * planes());  // 2 blocks for nibbles
  const size_t packed_k_bytes = (packed_k2 + 1) / 2;

  xnnpack::Buffer<float> input(m() * k2);
  xnnpack::Buffer<int8_t> a((m() - 1) * a_stride() + k2 +
                            XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<xnn_qd8_quantization_params> quantization_params(mr());
  xnnpack::Buffer<uint8_t> b(n() * k2 / 2);
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float> kernel_scale(n());
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k_bytes +
      packed_n() * (sizeof(int32_t) + sizeof(float) * 2));
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<int32_t> acc(m() * n());
  xnnpack::Buffer<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    for (size_t i = 0; i < m(); ++i) {
      const float* input_ptr = &input[i * k2];
      const auto minmax = std::minmax_element(input_ptr, input_ptr + k2);
      float inv_scale;
      quantization_params[i] = xnn_f32_qd8_asymmetric_quantization_params(
          *minmax.first, *minmax.second, &inv_scale);
      for (size_t j = 0; j < k2; ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(
            scaled_input, float(std::numeric_limits<int8_t>::max() -
                                quantization_params[i].zero_point));
        scaled_input = std::max<float>(
            scaled_input, float(std::numeric_limits<int8_t>::min() -
                                quantization_params[i].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) +
                                       long(quantization_params[i].zero_point));
      }
    }
    for (size_t i = m(); i < mr(); ++i) {
      quantization_params[i].zero_point =
          quantization_params[m() - 1].zero_point;
      quantization_params[i].inv_scale = quantization_params[m() - 1].inv_scale;
    }

    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));

    std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));
    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_qc4w_packing_params packing_params = {/*input_zero_point=*/1,
                                                        b_zero_point()};
    pack(/*g=*/1, n(), k2, nr(), kr(), sr(), b.data(), /*bias=*/nullptr,
         /*scale=*/nullptr, packed_w.data(), 2 * sizeof(float) * nr(),
         &packing_params);
    // Fill in packed kernel scale
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(), nr() * (ks() * packed_k_bytes + 3 * sizeof(float)),
        kernel_scale.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() * (ks() * packed_k_bytes + sizeof(float))));

    // Fill in packed bias
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(), nr() * (ks() * packed_k_bytes + 3 * sizeof(float)),
        bias.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() * (ks() * packed_k_bytes + 2 * sizeof(float))));

    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        int32_t ksum = 0;
        for (size_t k_index = 0; k_index < k2; k_index++) {
          const size_t nb_index = (n_index * k2 + k_index) / 2;
          const int32_t bv =
              int32_t((k_index % 2 == 0) ? (b[nb_index] & UINT8_C(0xF))
                                         : (b[nb_index] >> 4)) -
              b_zero_point();
          ksum += bv;
          c_ref[m_index * n() + n_index] +=
              int32_t(a[m_index * a_stride() + k_index]) * int32_t(bv);
        }
        c_ref[m_index * n() + n_index] -=
            (quantization_params[m_index].zero_point * ksum);
        c_ref[m_index * n() + n_index] *=
            quantization_params[m_index].inv_scale * kernel_scale[n_index];
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    // Prepare parameters.
    xnn_f32_qc4w_minmax_params params;
    init_params(&params, min(), max(), 8);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], max()), min());
      }
    }

    if (unsigned_inputs()) {
      // Some architectures require that the input be unsigned.
      // Adjust the zero point and flip the sign of the input to mimic adding
      // 128 to the input with correct overflow behaviour.
      for (int i = 0; i < quantization_params.size(); ++i) {
        quantization_params[i].zero_point += 128;
      }
      for (int i = 0; i < a.size(); ++i) {
        a[i] ^= 0x80;
      }
    }
    gemm(m(), n(), k2, a.data(), a_stride() * sizeof(int8_t),
         static_cast<const void*>(packed_w.data()), c.data(),
         cm_stride() * sizeof(float), nr() * sizeof(float), &params,
         quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux
        // AArch64.
        const float tolerance =
            std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f);
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << " (accumulator = " << acc[i * n() + j] << "), optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k2;
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_qd8_f32_qb4w_gemm_ukernel_fn gemm,
                                 xnn_init_f32_qb4w_minmax_params_fn init_params,
                                 xnn_pack_qs8_qb4w_gemm_fn pack) const {
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f),
                          std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f),
                            std::ref(rng));
  auto w8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             0, std::numeric_limits<uint8_t>::max()),
                         std::ref(rng));

  const size_t planes = 2;  // 4 bit is 2 planes - low nibbles and high nibbles
  const size_t k2 = round_up_po2(k(), 2);  // tester assumes byte aligned rows
  const size_t packed_k2 =
      round_up_po2(k(), kr() * sr() * planes);  // 2 blocks for nibbles
  const size_t packed_k_bytes = (packed_k2 + 1) / 2;
  const size_t num_blocks = packed_k2 / bl();

  xnnpack::Buffer<float> input(m() * k2);
  xnnpack::Buffer<int8_t> a((m() - 1) * a_stride() + k2 +
                            XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<xnn_qd8_quantization_params> quantization_params(mr());
  xnnpack::Buffer<uint8_t> b(n() * k2 / 2);
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<xnn_bfloat16> kernel_scale2d(n() * k2 / bl());
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k_bytes +
      /* vksum */ packed_n() * sizeof(float) +
      /* scales */ packed_n() * num_blocks * sizeof(float) +
      /* bias */ packed_n() * sizeof(float));

  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < 1 /* kIterations */; iteration++) {
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    for (size_t i = 0; i < m(); ++i) {
      const float* input_ptr = &input[i * k2];
      const auto minmax = std::minmax_element(input_ptr, input_ptr + k2);
      float inv_scale;
      quantization_params[i] = xnn_f32_qd8_asymmetric_quantization_params(
          *minmax.first, *minmax.second, &inv_scale);
      for (size_t j = 0; j < k2; ++j) {
        float scaled_input = input_ptr[j] * inv_scale;
        scaled_input = std::min<float>(
            scaled_input, float(std::numeric_limits<int8_t>::max() -
                                quantization_params[i].zero_point));
        scaled_input = std::max<float>(
            scaled_input, float(std::numeric_limits<int8_t>::min() -
                                quantization_params[i].zero_point));
        a[i * a_stride() + j] = int8_t(std::lrintf(scaled_input) +
                                       long(quantization_params[i].zero_point));
      }
    }
    for (size_t i = m(); i < mr(); ++i) {
      quantization_params[i].zero_point =
          quantization_params[m() - 1].zero_point;
      quantization_params[i].inv_scale = quantization_params[m() - 1].inv_scale;
    }

    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale2d.begin(), kernel_scale2d.end(),
                  [&]() { return scalerng(); });

    std::fill(packed_w.begin(), packed_w.end(), 0);
    // Row sums are multiplied by input zero point, since we don't know it
    // until runtime, set it to 1.
    const xnn_qs8_qc4w_packing_params packing_params = {/*input_zero_point=*/1,
                                                        b_zero_point()};
    pack(/*g=*/1, n(), k2, nr(), kr(), sr(), bl(), b.data(), /*bias=*/nullptr,
         /*scale=*/kernel_scale2d.data(), packed_w.data(),
         sizeof(xnn_float16) * nr(), sizeof(float) * nr(), &packing_params);

    // Fill in packed kernel scale
    size_t stride =
        nr() *
        (packed_k_bytes + /* scales= */ num_blocks * sizeof(xnn_float16) +
         /* ksum= */ sizeof(float) + /* bias= */ sizeof(float));
    size_t block_stride = (bl() / 2 + sizeof(xnn_float16)) * nr();
    size_t start_offset = nr() * (packed_k_bytes / num_blocks + sizeof(float));
    uintptr_t start = (uintptr_t)packed_w.data() + start_offset;
    xnn_init_blockwise_scale_bf16_params(n(), nr(), stride,
                                         /*num_blocks=*/num_blocks,
                                         /*block_stride=*/block_stride,
                                         kernel_scale2d.data(), (void*)start);

    start = (uintptr_t)packed_w.data() + stride - sizeof(float) * nr();
    xnn_init_qs8_qc8w_scale_fp32_params(n(), nr(), stride, bias.data(),
                                        (void*)start);

    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        float kfsum = 0.0;
        for (size_t bl_index = 0; bl_index < num_blocks; ++bl_index) {
          int32_t ksum = 0;
          int32_t c_ref_acc = 0;
          for (size_t kr_index = 0; kr_index < bl(); kr_index++) {
            const size_t k_index = bl_index * bl() + kr_index;
            const size_t nb_index = (n_index * k2 + k_index) / 2;
            const int32_t bv =
                int32_t((k_index % 2 == 0) ? (b[nb_index] & UINT8_C(0xF))
                                           : (b[nb_index] >> 4)) -
                b_zero_point();
            ksum += bv;
            c_ref_acc +=
                int32_t(a[m_index * a_stride() + k_index]) * int32_t(bv);
          }
          size_t scale_index = n_index * num_blocks + bl_index;
          float scale = kernel_scale2d[scale_index];
          c_ref[m_index * n() + n_index] += c_ref_acc * scale;
          kfsum += scale * ksum;
        }
        c_ref[m_index * n() + n_index] -=
            (quantization_params[m_index].zero_point * kfsum);
        c_ref[m_index * n() + n_index] *=
            quantization_params[m_index].inv_scale;
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    // Prepare parameters.
    xnn_f32_qb4w_minmax_params params;
    init_params(&params, min(), max(), 8, bl());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], max()), min());
      }
    }

    if (unsigned_inputs()) {
      // Some architectures require that the input be unsigned.
      // Adjust the zero point and flip the sign of the input to mimic adding
      // 128 to the input with correct overflow behaviour.
      for (int i = 0; i < quantization_params.size(); ++i) {
        quantization_params[i].zero_point += 128;
      }
      for (int i = 0; i < a.size(); ++i) {
        a[i] ^= 0x80;
      }
    }
    gemm(m(), n(), k2, a.data(), a_stride() * sizeof(int8_t),
         static_cast<const void*>(packed_w.data()), c.data(),
         cm_stride() * sizeof(float), nr() * sizeof(float), &params,
         quantization_params.data());

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux
        // AArch64.
        const float tolerance =
            std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 1.0e-5f);
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k2;
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

  xnnpack::Buffer<float> input_f32(m() * k2);
  xnnpack::Buffer<uint8_t> b(n() * k2 / 2);
  xnnpack::Buffer<float> bias(n(), 0.0f);
  xnnpack::Buffer<float> kernel_scale(n());
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<int32_t> acc(m() * n());
  xnnpack::Buffer<float> c_ref(m() * n(), 0);

  // Create a fake `gemm_config` for the packing functions.
  struct xnn_gemm_config gemm_config;
  gemm_config.mr = static_cast<uint8_t>(mr());
  gemm_config.mr_packed = static_cast<uint8_t>(mr_packed());
  gemm_config.nr = static_cast<uint8_t>(nr());
  gemm_config.log2_kr = static_cast<uint8_t>(31 - math_clz_nonzero_u32(kr()));
  gemm_config.log2_sr = static_cast<uint8_t>(31 - math_clz_nonzero_u32(sr()));

  const size_t packed_w_stride =
      packed_stride(&gemm_config, k2, /*unused_block_size=*/0, /*k_stride=*/k2,
                    /*extra_bytes=*/0);
  const size_t packed_w_size = packed_w_stride * round_up(n(), nr());
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(packed_w_size);

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(input_f32.begin(), input_f32.end(), std::ref(f32rng));

    // Quantize the left-hand operand.
    const size_t input_packed_size =
        xnn_x8_packq_f32qp8_packed_size(m(), k2, mr_packed(), kr(), sr());
    xnnpack::Buffer<int8_t> input_qp8(input_packed_size);
    xnn_x8_packq_f32qp8_ukernel__scalar_u1(m(), k2, mr_packed(), kr(), sr(),
                                           /*m_idx_start=*/0, input_f32.data(),
                                           /*lhs_stride=*/k2 * sizeof(float),
                                           input_qp8.data());

    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));
    std::fill(packed_w.begin(), packed_w.end(), 0);

    // RHS packing.
    struct xnn_qs8_qc4w_packing_params params;
    params.input_zero_point = 1;
    params.kernel_zero_point = b_zero_point();
    pack(/*flags=*/0, &gemm_config, k2, n(),
         /*groups=*/1, /*unused_block_size=*/0,
         /*k_stride=*/k2,
         /*accumulator_init=*/nullptr,
         /*weights=*/b.data(),
         /*int_extra_data0_fn=*/nullptr,
         /*extra_data0=*/bias.data(),
         /*extra_data0_size=*/sizeof(float),
         /*init_extra_data1_fn=*/
         nullptr,
         /*extra_data1=*/kernel_scale.data(),
         /*extra_data1_size=*/sizeof(float),
         /*packed_weights_ptr=*/packed_w.data(), &params);

    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
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

    // Prepare parameters.
    xnn_f32_minmax_params minmax_params;
    init_minmax_params(&minmax_params, min(), max());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], max()), min());
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
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << " (accumulator = " << acc[i * n() + j] << "), optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k2
            << ", nr = " << nr() << ", cm_stride = " << cm_stride();
      }
    }
  }
}

void GemmMicrokernelTester::Test_QP8F32QC8W(
    xnn_qp8_f32_qc8w_gemm_minmax_ukernel_fn gemm,
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

  xnnpack::Buffer<float> input_f32(m() * k());
  xnnpack::Buffer<uint8_t> b(n() * k());
  xnnpack::Buffer<float> bias(n(), 0.0f);
  xnnpack::Buffer<float> kernel_scale(n());
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<int32_t> acc(m() * n());
  xnnpack::Buffer<float> c_ref(m() * n(), 0);

  // Create a fake `gemm_config` for the packing functions.
  struct xnn_gemm_config gemm_config;
  gemm_config.mr = static_cast<uint8_t>(mr());
  gemm_config.mr_packed = static_cast<uint8_t>(mr_packed());
  gemm_config.nr = static_cast<uint8_t>(nr());
  gemm_config.log2_kr = static_cast<uint8_t>(31 - math_clz_nonzero_u32(kr()));
  gemm_config.log2_sr = static_cast<uint8_t>(31 - math_clz_nonzero_u32(sr()));

  const size_t packed_w_stride =
      packed_stride(&gemm_config, k(), /*unused_block_size=*/0,
                    /*k_stride=*/k(), /*extra_bytes=*/0);
  const size_t packed_w_size = packed_w_stride * round_up(n(), nr());
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(packed_w_size);

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(input_f32.begin(), input_f32.end(), std::ref(f32rng));

    // Quantize the left-hand operand.
    const size_t input_packed_size =
        xnn_x8_packq_f32qp8_packed_size(m(), k(), mr_packed(), kr(), sr());
    xnnpack::Buffer<int8_t> input_qp8(input_packed_size);
    xnn_x8_packq_f32qp8_ukernel__scalar_u1(m(), k(), mr_packed(), kr(), sr(),
                                           /*m_idx_start=*/0, input_f32.data(),
                                           /*lhs_stride=*/k() * sizeof(float),
                                           input_qp8.data());

    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));
    std::fill(packed_w.begin(), packed_w.end(), 0);

    // RHS packing.
    struct xnn_qs8_qc8w_packing_params params;
    params.input_zero_point = 1;
    params.scale_multiplier = 1.0f;
    pack(/*flags=*/0, &gemm_config, k(), n(),
         /*groups=*/1, /*unused_block_size=*/0,
         /*k_stride=*/k(),
         /*accumulator_init=*/nullptr,
         /*weights=*/b.data(),
         /*int_extra_data0_fn=*/nullptr,
         /*extra_data0=*/bias.data(),
         /*extra_data0_size=*/sizeof(float),
         /*init_extra_data1_fn=*/
         nullptr,
         /*extra_data1=*/kernel_scale.data(),
         /*extra_data1_size=*/sizeof(float),
         /*packed_weights_ptr=*/packed_w.data(), &params);

    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          const size_t nb_index = (n_index * k() + k_index) / 2;
          const int32_t bv = static_cast<int32_t>(
              (k_index % 2 == 0) ? (b[nb_index] & UINT8_C(0xF))
                                 : (b[nb_index] >> 4));
          c_ref[m_index * n() + n_index] +=
              xnn_x8_packq_f32qp8_get_dequantized(m_index, k_index,
                                                  input_qp8.data(), k(),
                                                  mr_packed(), kr(), sr()) *
              static_cast<int32_t>(bv);
        }
        c_ref[m_index * n() + n_index] *= kernel_scale[n_index];
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    // Prepare parameters.
    xnn_f32_minmax_params minmax_params;
    init_minmax_params(&minmax_params, min(), max());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], max()), min());
      }
    }

    gemm(m(), n(), k(), input_qp8.data(), packed_w.data(), c.data(),
         cm_stride() * sizeof(float), sizeof(float), &minmax_params);

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux
        // AArch64.
        const float tolerance =
            std::max(1.1e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f);
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << " (accumulator = " << acc[i * n() + j] << "), optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", nr = " << nr() << ", cm_stride = " << cm_stride();
      }
    }
  }
}

void GemmMicrokernelTester::Test_PF32(
    xnn_pf32_gemm_minmax_ukernel_fn gemm,
    xnn_init_f32_minmax_params_fn init_minmax_params,
    xnn_pack_weights_and_biases_fn pack,
    xnn_packed_stride_weights_and_biases_fn packed_stride) {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f),
                          std::ref(rng));

  xnnpack::Buffer<float> input_f32(m() * k());
  xnnpack::Buffer<float> weights(n() * k());
  xnnpack::Buffer<float> bias(n(), 0.0f);
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<float> c_ref(m() * n(), 0);

  // Create a fake `gemm_config` for the packing functions.
  struct xnn_gemm_config gemm_config;
  gemm_config.mr = static_cast<uint8_t>(mr());
  gemm_config.mr_packed = static_cast<uint8_t>(mr_packed());
  gemm_config.nr = static_cast<uint8_t>(nr());
  gemm_config.log2_kr = static_cast<uint8_t>(31 - math_clz_nonzero_u32(kr()));
  gemm_config.log2_sr = static_cast<uint8_t>(31 - math_clz_nonzero_u32(sr()));

  const size_t packed_w_stride =
      packed_stride(&gemm_config, k(), /*unused_block_size=*/0,
                    /*k_stride=*/k(), /*extra_bytes=*/0);
  const size_t packed_w_size = packed_w_stride * round_up(n(), nr());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_w(packed_w_size);

  // Get the LHS packing config.
  const struct xnn_pack_lh_config* pack_lh_config =
      xnn_init_x32_pack_lh_config();
  ASSERT_NE(pack_lh_config, nullptr);

  // Loop over the iterations.
  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(input_f32.begin(), input_f32.end(), std::ref(f32rng));

    // Pack the left-hand operand.
    const size_t input_packed_size =
        pack_lh_config->size_fn(m(), k(), mr_packed(), kr(), sr());
    xnnpack::Buffer<int8_t> input_packed(input_packed_size);
    pack_lh_config->ukernel(m(), k(), mr_packed(), kr(), sr(),
                            /*m_idx_start=*/0, input_f32.data(),
                            /*lhs_stride=*/k() * sizeof(float),
                            input_packed.data());

    std::generate(weights.begin(), weights.end(), std::ref(f32rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::fill(packed_w.begin(), packed_w.end(), 0);

    // RHS packing.
    struct xnn_qs8_qc8w_packing_params params;
    params.input_zero_point = 1;
    params.scale_multiplier = 1.0f;
    pack(/*flags=*/0, &gemm_config, k(), n(),
         /*groups=*/1, /*unused_block_size=*/0,
         /*k_stride=*/k(),
         /*accumulator_init=*/bias.data(),
         /*weights=*/weights.data(),
         /*int_extra_data0_fn=*/nullptr,
         /*extra_data0=*/nullptr,
         /*extra_data0_size=*/0,
         /*init_extra_data1_fn=*/
         nullptr,
         /*extra_data1=*/nullptr,
         /*extra_data1_size=*/0,
         /*packed_weights_ptr=*/packed_w.data(), &params);

    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          c_ref[m_index * n() + n_index] += input_f32[m_index * k() + k_index] *
                                            weights[n_index * k() + k_index];
        }
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    // Prepare parameters.
    xnn_f32_minmax_params minmax_params;
    init_minmax_params(&minmax_params, min(), max());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], max()), min());
      }
    }

    gemm(m(), n(), k() * sizeof(float), input_packed.data(), packed_w.data(),
         c.data(), cm_stride() * sizeof(float), sizeof(float), &minmax_params);

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        // Extract tolerance into variable to workaround test failures on Linux
        // AArch64.
        const float tolerance =
            std::max(1.1e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f);
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", nr = " << nr() << ", cm_stride = " << cm_stride();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
    xnn_qp8_f32_qb4w_gemm_minmax_ukernel_fn gemm,
    xnn_init_f32_qb4w_minmax_params_fn init_minmax_params,
    xnn_pack_weights_and_biases_fn pack,
    xnn_packed_stride_weights_and_biases_fn packed_stride) {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-5.f, 5.f),
                          std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f),
                            std::ref(rng));
  auto w8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             0, std::numeric_limits<uint8_t>::max()),
                         std::ref(rng));

  const size_t k2 = round_up_po2(k(), 2);  // tester assumes byte aligned rows

  const size_t packed_k2 =
      round_up_po2(k(), kr() * sr());  // 2 blocks for nibbles
  const size_t packed_k_bytes = (packed_k2 + 1) / 2;
  const size_t num_blocks = packed_k2 / bl();

  xnnpack::Buffer<float> input_f32(m() * k2);
  xnnpack::Buffer<uint8_t> b(n() * k2 / 2);
  xnnpack::Buffer<float> bias(n(), 0.0f);
  xnnpack::Buffer<uint16_t> kernel_scale2d(n() * packed_k2 / bl());
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<int32_t> acc(m() * n());
  xnnpack::Buffer<float> c_ref(m() * n(), 0);

  // Create a fake `gemm_config` for the packing functions.
  struct xnn_gemm_config gemm_config;
  gemm_config.mr = static_cast<uint8_t>(mr());
  gemm_config.mr_packed = static_cast<uint8_t>(mr_packed());
  gemm_config.nr = static_cast<uint8_t>(nr());
  gemm_config.log2_kr = static_cast<uint8_t>(31 - math_clz_nonzero_u32(kr()));
  gemm_config.log2_sr = static_cast<uint8_t>(31 - math_clz_nonzero_u32(sr()));

  const size_t packed_w_stride =
      packed_stride(&gemm_config, k2, /*block_size=*/bl(), /*k_stride=*/k2,
                    /*extra_bytes=*/0);
  const size_t packed_w_size = packed_w_stride * round_up(n(), nr());
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(packed_w_size);

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(input_f32.begin(), input_f32.end(), std::ref(f32rng));
    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::generate(kernel_scale2d.begin(), kernel_scale2d.end(),
                  [&]() { return math_cvt_bf16_fp32(scalerng()); });
    std::fill(packed_w.begin(), packed_w.end(), 0);

    // Quantize the left-hand operand.
    const size_t input_packed_size =
        xnn_x8_packq_f32qp8_packed_size(m(), k2, mr_packed(), kr(), sr());
    xnnpack::Buffer<int8_t> input_qp8(input_packed_size);
    xnn_x8_packq_f32qp8_ukernel__scalar_u1(
        m(), k2, mr_packed(), kr(), sr(),
        /*m_idx_start=*/0, reinterpret_cast<const float*>(input_f32.data()),
        /*lhs_stride=*/k2 * sizeof(float), input_qp8.data());

    // RHS packing.
    struct xnn_qs8_qc4w_packing_params params;
    params.input_zero_point = 1;
    params.kernel_zero_point = b_zero_point();
    pack(/*flags=*/0, &gemm_config, k2, n(),
         /*groups=*/1, /*block_size=*/bl(),
         /*k_stride=*/k2,
         /*accumulator_init=*/nullptr,
         /*weights=*/b.data(),
         /*int_extra_data0_fn=*/nullptr,
         /*extra_data0=*/nullptr,
         /*extra_data0_size=*/0,
         /*init_extra_data1_fn=*/
         nullptr,
         /*extra_data1=*/kernel_scale2d.data(),
         /*extra_data1_size=*/sizeof(float),
         /*packed_weights_ptr=*/packed_w.data(), &params);

    size_t stride =
        nr() * (packed_k_bytes + /* scales= */ num_blocks * sizeof(uint16_t) +
                /* ksum= */ sizeof(float) + /* bias= */ sizeof(float));
    uintptr_t start =
        (uintptr_t)packed_w.data() + stride - sizeof(float) * nr();

    xnn_init_qs8_qc8w_scale_fp32_params(n(), nr(), stride, bias.data(),
                                        (void*)start);

    // Compute 32-bit results and output quantization arguments.
    std::fill(c_ref.begin(), c_ref.end(), 0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        float kfsum = 0.0;
        for (size_t bl_index = 0; bl_index < num_blocks; ++bl_index) {
          int32_t ksum = 0;
          int32_t c_ref_acc = 0;
          for (size_t kr_index = 0; kr_index < bl(); kr_index++) {
            const size_t k_index = bl_index * bl() + kr_index;
            const size_t nb_index = (n_index * k2 + k_index) / 2;
            const int32_t bv =
                int32_t((k_index % 2 == 0) ? (b[nb_index] & UINT8_C(0xF))
                                           : (b[nb_index] >> 4)) -
                b_zero_point();
            ksum += bv;
            c_ref_acc += int32_t(xnn_x8_packq_f32qp8_get_quantized(
                             m_index, k_index, input_qp8.data(), k2,
                             mr_packed(), kr(), sr())) *
                         int32_t(bv);
          }
          size_t scale_index = n_index * num_blocks + bl_index;
          float scale = math_cvt_fp32_bf16(kernel_scale2d[scale_index]);
          c_ref[m_index * n() + n_index] += c_ref_acc * scale;
          kfsum += scale * ksum;
        }
        float inv_scale = xnn_x8_packq_f32qp8_get_recip_scale(
            m_index, input_qp8.data(), k2, mr_packed(), kr(), sr());
        int32_t neg_nudged_zero_point = xnn_x8_packq_f32qp8_get_neg_nudged_zp(
            m_index, input_qp8.data(), k2, mr_packed(), kr(), sr());
        c_ref[m_index * n() + n_index] += (neg_nudged_zero_point * kfsum);
        c_ref[m_index * n() + n_index] *= inv_scale;
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    // Prepare parameters.
    xnn_f32_qb4w_minmax_params minmax_params;
    init_minmax_params(&minmax_params, min(), max(), 8, bl());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], max()), min());
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
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], tolerance)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << " (accumulator = " << acc[i * n() + j] << "), optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k2
            << ", nr = " << nr() << ", cm_stride = " << cm_stride();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_qs8_gemm_minmax_ukernel_fn gemm,
                                 xnn_init_qs8_conv_minmax_params_fn init_params,
                                 xnn_pack_qs8_gemm_fn pack,
                                 xnn_qs8_requantize_fn requantize) const {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000),
                          std::ref(rng));
  auto w8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));

  xnnpack::Buffer<int8_t> a((m() - 1) * a_stride() + k() +
                            XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<int8_t> b(n() * k());
  xnnpack::Buffer<int32_t> bias(n());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k() + packed_n() * sizeof(int32_t) / sizeof(int8_t));
  xnnpack::Buffer<int8_t> c((mr() - 1) * cm_stride() +
                            ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<int32_t> acc(m() * n());
  xnnpack::Buffer<int8_t> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    xnnpack::fill_uniform_random_bits(a.data(), a.size(), rng);
    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = {
        int8_t(a_zero_point() - 0x80)};
    void* const packed_data = packed_w.data();
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_data, /*extra_bytes=*/0, &packing_params);

    // Compute 32-bit results and output quantization arguments.
    std::fill(acc.begin(), acc.end(), 0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          acc[m_index * n() + n_index] +=
              (int32_t(a[m_index * a_stride() + k_index]) -
               int32_t(a_zero_point() - 0x80)) *
              int32_t(b[n_index * k() + k_index]);
        }
        acc[m_index * n() + n_index] += bias[n_index];
      }
    }

    const int32_t accumulated_min = *std::min_element(acc.cbegin(), acc.cend());
    const int32_t accumulated_max = *std::max_element(acc.cbegin(), acc.cend());
    const double c_scale =
        uint32_t(accumulated_max - accumulated_min) >= 256
            ? double(uint32_t(accumulated_max - accumulated_min)) / 255.0
            : 1.00001;
    const int8_t c_zero_point = int8_t(std::max(
        std::min(lrint(-0.5 - 0.5 * double(accumulated_min + accumulated_max) /
                                  c_scale),
                 long(std::numeric_limits<int8_t>::max())),
        long(std::numeric_limits<int8_t>::min())));

    const float requantization_scale = 1.0f / float(c_scale);
    union xnn_qs8_conv_minmax_params quantization_params;
    init_params(&quantization_params, requantization_scale, c_zero_point,
                int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

    gemm(m(), n(), k(), a.data(), a_stride() * sizeof(int8_t), packed_data,
         c.data(), cm_stride() * sizeof(int8_t), nr() * sizeof(int8_t),
         &quantization_params);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = requantize(
            acc[m_index * n() + n_index], requantization_scale, c_zero_point,
            int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(int32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  int32_t(qmax()) - 0x80);
        ASSERT_GE(int32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  int32_t(qmin()) - 0x80);
        ASSERT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  int32_t(c_ref[i * n() + j]))
            << "at " << i << ", " << j
            << ": reference = " << int32_t(c_ref[i * n() + j])
            << " (accumulator = " << acc[i * n() + j] << "), optimized = "
            << int32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()])
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", requantization scale = " << requantization_scale
            << ", output zero point = " << int32_t(c_zero_point);
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_qs8_igemm_minmax_ukernel_fn igemm,
                                 xnn_init_qs8_conv_minmax_params_fn init_params,
                                 xnn_pack_qs8_igemm_fn pack,
                                 xnn_qs8_requantize_fn requantize) const {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000),
                          std::ref(rng));
  auto w8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));

  xnnpack::Buffer<int8_t> a((mr() - 1) * a_stride() + k() +
                            XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<int8_t> b(n() * ks() * k());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      ks() * packed_n() * packed_k() +
      packed_n() * sizeof(int32_t) / sizeof(int8_t));
  xnnpack::Buffer<int32_t> bias(n());
  xnnpack::Buffer<int8_t> c((mr() - 1) * cm_stride() +
                            ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<int32_t> acc(m() * n());
  xnnpack::Buffer<int8_t> c_ref(m() * n());
  xnnpack::Buffer<int8_t> junk(k() + XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<const int8_t*> im2col(mr() * ks());

  {  // for (size_t iteration = 0; iteration < kIterations; iteration++) {
    xnnpack::fill_uniform_random_bits(a.data(), a.size(), rng);
    std::generate(b.begin(), b.end(), std::ref(w8rng));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = {
        int8_t(a_zero_point() - 0x80)};
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0,
         &packing_params);

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] =
            a.data() + a_stride() * m_index - a_offset();
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
                  (int32_t(im2col[ks_index * mr() + m_index][k_index]) -
                   int32_t(a_zero_point() - 0x80)) *
                  int32_t(b[(n_index * ks() + ks_index) * k() + k_index]);
            } else {
              acc[m_index * n() + n_index] +=
                  (int32_t(im2col[ks_index * mr() + m_index]
                                 [k_index + a_offset()]) -
                   int32_t(a_zero_point() - 0x80)) *
                  int32_t(b[(n_index * ks() + ks_index) * k() + k_index]);
            }
          }
        }
        acc[m_index * n() + n_index] += bias[n_index];
      }
    }

    const int32_t accumulated_min = *std::min_element(acc.cbegin(), acc.cend());
    const int32_t accumulated_max = *std::max_element(acc.cbegin(), acc.cend());
    const double c_scale =
        uint32_t(accumulated_max - accumulated_min) >= 256
            ? double(uint32_t(accumulated_max - accumulated_min)) / 255.0
            : 1.00001;
    const uint8_t c_zero_point = uint8_t(std::max(
        std::min(lrint(-0.5 - 0.5 * double(accumulated_min + accumulated_max) /
                                  c_scale),
                 long(std::numeric_limits<int8_t>::max())),
        long(std::numeric_limits<int8_t>::min())));

    const float requantization_scale = 1.0f / float(c_scale);
    union xnn_qs8_conv_minmax_params quantization_params;
    init_params(&quantization_params, requantization_scale, c_zero_point,
                int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

    const int8_t* zero_pointer =
        (zero_index() != SIZE_MAX) ? a.data() : nullptr;

    igemm(m(), n(), k(), ks() * mr() * sizeof(void*), im2col.data(),
          packed_w.data(), c.data(), cm_stride() * sizeof(int8_t),
          nr() * sizeof(int8_t), a_offset() * sizeof(uint8_t), zero_pointer,
          &quantization_params);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = requantize(
            acc[m_index * n() + n_index], requantization_scale, c_zero_point,
            int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(int32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  int32_t(qmax()) - 0x80);
        ASSERT_GE(int32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  int32_t(qmin()) - 0x80);
        ASSERT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * nr() + j % nr()]),
                  int32_t(c_ref[i * n() + j]))
            << "at " << i << ", " << j
            << ": reference = " << uint32_t(c_ref[i * n() + j])
            << " (accumulator = " << acc[i * n() + j] << "), optimized = "
            << (uint32_t)c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", requantization scale = " << requantization_scale
            << ", output zero point = " << int32_t(c_zero_point);
      }
    }
  }
}

void GemmMicrokernelTester::Test(
    xnn_bf16_f32_gemm_minmax_ukernel_fn gemm_minmax,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_bf16_f32_gemm_fn pack) const {
  if (a_stride() < k()) {
    return;
  }
  if (m() > mr()) {
    return;
  }
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f),
                          std::ref(rng));

  xnnpack::Buffer<xnn_bfloat16> a((m() - 1) * a_stride() + k() +
                                  XNN_EXTRA_BYTES / sizeof(xnn_bfloat16));
  xnnpack::Buffer<xnn_bfloat16> b(n() * k());
  xnnpack::Buffer<xnn_bfloat16, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k() + packed_n() * 2);
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), [&] { return f32rng(rng); });
    std::generate(b.begin(), b.end(), [&] { return f32rng(rng); });
    std::generate(bias.begin(), bias.end(), [&] { return f32rng(rng); });
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(),
         /*extra_bytes=*/0, /*params=*/nullptr);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = bias[n_index];
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          ASSERT_LT(m_index * k() + k_index, a.size());
          c_ref[m_index * n() + n_index] +=
              a[m_index * a_stride() + k_index] * b[n_index * k() + k_index];
        }
      }
    }

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, min(), max());

    for (float& c_value : c_ref) {
      c_value = std::max(std::min(c_value, max()), min());
    }

    gemm_minmax(m(), n(), k() * sizeof(xnn_bfloat16),
                reinterpret_cast<const uint16_t*>(a.data()),
                a_stride() * sizeof(xnn_bfloat16), packed_w.data(), c.data(),
                cm_stride() * sizeof(float), /*unused_cn_stride=*/0, &params);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j],
                    std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 3.0e-2f))
            << "at " << i << ", " << j << ": Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n()
            << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_bf16_gemm_minmax_ukernel_fn gemm_minmax,
                                 xnn_init_bf16_minmax_params_fn init_params,
                                 xnn_pack_f16_gemm_fn pack) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f),
                          std::ref(rng));

  xnnpack::Buffer<xnn_bfloat16> a((m() - 1) * a_stride() + k() +
                                  XNN_EXTRA_BYTES / sizeof(xnn_bfloat16));
  xnnpack::Buffer<xnn_bfloat16> b(n() * k());
  xnnpack::Buffer<xnn_bfloat16, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k() + packed_n());
  xnnpack::Buffer<xnn_bfloat16> bias(n());
  xnnpack::Buffer<xnn_bfloat16> c((mr() - 1) * cm_stride() +
                                  ((n() - 1) / nr()) * nr() + (n() - 1) % nr() +
                                  1);
  xnnpack::Buffer<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), [&] { return f32rng(rng); });
    std::generate(b.begin(), b.end(), [&] { return f32rng(rng); });
    std::generate(bias.begin(), bias.end(), [&] { return f32rng(rng); });
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
         reinterpret_cast<const uint16_t*>(b.data()),
         reinterpret_cast<const uint16_t*>(bias.data()), /*scale=*/nullptr,
         reinterpret_cast<uint16_t*>(packed_w.data()),
         /*extra_bytes=*/0, /*params=*/nullptr);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = bias[n_index];
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          ASSERT_LT(m_index * k() + k_index, a.size());
          c_ref[m_index * n() + n_index] +=
              a[m_index * a_stride() + k_index] * b[n_index * k() + k_index];
        }
      }
    }

    // Prepare parameters.
    xnn_bf16_minmax_params params;
    init_params(&params, min(), max());

    for (float& c_value : c_ref) {
      c_value = std::max(std::min(c_value, max()), min());
    }

    gemm_minmax(m(), n(), k() * sizeof(xnn_bfloat16), a.data(),
                a_stride() * sizeof(xnn_bfloat16), packed_w.data(), c.data(),
                cm_stride() * sizeof(xnn_bfloat16), nr() * sizeof(xnn_bfloat16),
                &params);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j],
                    std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 3.0e-2f))
            << "at " << i << ", " << j << ": Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n()
            << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f16_gemm_minmax_ukernel_fn gemm_minmax,
                                 xnn_init_f16_minmax_params_fn init_params,
                                 xnn_pack_f16_gemm_fn pack) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  xnnpack::Buffer<xnn_float16> a((m() - 1) * a_stride() + k() +
                                 XNN_EXTRA_BYTES / sizeof(xnn_float16));
  xnnpack::Buffer<xnn_float16> b(n() * k());
  xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k() + packed_n());
  xnnpack::Buffer<xnn_float16> bias(n());
  xnnpack::Buffer<xnn_float16> c((mr() - 1) * cm_stride() +
                                 ((n() - 1) / nr()) * nr() + (n() - 1) % nr() +
                                 1);
  xnnpack::Buffer<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), f32rng);
    std::generate(b.begin(), b.end(), f32rng);
    std::generate(bias.begin(), bias.end(), f32rng);
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(),
         reinterpret_cast<const uint16_t*>(b.data()),
         reinterpret_cast<const uint16_t*>(bias.data()),
         /*scale=*/nullptr, reinterpret_cast<uint16_t*>(packed_w.data()),
         /*extra_bytes=*/0, /*params=*/nullptr);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          ASSERT_LT(m_index * k() + k_index, a.size());
          c_ref[m_index * n() + n_index] +=
              a[m_index * a_stride() + k_index] * b[n_index * k() + k_index];
        }
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    // Prepare parameters.
    xnn_f16_minmax_params params;
    init_params(&params, static_cast<xnn_float16>(min()),
                static_cast<xnn_float16>(max()));

    for (float& c_value : c_ref) {
      c_value = std::max(std::min(c_value, max()), min());
    }

    gemm_minmax(m(), n(), k() * sizeof(xnn_float16), a.data(),
                a_stride() * sizeof(xnn_float16), packed_w.data(), c.data(),
                cm_stride() * sizeof(xnn_float16), nr() * sizeof(xnn_float16),
                &params);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j],
                    std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 1.0e-2f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << (float)c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f16_igemm_minmax_ukernel_fn igemm_minmax,
                                 xnn_init_f16_minmax_params_fn init_params,
                                 xnn_pack_f16_igemm_fn pack) const {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f),
                          std::ref(rng));

  xnnpack::Buffer<xnn_float16> a((mr() - 1) * a_stride() + k() +
                                 XNN_EXTRA_BYTES / sizeof(xnn_float16));
  xnnpack::Buffer<xnn_float16> b(n() * ks() * k());
  xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> packed_w(
      ks() * packed_k() * packed_n() + packed_n());
  xnnpack::Buffer<xnn_float16> bias(n());
  xnnpack::Buffer<xnn_float16> c((mr() - 1) * cm_stride() +
                                 ((n() - 1) / nr()) * nr() + (n() - 1) % nr() +
                                 1);
  xnnpack::Buffer<float> c_ref(m() * n());
  xnnpack::Buffer<xnn_float16> junk(k() +
                                    XNN_EXTRA_BYTES / sizeof(xnn_float16));
  xnnpack::Buffer<const xnn_float16*> im2col(mr() * ks());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), f32rng);
    std::generate(b.begin(), b.end(), f32rng);
    std::generate(bias.begin(), bias.end(), f32rng);
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(),
         reinterpret_cast<const uint16_t*>(b.data()),
         reinterpret_cast<const uint16_t*>(bias.data()), /*scale=*/nullptr,
         reinterpret_cast<uint16_t*>(packed_w.data()),
         /*extra_bytes=*/0, /*params=*/nullptr);

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] =
            a.data() + a_stride() * m_index - a_offset();
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
                  im2col[ks_index * mr() + m_index][k_index] *
                  b[(n_index * ks() + ks_index) * k() + k_index];
            } else {
              c_ref[m_index * n() + n_index] +=
                  im2col[ks_index * mr() + m_index][k_index + a_offset()] *
                  b[(n_index * ks() + ks_index) * k() + k_index];
            }
          }
        }
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    // Prepare parameters.
    xnn_f16_minmax_params params;
    init_params(&params, static_cast<xnn_float16>(min()),
                static_cast<xnn_float16>(max()));

    for (float& c_value : c_ref) {
      c_value = std::max(std::min(c_value, max()), min());
    }

    const xnn_float16* zero_pointer =
        (zero_index() != SIZE_MAX) ? a.data() : nullptr;

    igemm_minmax(
        m(), n(), k() * sizeof(xnn_float16), ks() * mr() * sizeof(void*),
        reinterpret_cast<const xnn_float16**>(im2col.data()), packed_w.data(),
        c.data(), cm_stride() * sizeof(xnn_float16), nr() * sizeof(xnn_float16),
        a_offset() * sizeof(xnn_float16), zero_pointer, &params);

    // Compute an upper bound for the summation error of the inner products
    // assuming `a` and `b` are in the range `[-1, 1]`.
    const float nu = k() * xnnpack::NumericLimits<xnn_float16>::epsilon();
    ASSERT_LT(k() * nu, 1.0f)
        << "Unreasonable dimensions for `xnn_float16` tolerance.";
    float max_abs_err = k() * nu / (1.0f - nu);

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], max())
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << (float)c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k()
            << " x " << ks();
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], min())
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << (float)c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k()
            << " x " << ks();
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], max_abs_err)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << (float)c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k()
            << " x " << ks();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_ppmm_minmax_ukernel_fn ppmm_minmax,
                                 xnn_init_f32_minmax_params_fn init_params,
                                 xnn_pack_f32_gemm_fn pack) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

  xnnpack::Buffer<float> a(packed_k() * mr());
  xnnpack::Buffer<float> b(n() * k());
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k() + packed_n());
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0,
         /*params=*/nullptr);

    for (size_t i = m(); i < mr(); i++) {
      for (size_t l = 0; l < k(); l++) {
        a[l * mr() + i] = a[l * mr() + m() - 1];
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        for (size_t l = 0; l < k(); l++) {
          c_ref[i * n() + j] += a[l * mr() + i] * b[j * k() + l];
        }
        c_ref[i * n() + j] += bias[j];
      }
    }

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, min(), max());

    for (float& c_value : c_ref) {
      c_value = std::max(std::min(c_value, max()), min());
    }

    ppmm_minmax(m(), n(), k() * sizeof(float), a.data(), packed_w.data(),
                c.data(), cm_stride() * sizeof(float), nr() * sizeof(float),
                &params);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j],
                    std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_gemm_ukernel_fn gemm,
                                 xnn_pack_f32_gemm_fn pack) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  xnnpack::Buffer<float> a((m() - 1) * a_stride() + k() +
                           XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<float> b(n() * k());
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k() + packed_n());
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0,
         /*params=*/nullptr);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          c_ref[m_index * n() + n_index] +=
              a[m_index * a_stride() + k_index] * b[n_index * k() + k_index];
        }
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    gemm(m(), n(), k() * sizeof(float), a.data(), a_stride() * sizeof(float),
         packed_w.data(), c.data(), cm_stride() * sizeof(float),
         nr() * sizeof(float), nullptr);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j],
                    std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_gemm_relu_ukernel_fn gemm_relu,
                                 xnn_pack_f32_gemm_fn pack) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  xnnpack::Buffer<float> a((m() - 1) * a_stride() + k() +
                           XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<float> b(n() * k());
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k() + packed_n());
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0,
         /*params=*/nullptr);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          c_ref[m_index * n() + n_index] +=
              a[m_index * a_stride() + k_index] * b[n_index * k() + k_index];
        }
        c_ref[m_index * n() + n_index] =
            std::max(0.0f, c_ref[m_index * n() + n_index] + bias[n_index]);
      }
    }

    gemm_relu(m(), n(), k() * sizeof(float), a.data(),
              a_stride() * sizeof(float), packed_w.data(), c.data(),
              cm_stride() * sizeof(float), nr() * sizeof(float), nullptr);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], 0.0f)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j],
                    std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_gemm_minmax_ukernel_fn gemm_minmax,
                                 xnn_init_f32_minmax_params_fn init_params,
                                 xnn_pack_f32_gemm_fn pack) const {
  if (a_stride() < k()) {
    return;
  }
  if (m() > mr()) {
    return;
  }
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

  xnnpack::Buffer<float> a((m() - 1) * a_stride() + k() +
                           XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<float> b(n() * k());
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k() + packed_n());
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0,
         /*params=*/nullptr);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          c_ref[m_index * n() + n_index] +=
              a[m_index * a_stride() + k_index] * b[n_index * k() + k_index];
        }
        c_ref[m_index * n() + n_index] += bias[n_index];
      }
    }

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, min(), max());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], max()), min());
      }
    }

    gemm_minmax(m(), n(), k() * sizeof(float), a.data(),
                a_stride() * sizeof(float), packed_w.data(), c.data(),
                cm_stride() * sizeof(float), nr() * sizeof(float), &params);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], max())
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], min())
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j],
                    std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
    xnn_f32_gemm_goi_minmax_ukernel_fn gemm_minmax,
    xnn_init_f32_minmax_params_fn init_params) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

  xnnpack::Buffer<float> a((m() - 1) * a_stride() + k() +
                           XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<float> b(n() * k());
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          c_ref[m_index * n() + n_index] +=
              a[m_index * a_stride() + k_index] * b[n_index * k() + k_index];
        }
      }
    }

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, min(), max());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], max()), min());
      }
    }

    gemm_minmax(m(), n(), k() * sizeof(float), a.data(),
                a_stride() * sizeof(float), b.data(), c.data(),
                cm_stride() * sizeof(float), nr() * sizeof(float), &params);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], max())
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], min())
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j],
                    std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
    xnn_f32_qc4w_gemm_minmax_ukernel_fn gemm_minmax,
    xnn_init_f32_qc4w_minmax_params_fn init_params,
    xnn_pack_f32_qc4w_gemm_fn pack) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);
  std::uniform_int_distribution<int32_t> i8dist(
      -1, std::numeric_limits<uint8_t>::max());

  const size_t k_stride = (k() + 1) / 2;
  const size_t packed_k_bytes = (packed_k() + 1) / 2;
  xnnpack::Buffer<float> a((m() - 1) * a_stride() + k() +
                           XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<uint8_t> b(n() * k_stride);
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float> scale(n());
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k_bytes + packed_n() * sizeof(float) * 2);
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<double> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
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

    pack(/*g=*/1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(), nr() * sizeof(float),
         /*params=*/nullptr);

    // Fill in packed scale
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(),
        nr() * (ks() * packed_k_bytes + (sizeof(float) + sizeof(float))),
        scale.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() * (ks() * packed_k_bytes + sizeof(float))));

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = double(bias[n_index]);
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          const size_t nb_index = n_index * k_stride + k_index / 2;
          const int16_t bv =
              int16_t((k_index % 2 == 0) ? (b[nb_index] & UINT8_C(0xF))
                                         : (b[nb_index] >> 4u)) -
              b_zero_point();

          c_ref[m_index * n() + n_index] +=
              double(a[m_index * a_stride() + k_index]) * double(bv);
        }
        c_ref[m_index * n() + n_index] *= double(scale[n_index]);
      }
    }

    // Prepare parameters.
    xnn_f32_qc4w_minmax_params params;
    init_params(&params, min(), max(), b_zero_point());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index],
                              static_cast<double>(max())),
                     static_cast<double>(min()));
      }
    }

    gemm_minmax(m(), n(),
                k() * sizeof(float),  // Note KC measured in bytes of input
                a.data(), a_stride() * sizeof(float), packed_w.data(), c.data(),
                cm_stride() * sizeof(float), nr() * sizeof(float), &params);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], max())
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], min())
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j],
                    std::max(1.0e-5, std::abs(c_ref[i * n() + j]) * 1.0e-6))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_qc8w_gemm_ukernel_fn gemm,
                                 xnn_pack_f32_qs8w_gemm_fn pack) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_int_distribution<int32_t> i8dist(
      -1, std::numeric_limits<int8_t>::max());

  xnnpack::Buffer<float> a((m() - 1) * a_stride() + k() +
                           XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<int8_t> b(n() * k());
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float> scale(n());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k() + packed_n() * sizeof(float) * 2);
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return i8dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    std::fill(packed_w.begin(), packed_w.end(), 0);

    pack(/*g=*/1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(), nr() * sizeof(float),
         /*params=*/nullptr);

    // Fill in packed scale
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(),
        nr() * (ks() * packed_k() * sizeof(int8_t) +
                (sizeof(float) + sizeof(float))),
        scale.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(float))));

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          c_ref[m_index * n() + n_index] += a[m_index * a_stride() + k_index] *
                                            (float)b[n_index * k() + k_index];
        }
        c_ref[m_index * n() + n_index] += bias[n_index];
        c_ref[m_index * n() + n_index] *= scale[n_index];
      }
    }

    gemm(m(), n(), k() * sizeof(float),  // Note KC measured in bytes of input
         a.data(), a_stride() * sizeof(float), packed_w.data(), c.data(),
         cm_stride() * sizeof(float), nr() * sizeof(float), nullptr);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], 0.1f);
      }
    }

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j],
                    std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_qc8w_gemm_relu_ukernel_fn gemm_relu,
                                 xnn_pack_f32_qs8w_gemm_fn pack) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_int_distribution<int32_t> i8dist(
      -1, std::numeric_limits<int8_t>::max());

  xnnpack::Buffer<float> a((m() - 1) * a_stride() + k() +
                           XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<int8_t> b(n() * k());
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float> scale(n());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k() + packed_n() * sizeof(float) * 2);
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return i8dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    std::fill(packed_w.begin(), packed_w.end(), 0);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(), nr() * sizeof(float),
         /*params=*/nullptr);

    // Fill in packed scale
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(),
        nr() * (ks() * packed_k() * sizeof(int8_t) +
                (sizeof(float) + sizeof(float))),
        scale.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(float))));

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          c_ref[m_index * n() + n_index] += a[m_index * a_stride() + k_index] *
                                            (float)b[n_index * k() + k_index];
        }
        c_ref[m_index * n() + n_index] =
            std::max(0.0f, c_ref[m_index * n() + n_index] + bias[n_index]);
        c_ref[m_index * n() + n_index] *= scale[n_index];
      }
    }

    gemm_relu(m(), n(),
              k() * sizeof(float),  // Note KC measured in bytes of input
              a.data(), a_stride() * sizeof(float), packed_w.data(), c.data(),
              cm_stride() * sizeof(float), nr() * sizeof(float), nullptr);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], 0.0f)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j],
                    std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
    xnn_f32_qc8w_gemm_minmax_ukernel_fn gemm_minmax,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_f32_qs8w_gemm_fn pack) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);
  std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

  xnnpack::Buffer<float> a((m() - 1) * a_stride() + k() +
                           XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<int8_t> b(n() * k());
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float> scale(n());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k() + packed_n() * sizeof(float) * 2);
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<double> c_ref(m() * n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return i8dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::generate(scale.begin(), scale.end(), [&]() {
      return std::abs(f32dist(rng)) / std::numeric_limits<int8_t>::max();
    });
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    std::fill(packed_w.begin(), packed_w.end(), 0);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(), nr() * sizeof(float),
         /*params=*/nullptr);

    // Fill in packed scale
    xnn_init_qs8_qc8w_scale_fp32_params(
        n(), nr(),
        nr() * (ks() * packed_k() * sizeof(int8_t) +
                (sizeof(float) + sizeof(float))),
        scale.data(),
        (void*)((uintptr_t)packed_w.data() +
                nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(float))));

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

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, min(), max());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index],
                              static_cast<double>(max())),
                     static_cast<double>(min()));
      }
    }

    gemm_minmax(m(), n(),
                k() * sizeof(float),  // Note KC measured in bytes of input
                a.data(), a_stride() * sizeof(float), packed_w.data(), c.data(),
                cm_stride() * sizeof(float), nr() * sizeof(float), &params);

    // Compute an upper bound for the summation error of the inner products
    // assuming `a` and `b` are in the range `[-1, 1]`.
    const float nu = k() * xnnpack::NumericLimits<float>::epsilon();
    ASSERT_LT(k() * nu, 1.0f)
        << "Unreasonable dimensions for `float` tolerance.";
    float max_abs_err = k() * nu / (1.0f - nu);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], max())
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], min())
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], max_abs_err)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_gemminc_minmax_ukernel_fn gemminc,
                                 xnn_init_f32_minmax_params_fn init_params,
                                 xnn_pack_f32_gemminc_fn pack) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

  xnnpack::Buffer<float> a((m() - 1) * a_stride() + k() +
                           XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<float> b(n() * k());
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_w(
      packed_n() * packed_k());  // no packed_n()
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<float> c_ref(m() * n());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> acc(mr() * packed_n());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    std::generate(acc.begin(), acc.end(), [&]() { return f32dist(rng); });

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), k(), nr(), kr(), sr(), b.data(), packed_w.data(),
         /*params=*/nullptr);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          c_ref[m_index * n() + n_index] +=
              a[m_index * a_stride() + k_index] * b[n_index * k() + k_index];
        }
        c_ref[m_index * n() + n_index] +=
            acc[n_index / nr() * nr() * mr() + m_index % mr() * nr() +
                n_index % nr()];
      }
    }

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, min(), max());

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::max(std::min(c_ref[m_index * n() + n_index], max()), min());
      }
    }

    gemminc(m(), n(), k() * sizeof(float), a.data(), a_stride() * sizeof(float),
            packed_w.data(), c.data(), cm_stride() * sizeof(float),
            nr() * sizeof(float), acc.data(), &params);

    // Compute an upper bound for the summation error of the inner products.
    const float nu = k() * xnnpack::NumericLimits<float>::epsilon();
    ASSERT_LT(k() * nu, 1.0f)
        << "Unreasonable dimensions for `float` tolerance.";
    float max_abs_err = k() * nu / (1.0f - nu);

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], max())
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], min())
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j], max_abs_err)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_igemm_ukernel_fn igemm,
                                 xnn_pack_f32_igemm_fn pack) const {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

  xnnpack::Buffer<float> a((mr() - 1) * a_stride() + k() +
                           XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<float> b(n() * ks() * k());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_w(
      ks() * packed_k() * packed_n() + packed_n());
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<float> c_ref(m() * n());
  xnnpack::Buffer<float> junk(k() + XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<const float*> im2col(mr() * ks());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0,
         /*params=*/nullptr);

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] =
            a.data() + a_stride() * m_index - a_offset();
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

    igemm(m(), n(), k() * sizeof(float), ks() * mr() * sizeof(void*),
          im2col.data(), packed_w.data(), c.data(), cm_stride() * sizeof(float),
          nr() * sizeof(float), a_offset() * sizeof(float), zero_pointer,
          nullptr);

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j],
                    std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k()
            << " x " << ks();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_igemm_relu_ukernel_fn igemm_relu,
                                 xnn_pack_f32_igemm_fn pack) const {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

  xnnpack::Buffer<float> a((mr() - 1) * a_stride() + k() +
                           XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<float> b(n() * ks() * k());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_w(
      ks() * packed_k() * packed_n() + packed_n());
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<float> c_ref(m() * n());
  xnnpack::Buffer<float> junk(k() + XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<const float*> im2col(mr() * ks());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0,
         /*params=*/nullptr);

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] =
            a.data() + a_stride() * m_index - a_offset();
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
        c_ref[m_index * n() + n_index] =
            std::max(0.0f, bias[n_index] + c_ref[m_index * n() + n_index]);
      }
    }

    const float* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : nullptr;

    igemm_relu(m(), n(), k() * sizeof(float), ks() * mr() * sizeof(void*),
               im2col.data(), packed_w.data(), c.data(),
               cm_stride() * sizeof(float), nr() * sizeof(float),
               a_offset() * sizeof(float), zero_pointer, nullptr);

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], 0.0f)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k()
            << " x " << ks();
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j],
                    std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k()
            << " x " << ks();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_igemm_minmax_ukernel_fn igemm_minmax,
                                 xnn_init_f32_minmax_params_fn init_params,
                                 xnn_pack_f32_igemm_fn pack) const {
  ASSERT_LE(m(), mr());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

  xnnpack::Buffer<float> a((mr() - 1) * a_stride() + k() +
                           XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<float> b(n() * ks() * k());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_w(
      ks() * packed_k() * packed_n() + packed_n());
  xnnpack::Buffer<float> bias(n());
  xnnpack::Buffer<float> c((mr() - 1) * cm_stride() +
                           ((n() - 1) / nr()) * nr() + (n() - 1) % nr() + 1);
  xnnpack::Buffer<float> c_ref(m() * n());
  xnnpack::Buffer<float> junk(k() + XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<const float*> im2col(mr() * ks());

  for (size_t iteration = 0; iteration < kIterations; iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    pack(/*g=*/1, n(), ks(), k(), nr(), kr(), sr(), b.data(), bias.data(),
         /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0,
         /*params=*/nullptr);

    for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
      for (size_t m_index = 0; m_index < mr(); m_index++) {
        im2col[ks_index * mr() + m_index] =
            a.data() + a_stride() * m_index - a_offset();
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

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] =
            std::min(c_ref[m_index * n() + n_index], max());
        c_ref[m_index * n() + n_index] =
            std::max(c_ref[m_index * n() + n_index], min());
      }
    }

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, min(), max());

    const float* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : nullptr;

    igemm_minmax(m(), n(), k() * sizeof(float), ks() * mr() * sizeof(void*),
                 im2col.data(), packed_w.data(), c.data(),
                 cm_stride() * sizeof(float), nr() * sizeof(float),
                 a_offset() * sizeof(float), zero_pointer, &params);

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], max())
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k()
            << " x " << ks();
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * nr() + j % nr()], min())
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k()
            << " x " << ks();
        ASSERT_NEAR(c[i * cm_stride() + (j / nr()) * nr() + j % nr()],
                    c_ref[i * n() + j],
                    std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = "
            << c[i * cm_stride() + (j / nr()) * nr() + j % nr()]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k()
            << " x " << ks();
      }
    }
  }
}
