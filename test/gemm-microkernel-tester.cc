#include "gemm-microkernel-tester.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <fp16.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>
#include <xnnpack/requantization.h>

void GemmMicrokernelTester::Test(
  xnn_qu8_gemm_minmax_ukernel_function gemm,
  xnn_init_qu8_conv_minmax_params_fn init_params,
  xnn_qu8_requantize_fn requantize) const
{
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto u8rng = std::bind(
    std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), std::ref(rng));

  std::vector<uint8_t> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<uint8_t> b(n() * k());
  std::vector<int32_t> bias(n());
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> packed_w(packed_n() * packed_k() + packed_n() * sizeof(int32_t) / sizeof(uint8_t));
  std::vector<uint8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<uint8_t> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(a.begin(), a.end(), std::ref(u8rng));
    } while (a.size() > 1 && *std::max_element(a.cbegin(), a.cend()) == *std::min_element(a.cbegin(), a.cend()));
    do {
      std::generate(b.begin(), b.end(), std::ref(u8rng));
    } while (b.size() > 1 && *std::max_element(b.cbegin(), b.cend()) == *std::min_element(b.cbegin(), b.cend()));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), b_zero_point());
    const xnn_qu8_packing_params packing_params = { a_zero_point(), b_zero_point() };
    xnn_pack_qu8_gemm_goi_w(1, n(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), packed_w.data(), 0, &packing_params);

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
        ASSERT_LE(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(qmax()));
        ASSERT_GE(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(qmin()));
        ASSERT_EQ(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(c_ref[i * n() + j]))
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
  xnn_qu8_igemm_minmax_ukernel_function igemm,
  xnn_init_qu8_conv_minmax_params_fn init_params,
  xnn_qu8_requantize_fn requantize)
{
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto u8rng = std::bind(
    std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), std::ref(rng));

  std::vector<uint8_t> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<uint8_t> b(n() * ks() * k());
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> packed_w(ks() * packed_n() * packed_k() + packed_n() * sizeof(int32_t) / sizeof(uint8_t));
  std::vector<int32_t> bias(n());
  std::vector<uint8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<uint8_t> c_ref(m() * n());
  std::vector<uint8_t> junk(k() + 8);
  std::vector<const uint8_t*> im2col(mr() * ks());

  std::fill(junk.begin(), junk.end(), 0xA5);

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(a.begin(), a.end(), std::ref(u8rng));
    } while (a.size() > 1 && *std::max_element(a.cbegin(), a.cend()) == *std::min_element(a.cbegin(), a.cend()));
    do {
      std::generate(b.begin(), b.end(), std::ref(u8rng));
    } while (b.size() > 1 && *std::max_element(b.cbegin(), b.cend()) == *std::min_element(b.cbegin(), b.cend()));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), b_zero_point());
    const xnn_qu8_packing_params packing_params = { a_zero_point(), b_zero_point() };
    xnn_pack_qu8_conv_goki_w(
      1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), packed_w.data(), 0 /* extra bytes */, &packing_params);

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

    const uint8_t* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : NULL;

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
        ASSERT_LE(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(qmax()));
        ASSERT_GE(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(qmin()));
        ASSERT_EQ(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(c_ref[i * n() + j]))
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
  xnn_qc8_gemm_minmax_ukernel_function gemm,
  xnn_init_qs8_minmax_params_fn init_params,
  xnn_qs8_requantize_fn requantize) const
{
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
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
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_w(packed_n() * packed_k() + packed_n() * (sizeof(int32_t) + sizeof(float)) / sizeof(int8_t));
  std::vector<int16_t, AlignedAllocator<int16_t, 64>> packed_xw(packed_n() * packed_k() + packed_n() * (sizeof(int32_t) + sizeof(float)) / sizeof(int16_t));
  std::vector<int8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<float> scale(n());
  std::vector<int8_t> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(a.begin(), a.end(), std::ref(i8rng));
    } while (a.size() > 1 && *std::max_element(a.cbegin(), a.cend()) == *std::min_element(a.cbegin(), a.cend()));
    do {
      std::generate(b.begin(), b.end(), std::ref(w8rng));
    } while (b.size() > 1 && *std::max_element(b.cbegin(), b.cend()) == *std::min_element(b.cbegin(), b.cend()));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = { int8_t(a_zero_point() - 0x80) };
    if (extended_weights()) {
      xnn_pack_qs8_gemm_xw_goi_w(1, n(), k(), nr(), kr(), sr(),
        b.data(), bias.data(), packed_xw.data(), nr() * sizeof(float), &packing_params);
    } else {
      xnn_pack_qs8_gemm_goi_w(1, n(), k(), nr(), kr(), sr(),
        b.data(), bias.data(), packed_w.data(), nr() * sizeof(float), &packing_params);
    }

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

    if (extended_weights()) {
      xnn_init_qc8_scale_fp32_params(
        n(), nr(),
        nr() * (packed_k() * sizeof(int16_t) + (sizeof(int32_t) + sizeof(float))), scale.data(),
        (void*) ((uintptr_t) packed_xw.data() + nr() * (packed_k() * sizeof(int16_t) + sizeof(int32_t))));
    } else {
      xnn_init_qc8_scale_fp32_params(
        n(), nr(),
        nr() * (packed_k() * sizeof(int8_t) + (sizeof(int32_t) + sizeof(float))), scale.data(),
        (void*) ((uintptr_t) packed_w.data() + nr() * (packed_k() * sizeof(int8_t) + sizeof(int32_t))));
    }

    union xnn_qs8_minmax_params minmax_params;
    init_params(&minmax_params,
      c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

    gemm(
      m(), n(), k(),
      a.data(), a_stride() * sizeof(int8_t),
      extended_weights() ? static_cast<const void*>(packed_xw.data()) : static_cast<const void*>(packed_w.data()),
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
        ASSERT_LE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmax()) - 0x80);
        ASSERT_GE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmin()) - 0x80);
        ASSERT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(c_ref[i * n() + j]))
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
  xnn_qc8_igemm_minmax_ukernel_function igemm,
  xnn_init_qs8_minmax_params_fn init_params,
  xnn_qs8_requantize_fn requantize) const
{
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));
  auto w8rng = std::bind(
    std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<int8_t> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<int8_t> b(n() * ks() * k());
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_w(ks() * packed_n() * packed_k() + packed_n() * (sizeof(int32_t) + sizeof(float)) / sizeof(int8_t));
  std::vector<int32_t> bias(n());
  std::vector<int8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<float> scale(n());
  std::vector<int8_t> c_ref(m() * n());
  std::vector<int8_t> junk(k() + 8);
  std::vector<const int8_t*> im2col(mr() * ks());

  std::fill(junk.begin(), junk.end(), 0xA5);

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(a.begin(), a.end(), std::ref(i8rng));
    } while (a.size() > 1 && *std::max_element(a.cbegin(), a.cend()) == *std::min_element(a.cbegin(), a.cend()));
    do {
      std::generate(b.begin(), b.end(), std::ref(w8rng));
    } while (b.size() > 1 && *std::max_element(b.cbegin(), b.cend()) == *std::min_element(b.cbegin(), b.cend()));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = { int8_t(a_zero_point() - 0x80) };
    xnn_pack_qs8_conv_goki_w(
      1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), packed_w.data(), nr() * sizeof(float), &packing_params);

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

    xnn_init_qc8_scale_fp32_params(
      n(), nr(),
      nr() * (ks() * packed_k() * sizeof(int8_t) + (sizeof(int32_t) + sizeof(float))), scale.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(int32_t))));

    union xnn_qs8_minmax_params minmax_params;
    init_params(&minmax_params,
      c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

    const int8_t* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : NULL;

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
        ASSERT_LE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmax()) - 0x80);
        ASSERT_GE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmin()) - 0x80);
        ASSERT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(c_ref[i * n() + j]))
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
  xnn_qs8_gemm_minmax_ukernel_function gemm,
  xnn_init_qs8_conv_minmax_params_fn init_params,
  xnn_qs8_requantize_fn requantize) const
{
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
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
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_w(packed_n() * packed_k() + packed_n() * sizeof(int32_t) / sizeof(int8_t));
  std::vector<int16_t, AlignedAllocator<int16_t, 64>> packed_xw(packed_n() * packed_k() + packed_n() * sizeof(int32_t) / sizeof(int16_t));
  std::vector<int8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<int8_t> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(a.begin(), a.end(), std::ref(i8rng));
    } while (a.size() > 1 && *std::max_element(a.cbegin(), a.cend()) == *std::min_element(a.cbegin(), a.cend()));
    do {
      std::generate(b.begin(), b.end(), std::ref(w8rng));
    } while (b.size() > 1 && *std::max_element(b.cbegin(), b.cend()) == *std::min_element(b.cbegin(), b.cend()));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = { int8_t(a_zero_point() - 0x80) };
    if (extended_weights()) {
      xnn_pack_qs8_gemm_xw_goi_w(1, n(), k(), nr(), kr(), sr(),
        b.data(), bias.data(), packed_xw.data(), 0, &packing_params);
    } else {
      xnn_pack_qs8_gemm_goi_w(1, n(), k(), nr(), kr(), sr(),
        b.data(), bias.data(), packed_w.data(), 0, &packing_params);
    }

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
      extended_weights() ? static_cast<const void*>(packed_xw.data()) : static_cast<const void*>(packed_w.data()),
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
        ASSERT_LE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmax()) - 0x80);
        ASSERT_GE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmin()) - 0x80);
        ASSERT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(c_ref[i * n() + j]))
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
  xnn_qs8_igemm_minmax_ukernel_function igemm,
  xnn_init_qs8_conv_minmax_params_fn init_params,
  xnn_qs8_requantize_fn requantize) const
{
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));
  auto w8rng = std::bind(
    std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<int8_t> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<int8_t> b(n() * ks() * k());
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_w(ks() * packed_n() * packed_k() + packed_n() * sizeof(int32_t) / sizeof(int8_t));
  std::vector<int32_t> bias(n());
  std::vector<int8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<int8_t> c_ref(m() * n());
  std::vector<int8_t> junk(k() + 8);
  std::vector<const int8_t*> im2col(mr() * ks());

  std::fill(junk.begin(), junk.end(), 0xA5);

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(a.begin(), a.end(), std::ref(i8rng));
    } while (a.size() > 1 && *std::max_element(a.cbegin(), a.cend()) == *std::min_element(a.cbegin(), a.cend()));
    do {
      std::generate(b.begin(), b.end(), std::ref(w8rng));
    } while (b.size() > 1 && *std::max_element(b.cbegin(), b.cend()) == *std::min_element(b.cbegin(), b.cend()));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = { int8_t(a_zero_point() - 0x80) };
    xnn_pack_qs8_conv_goki_w(
      1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), packed_w.data(), 0 /* extra bytes */, &packing_params);

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

    const int8_t* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : NULL;

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
        ASSERT_LE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmax()) - 0x80);
        ASSERT_GE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmin()) - 0x80);
        ASSERT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(c_ref[i * n() + j]))
            << "at " << i << ", " << j << ": reference = " << uint32_t(c_ref[i * n() + j])
            << " (accumulator = " << acc[i * n() + j]
            << "), optimized = " << (uint32_t) c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", requantization scale = " << requantization_scale << ", output zero point = " << int32_t(c_zero_point);
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f16_gemm_minmax_ukernel_function gemm_minmax, xnn_init_f16_scaleminmax_params_fn init_params) const
{
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  std::vector<uint16_t> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<uint16_t> b(n() * k());
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> packed_w(packed_n() * packed_k() + packed_n());
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
    xnn_pack_f16_gemm_goi_w(1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(), packed_w.data(), 0, nullptr);

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t k_index = 0; k_index < k(); k_index++) {
          ASSERT_LE(n(), packed_n());
          ASSERT_LT(m_index * n() + n_index, c_ref.size());
          ASSERT_LT(m_index * k() + k_index, a.size());
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
    xnn_f16_scaleminmax_params params;
    init_params(&params,
      UINT16_C(0x3C00) /* 1.0 */,
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
        ASSERT_NEAR(fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), c_ref[i * n() + j], std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 1.0e-2f))
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]) << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f16_igemm_minmax_ukernel_function igemm_minmax, xnn_init_f16_scaleminmax_params_fn init_params) const {
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  std::vector<uint16_t> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<uint16_t> b(n() * ks() * k());
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> packed_w(ks() * packed_k() * packed_n() + packed_n());
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
    std::fill(c_ref.begin(), c_ref.end(), 0);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    xnn_pack_f16_conv_goki_w(
      1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), packed_w.data(), 0 /* extra bytes */, nullptr);

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

    std::fill(c_ref.begin(), c_ref.end(), 0.0);
    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            ASSERT_LT(ks_index * mr() + m_index, im2col.size());
            ASSERT_LT(k_index, k());
            ASSERT_LT(k_index, a_stride());
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
    xnn_f16_scaleminmax_params params;
    init_params(&params,
      UINT16_C(0x3C00) /* 1.0 */,
      fp16_ieee_from_fp32_value(c_min),
      fp16_ieee_from_fp32_value(c_max));

    for (float& c_value : c_ref) {
      c_value = std::max(std::min(c_value, c_max), c_min);
    }

    const uint16_t* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : NULL;

    igemm_minmax(
      m(), n(), k() * sizeof(uint16_t), ks() * mr() * sizeof(void*),
      reinterpret_cast<const void**>(im2col.data()), packed_w.data(),
      c.data(), cm_stride() * sizeof(uint16_t), cn_stride() * sizeof(uint16_t),
      a_offset() * sizeof(uint16_t), zero_pointer,
      &params);

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), c_max)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]) << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
        ASSERT_GE(fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), c_min)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]) << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
        ASSERT_NEAR(fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), c_ref[i * n() + j], std::max(1.0e-4f, std::abs(c_ref[i * n() + j]) * 1.0e-2f))
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]) << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_ppmm_minmax_ukernel_function ppmm_minmax, xnn_init_f32_minmax_params_fn init_params) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(cm_stride(), n());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a(packed_k() * mr());
  std::vector<float> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float, AlignedAllocator<float, 64>> packed_w(packed_n() * packed_k() + packed_n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(f32rng));
    std::generate(b.begin(), b.end(), std::ref(f32rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    xnn_pack_f32_gemm_goi_w(1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(), packed_w.data(), 0, nullptr);

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
        ASSERT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::abs(c_ref[i * n() + j]) * 1.0e-6f)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_gemm_ukernel_function gemm) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float, AlignedAllocator<float, 64>> packed_w(packed_n() * packed_k() + packed_n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(f32rng));
    std::generate(b.begin(), b.end(), std::ref(f32rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    xnn_pack_f32_gemm_goi_w(1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(), packed_w.data(), 0, nullptr);

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
        ASSERT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::abs(c_ref[i * n() + j]) * 1.0e-6f)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_gemm_relu_ukernel_function gemm_relu) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float, AlignedAllocator<float, 64>> packed_w(packed_n() * packed_k() + packed_n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(f32rng));
    std::generate(b.begin(), b.end(), std::ref(f32rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    xnn_pack_f32_gemm_goi_w(1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(), packed_w.data(), 0, nullptr);

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
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], 0.0f)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::abs(c_ref[i * n() + j]) * 1.0e-6f)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_gemm_minmax_ukernel_function gemm_minmax, xnn_init_f32_minmax_params_fn init_params) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float, AlignedAllocator<float, 64>> packed_w(packed_n() * packed_k() + packed_n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(f32rng));
    std::generate(b.begin(), b.end(), std::ref(f32rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    xnn_pack_f32_gemm_goi_w(1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(), packed_w.data(), 0, nullptr);

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
        ASSERT_LE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_max)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_min)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::abs(c_ref[i * n() + j]) * 1.0e-6f)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_gemminc_minmax_ukernel_function gemminc, xnn_init_f32_minmax_params_fn init_params) const {
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float, AlignedAllocator<float, 64>> packed_w(packed_n() * packed_k());  // no packed_n()
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());
  std::vector<float, AlignedAllocator<float, 64>> acc(mr() * packed_n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(f32rng));
    std::generate(b.begin(), b.end(), std::ref(f32rng));
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);
    std::generate(acc.begin(), acc.end(), std::ref(f32rng));

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    xnn_pack_f32_gemminc_goi_w(1, n(), k(), nr(), kr(), sr(), b.data(), packed_w.data(), nullptr);

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
        ASSERT_LE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_max)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_min)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::abs(c_ref[i * n() + j]) * 1.0e-6f)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_igemm_ukernel_function igemm) const {
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * ks() * k());
  std::vector<float, AlignedAllocator<float, 64>> packed_w(ks() * packed_k() * packed_n() + packed_n());
  std::vector<float> bias(n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());
  std::vector<float> junk(k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<const float*> im2col(mr() * ks());
  std::fill(junk.begin(), junk.end(), nanf(""));

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(f32rng));
    std::generate(b.begin(), b.end(), std::ref(f32rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    xnn_pack_f32_conv_goki_w(
      1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), packed_w.data(), 0 /* extra bytes */, nullptr);

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

    std::fill(c_ref.begin(), c_ref.end(), 0.0);
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

    const float* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : NULL;

    igemm(
      m(), n(), k() * sizeof(float), ks() * mr() * sizeof(void*),
      im2col.data(), packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
      a_offset() * sizeof(float), zero_pointer,
      nullptr);

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::abs(c_ref[i * n() + j]) * 1.0e-6f)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_igemm_relu_ukernel_function igemm_relu) const {
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * ks() * k());
  std::vector<float, AlignedAllocator<float, 64>> packed_w(ks() * packed_k() * packed_n() + packed_n());
  std::vector<float> bias(n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());
  std::vector<float> junk(k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<const float*> im2col(mr() * ks());
  std::fill(junk.begin(), junk.end(), nanf(""));

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(f32rng));
    std::generate(b.begin(), b.end(), std::ref(f32rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    xnn_pack_f32_conv_goki_w(
      1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), packed_w.data(), 0 /* extra bytes */, nullptr);

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

    std::fill(c_ref.begin(), c_ref.end(), 0.0);
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

    const float* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : NULL;

    igemm_relu(
      m(), n(), k() * sizeof(float), ks() * mr() * sizeof(void*),
      im2col.data(), packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
      a_offset() * sizeof(float), zero_pointer,
      nullptr);

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], 0.0f)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
        ASSERT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::abs(c_ref[i * n() + j]) * 1.0e-6f)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
      }
    }
  }
}

void GemmMicrokernelTester::Test(xnn_f32_igemm_minmax_ukernel_function igemm_minmax, xnn_init_f32_minmax_params_fn init_params) const {
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * ks() * k());
  std::vector<float, AlignedAllocator<float, 64>> packed_w(ks() * packed_k() * packed_n() + packed_n());
  std::vector<float> bias(n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());
  std::vector<float> junk(k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<const float*> im2col(mr() * ks());
  std::fill(junk.begin(), junk.end(), nanf(""));

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(f32rng));
    std::generate(b.begin(), b.end(), std::ref(f32rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    xnn_pack_f32_conv_goki_w(
      1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), packed_w.data(), 0 /* extra bytes */, nullptr);

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

    std::fill(c_ref.begin(), c_ref.end(), 0.0);
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

    const float* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : NULL;

    igemm_minmax(
      m(), n(), k() * sizeof(float), ks() * mr() * sizeof(void*),
      im2col.data(), packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
      a_offset() * sizeof(float), zero_pointer,
      &params);

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_max)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_min)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
        ASSERT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::abs(c_ref[i * n() + j]) * 1.0e-6f)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
      }
    }
  }
}

#if XNN_PLATFORM_JIT
void GemmMicrokernelTester::Test(
    xnn_jit_gemm_code_generator_function gemm_generator,
    xnn_init_f32_minmax_params_fn init_params) const
{
  ASSERT_LE(m(), mr());
  ASSERT_GE(a_stride(), k());
  ASSERT_GE(cm_stride(), n());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * k());
  std::vector<float> bias(n());
  std::vector<float, AlignedAllocator<float, 64>> packed_w(packed_n() * packed_k() + packed_n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(f32rng));
    std::generate(b.begin(), b.end(), std::ref(f32rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    xnn_pack_f32_gemm_goi_w(1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(), packed_w.data(), 0, nullptr);

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

    struct xnn_code_buffer code_buffer;
    ASSERT_EQ(xnn_status_success, xnn_allocate_code_memory(&code_buffer, XNN_DEFAULT_CODE_BUFFER_SIZE));
    jit_gemm_params p = (jit_gemm_params) {
      .f32_minmax = {
        .min = c_min,
        .max = c_max
      }
    };
    ASSERT_EQ(xnn_status_success, gemm_generator(&code_buffer, n() % nr(), k() * sizeof(float), &p));
    xnn_f32_gemm_minmax_ukernel_function gemm_minmax =
        reinterpret_cast<xnn_f32_gemm_minmax_ukernel_function>(code_buffer.code);

    gemm_minmax(m(), n(), k() * sizeof(float),
      a.data(), a_stride() * sizeof(float),
      packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
      &params);

    ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&code_buffer));

    // Validate micro-kernel outputs.
    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_max)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_min)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        ASSERT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::abs(c_ref[i * n() + j]) * 1.0e-6f)
            << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
    xnn_jit_igemm_code_generator_function igemm_generator,
    xnn_init_f32_minmax_params_fn init_params) const
{
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(n() * ks() * k());
  std::vector<float, AlignedAllocator<float, 64>> packed_w(ks() * packed_k() * packed_n() + packed_n());
  std::vector<float> bias(n());
  std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<float> c_ref(m() * n());
  std::vector<float> junk(k() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<const float*> im2col(mr() * ks());
  std::fill(junk.begin(), junk.end(), nanf(""));

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(f32rng));
    std::generate(b.begin(), b.end(), std::ref(f32rng));
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));
    std::fill(c.begin(), c.end(), nanf(""));
    std::fill(c_ref.begin(), c_ref.end(), 0.0f);

    std::fill(packed_w.begin(), packed_w.end(), 0.0f);
    xnn_pack_f32_conv_goki_w(
      1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), packed_w.data(), 0 /* extra bytes */, nullptr);

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

    std::fill(c_ref.begin(), c_ref.end(), 0.0);
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

    const float* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : NULL;

    struct xnn_code_buffer code_buffer;
    ASSERT_EQ(xnn_status_success, xnn_allocate_code_memory(&code_buffer, XNN_DEFAULT_CODE_BUFFER_SIZE));
    jit_gemm_params p = (jit_gemm_params) {
      .f32_minmax = {
        .min = c_min,
        .max = c_max
      }
    };
    ASSERT_EQ(xnn_status_success,
              igemm_generator(&code_buffer, n() % nr(), k() * sizeof(float), ks() * mr() * sizeof(void *), &p));
    xnn_f32_igemm_minmax_ukernel_function igemm_minmax =
        reinterpret_cast<xnn_f32_igemm_minmax_ukernel_function>(code_buffer.code);

    igemm_minmax(
      m(), n(), k() * sizeof(float), ks() * mr() * sizeof(void*),
      im2col.data(), packed_w.data(),
      c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
      a_offset() * sizeof(float), zero_pointer,
      &params);

    ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&code_buffer));

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_max)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
        ASSERT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_min)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
        ASSERT_NEAR(
            c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
            c_ref[i * n() + j],
            std::abs(c_ref[i * n() + j]) * 1.0e-6f)
            << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
            << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
      }
    }
  }
}

void GemmMicrokernelTester::Test(
  xnn_jit_gemm_code_generator_function gemm_generator,
  xnn_init_qs8_minmax_params_fn init_params,
  xnn_qs8_requantize_fn requantize) const
{
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
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
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_w(packed_n() * packed_k() + packed_n() * (sizeof(int32_t) + sizeof(float)) / sizeof(int8_t));
  std::vector<int16_t, AlignedAllocator<int16_t, 64>> packed_xw(packed_n() * packed_k() + packed_n() * (sizeof(int32_t) + sizeof(float)) / sizeof(int16_t));
  std::vector<int8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<float> scale(n());
  std::vector<int8_t> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(a.begin(), a.end(), std::ref(i8rng));
    } while (a.size() > 1 && *std::max_element(a.cbegin(), a.cend()) == *std::min_element(a.cbegin(), a.cend()));
    do {
      std::generate(b.begin(), b.end(), std::ref(w8rng));
    } while (b.size() > 1 && *std::max_element(b.cbegin(), b.cend()) == *std::min_element(b.cbegin(), b.cend()));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = { int8_t(a_zero_point() - 0x80) };
    if (extended_weights()) {
      xnn_pack_qs8_gemm_xw_goi_w(1, n(), k(), nr(), kr(), sr(),
        b.data(), bias.data(), packed_xw.data(), nr() * sizeof(float), &packing_params);
    } else {
      xnn_pack_qs8_gemm_goi_w(1, n(), k(), nr(), kr(), sr(),
        b.data(), bias.data(), packed_w.data(), nr() * sizeof(float), &packing_params);
    }

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

    if (extended_weights()) {
      xnn_init_qc8_scale_fp32_params(
        n(), nr(),
        nr() * (packed_k() * sizeof(int16_t) + (sizeof(int32_t) + sizeof(float))), scale.data(),
        (void*) ((uintptr_t) packed_xw.data() + nr() * (packed_k() * sizeof(int16_t) + sizeof(int32_t))));
    } else {
      xnn_init_qc8_scale_fp32_params(
        n(), nr(),
        nr() * (packed_k() * sizeof(int8_t) + (sizeof(int32_t) + sizeof(float))), scale.data(),
        (void*) ((uintptr_t) packed_w.data() + nr() * (packed_k() * sizeof(int8_t) + sizeof(int32_t))));
    }

    union xnn_qs8_minmax_params minmax_params;
    init_params(&minmax_params,
      c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

    struct xnn_code_buffer code_buffer;
    ASSERT_EQ(xnn_status_success, xnn_allocate_code_memory(&code_buffer, XNN_DEFAULT_CODE_BUFFER_SIZE));
    ASSERT_EQ(xnn_status_success, gemm_generator(&code_buffer, n() % nr(), k(), nullptr));
    xnn_qc8_gemm_minmax_ukernel_function gemm = reinterpret_cast<xnn_qc8_gemm_minmax_ukernel_function>(code_buffer.code);

    gemm(
      m(), n(), k(),
      a.data(), a_stride() * sizeof(int8_t),
      extended_weights() ? static_cast<const void*>(packed_xw.data()) : static_cast<const void*>(packed_w.data()),
      c.data(), cm_stride() * sizeof(int8_t), cn_stride() * sizeof(int8_t),
      &minmax_params);

    ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&code_buffer));

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = requantize(
          acc[m_index * n() + n_index], scale[n_index], c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmax()) - 0x80);
        ASSERT_GE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmin()) - 0x80);
        ASSERT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(c_ref[i * n() + j]))
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
  xnn_jit_igemm_code_generator_function igemm_generator,
  xnn_init_qs8_minmax_params_fn init_params,
  xnn_qs8_requantize_fn requantize) const
{
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));
  auto w8rng = std::bind(
    std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<int8_t> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<int8_t> b(n() * ks() * k());
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_w(ks() * packed_n() * packed_k() + packed_n() * (sizeof(int32_t) + sizeof(float)) / sizeof(int8_t));
  std::vector<int32_t> bias(n());
  std::vector<int8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<float> scale(n());
  std::vector<int8_t> c_ref(m() * n());
  std::vector<int8_t> junk(k() + 8);
  std::vector<const int8_t*> im2col(mr() * ks());

  std::fill(junk.begin(), junk.end(), 0xA5);

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(a.begin(), a.end(), std::ref(i8rng));
    } while (a.size() > 1 && *std::max_element(a.cbegin(), a.cend()) == *std::min_element(a.cbegin(), a.cend()));
    do {
      std::generate(b.begin(), b.end(), std::ref(w8rng));
    } while (b.size() > 1 && *std::max_element(b.cbegin(), b.cend()) == *std::min_element(b.cbegin(), b.cend()));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = { int8_t(a_zero_point() - 0x80) };
    xnn_pack_qs8_conv_goki_w(
      1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), packed_w.data(), nr() * sizeof(float), &packing_params);

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

    xnn_init_qc8_scale_fp32_params(
      n(), nr(),
      nr() * (ks() * packed_k() * sizeof(int8_t) + (sizeof(int32_t) + sizeof(float))), scale.data(),
      (void*) ((uintptr_t) packed_w.data() + nr() * (ks() * packed_k() * sizeof(int8_t) + sizeof(int32_t))));

    union xnn_qs8_minmax_params minmax_params;
    init_params(&minmax_params,
      c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

    const int8_t* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : NULL;

    struct xnn_code_buffer code_buffer;
    ASSERT_EQ(xnn_status_success, xnn_allocate_code_memory(&code_buffer, XNN_DEFAULT_CODE_BUFFER_SIZE));
    ASSERT_EQ(xnn_status_success, igemm_generator(&code_buffer, n() % nr(), k(), ks() * mr() * sizeof(void *), nullptr));
    xnn_qc8_igemm_minmax_ukernel_function igemm = reinterpret_cast<xnn_qc8_igemm_minmax_ukernel_function>(code_buffer.code);

    igemm(
      m(), n(), k(), ks() * mr() * sizeof(void*),
      im2col.data(), packed_w.data(),
      c.data(), cm_stride() * sizeof(int8_t), cn_stride() * sizeof(int8_t),
      a_offset() * sizeof(uint8_t), zero_pointer,
      &minmax_params);

    ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&code_buffer));

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = requantize(
          acc[m_index * n() + n_index], scale[n_index], c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmax()) - 0x80);
        ASSERT_GE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmin()) - 0x80);
        ASSERT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(c_ref[i * n() + j]))
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
  xnn_jit_gemm_code_generator_function gemm_generator,
  xnn_init_qs8_conv_minmax_params_fn init_params,
  xnn_qs8_requantize_fn requantize) const
{
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
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
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_w(packed_n() * packed_k() + packed_n() * sizeof(int32_t) / sizeof(int8_t));
  std::vector<int16_t, AlignedAllocator<int16_t, 64>> packed_xw(packed_n() * packed_k() + packed_n() * sizeof(int32_t) / sizeof(int16_t));
  std::vector<int8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<int8_t> c_ref(m() * n());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(a.begin(), a.end(), std::ref(i8rng));
    } while (a.size() > 1 && *std::max_element(a.cbegin(), a.cend()) == *std::min_element(a.cbegin(), a.cend()));
    do {
      std::generate(b.begin(), b.end(), std::ref(w8rng));
    } while (b.size() > 1 && *std::max_element(b.cbegin(), b.cend()) == *std::min_element(b.cbegin(), b.cend()));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = { int8_t(a_zero_point() - 0x80) };
    if (extended_weights()) {
      xnn_pack_qs8_gemm_xw_goi_w(1, n(), k(), nr(), kr(), sr(),
        b.data(), bias.data(), packed_xw.data(), 0, &packing_params);
    } else {
      xnn_pack_qs8_gemm_goi_w(1, n(), k(), nr(), kr(), sr(),
        b.data(), bias.data(), packed_w.data(), 0, &packing_params);
    }

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

    struct xnn_code_buffer code_buffer;
    ASSERT_EQ(xnn_status_success, xnn_allocate_code_memory(&code_buffer, XNN_DEFAULT_CODE_BUFFER_SIZE));
    ASSERT_EQ(xnn_status_success, gemm_generator(&code_buffer, n() % nr(), k(), nullptr));
    xnn_qs8_gemm_minmax_ukernel_function gemm = reinterpret_cast<xnn_qs8_gemm_minmax_ukernel_function >(code_buffer.code);

    gemm(
      m(), n(), k(),
      a.data(), a_stride() * sizeof(int8_t),
      extended_weights() ? static_cast<const void*>(packed_xw.data()) : static_cast<const void*>(packed_w.data()),
      c.data(), cm_stride() * sizeof(int8_t), cn_stride() * sizeof(int8_t),
      &quantization_params);

    ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&code_buffer));

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = requantize(
          acc[m_index * n() + n_index], requantization_scale, c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmax()) - 0x80);
        ASSERT_GE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmin()) - 0x80);
        ASSERT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(c_ref[i * n() + j]))
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
  xnn_jit_igemm_code_generator_function igemm_generator,
  xnn_init_qs8_conv_minmax_params_fn init_params,
  xnn_qs8_requantize_fn requantize) const
{
  ASSERT_LE(m(), mr());

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));
  auto w8rng = std::bind(
    std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<int8_t> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<int8_t> b(n() * ks() * k());
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_w(ks() * packed_n() * packed_k() + packed_n() * sizeof(int32_t) / sizeof(int8_t));
  std::vector<int32_t> bias(n());
  std::vector<int8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
  std::vector<int32_t> acc(m() * n());
  std::vector<int8_t> c_ref(m() * n());
  std::vector<int8_t> junk(k() + 8);
  std::vector<const int8_t*> im2col(mr() * ks());

  std::fill(junk.begin(), junk.end(), 0xA5);

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(a.begin(), a.end(), std::ref(i8rng));
    } while (a.size() > 1 && *std::max_element(a.cbegin(), a.cend()) == *std::min_element(a.cbegin(), a.cend()));
    do {
      std::generate(b.begin(), b.end(), std::ref(w8rng));
    } while (b.size() > 1 && *std::max_element(b.cbegin(), b.cend()) == *std::min_element(b.cbegin(), b.cend()));
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(c.begin(), c.end(), 0xA5);

    std::fill(packed_w.begin(), packed_w.end(), 0);
    const xnn_qs8_packing_params packing_params = { int8_t(a_zero_point() - 0x80) };
    xnn_pack_qs8_conv_goki_w(
      1, n(), ks(), k(), nr(), kr(), sr(),
      b.data(), bias.data(), packed_w.data(), 0 /* extra bytes */, &packing_params);

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

    const int8_t* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : NULL;

    struct xnn_code_buffer code_buffer;
    ASSERT_EQ(xnn_status_success, xnn_allocate_code_memory(&code_buffer, XNN_DEFAULT_CODE_BUFFER_SIZE));
    ASSERT_EQ(xnn_status_success, igemm_generator(&code_buffer, n() % nr(), k(), ks() * mr() * sizeof(void *), nullptr));
    xnn_qs8_igemm_minmax_ukernel_function igemm = reinterpret_cast<xnn_qs8_igemm_minmax_ukernel_function>(code_buffer.code);

    igemm(
      m(), n(), k(), ks() * mr() * sizeof(void*),
      im2col.data(), packed_w.data(),
      c.data(), cm_stride() * sizeof(int8_t), cn_stride() * sizeof(int8_t),
      a_offset() * sizeof(uint8_t), zero_pointer,
      &quantization_params);

    ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&code_buffer));

    for (size_t m_index = 0; m_index < m(); m_index++) {
      for (size_t n_index = 0; n_index < n(); n_index++) {
        c_ref[m_index * n() + n_index] = requantize(
          acc[m_index * n() + n_index], requantization_scale, c_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
      }
    }

    for (size_t i = 0; i < m(); i++) {
      for (size_t j = 0; j < n(); j++) {
        ASSERT_LE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmax()) - 0x80);
        ASSERT_GE(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(qmin()) - 0x80);
        ASSERT_EQ(int32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), int32_t(c_ref[i * n() + j]))
            << "at " << i << ", " << j << ": reference = " << uint32_t(c_ref[i * n() + j])
            << " (accumulator = " << acc[i * n() + j]
            << "), optimized = " << (uint32_t) c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
            << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k()
            << ", requantization scale = " << requantization_scale << ", output zero point = " << int32_t(c_zero_point);
      }
    }
  }
}

#endif  // XNN_PLATFORM_JIT
