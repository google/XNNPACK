// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/pack-lh.h"

namespace {

#if XNN_ARCH_ARM64 && (XNN_ENABLE_ARM_SME || XNN_ENABLE_ARM_SME2) && \
    XNN_ENABLE_ARM_SME_ACLE

using PackLhFn = void (*)(size_t, size_t, size_t, size_t, size_t, size_t,
                          const int8_t*, size_t, void*);
using PackLhSizeFn = size_t (*)(size_t, size_t, size_t, size_t, size_t);
using PackLhOffsetFn = size_t (*)(size_t, size_t, size_t, size_t, size_t);

void ReferencePackLh(size_t m, size_t k, size_t mr, const int8_t* lhs,
                     size_t lhs_stride, int8_t* packed) {
  if (mr == 1) {
    for (size_t row = 0; row < m; row++) {
      std::memcpy(packed + row * k, lhs + row * lhs_stride, k);
    }
    return;
  }
  const size_t k_rounded = (k + 3) / 4 * 4;
  const size_t m_rounded = (m + mr - 1) / mr * mr;
  for (size_t block = 0; block < m_rounded; block += mr) {
    for (size_t group = 0; group < k_rounded / 4; group++) {
      for (size_t row = 0; row < mr; row++) {
        for (size_t j = 0; j < 4; j++) {
          const size_t kk = 4 * group + j;
          const bool valid = block + row < m && kk < k;
          packed[block * k_rounded + group * 4 * mr + row * 4 + j] =
              valid ? lhs[(block + row) * lhs_stride + kk] : 0;
        }
      }
    }
  }
}

void TestPackLh(PackLhFn pack_fn, PackLhSizeFn size_fn, size_t m, size_t k,
                size_t lhs_stride, size_t mr) {
  const size_t packed_size = size_fn(m, k, mr, 4, 1);
  const size_t lhs_count = m == 0 ? 0 : (m - 1) * lhs_stride + k;
  xnnpack::Buffer<int8_t> lhs(lhs_count);
  for (size_t i = 0; i < lhs_count; i++) {
    lhs[i] = static_cast<int8_t>((i * 131 + 7) & 0xFF);
  }
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed(packed_size);
  xnnpack::Buffer<int8_t> expected(packed_size);
  std::fill(packed.begin(), packed.end(), static_cast<int8_t>(0x5A));
  std::fill(expected.begin(), expected.end(), static_cast<int8_t>(0xA5));
  ReferencePackLh(m, k, mr, lhs.data(), lhs_stride, expected.data());
  pack_fn(m, k, mr, 4, 1, 0, lhs.data(), lhs_stride, packed.data());
  for (size_t i = 0; i < packed_size; i++) {
    ASSERT_EQ(packed[i], expected[i])
        << "m=" << m << " k=" << k << " lhs_stride=" << lhs_stride
        << " mr=" << mr << " at byte " << i;
  }
}

void TestPackLhOffset(PackLhFn pack_fn, PackLhSizeFn size_fn,
                      PackLhOffsetFn offset_fn, size_t m, size_t k,
                      size_t lhs_stride, size_t mr) {
  const size_t packed_size = size_fn(m, k, mr, 4, 1);
  const size_t lhs_count = m == 0 ? 0 : (m - 1) * lhs_stride + k;
  xnnpack::Buffer<int8_t> lhs(lhs_count);
  for (size_t i = 0; i < lhs_count; i++) {
    lhs[i] = static_cast<int8_t>((i * 131 + 7) & 0xFF);
  }
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed(packed_size);
  std::fill(packed.begin(), packed.end(), static_cast<int8_t>(0x5A));
  pack_fn(m, k, mr, 4, 1, 0, lhs.data(), lhs_stride, packed.data());

  ASSERT_EQ(offset_fn(0, k, mr, 4, 1), 0u);
  for (size_t row = mr; row < m; row += mr) {
    const size_t off = offset_fn(row, k, mr, 4, 1);
    const size_t rem_m = m - row;
    const size_t rem_size = size_fn(rem_m, k, mr, 4, 1);
    ASSERT_EQ(off + rem_size, packed_size)
        << "m=" << m << " k=" << k << " mr=" << mr << " row=" << row;
    xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> sub_packed(rem_size);
    std::fill(sub_packed.begin(), sub_packed.end(), static_cast<int8_t>(0xA5));
    pack_fn(rem_m, k, mr, 4, 1, 0, lhs.data() + row * lhs_stride, lhs_stride,
            sub_packed.data());
    for (size_t i = 0; i < rem_size; i++) {
      ASSERT_EQ(packed[off + i], sub_packed[i])
          << "offset slice mismatch at m=" << m << " k=" << k << " mr=" << mr
          << " row=" << row << " byte=" << i;
    }
  }
}

std::vector<size_t> Ks(size_t vl) {
  return {1,          2,          3,          4,           5,
          7,          4 * vl - 1, 4 * vl,     4 * vl + 1,  4 * vl + 5,
          8 * vl - 1, 8 * vl,     8 * vl + 3, 12 * vl + 2, 16 * vl + 9};
}

void TestMrEq1(PackLhFn pack_fn, PackLhSizeFn size_fn, size_t vl) {
  for (size_t m = 1; m <= 5; m++) {
    for (size_t k : Ks(vl)) {
      TestPackLh(pack_fn, size_fn, m, k, k, /*mr=*/1);
    }
  }
}

void TestMLeVl(PackLhFn pack_fn, PackLhSizeFn size_fn, size_t mr) {
  const size_t vl = mr / 2;
  for (size_t m = 1; m <= vl; m++) {
    for (size_t k : Ks(vl)) {
      TestPackLh(pack_fn, size_fn, m, k, k, mr);
    }
  }
}

void TestMGtVl(PackLhFn pack_fn, PackLhSizeFn size_fn, size_t mr) {
  const size_t vl = mr / 2;
  for (size_t m = vl + 1; m <= mr; m++) {
    for (size_t k : Ks(vl)) {
      TestPackLh(pack_fn, size_fn, m, k, k, mr);
    }
  }
}

void TestMGtMr(PackLhFn pack_fn, PackLhSizeFn size_fn, size_t mr) {
  const size_t vl = mr / 2;
  for (size_t m :
       {mr + 1, mr + vl, mr + vl + 3, 2 * mr, 2 * mr + 1, 3 * mr + 5}) {
    for (size_t k : Ks(vl)) {
      TestPackLh(pack_fn, size_fn, m, k, k, mr);
    }
  }
}

void TestStridedLhs(PackLhFn pack_fn, PackLhSizeFn size_fn, size_t mr) {
  const size_t vl = mr / 2;
  for (size_t m : {static_cast<size_t>(1), static_cast<size_t>(3), vl - 1,
                   vl + 1, mr, mr + 1, mr + vl + 3, 2 * mr + 3}) {
    for (size_t k : Ks(vl)) {
      TestPackLh(pack_fn, size_fn, m, k, k + 1, mr);
      TestPackLh(pack_fn, size_fn, m, k, k + 13, mr);
      TestPackLh(pack_fn, size_fn, m, k, k + 13, /*mr=*/1);
    }
  }
}

void TestKLarge(PackLhFn pack_fn, PackLhSizeFn size_fn, size_t mr) {
  const size_t vl = mr / 2;
  for (size_t m : {static_cast<size_t>(1), vl - 1, mr, mr + vl + 1}) {
    for (size_t k : {256 * vl - 1, 256 * vl, 256 * vl + 3,
                     256 * vl + 12 * vl + 7, 272 * vl}) {
      TestPackLh(pack_fn, size_fn, m, k, k, mr);
    }
  }
}

void TestOffsets(PackLhFn pack_fn, PackLhSizeFn size_fn,
                 PackLhOffsetFn offset_fn, size_t mr) {
  const size_t vl = mr / 2;
  for (size_t k : {static_cast<size_t>(1), static_cast<size_t>(7), 4 * vl - 1,
                   4 * vl, 8 * vl + 3}) {
    TestPackLhOffset(pack_fn, size_fn, offset_fn, /*m=*/4, k, k + 5, /*mr=*/1);
    for (size_t m : {mr + 1, mr + vl + 3, 2 * mr, 3 * mr + 5}) {
      TestPackLhOffset(pack_fn, size_fn, offset_fn, m, k, k + 5, mr);
    }
  }
}

void TestZeroMOrK(PackLhFn pack_fn, PackLhSizeFn size_fn, size_t mr) {
  xnnpack::Buffer<int8_t> lhs(64);
  for (size_t i = 0; i < lhs.size(); i++) {
    lhs[i] = 1;
  }
  for (size_t mr_value : {static_cast<size_t>(1), mr}) {
    ASSERT_EQ(size_fn(0, 16, mr_value, 4, 1), 0u);
    ASSERT_EQ(size_fn(3, 0, mr_value, 4, 1), 0u);
    xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_zero_m(0);
    pack_fn(0, 16, mr_value, 4, 1, 0, lhs.data(), 16, packed_zero_m.data());
    xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_zero_k(0);
    pack_fn(3, 0, mr_value, 4, 1, 0, lhs.data(), 16, packed_zero_k.data());
  }
}

#if XNN_ENABLE_ARM_SME2
size_t MrSme2() {
  return xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__sme2_get_mr();
}

TEST(X8_PACK_LH__SME2, mr_eq_1) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme2);
  TestMrEq1(xnn_x8_pack_lh_ukernel__sme2, xnn_x8_pack_lh_size__sme2,
            MrSme2() / 2);
}

TEST(X8_PACK_LH__SME2, m_le_vl) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme2);
  TestMLeVl(xnn_x8_pack_lh_ukernel__sme2, xnn_x8_pack_lh_size__sme2, MrSme2());
}

TEST(X8_PACK_LH__SME2, m_gt_vl) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme2);
  TestMGtVl(xnn_x8_pack_lh_ukernel__sme2, xnn_x8_pack_lh_size__sme2, MrSme2());
}

TEST(X8_PACK_LH__SME2, m_gt_mr) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme2);
  TestMGtMr(xnn_x8_pack_lh_ukernel__sme2, xnn_x8_pack_lh_size__sme2, MrSme2());
}

TEST(X8_PACK_LH__SME2, strided_lhs) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme2);
  TestStridedLhs(xnn_x8_pack_lh_ukernel__sme2, xnn_x8_pack_lh_size__sme2,
                 MrSme2());
}

TEST(X8_PACK_LH__SME2, k_large) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme2);
  TestKLarge(xnn_x8_pack_lh_ukernel__sme2, xnn_x8_pack_lh_size__sme2, MrSme2());
}

TEST(X8_PACK_LH__SME2, packed_offset) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme2);
  TestOffsets(xnn_x8_pack_lh_ukernel__sme2, xnn_x8_pack_lh_size__sme2,
              xnn_x8_pack_lh_offset__sme2, MrSme2());
}

TEST(X8_PACK_LH__SME2, zero_m_or_k) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme2);
  TestZeroMOrK(xnn_x8_pack_lh_ukernel__sme2, xnn_x8_pack_lh_size__sme2,
               MrSme2());
}
#endif  // XNN_ENABLE_ARM_SME2

#if XNN_ENABLE_ARM_SME
size_t MrSme() {
  return xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__sme_get_mr();
}

TEST(X8_PACK_LH__SME, mr_eq_1) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme);
  TestMrEq1(xnn_x8_pack_lh_ukernel__sme, xnn_x8_pack_lh_size__sme, MrSme() / 2);
}

TEST(X8_PACK_LH__SME, m_le_vl) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme);
  TestMLeVl(xnn_x8_pack_lh_ukernel__sme, xnn_x8_pack_lh_size__sme, MrSme());
}

TEST(X8_PACK_LH__SME, m_gt_vl) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme);
  TestMGtVl(xnn_x8_pack_lh_ukernel__sme, xnn_x8_pack_lh_size__sme, MrSme());
}

TEST(X8_PACK_LH__SME, m_gt_mr) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme);
  TestMGtMr(xnn_x8_pack_lh_ukernel__sme, xnn_x8_pack_lh_size__sme, MrSme());
}

TEST(X8_PACK_LH__SME, strided_lhs) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme);
  TestStridedLhs(xnn_x8_pack_lh_ukernel__sme, xnn_x8_pack_lh_size__sme,
                 MrSme());
}

TEST(X8_PACK_LH__SME, k_large) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme);
  TestKLarge(xnn_x8_pack_lh_ukernel__sme, xnn_x8_pack_lh_size__sme, MrSme());
}

TEST(X8_PACK_LH__SME, packed_offset) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme);
  TestOffsets(xnn_x8_pack_lh_ukernel__sme, xnn_x8_pack_lh_size__sme,
              xnn_x8_pack_lh_offset__sme, MrSme());
}

TEST(X8_PACK_LH__SME, zero_m_or_k) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme);
  TestZeroMOrK(xnn_x8_pack_lh_ukernel__sme, xnn_x8_pack_lh_size__sme, MrSme());
}
#endif  // XNN_ENABLE_ARM_SME

#if XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME2
TEST(X8_PACK_LH__SME_VS_SME2, byte_identical) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme | xnn_arch_arm_sme2);
  const size_t mr = MrSme();
  ASSERT_EQ(mr, MrSme2());
  const size_t vl = mr / 2;
  for (size_t m : {static_cast<size_t>(1), vl - 1, vl, vl + 1, mr - 1, mr,
                   mr + 1, mr + vl + 3, 2 * mr + 5}) {
    for (size_t k : {static_cast<size_t>(1), static_cast<size_t>(7), 4 * vl - 1,
                     4 * vl + 5, 8 * vl + 3, 256 * vl + 3}) {
      const size_t stride = k + 7;
      const size_t sz1 = xnn_x8_pack_lh_size__sme(m, k, mr, 4, 1);
      const size_t sz2 = xnn_x8_pack_lh_size__sme2(m, k, mr, 4, 1);
      ASSERT_EQ(sz1, sz2);
      ASSERT_EQ(xnn_x8_pack_lh_offset__sme(m, k, mr, 4, 1),
                xnn_x8_pack_lh_offset__sme2(m, k, mr, 4, 1));
      const size_t lhs_count = (m - 1) * stride + k;
      xnnpack::Buffer<int8_t> lhs(lhs_count);
      for (size_t i = 0; i < lhs_count; i++) {
        lhs[i] = static_cast<int8_t>((i * 97 + 19) & 0xFF);
      }
      xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> p1(sz1);
      xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> p2(sz2);
      std::fill(p1.begin(), p1.end(), static_cast<int8_t>(0x5A));
      std::fill(p2.begin(), p2.end(), static_cast<int8_t>(0xA5));
      xnn_x8_pack_lh_ukernel__sme(m, k, mr, 4, 1, 0, lhs.data(), stride,
                                  p1.data());
      xnn_x8_pack_lh_ukernel__sme2(m, k, mr, 4, 1, 0, lhs.data(), stride,
                                   p2.data());
      for (size_t i = 0; i < sz1; i++) {
        ASSERT_EQ(p1[i], p2[i]) << "m=" << m << " k=" << k << " byte=" << i;
      }
    }
  }
}
#endif  // XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME2

#endif  // XNN_ARCH_ARM64 && (XNN_ENABLE_ARM_SME || XNN_ENABLE_ARM_SME2) &&
        // XNN_ENABLE_ARM_SME_ACLE

}  // namespace
