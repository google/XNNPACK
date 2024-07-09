// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef __XNNPACK_TEST_PACKQ_MICROKERNEL_TESTER_H
#define __XNNPACK_TEST_PACKQ_MICROKERNEL_TESTER_H

#include <cassert>
#include <cstddef>
#include <cstdlib>

#include "xnnpack/aligned-allocator.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/pack.h"

namespace xnnpack {

class PackQMicrokernelTester {
 public:
  PackQMicrokernelTester&m(size_t m) {
    this->m_ = m;
    return *this;
  }

  size_t m() const { return this->m_; }

  PackQMicrokernelTester& k(size_t k) {
    assert(k != 0);
    this->k_ = k;
    return *this;
  }

  size_t k() const { return this->k_; }

  PackQMicrokernelTester& mr(size_t mr) {
    this->mr_ = mr;
    return *this;
  }

  size_t mr() const { return this->mr_; }

  PackQMicrokernelTester& kr(size_t kr) {
    this->kr_ = kr;
    return *this;
  }

  size_t kr() const { return this->kr_; }

  PackQMicrokernelTester& sr(size_t sr) {
    this->sr_ = sr;
    return *this;
  }

  size_t sr() const { return this->sr_; }

  size_t packed_k() const { return round_up_po2(k(), kr() * sr()); }

  size_t packed_m() const { return round_up(m(), mr()); }

  PackQMicrokernelTester& nullbias(bool nullbias) {
    this->nullbias_ = nullbias;
    return *this;
  }

  bool nullbias() const { return this->nullbias_; }

  void Test(xnn_x8_packq_f32qp8_ukernel_fn packq) const;

 private:
  size_t m_{1};
  size_t k_{1};
  size_t mr_{1};
  size_t kr_{1};
  size_t sr_{1};
  bool nullbias_{false};
};

};  // namespace xnnpack

#endif  // __XNNPACK_TEST_PACKQ_MICROKERNEL_TESTER_H
