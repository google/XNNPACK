// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Paired, mutable-RHS graph benchmark. Internal structs are used only to audit
// dispatch and workspace; execution uses the public graph/runtime APIs.
#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pthreadpool.h"
#include "include/xnnpack.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/subgraph.h"

namespace {
using Clock = std::chrono::steady_clock;
#if defined(__EXCEPTIONS) || defined(__cpp_exceptions)
struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void Check(xnn_status status) {
  if (status == xnn_status_unsupported_hardware)
    throw Unsupported("unsupported hardware");
  if (status != xnn_status_success)
    throw std::runtime_error("XNNPACK status " + std::to_string(status));
}
#else
void Require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}
void Check(xnn_status status) {
  if (status == xnn_status_unsupported_hardware) {
    std::cerr << "SKIP: unsupported hardware\n";
    std::exit(77);
  }
  if (status != xnn_status_success) {
    std::cerr << "FAIL: XNNPACK status " << status << '\n';
    std::exit(1);
  }
}
#endif
size_t Mul(size_t a, size_t b) {
  Require(b == 0 || a <= std::numeric_limits<size_t>::max() / b,
          "size overflow");
  return a * b;
}
uint32_t Hash(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352d;
  x ^= x >> 15;
  x *= 0x846ca68b;
  return x ^ (x >> 16);
}
std::string Csv(const std::string& value) {
  std::string result = "\"";
  for (char c : value) {
    if (c == '"') result += '"';
    result += c;
  }
  return result + '"';
}
struct Options {
  size_t m = 8, k = 512, n = 1024, ba = 1, bb = 1;
  size_t threads = 1, passes = 4, churn_bytes = 0;
  uint32_t seed = 1;
  bool transpose = true, positive = false, off_first = false;
  double ms = 60, max_pair_mib = 512;
  float scale = 0.03125f;
};
Options Parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string s(argv[i]);
    const size_t eq = s.find('=');
    Require(s.rfind("--", 0) == 0 && eq != std::string::npos,
            "use --key=value");
    const std::string key = s.substr(2, eq - 2), v = s.substr(eq + 1);
    Require(!v.empty() && v[0] != '-', "expected nonnegative value");
    if (key == "scale")
      o.scale = std::stof(v);
    else if (key == "ms")
      o.ms = std::stod(v);
    else if (key == "max-pair-mib")
      o.max_pair_mib = std::stod(v);
    else {
      size_t end = 0;
      const size_t number = std::stoull(v, &end);
      Require(end == v.size(), "invalid integer option");
      if (key == "m")
        o.m = number;
      else if (key == "k")
        o.k = number;
      else if (key == "n")
        o.n = number;
      else if (key == "batch-a")
        o.ba = number;
      else if (key == "batch-b")
        o.bb = number;
      else if (key == "threads")
        o.threads = number;
      else if (key == "passes")
        o.passes = number;
      else if (key == "churn-mib")
        o.churn_bytes = Mul(number, 1024 * 1024);
      else if (key == "seed")
        o.seed = number;
      else if (key == "transpose-b") {
        Require(number <= 1, "transpose-b must be 0/1");
        o.transpose = number;
      } else if (key == "positive-a") {
        Require(number <= 1, "positive-a must be 0/1");
        o.positive = number;
      } else if (key == "off-first") {
        Require(number <= 1, "off-first must be 0/1");
        o.off_first = number;
      } else {
        Require(false, ("unknown option: " + key).c_str());
      }
    }
  }
  Require(o.m && o.k && o.n && o.ba && o.bb, "dimensions must be positive");
  Require(o.ba == o.bb || o.ba == 1 || o.bb == 1, "incompatible batch sizes");
  Require(o.threads >= 1 && o.threads <= 256 && o.passes >= 2 &&
              o.passes <= 100 && o.passes % 2 == 0,
          "threads 1..256; passes must be even, 2..100");
  Require(std::isfinite(o.ms) && o.ms >= 10 && o.ms <= 10000 &&
              std::isfinite(o.scale) && o.scale > 0 &&
              std::isfinite(o.max_pair_mib) && o.max_pair_mib > 0,
          "invalid timing/scale/budget");
  return o;
}
struct Probe {
  Options o;
  bool rewritten;
  xnn_runtime_t runtime = nullptr;
  xnn_operator_t bmm = nullptr;
  std::vector<float> a, c;
  std::vector<int8_t> b;
  std::vector<xnn_external_value> bindings;
  double create_us = 0, reference_error = 0;
  size_t scale_bytes = 0;
  std::string kernel_offsets;

  Probe(Options options, bool rewrite, pthreadpool_t pool)
      : o(options), rewritten(rewrite) {
    a.resize(Mul(Mul(o.ba, o.m), o.k) + XNN_EXTRA_BYTES / sizeof(float));
    b.resize(Mul(Mul(o.bb, o.k), o.n) + XNN_EXTRA_BYTES);
    c.resize(Mul(Mul(std::max(o.ba, o.bb), o.m), o.n));
    for (size_t i = 0; i < a.size(); ++i) {
      const float v = float(int(Hash(uint32_t(i) + o.seed) % 1024) - 512) / 512;
      a[i] = o.positive ? (v + 1) / float(o.k) : v / std::sqrt(float(o.k));
    }
    for (size_t batch = 0; batch < o.bb; ++batch)
      for (size_t k = 0; k < o.k; ++k)
        for (size_t n = 0; n < o.n; ++n)
          b[batch * o.k * o.n + (o.transpose ? n * o.k + k : k * o.n + n)] =
              int(Hash(uint32_t((batch * o.k + k) * o.n + n) + o.seed + 43) %
                  255) -
              127;
    const size_t ash[] = {o.ba, o.m, o.k};
    const size_t bsh[] = {o.bb, o.transpose ? o.n : o.k,
                          o.transpose ? o.k : o.n};
    const size_t csh[] = {std::max(o.ba, o.bb), o.m, o.n};
    xnn_subgraph_t graph = nullptr;
    Check(xnn_create_subgraph(3, 0, &graph));
    std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> owner(
        graph, xnn_delete_subgraph);
    uint32_t ai, bi, fi, ci;
    Check(xnn_define_tensor_value(graph, xnn_datatype_fp32, 3, ash, nullptr, 0,
                                  XNN_VALUE_FLAG_EXTERNAL_INPUT, &ai));
    Check(xnn_define_quantized_tensor_value(
        graph, xnn_datatype_qint8, 0, o.scale, 3, bsh, nullptr, 1,
        XNN_VALUE_FLAG_EXTERNAL_INPUT, &bi));
    Check(xnn_define_tensor_value(graph, xnn_datatype_fp32, 3, bsh, nullptr,
                                  XNN_INVALID_VALUE_ID, 0, &fi));
    Check(xnn_define_tensor_value(graph, xnn_datatype_fp32, 3, csh, nullptr, 2,
                                  XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &ci));
    Check(xnn_define_unary(graph, xnn_unary_convert, nullptr, bi, fi, 0));
    Check(xnn_define_batch_matrix_multiply(
        graph, ai, fi, ci,
        (o.transpose ? XNN_FLAG_TRANSPOSE_B : 0) |
            (rewritten ? 0 : XNN_FLAG_NO_BMM_DEQUANTIZATION_FUSION)));
    const auto start = Clock::now();
    Check(xnn_create_runtime_v3(graph, nullptr, pool, 0, &runtime));
    create_us =
        std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    bindings = {{ai, a.data()}, {bi, b.data()}, {ci, c.data()}};
    Run();
    size_t count = 0;
    for (size_t i = 0; i < runtime->num_ops; ++i) {
      for (auto* op : runtime->opdata[i].operator_objects) {
        if (!op) continue;
        scale_bytes += op->channelwise_quantization_buffer_capacity;
        if (runtime->opdata[i].type != xnn_node_type_batch_matrix_multiply)
          continue;
        bmm = op;
        ++count;
      }
    }
    Require(count == 1, "expected one BMM");
    const bool mixed =
        bmm->type == xnn_operator_type_batch_matrix_multiply_nc_f32_qc8w;
    const bool fp32 =
        bmm->type == xnn_operator_type_batch_matrix_multiply_nc_f32 ||
        bmm->type == xnn_operator_type_batch_matrix_multiply_nc_pf32;
    Require(rewritten ? mixed : fp32,
            "unexpected dispatch; do not classify this run as an on/off "
            "comparison");
    Require(scale_bytes == (rewritten ? o.bb * o.n * sizeof(float) : 0),
            "unexpected scale allocation");
    const auto& ctx = bmm->dynamic_context.gemm->gemm;
    for (size_t i = 0; i < XNN_MAX_UARCH_TYPES; ++i) {
      const void* function =
          reinterpret_cast<const void*>(ctx.ukernel.function[i]);
      if (!function) continue;
#if !defined(_WIN32)
      Dl_info info{};
      Require(dladdr(function, &info) != 0, "cannot identify kernel image");
      const uintptr_t offset = reinterpret_cast<uintptr_t>(function) -
                               reinterpret_cast<uintptr_t>(info.dli_fbase);
      if (!kernel_offsets.empty()) kernel_offsets += ';';
      kernel_offsets += std::to_string(i) + ':' + std::to_string(offset);
#else
      if (!kernel_offsets.empty()) kernel_offsets += ';';
      kernel_offsets += std::to_string(i) + ":0";
#endif
    }
    Validate();
  }
  ~Probe() {
    if (runtime) xnn_delete_runtime(runtime);
  }
  void Run() {
    Check(xnn_reshape_runtime(runtime));
    Check(xnn_setup_runtime_v2(runtime, bindings.size(), bindings.data()));
    Invoke();
  }
  void Invoke() { Check(xnn_invoke_runtime(runtime)); }
  void Validate() {
    for (size_t sample = 0; sample < std::min<size_t>(256, c.size());
         ++sample) {
      const size_t index = sample == 0 ? 0
                           : sample == 1
                               ? c.size() - 1
                               : Hash(uint32_t(sample + 11)) % c.size();
      const size_t batch = index / (o.m * o.n), row = index / o.n % o.m,
                   col = index % o.n;
      double ref = 0;
      for (size_t k = 0; k < o.k; ++k) {
        const float bv =
            float(b[(o.bb == 1 ? 0 : batch * o.k * o.n) +
                    (o.transpose ? col * o.k + k : k * o.n + col)]) *
            o.scale;
        ref += double(a[((o.ba == 1 ? 0 : batch) * o.m + row) * o.k + k]) * bv;
      }
      const double error = std::abs(double(c[index]) - ref);
      Require(std::isfinite(c[index]) && error <= 1e-4 + 5e-5 * std::abs(ref),
              "FP64 reference check failed");
      reference_error = std::max(reference_error, error);
    }
  }
};
struct Timing {
  double us;
  size_t calls;
};
template <class Fn>
Timing Measure(Fn fn, double target_ms, std::vector<uint8_t>& churn) {
  for (int i = 0; i < 3; ++i) fn();
  const auto start = Clock::now();
  double elapsed_us = 0;
  size_t calls = 0;
  do {
    // Deliberate cache pressure, outside the measured call. This is not a
    // hardware cache flush and is labeled "churn", not "cold cache".
    volatile uint8_t* bytes = churn.data();
    for (size_t i = 0; i < churn.size(); i += 64)
      bytes[i] = uint8_t(bytes[i] + 1);
    const auto begin = Clock::now();
    fn();
    elapsed_us +=
        std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
    ++calls;
  } while (
      calls < 3 ||
      std::chrono::duration<double, std::milli>(Clock::now() - start).count() <
          target_ms);
  return {elapsed_us / calls, calls};
}
}  // namespace

int main(int argc, char** argv) {
#if defined(__EXCEPTIONS) || defined(__cpp_exceptions)
  try {
#endif
    const Options o = Parse(argc, argv);
    const size_t a_bytes = Mul(Mul(Mul(o.ba, o.m), o.k), sizeof(float));
    const size_t b_bytes = Mul(Mul(o.bb, o.k), o.n);
    const size_t c_bytes =
        Mul(Mul(Mul(std::max(o.ba, o.bb), o.m), o.n), sizeof(float));
    // Deliberately conservative planning estimate; not a measured RSS limit.
    const double estimated =
        (4.0 * a_bytes + 16.0 * b_bytes + 4.0 * c_bytes + o.churn_bytes) /
            1048576 +
        16;
#if defined(__EXCEPTIONS) || defined(__cpp_exceptions)
    if (estimated > o.max_pair_mib)
      throw Unsupported("estimated paired allocation exceeds --max-pair-mib");
#else
  if (estimated > o.max_pair_mib) {
    std::cerr << "SKIP: estimated paired allocation exceeds --max-pair-mib\n";
    return 77;
  }
#endif
    Check(xnn_initialize(nullptr));
    std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> pool(
        pthreadpool_create(o.threads), pthreadpool_destroy);
    Require(bool(pool), "threadpool creation failed");
    std::unique_ptr<Probe> probes[2];
    // Vary construction order as well as measurement order across processes.
    for (size_t step = 0; step < 2; ++step) {
      size_t index = (step + o.off_first) % 2;
      probes[index] = std::make_unique<Probe>(o, index == 0, pool.get());
    }
    double difference = 0;
    for (size_t i = 0; i < probes[0]->c.size(); ++i) {
      const double delta = std::abs(double(probes[0]->c[i]) - probes[1]->c[i]);
      Require(delta <= 1e-4 + 5e-5 * std::max(std::abs(probes[0]->c[i]),
                                              std::abs(probes[1]->c[i])),
              "paths differ beyond tolerance");
      difference = std::max(difference, delta);
    }
    std::vector<uint8_t> churn(o.churn_bytes, 1);
    std::cout
        << "pass,order,rewrite,m,k,n,batch_a,batch_b,transpose_b,positive_a,"
           "threads,scale,seed,churn_bytes,run_us,run_calls,invoke_us,invoke_"
           "calls,create_us,workspace_bytes,scale_bytes,external_bytes,rhs_"
           "bytes,operator_count,bmm_type,mr_config,mr_selected,nr,kernel_"
           "offsets,reference_max_abs,path_max_abs\n";
    for (size_t pass = 0; pass < o.passes; ++pass) {
      for (size_t step = 0; step < 2; ++step) {
        auto& p = *probes[(step + pass + o.off_first) % 2];
        const auto run = Measure([&] { p.Run(); }, o.ms, churn);
        const auto invoke = Measure([&] { p.Invoke(); }, o.ms, churn);
        p.Validate();
        const auto& ctx = p.bmm->dynamic_context.gemm->gemm;
        std::cout << std::setprecision(12) << pass << ',' << step << ','
                  << p.rewritten << ',' << o.m << ',' << o.k << ',' << o.n
                  << ',' << o.ba << ',' << o.bb << ',' << o.transpose << ','
                  << o.positive << ',' << o.threads << ',' << o.scale << ','
                  << o.seed << ',' << o.churn_bytes << ',' << run.us << ','
                  << run.calls << ',' << invoke.us << ',' << invoke.calls << ','
                  << p.create_us << ',' << p.runtime->workspace->size << ','
                  << p.scale_bytes << ',' << a_bytes + b_bytes + c_bytes << ','
                  << b_bytes << ',' << p.runtime->num_ops << ','
                  << Csv(xnn_operator_type_to_string(p.bmm->type)) << ','
                  << unsigned(p.bmm->gemm_config->mr) << ',' << ctx.mr << ','
                  << unsigned(p.bmm->gemm_config->nr) << ','
                  << Csv(p.kernel_offsets) << ',' << p.reference_error << ','
                  << difference << '\n'
                  << std::flush;
      }
    }
    std::cerr << "PASS: both dispatch paths, scales, and numerical checks\n";
#if defined(__EXCEPTIONS) || defined(__cpp_exceptions)
  } catch (const Unsupported& e) {
    std::cerr << "SKIP: " << e.what() << '\n';
    return 77;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
#endif
  return 0;
}
