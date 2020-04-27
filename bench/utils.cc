// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#ifdef __linux__
  #include <sched.h>
#endif
#if defined(__ANDROID__) || defined(_WIN32) || defined(__CYGWIN__)
  #include <malloc.h>
#endif
#if defined(__SSE__) || defined(__x86_64__)
  #include <xmmintrin.h>
#endif

#include <cpuinfo.h>

#include "bench/utils.h"


static void* wipe_buffer = nullptr;
static size_t wipe_buffer_size = 0;

static std::once_flag wipe_buffer_guard;

static void InitWipeBuffer() {
  // Default: the largest know cache size (128 MB Intel Crystalwell L4 cache).
  wipe_buffer_size = 128 * 1024 * 1024;
  if (cpuinfo_initialize()) {
    wipe_buffer_size = benchmark::utils::GetMaxCacheSize();
  }
#if defined(_WIN32)
  wipe_buffer = _aligned_malloc(wipe_buffer_size, 128);
#elif defined(__ANDROID__) || defined(__CYGWIN__)
  // memalign is obsolete, but it is the only option on Android until API level 17.
  wipe_buffer = memalign(128, wipe_buffer_size);
#else
  (void) posix_memalign((void**) &wipe_buffer, 128, wipe_buffer_size);
#endif
  if (wipe_buffer != nullptr) {
    memset(wipe_buffer, 0xA5, wipe_buffer_size);
  }
}

namespace benchmark {
namespace utils {

uint32_t PrefetchToL1(const void* ptr, size_t size) {
  uint32_t step = 16;
  if (cpuinfo_initialize()) {
    step = cpuinfo_get_l1d_cache(0)->line_size;
  }
  const uint8_t* u8_ptr = static_cast<const uint8_t*>(ptr);
  // Compute and return sum of data to prevent compiler from removing data reads.
  uint32_t sum = 0;
  while (size >= step) {
    sum += uint32_t(*u8_ptr);
    u8_ptr += step;
    size -= step;
  }
  return sum;
}

uint32_t WipeCache() {
  std::call_once(wipe_buffer_guard, InitWipeBuffer);
  return PrefetchToL1(wipe_buffer, wipe_buffer_size);
}

void DisableDenormals() {
#if defined(__SSE__) || defined(__x86_64__)
  _mm_setcsr(_mm_getcsr() | 0x8040);
#elif defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
  uint32_t fpscr;
  __asm__ __volatile__(
      "VMRS %[fpscr], fpscr\n"
      "ORR %[fpscr], #0x1000000\n"
      "VMSR fpscr, %[fpscr]\n"
    : [fpscr] "=r" (fpscr));
#elif defined(__aarch64__)
  uint64_t fpcr;
  __asm__ __volatile__(
      "MRS %[fpcr], fpcr\n"
      "ORR %w[fpcr], %w[fpcr], 0x1000000\n"
      "ORR %w[fpcr], %w[fpcr], 0x80000\n"
      "MSR fpcr, %[fpcr]\n"
    : [fpcr] "=r" (fpcr));
#endif
}

// Return clockrate in Hz
uint64_t GetCurrentCpuFrequency() {
#ifdef __linux__
  int freq = 0;
  char cpuinfo_name[512];
  int cpu = sched_getcpu();
  snprintf(cpuinfo_name, sizeof(cpuinfo_name),
    "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);

  FILE* f = fopen(cpuinfo_name, "r");
  if (f) {
    if (fscanf(f, "%d", &freq)) {
      fclose(f);
      return uint64_t(freq) * 1000;
    }
    fclose(f);
  }
#endif  // __linux__
  return 0;
}

size_t GetMaxCacheSize() {
  if (!cpuinfo_initialize()) {
    #if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
      // DynamIQ max: 4 MB
      return 4 * 1024 * 1024;
    #else
      // Intel eDRAM max: 128 MB
      return 128 * 1024 * 1024;
    #endif
  }
  return cpuinfo_get_max_cache_size();
}

void MultiThreadingParameters(benchmark::internal::Benchmark* benchmark) {
  benchmark->ArgName("T");

  // Disabled thread pool (execution on the caller thread only).
  benchmark->Arg(1);

  if (cpuinfo_initialize()) {
    // All cores except the little ones.
    uint32_t max_cores = cpuinfo_get_cores_count();
    if (cpuinfo_get_clusters_count() > 1) {
      max_cores -= cpuinfo_get_cluster(cpuinfo_get_clusters_count() - 1)->core_count;
    }
    for (uint32_t t = 2; t <= max_cores; t++) {
      benchmark->Arg(t);
    }

    // All cores (if more than one cluster).
    if (cpuinfo_get_cores_count() > max_cores) {
      benchmark->Arg(cpuinfo_get_cores_count());
    }

    // All cores + hyperthreads (only if hyperthreading supported).
    if (cpuinfo_get_processors_count() > cpuinfo_get_cores_count()) {
      benchmark->Arg(cpuinfo_get_processors_count());
    }
  }
}


bool CheckNEON(benchmark::State& state) {
  if (!cpuinfo_initialize() || !cpuinfo_has_arm_neon()) {
    state.SkipWithError("no NEON extension");
    return false;
  }
  return true;
}

bool CheckNEONFMA(benchmark::State& state) {
  if (!cpuinfo_initialize() || !cpuinfo_has_arm_neon_fma()) {
    state.SkipWithError("no NEON-FMA extension");
    return false;
  }
  return true;
}

bool CheckSSE41(benchmark::State& state) {
  if (!cpuinfo_initialize() || !cpuinfo_has_x86_sse4_1()) {
    state.SkipWithError("no SSE4.1 extension");
    return false;
  }
  return true;
}

bool CheckAVX(benchmark::State& state) {
  if (!cpuinfo_initialize() || !cpuinfo_has_x86_avx()) {
    state.SkipWithError("no AVX extension");
    return false;
  }
  return true;
}

bool CheckFMA3(benchmark::State& state) {
  if (!cpuinfo_initialize() || !cpuinfo_has_x86_fma3()) {
    state.SkipWithError("no FMA3 extension");
    return false;
  }
  return true;
}

bool CheckAVX2(benchmark::State& state) {
  if (!cpuinfo_initialize() || !cpuinfo_has_x86_avx2()) {
    state.SkipWithError("no AVX2 extension");
    return false;
  }
  return true;
}

bool CheckAVX512F(benchmark::State& state) {
  if (!cpuinfo_initialize() || !cpuinfo_has_x86_avx512f()) {
    state.SkipWithError("no AVX512F extension");
    return false;
  }
  return true;
}

}  // namespace utils
}  // namespace benchmark
