// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "bench/utils.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
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

#if XNN_ENABLE_CPUINFO
#include <cpuinfo.h>
#endif  // XNN_ENABLE_CPUINFO

#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"
#include <benchmark/benchmark.h>
#include <pthreadpool.h>

// Common flags for all benchmarks.
int FLAGS_num_threads = 1;
int FLAGS_batch_size = 1;
uint32_t FLAGS_xnn_runtime_flags = 0;
uint32_t FLAGS_benchmark_min_iters = 1;

namespace benchmark {
namespace utils {
namespace {

static void* wipe_buffer = nullptr;
static size_t wipe_buffer_size = 0;

static std::once_flag wipe_buffer_guard;

static void InitWipeBuffer() {
#if XNN_ENABLE_CPUINFO
  if (cpuinfo_initialize()) {
    wipe_buffer_size = GetMaxCacheSize();
  }
#endif  // XNN_ENABLE_CPUINFO
  if (wipe_buffer_size == 0) {
    return;
  }

#if defined(_WIN32)
  wipe_buffer = _aligned_malloc(wipe_buffer_size, 128);
#elif defined(__ANDROID__) || defined(__CYGWIN__)
  // memalign is obsolete, but it is the only option on Android until API
  // level 17.
  wipe_buffer = memalign(128, wipe_buffer_size);
#else
  (void)posix_memalign((void**)&wipe_buffer, 128, wipe_buffer_size);
#endif
  if (wipe_buffer != nullptr) {
    memset(wipe_buffer, 0xA5, wipe_buffer_size);
  }
}

// Pthreadpool-compatible function to wipe the cache in each thread.
void PthreadpoolClearL2Cache(std::atomic<size_t>* counter, size_t id) {
#if XNN_ENABLE_CPUINFO
  static const size_t wipe_buffer_size = []() {
    const auto* l2_cache = cpuinfo_get_l2_cache(0);
    return l2_cache == nullptr ? 0 : l2_cache->size;
  }();
  static const char* wipe_buffer = wipe_buffer_size ? [&]() -> char* {
    char* const buff = (char*)malloc(wipe_buffer_size);
    memset(buff, 0xA5, wipe_buffer_size);
    return buff;
  }()
      : nullptr;
  if (wipe_buffer_size) {
    PrefetchToL1(wipe_buffer, wipe_buffer_size);
  } else {
    WipeCache();
  }
#else
  WipeCache();
#endif  // XNN_ENABLE_CPUINFO
  // Spin until all threads are done. This ensures that each thread calls this
  // function exactly once.
  counter->fetch_sub(1, std::memory_order_acquire);
  while (counter->load(std::memory_order_acquire) > 0) {
    std::atomic_thread_fence(std::memory_order_acquire);
  }
}

};  // namespace

int ProcessArgs(int& argc, char**& argv) {
  for (int i = 1; i < argc;) {
    if (strncmp(argv[i], "--num_threads=", 14) == 0) {
      FLAGS_num_threads = atoi(argv[i] + 14);
      if (FLAGS_num_threads <= 0) {
        std::cerr << "Invalid --num_threads: " << FLAGS_num_threads << "\n";
        return 1;
      }
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "--batch_size=", 13) == 0) {
      FLAGS_batch_size = atoi(argv[i] + 13);
      if (FLAGS_batch_size <= 0) {
        std::cerr << "Invalid --batch_size: " << FLAGS_batch_size << "\n";
        return 1;
      }
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "--xnn_runtime_flags=", 20) == 0) {
      const char* v = argv[i] + 20;
      if (strlen(v) > 2 && strncmp(v, "0x", 2) == 0) {
        FLAGS_xnn_runtime_flags = strtoul(v + 2, nullptr, 16);
      } else {
        FLAGS_xnn_runtime_flags = strtoul(v, nullptr, 10);
      }
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "--benchmark_min_iters=", 22) == 0) {
      FLAGS_benchmark_min_iters = atoi(argv[i] + 22);
      if (FLAGS_benchmark_min_iters <= 0) {
        std::cerr << "Invalid --benchmark_min_iters: "
                  << FLAGS_benchmark_min_iters << "\n";
        return 1;
      }
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else {
      ++i;
    }
  }
#if !XNN_PLATFORM_QURT
  // InitGoogle(...);
#endif
  return 0;
}

uint32_t PrefetchToL1(const void* ptr, size_t size) {
  uint32_t step = 16;
#if XNN_ENABLE_CPUINFO
  if (cpuinfo_initialize()) {
    const struct cpuinfo_cache* cpuinfo_cache_info = cpuinfo_get_l1d_cache(0);
    if (cpuinfo_cache_info) {
      step = cpuinfo_cache_info->line_size;
    }
  }
#endif  // XNN_ENABLE_CPUINFO

  const uint8_t* u8_ptr = static_cast<const uint8_t*>(ptr);
  // Compute and return sum of data to prevent compiler from removing data
  // reads.
  uint32_t sum = 0;
  while (size >= step) {
    sum += uint32_t(*u8_ptr);
    u8_ptr += step;
    size -= step;
  }
  return sum;
}

void WipePthreadpoolL2Caches(benchmark::State& state,
                             pthreadpool_t threadpool) {
  state.PauseTiming();
  std::atomic<size_t> counter(pthreadpool_get_threads_count(threadpool));
  pthreadpool_parallelize_1d(
      threadpool, (pthreadpool_task_1d_t)PthreadpoolClearL2Cache, &counter,
      pthreadpool_get_threads_count(threadpool), 0);
  state.ResumeTiming();
}

uint32_t WipeCache() {
  std::call_once(wipe_buffer_guard, InitWipeBuffer);
  if (!wipe_buffer) {
    return 0;
  }
  return PrefetchToL1(wipe_buffer, wipe_buffer_size);
}

void DisableDenormals() {
#if defined(__SSE__) || defined(__x86_64__)
  _mm_setcsr(_mm_getcsr() | 0x8040);
#elif defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
  uint32_t fpscr;
#if defined(__thumb__) && !defined(__thumb2__)
  __asm__ __volatile__(
      "VMRS %[fpscr], fpscr\n"
      "ORRS %[fpscr], %[bitmask]\n"
      "VMSR fpscr, %[fpscr]\n"
      : [fpscr] "=l"(fpscr)
      : [bitmask] "l"(0x1000000)
      : "cc");
#else
  __asm__ __volatile__(
      "VMRS %[fpscr], fpscr\n"
      "ORR %[fpscr], #0x1000000\n"
      "VMSR fpscr, %[fpscr]\n"
      : [fpscr] "=r"(fpscr));
#endif
#elif defined(__aarch64__)
  uint64_t fpcr;
  __asm__ __volatile__(
      "MRS %[fpcr], fpcr\n"
      "ORR %w[fpcr], %w[fpcr], 0x1000000\n"
      "ORR %w[fpcr], %w[fpcr], 0x80000\n"
      "MSR fpcr, %[fpcr]\n"
      : [fpcr] "=r"(fpcr));
#endif
}

// Return clock rate in Hz.
uint64_t GetCurrentCpuFrequency() {
#ifdef __linux__
  int freq = 0;
  char cpuinfo_name[512];
  int cpu = sched_getcpu();
  snprintf(cpuinfo_name, sizeof(cpuinfo_name),
           "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);

  FILE* f = fopen(cpuinfo_name, "r");
  if (f != nullptr) {
    if (fscanf(f, "%d", &freq) != 0) {
      fclose(f);
      return static_cast<uint64_t>(freq) * 1000;
    }
    fclose(f);
  }
#endif  // __linux__
  return 0;
}

size_t GetMaxCacheSize() {
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  // DynamIQ max: 4 MB
  size_t max_cache_size = 4 * 1024 * 1024;
#else
  // Intel eDRAM max: 128 MB
  size_t max_cache_size = 128 * 1024 * 1024;
#endif
#if XNN_ENABLE_CPUINFO
  if (cpuinfo_initialize()) {
    max_cache_size = cpuinfo_get_max_cache_size();
  }
#endif  // XNN_ENABLE_CPUINFO
  return max_cache_size;
}

bool CheckArchFlags(benchmark::State& state, uint64_t arch_flags) {
  const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == nullptr) {
    state.SkipWithError("no hardware config");
    return false;
  } 
  else if ((hardware_config->arch_flags & arch_flags) != arch_flags) {
    state.SkipWithError("architecture unsupported");
    return false;
  }

  return true;
}

}  // namespace utils
}  // namespace benchmark
