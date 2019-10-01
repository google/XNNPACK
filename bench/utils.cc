// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <malloc.h>
#include <pthread.h>
#include <sched.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cpuinfo.h>

#include "bench/utils.h"


static void* wipe_buffer = nullptr;
static size_t wipe_buffer_size = 0;

static pthread_once_t wipeBufferGuard = PTHREAD_ONCE_INIT;

static void initWipeBuffer() {
  // Default: the largest know cache size (128 MB Intel Crystalwell L4 cache).
  wipe_buffer_size = 128 * 1024 * 1024;
  if (cpuinfo_initialize()) {
    wipe_buffer_size = benchmark::utils::GetMaxCacheSize();
  }
#if defined(__ANDROID__)
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

uint32_t prefetchToL1(const void* ptr, size_t size) {
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

uint32_t wipeCache() {
  pthread_once(&wipeBufferGuard, &initWipeBuffer);
  return prefetchToL1(wipe_buffer, wipe_buffer_size);
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
  const cpuinfo_processor* processor = cpuinfo_get_processor(0);
  #if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
    // There is no precise way to detect cache size on ARM/ARM64, and cache size reported by cpuinfo
    // may underestimate the actual cache size. Thus, we use microarchitecture-specific maximum.
    switch (processor->core->uarch) {
      case cpuinfo_uarch_xscale:
      case cpuinfo_uarch_arm11:
      case cpuinfo_uarch_scorpion:
      case cpuinfo_uarch_krait:
      case cpuinfo_uarch_kryo:
      case cpuinfo_uarch_exynos_m1:
      case cpuinfo_uarch_exynos_m2:
      case cpuinfo_uarch_exynos_m3:
        // cpuinfo-detected cache size always correct.
        break;
      case cpuinfo_uarch_cortex_a5:
        // Max observed (NXP Vybrid SoC)
        return 512 * 1024;
      case cpuinfo_uarch_cortex_a7:
        // Cortex-A7 MPCore Technical Reference Manual:
        // 7.1. About the L2 Memory system
        //   The L2 memory system consists of an:
        //    - Optional tightly-coupled L2 cache that includes:
        //      - Configurable L2 cache size of 128KB, 256KB, 512KB, and 1MB.
        return 1024 * 1024;
      case cpuinfo_uarch_cortex_a8:
        // Cortex-A8 Technical Reference Manual:
        // 8.1. About the L2 memory system
        //   The key features of the L2 memory system include:
        //    - configurable cache size of 0KB, 128KB, 256KB, 512KB, and 1MB
        return 1024 * 1024;
      case cpuinfo_uarch_cortex_a9:
        // Max observed (e.g. Exynos 4212)
        return 1024 * 1024;
      case cpuinfo_uarch_cortex_a12:
      case cpuinfo_uarch_cortex_a17:
        // ARM Cortex-A17 MPCore Processor Technical Reference Manual:
        // 7.1. About the L2 Memory system
        //   The key features of the L2 memory system include:
        //    - An integrated L2 cache:
        //      - The cache size is implemented as either 256KB, 512KB, 1MB, 2MB, 4MB or 8MB.
        return 8 * 1024 * 1024;
      case cpuinfo_uarch_cortex_a15:
        // ARM Cortex-A15 MPCore Processor Technical Reference Manual:
        // 7.1. About the L2 memory system
        //   The features of the L2 memory system include:
        //    - Configurable L2 cache size of 512KB, 1MB, 2MB and 4MB.
        return 4 * 1024 * 1024;
      case cpuinfo_uarch_cortex_a35:
        // ARM Cortex‑A35 Processor Technical Reference Manual:
        // 7.1 About the L2 memory system
        //   L2 cache
        //    - Further features of the L2 cache are:
        //      - Configurable size of 128KB, 256KB, 512KB, and 1MB.
        return 1024 * 1024;
      case cpuinfo_uarch_cortex_a53:
        // ARM Cortex-A53 MPCore Processor Technical Reference Manual:
        // 7.1. About the L2 memory system
        //   The L2 memory system consists of an:
        //    - Optional tightly-coupled L2 cache that includes:
        //      - Configurable L2 cache size of 128KB, 256KB, 512KB, 1MB and 2MB.
        return 2 * 1024 * 1024;
      case cpuinfo_uarch_cortex_a57:
        // ARM Cortex-A57 MPCore Processor Technical Reference Manual:
        // 7.1 About the L2 memory system
        //   The features of the L2 memory system include:
        //    - Configurable L2 cache size of 512KB, 1MB, and 2MB.
        return 2 * 1024 * 1024;
      case cpuinfo_uarch_cortex_a72:
        // ARM Cortex-A72 MPCore Processor Technical Reference Manual:
        // 7.1 About the L2 memory system
        //   The features of the L2 memory system include:
        //    - Configurable L2 cache size of 512KB, 1MB, 2MB and 4MB.
        return 4 * 1024 * 1024;
      case cpuinfo_uarch_cortex_a73:
        // ARM Cortex‑A73 MPCore Processor Technical Reference Manual
        // 7.1 About the L2 memory system
        //   The L2 memory system consists of:
        //    - A tightly-integrated L2 cache with:
        //       - A configurable size of 256KB, 512KB, 1MB, 2MB, 4MB, or 8MB.
        return 8 * 1024 * 1024;
      default:
        // ARM DynamIQ Shared Unit Technical Reference Manual
        // 1.3 Implementation options
        //   L3_CACHE_SIZE
        //    - 256KB
        //    - 512KB
        //    - 1024KB
        //    - 1536KB
        //    - 2048KB
        //    - 3072KB
        //    - 4096KB
        return 4 * 1024 * 1024;
    }
  #endif
  if (processor->cache.l4 != NULL) {
    return processor->cache.l4->size;
  } else if (processor->cache.l3 != NULL) {
    return processor->cache.l3->size;
  } else if (processor->cache.l2 != NULL) {
    return processor->cache.l2->size;
  } else if (processor->cache.l1d != NULL) {
    return processor->cache.l1d->size;
  } else {
    return 0;
  }
}

}  // namespace utils
}  // namespace benchmark
