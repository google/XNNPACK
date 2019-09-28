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
    wipe_buffer_size = cpuinfo_get_max_cache_size();
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
uint64_t GetCurrentCpuFrequency(void) {
#ifdef __linux__
  int freq = 0;
  char cpuinfo_name[512];
  int cpu = sched_getcpu();
  sprintf(cpuinfo_name,
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

}  // namespace utils
}  // namespace benchmark
