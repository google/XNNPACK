// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "xnnpack/common.h"

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

#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/hardware-config.h"

#include "bench/utils.h"

static void* wipe_buffer = nullptr;
static size_t wipe_buffer_size = 0;

static std::once_flag wipe_buffer_guard;

static void InitWipeBuffer() {
  // Default: the largest known cache size (128 MB Intel Crystalwell L4 cache).
  wipe_buffer_size = 128 * 1024 * 1024;
  #if XNN_ENABLE_CPUINFO
    if (cpuinfo_initialize()) {
      wipe_buffer_size = benchmark::utils::GetMaxCacheSize();
    }
  #endif  // XNN_ENABLE_CPUINFO

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
  #if XNN_ENABLE_CPUINFO
    if (cpuinfo_initialize()) {
      step = cpuinfo_get_l1d_cache(0)->line_size;
    }
  #endif  // XNN_ENABLE_CPUINFO

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
  #if defined(__thumb__) && !defined(__thumb2__)
    __asm__ __volatile__(
        "VMRS %[fpscr], fpscr\n"
        "ORRS %[fpscr], %[bitmask]\n"
        "VMSR fpscr, %[fpscr]\n"
        : [fpscr] "=l" (fpscr)
        : [bitmask] "l" (0x1000000)
        : "cc");
  #else
    __asm__ __volatile__(
        "VMRS %[fpscr], fpscr\n"
        "ORR %[fpscr], #0x1000000\n"
        "VMSR fpscr, %[fpscr]\n"
        : [fpscr] "=r" (fpscr));
  #endif
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
  if (f != nullptr) {
    if (fscanf(f, "%d", &freq) != 0) {
      fclose(f);
      return uint64_t(freq) * 1000;
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

void MultiThreadingParameters(benchmark::internal::Benchmark* benchmark) {
  benchmark->ArgName("T");

  // Disabled thread pool (execution on the caller thread only).
  benchmark->Arg(1);

  #if XNN_ENABLE_CPUINFO
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
  #endif  // XNN_ENABLE_CPUINFO
}

#if XNN_ARCH_ARM
  bool CheckVFP(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !(hardware_config->use_arm_vfpv2 || hardware_config->use_arm_vfpv3)) {
      state.SkipWithError("no VFP extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_ARM
  bool CheckARMV6(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_arm_v6) {
      state.SkipWithError("no ARMv6 extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  bool CheckFP16ARITH(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_arm_fp16_arith) {
      state.SkipWithError("no FP16-ARITH extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  bool CheckNEON(benchmark::State& state) {
    #if XNN_ARCH_ARM64
      return true;
    #else
      const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      if (hardware_config == nullptr || !hardware_config->use_arm_neon) {
        state.SkipWithError("no NEON extension");
        return false;
      }
      return true;
    #endif
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  bool CheckNEONFP16(benchmark::State& state) {
    #if XNN_ARCH_ARM64
      return true;
    #else
      const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      if (hardware_config == nullptr || !hardware_config->use_arm_neon_fp16) {
        state.SkipWithError("no NEON-FP16 extension");
        return false;
      }
      return true;
    #endif
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  bool CheckNEONFMA(benchmark::State& state) {
    #if XNN_ARCH_ARM64
      return true;
    #else
      const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      if (hardware_config == nullptr || !hardware_config->use_arm_neon_fma) {
        state.SkipWithError("no NEON-FMA extension");
        return false;
      }
      return true;
    #endif
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  bool CheckNEONV8(benchmark::State& state) {
    #if XNN_ARCH_ARM64
      return true;
    #else
      const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      if (hardware_config == nullptr || !hardware_config->use_arm_neon_v8) {
        state.SkipWithError("no NEON-V8 extension");
        return false;
      }
      return true;
    #endif
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  bool CheckNEONFP16ARITH(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_arm_neon_fp16_arith) {
      state.SkipWithError("no NEON-FP16-ARITH extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  bool CheckNEONBF16(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_arm_neon_bf16) {
      state.SkipWithError("no NEON-BF16 extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  bool CheckNEONDOT(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_arm_neon_dot) {
      state.SkipWithError("no NEON-DOT extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  bool CheckNEONI8MM(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_arm_neon_i8mm) {
      state.SkipWithError("no NEON-I8MM extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_RISCV
  bool CheckRVV(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_riscv_vector) {
      state.SkipWithError("no RVV extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_RISCV

#if XNN_ARCH_RISCV
  bool CheckRVVFP16ARITH(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_riscv_vector_fp16_arith) {
      state.SkipWithError("no RVV-FP16-ARITH extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_RISCV

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckSSSE3(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_ssse3) {
      state.SkipWithError("no SSSE3 extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckSSE41(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_sse4_1) {
      state.SkipWithError("no SSE4.1 extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckAVX(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_avx) {
      state.SkipWithError("no AVX extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckF16C(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_f16c) {
      state.SkipWithError("no F16C extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckFMA3(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_fma3) {
      state.SkipWithError("no FMA3 extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckAVX2(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_avx2) {
      state.SkipWithError("no AVX2 extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckAVX512F(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_avx512f) {
      state.SkipWithError("no AVX512F extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckAVX512SKX(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_avx512skx) {
      state.SkipWithError("no AVX512 SKX extensions");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckAVX512VBMI(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_avx512vbmi) {
      state.SkipWithError("no AVX512 VBMI extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckAVX512VNNI(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_avx512vnni) {
      state.SkipWithError("no AVX512 VNNI extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckAVX512AMX(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_avx512amx) {
      state.SkipWithError("no AVX512 AMX extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckAVX512FP16(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_avx512fp16) {
      state.SkipWithError("no AVX512 FP16 extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckAVX512VNNIGFNI(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_avx512vnnigfni) {
      state.SkipWithError("no GFNI extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckAVXVNNI(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_avxvnni) {
      state.SkipWithError("no AVX VNNI extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckAVX256SKX(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_avx256skx) {
      state.SkipWithError("no AVX256SKX extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckAVX256VNNI(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_avx256vnni) {
      state.SkipWithError("no AVX256VNNI extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool CheckAVX256VNNIGFNI(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_x86_avx256vnnigfni) {
      state.SkipWithError("no AVX256VNNIGFNI extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_HEXAGON
  bool CheckHVX(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_hvx) {
      state.SkipWithError("no HVX extension");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_HEXAGON

#if XNN_ARCH_WASMRELAXEDSIMD
  bool CheckWAsmPSHUFB(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_wasm_pshufb) {
      state.SkipWithError("no WAsm PSHUFB support");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
  bool CheckWAsmSDOT(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_wasm_sdot) {
      state.SkipWithError("no WAsm SDOT support");
      return false;
    }
    return true;
  }

  bool CheckWAsmUSDOT(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_wasm_usdot) {
      state.SkipWithError("no WAsm USDOT support");
      return false;
    }
    return true;
  }

  bool CheckWAsmBLENDVPS(benchmark::State& state) {
    const xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == nullptr || !hardware_config->use_wasm_blendvps) {
      state.SkipWithError("no WAsm BLEND support");
      return false;
    }
    return true;
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


}  // namespace utils
}  // namespace benchmark
