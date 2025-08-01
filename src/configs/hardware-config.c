// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#if XNN_ENABLE_CPUINFO
#include <cpuinfo.h>
#endif  // XNN_ENABLE_CPUINFO

#include "src/xnnpack/common.h"

#if _WIN32
  #include <windows.h>

  #ifndef PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
    #define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
  #endif
#endif
#if XNN_ARCH_X86_64 && defined(__linux__) && !defined(CHROMIUM)
#include <sys/syscall.h>
#include <unistd.h>

#define XFEATURE_XTILEDATA 18
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#if XNN_ENABLE_CPUINFO
  #include <cpuinfo.h>
#endif  // XNN_ENABLE_CPUINFO
#if XNN_ARCH_HEXAGON
// TODO - use CPUINFO
#include <qurt.h>
#endif

#if XNN_ARCH_RISCV
#include <inttypes.h>
#include <sys/auxv.h>

#define COMPAT_HWCAP_ISA_V (1 << ('V' - 'A'))
#endif

#if XNN_ARCH_PPC64
  #include <sys/auxv.h>
#endif

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
#include <math.h>
#endif

#if XNN_ARCH_WASMRELAXEDSIMD
#include <wasm_simd128.h>
#endif

#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/log.h"

#if XNN_ARCH_X86_64 && defined(__linux__) && !defined(CHROMIUM)
ssize_t xnn_syscall(size_t rax, size_t rdi, size_t rsi, size_t rdx) {
  __asm (
    "syscall"
    : "+a" (rax)
    : "D"(rdi), "S"(rsi), "d"(rdx)
    : "rcx", "r11", "memory"
  );
  return rax;
}
#endif

#if XNN_ENABLE_CPUINFO
static enum xnn_uarch cpuinfo_to_xnn_uarch(enum cpuinfo_uarch uarch) {
  switch (uarch) {
    case cpuinfo_uarch_dhyana: return xnn_uarch_dhyana;
    case cpuinfo_uarch_zen: return xnn_uarch_zen;
    case cpuinfo_uarch_zen4: return xnn_uarch_zen4;

    case cpuinfo_uarch_cortex_a5: return xnn_uarch_cortex_a5;
    case cpuinfo_uarch_cortex_a7: return xnn_uarch_cortex_a7;
    case cpuinfo_uarch_cortex_a32: return xnn_uarch_cortex_a32;
    case cpuinfo_uarch_cortex_a35: return xnn_uarch_cortex_a35;
    case cpuinfo_uarch_cortex_a53: return xnn_uarch_cortex_a53;
    case cpuinfo_uarch_cortex_a55r0: return xnn_uarch_cortex_a55r0;
    case cpuinfo_uarch_cortex_a55: return xnn_uarch_cortex_a55;
    case cpuinfo_uarch_cortex_a57: return xnn_uarch_cortex_a57;
    case cpuinfo_uarch_cortex_a72: return xnn_uarch_cortex_a72;
    case cpuinfo_uarch_cortex_a73: return xnn_uarch_cortex_a73;
    case cpuinfo_uarch_cortex_a75: return xnn_uarch_cortex_a75;
    case cpuinfo_uarch_cortex_a76: return xnn_uarch_cortex_a76;
    case cpuinfo_uarch_cortex_a77: return xnn_uarch_cortex_a77;
    case cpuinfo_uarch_cortex_a78: return xnn_uarch_cortex_a78;
    case cpuinfo_uarch_cortex_a510: return xnn_uarch_cortex_a510;
    case cpuinfo_uarch_cortex_a710: return xnn_uarch_cortex_a710;
    case cpuinfo_uarch_cortex_a715: return xnn_uarch_cortex_a715;
    case cpuinfo_uarch_cortex_x1: return xnn_uarch_cortex_x1;
    case cpuinfo_uarch_cortex_x2: return xnn_uarch_cortex_x2;
    case cpuinfo_uarch_cortex_x3: return xnn_uarch_cortex_x3;
    case cpuinfo_uarch_cortex_x4: return xnn_uarch_cortex_x4;
    case cpuinfo_uarch_exynos_m1: return xnn_uarch_exynos_m1;
    case cpuinfo_uarch_exynos_m2: return xnn_uarch_exynos_m2;
    case cpuinfo_uarch_exynos_m3: return xnn_uarch_exynos_m3;
    case cpuinfo_uarch_exynos_m4: return xnn_uarch_exynos_m4;
    case cpuinfo_uarch_exynos_m5: return xnn_uarch_exynos_m5;
    case cpuinfo_uarch_krait: return xnn_uarch_krait;
    case cpuinfo_uarch_kryo: return xnn_uarch_kryo;
    case cpuinfo_uarch_neoverse_n1: return xnn_uarch_neoverse_n1;
    case cpuinfo_uarch_neoverse_n2: return xnn_uarch_neoverse_n2;
    case cpuinfo_uarch_neoverse_v1: return xnn_uarch_neoverse_v1;
    case cpuinfo_uarch_neoverse_v2: return xnn_uarch_neoverse_v2;
    case cpuinfo_uarch_oryon: return xnn_uarch_oryon;
    default: return xnn_uarch_unknown;
  }
}
#endif  // XNN_ENABLE_CPUINFO

static struct xnn_hardware_config hardware_config = {0};

XNN_INIT_ONCE_GUARD(hardware);

static void set_arch_flag(uint64_t flag, bool value) {
  if (value) {
    hardware_config.arch_flags |= flag;
  } else {
    hardware_config.arch_flags &= ~flag;
  }
}

static void init_hardware_config(void) {
  hardware_config.arch_flags = 0;
#if XNN_ARCH_ARM64 || XNN_ARCH_ARM
#if XNN_PLATFORM_WINDOWS
  SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);
  switch (system_info.wProcessorLevel) {
    case 0x803:  // Kryo 385 Silver
      set_arch_flag(xnn_arch_arm_neon_fp16_arith, true);
      break;
    default:
      // Assume that Dot Product support implies FP16 support.
      // ARM manuals don't guarantee that, but it holds in practice.
      set_arch_flag(xnn_arch_arm_neon_fp16_arith,
          !!IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE));
      break;
  }
  set_arch_flag(xnn_arch_arm_fp16_arith, hardware_config.arch_flags & xnn_arch_arm_neon_fp16_arith);

  set_arch_flag(xnn_arch_arm_neon_bf16, false);
  set_arch_flag(xnn_arch_arm_neon_dot, !!IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE));
#else
  set_arch_flag(xnn_arch_arm_fp16_arith, cpuinfo_has_arm_fp16_arith());
  set_arch_flag(xnn_arch_arm_neon_fp16_arith, cpuinfo_has_arm_neon_fp16_arith());
  set_arch_flag(xnn_arch_arm_neon_bf16, cpuinfo_has_arm_neon_bf16());
  set_arch_flag(xnn_arch_arm_neon_dot, cpuinfo_has_arm_neon_dot());
#endif
  set_arch_flag(xnn_arch_arm_vfpv3, cpuinfo_has_arm_vfpv3());
  set_arch_flag(xnn_arch_arm_neon, cpuinfo_has_arm_neon());
  set_arch_flag(xnn_arch_arm_neon_fp16, cpuinfo_has_arm_neon_fp16());
  set_arch_flag(xnn_arch_arm_neon_fma, cpuinfo_has_arm_neon_fma());
  set_arch_flag(xnn_arch_arm_neon_v8, cpuinfo_has_arm_neon_v8());
#endif

#if XNN_ARCH_ARM
  set_arch_flag(xnn_arch_arm_v6, cpuinfo_has_arm_v6());
  set_arch_flag(xnn_arch_arm_vfpv2, cpuinfo_has_arm_vfpv2());
#endif

#if XNN_ARCH_ARM64
  set_arch_flag(xnn_arch_arm_neon_i8mm, cpuinfo_has_arm_i8mm());
  set_arch_flag(xnn_arch_arm_sve, cpuinfo_has_arm_sve());
  set_arch_flag(xnn_arch_arm_sve2, cpuinfo_has_arm_sve2());
  set_arch_flag(xnn_arch_arm_sme, cpuinfo_has_arm_sme());
  set_arch_flag(xnn_arch_arm_sme2, cpuinfo_has_arm_sme2());
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  const bool use_x86_avx512f = XNN_ENABLE_AVX512F && cpuinfo_has_x86_avx512f();
  const bool use_x86_avx512skx = XNN_ENABLE_AVX512SKX && use_x86_avx512f &&
      cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl();
  const bool use_x86_avx512vnni = XNN_ENABLE_AVX512VNNI && use_x86_avx512skx && cpuinfo_has_x86_avx512vnni();
  const bool use_x86_avx512vnnigfni = XNN_ENABLE_AVX512VNNIGFNI && use_x86_avx512vnni && cpuinfo_has_x86_gfni();
  const bool use_x86_avx512amx = XNN_ENABLE_AVX512AMX && XNN_ARCH_X86_64 && use_x86_avx512vnnigfni && cpuinfo_has_x86_amx_int8();
  const bool use_x86_avx2 = cpuinfo_has_x86_avx2();
  const bool use_x86_avx256vnni = XNN_ENABLE_AVX256VNNI && use_x86_avx512skx && cpuinfo_has_x86_avx512vnni();

  set_arch_flag(xnn_arch_x86_ssse3, cpuinfo_has_x86_ssse3());
  set_arch_flag(xnn_arch_x86_sse4_1, cpuinfo_has_x86_sse4_1());
  set_arch_flag(xnn_arch_x86_avx, cpuinfo_has_x86_avx());
  set_arch_flag(xnn_arch_x86_f16c, cpuinfo_has_x86_f16c());
  set_arch_flag(xnn_arch_x86_fma3, cpuinfo_has_x86_fma3());
  set_arch_flag(xnn_arch_x86_avx2, use_x86_avx2);
  set_arch_flag(xnn_arch_x86_avx512f, use_x86_avx512f);
  set_arch_flag(xnn_arch_x86_avx512skx, use_x86_avx512skx);
  set_arch_flag(xnn_arch_x86_avx512vbmi, XNN_ENABLE_AVX512VBMI && use_x86_avx512skx && cpuinfo_has_x86_avx512vbmi());
  set_arch_flag(xnn_arch_x86_avx512vnni, use_x86_avx512vnni);
  set_arch_flag(xnn_arch_x86_avx512vnnigfni, use_x86_avx512vnnigfni);
  set_arch_flag(xnn_arch_x86_avx512fp16, XNN_ENABLE_AVX512FP16 && cpuinfo_has_x86_avx512fp16());
  set_arch_flag(xnn_arch_x86_avx512bf16, XNN_ENABLE_AVX512BF16 && cpuinfo_has_x86_avx512bf16());
  set_arch_flag(xnn_arch_x86_avx512amx, use_x86_avx512amx);
#if XNN_ARCH_X86_64 && defined(__linux__) && !defined(CHROMIUM)
  if (use_x86_avx512amx) {
    size_t status = xnn_syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA, 0);
    if (status) {
      xnn_log_info("XFEATURE_XTILEDATA setup is failed, TMUL usage is not allowed");
      set_arch_flag(xnn_arch_x86_avx512amx, 0);
    }
  }
#endif
  set_arch_flag(xnn_arch_x86_avxvnni, XNN_ENABLE_AVXVNNI && use_x86_avx2 && cpuinfo_has_x86_avxvnni());
  set_arch_flag(xnn_arch_x86_avxvnniint8, XNN_ENABLE_AVXVNNIINT8 && use_x86_avx2 && cpuinfo_has_x86_avx_vnni_int8());
  set_arch_flag(xnn_arch_x86_avx256skx, XNN_ENABLE_AVX256SKX && use_x86_avx512skx);
  set_arch_flag(xnn_arch_x86_avx256vnni, use_x86_avx256vnni);
  set_arch_flag(xnn_arch_x86_avx256vnnigfni, XNN_ENABLE_AVX256VNNIGFNI && use_x86_avx256vnni && cpuinfo_has_x86_gfni());
#endif  // !XNN_ARCH_X86 && !XNN_ARCH_X86_64

#if XNN_ARCH_HEXAGON
  qurt_arch_version_t vers = {0};
  int ret = 0;
  int version = 0;

  ret = qurt_sysenv_get_arch_version(&vers);
  if (QURT_EOK == ret) {
    // Lower 8 bits represents the version number in hex form
    if ((vers.arch_version & 0xff) == 0x73) {
      version = 73;
    } else if ((vers.arch_version & 0xff) == 0x75) {
      version = 75;
    } else if ((vers.arch_version & 0xff) == 0x79) {
      version = 79;
    }
    // TODO: use xnn_log_info
    printf("HEXAGON UARCH VERSION %d\n", version);
    printf("HEXAGON sizeof(max_align_t) %zd\n", sizeof(max_align_t));

    // TODO(b/435522481): Support v69
    if (version >= 73) {
      set_arch_flag(xnn_arch_hvx, XNN_ENABLE_HVX);
    }
  }
#endif  // XNN_ARCH_HEXAGON

  #if XNN_ARCH_RISCV
    const long hwcap = getauxval(AT_HWCAP);
    xnn_log_debug("getauxval(AT_HWCAP) = %08lX", hwcap);
    const bool use_riscv_vector = (hwcap & COMPAT_HWCAP_ISA_V) != 0;
    set_arch_flag(xnn_arch_riscv_vector, use_riscv_vector);

    /* There is no HWCAP for fp16 so disable by default */
    set_arch_flag(xnn_arch_riscv_vector_fp16_arith, false);

    if (use_riscv_vector) {
      register uint32_t vlenb __asm__ ("t0");
      __asm__(".word 0xC22022F3"  /* CSRR t0, vlenb */ : "=r" (vlenb));
      hardware_config.vlenb = vlenb;
      xnn_log_info("RISC-V VLENB: %" PRIu32, vlenb);
    }
  #endif

  #if XNN_ARCH_PPC64
    const unsigned long HWCAPs = getauxval(AT_HWCAP);
    const unsigned long HWCAPs_2 = getauxval(AT_HWCAP2);
    if (HWCAPs & PPC_FEATURE_HAS_VSX) {
      set_arch_flag(xnn_arch_vsx, 1);
    }
    #if defined PPC_FEATURE2_ARCH_3_1
      if (HWCAPs_2 & PPC_FEATURE2_ARCH_3_1) {
        set_arch_flag(xnn_arch_vsx3, 1);
      }
      if (HWCAPs_2 & PPC_FEATURE2_MMA) {
        set_arch_flag(xnn_arch_mma, 1);
      }
    #endif
  #endif

  #if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    // Unlike most other architectures, on x86/x86-64 when floating-point instructions
    // have no NaN arguments, but produce NaN output, the output NaN has sign bit set.
    // We use it to distinguish x86/x86-64 from other architectures, by doing subtraction
    // of two infinities (must produce NaN per IEEE 754 standard).
    static const volatile float inf = INFINITY;
    hardware_config.is_x86 = signbit(inf - inf);
  #endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

  #if XNN_ARCH_WASMRELAXEDSIMD
    {
      // Check if out-of-bounds behavior of Relaxed Swizzle is consistent with PSHUFB.
      const v128_t table = wasm_i8x16_const(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
      const v128_t index_mask = wasm_i8x16_const_splat(INT8_C(0x8F));
      const volatile v128_t index_increment = wasm_i8x16_const_splat(16);  // volatile to confuse Clang which otherwise mis-compiles
      v128_t index = wasm_i8x16_const(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
      v128_t diff = wasm_i8x16_const_splat(0);
      for (uint32_t i = 16; i != 0; i--) {
        const v128_t pshufb_result = wasm_i8x16_swizzle(table, wasm_v128_and(index, index_mask));
        const v128_t relaxed_result = wasm_i8x16_relaxed_swizzle(table, index);
        diff = wasm_v128_or(diff, wasm_v128_xor(pshufb_result, relaxed_result));
        index = wasm_i8x16_add(index, index_increment);
      }
      set_arch_flag(xnn_arch_wasm_pshufb, !wasm_v128_any_true(diff));
    }

    {
      // Check out-of-bounds behaviour of Relaxed Integer Dot Product with Accumulation.
      const v128_t int8_input = wasm_i8x16_const(0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0);
      const volatile v128_t xint8_input = wasm_i8x16_const(0, 0, 0, -128, 0, 0, -128, 0, 0, -128, 0, 0, -128, 0, 0, 0);  // volatile to confuse Clang which otherwise ICE's
      const v128_t xint8_output = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(int8_input, xint8_input, wasm_i8x16_const_splat(0));

      const volatile v128_t overflow_input = wasm_i8x16_const(-128, -128, -128, -128, -128, -128, -1, -1, -1, -1, -128, -128, -1, -1, -1, -1);  // volatile to confuse Clang which otherwise ICE's
      const v128_t overflow_output = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(wasm_i8x16_const_splat(-128), overflow_input, wasm_i8x16_const_splat(0));
      set_arch_flag(xnn_arch_wasm_sdot, !wasm_v128_any_true(wasm_v128_or(
        wasm_v128_xor(xint8_output, wasm_i32x4_const_splat(-128)),
        wasm_v128_xor(overflow_output, wasm_i32x4_const(65536, 33024, 33024, 512)))));
    }
    {
      // Check out-of-bounds behaviour of Relaxed Integer Dot Product with Accumulation with signed and unsigned input (e.g. vpdpbusd).
      const v128_t int8_input = wasm_i8x16_const(0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0);
      const volatile v128_t xint8_input = wasm_i8x16_const(0, 0, 0, -128, 0, 0, -128, 0, 0, -128, 0, 0, -128, 0, 0, 0);  // volatile to confuse Clang which otherwise ICE's
      const v128_t xint8_output = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(int8_input, xint8_input, wasm_i8x16_const_splat(0));

      const volatile v128_t overflow_input = wasm_i8x16_const(-128, -128, -128, -128, -128, -128, -1, -1, -1, -1, -128, -128, -1, -1, -1, -1);  // volatile to confuse Clang which otherwise ICE's
      const v128_t overflow_output = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(wasm_i8x16_const_splat(-128), overflow_input, wasm_i8x16_const_splat(0));
      set_arch_flag(xnn_arch_wasm_usdot, !wasm_v128_any_true(wasm_v128_or(
        wasm_v128_xor(xint8_output, wasm_i32x4_const_splat(128)),
        wasm_v128_xor(overflow_output, wasm_i32x4_const(-65536, -98048, -98048, -130560)))));
    }
    {
      const v128_t input1 = wasm_i32x4_const(0xF0F0F0F0, 0xAAAAAAAA, 0xCCCCCCCC, 0x99999999);
      const v128_t input2 = wasm_i32x4_const(0x0F0F0F0F, 0x55555555, 0x33333333, 0x66666666);
      v128_t diff = wasm_i8x16_const_splat(0);
      for (uint32_t shift = 0; shift < 32; ++shift) {
        const uint32_t mask = UINT32_C(1) << shift;
        const volatile v128_t vmask = wasm_u32x4_splat(mask);
        const v128_t blendvps_result = wasm_v128_bitselect(input1, input2, wasm_i32x4_shr(vmask, 31));
        const v128_t relaxed_result = wasm_i32x4_relaxed_laneselect(input1, input2, vmask);
        diff = wasm_v128_or(diff, wasm_v128_xor(blendvps_result, relaxed_result));
      }
      set_arch_flag(xnn_arch_wasm_blendvps, !wasm_v128_any_true(diff));
    }
    {
      const v128_t input1 = wasm_f32x4_const(16777218.f, 0.f, 0.f, 0.f);
      const v128_t input2 = wasm_f32x4_const(3.f, 0.f, 0.f, 0.f);
      const v128_t input3 = wasm_f32x4_const(3.f, 0.f, 0.f, 0.f);
      v128_t diff = wasm_i8x16_const_splat(0);
      const v128_t relaxed_result = wasm_f32x4_relaxed_madd(input1, input2, input3);
      const v128_t mul_result = wasm_f32x4_add(input3, wasm_f32x4_mul(input1, input2));
      diff = wasm_v128_or(diff, wasm_v128_xor(mul_result, relaxed_result));
      set_arch_flag(xnn_arch_wasm_fma, !wasm_v128_any_true(diff));
    }
  #endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ENABLE_CPUINFO
    // Set the size of the L1 and L2 data caches.
    if (!cpuinfo_initialize()) {
      xnn_log_warning(
          "Failed to initialize cpuinfo, unable to determine L1/L2 data cache "
          "properties.");
    } else {
      const struct cpuinfo_processor* proc_info = cpuinfo_get_processor(0);
      if (proc_info != NULL) {
        // Get the L1 cache information.
        const struct cpuinfo_cache* l1_data_cache = proc_info->cache.l1d;
        if (l1_data_cache != NULL) {
          hardware_config.l1_data_cache_bytes = l1_data_cache->size;
          hardware_config.l1_data_cache_line_size = l1_data_cache->line_size;
          hardware_config.l1_data_cache_associativity =
              l1_data_cache->associativity;
          hardware_config.l1_data_cache_num_sets = l1_data_cache->sets;
          xnn_log_info(
              "l1_data_cache_bytes=%zu, l1_data_cache_line_size=%zu, "
              "l1_data_cache_associativity=%zu, l1_data_cache_num_sets=%zu.",
              hardware_config.l1_data_cache_bytes,
              hardware_config.l1_data_cache_line_size,
              hardware_config.l1_data_cache_associativity,
              hardware_config.l1_data_cache_num_sets);
        } else {
          xnn_log_warning("Unable to determine L1 data cache properties.");
        }

        // Get the L2 cache information.
        const struct cpuinfo_cache* l2_data_cache = proc_info->cache.l2;
        if (l2_data_cache != NULL) {
          hardware_config.l2_data_cache_bytes = l2_data_cache->size;
          hardware_config.l2_data_cache_line_size = l2_data_cache->line_size;
          hardware_config.l2_data_cache_associativity =
              l2_data_cache->associativity;
          hardware_config.l2_data_cache_num_sets = l2_data_cache->sets;
          xnn_log_info(
              "l2_data_cache_bytes=%zu, l2_data_cache_line_size=%zu, "
              "l2_data_cache_associativity=%zu, l2_data_cache_num_sets=%zu.",
              hardware_config.l2_data_cache_bytes,
              hardware_config.l2_data_cache_line_size,
              hardware_config.l2_data_cache_associativity,
              hardware_config.l2_data_cache_num_sets);
        } else {
          xnn_log_warning("Unable to determine L2 data cache properties.");
        }
      } else {
        xnn_log_warning("Unable to determine L1/L2 data cache properties.");
      }
    }

#if XNN_MAX_UARCH_TYPES > 1
    // Print what we think we know about the microarchs.
    xnn_log_info("cpuinfo_get_uarchs_count: %u.", cpuinfo_get_uarchs_count());
    for (int i = 0; i < cpuinfo_get_uarchs_count(); i++) {
      xnn_log_info("cpu_get_uarch(%i): 0x%x", i, cpuinfo_get_uarch(i)->uarch);
    }
#endif  // XNN_MAX_UARCH_TYPES > 1
    for (size_t i = 0; i < XNN_MAX_UARCH_TYPES; ++i) {
      const struct cpuinfo_uarch_info* uarch = cpuinfo_get_uarch(i);
      hardware_config.uarch[i] = uarch ? cpuinfo_to_xnn_uarch(uarch->uarch) : xnn_uarch_unknown;
    }
#else
  xnn_log_warning("Unable to determine L1/L2 data cache properties.");
  for (size_t i = 0; i < XNN_MAX_UARCH_TYPES; ++i) {
    hardware_config.uarch[i] = xnn_uarch_unknown;
  }
#endif  // XNN_ENABLE_CPUINFO
}

const struct xnn_hardware_config* xnn_init_hardware_config() {
  #if !XNN_PLATFORM_WEB && !XNN_ARCH_RISCV && !XNN_ARCH_PPC64 && XNN_ENABLE_CPUINFO
    if (!cpuinfo_initialize()) {
      xnn_log_error("failed to initialize cpuinfo");
      return NULL;
    }
  #endif  // !XNN_PLATFORM_WEB && !XNN_ARCH_RISCV && !XNN_ARCH_PPC64
  #if XNN_ARCH_ARM
    if (!cpuinfo_has_arm_v6()) {
      xnn_log_debug("unsupported hardware: ARMv6 not detected");
      return NULL;
    }

    if (!cpuinfo_has_arm_vfpv2() && !cpuinfo_has_arm_vfpv3()) {
      xnn_log_debug("unsupported hardware: VFP FPU not detected");
      return NULL;
    }
  #endif  // XNN_ARCH_ARM
  #if XNN_ARCH_X86
    if (!cpuinfo_has_x86_sse2()) {
      xnn_log_debug("unsupported hardware: SSE2 not detected");
      return NULL;
    }
  #endif  // XNN_ARCH_X86

  XNN_INIT_ONCE(hardware);
  return &hardware_config;
}
