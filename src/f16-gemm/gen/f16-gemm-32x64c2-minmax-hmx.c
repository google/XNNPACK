// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/c2-hmx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <HAP_compute_res.h>
#include <HAP_power.h>
#include <assert.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <pthread.h>
#include <qurt.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/unaligned.h"

// TODO(b/574286664): Support qc4w and qc8w weights (f16-qc4w-gemm and
// f16-qc8w-gemm) by using HVX to unpack/dequantize 4-bit and 8-bit weights into
// the VTCM FP16 weight buffers in parallel with HMX 32x32x2 tile execution.

// HMX activation/weight operand sizes are limited to 64 KB (16-bit byte count),
// i.e. 32 rows x 1024 FP16 elements per issue on Hexagon V81; earlier
// generations (V73 simulator) stall above ~32 KB, so `max_k_chunk` is chosen at
// init. Larger K is split into chunks accumulated in FP16.
#define XNN_HMX_MAX_K_CHUNK 1024

static const uint8_t k_vrdelta_controls_out[128] XNN_ALIGN(128) = {
    0x00, 0x00, 0x40, 0x40, 0x42, 0x42, 0x02, 0x02, 0x44, 0x44, 0x04, 0x04,
    0x06, 0x06, 0x46, 0x46, 0x48, 0x48, 0x08, 0x08, 0x0a, 0x0a, 0x4a, 0x4a,
    0x0c, 0x0c, 0x4c, 0x4c, 0x4e, 0x4e, 0x0e, 0x0e, 0x50, 0x50, 0x10, 0x10,
    0x12, 0x12, 0x52, 0x52, 0x14, 0x14, 0x54, 0x54, 0x56, 0x56, 0x16, 0x16,
    0x18, 0x18, 0x58, 0x58, 0x5a, 0x5a, 0x1a, 0x1a, 0x5c, 0x5c, 0x1c, 0x1c,
    0x1e, 0x1e, 0x5e, 0x5e, 0x20, 0x20, 0x60, 0x60, 0x62, 0x62, 0x22, 0x22,
    0x64, 0x64, 0x24, 0x24, 0x26, 0x26, 0x66, 0x66, 0x68, 0x68, 0x28, 0x28,
    0x2a, 0x2a, 0x6a, 0x6a, 0x2c, 0x2c, 0x6c, 0x6c, 0x6e, 0x6e, 0x2e, 0x2e,
    0x70, 0x70, 0x30, 0x30, 0x32, 0x32, 0x72, 0x72, 0x34, 0x34, 0x74, 0x74,
    0x76, 0x76, 0x36, 0x36, 0x38, 0x38, 0x78, 0x78, 0x7a, 0x7a, 0x3a, 0x3a,
    0x7c, 0x7c, 0x3c, 0x3c, 0x3e, 0x3e, 0x7e, 0x7e,
};

static const uint8_t k_vdelta_controls_out[128] XNN_ALIGN(128) = {
    0x00, 0x00, 0x3c, 0x3c, 0x38, 0x38, 0x04, 0x04, 0x30, 0x30, 0x0c, 0x0c,
    0x08, 0x08, 0x34, 0x34, 0x20, 0x20, 0x1c, 0x1c, 0x18, 0x18, 0x24, 0x24,
    0x10, 0x10, 0x2c, 0x2c, 0x28, 0x28, 0x14, 0x14, 0x00, 0x00, 0x3c, 0x3c,
    0x38, 0x38, 0x04, 0x04, 0x30, 0x30, 0x0c, 0x0c, 0x08, 0x08, 0x34, 0x34,
    0x20, 0x20, 0x1c, 0x1c, 0x18, 0x18, 0x24, 0x24, 0x10, 0x10, 0x2c, 0x2c,
    0x28, 0x28, 0x14, 0x14, 0x3e, 0x3e, 0x02, 0x02, 0x06, 0x06, 0x3a, 0x3a,
    0x0e, 0x0e, 0x32, 0x32, 0x36, 0x36, 0x0a, 0x0a, 0x1e, 0x1e, 0x22, 0x22,
    0x26, 0x26, 0x1a, 0x1a, 0x2e, 0x2e, 0x12, 0x12, 0x16, 0x16, 0x2a, 0x2a,
    0x3e, 0x3e, 0x02, 0x02, 0x06, 0x06, 0x3a, 0x3a, 0x0e, 0x0e, 0x32, 0x32,
    0x36, 0x36, 0x0a, 0x0a, 0x1e, 0x1e, 0x22, 0x22, 0x26, 0x26, 0x1a, 0x1a,
    0x2e, 0x2e, 0x12, 0x12, 0x16, 0x16, 0x2a, 0x2a,
};

// The HMX unit and its VTCM working set are a single shared resource per DSP,
// so all HMX GEMM microkernels serialize on `mutex`. HMX instructions may only
// be issued by a thread holding the HMX lock: each thread takes a shared lock
// once (HAP_compute_res_hmx_lock2) and keeps it, since lock/unlock per call
// halves HMX throughput. If the DSP firmware does not support shared locks,
// only the thread that initialized the context holds the (exclusive) lock and
// other threads fall back to HVX.
// TODO(b/574286664): Add multi-threading support (e.g., overlapping HVX
// LHS/RHS VTCM staging across worker threads with HMX tile execution, or
// pipelining threads across VTCM buffers) to improve performance.
struct xnn_hmx_vtcm_context {
  pthread_mutex_t mutex;
  bool initialized;
  bool init_failed;
  bool shared_lock;
  bool cached_pingpong;
  uint32_t context_id;
  int power_ctx;
  uint8_t* vtcm_base;
  uint8_t* vtcm_bias_shift_scale;
  uint8_t* vtcm_out;
  uint8_t* vtcm_act;
  uint8_t* vtcm_wgt;
  uint8_t* vtcm_wgt_pong;
  size_t max_k_chunk;
  size_t pingpong_n_offset;
  const void* cached_a;
  const void* next_expected_c;
  size_t cached_mr;
  size_t cached_kc;
  size_t cached_a_stride;
};

XNN_INTERNAL __attribute__((weak)) struct xnn_hmx_vtcm_context xnn_hmx_ctx = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
};

XNN_INTERNAL __attribute__((weak)) _Thread_local bool xnn_hmx_thread_locked;

// Must be called with `xnn_hmx_ctx.mutex` held.
static bool xnn_hmx_lock_thread(void) {
  if (xnn_hmx_thread_locked) {
    return true;
  }
  if (!xnn_hmx_ctx.shared_lock ||
      HAP_compute_res_hmx_lock2(xnn_hmx_ctx.context_id,
                                HAP_COMPUTE_RES_HMX_SHARED) != 0) {
    return false;
  }
  xnn_hmx_thread_locked = true;
  return true;
}

static XNN_INLINE void hmx_mpy_hf_32x32(
    const void* vtcm_act,
    uint32_t act_ctrl,
    const void* vtcm_wgt,
    uint32_t wgt_ctrl,
    void* vtcm_out,
    uint32_t out_ctrl,
    const void* vtcm_bias_shift_scale) {
  register const void* r0 __asm__("r0") = vtcm_act;
  register uint32_t r1 __asm__("r1") = act_ctrl;
  register const void* r2 __asm__("r2") = vtcm_wgt;
  register uint32_t r3 __asm__("r3") = wgt_ctrl;
  register void* r4 __asm__("r4") = vtcm_out;
  register uint32_t r5 __asm__("r5") = out_ctrl;
  register uint32_t r6 __asm__("r6") = 2;
  register const void* r7 __asm__("r7") = vtcm_bias_shift_scale;
  __asm__ volatile(
      "{\n\t"
      "  activation.hf = mxmem(%0, %1):deep\n\t"
      "  weight.hf = mxmem(%2, %3)\n\t"
      "}\n\t"
      "bias = mxmem2(%7)\n\t"
      "cvt.hf = acc(%6)\n\t"
      "mxmem(%4, %5) = cvt\n\t"
      :
      : "r"(r0), "r"(r1), "r"(r2), "r"(r3),
        "r"(r4), "r"(r5), "r"(r6), "r"(r7)
      : "memory");
}

static XNN_INLINE void transfer_lhs_to_vtcm(
    size_t mr,
    size_t k_elems,
    size_t k_padded,
    const xnn_float16* a,
    size_t a_stride,
    uint8_t* vtcm_act) {
  HVX_Vector* vtcm_act_vec = (HVX_Vector*) vtcm_act;
  const size_t num_k32 = k_padded >> 5;
  const size_t tail_rem = k_elems & 63;
  const HVX_VectorPred q_rem = Q6_Q_vsetq_R(tail_rem * sizeof(xnn_float16));
  for (size_t i = 0; i < 16; ++i) {
    if (2 * i >= mr) {
      for (size_t c32 = 0; c32 < num_k32; ++c32) {
        vtcm_act_vec[c32 * 16 + i] = Q6_V_vzero();
      }
      continue;
    }
    const xnn_float16* a_even =
        (const xnn_float16*) ((uintptr_t) a + (2 * i) * a_stride);
    const xnn_float16* a_odd =
        (2 * i + 1 < mr)
            ? (const xnn_float16*) ((uintptr_t) a + (2 * i + 1) * a_stride)
            : NULL;
    size_t c = 0;
    size_t k = 0;
    for (; k + 64 <= k_elems; k += 64, ++c) {
      const HVX_Vector v_even = *((const HVX_UVector*) (a_even + k));
      const HVX_Vector v_odd =
          a_odd ? *((const HVX_UVector*) (a_odd + k)) : Q6_V_vzero();
      const HVX_VectorPair pair = Q6_W_vshuff_VVR(v_odd, v_even, -2);
      vtcm_act_vec[(2 * c) * 16 + i] = Q6_V_lo_W(pair);
      vtcm_act_vec[(2 * c + 1) * 16 + i] = Q6_V_hi_W(pair);
    }
    if (tail_rem != 0) {
      const HVX_Vector v_even =
          Q6_V_vand_QV(q_rem, *((const HVX_UVector*) (a_even + k)));
      const HVX_Vector v_odd =
          a_odd ? Q6_V_vand_QV(q_rem, *((const HVX_UVector*) (a_odd + k)))
                : Q6_V_vzero();
      const HVX_VectorPair pair = Q6_W_vshuff_VVR(v_odd, v_even, -2);
      vtcm_act_vec[(2 * c) * 16 + i] = Q6_V_lo_W(pair);
      if (2 * c + 1 < num_k32) {
        vtcm_act_vec[(2 * c + 1) * 16 + i] = Q6_V_hi_W(pair);
      }
    }
  }
}

static XNN_INLINE void transfer_rhs_to_vtcm(
    size_t k_elems,
    size_t k_padded,
    const xnn_float16* w_ptr,
    uint8_t* vtcm_wgt) {
  const size_t k_pairs = (k_elems + 1) >> 1;
  const size_t k_pairs_padded = k_padded >> 1;
  HVX_Vector* vtcm_wgt0 = (HVX_Vector*) vtcm_wgt;
  HVX_Vector* vtcm_wgt1 = (HVX_Vector*) (vtcm_wgt + k_padded * 64);
  for (size_t kp = 0; kp < k_pairs; ++kp) {
    vtcm_wgt0[kp] = *((const HVX_UVector*) w_ptr);
    w_ptr += 64;
    vtcm_wgt1[kp] = *((const HVX_UVector*) w_ptr);
    w_ptr += 64;
  }
  for (size_t kp = k_pairs; kp < k_pairs_padded; ++kp) {
    vtcm_wgt0[kp] = Q6_V_vzero();
    vtcm_wgt1[kp] = Q6_V_vzero();
  }
}

// Must be called with `xnn_hmx_ctx.mutex` held.
static bool xnn_init_hmx_vtcm(void) {
  HAP_power_request_t request;
  memset(&request, 0, sizeof(request));
  request.type = HAP_power_set_apptype;
  request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
  HAP_power_set(&xnn_hmx_ctx.power_ctx, &request);

  memset(&request, 0, sizeof(request));
  request.type = HAP_power_set_HVX;
  request.hvx.power_up = 1;
  HAP_power_set(&xnn_hmx_ctx.power_ctx, &request);

  memset(&request, 0, sizeof(request));
  request.type = HAP_power_set_HMX;
  request.hmx.power_up = 1;
  if (HAP_power_set(&xnn_hmx_ctx.power_ctx, &request) != 0) {
    return false;
  }

  const unsigned int required_vtcm_bytes = 384 * 1024;
  compute_res_attr_t compute_res;
  HAP_compute_res_attr_init(&compute_res);
  HAP_compute_res_attr_set_vtcm_param(&compute_res, required_vtcm_bytes, 1);
  HAP_compute_res_attr_set_hmx_param(&compute_res, 1);
  xnn_hmx_ctx.context_id = HAP_compute_res_acquire(&compute_res, 100000);
  if (xnn_hmx_ctx.context_id == 0) {
    return false;
  }
  xnn_hmx_ctx.vtcm_base =
      (uint8_t*) HAP_compute_res_attr_get_vtcm_ptr(&compute_res);
  if (xnn_hmx_ctx.vtcm_base == NULL) {
    HAP_compute_res_release(xnn_hmx_ctx.context_id);
    xnn_hmx_ctx.context_id = 0;
    return false;
  }
  if (HAP_compute_res_hmx_lock2(xnn_hmx_ctx.context_id,
                                HAP_COMPUTE_RES_HMX_SHARED) == 0) {
    xnn_hmx_ctx.shared_lock = true;
  } else if (HAP_compute_res_hmx_lock(xnn_hmx_ctx.context_id) != 0) {
    HAP_compute_res_release(xnn_hmx_ctx.context_id);
    xnn_hmx_ctx.context_id = 0;
    xnn_hmx_ctx.vtcm_base = NULL;
    return false;
  }
  xnn_hmx_thread_locked = true;
  memset(xnn_hmx_ctx.vtcm_base, 0, required_vtcm_bytes);

  qurt_arch_version_t arch = {0};
  const unsigned int arch_version =
      (qurt_sysenv_get_arch_version(&arch) == QURT_EOK) ? (arch.arch_version & 0xFF)
                                                        : 0;
  xnn_hmx_ctx.max_k_chunk = arch_version >= 0x81 ? XNN_HMX_MAX_K_CHUNK
                                                  : XNN_HMX_MAX_K_CHUNK / 2;

  // VTCM layout for XNN_HMX_MAX_K_CHUNK = 1024:
  //   [0 .. 2KB):       bias/shift/scale table (2 KB)
  //   [2KB .. 6KB):     2 x 32x32 FP16 output tiles (4 KB)
  //   [6KB .. 70KB):    32 x 1024 FP16 activation buffer (64 KB)
  //   [70KB .. 198KB):  1024 x 64 FP16 weight buffer 0 (128 KB)
  //   [198KB .. 326KB): 1024 x 64 FP16 weight buffer 1 (128 KB, ping-pong)
  xnn_hmx_ctx.vtcm_bias_shift_scale = xnn_hmx_ctx.vtcm_base;
  xnn_hmx_ctx.vtcm_out = xnn_hmx_ctx.vtcm_base + 2048;
  xnn_hmx_ctx.vtcm_act = xnn_hmx_ctx.vtcm_base + 6144;
  xnn_hmx_ctx.vtcm_wgt =
      xnn_hmx_ctx.vtcm_act + 32 * XNN_HMX_MAX_K_CHUNK * sizeof(xnn_float16);
  xnn_hmx_ctx.vtcm_wgt_pong =
      xnn_hmx_ctx.vtcm_wgt + 64 * XNN_HMX_MAX_K_CHUNK * sizeof(xnn_float16);

  XNN_ALIGN(128) uint8_t bias_init[256];
  memset(bias_init, 0, sizeof(bias_init));
  for (size_t i = 0; i < 128; i += 4) {
    bias_init[i + 1] = 0x3c;  // 1.0 in IEEE FP16 (0x3c00)
  }
  const HVX_Vector* src_vec = (const HVX_Vector*) bias_init;
  HVX_Vector* dst_vec = (HVX_Vector*) xnn_hmx_ctx.vtcm_bias_shift_scale;
  dst_vec[0] = src_vec[0];
  dst_vec[1] = src_vec[1];

  xnn_hmx_ctx.initialized = true;
  return true;
}

// Must be called with `xnn_hmx_ctx.mutex` held and the HMX lock taken by this
// thread.
static void xnn_f16_gemm_minmax_32x64c2__hmx_locked(
    size_t mr,
    size_t nc,
    size_t kc,
    const xnn_float16* restrict a,
    size_t a_stride,
    const xnn_float16* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_minmax_params* restrict params) {
  const size_t k_elems = kc >> 1;
  const size_t k_pairs_total = (k_elems + 1) >> 1;
  const size_t kc_stride = k_pairs_total << 1;
  const size_t panel_w_stride = 64 * (kc_stride + 1);

  const HVX_Vector v_vrdelta_out =
      *((const HVX_Vector*) k_vrdelta_controls_out);
  const HVX_Vector v_vdelta_out = *((const HVX_Vector*) k_vdelta_controls_out);
  const HVX_Vector* vtcm_out0 = (const HVX_Vector*) xnn_hmx_ctx.vtcm_out;
  const HVX_Vector* vtcm_out1 =
      (const HVX_Vector*) (xnn_hmx_ctx.vtcm_out + 2048);

  const uint16_t min_bits = *((const uint16_t*) &params->scalar.min);
  const uint16_t max_bits = *((const uint16_t*) &params->scalar.max);
  const HVX_Vector v_min = Q6_Vh_vsplat_R(min_bits);
  const HVX_Vector v_max = Q6_Vh_vsplat_R(max_bits);

  const size_t max_k_chunk = xnn_hmx_ctx.max_k_chunk;
  if XNN_UNLIKELY(k_elems > max_k_chunk) {
    xnn_hmx_ctx.cached_a = NULL;
    xnn_hmx_ctx.next_expected_c = NULL;

    do {
      const size_t step_nc = min(nc, 64);
      const HVX_Vector v_bias = *((const HVX_UVector*) w);
      const xnn_float16* w_ptr = w + 64;
      HVX_Vector acc_even[16];
      HVX_Vector acc_odd[16];
      for (size_t i = 0; i < 16; ++i) {
        acc_even[i] = v_bias;
        acc_odd[i] = v_bias;
      }

      for (size_t k_base = 0; k_base < k_elems; k_base += max_k_chunk) {
        const size_t chunk_k = min(k_elems - k_base, max_k_chunk);
        const size_t chunk_padded = (chunk_k + 31) & ~(size_t) 31;
        const uint32_t wgt_size = (uint32_t) (chunk_padded * 64);
        const uint32_t act_ctrl = wgt_size | 0x7c;
        const uint32_t wgt_ctrl = wgt_size - 1;

        transfer_lhs_to_vtcm(mr, chunk_k, chunk_padded, a + k_base, a_stride,
                             xnn_hmx_ctx.vtcm_act);
        transfer_rhs_to_vtcm(chunk_k, chunk_padded, w_ptr + 64 * k_base,
                             xnn_hmx_ctx.vtcm_wgt);
        hmx_mpy_hf_32x32(xnn_hmx_ctx.vtcm_act, act_ctrl, xnn_hmx_ctx.vtcm_wgt,
                         wgt_ctrl, xnn_hmx_ctx.vtcm_out, 0,
                         xnn_hmx_ctx.vtcm_bias_shift_scale);
        if (step_nc > 32) {
          hmx_mpy_hf_32x32(xnn_hmx_ctx.vtcm_act, act_ctrl,
                           xnn_hmx_ctx.vtcm_wgt + wgt_size, wgt_ctrl,
                           xnn_hmx_ctx.vtcm_out + 2048, 0,
                           xnn_hmx_ctx.vtcm_bias_shift_scale);
        }

        for (size_t i = 0; i < 16; ++i) {
          if (2 * i >= mr) break;
          const HVX_Vector v8 = Q6_V_vdelta_VV(
              Q6_V_vrdelta_VV(vtcm_out0[i], v_vrdelta_out), v_vdelta_out);
          const HVX_Vector v9 =
              (step_nc > 32)
                  ? Q6_V_vdelta_VV(
                        Q6_V_vrdelta_VV(vtcm_out1[i], v_vrdelta_out),
                        v_vdelta_out)
                  : Q6_V_vzero();
          const HVX_VectorPair v_rows = Q6_W_vshuff_VVR(v9, v8, -64);
          acc_even[i] = Q6_Vhf_vadd_VhfVhf(acc_even[i], Q6_V_lo_W(v_rows));
          if (2 * i + 1 < mr) {
            acc_odd[i] = Q6_Vhf_vadd_VhfVhf(acc_odd[i], Q6_V_hi_W(v_rows));
          }
        }
      }

      for (size_t i = 0; i < 16; ++i) {
        if (2 * i >= mr) break;
        const HVX_Vector v_even =
            Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(acc_even[i], v_min), v_max);
        xnn_float16* c_even =
            (xnn_float16*) ((uintptr_t) c + (2 * i) * cm_stride);
        if (step_nc >= 64) {
          *((HVX_UVector*) c_even) = v_even;
        } else {
          Q6_V_vstu_variable(c_even, step_nc * sizeof(xnn_float16), v_even);
        }
        if (2 * i + 1 < mr) {
          const HVX_Vector v_odd =
              Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(acc_odd[i], v_min), v_max);
          xnn_float16* c_odd =
              (xnn_float16*) ((uintptr_t) c + (2 * i + 1) * cm_stride);
          if (step_nc >= 64) {
            *((HVX_UVector*) c_odd) = v_odd;
          } else {
            Q6_V_vstu_variable(c_odd, step_nc * sizeof(xnn_float16), v_odd);
          }
        }
      }

      if (nc > 64) {
        w += panel_w_stride;
        c = (xnn_float16*) ((uintptr_t) c + cn_stride);
        nc -= 64;
      } else {
        nc = 0;
      }
    } while (nc != 0);
    return;
  }

  const size_t k_padded = (k_elems + 31) & ~(size_t) 31;
  const uint32_t wgt_size = (uint32_t) (k_padded * 64);
  const uint32_t act_ctrl = wgt_size | 0x7c;
  const uint32_t wgt_ctrl = wgt_size - 1;

  const bool reuse_lhs = !xnn_hmx_ctx.cached_pingpong &&
                         (a == xnn_hmx_ctx.cached_a) &&
                         (c == xnn_hmx_ctx.next_expected_c) &&
                         (mr == xnn_hmx_ctx.cached_mr) &&
                         (kc == xnn_hmx_ctx.cached_kc) &&
                         (a_stride == xnn_hmx_ctx.cached_a_stride);
  if (!reuse_lhs) {
    transfer_lhs_to_vtcm(mr, k_elems, k_padded, a, a_stride,
                         xnn_hmx_ctx.vtcm_act);
    xnn_hmx_ctx.cached_pingpong = false;
    xnn_hmx_ctx.cached_a = a;
    xnn_hmx_ctx.cached_mr = mr;
    xnn_hmx_ctx.cached_kc = kc;
    xnn_hmx_ctx.cached_a_stride = a_stride;
  }

  do {
    const size_t step_nc = min(nc, 64);
    const HVX_Vector v_bias = *((const HVX_UVector*) w);
    const xnn_float16* w_ptr = w + 64;

    transfer_rhs_to_vtcm(k_elems, k_padded, w_ptr, xnn_hmx_ctx.vtcm_wgt);

    hmx_mpy_hf_32x32(xnn_hmx_ctx.vtcm_act, act_ctrl, xnn_hmx_ctx.vtcm_wgt,
                     wgt_ctrl, xnn_hmx_ctx.vtcm_out, 0,
                     xnn_hmx_ctx.vtcm_bias_shift_scale);
    if (step_nc > 32) {
      hmx_mpy_hf_32x32(xnn_hmx_ctx.vtcm_act, act_ctrl,
                       xnn_hmx_ctx.vtcm_wgt + wgt_size, wgt_ctrl,
                       xnn_hmx_ctx.vtcm_out + 2048, 0,
                       xnn_hmx_ctx.vtcm_bias_shift_scale);
    }

    for (size_t i = 0; i < 16; ++i) {
      if (2 * i >= mr) break;
      const HVX_Vector v8 = Q6_V_vdelta_VV(
          Q6_V_vrdelta_VV(vtcm_out0[i], v_vrdelta_out), v_vdelta_out);
      const HVX_Vector v9 =
          (step_nc > 32)
              ? Q6_V_vdelta_VV(
                    Q6_V_vrdelta_VV(vtcm_out1[i], v_vrdelta_out),
                    v_vdelta_out)
              : Q6_V_vzero();
      const HVX_VectorPair v_rows = Q6_W_vshuff_VVR(v9, v8, -64);

      HVX_Vector v_even = Q6_Vhf_vadd_VhfVhf(Q6_V_lo_W(v_rows), v_bias);
      v_even = Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(v_even, v_min), v_max);
      xnn_float16* c_even =
          (xnn_float16*) ((uintptr_t) c + (2 * i) * cm_stride);
      if (step_nc >= 64) {
        *((HVX_UVector*) c_even) = v_even;
      } else {
        Q6_V_vstu_variable(c_even, step_nc * sizeof(xnn_float16), v_even);
      }

      if (2 * i + 1 < mr) {
        HVX_Vector v_odd = Q6_Vhf_vadd_VhfVhf(Q6_V_hi_W(v_rows), v_bias);
        v_odd = Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(v_odd, v_min), v_max);
        xnn_float16* c_odd =
            (xnn_float16*) ((uintptr_t) c + (2 * i + 1) * cm_stride);
        if (step_nc >= 64) {
          *((HVX_UVector*) c_odd) = v_odd;
        } else {
          Q6_V_vstu_variable(c_odd, step_nc * sizeof(xnn_float16), v_odd);
        }
      }
    }

    if (nc > 64) {
      w += panel_w_stride;
      c = (xnn_float16*) ((uintptr_t) c + cn_stride);
      nc -= 64;
    } else {
      xnn_hmx_ctx.next_expected_c = (const void*) ((uintptr_t) c + cn_stride);
      nc = 0;
    }
  } while (nc != 0);
}

static void xnn_f16_gemm_minmax_hvx_fallback(
    size_t mr,
    size_t nc,
    size_t kc,
    const xnn_float16* restrict a,
    size_t a_stride,
    const xnn_float16* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_minmax_params* restrict params) {
  const uint16_t min_bits = *((const uint16_t*) &params->scalar.min);
  const uint16_t max_bits = *((const uint16_t*) &params->scalar.max);
  const HVX_Vector v_min = Q6_Vh_vsplat_R(min_bits);
  const HVX_Vector v_max = Q6_Vh_vsplat_R(max_bits);
  const HVX_Vector v_one = Q6_Vh_vsplat_R(0x3c00);

  for (size_t m = 0; m < mr; ++m) {
    const uint8_t* a0 = (const uint8_t*) ((uintptr_t) a + m * a_stride);
    xnn_float16* c0 = (xnn_float16*) ((uintptr_t) c + m * cm_stride);
    const HVX_UVector* w_vec = (const HVX_UVector*) w;
    size_t rem_nc = nc;
    do {
      const HVX_Vector v_bias_raw = *w_vec++;
      HVX_Vector vacc0_0 = Q6_V_vzero();
      HVX_Vector vacc0_1 = Q6_V_vzero();

      size_t k = kc;
      const uint8_t* a_ptr = a0;
      for (; k >= 2 * sizeof(xnn_float16); k -= 2 * sizeof(xnn_float16)) {
        const HVX_Vector va0 = Q6_V_vsplat_R(unaligned_load_u32(a_ptr));
        a_ptr += sizeof(uint32_t);
        const HVX_Vector vb0 = *w_vec++;
        const HVX_Vector vb1 = *w_vec++;
        const HVX_VectorPair p00 = Q6_Wqf32_vmpy_VhfVhf(va0, vb0);
        const HVX_VectorPair p01 = Q6_Wqf32_vmpy_VhfVhf(va0, vb1);
        vacc0_0 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc0_0,
            Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p00), Q6_V_hi_W(p00)));
        vacc0_1 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc0_1,
            Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p01), Q6_V_hi_W(p01)));
      }
      if XNN_UNLIKELY(k != 0) {
        const HVX_Vector va0 =
            Q6_V_vsplat_R((uint32_t) *((const uint16_t*) a_ptr));
        const HVX_Vector vb0 = *w_vec++;
        const HVX_Vector vb1 = *w_vec++;
        const HVX_VectorPair p00 = Q6_Wqf32_vmpy_VhfVhf(va0, vb0);
        const HVX_VectorPair p01 = Q6_Wqf32_vmpy_VhfVhf(va0, vb1);
        vacc0_0 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc0_0,
            Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p00), Q6_V_hi_W(p00)));
        vacc0_1 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc0_1,
            Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p01), Q6_V_hi_W(p01)));
      }

      const HVX_Vector v_bias_shuff = Q6_Vh_vshuff_Vh(v_bias_raw);
      const HVX_VectorPair b_pair = Q6_Wqf32_vmpy_VhfVhf(v_bias_shuff, v_one);
      vacc0_0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0_0, Q6_V_lo_W(b_pair));
      vacc0_1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0_1, Q6_V_hi_W(b_pair));
      HVX_Vector vout0 = Q6_Vh_vdeal_Vh(
          Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vacc0_1, vacc0_0)));
      vout0 = Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(vout0, v_min), v_max);

      if XNN_LIKELY(rem_nc >= 64) {
        *((HVX_UVector*) c0) = vout0;
        c0 = (xnn_float16*) ((uintptr_t) c0 + cn_stride);
        rem_nc -= 64;
      } else {
        Q6_V_vstu_variable(c0, rem_nc * sizeof(xnn_float16), vout0);
        rem_nc = 0;
      }
    } while (rem_nc != 0);
  }
}

void xnn_f16_gemm_minmax_ukernel_32x64c2__hmx(
    size_t mr,
    size_t nc,
    size_t kc,
    const xnn_float16* restrict a,
    size_t a_stride,
    const xnn_float16* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_minmax_params* restrict params) {
  assert(mr != 0);
  assert(mr <= 32);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(xnn_float16) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  pthread_mutex_lock(&xnn_hmx_ctx.mutex);
  // A failed HMX/VTCM acquisition is not retried: every attempt may block for
  // the acquire timeout, and the HVX fallback keeps the results correct.
  if XNN_UNLIKELY(!xnn_hmx_ctx.initialized && !xnn_hmx_ctx.init_failed) {
    xnn_hmx_ctx.init_failed = !xnn_init_hmx_vtcm();
  }
  const bool hmx_ready = xnn_hmx_ctx.initialized && xnn_hmx_lock_thread();
  if XNN_LIKELY(hmx_ready) {
    xnn_f16_gemm_minmax_32x64c2__hmx_locked(
        mr, nc, kc, a, a_stride, w, c, cm_stride, cn_stride, params);
  }
  pthread_mutex_unlock(&xnn_hmx_ctx.mutex);

  if XNN_UNLIKELY(!hmx_ready) {
    // HMX/VTCM could not be acquired (e.g. held by another client) or this
    // thread cannot lock HMX: compute on HVX with the same packed weights.
    xnn_f16_gemm_minmax_hvx_fallback(
        mr, nc, kc, a, a_stride, w, c, cm_stride, cn_stride, params);
  }
}
