// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdint.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>


static inline void xnn_pack_q8_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  uint32_t nr,
  uint32_t kr,
  uint8_t izp,
  uint8_t kzp,
  const uint8_t* k,
  const int32_t* b,
  void* packed_w)
{
  const int32_t boff = (int32_t) kc * (int32_t) izp * (int32_t) kzp;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_w;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          *((int32_t*) packed_w) = b[nr_block_start + nr_block_offset] + boff;
          packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
        }
      } else {
        size_t n = nr_block_size;
        do {
          *((int32_t*) packed_w) = boff;
          packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_w = (void*) ((uintptr_t) packed_w + (nr - nr_block_size) * sizeof(int32_t));
      for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
        const size_t kr_block_size = min(kc - kr_block_start, kr);
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          int32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
            const uint8_t kv = k[(nr_block_start + nr_block_offset) * kc + (kr_block_start + kr_block_offset)];
            ksum += (int32_t) kv;
            *((uint8_t*) packed_w) = kv;
            packed_w = (void*) ((uintptr_t) packed_w + sizeof(uint8_t));
          }
          packed_b[nr_block_offset] -= ksum * (int32_t) izp;
          packed_w = (void*) ((uintptr_t) packed_w + (kr - kr_block_size) * sizeof(uint8_t));
        }
        packed_w = (void*) ((uintptr_t) packed_w + (nr - nr_block_size) * kr * sizeof(uint8_t));
      }
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

static inline void xnn_pack_q8_gemm_io_w(
  size_t nc,
  size_t kc,
  uint32_t nr,
  uint32_t kr,
  uint8_t izp,
  uint8_t kzp,
  const uint8_t* k,
  const int32_t* b,
  void* packed_w)
{
  const int32_t boff = (int32_t) kc * (int32_t) izp * (int32_t) kzp;
  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    int32_t* packed_b = (int32_t*) packed_w;
    if XNN_LIKELY(b != NULL) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        *((int32_t*) packed_w) = b[nr_block_start + nr_block_offset] + boff;
        packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
      }
    } else {
      size_t n = nr_block_size;
      do {
        *((int32_t*) packed_w) = boff;
        packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
      } while (--n != 0);
    }
    packed_w = (void*) ((uintptr_t) packed_w + (nr - nr_block_size) * sizeof(int32_t));
    for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr);
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        int32_t ksum = 0;
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
          const uint8_t kv = k[(kr_block_start + kr_block_offset) * nc + (nr_block_start + nr_block_offset)];
          ksum += (int32_t) kv;
          *((uint8_t*) packed_w) = kv;
          packed_w = (void*) ((uintptr_t) packed_w + sizeof(uint8_t));
        }
        packed_b[nr_block_offset] -= ksum * (int32_t) izp;
        packed_w = (void*) ((uintptr_t) packed_w + (kr - kr_block_size) * sizeof(uint8_t));
      }
      packed_w = (void*) ((uintptr_t) packed_w + (nr - nr_block_size) * kr * sizeof(uint8_t));
    }
  }
}

static inline void xnn_pack_q8_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  uint32_t nr,
  uint32_t kr,
  uint8_t izp,
  uint8_t kzp,
  const uint8_t* k,
  const int32_t* b,
  void* packed_w)
{
  const int32_t boff = (int32_t) ks * (int32_t) kc * (int32_t) izp * (int32_t) kzp;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_w;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          *((int32_t*) packed_w) = b[nr_block_start + nr_block_offset] + boff;
          packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
        }
      } else {
        size_t n = nr_block_size;
        do {
          *((int32_t*) packed_w) = boff;
          packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_w = (void*) ((uintptr_t) packed_w + (nr - nr_block_size) * sizeof(int32_t));
      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
          const size_t kr_block_size = min(kc - kr_block_start, kr);
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            int32_t ksum = 0;
            for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
              const uint8_t kv =
                k[((nr_block_start + nr_block_offset) * ks + ki) * kc + (kr_block_start + kr_block_offset)];
              ksum += (int32_t) kv;
              *((uint8_t*) packed_w) = kv;
              packed_w = (void*) ((uintptr_t) packed_w + sizeof(uint8_t));
            }
            packed_b[nr_block_offset] -= ksum * (int32_t) izp;
            packed_w = (void*) ((uintptr_t) packed_w + (kr - kr_block_size) * sizeof(uint8_t));
          }
          packed_w = (void*) ((uintptr_t) packed_w + (nr - nr_block_size) * kr * sizeof(uint8_t));
        }
      }
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

static inline void xnn_pack_q8_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  uint32_t nr,
  uint32_t kr,
  uint8_t izp,
  uint8_t kzp,
  const uint8_t* k,
  const int32_t* b,
  void* packed_w)
{
  const int32_t boff = (int32_t) ks * (int32_t) izp * (int32_t) kzp;
  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_w;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          *((int32_t*) packed_w) = b[nr_block_start + nr_block_offset] + boff;
          packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
        }
      } else {
        size_t n = nr_block_size;
        do {
          *((int32_t*) packed_w) = boff;
          packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_w = (void*) ((uintptr_t) packed_w + (nr - nr_block_size) * sizeof(int32_t));
      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          const uint8_t kv =
            k[ki * g * nc + (nr_block_start + nr_block_offset)];
          *((uint8_t*) packed_w) = kv;
          packed_b[nr_block_offset] -= (int32_t) kv * (int32_t) izp;
          packed_w = (void*) ((uintptr_t) packed_w + kr * sizeof(uint8_t));
        }
        packed_w = (void*) ((uintptr_t) packed_w + (nr - nr_block_size) * kr * sizeof(uint8_t));
      }
    }
    k += nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

static inline void xnn_pack_q8_deconv_goki_w(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  uint8_t izp,
  uint8_t kzp,
  const uint8_t* k,
  const int32_t* b,
  void* packed_w,
  struct subconvolution_params* params)
{
  for (size_t i = 0; i < g; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*params++).weights = packed_w;
        }
        const int32_t boff = (int32_t) divide_round_up(kh - oy, sh) * (int32_t) divide_round_up(kw - ox, sw) * (int32_t) kc * (int32_t) izp * (int32_t) kzp;
        for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          int32_t* packed_b = (int32_t*) packed_w;
          if XNN_LIKELY(b != 0) {
            for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
              *((int32_t*) packed_w) = b[nr_block_start + nr_block_offset] + boff;
              packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
            }
          } else {
            size_t n = nr_block_size;
            do {
              *((int32_t*) packed_w) = boff;
              packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
            } while (--n != 0);
          }
          packed_w = (void*) ((uintptr_t) packed_w + (nr - nr_block_size) * sizeof(int32_t));
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
                const size_t kr_block_size = min(kc - kr_block_start, kr);
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                  int32_t ksum = 0;
                  for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
                    const uint8_t kv =
                      k[(((nr_block_start + nr_block_offset) * kh + ky) * kw + kx) * kc + (kr_block_start + kr_block_offset)];
                    ksum += (int32_t) kv;
                    *((uint8_t*) packed_w) = kv;
                    packed_w = (void*) ((uintptr_t) packed_w + sizeof(uint8_t));
                  }
                  packed_b[nr_block_offset] -= ksum * (int32_t) izp;
                  packed_w = (void*) ((uintptr_t) packed_w + (kr - kr_block_size) * sizeof(uint8_t));
                }
                packed_w = (void*) ((uintptr_t) packed_w + (nr - nr_block_size) * kr * sizeof(uint8_t));
              }
            }
          }
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

static inline void xnn_pack_q8_dwconv_ghw_w(
  size_t h,
  size_t w,
  size_t c,
  size_t cr,
  uint8_t izp,
  uint8_t kzp,
  const uint8_t* k,
  const int32_t* b,
  void* packed_w)
{
  const int32_t boff = (int32_t) h * (int32_t) w * (int32_t) izp * (int32_t) kzp;
  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    int32_t* packed_b = (int32_t*) packed_w;
    if XNN_LIKELY(b != NULL) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
        *((int32_t*) packed_w) = b[cr_block_start + cr_block_offset] + boff;
        packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
      }
    } else {
      size_t n = cr_block_size;
      do {
        *((int32_t*) packed_w) = boff;
        packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
      } while (--n != 0);
    }
    packed_w = (void*) ((uintptr_t) packed_w + (cr - cr_block_size) * sizeof(int32_t));
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          packed_b[cr_block_offset] -= (int32_t) kv * (int32_t) izp;
          *((uint8_t*) packed_w) = kv;
          packed_w = (void*) ((uintptr_t) packed_w + sizeof(uint8_t));
        }
        packed_w = (void*) ((uintptr_t) packed_w + (cr - cr_block_size) * sizeof(uint8_t));
      }
    }
  }
}

static inline void xnn_pack_q8_dwconv_hwg_w(
  size_t h,
  size_t w,
  size_t c,
  size_t cr,
  uint8_t izp,
  uint8_t kzp,
  const uint8_t* k,
  const int32_t* b,
  void* packed_w)
{
  const int32_t boff = (int32_t) h * (int32_t) w * (int32_t) izp * (int32_t) kzp;
  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    int32_t* packed_b = (int32_t*) packed_w;
    if XNN_LIKELY(b != NULL) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
        *((int32_t*) packed_w) = b[cr_block_start + cr_block_offset] + boff;
        packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
      }
    } else {
      size_t n = cr_block_size;
      do {
        *((int32_t*) packed_w) = boff;
        packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
      } while (--n != 0);
    }
    packed_w = (void*) ((uintptr_t) packed_w + (cr - cr_block_size) * sizeof(int32_t));
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          packed_b[cr_block_offset] -= (int32_t) kv * (int32_t) izp;
          *((uint8_t*) packed_w) = kv;
          packed_w = (void*) ((uintptr_t) packed_w + sizeof(uint8_t));
        }
        packed_w = (void*) ((uintptr_t) packed_w + (cr - cr_block_size) * sizeof(uint8_t));
      }
    }
  }
}

static inline void xnn_pack_f16_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w)
{
  const size_t skr = sr * kr;
  const size_t skc = round_down_po2(kc, skr);
  const size_t sr_mask = (sr - 1) * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_w[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      }
      packed_w += nr;

      for (size_t kr_block_start = 0; kr_block_start < skc; kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            *packed_w++ =
              k[(nr_block_start + nr_block_offset) * kc + round_down_po2(kr_block_start, skr) + ((kr_block_start + nr_block_offset * kr) & sr_mask) + kr_block_offset];
          }
        }
        packed_w += (nr - nr_block_size) * kr;
      }

      for (size_t kr_block_start = skc; kr_block_start < kc; kr_block_start += kr) {
        const size_t kr_block_size = min(kc - kr_block_start, kr);
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
            *packed_w++ =
              k[(nr_block_start + nr_block_offset) * kc + (kr_block_start + kr_block_offset)];
          }
          packed_w += kr - kr_block_size;
        }
        packed_w += (nr - nr_block_size) * kr;
      }
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

static inline void xnn_pack_f16_gemm_io_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w)
{
  const size_t skr = sr * kr;
  const size_t skc = round_down_po2(kc, skr);
  const size_t sr_mask = (sr - 1) * kr;
  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    if XNN_LIKELY(b != NULL) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        packed_w[nr_block_offset] = b[nr_block_start + nr_block_offset];
      }
    }
    packed_w += nr;

    for (size_t kr_block_start = 0; kr_block_start < skc; kr_block_start += kr) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
          *packed_w++ =
            k[(round_down_po2(kr_block_start, skr) + ((kr_block_start + nr_block_offset * kr) & sr_mask) + kr_block_offset) * nc + (nr_block_start + nr_block_offset)];
        }
      }
      packed_w += (nr - nr_block_size) * kr;
    }

    for (size_t kr_block_start = skc; kr_block_start < kc; kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr);
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
          *packed_w++ =
            k[(kr_block_start + kr_block_offset) * nc + (nr_block_start + nr_block_offset)];
        }
        packed_w += kr - kr_block_size;
      }
      packed_w += (nr - nr_block_size) * kr;
    }
  }
}

static inline void xnn_pack_f16_gemminc_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  uint16_t* packed_w)
{
  const size_t skr = sr * kr;
  const size_t skc = round_down_po2(kc, skr);
  const size_t sr_mask = (sr - 1) * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);

      for (size_t kr_block_start = 0; kr_block_start < skc; kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            *packed_w++ =
              k[(nr_block_start + nr_block_offset) * kc + round_down_po2(kr_block_start, skr) + ((kr_block_start + nr_block_offset * kr) & sr_mask) + kr_block_offset];
          }
        }
        packed_w += (nr - nr_block_size) * kr;
      }

      for (size_t kr_block_start = skc; kr_block_start < kc; kr_block_start += kr) {
        const size_t kr_block_size = min(kc - kr_block_start, kr);
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
            *packed_w++ =
              k[(nr_block_start + nr_block_offset) * kc + (kr_block_start + kr_block_offset)];
          }
          packed_w += kr - kr_block_size;
        }
        packed_w += (nr - nr_block_size) * kr;
      }
    }
    k += nc * kc;
  } while (--g != 0);
}

static inline void xnn_pack_f16_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w)
{
  const size_t skr = sr * kr;
  const size_t skc = round_down_po2(kc, skr);
  const size_t sr_mask = (sr - 1) * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_w[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      }
      packed_w += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < skc; kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
              *packed_w++ =
                k[((nr_block_start + nr_block_offset) * ks + ki) * kc + round_down_po2(kr_block_start, skr) + ((kr_block_start + nr_block_offset * kr) & sr_mask) + kr_block_offset];
            }
          }
          packed_w += (nr - nr_block_size) * kr;
        }

        for (size_t kr_block_start = skc; kr_block_start < kc; kr_block_start += kr) {
          const size_t kr_block_size = min(kc - kr_block_start, kr);
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
              *packed_w++ =
                k[((nr_block_start + nr_block_offset) * ks + ki) * kc + (kr_block_start + kr_block_offset)];
            }
            packed_w += kr - kr_block_size;
          }
          packed_w += (nr - nr_block_size) * kr;
        }
      }
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

static inline void xnn_pack_f16_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w)
{
  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_w[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      }
      packed_w += nr;
      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          *packed_w =
            k[ki * g * nc + (nr_block_start + nr_block_offset)];
          packed_w += kr;
        }
        packed_w += (nr - nr_block_size) * kr;
      }
    }
    k += nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

static inline void xnn_pack_f16_dconv_oki_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kh,
  size_t kw,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w)
{
  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    if XNN_LIKELY(b != NULL) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
        *packed_w++ = b[min(nr_block_offset, nr_block_size - 1)];
      }
    } else {
      size_t n = nr;
      do {
        *packed_w++ = 0;
      } while (--n != 0);
    }

    for (size_t kx = 0; kx < kw; kx++) {
      for (size_t c = 0; c < kc; c++) {
        for (size_t ky = 0; ky < kh; ky++) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
            *packed_w++ = k[(((nr_block_start + min(nr_block_offset, nr_block_size - 1)) * kh + ky) * kw + kx) * kc + c];
          }
        }
      }
    }
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nr;
    }
  }
}

static inline void xnn_pack_f16_deconv_goki_w(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w,
  struct subconvolution_params* params)
{
  const size_t skr = sr * kr;
  const size_t skc = round_down_po2(kc, skr);
  const size_t sr_mask = (sr - 1) * kr;
  for (size_t i = 0; i < g; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*params++).weights = packed_w;
        }
        for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          if XNN_LIKELY(b != NULL) {
            for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
              packed_w[nr_block_offset] = b[nr_block_start + nr_block_offset];
            }
          }
          packed_w += nr;
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0; kr_block_start < skc; kr_block_start += kr) {
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                  for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
                    *packed_w++ =
                      k[(((nr_block_start + nr_block_offset) * kh + ky) * kw + kx) * kc + round_down_po2(kr_block_start, skr) + ((kr_block_start + nr_block_offset * kr) & sr_mask) + kr_block_offset];
                  }
                }
                packed_w += (nr - nr_block_size) * kr;
              }

              for (size_t kr_block_start = skc; kr_block_start < kc; kr_block_start += kr) {
                const size_t kr_block_size = min(kc - kr_block_start, kr);
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                  for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
                    *packed_w++ =
                      k[(((nr_block_start + nr_block_offset) * kh + ky) * kw + kx) * kc + (kr_block_start + kr_block_offset)];
                  }
                  packed_w += kr - kr_block_size;
                }
                packed_w += (nr - nr_block_size) * kr;
              }
            }
          }
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

static inline void xnn_pack_f16_dwconv_ghw_w(
  size_t h,
  size_t w,
  size_t c,
  size_t cr,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w)
{
  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    if XNN_LIKELY(b != NULL) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
        *packed_w++ = b[cr_block_start + cr_block_offset];
      }
    } else {
      size_t n = cr_block_size;
      do {
        *packed_w++ = 0;
      } while (--n != 0);
    }
    packed_w += cr - cr_block_size;
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_w++ = kv;
        }
        packed_w += cr - cr_block_size;
      }
    }
  }
}

static inline void xnn_pack_f16_dwconv_hwg_w(
  size_t h,
  size_t w,
  size_t c,
  size_t cr,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w)
{
  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    if XNN_LIKELY(b != NULL) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
        *packed_w++ = b[cr_block_start + cr_block_offset];
      }
    } else {
      size_t n = cr_block_size;
      do {
        *packed_w++ = 0;
      } while (--n != 0);
    }
    packed_w += cr - cr_block_size;
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_w++ = kv;
        }
        packed_w += cr - cr_block_size;
      }
    }
  }
}

static inline void xnn_pack_f16_spchw_dwconv_ghw_w(
  size_t kernel_size,
  size_t groups,
  const uint16_t* kernel,
  const uint16_t* bias,
  uint16_t* packed_weights)
{
  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY(bias != NULL) {
      *packed_weights = *bias++;
    } else {
      *packed_weights = 0;
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = kernel[g * kernel_size + i];
    }
  }
}

static inline void xnn_pack_f32_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  float* packed_w)
{
  const size_t skr = sr * kr;
  const size_t skc = round_down_po2(kc, skr);
  const size_t sr_mask = (sr - 1) * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_w[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      }
      packed_w += nr;

      for (size_t kr_block_start = 0; kr_block_start < skc; kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            *packed_w++ =
              k[(nr_block_start + nr_block_offset) * kc + round_down_po2(kr_block_start, skr) + ((kr_block_start + nr_block_offset * kr) & sr_mask) + kr_block_offset];
          }
        }
        packed_w += (nr - nr_block_size) * kr;
      }

      for (size_t kr_block_start = skc; kr_block_start < kc; kr_block_start += kr) {
        const size_t kr_block_size = min(kc - kr_block_start, kr);
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
            *packed_w++ =
              k[(nr_block_start + nr_block_offset) * kc + (kr_block_start + kr_block_offset)];
          }
          packed_w += kr - kr_block_size;
        }
        packed_w += (nr - nr_block_size) * kr;
      }
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

static inline void xnn_pack_f32_gemm_io_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  float* packed_w)
{
  const size_t skr = sr * kr;
  const size_t skc = round_down_po2(kc, skr);
  const size_t sr_mask = (sr - 1) * kr;
  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    if XNN_LIKELY(b != NULL) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        packed_w[nr_block_offset] = b[nr_block_start + nr_block_offset];
      }
    }
    packed_w += nr;

    for (size_t kr_block_start = 0; kr_block_start < skc; kr_block_start += kr) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
          *packed_w++ =
            k[(round_down_po2(kr_block_start, skr) + ((kr_block_start + nr_block_offset * kr) & sr_mask) + kr_block_offset) * nc + (nr_block_start + nr_block_offset)];
        }
      }
      packed_w += (nr - nr_block_size) * kr;
    }

    for (size_t kr_block_start = skc; kr_block_start < kc; kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr);
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
          *packed_w++ =
            k[(kr_block_start + kr_block_offset) * nc + (nr_block_start + nr_block_offset)];
        }
        packed_w += kr - kr_block_size;
      }
      packed_w += (nr - nr_block_size) * kr;
    }
  }
}

static inline void xnn_pack_f32_gemminc_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  float* packed_w)
{
  const size_t skr = sr * kr;
  const size_t skc = round_down_po2(kc, skr);
  const size_t sr_mask = (sr - 1) * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);

      for (size_t kr_block_start = 0; kr_block_start < skc; kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            *packed_w++ =
              k[(nr_block_start + nr_block_offset) * kc + round_down_po2(kr_block_start, skr) + ((kr_block_start + nr_block_offset * kr) & sr_mask) + kr_block_offset];
          }
        }
        packed_w += (nr - nr_block_size) * kr;
      }

      for (size_t kr_block_start = skc; kr_block_start < kc; kr_block_start += kr) {
        const size_t kr_block_size = min(kc - kr_block_start, kr);
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
            *packed_w++ =
              k[(nr_block_start + nr_block_offset) * kc + (kr_block_start + kr_block_offset)];
          }
          packed_w += kr - kr_block_size;
        }
        packed_w += (nr - nr_block_size) * kr;
      }
    }
    k += nc * kc;
  } while (--g != 0);
}

static inline void xnn_pack_f32_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  float* packed_w)
{
  const size_t skr = sr * kr;
  const size_t skc = round_down_po2(kc, skr);
  const size_t sr_mask = (sr - 1) * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_w[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      }
      packed_w += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < skc; kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
              *packed_w++ =
                k[((nr_block_start + nr_block_offset) * ks + ki) * kc + round_down_po2(kr_block_start, skr) + ((kr_block_start + nr_block_offset * kr) & sr_mask) + kr_block_offset];
            }
          }
          packed_w += (nr - nr_block_size) * kr;
        }

        for (size_t kr_block_start = skc; kr_block_start < kc; kr_block_start += kr) {
          const size_t kr_block_size = min(kc - kr_block_start, kr);
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
              *packed_w++ =
                k[((nr_block_start + nr_block_offset) * ks + ki) * kc + (kr_block_start + kr_block_offset)];
            }
            packed_w += kr - kr_block_size;
          }
          packed_w += (nr - nr_block_size) * kr;
        }
      }
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

static inline void xnn_pack_f32_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  const float* k,
  const float* b,
  float* packed_w)
{
  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_w[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      }
      packed_w += nr;
      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          *packed_w =
            k[ki * g * nc + (nr_block_start + nr_block_offset)];
          packed_w += kr;
        }
        packed_w += (nr - nr_block_size) * kr;
      }
    }
    k += nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

static inline void xnn_pack_f32_dconv_oki_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kh,
  size_t kw,
  const float* k,
  const float* b,
  float* packed_w)
{
  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    if XNN_LIKELY(b != NULL) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
        *packed_w++ = b[min(nr_block_offset, nr_block_size - 1)];
      }
    } else {
      size_t n = nr;
      do {
        *packed_w++ = 0.0f;
      } while (--n != 0);
    }

    for (size_t kx = 0; kx < kw; kx++) {
      for (size_t c = 0; c < kc; c++) {
        for (size_t ky = 0; ky < kh; ky++) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
            *packed_w++ = k[(((nr_block_start + min(nr_block_offset, nr_block_size - 1)) * kh + ky) * kw + kx) * kc + c];
          }
        }
      }
    }
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nr;
    }
  }
}

static inline void xnn_pack_f32_deconv_goki_w(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  float* packed_w,
  struct subconvolution_params* params)
{
  const size_t skr = sr * kr;
  const size_t skc = round_down_po2(kc, skr);
  const size_t sr_mask = (sr - 1) * kr;
  for (size_t i = 0; i < g; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*params++).weights = packed_w;
        }
        for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          if XNN_LIKELY(b != NULL) {
            for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
              packed_w[nr_block_offset] = b[nr_block_start + nr_block_offset];
            }
          }
          packed_w += nr;
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0; kr_block_start < skc; kr_block_start += kr) {
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                  for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
                    *packed_w++ =
                      k[(((nr_block_start + nr_block_offset) * kh + ky) * kw + kx) * kc + round_down_po2(kr_block_start, skr) + ((kr_block_start + nr_block_offset * kr) & sr_mask) + kr_block_offset];
                  }
                }
                packed_w += (nr - nr_block_size) * kr;
              }

              for (size_t kr_block_start = skc; kr_block_start < kc; kr_block_start += kr) {
                const size_t kr_block_size = min(kc - kr_block_start, kr);
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                  for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
                    *packed_w++ =
                      k[(((nr_block_start + nr_block_offset) * kh + ky) * kw + kx) * kc + (kr_block_start + kr_block_offset)];
                  }
                  packed_w += kr - kr_block_size;
                }
                packed_w += (nr - nr_block_size) * kr;
              }
            }
          }
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

static inline void xnn_pack_f32_dwconv_ghw_w(
  size_t h,
  size_t w,
  size_t c,
  size_t cr,
  const float* k,
  const float* b,
  float* packed_w)
{
  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    if XNN_LIKELY(b != NULL) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
        *packed_w++ = b[cr_block_start + cr_block_offset];
      }
    } else {
      size_t n = cr_block_size;
      do {
        *packed_w++ = 0.0f;
      } while (--n != 0);
    }
    packed_w += cr - cr_block_size;
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_w++ = kv;
        }
        packed_w += cr - cr_block_size;
      }
    }
  }
}

static inline void xnn_pack_f32_dwconv_hwg_w(
  size_t h,
  size_t w,
  size_t c,
  size_t cr,
  const float* k,
  const float* b,
  float* packed_w)
{
  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    if XNN_LIKELY(b != NULL) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
        *packed_w++ = b[cr_block_start + cr_block_offset];
      }
    } else {
      size_t n = cr_block_size;
      do {
        *packed_w++ = 0.0f;
      } while (--n != 0);
    }
    packed_w += cr - cr_block_size;
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_w++ = kv;
        }
        packed_w += cr - cr_block_size;
      }
    }
  }
}

static inline void xnn_pack_f32_spchw_dwconv_ghw_w(
  size_t kernel_size,
  size_t groups,
  const float* kernel,
  const float* bias,
  float* packed_weights)
{
  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY(bias != NULL) {
      *packed_weights = *bias++;
    } else {
      *packed_weights = 0.0f;
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = kernel[g * kernel_size + i];
    }
  }
}

static inline void xnn_pack_f32_vmulcaddc_w(
  size_t c,
  size_t cr,
  const float* s,
  const float* b,
  float* packed_w)
{
  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
      *packed_w++ = s[cr_block_start + cr_block_offset];
    }
    packed_w += cr - cr_block_size;
    if XNN_LIKELY(b != NULL) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
        *packed_w++ = b[cr_block_start + cr_block_offset];
      }
    } else {
      size_t n = cr_block_size;
      do {
        *packed_w++ = 0.0f;
      } while (--n != 0);
    }
    packed_w += cr - cr_block_size;
  }
}
