// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/zip.h>


void xnn_x32_zip_x3_ukernel__sse2(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);

  const float* x = (const float*) input;
  const float* y = (const float*) ((uintptr_t) x + n);
  const float* z = (const float*) ((uintptr_t) y + n);
  float* o = (float*) output;

  while (n >= 16) {
    // vx = ( x3, x2, x1, x0 )
    const __m128 vx = _mm_loadu_ps(x);
    x += 4;
    // vy = ( y3, y2, y1, y0 )
    const __m128 vy = _mm_loadu_ps(y);
    y += 4;
    // vz = ( z3, z2, z1, z0 )
    const __m128 vz = _mm_loadu_ps(z);
    z += 4;

    // vxy = ( y2, y0, x2, x0 )
    const __m128 vxy = _mm_shuffle_ps(vx, vy, _MM_SHUFFLE(2, 0, 2, 0));
    // vyz = ( z3, z1, y3, y1 )
    const __m128 vyz = _mm_shuffle_ps(vy, vz, _MM_SHUFFLE(3, 1, 3, 1));
    // vzx = ( x3, x1, z2, z0 )
    const __m128 vzx = _mm_shuffle_ps(vz, vx, _MM_SHUFFLE(3, 1, 2, 0));

    // vxyz0 = ( x1, z0, y0, x0 )
    const __m128 vxyz0 = _mm_shuffle_ps(vxy, vzx, _MM_SHUFFLE(2, 0, 2, 0));
    // vxyz1 = ( y2, x2, z1, y1 )
    const __m128 vxyz1 = _mm_shuffle_ps(vyz, vxy, _MM_SHUFFLE(3, 1, 2, 0));
    // vxyz2 = ( z3, y3, x3, z2 )
    const __m128 vxyz2 = _mm_shuffle_ps(vzx, vyz, _MM_SHUFFLE(3, 1, 3, 1));

    _mm_storeu_ps(o, vxyz0);
    _mm_storeu_ps(o + 4, vxyz1);
    _mm_storeu_ps(o + 8, vxyz2);
    o += 12;
    n -= 16;
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & 8) {
      // vx = ( -, -, x1, x0 )
      const __m128 vx = _mm_castpd_ps(_mm_load_sd((const double*) x));
      x += 2;
      // vy = ( -, -, y1, y0 )
      const __m128 vy = _mm_castpd_ps(_mm_load_sd((const double*) y));
      y += 2;
      // vz = ( -, -, z1, z0 )
      const __m128 vz = _mm_castpd_ps(_mm_load_sd((const double*) z));
      z += 2;

      // vxy = ( y1, x1, y0, x0 )
      const __m128 vxy = _mm_unpacklo_ps(vx, vy);
      // vzx = ( x1, z1, x0, z0 )
      const __m128 vzx = _mm_unpacklo_ps(vz, vx);
      // vyz = ( z1, y1, z0, y0 )
      const __m128 vyz = _mm_unpacklo_ps(vy, vz);

      _mm_storeu_ps(o, _mm_shuffle_ps(vxy, vzx, _MM_SHUFFLE(3, 0, 1, 0)));
      _mm_storeh_pi((__m64*) (o + 4), vyz);
      o += 6;
    }
    if (n & 4) {
      const __m128 vx = _mm_load_ss(x);
      const __m128 vy = _mm_load_ss(y);
      const __m128 vz = _mm_load_ss(z);
      _mm_store_ss(o, vx);
      _mm_store_ss(o + 1, vy);
      _mm_store_ss(o + 2, vz);
    }
  }
}
