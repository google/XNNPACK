# Copyright (C) 2024 Intel Corporation
#  
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#  
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#  
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  
#  
# SPDX-License-Identifier: BSD-3-Clause

#################################### Scalar ###################################
tools/xngen src/qs8-vrpreluc/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -o src/qs8-vrpreluc/gen/qs8-vrpreluc-scalar-u1.c &
tools/xngen src/qs8-vrpreluc/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -o src/qs8-vrpreluc/gen/qs8-vrpreluc-scalar-u2.c &
tools/xngen src/qs8-vrpreluc/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vrpreluc/gen/qs8-vrpreluc-scalar-u4.c &
tools/xngen src/qs8-vrpreluc/scalar.c.in -D BATCH_TILE=8 -D DATATYPE=QS8 -o src/qs8-vrpreluc/gen/qs8-vrpreluc-scalar-u8.c &
 
tools/xngen src/qs8-vrpreluc/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -o src/qu8-vrpreluc/gen/qu8-vrpreluc-scalar-u1.c &
tools/xngen src/qs8-vrpreluc/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -o src/qu8-vrpreluc/gen/qu8-vrpreluc-scalar-u2.c &
tools/xngen src/qs8-vrpreluc/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vrpreluc/gen/qu8-vrpreluc-scalar-u4.c &
tools/xngen src/qs8-vrpreluc/scalar.c.in -D BATCH_TILE=8 -D DATATYPE=QU8 -o src/qu8-vrpreluc/gen/qu8-vrpreluc-scalar-u8.c &

#################################### AVX2 ###################################
tools/xngen src/qs8-vrpreluc/avx2.c.in -D BATCH_TILE=16 -D AVX=1 -D DATATYPE=QS8 -o src/qs8-vrpreluc/gen/qs8-vrpreluc-avx2-u16.c &

tools/xngen src/qs8-vrpreluc/avx2.c.in -D BATCH_TILE=16 -D AVX=1 -D DATATYPE=QU8 -o src/qu8-vrpreluc/gen/qu8-vrpreluc-avx2-u16.c &


wait
