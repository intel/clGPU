/* Copyright (c) 2017-2018 Intel Corporation
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define SIMD 16
#define VEC_SIZE 4
#define TILE (SIMD * VEC_SIZE)

#define VLIST2(tab, start, inc) (float2)(tab[(start)], tab[(start) + (inc)])
#define VLIST4(tab, start, inc) (float4)(VLIST2(tab, start, inc), VLIST2(tab, (start) + 2 * (inc), inc))

// Each tile has size 64x64 and is computed by single sub group
// Each work item computes 4x64 values from a tile using float4s
// Work items in sub group share values of y vector using intel_sub_group_shuffle
__attribute__((reqd_work_group_size(SIMD, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SIMD)))
__kernel void Sger_simd16x4x4(uint m, uint n, float alpha, __global float* x, uint incx, __global float* y, uint incy, __global float* a, uint lda)
{
    lda /= VEC_SIZE;
    const uint sid = get_sub_group_local_id();

    const uint gid_m = get_group_id(0);
    const uint gid_n = get_group_id(1);
    const uint group_start_m = gid_m * TILE;
    const uint group_start_n = gid_n * TILE;

    const uint ind_m = group_start_m + VEC_SIZE * sid;
    const uint ind_n = group_start_n + VEC_SIZE * sid;

    // Check if this tile exceeds target region in some dimension
    bool over_m = (gid_m + 1) * TILE > m;
    bool over_n = (gid_n + 1) * TILE > n;

    uint index_a = group_start_n * lda + ind_m / VEC_SIZE;

    //////// Helper macros, assuming this_x and this_y of correct type exist
    // Update 4 values in a using float4
    #define SGER_UPDATE_A4(ty_idx, shuffle_idx) {\
        float4 this_a = vload4(index_a, a); \
        float4 prod = this_x * (float4) intel_sub_group_shuffle(this_y[ty_idx], shuffle_idx); \
        this_a = mad((float4)alpha, prod, this_a); \
        vstore4(this_a, index_a, a); }
    // Update 1 value in a using float
    #define SGER_UPDATE_A1(ty_idx, tx_idx, shuffle_idx) { \
        float this_a = a[index_a]; \
        float prod = this_x[tx_idx] * intel_sub_group_shuffle(this_y[ty_idx], shuffle_idx); \
        this_a = mad(alpha, prod, this_a); \
        a[index_a] = this_a; }
    ////////

    // If tile exceeds in both directions perform update using floats
    if (over_m && over_n) {
        float this_x[VEC_SIZE];
        __attribute__((opencl_unroll_hint(VEC_SIZE)))
        for (int tx_idx = 0, row = ind_m; tx_idx < VEC_SIZE; tx_idx++, row++) {
            if (row < m) this_x[tx_idx] = x[row * incx];
        }

        float this_y[VEC_SIZE];
        __attribute__((opencl_unroll_hint(VEC_SIZE)))
        for (uint ty_idx = 0, col = ind_n; ty_idx < VEC_SIZE; ty_idx++, col++) {
            if (col < n) this_y[ty_idx] = y[col * incy];
        }

        index_a *= VEC_SIZE;

        for (uint i = 0, col = group_start_n; col < n; i++, col++) {
            __attribute__((opencl_unroll_hint(VEC_SIZE)))
            for (uint tx_idx = 0, row = ind_m; tx_idx < VEC_SIZE; tx_idx++, row++) {

                float prod = intel_sub_group_shuffle(this_y[i % VEC_SIZE], i / VEC_SIZE);
                if (row < m) {
                    prod *= this_x[tx_idx];

                    float this_a = a[index_a];
                    this_a = mad(alpha, prod, this_a);
                    a[index_a] = this_a;
                }
                index_a++;
            }
            index_a -= VEC_SIZE;
            index_a += lda * VEC_SIZE;
        }
    // If tile exceeds only in first dimension, check for possibility to use float4
    // If cant use float4s use single float
    } else if (over_m) {
        float4 this_y = VLIST4(y, ind_n * incy, incy);

        const uint rows = m - group_start_m;
        // Can use float4 for calculations
        if (rows % VEC_SIZE == 0) {
            float4 this_x = 0;
            if (ind_m < m) this_x = VLIST4(x, ind_m * incx, incx);

            __attribute__((opencl_unroll_hint(SIMD)))
            for (uint shuffle_idx = 0; shuffle_idx < SIMD; shuffle_idx++) {
                __attribute__((opencl_unroll_hint(VEC_SIZE)))
                for (uint ty_idx = 0; ty_idx < VEC_SIZE; ty_idx++) {

                    float4 prod = intel_sub_group_shuffle(this_y[ty_idx], shuffle_idx);
                    if (ind_m < m) {
                        prod *= this_x;

                        float4 this_a = vload4(index_a, a);
                        this_a = mad(alpha, prod, this_a);
                        vstore4(this_a, index_a, a);

                        index_a += lda;
                    }
                }
            }
        // Cant use float4 -> use float
        } else {
            float this_x[VEC_SIZE];
            __attribute__((opencl_unroll_hint(VEC_SIZE)))
            for (uint tx_idx = 0, row = ind_m; tx_idx < VEC_SIZE; tx_idx++, row++) {
                if (row < m) this_x[tx_idx] = x[row * incx];
            }

            index_a *= VEC_SIZE;

            __attribute__((opencl_unroll_hint(SIMD)))
            for (uint shuffle_idx = 0; shuffle_idx < SIMD; shuffle_idx++) {
                __attribute__((opencl_unroll_hint(VEC_SIZE)))
                for (uint ty_idx = 0; ty_idx < VEC_SIZE; ty_idx++) {
                    __attribute__((opencl_unroll_hint(VEC_SIZE)))
                    for (uint tx_idx = 0; tx_idx < VEC_SIZE; tx_idx++) {
                        float prod = intel_sub_group_shuffle(this_y[ty_idx], shuffle_idx);
                        if (ind_m + tx_idx < m) {
                            prod *= this_x[tx_idx];

                            float this_a = a[index_a];
                            this_a = mad(alpha, prod, this_a);
                            a[index_a] = this_a;
                        }
                        index_a += 1;
                    }
                    index_a -= VEC_SIZE;
                    index_a += VEC_SIZE * lda;
                }
            }
        }
    // If tile exeeds only in second dimension all work_items use float4 and calulate values up to the edge
    } else if (over_n) {
        float4 this_x = VLIST4(x, ind_m * incx, incx);

        float this_y[VEC_SIZE];
        __attribute__((opencl_unroll_hint(VEC_SIZE)))
        for (int ty_idx = 0, col = ind_n; ty_idx < VEC_SIZE; ty_idx++, col++) {
            if (col < n) this_y[ty_idx] = y[col * incy];
        }

        uint shuffle_idx = 0;
        uint col = group_start_n;
        for (; col + VEC_SIZE - 1 < n; col += VEC_SIZE, shuffle_idx++) {
            __attribute__((opencl_unroll_hint(VEC_SIZE)))
            for (uint ty_idx = 0; ty_idx < VEC_SIZE; ty_idx++) {
                SGER_UPDATE_A4(ty_idx, shuffle_idx);
                index_a += lda;
            }
        }

        for (uint ty_idx = 0; col < n; col++, ty_idx++) {
            SGER_UPDATE_A4(ty_idx, shuffle_idx);
            index_a += lda;
        }
    // If whole tile fits into target area perform calculation on all work items without any bounds checking
    } else {
        float4 this_x = VLIST4(x, ind_m * incx, incx);
        float4 this_y = VLIST4(y, ind_n * incy, incy);

        __attribute__((opencl_unroll_hint(SIMD)))
        for (uint shuffle_idx = 0; shuffle_idx < SIMD; shuffle_idx++) {
            __attribute__((opencl_unroll_hint(VEC_SIZE)))
            for (uint ty_idx = 0; ty_idx < VEC_SIZE; ty_idx++) {
                SGER_UPDATE_A4(ty_idx, shuffle_idx);
                index_a += lda;
            }
        }
    }
    #undef SGER_UPDATE_A4
    #undef SGER_UPDATE_A1
}
