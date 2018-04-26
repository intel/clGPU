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

#define IDX(m, n, ld) ((n)*(ld) + (m))

#define SIMD (16)
#define VEC_SIZE (4)
#define TILE (SIMD * VEC_SIZE)

#define VLIST2(tab, start, inc) (float2)(tab[(start)], tab[(start) + (inc)])
#define VLIST4(tab, start, inc) (float4)(VLIST2(tab, start, inc), VLIST2(tab, (start) + 2 * (inc), inc))

// Each tile has size 64x64 - calculated by single sub group
// One work item calculates 4x64 fragment
__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(SIMD, 1, 1)))
__kernel void Ssyr_simd16x4x4_upper(uint n, float alpha, __global float* x, uint incx, __global float* a, uint lda)
{
    lda /= VEC_SIZE;
    const uint slid = get_sub_group_local_id();

    const uint tiles = (n + TILE - 1) / TILE;
    const uint tile_number = get_group_id(0);

    const bool perfect_fit = tiles * TILE == n;
    
    // Calculate starting coordinates of sub_group tile
    uint sub_group_start_m = 0;
    uint sub_group_start_n = 0;
    if (slid == 0) {
        uint tiles_left = tile_number;
        while (tiles_left + sub_group_start_m >= tiles) {
            tiles_left = tiles_left - tiles + sub_group_start_m;
            sub_group_start_n += 1;
            sub_group_start_m = sub_group_start_n;
        }
        sub_group_start_m += tiles_left;
        sub_group_start_m -= sub_group_start_n;
        sub_group_start_n = tiles - sub_group_start_n - 1;

        sub_group_start_m *= TILE;
        sub_group_start_n *= TILE;
    }

    sub_group_start_m = sub_group_broadcast(sub_group_start_m, 0);
    sub_group_start_n = sub_group_broadcast(sub_group_start_n, 0);

    const uint ind_m = sub_group_start_m + slid * VEC_SIZE;
    const uint ind_n = sub_group_start_n + slid * VEC_SIZE;
    
    float4 left_x = (float4)0.f;
    float4 right_x = (float4)0.f;

    uint index_a = IDX(ind_m / VEC_SIZE, sub_group_start_n, lda);

    #define SSYR4_UPDATE_PROD(prod)                \
        float4 this_a = vload4(index_a, a);        \
        this_a = mad((float4)alpha, prod, this_a); \
        vstore4(this_a, index_a, a)

    #define SSYR4_SIMD_UPDATE(shuffle_idx, right_idx)                                            \
        float4 prod = left_x * (float4)intel_sub_group_shuffle(right_x[right_idx], shuffle_idx); \
        SSYR4_UPDATE_PROD(prod)

    // Handle tiles that exceed target area
    if (!perfect_fit && (tiles - 1) * TILE == sub_group_start_n) {
        if (sub_group_start_m == sub_group_start_n) {
            __attribute__((opencl_unroll_hint(VEC_SIZE)))
            for (uint i = 0; i < VEC_SIZE; i++) {
                if (ind_n + i < n) right_x[i] = x[(ind_n + i) * incx];
            }
            left_x = right_x;

            uint columns = n - sub_group_start_n;
            for (uint i = 0; i < columns; i++) {
                if (i >= slid * VEC_SIZE) {
                    float4 this_a = vload4(index_a, a);
                    float4 prod = left_x * (float4)intel_sub_group_shuffle(right_x[i%VEC_SIZE], i/VEC_SIZE);
                    this_a = mad((float4)alpha, prod, this_a);
                    if (i >= slid * VEC_SIZE + 3) {
                        vstore4(this_a, index_a, a);
                    } else {
                        for (int j=0; j< i % VEC_SIZE + 1; j++) {
                            a[VEC_SIZE * index_a + j] = this_a[j];
                        }
                    }
                }
                index_a += lda;
            }
        } else {
            left_x = VLIST4(x, ind_m * incx, incx);
            __attribute__((opencl_unroll_hint(VEC_SIZE)))
            for (uint i = 0; i < VEC_SIZE; i++) {
                if (ind_n + i < n) right_x[i] = x[(ind_n + i) * incx];
            }

            uint iterations = (n - sub_group_start_n + VEC_SIZE - 1) / VEC_SIZE;
            for (uint i = 0; i < iterations; i++) {
                __attribute__((opencl_unroll_hint(VEC_SIZE)))
                for (uint j = 0; j < VEC_SIZE; j++) {
                    SSYR4_SIMD_UPDATE(i, j);
                    index_a += lda;
                }
            }
        }
    // Tiles that fit into target area and are on a diagonal
    } else if (sub_group_start_m == sub_group_start_n) {
        left_x = VLIST4(x, ind_m * incx, incx);
        right_x = left_x;

        index_a += slid * lda * VEC_SIZE;

        __attribute__((opencl_unroll_hint(SIMD)))
        for(int i = 0; i < SIMD; i++) {
            if (i >= slid) {
                __attribute__((opencl_unroll_hint(VEC_SIZE)))
                for (uint j = 0; j < VEC_SIZE; j++) {
                    float4 this_a = vload4(index_a, a);
                    float4 prod = left_x * (float4)intel_sub_group_shuffle(right_x[j], i);
                    this_a = mad((float4)alpha, prod, this_a);
                    if (slid == i && j != 3) {
                        for (uint k = 0; k < j + 1; k++) {
                            a[VEC_SIZE*index_a + k] = this_a[k];
                        }
                    } else {
                        vstore4(this_a, index_a, a);
                    }
                    index_a += lda;
                }
            }
        }
    // Tiles that fit and are not on a diagonal
    } else {
        left_x = VLIST4(x, ind_m * incx, incx);
        right_x = VLIST4(x, ind_n * incx, incx);

        __attribute__((opencl_unroll_hint(SIMD)))
        for (uint i = 0; i < SIMD; i++) {
            __attribute__((opencl_unroll_hint(VEC_SIZE)))
            for (uint j = 0; j < VEC_SIZE; j++) {
                SSYR4_SIMD_UPDATE(i, j);
                index_a += lda;
            }
        }
    }

    #undef SSYR4_SIMD_UPDATE
    #undef SSYR4_UPDATE_PROD
}
