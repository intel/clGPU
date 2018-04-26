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

// Each tile has size 16x16 and is computed by single sub group
// Each work_item computes 1x16 values from a tile
// Work items in sub group share values of y vector using intel_sub_group_shuffle
__attribute__((reqd_work_group_size(SIMD, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SIMD)))
__kernel void Sger_simd16x1x1(uint m, uint n, float alpha, __global float* x, uint incx, __global float* y, uint incy, __global float* a, uint lda)
{
    const uint sid = get_sub_group_local_id();

    const uint gid_m = get_group_id(0);
    const uint gid_n = get_group_id(1);
    const uint group_start_m = gid_m * SIMD;
    const uint group_start_n = gid_n * SIMD;

    const uint ind_m = group_start_m + sid;
    const uint ind_n = group_start_n + sid;

    // Check if this tile exceeds target region in some dimension
    bool over_m = (gid_m + 1) * SIMD > m;
    bool over_n = (gid_n + 1) * SIMD > n;

    float this_x;
    float this_y;

    __global float* ptr_a = a + group_start_n * lda + ind_m;

    #define SGER_UPDATE_PROD(ptr, prod)     \
        float this_a = ptr[0];              \
        this_a = mad(alpha, prod, this_a);  \
        ptr[0] = this_a

    #define SGER_UPDATE(ptr, col)                                   \
        float prod = this_x * intel_sub_group_shuffle(this_y, col); \
        SGER_UPDATE_PROD(ptr, prod)

    // If tile exceeds update matrix a using subset of sub group work items and calculate values up to the edge
    if (over_m && over_n) {
        if (ind_m < m) this_x = x[ind_m * incx];
        if (ind_n < n) this_y = y[ind_n * incy];

        const uint columns = n - group_start_n;
        for (uint col = 0; col < columns; col++) {
            float prod = sub_group_broadcast(this_y, col);
            if (ind_m < m) {
                prod *= this_x;
                SGER_UPDATE_PROD(ptr_a, prod);
                ptr_a += lda;
            }
        }
    // If tile exceed only in first dimension work items that fit into target area are only ones performing calculations
    } else if (over_m) {
        if (ind_m < m) this_x = x[ind_m * incx];
        this_y = y[ind_n * incy];

        __attribute__((opencl_unroll_hint(SIMD)))
        for (uint col = 0; col < SIMD; col++) {
            float prod = sub_group_broadcast(this_y, col);
            if (ind_m < m) {
                prod *= this_x;
                SGER_UPDATE_PROD(ptr_a, prod);
                ptr_a += lda;
            }
        }
    // If tile exceeds only in second dimension all work items perform calculations up to the edge
    } else if (over_n) {
        this_x = x[ind_m * incx];
        if (ind_n < n) this_y = y[ind_n * incy];

        const uint columns = n - group_start_n;
        for (uint col = 0; col < columns; col++) {
            SGER_UPDATE(ptr_a, col);
            ptr_a += lda;
        }
    // If whole tile fits into target area perform calculation on all work items without any bounds checking
    } else {
        this_x = x[ind_m * incx];
        this_y = y[ind_n * incy];

        __attribute__((opencl_unroll_hint(SIMD)))
        for (uint col = 0; col < SIMD; col++) {
            SGER_UPDATE(ptr_a, col);
            ptr_a += lda;
        }
    }

    #undef SGER_UPDATE
    #undef SGER_UPDATE_PROD
}
