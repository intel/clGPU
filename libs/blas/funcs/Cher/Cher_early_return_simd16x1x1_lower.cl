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

#define IDX(m, n, ld) (n)*(ld) + (m)

#define SIMD (16)

// Each sub group calculates tile of size 16x16, one work item calulates 16 values from a row
// Work size is n x n/16 and sub groups in upper part of matrix are returned before calculations
__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(SIMD, 1, 1)))
__kernel void Cher_early_return_simd16x1x1_lower(uint n, float alpha, __global complex_t* x, uint incx, __global complex_t* a, uint lda)
{
    const uint group_m = get_group_id(0);
    const uint group_n = get_group_id(1);
    const uint group_start_m = group_m * SIMD;
    const uint group_start_n = group_n * SIMD;

    const bool upper = group_start_m < group_start_n;
    if (upper) return; // Early return tiles that are in upper part of matrix

    const uint slid = get_sub_group_local_id();
    const uint ind_m = group_start_m + slid;
    const uint ind_n = group_start_n + slid;

    const bool diagonal = group_start_m == group_start_n; // Tile is on a diagonal
    const bool last_row = group_start_m + SIMD > n; // Tile is in last row and exceeds target area

    if (last_row && diagonal) {
        if (ind_m >= n) return;
        complex_t left_x = x[ind_m * incx];
        complex_t right_x = conjg(left_x);

        uint index_a = IDX(ind_m, group_start_n, lda);
        uint columns = n - group_start_n;
        for (int i = 0; i < columns; i++) {
            if (i <= slid) {
                complex_t prod = cmul(left_x, intel_sub_group_shuffle(right_x, i));
                complex_t this_a = a[index_a];
                this_a = cfmaf(prod, alpha, this_a);
                if (i == slid) this_a.y = 0.f;
                a[index_a] = this_a;
                index_a += lda;
            }
        }
    } else if (last_row) {
        complex_t left_x;
        if (ind_m < n) left_x = x[ind_m * incx];
        complex_t right_x = conjg(x[ind_n * incx]);

        uint index_a = IDX(ind_m, group_start_n, lda);

        __attribute__((opencl_unroll_hint(SIMD)))
        for (int i = 0; i < SIMD; i++) {
            complex_t prod = cmul(left_x, intel_sub_group_shuffle(right_x, i));
            if (ind_m < n) {
                complex_t this_a = a[index_a];
                this_a = cfmaf(prod, alpha, this_a);
                a[index_a] = this_a;
                index_a += lda;
            }
        }
    } else if (diagonal) {
        complex_t left_x = x[ind_m * incx];
        complex_t right_x = conjg(left_x);

        uint index_a = IDX(ind_m, group_start_n, lda);

        __attribute__((opencl_unroll_hint(SIMD)))
        for (int i = 0; i < SIMD; i++) {
            if (i <= slid) {
                complex_t this_a = a[index_a];
                complex_t prod = cmul(left_x, intel_sub_group_shuffle(right_x, i));
                this_a = cfmaf(prod, alpha, this_a);
                if (i == slid) this_a.y = 0.f;
                a[index_a] = this_a;
                index_a += lda;
            }
        }
    } else {
        complex_t left_x = x[ind_m * incx];
        complex_t right_x = conjg(x[ind_n * incx]);

        uint index_a = IDX(ind_m, group_start_n, lda);

        __attribute__((opencl_unroll_hint(SIMD)))
        for (int i = 0; i < SIMD; i++) {
            complex_t this_a = a[index_a];
            complex_t prod = cmul(left_x, intel_sub_group_shuffle(right_x, i));
            this_a = cfmaf(prod, alpha, this_a);
            a[index_a] = this_a;
            index_a += lda;
        }
    }
}
