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

#define ICLBLAS_FILL_MODE_LOWER (1)

#define IDX(m, n, ld) (n)*(ld) + (m)

// Every work item calculates two values, the second being directly under first in ouput matrix
// Work size is n/2 x n and work items in triangle other than targeted are returned before calculations
__kernel void Cher_early_return_float4(int uplo, uint n, float alpha, __global complex_t* x, uint incx, __global float* a, uint lda)
{
    const uint ind_m = get_global_id(0) * 2;
    const uint ind_n = get_global_id(1);

    const bool ltriangle = uplo == ICLBLAS_FILL_MODE_LOWER;

    const bool upper = ind_m + 1 < ind_n;
    const bool lower = ind_m > ind_n;

    if (ltriangle && upper) return;
    if (!ltriangle && lower) return;

    // Check if tile is on diagonal and only one element is in target area
    const bool diag = ltriangle ? ind_m + 1 == ind_n : ind_m == ind_n;

    const bool last_row = ind_m + 2 > n; // Tile is in last row and exceeds target area

    uint index_a = IDX(ind_m, ind_n, lda);

    complex_t right_x = conjg(x[ind_n * incx]);

    // In last row, when target area is lower triangle, calculate only first value
    if (ltriangle && last_row) {
        complex_t left_x = x[ind_m * incx];
        complex_t this_a = vload2(index_a, a);

        complex_t prod = cmul(left_x, right_x);
        this_a = cfmaf(prod, alpha, this_a);
        if (ind_m == ind_n) this_a.y = 0.f;

        vstore2(this_a, index_a, a);
    } else if (diag) {
        complex_t left_x;
        if (ltriangle) {
            index_a += 1;
            left_x = x[(ind_m + 1)*incx];
        } else {
            left_x = x[ind_m * incx];
        }
        complex_t this_a = vload2(index_a, a);

        complex_t prod = cmul(left_x, right_x);
        this_a = cfmaf(prod, alpha, this_a);
        this_a.y = 0.f;

        vstore2(this_a, index_a, a);
    } else {
        float4 left_x = (float4)(x[ind_m * incx], x[ind_m * incx + incx]);
        index_a /= 2;
        float4 this_a = vload4(index_a, a);

        float4 prod = (float4)(cmul(left_x.xy, right_x), cmul(left_x.zw, right_x));
        this_a = fma(prod, alpha, this_a);

        if (ltriangle && ind_m == ind_n) this_a.y = 0.f;
        else if (!ltriangle && ind_m + 1 == ind_n) this_a.w = 0.f;

        vstore4(this_a, index_a, a);
    }
}
