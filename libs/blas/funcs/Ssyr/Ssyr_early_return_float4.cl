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

// Separate into header file
#define ICLBLAS_FILL_MODE_LOWER (1)
//

#define IDX(m, n, ld) ((n)*(ld) + (m))

#define VEC_SIZE (4)

__kernel void Ssyr_early_return_float4(int uplo, uint n, float alpha, __global float* x, uint incx, __global float* a, uint lda)
{
    const uint ind_m = get_global_id(0) * VEC_SIZE;
    const uint ind_n = get_global_id(1);

    const bool ltriangle = uplo == ICLBLAS_FILL_MODE_LOWER;

    const bool upper = ind_m + VEC_SIZE - 1 < ind_n;
    const bool lower = ind_m > ind_n;

    if (ltriangle && upper) return;
    if (!ltriangle && lower) return;

    const bool diag = ind_m / VEC_SIZE == ind_n / VEC_SIZE;
    const bool last_row = ind_m + VEC_SIZE > n;

    float right_x = x[ind_n * incx];

    #define SSYR_UPDATE(row, col)                     \
        float this_a = a[IDX(row, col, lda)];         \
        float left_x = x[(row) * incx];               \
        float prod = left_x * right_x;                \
        this_a = mad(alpha, prod, this_a);            \
        a[IDX(row, col, lda)] = this_a

    if (ltriangle && diag && last_row) {
        __attribute__((opencl_unroll_hint(VEC_SIZE)))
        for (int i = 0; i < VEC_SIZE; i++) {
            if (ind_m + i >= ind_n && ind_m + i < n) {
                SSYR_UPDATE(ind_m + i, ind_n);
            }
        }
    } else if (ltriangle && last_row) {
        __attribute__((opencl_unroll_hint(VEC_SIZE)))
        for (int i = 0; i < VEC_SIZE; i++) {
            if (ind_m + i < n) {
                SSYR_UPDATE(ind_m + i, ind_n);
            }
        }
    } else if (ltriangle && diag) {
        __attribute__((opencl_unroll_hint(VEC_SIZE)))
        for (int i = 0; i < VEC_SIZE; i++) {
            if (ind_m + i >= ind_n) {
                SSYR_UPDATE(ind_m + i, ind_n);
            }
        }
    } else if (!ltriangle && diag) {
        __attribute__((opencl_unroll_hint(VEC_SIZE)))
        for (int i = 0; i < VEC_SIZE; i++) {
            if (ind_m + i <= ind_n) {
                SSYR_UPDATE(ind_m + i, ind_n);
            }
        }
    } else {
        float4 left_x = (float4)(x[ind_m * incx], x[ind_m * incx + incx], x[ind_m * incx + 2 * incx], x[ind_m * incx + 3 * incx]);

        uint index_a = IDX(ind_m, ind_n, lda);
        index_a /= VEC_SIZE;
        float4 this_a = vload4(index_a, a);
        float4 prod = left_x * (float4)right_x;
        this_a = mad((float4)alpha, prod, this_a);
        vstore4(this_a, index_a, a);
    }

    #undef SSYR_UPDATE
}
