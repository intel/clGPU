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

#define IDX_PACKED_UPPER(m, n) (((n) + 1) * (n) / 2 + (m))
#define IDX_PACKED_LOWER(m, n, N) ((2 * (N) - (n) - 1) * (n) / 2 + (m))

__kernel void Chpr2_early_return(int uplo, uint n, complex_t alpha, __global complex_t* x, uint incx, __global complex_t* y, uint incy, __global complex_t* a)
{
    bool ltriangle = uplo == ICLBLAS_FILL_MODE_LOWER;

    const uint ind_m = get_global_id(0);
    const uint ind_n = get_global_id(1);

    const bool upper = ind_m < ind_n;
    const bool lower = ind_m > ind_n;

    if (ltriangle && upper) return;
    if (!ltriangle && lower) return;

    uint index_a;
    if (ltriangle) {
        index_a = IDX_PACKED_LOWER(ind_m, ind_n, n);
    } else {
        index_a = IDX_PACKED_UPPER(ind_m, ind_n);
    }

    complex_t left_x = x[ind_m * incx];
    complex_t left_y = y[ind_m * incy];
    complex_t right_x = conjg(x[ind_n * incx]);
    complex_t right_y = conjg(y[ind_n * incy]);
    complex_t this_a = a[index_a];

    complex_t prod1 = cmul(left_x, right_y);
    complex_t prod2 = cmul(left_y, right_x);

    this_a += cmul(alpha, prod1);
    this_a += cmul(conjg(alpha), prod2);
    if (ind_m == ind_n) this_a.y = 0.f;
    a[index_a] = this_a;
}
