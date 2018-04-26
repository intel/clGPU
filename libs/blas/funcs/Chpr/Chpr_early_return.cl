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

__kernel void Chpr_early_return(int uplo, int n, float alpha, __global complex_t* x, uint incx, __global complex_t* a)
{
    bool ltriangle = uplo == ICLBLAS_FILL_MODE_LOWER;

    const uint ind_m = get_global_id(0);
    const uint ind_n = get_global_id(1);

    const bool upper = ind_m < ind_n;
    const bool lower = ind_m > ind_n;

    if (ltriangle && upper) return;
    if (!ltriangle && lower) return;

    const uint index_a = ltriangle ? IDX_PACKED_LOWER(ind_m, ind_n, n) : IDX_PACKED_UPPER(ind_m, ind_n);

    complex_t left_x = x[ind_m * incx];
    complex_t right_x = x[ind_n * incx];
    complex_t this_a = a[index_a];

    complex_t prod = cmul(left_x, conjg(right_x));
    this_a = cfmaf(prod, alpha, this_a);
    if (ind_m == ind_n) this_a.y = 0.f;

    a[index_a] = this_a;
}
