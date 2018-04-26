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

__kernel void Ssyr_early_return(int uplo, float alpha, __global float* x, uint incx, __global float* a, uint lda)
{
    const uint ind_m = get_global_id(0);
    const uint ind_n = get_global_id(1);

    const bool ltriangle = uplo == ICLBLAS_FILL_MODE_LOWER;

    const bool upper = ind_m < ind_n;
    const bool lower = ind_m > ind_n;

    if (ltriangle && upper) return;
    if (!ltriangle && lower) return;

    float left_x = x[ind_m * incx];
    float right_x = x[ind_n * incx];
    float this_a = a[IDX(ind_m, ind_n, lda)];
    float prod = left_x * right_x;
    this_a = mad(alpha, prod, this_a);
    a[IDX(ind_m, ind_n, lda)] = this_a;
}
