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

__kernel void Cgeru_naive(complex_t alpha, __global complex_t* x, uint incx, __global complex_t* y, uint incy, __global complex_t* a, uint lda)
{
    const uint ind_m = get_global_id(0);
    const uint ind_n = get_global_id(1);

    uint index_a = IDX(ind_m, ind_n, lda);

    complex_t this_a = a[index_a];
    complex_t left_x = x[ind_m * incx];
    complex_t right_y = y[ind_n * incy];

    this_a += cmul(alpha, cmul(left_x, right_y));
    a[index_a] = this_a;
}
