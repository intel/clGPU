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

__kernel void Sger_naive(float alpha, __global float* x, uint incx, __global float* y, uint incy, __global float* a, uint lda)
{
    const uint ind_m = get_global_id(0);
    const uint ind_n = get_global_id(1);

    const uint index_a = ind_n*lda + ind_m;
    float this_a = a[index_a];
    float prod = x[ind_m * incx] * y[ind_n * incy];
    this_a = mad(alpha, prod, this_a);
    a[index_a] = this_a;
}
