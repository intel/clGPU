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

__kernel void Sger_naive(float alpha, __global float* x, int incx, __global float* y, int incy, __global float* a, int lda)
{
    int ind_x = get_global_id(0);
    int ind_y = get_global_id(1);
    a[ind_y*lda + ind_x] += alpha * x[ind_x*incx] * y[ind_y*incy];
}
