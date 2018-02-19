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

__attribute__((reqd_sub_group_size(SIMD)))
__attribute__((intel_reqd_sub_group_size(SIMD)))
__kernel void Snrm2_simd16_single_thread(__global float* result, uint n, __global float* x, uint incx)
{
    const uint id = get_sub_group_local_id();
    uint ind = id*incx;
    const uint inc = SIMD*incx;
    const uint max_ind = n*incx;

    float subsum = 0.f;
    for (; ind < max_ind; ind += inc) {
        subsum += x[ind] * x[ind];
    }

    float sum = sub_group_reduce_add(subsum);
    result[0] = sqrt(sum);
}
