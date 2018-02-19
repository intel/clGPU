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
#define SIMD_PER_GROUP 16

#define TILE SIMD*SIMD_PER_GROUP

__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(SIMD, SIMD_PER_GROUP, 1)))
__kernel void Scasum_simd16x16(int n, __global complex_t* x, int incx, __global float* result)
{
    const uint lid = get_sub_group_local_id();
    const uint sid = get_sub_group_id();
    uint id = sid*SIMD + lid;
    uint ind_x = id*incx;
    const uint inc_x = TILE*incx;

    float subsum = 0.f;
    for (; id < n; id += TILE, ind_x += inc_x) 
    {
        subsum += scabs1(x[ind_x]);
    }

    float sum = sub_group_reduce_add(subsum);

    // Reduce work_groups sum using first sub_group
    __local float local_sums[SIMD_PER_GROUP];
    local_sums[sid] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (sid == 0) 
    {
        subsum = local_sums[lid];
        sum = sub_group_reduce_add(subsum);
        result[0] = sum;
    }
}
