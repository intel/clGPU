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
#define WORK_GROUPS 256

#define GROUP_TILE SIMD*SIMD_PER_GROUP
#define TILE GROUP_TILE*WORK_GROUPS

__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(SIMD, SIMD_PER_GROUP, 1)))
__kernel void Snrm2_simd16_first_stage(__global float* result, uint n, __global float* x, uint incx)
{
    const uint slid = get_sub_group_local_id();
    const uint sid = get_sub_group_id();
    const uint gid = get_group_id(2);
    const uint id = gid*GROUP_TILE + sid*SIMD + slid;
    uint ind = id*incx;
    const uint inc = TILE*incx;
    const uint max_ind = n*incx;

    float subsum = 0.f;
    for (; ind < max_ind; ind += inc) {
        subsum += x[ind] * x[ind];
    }

    float sum = sub_group_reduce_add(subsum);

    // Reduce work_group sums using first sub_group
    __local float local_sums[SIMD_PER_GROUP];
    local_sums[sid] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (sid == 0) {
        subsum = local_sums[slid];
        sum = sub_group_reduce_add(subsum);
        result[gid] = sum;
    }
}

__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(SIMD, SIMD_PER_GROUP, 1)))
__kernel void Snrm2_simd16_second_stage(__global float* result, __global float* x) {
    const uint slid = get_sub_group_local_id();
    const uint sid = get_sub_group_id();
    const uint ind = sid*SIMD + slid;

    float subsum = x[ind];

    float sum = sub_group_reduce_add(subsum);

    // Reduce all sub_groups using first sub_group and slm
    __local float local_sums[SIMD_PER_GROUP];
    local_sums[sid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sid == 0) {
        subsum = local_sums[slid];
        sum = sub_group_reduce_add(subsum);
        result[0] = sqrt(sum);
    }
}
