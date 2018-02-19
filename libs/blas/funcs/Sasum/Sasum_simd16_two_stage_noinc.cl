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
#define VEC_SIZE 4

#define SIMD_TILE SIMD*VEC_SIZE
#define GROUP_TILE SIMD_TILE*SIMD_PER_GROUP
#define TILE GROUP_TILE*WORK_GROUPS

__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(SIMD, SIMD_PER_GROUP, 1)))
__kernel void Sasum_simd16_first_stage_noinc(uint n, __global float* x, __global float* result)
{
    __local float group_sum[SIMD_PER_GROUP];
    const uint simd_id = get_sub_group_id();
    const uint simd_lid = get_sub_group_local_id();
    const uint group_id = get_group_id(2);
    const uint id = simd_lid*VEC_SIZE + simd_id*SIMD_TILE + group_id*GROUP_TILE;
    uint ind = id;
    const uint inc = TILE;
    const uint max_ind = n;
    
    float4 subsum = (float4)0.f;
    for (; ind + VEC_SIZE < max_ind; ind += inc) {
        float4 this_x = vload4(ind/4, x);
        subsum += fabs(this_x);
    }

    // Handle leftovers
    for (; ind < max_ind; ind += 1) {
        subsum.x += fabs(x[ind]);
    }

    subsum.x = subsum.x + subsum.y + subsum.z + subsum.w;
    float sum = sub_group_reduce_add(subsum.x);
    group_sum[simd_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    // Reduce all simds in work_group using first simd
    if (simd_id == 0) {
        float simd_sum = group_sum[simd_lid];
        sum = sub_group_reduce_add(simd_sum);
        result[group_id] = sum;
    }
}

__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(SIMD, SIMD_PER_GROUP, 1)))
__kernel void Sasum_simd16_second_stage_noinc(__global float* x, __global float* result)
{
    __local float group_sum[SIMD_PER_GROUP];
    const uint simd_id = get_sub_group_id();
    const uint simd_lid = get_sub_group_local_id();
    const uint ind = simd_lid + simd_id*SIMD_PER_GROUP;
    
    float subsum = x[ind];

    float sum = sub_group_reduce_add(subsum);
    group_sum[simd_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    // Reduce all simds in work_group using first simd
    if (simd_id == 0) {
        subsum = group_sum[simd_lid];
        sum = sub_group_reduce_add(subsum);
        result[0] = sum;
    }
}
