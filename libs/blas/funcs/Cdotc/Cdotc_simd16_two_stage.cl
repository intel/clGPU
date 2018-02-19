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
__attribute__((reqd_work_group_size(SIMD, SIMD_PER_GROUP, 1)))
__kernel void Cdotc_simd16_first_stage(int n, __global complex_t* x, int incx, __global complex_t* y, int incy, __global complex_t* result)
{
    const uint lid = get_sub_group_local_id();
    const uint sid = get_sub_group_id();
    const uint gid = get_group_id(2);
    uint id = gid*GROUP_TILE + sid*SIMD + lid;
    uint ind_x = id*incx;
    uint ind_y = id*incy;
    const uint inc_x = TILE*incx;
    const uint inc_y = TILE*incy;

    complex_t subsum = (float2)0.f;
    for (; id < n; id += TILE, ind_x += inc_x, ind_y += inc_y) {
        complex_t this_x = x[ind_x];
        complex_t this_y = y[ind_y];
        this_x = conjg(this_x);
        subsum += cmul(this_x, this_y);
    }

    complex_t sum;
    sum.x = sub_group_reduce_add(subsum.x);
    sum.y = sub_group_reduce_add(subsum.y);

    // Reduce work_groups sum using first sub_group
    __local complex_t local_sums[SIMD_PER_GROUP];
    local_sums[sid] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (sid == 0) {
        subsum = local_sums[lid];
        sum.x = sub_group_reduce_add(subsum.x);
        sum.y = sub_group_reduce_add(subsum.y);
        result[gid] = sum;
    }
}

__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(SIMD, SIMD_PER_GROUP, 1)))
__kernel void Cdotc_simd16_second_stage(__global complex_t* x, __global complex_t* result) {
    const uint lid = get_sub_group_local_id();
    const uint sid = get_sub_group_id();
    const uint ind = sid*SIMD + lid;

    complex_t subsum = x[ind];

    complex_t sum;
    sum.x = sub_group_reduce_add(subsum.x);
    sum.y = sub_group_reduce_add(subsum.y);

    // Reduce work_groups sum using first sub_group
    __local complex_t local_sums[SIMD_PER_GROUP];
    local_sums[sid] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (sid == 0) {
        subsum = local_sums[lid];
        sum.x = sub_group_reduce_add(subsum.x);
        sum.y = sub_group_reduce_add(subsum.y);
        result[0] = sum;
    }
}
