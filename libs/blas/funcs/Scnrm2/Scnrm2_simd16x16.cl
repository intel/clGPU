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
__kernel void Scnrm2_simd16x16(uint n, __global complex_t* x, uint incx, __global float* res)
{
    const uint slid = get_sub_group_local_id();
    const uint sid = get_sub_group_id();
    const uint id = sid*SIMD + slid;
    const uint max_ind = n*incx;
    const uint inc = TILE*incx;
    uint ind = id*incx;

    float subsum = 0.f;
    for (; ind < max_ind; ind += inc) {
        subsum += cnorm(x[ind]);
    }

    float sum = sub_group_reduce_add(subsum);
    // Reduce 16 sub_groups in work_group using first work_group and slm
    __local float local_sum[SIMD_PER_GROUP];
    local_sum[sid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sid == 0) {
        subsum = local_sum[slid];
        sum = sub_group_reduce_add(subsum);
        res[0] = sqrt(sum);
    }
}
