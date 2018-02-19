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

#define GROUP_SIZE 256

__kernel void Sasum_slm_reduction(int n, __global float* x, int incx, __global float* result)
{
    const int lid = get_local_id(0);
    const int gid = get_group_id(0);
    const int gnum = get_num_groups(0);
    const int ind = get_global_id(0);
    const int lsz = get_local_size(0);

    __local float local_sum[GROUP_SIZE];

    local_sum[lid] = fabs(x[ind*incx]);
    // Add leftovers before reduction, only in the last work_group
    if (gid == gnum - 1 && ind + GROUP_SIZE < n) {
        local_sum[lid] += fabs(x[(ind + GROUP_SIZE)*incx]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    int plsz = lsz;
    for (int i = (lsz + 1)>> 1; plsz > 1; plsz = i, i = (i + 1) >> 1) {
        if (lid < i && plsz - lid - 1 != lid) local_sum[lid] += local_sum[plsz - lid - 1];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) result[gid] = local_sum[0];
}
