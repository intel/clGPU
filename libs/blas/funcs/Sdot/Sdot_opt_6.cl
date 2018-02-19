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


#define SIMDS_COUNT (256/16)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void Sdot_opt_6(int n, __global float* x, int incx, __global float* y, int incy, __global float* result)
{
    int gid = get_global_id(0);
    int simd_id = get_sub_group_id();
    int groups_num = get_num_groups(0);
    int group_id = get_group_id(0);

    __local float local_acc[SIMDS_COUNT];

    float acc = 0.f;
    int element_id = gid;

    for (int i = 0; i<(n / (256 * groups_num)); ++i)
    {
        acc += x[element_id * incx] * y[element_id * incy];
        element_id += 256 * groups_num;
    }
    // leftovers
    if (element_id < n)
        acc += x[element_id * incx] * y[element_id * incy];

    // sum all of the accumulators
    float simd_acc = sub_group_reduce_add(acc);

    local_acc[simd_id] = simd_acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (simd_id == 0)
    {
        float _local_acc = local_acc[get_sub_group_local_id()];

        float final_acc = sub_group_reduce_add(_local_acc);
        result[group_id] = final_acc;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void sum(__global float *x, __global float* result)
{
    int gid = get_global_id(0);

    float acc = x[gid];
    float simd_acc = sub_group_reduce_add(acc);
    result[0] = simd_acc;
}
