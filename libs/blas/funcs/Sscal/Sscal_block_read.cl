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
#define WI_ELEMS 1
#define VEC_SIZE 4

#include "vector_operations.h"

#define WIDTH ((SIMD) * (VEC_SIZE))

__attribute__((reqd_work_group_size(SIMD, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SIMD)))
__kernel void Sscal_block_read(uint n, float alpha, __global float* x)
{
    const uint gid = get_group_id(0);
    const uint gsz = get_global_size(0);
    const uint sglid = get_sub_group_local_id();

    const uint inc = gsz * VEC_SIZE;

    uint group_ind = gid * WIDTH;

    __attribute__((opencl_unroll_hint(WI_ELEMS)))
    for (uint elem = 0; elem < WI_ELEMS; ++elem)
    {
        FLOAT_VS this_x = AS_FLOAT_VS( BLOCK_READ_VS( (__global uint*)x + group_ind ) );
        this_x *= alpha;
        BLOCK_WRITE_VS( (__global uint*)x + group_ind, AS_UINT_VS( this_x ));

        group_ind += inc;
    }
    // Leftovers
    while (group_ind < n)
    {
        uint ind = group_ind + sglid;
        __attribute__((opencl_unroll_hint(VEC_SIZE)))
        for (uint elem = 0; elem < VEC_SIZE; ++elem)
        {
            if (ind < n)
            {
                float this_x = x[ind];
                this_x *= alpha;
                x[ind] = this_x;

                ind += SIMD;
            }
        }

        group_ind += inc;
    }
}
