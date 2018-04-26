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

#include "Srotm_helpers.h"

#ifndef ROT_TYPE
#define ROT_TYPE ROT_FULL
#endif

#ifndef WI_ELEMS
#define WI_ELEMS 4
#endif

#ifndef SIMD
#define SIMD 16
#endif

#ifndef VEC_SIZE
#define VEC_SIZE 1
#endif

#include "vector_operations.h"

#define WIDTH ((SIMD) * (VEC_SIZE))

__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(SIMD, 1, 1)))
__kernel void Srotm_block_read(uint n, __global float* x, __global float* y, __global float* param)
{
    const uint gid = get_group_id(0);
    const uint gsz = get_global_size(0);
    const uint sglid = get_sub_group_local_id();

    uint ind = gid * WIDTH;
    uint inc = gsz * VEC_SIZE;

    __attribute__((opencl_unroll_hint(WI_ELEMS)))
    for (uint elem = 0; elem < WI_ELEMS; ++elem)
    {
        FLOAT_VS this_x = AS_FLOAT_VS( BLOCK_READ_VS( (__global uint*)x + ind ) );
        FLOAT_VS this_y = AS_FLOAT_VS( BLOCK_READ_VS( (__global uint*)y + ind ) );

        FLOAT_VS new_x;
        FLOAT_VS new_y;

        ROT_IMPL( ROT_TYPE, new_x, new_y, this_x, this_y, param );

        BLOCK_WRITE_VS( (__global uint*)x + ind, AS_UINT_VS( new_x ) );
        BLOCK_WRITE_VS( (__global uint*)y + ind, AS_UINT_VS( new_y ) );

        ind += inc;
    }

    ind += sglid;
    // Leftovers
    while (ind < n)
    {
        float this_x = x[ind];
        float this_y = y[ind];

        float new_x;
        float new_y;

        ROT_IMPL( ROT_TYPE, new_x, new_y, this_x, this_y, param );

        x[ind] = new_x;
        y[ind] = new_y;

        ind += gsz;
    }
}

#undef WIDTH
