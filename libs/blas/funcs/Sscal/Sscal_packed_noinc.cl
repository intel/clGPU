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

#define LWG_SIZE 32
#define WI_ELEMS 1
#define VEC_SIZE 4

#include "vector_operations.h"

__attribute__((reqd_work_group_size(LWG_SIZE, 1, 1)))
__kernel void Sscal_packed_noinc(uint n, float alpha, __global float* x)
{
    const uint gid = get_global_id(0);
    uint ind = gid;

    const uint gsz = get_global_size(0);
    const uint inc = gsz;

    __attribute__((opencl_unroll_hint(WI_ELEMS)))
    for (int i = 0; i < WI_ELEMS; i++)
    {
        FLOAT_VS this_x = VLOAD_VS( ind, x );
        this_x *= alpha;
        VSTORE_VS( this_x, ind, x );
        ind += inc;
    }

    // leftovers
    ind *= VEC_SIZE;
    while (ind < n)
    {
        __attribute__((opencl_unroll_hint(VEC_SIZE)))
        for (uint elem = 0; elem < VEC_SIZE; ++elem)
        {
            if (ind + elem < n)
            {
                float this_x = x[ind + elem];
                this_x *= alpha;
                x[ind + elem] = this_x;
            }
        }

        ind += inc;
    }
}
