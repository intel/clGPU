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

__kernel void Srotm_packed(uint n, __global float* x, uint incx, __global float* y, uint incy, __global float* param)
{
    const uint gid = get_global_id(0);
    const uint gsz = get_global_size(0);

    uint ind = gid;
    uint ind_x = ind;
    uint ind_y = ind;

    uint inc_x = gsz;
    uint inc_y = gsz;

#ifndef NOINCX
    ind_x *= incx;
    inc_x *= incx;
#endif

#ifndef NOINCY
    ind_y *= incy;
    inc_y *= incy;
#endif

    __attribute__((opencl_unroll_hint(WI_ELEMS)))
    for (uint elem = 0; elem < WI_ELEMS; ++elem)
    {
        float this_x = x[ind_x];
        float this_y = y[ind_y];

        float new_x;
        float new_y;

        ROT_IMPL( ROT_TYPE, new_x, new_y, this_x, this_y, param );

        x[ind_x] = new_x;
        y[ind_y] = new_y;
            
        ind += gsz;
        ind_x += inc_x;
        ind_y += inc_y;
    }

    // Leftovers
    if (ind < n)
    {
        float this_x = x[ind_x];
        float this_y = y[ind_y];

        float new_x;
        float new_y;

        ROT_IMPL( ROT_TYPE, new_x, new_y, this_x, this_y, param );

        x[ind_x] = new_x;
        y[ind_y] = new_y;
    }
}
