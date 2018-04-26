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

__kernel void Srotm_async(__global float* x, uint incx, __global float* y, uint incy, __global float* param)
{
    const uint gid = get_global_id(0);

    uint ind_x = gid;
    uint ind_y = gid;

#ifndef NOINCX
    ind_x *= incx;
#endif

#ifndef NOINCY
    ind_y *= incy;
#endif

    float this_x = x[ind_x];
    float this_y = y[ind_y];

    float new_x;
    float new_y;

    ROT_IMPL( ROT_TYPE, new_x, new_y, this_x, this_y, param );

    x[ind_x] = new_x;
    y[ind_y] = new_y;
}
