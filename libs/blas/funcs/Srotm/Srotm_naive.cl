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

__kernel void Srotm_naive(uint n, __global float* x, uint incx, __global float* y, uint incy, __global float* param)
{
    if (param[0] == -1.f)
    {
        for (uint i = 0; i < n; i++)
        {
            float this_x = x[i * incx];
            float this_y = y[i * incy];

            ROT_IMPL( ROT_FULL, x[i * incx], y[i * incy], this_x, this_y, param );
        }
    }
    else if (param[0] == 0.f)
    {
        for (uint i = 0; i < n; i++)
        {
            float this_x = x[i * incx];
            float this_y = y[i * incy];

            ROT_IMPL( ROT_DIAGONAL_ONES, x[i * incx], y[i * incy], this_x, this_y, param );
        }
    }
    else if (param[0] == 1.f)
    {
        for (uint i = 0; i < n; i++)
        {
            float this_x = x[i * incx];
            float this_y = y[i * incy];

            ROT_IMPL( ROT_ANTI_DIAGONAL_ONES, x[i * incx], y[i * incy], this_x, this_y, param );
        }
    }
}
