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

__kernel void Srotm_async(__global float* x, int incx, __global float* y, int incy, __global float* param)
{
    int gid = get_global_id(0);

    float _x = param[1] * x[gid * incx] + param[2] * y[gid * incy];
    y[gid * incy] = param[3] * x[gid * incx] + param[4] * y[gid * incy];
    x[gid * incx] = _x;
}
