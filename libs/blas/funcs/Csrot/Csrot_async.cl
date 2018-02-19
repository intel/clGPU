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

__kernel void Csrot_async(__global complex_t* x, int incx, __global complex_t* y, int incy, float c, float s)
{
    uint gid = get_global_id(0);

    uint idx = gid * incx;
    uint idy = gid * incy;

    complex_t x_this = x[idx];
    complex_t y_this = y[idy];

    x[idx] = cfmaf(x_this, c,  cmulf(y_this, s));
    y[idy] = cfmaf(x_this, -s, cmulf(y_this, c));
}
