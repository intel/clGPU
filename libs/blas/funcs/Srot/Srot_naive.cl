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

__kernel void Srot_naive(uint n, __global float* x, int incx, __global float* y, int incy, float c, float s)
{
    for (uint i = 0; i < n; i++)
    {
        uint idx = i * incx;
        uint idy = i * incy;

        float current_x = x[idx];
        float current_y = y[idy];

        float sy = s * current_y;
        float cy = c * current_y;

        float _x = fma(c, current_x, sy);
        y[idy] = fma(-s, current_x, cy);

        x[idx] = _x;
    }
}
