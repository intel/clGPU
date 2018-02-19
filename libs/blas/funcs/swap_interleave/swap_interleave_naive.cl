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

__kernel void swap_interleave_naive(__global char* x, uint incx, __global char* y, uint incy, int elem_size)
{
    uint gid = get_global_id(0);
    for (uint i = 0; i<elem_size; i++) {
        char this_x = x[gid*incx + i];
        char this_y = y[gid*incy + i];
        x[gid*incx + i] = this_y;
        y[gid*incy + i] = this_x;
    }
}
