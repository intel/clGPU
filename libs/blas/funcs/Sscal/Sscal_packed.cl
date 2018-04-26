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

__attribute__((reqd_work_group_size(LWG_SIZE, 1, 1)))
__kernel void Sscal_packed(uint n, float alpha, __global float* x, uint incx)
{
    const uint gid = get_global_id(0);
    uint ind = gid * incx;
    const uint max_ind = n * incx;
    const uint gsz = get_global_size(0);
    const uint inc = gsz * incx;

    __attribute__((opencl_unroll_hint(WI_ELEMS)))
    for (uint elem = 0; elem < WI_ELEMS; elem++)
    {
        x[ind] *= alpha;
        ind += inc;
    }

    // leftovers
    while (ind < max_ind)
    {
        x[ind] *= alpha;
        ind += inc;
    }
}
