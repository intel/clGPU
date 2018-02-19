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

__kernel void Csrot_noinc(__global complex_t* x, __global complex_t* y, float c, float s)
{
    uint gid = get_global_id(0);

    complex_t x_this = x[gid];
    complex_t y_this = y[gid];

    x[gid] = cfmaf(x_this, c, cmulf(y_this, s));
    y[gid] = cfmaf(x_this, -s, cmulf(y_this, c));
}
