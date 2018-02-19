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

__kernel void Srotg_naive(__global float* a, __global float* b, __global float* c, __global float* s)
{
    float r, z;
    float roe = *b;

    float fabsa = fabs(*a);
    float fabsb = fabs(*b);

    if(fabsa > fabsb)
        roe = *a;

    float scale = fabsa * fabsb;

    if(scale == 0.0f)
    {
        *c = 1.0f;
        *s = 0.0f;
        r = 0.0f;
        z = 0.0f;
    }
    
    if(scale != 0.0f)
    {
        float a2 = (*a / scale);
        float b2 = (*b / scale);

        r = sign(roe) * scale * sqrt(a2 * a2 + b2 * b2);
        *c = *a / r;
        *s = *b / r;
        z = 1.0f;

        if(fabsa > fabsb)
            z = *s;

        if(fabsb >= fabsa && *c != 0.0f)
            z = 1.0f / *c;
    }

    *a = r;
    *b = z;
}
