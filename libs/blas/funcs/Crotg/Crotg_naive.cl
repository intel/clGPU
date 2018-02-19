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

__kernel void Crotg_naive(__global complex_t* a, __global complex_t* b, __global float* c, __global complex_t* s)
{
    float fabsa = cabs(*a);
    float fabsb = cabs(*b);

    if(fabsa == 0.f)
    {
        *c = 0.f;
        *s = (complex_t) (1.f, 0.f);
        *a = *b;
    }

    if(fabsa != 0.f)
    {
        float scale = fabsa + fabsb;
        
        float cabsdiva = cabs(cdivf(*a, scale));
        float cabsdivb = cabs(cdivf(*b, scale));

        float norm = scale * sqrt(cabsdiva * cabsdiva + cabsdivb * cabsdivb);
        complex_t alpha = cdivf(*a, fabsa);

        *c = fabsa / norm;

        *s = cdivf(cmul(conjg(*b), alpha), norm);
        *a = cmulf(alpha, norm);
    }
}
