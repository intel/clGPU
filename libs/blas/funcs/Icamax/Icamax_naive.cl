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

typedef struct /* Index and Value type that holds index and value used in this kernel */
{
    uint index; 
    float value; 
} iav_type;

__kernel void Icamax_naive(int n, __global complex_t* x, int incx, __global int* res)
{
    iav_type lowest;
    lowest.index = 0;
    lowest.value = scabs1(x[0]);

    for(uint i = 1; i<n; ++i)
    {
        iav_type element;
        element.index = i * incx;
        element.value = scabs1(x[element.index]);

        if(element.value > lowest.value)
        {   
            lowest.index = element.index;
            lowest.value = element.value;
        }
    }

    res[0] = lowest.index;
}
