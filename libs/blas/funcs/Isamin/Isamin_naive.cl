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

__kernel void Isamin_naive(int n, __global float* x, int incx, __global int* res)
{
    int lowest_index = 0;
    float lowest_value = fabs(x[0]);

    for(uint i = 1; i<n; ++i)
    {
        float current_value = fabs(x[i * incx]);

        if(current_value < lowest_value)
        {   
            lowest_value = current_value;
            lowest_index = i * incx;
        }
    }

    res[0] = lowest_index;
}
