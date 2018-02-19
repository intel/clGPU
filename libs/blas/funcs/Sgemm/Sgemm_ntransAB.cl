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

__kernel void Sgemm_ntransAB(uint k, float alpha, __global float* A, uint lda, __global float* B, uint ldb, float beta, __global float* C, uint ldc)
{
    uint row = get_global_id(0);
    uint column = get_global_id(1);

    float value = 0.f;

    for(uint i = 0; i < k; i++)
    {
        uint indexA = i * lda + row;
        float valueA = A[indexA];

        uint indexB = column * ldb + i;
        float valueB = B[indexB];

        value = fma(valueA, valueB, value);
    }

    uint indexC = column * ldc + row;
    float valueC = C[indexC];

    float betaC = beta * valueC;
    
    C[indexC] = fma(alpha, value, betaC);
}
