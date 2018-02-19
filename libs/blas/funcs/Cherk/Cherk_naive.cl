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

__kernel void Cherk_naive(int uplo, int trans, int n, int k, float alpha, __global complex_t* A, int lda, float beta, __global complex_t* C, int ldc)
{
    bool isANTrans = (trans == 0);
    bool isCUp = (uplo == 0);

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            complex_t value = (complex_t)(0.f, 0.f);
            
            if(isANTrans)
            {
                for(int l = 0; l < k; l++)
                {
                    value += cmul(A[i * lda + l], conjg(A[i * lda + l]));
                }
            }

            if(!isANTrans)
            {
                for(int l = 0; l < k; l++)
                {
                    value += cmul(conjg(A[l * lda + i]), A[l * lda + i]);
                }
            }

            int ildc = i * ldc;
            int jldc = j * ldc;

            if(ildc == jldc)
                C[jldc + i].y = 0.f;

            if (uplo == 0 && jldc >= ildc || uplo == 1 && jldc <= ildc)
            {
                C[jldc + i] = cmulf(value, alpha) + cmulf(C[jldc + i], beta);
            }
        }
    }
}
