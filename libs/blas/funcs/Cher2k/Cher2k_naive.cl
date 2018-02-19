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

__kernel void Cher2k_naive(int uplo, int trans, int n, int k, complex_t alpha, __global complex_t* A, int lda, __global complex_t* B, int ldb, float beta, __global complex_t* C, int ldc)
{
    bool isABNTrans = (trans == 0);
    bool isCUp = (uplo == 0);

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            complex_t value1 = (complex_t)(0.f, 0.f);
            complex_t value2 = (complex_t)(0.f, 0.f);
            
            if(isABNTrans)
            {
                for(int l = 0; l < k; l++)
                {
                    value1 += cmul(A[i * lda + l], conjg(B[i * ldb + l]));
                    value2 += cmul(B[i * ldb + l], conjg(A[i * lda + l]));
                }
            }

            if(!isABNTrans)
            {
                for(int l = 0; l < k; l++)
                {
                    value1 += cmul(conjg(A[l * lda + i]), B[l * ldb + i]);
                    value2 += cmul(conjg(B[l * ldb + i]), A[l * lda + i]);
                }
            }

            int ildc = i * ldc;
            int jldc = j * ldc;

            if(ildc == jldc)
                C[jldc + i].y = 0.f;

            if (isCUp && jldc >= ildc || !isCUp && jldc <= ildc)
            {
                C[jldc + i] = cmul(value1, alpha) + cmul(value2, conjg(alpha)) + cmulf(C[jldc + i], beta);
            }
        }
    }
}
