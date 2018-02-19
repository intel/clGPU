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

#define ACCESS(A, i, j, N) A[(j)*(N) + (i)]

/* Access for Lower Hermitian banded matrix coordinates from normal matrix coordinates */
#define LHB_ACCESS(A, i, j, N) ACCESS(A, i-j, j, N)

/* Access for Upper Hermitian banded matrix coordinates from normal matrix coordinates 
* k = number of sub-diagonals in matrix
*/
#define UHB_ACCESS(A, i, j, k, N) ACCESS(A, k+i-j, j, N)

__kernel void Chbmv_naive(int uplo, int n, int k, complex_t alpha, __global complex_t* A, int lda, __global complex_t* x, int incx, complex_t beta, __global complex_t* y, int incy)
{
    /* Global ID - One thread calculate it's own result for corresponding vector index */
    int global_id = get_global_id(0);
    int startIndex;
    int maxIndex;
    
    /* Lower */
    if(uplo == 1)
    {
        startIndex = max(global_id - k, 0);
        maxIndex = min(global_id + k, n-1);
    
        complex_t l_result = (complex_t)(0,0);
    
        for(int i = startIndex; i<=maxIndex; ++i)
        {
            /* Since the entry matrix contains only maindiagonal and k-subdiagonals, we inferred upper k-superdiagonals by
            /* conjugating complex number stored in transposed position
            */
            if(i > global_id)
                {
                    l_result += cmul(cmul(conjg(LHB_ACCESS(A, global_id, i, lda)), alpha), x[i * incx]);
                }
            else
                {
                    l_result += cmul(cmul(LHB_ACCESS(A, global_id, i, lda), alpha), x[i * incx]);
                }
        }
        y[global_id * incy] = l_result + cmul(beta, y[global_id * incy]);
    }
    
    /* Upper */
    else
    {
        startIndex = max(global_id - k, 0);
        maxIndex = min(global_id + k, n-1);
    
        complex_t l_result = (complex_t)(0,0);
    
        for(int i = startIndex; i<=maxIndex; ++i)
        {
            /* Since the entry matrix contains only maindiagonal and k-superdiagonals, we inferred lower k-subdiagonals by
            /* conjugating complex number stored in transposed position
            */
            if(i < global_id)
                {
                    l_result += cmul(cmul(conjg(LHB_ACCESS(A, global_id, i, lda)), alpha), x[i * incx]);
                }
            else
                {
                    l_result += cmul(cmul(LHB_ACCESS(A, global_id, i, lda), alpha), x[i * incx]);
                }
        }
        y[global_id * incy] = l_result + cmul(beta, y[global_id * incy]);
    }
}
