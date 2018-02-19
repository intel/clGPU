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

__kernel void Ctrmm_naive(int side, int uplo, int transa, int diag, int m, int n, complex_t alpha, __global complex_t* A, int lda, __global complex_t* B, int ldb, __global complex_t* C, int ldc)
{
    //Quick return if possible
    if(m == 0 || n == 0) return;

    const int globalRow_id = get_global_id(0);
    const int globalCol_id = get_global_id(1); 

    /********************************************************* Left Side ***************************************************************************/
    if(side == 0)
    {
        int K = m;

        /* Upper */
        if(uplo == 0)
        {    
            /* Upper Check */
            if(globalCol_id >= globalRow_id)
            {
                /* OP(A) = A */
                if(transa == 0)
                {
                    complex_t acc = (complex_t)(0, 0);
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalRow_id && diag == 1) /* Check if is on diagonal, if yes and have diag == 1 then assume that A have here 1 + 0i, else do normal multiplication */
                            acc += cmul(cmul((complex_t)(1, 0), alpha), ACCESS(B, k, globalCol_id, ldb));
                        else
                            acc += cmul(cmul(ACCESS(A, globalRow_id, k, lda), alpha), ACCESS(B, k, globalCol_id, ldb)); //[globalCol_id*K + k]
                    }

                    C[globalCol_id*m + globalRow_id] = acc; /* Write result to buffer */
                }

                else if(transa == 1) /* OP(A) = A**T */
                {
                    complex_t acc = (complex_t)(0, 0);
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalRow_id && diag == 1) /* Check if is on diagonal, if yes and have diag == 1 then assume that A have here 1 + 0i, else do normal multiplication */
                            acc += cmul(cmul((complex_t)(1, 0), alpha),ACCESS(B, k, globalCol_id, ldb));
                        else
                            acc += cmul(cmul(ACCESS(A, k, globalRow_id, lda), alpha), ACCESS(B, k, globalCol_id, ldb));
                    }

                    C[globalCol_id*m + globalRow_id] = acc; /* Write result to buffer */
                }

                else if(transa == 2) //If OP(A) = A**T
                {
                    complex_t acc = (complex_t)(0, 0);
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalRow_id && diag == 1) /* Check if is on diagonal, if yes and have diag == 1 then assume that A have here 1 + 0i, else do normal multiplication */
                            acc += cmul(cmul((complex_t)(1, 0), alpha), ACCESS(B, k, globalCol_id, ldb));
                        else
                            acc += cmul(cmul(ACCESS(A, k, globalRow_id, lda), alpha), ACCESS(B, k, globalCol_id, ldb));
                    }

                    C[globalCol_id*m + globalRow_id] = acc; /* Write result to buffer */
                }
            }
        }

        /* Lower */
        else
        {
            /* Lower check */
            if(globalRow_id >= globalCol_id)
            {
                
                //if OP(A) = A
                if(transa == 0)
                {
                    complex_t acc = (complex_t)(0, 0);
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalRow_id && diag == 1) /* Check if is on diagonal, if yes and have diag == 1 then assume that A have here 1 + 0i, else do normal multiplication */
                            acc += cmul(cmul((complex_t)(1, 0), alpha), ACCESS(B, k, globalCol_id, ldb));
                        else
                            acc += cmul(cmul(ACCESS(A, globalRow_id, k, lda), alpha), ACCESS(B, k, globalCol_id, ldb));
                    }

                    C[globalCol_id*m + globalRow_id] = acc; /* Write result to buffer */
                }
                else if(transa == 1) //If OP(A) = A**T 
                {
                    complex_t acc = (complex_t)(0, 0);
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalRow_id && diag == 1) /* Check if is on diagonal, if yes and have diag == 1 then assume that A have here 1 + 0i, else do normal multiplication */
                            acc += cmul(cmul((complex_t)(1, 0), alpha), ACCESS(B, k, globalCol_id, ldb));
                        else
                            acc += cmul(cmul(ACCESS(A, k, globalRow_id, lda), alpha), ACCESS(B, k, globalCol_id, ldb));
                    }

                    C[globalCol_id*m + globalRow_id] = acc; /* Write result to buffer */
                }
                else if(transa == 2) //If OP(A) = A**T 
                {
                    complex_t acc = (complex_t)(0, 0);
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalRow_id && diag == 1) /* Check if is on diagonal, if yes and have diag == 1 then assume that A have here 1 + 0i, else do normal multiplication */
                            acc += cmul(cmul((complex_t)(1, 0), alpha), ACCESS(B, k, globalCol_id, ldb));
                        else
                            acc += cmul(cmul(ACCESS(A, k, globalRow_id, lda), alpha), ACCESS(B, k, globalCol_id, ldb));
                    }

                    C[globalCol_id*m + globalRow_id] = acc; /* Write result to buffer */
                }
            }
        }
    }

    /********************************************************* Right Side ***************************************************************************/
    else
    {
        int K = n;

        /* Upper */
        if(uplo == 0)
        {    
            /* Upper Check */
            if(globalCol_id >= globalRow_id)
            {
                /* OP(A) = A */
                if(transa == 0)
                {
                    complex_t acc = (complex_t)(0, 0);
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalRow_id && diag == 1) /* Check if is on diagonal, if yes and have diag == 1 then assume that A have here 1 + 0i, else do normal multiplication */
                            acc += cmul(cmul((complex_t)(1, 0), alpha), B[globalCol_id*K + k]);
                        else
                            acc += cmul(cmul(ACCESS(B, globalRow_id, k, ldb), alpha), ACCESS(A, k, globalCol_id, lda));
                    }

                    C[globalCol_id*m + globalRow_id] = acc; /* Write result to buffer */
                }

                else if(transa == 1) /* OP(A) = A**T */
                {
                    complex_t acc = (complex_t)(0, 0);
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalRow_id && diag == 1) /* Check if is on diagonal, if yes and have diag == 1 then assume that A have here 1 + 0i, else do normal multiplication */
                            acc += cmul(cmul((complex_t)(1, 0), alpha), B[globalCol_id*K + k]);
                        else
                            acc += cmul(cmul(ACCESS(B, globalRow_id, k, ldb), alpha), ACCESS(A, globalCol_id, k, lda));
                    }

                    C[globalCol_id*m + globalRow_id] = acc; /* Write result to buffer */
                }

                else if(transa == 2) //If OP(A) = A**T
                {
                    complex_t acc = (complex_t)(0, 0);
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalRow_id && diag == 1) /* Check if is on diagonal, if yes and have diag == 1 then assume that A have here 1 + 0i, else do normal multiplication */
                            acc += cmul(cmul((complex_t)(1, 0), alpha), B[globalCol_id*K + k]);
                        else
                            acc += cmul(cmul(ACCESS(B, globalRow_id, k, ldb), alpha), ACCESS(A, globalCol_id, k, lda));
                    }

                    C[globalCol_id*m + globalRow_id] = acc; /* Write result to buffer */
                }
            }
        }

        /* Lower */
        else
        {
            /* Lower check */
            if(globalRow_id >= globalCol_id)
            {
                
                //if OP(A) = A
                if(transa == 0)
                {
                    complex_t acc = (complex_t)(0, 0);
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalRow_id && diag == 1) /* Check if is on diagonal, if yes and have diag == 1 then assume that A have here 1 + 0i, else do normal multiplication */
                            acc += cmul(cmul((complex_t)(1, 0), alpha), B[globalCol_id*K + k]);
                        else
                            acc += cmul(cmul(ACCESS(A, globalRow_id, k, lda), alpha), B[globalCol_id*K + k]);
                    }

                    C[globalCol_id*m + globalRow_id] = acc; /* Write result to buffer */
                }
                else if(transa == 1) //If OP(A) = A**T 
                {
                    complex_t acc = (complex_t)(0, 0);
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalRow_id && diag == 1) /* Check if is on diagonal, if yes and have diag == 1 then assume that A have here 1 + 0i, else do normal multiplication */
                            acc += cmul(cmul((complex_t)(1, 0), alpha), B[globalCol_id*K + k]);
                        else
                            acc += cmul(cmul(ACCESS(B, globalRow_id, k, ldb), alpha), ACCESS(A, globalCol_id, k, lda));
                    }

                    C[globalCol_id*m + globalRow_id] = acc; /* Write result to buffer */
                }
                else if(transa == 2) //If OP(A) = A**T 
                {
                    complex_t acc = (complex_t)(0, 0);
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalRow_id && diag == 1) /* Check if is on diagonal, if yes and have diag == 1 then assume that A have here 1 + 0i, else do normal multiplication */
                            acc += cmul(cmul((complex_t)(1, 0), alpha), B[globalCol_id*K + k]);
                        else
                            acc += cmul(cmul(ACCESS(B, globalRow_id, k, ldb), alpha), ACCESS(A, globalCol_id, k, lda));
                    }

                    C[globalCol_id*m + globalRow_id] = acc; /* Write result to buffer */
                }
            }
        }
    }

}
