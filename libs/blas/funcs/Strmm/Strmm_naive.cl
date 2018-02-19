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

__kernel void Strmm_naive(int side, int uplo, int transa, int diag, int m, int n, float alpha, __global float* A, int lda, __global float* B, int ldb, __global float* C, int ldc)
{
    //Quick return if possible
    if(m == 0 || n == 0) return;

    int i;
    int j;

    // Thread identifiers
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    //------------------------------------------------------------Left Side Operation---------------------------------------------------------------//
    if(side == 0)
    {
        int K = m;
        //Upper
        if(uplo == 0)
        {    
            //Upper
            if(globalCol >= globalRow)
            {
                //if OP(A) = A
                if(transa == 0)
                {
                    //Matrix Multiplication
                    float acc = 0.0f;
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalRow && diag == 1) acc += alpha * 1 * B[globalCol*K + k]; //assums that A[x] = 1 on diagonal if is unit, so can leave multiplication
                        else acc += alpha * A[k*m + globalRow] * B[globalCol*K + k]; 
                        
                    }
                    // Store the result in Global Array
                    C[globalCol*m + globalRow] = acc;
                }

                else if(transa == 1) //If OP(A) = A**T 
                {
                    //Matrix Multiplication
                    float acc = 0.0f;
                    for (int k=0; k<K; k++) 
                    {
                         //acc += alpha * A[globalRow*m + k] * B[globalCol*m + k]; //Ok
                        if(k == globalCol && diag == 1) acc += alpha * 1 * B[globalCol*K + k]; //assums that A[x] = 1 on diagonal if is unit, so can leave multiplication
                        else acc += alpha * A[globalRow*m + k] * B[globalCol*K + k];
                    }

                    // Store the result in Global Array
                    C[globalCol*m + globalRow] = acc;
                }

                else if(transa == 2) //If OP(A) = A**T / TODO!!!
                {
                    //Matrix Multiplication
                    float acc = 0.0f;
                    for (int k=0; k<K; k++) 
                    {
                        acc += alpha * A[globalCol*K + k] * B[globalCol*K + k]; //Todo
                    }

                    // Store the result in Global Array
                    C[globalCol*m + globalRow] = acc;
                }
            }
        }

        else//Lower
        {
            //Lower
            if(globalRow >= globalCol)
            {
                
                //if OP(A) = A
                if(transa == 0)
                {
                    //Matrix Multiplication
                    float acc = 0.0f;
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalRow && diag == 1) acc += alpha * 1 * B[globalCol*m + k]; //assums that A[x] = 1 on diagonal if is unit, so can leave multiplication
                        else acc += alpha * A[k*m + globalRow] * B[globalCol*m + k]; 
                        
                    }
                    // Store the result in Global Array
                    C[globalCol*m + globalRow] = acc;
                }
                else if(transa == 1) //If OP(A) = A**T 
                {
                    //Matrix Multiplication
                    float acc = 0.0f;
                    for (int k=0; k<K; k++) 
                    {
                         if(k == globalRow && globalCol==globalRow &&  diag == 1) acc += alpha * 1 * B[globalCol*m + k]; //assums that A[x] = 1 on diagonal if is unit, so can leave multiplication
                        else acc += alpha * A[globalRow*m + k] * B[globalCol*m + k];
                    }

                    // Store the result in Global Array
                    C[globalCol*m + globalRow] = acc;
                }
                else if(transa == 2) //If OP(A) = A**T 
                {
                    //Matrix Multiplication
                    float acc = 0.0f;
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalRow && globalCol==globalRow &&  diag == 1) acc += alpha * 1 * B[globalCol*m + k]; //assums that A[x] = 1 on diagonal if is unit, so can leave multiplication
                        else acc += alpha * A[globalRow*m + k] * B[globalCol*m + k];
                    }

                    // Store the result in Global Array
                    C[globalCol*m + globalRow] = acc;
                }
            }
        }
    }
    //------------------------------------------------------------Right Side Operation---------------------------------------------------------------//
    else
    {
        int K = n;
        //Upper
        if(uplo == 0)
        {    
            //Upper
            if(globalCol >= globalRow)
            {
                //if OP(A) = A
                if(transa == 0)
                {
                    //Matrix Multiplication
                    float acc = 0.0f;
                    for (int k=0; k<K; k++) 
                    {
                         if(k == globalCol && diag == 1) acc += alpha * B[k*m + globalRow] * 1; // * assums that A[x] = 1 on diagonal if is unit, so can leave multiplication
                         else acc += alpha * B[k*m + globalRow] * A[globalCol*K + k]; //Ok
                    }

                    // Store the result in Global Array
                    C[globalCol*m + globalRow] = acc;
                }
                else if(transa == 1) //If OP(A) = A**T
                {
                    //Matrix Multiplication
                    float acc = 0.0f;
                    for (int k=0; k<K; k++) 
                    {
                         if(k == globalCol && diag == 1)  acc += alpha * B[k*m + globalRow] * 1; //assums that A[x] = 1 on diagonal if is unit, so can leave multiplication
                        else acc += alpha * B[k*m + globalRow] * A[k*K + globalCol];
                    }

                    // Store the result in Global Array
                    C[globalCol*m + globalRow] = acc;
                }
                else if(transa == 2) //If OP(A) = A**T
                {
                    //Matrix Multiplication
                    float acc = 0.0f;
                    for (int k=0; k<K; k++) 
                    {
                         if(k == globalCol && diag == 1)  acc += alpha * B[k*m + globalRow] * 1; //assums that A[x] = 1 on diagonal if is unit, so can leave multiplication
                        else acc += alpha * B[k*m + globalRow] * A[k*K + globalCol];
                    }

                    // Store the result in Global Array
                    C[globalCol*m + globalRow] = acc;
                }
            }
        }
        else //Lower
        {
             //Lower
            if(globalRow >= globalCol)
            {
                //if OP(A) = A
                if(transa == 0)
                {
                    //Matrix Multiplication
                    float acc = 0.0f;
                    for (int k=0; k<K; k++) 
                    {
                         if(k == globalCol && diag == 1) acc += alpha * B[k*m + globalRow] * 1; // * assums that A[x] = 1 on diagonal if is unit, so can leave multiplication
                         else acc += alpha * B[k*m + globalRow] * A[globalCol*K + k]; //Ok
                    }

                    // Store the result in Global Array
                    C[globalCol*m + globalRow] = acc;
                }
                else if(transa == 1) //If OP(A) = A**T
                {
                    //Matrix Multiplication
                    float acc = 0.0f;
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalCol && diag == 1)  acc += alpha * B[k*m + globalRow] * 1; //assums that A[x] = 1 on diagonal if is unit, so can leave multiplication
                        else acc += alpha * B[k*m + globalRow] * A[k*K + globalCol];
                    }

                    // Store the result in Global Array
                    C[globalCol*m + globalRow] = acc;
                }
                else if(transa == 2) //If OP(A) = A**T / TODO!!!
                {
                    //Matrix Multiplication
                    float acc = 0.0f;
                    for (int k=0; k<K; k++) 
                    {
                        if(k == globalCol && diag == 1)  acc += alpha * B[k*m + globalRow] * 1; //assums that A[x] = 1 on diagonal if is unit, so can leave multiplication
                        else acc += alpha * B[k*m + globalRow] * A[k*K + globalCol];
                    }

                    // Store the result in Global Array
                    C[globalCol*m + globalRow] = acc;
                }
            }
        }
    }
}
