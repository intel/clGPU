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

__kernel void Ssyr2_naive_upper(int uplo, int n, float alpha, __global float* x, int incx, __global float* y, int incy, __global float* A, int lda,
 __global float* tempMatrix1, __global float* tempMatrix2, __global float* tempMatrix, __global float* tempX, __global float* tempY)
{
    int i = 0;
    int j = 0;

    //--------------- First Side Operation: alpha * x * y' ------------------
    
    //alpha * x
    for(i = 0; i<n; ++i)
        {
            tempX[i * incx] = alpha * x[i * incx];
        }

    // tempX * y'
    for(i = 0; i<n; ++i)
    {
        for(j = 0; j<n; ++j)
        {
            tempMatrix1[i*n + j] = tempX[i] * y[j];
        }
    }
    //--------------------------- First Side End ---------------------------------

    //--------------- Second Side Operation: alpha * y * x' ------------------
    
    //alpha * y
    for(i = 0; i<n; ++i)
        {
            tempY[i * incy] = alpha * y[i * incy];
        }

    // tempY * x'
    for(i = 0; i<n; ++i)
    {
        for(j = 0; j<n; ++j)
        {
            tempMatrix2[i*n + j] = tempY[i] * x[j];
        }
    }
    //--------------------------- Second Side End ---------------------------------

    //----------------------- Add Matrices tempX + tempY --------------------------
    for(i = 0; i<n; ++i)
    {
        for(j = 0; j<n; ++j)
        {
            tempMatrix[i*n + j] = tempMatrix1[i*n + j] + tempMatrix2[i*n + j];
        }
    }
    //----------------------- End Add Matrices tempMatrix1 + tempMatrix2 --------------------------

    //----------------------- Final Add Matrices tempMatrix + A [Upper] --------------------------
    for(i = 0; i<n; ++i)
    {
        for(j = 0; j<n; ++j)
        {
            if(j>= i) A[i*lda + j] = tempMatrix[i*lda + j] + A[i*lda + j];
        }
    }
    //----------------------- End Final Add Matrices tempMatrix + A [Upper] --------------------------
}
