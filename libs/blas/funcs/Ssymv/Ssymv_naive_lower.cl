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

__kernel void Ssymv_naive_lower(int n, float alpha, __global float* A, int lda, __global float* x, int incx, float beta, __global float* y, int incy, __global float* tempL, __global float* tempR)
{
    //----------------LEFT SIDE START--------------------
   //alpha * A
   for (int i=0; i<n; ++i)
    {
            for (int j=0; j<n; ++j)
            {
                if(i>=j) tempL[i*lda+j] = A[i*lda+j] * alpha;
            }
     }

    //alpha * A * x
    for (int i=0; i<n; ++i)
       {
            for (int j=0; j<n; ++j)
           {
                if(i>=j)  tempR[i] += tempL[i*lda+j] * x[j * incx];
           }
       }
      for (int i=0; i<n; ++i)
       {
            tempL[i] = tempR[i];
       }
       //-----------------LEFT SIDE END---------------------

       //---------------RIGHT SIDE START--------------------
        for (int i=0; i<n; ++i)
       {
                tempR[i] = beta * y[i];
       }
       //-----------------RIGHT SIDE END------------------


       //----------Result = Left Side + Right Side---------
      for (int i=0; i<n; ++i)
       {
           y[i] = tempL[i] + tempR[i];
       }
}
