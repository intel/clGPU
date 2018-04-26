// Copyright (c) 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "blas_reference.hpp"
#include <cmath>
#include <algorithm>
#include <complex>
#include <cassert>

template<typename DataTy>
int axpy_reference_impl(int n, DataTy alpha, array_ref<DataTy> x, int incx, array_ref<DataTy> y, int incy)
{
    assert(x.size() >= static_cast<size_t>(n*incx) && y.size() >= static_cast<size_t>(n*incy));

    if(n <= 0 || alpha == DataTy(0))
        return 0;
    if(incx < 1 || incy < 1) //do not support negative increments
        return 0;

    for(int i = 0; i < n; i++)
    {
        int ix = i * incx;
        int iy = i * incy;
        y[iy] += alpha * x[ix];
    }
    return 0;
}

int Caxpy_reference(int n, std::complex<float> alpha, array_ref<std::complex<float>> x, int incx, array_ref<std::complex<float>> y, int incy)
{
    return axpy_reference_impl(n, alpha, x, incx, y, incy);
}

template<typename DataTy>
int amax_reference_impl(int n, array_ref<DataTy> x, int incx)
{
    assert(x.size() >= static_cast<size_t>(n*incx));
    if(n < 1 || incx < 1)
        return 0;

    auto max_val = std::abs(x[0]);
    int max_idx = 0;
    for(int i = 1; i < n; i++)
    {
        auto curr_val = std::abs(x[i*incx]);
        if(curr_val > max_val)
        {
            max_idx = i;
            max_val = curr_val;
        }
    }
    return max_idx + 1; // Fortran index base is 1
}

int Isamax_reference(int n, array_ref<float> x, int incx)
{
    return amax_reference_impl<float>(n, x, incx);
}

template<typename DataTy>
int syr2_reference_impl(char uplo, int n, DataTy alpha, array_ref<DataTy> x, int incx, array_ref<DataTy> y, int incy, array_ref<DataTy> A, int lda)
{
    assert(x.size() >= static_cast<size_t>(n * incx) && y.size() >= static_cast<size_t>(n * incy) && A.size() >= static_cast<size_t>(n * lda));

    auto MAT_ACCESS = [](array_ref<DataTy>& A, int r, int c, int n)->DataTy&{ return A[c *n + r];};

    if(uplo != 'U' && uplo != 'u' && uplo != 'L' && uplo != 'l')
        return 1;
    if(n < 0)
        return 2;
    if(incx < 1) //do not support negative strides
        return 5;
    if(incy < 1) //do not support negative strides
        return 7;
    if(lda < std::max(1, n))
        return 9;

    if(n == 0 || alpha == DataTy(0))
        return 0;

    if(uplo == 'U' || uplo == 'u')
    {
        if(incx == 1 && incy == 1)
        {
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    float res_1 = alpha * x[i] * y[j];
                    float res_2 = alpha * y[i] * x[j];

                    if (j >= i)
                        MAT_ACCESS(A, i, j, lda) = res_1 + res_2 + MAT_ACCESS(A, i, j, lda);
                    else
                        MAT_ACCESS(A, i, j, lda) = MAT_ACCESS(A, i, j, lda);
                }
            }
        }
        else
        {
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    float res_1 = alpha * x[i * incx] * y[j * incy];
                    float res_2 = alpha * y[j * incy] * x[i * incx];

                    if (j >= i)
                        MAT_ACCESS(A, i, j, lda) = res_1 + res_2 + MAT_ACCESS(A, i, j, lda);
                    else
                        MAT_ACCESS(A, i, j, lda) = MAT_ACCESS(A, i, j, lda);
                }
            }
        }
    }
    else
    {
        if(incx == 1 && incy == 1)
        {

            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    float res_1 = alpha * x[i] * y[j];
                    float res_2 = alpha * y[i] * x[j];

                    if (j <= i)
                        MAT_ACCESS(A, i, j, lda) = res_1 + res_2 + MAT_ACCESS(A, i, j, lda);
                    else
                        MAT_ACCESS(A, i, j, lda) = MAT_ACCESS(A, i, j, lda);
                }
            }
        }
        else
        {
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    float res_1 = alpha * x[i * incx] * y[j * incy];
                    float res_2 = alpha * y[j * incy] * x[i * incx];

                    if (j <= i)
                        MAT_ACCESS(A, i, j, lda) = res_1 + res_2 + MAT_ACCESS(A, i, j, lda);
                    else
                        MAT_ACCESS(A, i, j, lda) = MAT_ACCESS(A, i, j, lda);
                }
            }
        }
    }

    return 0;
}

int Ssyr2_reference(char uplo, int n, float alpha, array_ref<float> x, int incx, array_ref<float> y, int incy, array_ref<float> A, int lda)
{
    return syr2_reference_impl(uplo, n, alpha, x, incx, y, incy, A, lda);
}
