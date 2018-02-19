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

#include <gtest/gtest.h>
#include <iclBLAS.h>
#include <vector>

TEST(Chbmv, Naive_L_NoINC_4x4_k1)
{
    iclblasFillMode_t transa = ICLBLAS_FILL_MODE_LOWER;
    const int n = 4;
    const int k = 1;
    const int incx = 1;
    const int incy = 1;
    int lda = 2;
    oclComplex_t alpha = { 1.f, 0.f };
    oclComplex_t beta = { 1.f, 0.f };
    
    oclComplex_t A[((k + 1)* n)] =
    {
        { 1,0 },
        { 5,5 },{ 6,0 },
        { 10,10 },{ 11,0 },
        { 15,15 },{ 16,0 }
    };
    
    oclComplex_t ex_result[] =
    {
        { 23,3 },{ 73,23 },{ 157,77 },{ 68, 158 }
    };
    
    oclComplex_t x[n*incx] = { { 1.f, 1.f },{ 2.f, 2.f },{ 3.f, 3.f },{ 4.f, 4.f } };
    oclComplex_t y[n*incy] = { { 2.f, 2.f },{ 1.f, 1.f },{ 4.f, 4.f },{ 4.f, 4.f } };
    
    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    
    status = iclblasChbmv(handle, transa, n, k, &alpha, A, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    
    for (int i = 0; i < n*incy; ++i)
    {
        EXPECT_FLOAT_EQ(ex_result[i].val[0], y[i].val[0]);
        EXPECT_FLOAT_EQ(ex_result[i].val[1], y[i].val[1]);
    }
}

TEST(Chbmv, Naive_L_NoINC_4x4_k0)
{
    iclblasFillMode_t transa = ICLBLAS_FILL_MODE_LOWER;
    const int n = 4;
    const int k = 0;
    const int incx = 1;
    const int incy = 1;
    int lda = 1;
    oclComplex_t alpha = { 1.f, 0.f };
    oclComplex_t beta = { 1.f, 0.f };

    oclComplex_t A[((k + 1)* n)] =
    {
        { 1,0 }, { 6,0 }, { 11,0 }, { 16,0 }
    };

    oclComplex_t ex_result[] =
    {
        { 3,3 },{ 13,13 },{ 37,37 },{ 68, 68 }
    };

    oclComplex_t x[n*incx] = { { 1.f, 1.f },{ 2.f, 2.f },{ 3.f, 3.f },{ 4.f, 4.f } };
    oclComplex_t y[n*incy] = { { 2.f, 2.f },{ 1.f, 1.f },{ 4.f, 4.f },{ 4.f, 4.f } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasChbmv(handle, transa, n, k, &alpha, A, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incy; ++i)
    {
        EXPECT_FLOAT_EQ(ex_result[i].val[0], y[i].val[0]);
        EXPECT_FLOAT_EQ(ex_result[i].val[1], y[i].val[1]);
    }
}

TEST(Chbmv, Naive_U_NoINC_4x4_k0)
{
    iclblasFillMode_t transa = ICLBLAS_FILL_MODE_UPPER;
    const int n = 4;
    const int k = 0;
    const int incx = 1;
    const int incy = 1;
    int lda = 1;
    oclComplex_t alpha = { 1.f, 0.f };
    oclComplex_t beta = { 1.f, 0.f };

    oclComplex_t A[((k + 1)* n)] =
    {
        { 1,0 },{ 6,0 },{ 11,0 },{ 16,0 }
    };

    oclComplex_t ex_result[] =
    {
        { 3,3 },{ 13,13 },{ 37,37 },{ 68, 68 }
    };

    oclComplex_t x[n*incx] = { { 1.f, 1.f },{ 2.f, 2.f },{ 3.f, 3.f },{ 4.f, 4.f } };
    oclComplex_t y[n*incy] = { { 2.f, 2.f },{ 1.f, 1.f },{ 4.f, 4.f },{ 4.f, 4.f } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasChbmv(handle, transa, n, k, &alpha, A, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incy; ++i)
    {
        EXPECT_FLOAT_EQ(ex_result[i].val[0], y[i].val[0]);
        EXPECT_FLOAT_EQ(ex_result[i].val[1], y[i].val[1]);
    }
}
