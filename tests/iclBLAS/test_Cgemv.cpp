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
#include <gtest_utils.hpp>

TEST(Cgemv, Naive_N_NonScalar_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_N;
    const int m = 3;
    const int n = 3;
    const int incx = 1;
    const int incy = 1;
    int lda = 3;
    oclComplex_t alpha = { 1.f, 0.f };
    oclComplex_t beta = { 1.f, 0.f };

    oclComplex_t A[] =
    {
        { 1,1 },{ 4,4 },{ 7,7 },
        { 2,2 },{ 5,5 },{ 8,8 },
        { 3,3 },{ 6,6 },{ 9,9 }
    };

    oclComplex_t ex_result[] =
    {
        { 2,30 },{ 1,65 },{ 4,104 },
    };

    oclComplex_t x[n*incx] = { { 1.f, 1.f },{ 2.f, 2.f },{ 3.f, 3.f } };
    oclComplex_t y[m*incy] = { { 2.f, 2.f },{ 1.f, 1.f },{ 4.f, 4.f } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgemv(handle, transa, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, ex_result, y);
}

TEST(Cgemv, Naive_N_Scalar_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_N;
    const int m = 3;
    const int n = 3;
    const int incx = 1;
    const int incy = 1;
    int lda = 3;
    oclComplex_t alpha = { 3.f, 2.f };
    oclComplex_t beta = { 2.f, 4.f };

    oclComplex_t A[] =
    {
        { 1,1 },{ 4,4 },{ 7,7 },
        { 2,2 },{ 5,5 },{ 8,8 },
        { 3,3 },{ 6,6 },{ 9,9 }
    };

    oclComplex_t ex_result[] =
    {
        { -60,96 },{ -130,198 },{ -208,324 },
    };

    oclComplex_t x[m*incx] = { { 1.f, 1.f },{ 2.f, 2.f },{ 3.f, 3.f } };
    oclComplex_t y[n*incy] = { { 2.f, 2.f },{ 1.f, 1.f },{ 4.f, 4.f } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgemv(handle, transa, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, ex_result, y);
}

TEST(Cgemv, Naive_T_NonScalar_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_T;
    const int m = 3;
    const int n = 3;
    const int incx = 1;
    const int incy = 1;
    int lda = 3;
    oclComplex_t alpha = { 1.f, 0.f };
    oclComplex_t beta = { 1.f, 0.f };

    oclComplex_t A[] =
    {
        { 1,1 },{ 2,2 },{ 3,3 },
        { 4,4 },{ 5,5 },{ 6,6 },
        { 7,7 },{ 8,8 },{ 9,9 }
    };

    oclComplex_t ex_result[] =
    {
        { 2,30 },{ 1,65 },{ 4,104 }
    };

    oclComplex_t x[m*incx] = { { 1.f, 1.f },{ 2.f, 2.f },{ 3.f, 3.f } };
    oclComplex_t y[n*incy] = { { 2.f, 2.f },{ 1.f, 1.f },{ 4.f, 4.f } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgemv(handle, transa, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, ex_result, y);
}

TEST(Cgemv, Naive_T_Scalar_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_T;
    const int m = 3;
    const int n = 3;
    const int incx = 1;
    const int incy = 1;
    int lda = 3;
    oclComplex_t alpha = { -2.f, 4.f };
    oclComplex_t beta = { -3.f, -3.f };

    oclComplex_t A[] =
    {
        { 1,1 },{ 2,2 },{ 3,3 },
        { 4,4 },{ 5,5 },{ 6,6 },
        { 7,7 },{ 8,8 },{ 9,9 }
    };

    oclComplex_t ex_result[] =
    {
        { -112, -68 },{ -256,-134 },{ -400,-224 }
    };

    oclComplex_t x[m*incx] = { { 1.f, 1.f },{ 2.f, 2.f },{ 3.f, 3.f } };
    oclComplex_t y[n*incy] = { { 2.f, 2.f },{ 1.f, 1.f },{ 4.f, 4.f } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgemv(handle, transa, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, ex_result, y);
}

TEST(Cgemv, Naive_H_Scalar_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_C;
    const int m = 3;
    const int n = 3;
    const int incx = 1;
    const int incy = 1;
    int lda = 3;
    oclComplex_t alpha = { -2.f, 4.f };
    oclComplex_t beta = { -3.f, -3.f };

    oclComplex_t A[] =
    {
        { 1,1 },{ 2,2 },{ 3,3 },
        { 4,4 },{ 5,5 },{ 6,6 },
        { 7,7 },{ 8,8 },{ 9,9 }
    };

    oclComplex_t ex_result[] =
    {
        { -56, 100 },{ -128,250 },{ -200,376 }
    };

    oclComplex_t x[m*incx] = { { 1.f, 1.f },{ 2.f, 2.f },{ 3.f, 3.f } };
    oclComplex_t y[n*incy] = { { 2.f, 2.f },{ 1.f, 1.f },{ 4.f, 4.f } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgemv(handle, transa, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, ex_result, y);
}

TEST(Cgemv, Naive_N_Scalar_Inc_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_N;
    const int m = 3;
    const int n = 3;
    const int incx = 2;
    const int incy = 2;
    int lda = 3;
    oclComplex_t alpha = { 3.f, 2.f };
    oclComplex_t beta = { 2.f, 4.f };

    oclComplex_t A[] =
    {
        { 1,1 },{ 4,4 },{ 7,7 },
        { 2,2 },{ 5,5 },{ 8,8 },
        { 3,3 },{ 6,6 },{ 9,9 }
    };

    oclComplex_t ex_result[] =
    {
        { -56,90 },{ 1,1 },{ -132,210 }, { 2,2 },{ -198, 300 },{ 4,4 }
    };

    oclComplex_t x[n*incx] = { { 1.f, 1.f },{ 2.f, 2.f },{ 3.f, 3.f }, { 1.f, 1.f },{ 2.f, 2.f },{ 3.f, 3.f } };
    oclComplex_t y[m*incy] = { { 2.f, 2.f },{ 1.f, 1.f },{ 4.f, 4.f },{ 2.f, 2.f },{ 1.f, 1.f },{ 4.f, 4.f } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgemv(handle, transa, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, ex_result, y);
}

TEST(Cgemv, Naive_N_NonScalar_2x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_N;
    const int m = 3;
    const int n = 2;
    const int incx = 1;
    const int incy = 1;
    int lda = 3;
    oclComplex_t alpha = { 1.f, 0.f };
    oclComplex_t beta = { 1.f, 0.f };

    oclComplex_t A[] =
    {
        { 1,1 },{ 4,4 },
        { 2,2 },{ 5,5 },
        { 3,3 },{ 6,6 }
    };

    oclComplex_t ex_result[] =
    {
        { 2,24 },{ 1,21 },{ 4,32 },
    };

    oclComplex_t x[n*incx] = { { 1.f, 1.f },{ 2.f, 2.f } };
    oclComplex_t y[m*incy] = { { 2.f, 2.f },{ 1.f, 1.f },{ 4.f, 4.f } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgemv(handle, transa, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, ex_result, y);
}
